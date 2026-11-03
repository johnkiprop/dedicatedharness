from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import httpx
import test_harness
import test_quality

from dedicatedharness.activity import activity_lock
from dedicatedharness.agent_engine import environment
from dedicatedharness.agent_gateway import Gateway, compact
from dedicatedharness.cli import main
from dedicatedharness.model_manager import cleanup_previous
from dedicatedharness.profiles import activate, read_profile, save_profile
from dedicatedharness.quality import add_case, report, run_check, start_trial
from dedicatedharness.supervisor import editable_paths, read_state, run_supervised
from dedicatedharness.system import HarnessError, load_config, save_config


class ModelServer:
    def __init__(self, callback, model="test"):
        self.payloads = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_arguments):
                pass

            def do_GET(self):
                body = json.dumps({"data": [{"id": model}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.payloads.append(payload)
                value = callback(payload)
                if isinstance(value, str):
                    delta = {"content": value}
                    reason = "stop"
                else:
                    name, arguments = value
                    delta = {"tool_calls": [{"index": 0, "id": f"call_{len(owner.payloads)}",
                                             "type": "function",
                                             "function": {"name": name, "arguments": json.dumps(arguments)}}]}
                    reason = "tool_calls"
                packet = {"id": "chatcmpl-fixture", "object": "chat.completion.chunk",
                          "created": 1, "model": model}
                frames = [
                    {**packet, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                    {**packet, "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
                     "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
                ]
                body = "".join("data: " + json.dumps(frame) + "\n\n" for frame in frames) + "data: [DONE]\n\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body.encode())) )
                self.end_headers()
                self.wfile.write(body.encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class GatewayTests(unittest.TestCase):
    def gateway(self, **overrides):
        return Gateway(test_harness.configuration(context_tokens=32768, max_output=128),
                       time.monotonic() + 10, 4, 100000, **overrides)

    def test_gateway_pins_model_tools_budget_sampling_and_authentication(self):
        server = ModelServer(lambda _: "OK")
        self.addCleanup(server.close)
        guarded = []
        gateway = Gateway(test_harness.configuration(base_url=server.url, context_tokens=32768),
                          time.monotonic() + 10, 2, 100000, guard=lambda: guarded.append(True))
        with gateway.serve() as url, httpx.Client(trust_env=False) as client:
            self.assertEqual(client.get(url + "/models").status_code, 401)
            response = client.post(url + "/chat/completions",
                                   headers={"Authorization": "Bearer " + gateway.key},
                                   json={"model": "test", "messages": [{"role": "user", "content": "Hi"}],
                                         "stream": True, "temperature": 2, "max_tokens": 999999})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn("[DONE]", response.text)
            sent = server.payloads[0]
            self.assertEqual(sent["max_tokens"], 128)
            self.assertEqual(sent["temperature"], 0)
            self.assertEqual(sent["top_p"], 1)
            self.assertEqual(gateway.calls[0]["status"], "completed")
        self.assertEqual(len(server.payloads), 1)
        self.assertEqual(len(guarded), 2)

    def test_wrong_model_tool_and_budget_are_refused_before_inference(self):
        payload = {"model": "other", "messages": [{"role": "user", "content": "Hi"}]}
        with self.assertRaisesRegex(HarnessError, "pinned"):
            self.gateway().generate(payload)
        payload["model"] = "test"
        payload["tools"] = [{"type": "function", "function": {"name": "bash"}}]
        with self.assertRaisesRegex(HarnessError, "allowlist"):
            self.gateway().generate(payload)
        payload.pop("tools")
        gateway = self.gateway()
        gateway.budget = 1
        with self.assertRaisesRegex(HarnessError, "admission budget"):
            gateway.generate(payload)

    def test_compaction_preserves_tool_call_relationships_and_goal(self):
        messages = [
            {"role": "user", "content": "Keep this goal"},
            {"role": "assistant", "tool_calls": [{"id": "call_1"}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "a" * 8000},
            {"role": "assistant", "tool_calls": [{"id": "call_2"}]},
            {"role": "tool", "tool_call_id": "call_2", "content": "latest"},
        ]
        result, count = compact(messages, clear_old=True)
        self.assertEqual(count, 1)
        self.assertEqual(result[0], messages[0])
        self.assertEqual(result[2]["tool_call_id"], "call_1")
        self.assertIn("cleared", result[2]["content"])
        self.assertEqual(result[-1]["content"], "latest")
        self.assertEqual(len(messages[2]["content"]), 8000)

    def test_context_limit_still_fails_after_compaction(self):
        gateway = self.gateway()
        gateway.config["context_tokens"] = 512
        with self.assertRaisesRegex(HarnessError, "Context budget exceeded"):
            gateway.generate({"model": "test", "messages": [{"role": "user", "content": "x" * 1000}]})
        self.assertEqual(gateway.calls[0]["status"], "failed")

    def test_request_deadline_and_request_count_are_enforced(self):
        gateway = self.gateway(calls=[{}] * 4)
        with self.assertRaisesRegex(HarnessError, "request limit"):
            gateway.generate({})
        gateway = self.gateway()
        gateway.deadline = time.monotonic() - 1
        with self.assertRaisesRegex(HarnessError, "deadline"):
            gateway.generate({})

    def test_file_and_patch_paths_are_checked_before_tools_execute(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "src").mkdir()
            gateway = self.gateway()
            gateway.root = root
            gateway.edits = [root / "src"]
            gateway.protected = {root / "src" / "grader.py"}
            gateway.validate_tool("write", {"filePath": "src/new.py"})
            gateway.validate_tool("apply_patch", {"patchText": "*** Begin Patch\n*** Add File: src/new.py\n+x\n*** End Patch"})
            for name in ("../outside.py", "src/grader.py", ".git/config"):
                with self.assertRaises(HarnessError):
                    gateway.validate_tool("write", {"filePath": name})
            with self.assertRaises(HarnessError):
                gateway.validate_tool("apply_patch", {"patchText":
                    "*** Begin Patch\n*** Update File: src/new.py\n*** Move to: ../outside.py\n+x\n*** End Patch"})

    def test_engine_environment_does_not_inherit_credentials_or_user_config(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {
            "OPENAI_API_KEY": "fixture-only", "OPENCODE_CONFIG": "user-global.json",
        }):
            env = environment(Path(temporary), {"model": "dedicatedharness/local"}, "fixture-only")
            self.assertNotIn("OPENAI_API_KEY", env)
            self.assertNotIn("OPENCODE_CONFIG", env)
            self.assertEqual(env["OPENCODE_DISABLE_PROJECT_CONFIG"], "true")
            self.assertEqual(env["OPENCODE_PURE"], "true")
            self.assertEqual(env["OPENCODE_DISABLE_MODELS_FETCH"], "true")


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        save_config(self.home, test_harness.configuration())

    def test_profiles_are_named_immutable_and_clear_verification(self):
        config = load_config(self.home)
        config.update(tools_verified=True, tools_verified_at="before")
        save_profile(self.home, "coding", config)
        self.assertFalse(read_profile(self.home, "coding")["tools_verified"])
        self.assertNotIn("tools_verified_at", read_profile(self.home, "coding"))
        with self.assertRaises(FileExistsError):
            save_profile(self.home, "coding")
        with self.assertRaises(HarnessError):
            save_profile(self.home, "../outside")

    def test_external_activation_validates_model_and_preserves_old_config_on_failure(self):
        save_profile(self.home, "reasoning", test_harness.configuration(model="different"))
        with patch("dedicatedharness.profiles.LocalClient.models", return_value=["test"]):
            with self.assertRaisesRegex(HarnessError, "not exposed"):
                activate(self.home, "reasoning")
        self.assertEqual(load_config(self.home)["model"], "test")
        with patch("dedicatedharness.profiles.LocalClient.models", return_value=["different"]):
            activate(self.home, "reasoning")
        self.assertEqual(load_config(self.home)["model"], "different")
        self.assertFalse(load_config(self.home)["tools_verified"])

    def test_saved_profile_prevents_upgrade_cleanup(self):
        previous = test_harness.configuration(repo="mlx-community/old", revision="a" * 40)
        save_profile(self.home, "rollback", previous)
        with patch("dedicatedharness.model_manager.inventory", return_value={
            "mlx-community/old@" + "a" * 40: {"snapshot": str(self.home / "hub" / "old")},
        }):
            self.assertIn("saved profile", cleanup_previous(
                self.home, previous, test_harness.configuration(repo="mlx-community/new", revision="b" * 40)))

    def test_active_operation_prevents_other_thread_and_process_changes(self):
        errors = []
        with activity_lock(self.home):
            def change():
                try:
                    save_config(self.home, test_harness.configuration(model="changed"))
                except HarnessError as error:
                    errors.append(str(error))
            thread = threading.Thread(target=change)
            thread.start()
            thread.join()
            script = (
                "from dedicatedharness.system import save_config; from pathlib import Path; "
                f"save_config(Path({str(self.home)!r}), {{'changed': True}})"
            )
            result = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"Another model operation", result.stderr)
        self.assertEqual(len(errors), 1)
        self.assertEqual(load_config(self.home)["model"], "test")
        save_config(self.home, test_harness.configuration(model="after"))
        self.assertEqual(load_config(self.home)["model"], "after")

    def managed_profile(self, **overrides):
        snapshot = (self.home / "hub" / "snapshots" / ("a" * 40)).resolve()
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text(json.dumps(test_harness.model_config()), encoding="utf-8")
        (snapshot / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": "{{ tools }} tool_call"}), encoding="utf-8")
        (snapshot / "model.safetensors").write_bytes(b"fixture weights")
        config = test_harness.configuration(
            backend="mlx", repo="mlx-community/test", revision="a" * 40, license="apache-2.0",
            snapshot=str(snapshot), model=str(snapshot), runtime_python=sys.executable,
            runtime_version="fixture", **overrides,
        )
        save_profile(self.home, "managed", config)
        (self.home / "owned-models.json").write_text(json.dumps({
            "mlx-community/test@" + "a" * 40: {
                "repo": "mlx-community/test", "revision": "a" * 40, "snapshot": str(snapshot),
            },
        }), encoding="utf-8")
        return snapshot

    def test_managed_profile_restores_offline_and_rechecks_current_hardware(self):
        snapshot = self.managed_profile()
        with (
            patch("dedicatedharness.profiles.detect_hardware", return_value=test_harness.hardware(disk_free_bytes=0)),
            patch("dedicatedharness.runtime.prepare_runtime", return_value=(Path(sys.executable), "fixture")),
            patch("dedicatedharness.model_manager.require_stopped"),
            patch("dedicatedharness.model_manager.verify_model") as verify,
            patch("dedicatedharness.models.Hub.inspect", side_effect=AssertionError("Restore must not need network")),
        ):
            activate(self.home, "managed")
            verify.assert_called_once_with(Path(sys.executable), snapshot)
        self.assertEqual(load_config(self.home)["snapshot"], str(snapshot))
        self.assertFalse(load_config(self.home)["tools_verified"])

    def test_managed_profile_rejects_pressure_and_runtime_drift_without_switching(self):
        self.managed_profile()
        with (
            patch("dedicatedharness.profiles.detect_hardware",
                  return_value=test_harness.hardware(available_bytes=1)),
            patch("dedicatedharness.model_manager.require_stopped"),
        ):
            with self.assertRaisesRegex(HarnessError, "memory-pressure"):
                activate(self.home, "managed")
        self.assertEqual(load_config(self.home)["model"], "test")
        with (
            patch("dedicatedharness.profiles.detect_hardware", return_value=test_harness.hardware()),
            patch("dedicatedharness.runtime.prepare_runtime", return_value=(Path(sys.executable), "different")),
            patch("dedicatedharness.model_manager.require_stopped"),
        ):
            with self.assertRaisesRegex(HarnessError, "runtime changed"):
                activate(self.home, "managed")
        self.assertEqual(load_config(self.home)["model"], "test")


class SupervisorFixture:
    def setUp(self):
        test_quality.GitQualityTests.setUp(self)
        config = load_config(self.home)
        config.update(tools_verified=True, context_tokens=32768)
        save_config(self.home, config)
        self.task = self.store.new("Set VALUE to 2", "Trusted grader passes", ["source.py"], ["tests"])
        self.store.checkpoint(None, decisions="Set VALUE to 2 without changing grader")
        self.store.approve(None, "Only change source.py")
        self.args = Namespace(root=self.root, id=None, opencode=sys.executable,
                              steps=6, repairs=1, requests=16, timeout=30, budget=100000,
                              allow_edit=[], resume=None, keep_trace=False, yes=True)

    def git(self, *arguments):
        return test_quality.GitQualityTests.git(self, *arguments)

    def invoke(self, values=(2,), *, change=True, initial_error=False, terminal=True, interrupt=False):
        root = self.root
        store = self.store

        class FakeEngine:
            def __init__(self, *_arguments):
                self.version = "1.19.4"
                self.index = -1

            def __enter__(self):
                if initial_error:
                    raise HarnessError("Injected engine startup failure")
                return self

            def __exit__(self, *_arguments):
                pass

            def start(self, _prompt):
                self.index += 1
                if change:
                    value = values[min(self.index, len(values) - 1)]
                    (root / "source.py").write_text(f"VALUE = {value}\n", encoding="utf-8")
                return "ses_fixture"

            def observe(self):
                if interrupt:
                    raise KeyboardInterrupt
                return {"terminal": True, "tools": {"write": 1}} if terminal else None

        with (
            patch("dedicatedharness.supervisor.Engine", FakeEngine),
            patch("dedicatedharness.supervisor.LocalClient.models", return_value=["test"]),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return run_supervised(self.args, self.home)

    def records(self):
        return [json.loads(path.read_text()) for path in (self.store.path.parent / "runs").glob("*.json")]


class SupervisorTests(SupervisorFixture, unittest.TestCase):
    def test_verified_changes_complete_without_trusting_model_text(self):
        result = self.invoke()
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["first_attempt_passed"])
        self.assertEqual(self.store.snapshot()[1].phase, "done")
        self.assertTrue(result["attempts"][0]["checks"][0]["success"])
        self.assertEqual(read_state(self.store, result["id"])["status"], "completed")

    def test_repair_is_bounded_and_does_not_replace_first_failure(self):
        result = self.invoke((3, 2))
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["first_attempt_passed"])
        self.assertEqual([attempt["status"] for attempt in result["attempts"]], ["failed", "passed"])

    def test_no_progress_does_not_retry_or_claim_success(self):
        with self.assertRaisesRegex(HarnessError, "no Git-observed progress"):
            self.invoke(change=False)
        record = self.records()[0]
        self.assertEqual(len(record["attempts"]), 1)
        self.assertEqual(record["status"], "blocked")
        self.assertNotEqual(self.store.snapshot()[1].phase, "done")

    def test_failed_checks_exhaust_repair_budget(self):
        with self.assertRaisesRegex(HarnessError, "Repair budget exhausted"):
            self.invoke((3, 4))
        self.assertEqual(len(self.records()[0]["attempts"]), 2)
        self.assertEqual(self.store.snapshot()[1].checks["tests"].status, "failed")

    def test_resume_rejects_git_drift(self):
        with self.assertRaisesRegex(HarnessError, "startup failure"):
            self.invoke(initial_error=True)
        self.args.resume = self.records()[0]["id"]
        (self.root / "source.py").write_text("VALUE = 77\n", encoding="utf-8")
        with self.assertRaisesRegex(HarnessError, "Resume refused"):
            self.invoke()

    def test_resume_uses_compact_state_and_preserves_limits(self):
        with self.assertRaises(HarnessError):
            self.invoke(initial_error=True)
        self.args.resume = self.records()[0]["id"]
        result = self.invoke()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["id"], self.args.resume)

    def test_interrupted_candidate_can_be_verified_on_resume_without_fake_edits(self):
        with self.assertRaises(KeyboardInterrupt):
            self.invoke(interrupt=True)
        self.args.resume = self.records()[0]["id"]
        result = self.invoke(change=False)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["attempts"][1]["resumed_existing_candidate"])
        self.assertFalse(result["first_attempt_passed"])

    def test_explicit_new_file_scope_is_local_and_literal(self):
        self.assertEqual(editable_paths(self.root, self.task, ["new_test.py"]), ["new_test.py"])
        for path in ("..", ".", ".git", "*.py"):
            with self.assertRaises(HarnessError):
                editable_paths(self.root, self.task, [path])

    def test_unapproved_or_unverified_work_cannot_run(self):
        self.store.checkpoint(None, decisions="Different plan")
        with self.assertRaisesRegex(HarnessError, "explicitly approved"):
            self.invoke()

    def test_deadline_stops_a_nonterminal_engine_and_releases_model_lock(self):
        self.args.timeout = 1
        with self.assertRaisesRegex(HarnessError, "deadline"):
            self.invoke(terminal=False)
        self.assertEqual(self.records()[0]["status"], "blocked")
        with activity_lock(self.home):
            pass

    def test_pending_device_check_is_never_fabricated(self):
        self.task = self.store.new("Set VALUE to 2", "Tests and phone check", ["source.py"], ["tests", "phone"])
        self.store.checkpoint(None, decisions="Set VALUE to 2")
        self.store.approve(None, "Only source.py")
        with self.assertRaisesRegex(HarnessError, "remaining required checks"):
            self.invoke()
        current = self.store.snapshot()[1]
        self.assertEqual(current.checks["phone"].status, "pending")
        self.assertNotEqual(current.phase, "done")

    def test_supervised_trial_records_observed_harness_separately(self):
        add_case(self.store, "fixture-v1", "Set VALUE to 2", "Grader passes", ["tests"], ["source.py"])
        start_trial(self.store, "fixture-v1", load_config(self.home))
        self.store.checkpoint(None, decisions="Set VALUE to 2")
        self.store.approve(None, "Only source.py")
        result = self.invoke((3, 2))
        group = report(self.store)["profiles"][0]
        self.assertIn("execution", group["profile"]["configuration"])
        self.assertEqual(group["passed"], 0)
        self.assertFalse(result["first_attempt_passed"])
        self.assertEqual(group["profile"]["configuration"]["execution"]["engine_version"], "1.19.4")

    def test_bound_trial_cannot_be_submitted_by_manual_quality_run(self):
        add_case(self.store, "fixture-v1", "Set VALUE to 2", "Grader passes", ["tests"], ["source.py"])
        start_trial(self.store, "fixture-v1", load_config(self.home))
        self.store.checkpoint(None, decisions="Set VALUE to 2")
        self.store.approve(None, "Only source.py")
        self.args.repairs = 0
        with self.assertRaises(HarnessError):
            self.invoke((3,))
        with self.assertRaisesRegex(HarnessError, "bound execution supervisor"):
            run_check(self.store, "tests", None)


@unittest.skipUnless(os.environ.get("DEDICATEDHARNESS_TEST_OPENCODE"), "Optional real OpenCode engine integration")
class NativeEngineTests(SupervisorFixture, unittest.TestCase):
    def invoke_native(self, target="source.py", *, expected=2, model="test"):
        phase = [0]

        def respond(payload):
            if not payload.get("tools"):
                return "Done"
            step = phase[0] % 3
            phase[0] += 1
            if step == 0:
                return "read", {"filePath": str(self.store.root / "source.py"), "offset": 1, "limit": 20}
            if step == 1:
                return "write", {"filePath": str(self.store.root / target), "content": f"VALUE = {expected}\n"}
            return "Done"

        server = ModelServer(respond, model)
        self.addCleanup(server.close)
        config = load_config(self.home)
        config["base_url"] = server.url
        config["model"] = model
        save_config(self.home, config)
        self.args.opencode = os.environ["DEDICATEDHARNESS_TEST_OPENCODE"]
        self.args.timeout = 90
        self.args.keep_trace = False
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_supervised(self.args, self.home)
        self.assertEqual(server.payloads[0]["model"], model)
        return result

    def test_real_opencode_reads_edits_and_is_verified(self):
        result = self.invoke_native()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["engine_version"], "1.19.4")
        self.assertIn("read", result["attempts"][0]["observation"]["tools"])
        self.assertIn("write", result["attempts"][0]["observation"]["tools"])

    def test_real_opencode_denies_protected_grader_edit(self):
        with self.assertRaisesRegex(HarnessError, "protected grader"):
            self.invoke_native("grader.py")
        self.assertIn("assert runpy", (self.root / "grader.py").read_text())

    def test_real_opencode_permission_denial_without_gateway_path_guard(self):
        with patch("dedicatedharness.agent_gateway.Gateway.validate_tool"):
            with self.assertRaisesRegex(HarnessError, "no Git-observed progress"):
                self.invoke_native("grader.py")
        self.assertIn("assert runpy", (self.root / "grader.py").read_text())

    def test_real_opencode_ignores_project_providers_and_plugins(self):
        (self.root / "opencode.json").write_text(json.dumps({
            "model": "cloud-fixture/forbidden", "plugin": ["file:///missing-untrusted-plugin.js"],
            "provider": {"cloud-fixture": {"options": {"baseURL": "https://example.invalid"}}},
        }), encoding="utf-8")
        result = self.invoke_native()
        self.assertEqual(result["status"], "completed")

    def test_real_opencode_pins_mac_snapshot_model_id_with_slashes_and_spaces(self):
        model = "/Users/fixture/Library/Application Support/dedicatedharness/hub/snapshots/" + "a" * 40
        result = self.invoke_native(model=model)
        self.assertEqual(result["model_configuration"]["model"], model)
        self.assertEqual(result["status"], "completed")


if __name__ == "__main__":
    unittest.main()
