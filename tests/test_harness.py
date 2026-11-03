from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import tempfile
import threading
import unittest
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dedicatedharness.cli import (
    check_tools, export_config, main, parser_for_cli, setup,
)
from dedicatedharness.local import LocalClient, Result, local_url
from dedicatedharness.models import Hub, assess
from dedicatedharness.prompts import (
    admit, brief, count_tokens, messages_for, read_context,
)
from dedicatedharness.runtime import prepare_runtime, serve
from dedicatedharness.system import (
    GIB, Hardware, HarnessError, detect_hardware, load_config,
    require_apple, save_config, write_json,
)


def hardware(**overrides) -> Hardware:
    original = Hardware(
        "Darwin", "arm64", "Apple M5", 24 * GIB, 18 * GIB,
        100 * GIB, "26.0", False,
    )
    return replace(original, **overrides)


def model_info(**overrides):
    values = {
        "id": "mlx-community/test-4bit", "sha": "a" * 40,
        "siblings": [
            SimpleNamespace(rfilename="model.safetensors", size=3 * GIB),
            SimpleNamespace(rfilename="config.json", size=1024),
            SimpleNamespace(rfilename="tokenizer_config.json", size=1024),
            SimpleNamespace(rfilename="unsafe.py", size=99999),
            SimpleNamespace(rfilename="alternative.gguf", size=30 * GIB),
        ],
        "pipeline_tag": "text-generation", "card_data": {"license": "apache-2.0"},
        "gated": False,
    }
    return SimpleNamespace(**(values | overrides))


def model_config(**overrides):
    return {
        "model_type": "qwen3", "quantization": {"bits": 4},
        "num_hidden_layers": 36, "num_key_value_heads": 8,
        "head_dim": 128, "max_position_embeddings": 262144,
    } | overrides


def configuration(**overrides):
    return {
        "version": 1, "backend": "external",
        "base_url": "http://127.0.0.1:1234/v1", "model": "test",
        "context_tokens": 4096, "max_output": 128, "tools_verified": False,
    } | overrides


class ModelTests(unittest.TestCase):
    def candidate(self, info=None, config=None, machine=None, context=4096, template=None):
        return assess(
            info or model_info(), config or model_config(),
            {"chat_template": "{{ tools }} tool_call"} if template is None else template,
            machine or hardware(), context,
        )

    def test_kv_uses_full_precision_not_weight_quantization(self):
        candidate = self.candidate()
        self.assertEqual(candidate.kv_bytes, 2 * 36 * 8 * 128 * 4096 * 2)
        self.assertEqual(candidate.fit, "plausible")
        self.assertGreater(candidate.required_bytes, 11 * GIB)
        self.assertEqual(candidate.download_bytes, 3 * GIB + 2048)
        self.assertTrue(candidate.has_tool_template)

    def test_24gb_rejects_large_weights_even_with_few_active_experts(self):
        info = model_info(siblings=[
            SimpleNamespace(rfilename="model.safetensors", size=20 * GIB),
        ])
        self.assertEqual(self.candidate(info=info).fit, "too-large")

    def test_current_memory_pressure_blocks_install(self):
        candidate = self.candidate(machine=hardware(available_bytes=2 * GIB))
        self.assertEqual(candidate.fit, "memory-pressure")

    def test_unknown_ram_is_uncertain(self):
        candidate = self.candidate(machine=hardware(memory_bytes=None, available_bytes=None))
        self.assertEqual(candidate.fit, "uncertain")

    def test_unknown_available_is_plausible_with_warning(self):
        candidate = self.candidate(machine=hardware(available_bytes=None))
        self.assertEqual(candidate.fit, "plausible")
        self.assertIn("Current available memory unknown", " ".join(candidate.reasons))

    def test_unknown_sizes_never_claim_fit(self):
        info = model_info(siblings=[SimpleNamespace(rfilename="model.safetensors", size=None)])
        self.assertEqual(self.candidate(info=info).fit, "uncertain")

    def test_vision_or_hybrid_architecture_is_not_mlx_lm_compatible_by_tag(self):
        for config in (
            model_config(model_type="qwen3_5"),
            model_config(vision_config={"hidden_size": 100}),
        ):
            with self.subTest(config=config):
                self.assertEqual(self.candidate(config=config).fit, "unsupported")

    def test_context_limit_and_disk_are_checked(self):
        self.assertEqual(
            self.candidate(config=model_config(max_position_embeddings=2048)).fit,
            "too-large",
        )
        self.assertEqual(
            self.candidate(machine=hardware(disk_free_bytes=4 * GIB)).fit, "disk-full"
        )
        self.assertEqual(
            self.candidate(config=model_config(max_position_embeddings=None)).fit, "uncertain"
        )

    def test_missing_template_rejected(self):
        self.assertEqual(self.candidate(template={}).fit, "unsupported")

    def test_missing_kv_metadata_and_no_quantization(self):
        self.assertEqual(
            self.candidate(config=model_config(num_key_value_heads=None)).fit, "uncertain"
        )
        self.assertEqual(
            self.candidate(config=model_config(quantization=None)).fit, "unsupported"
        )

    def test_download_is_revision_pinned_and_filtered(self):
        candidate = self.candidate()
        with patch("huggingface_hub.snapshot_download", return_value="snapshot") as download:
            hub = Hub(Path("cache"))
            self.assertEqual(hub.download(candidate), Path("snapshot"))
        self.assertEqual(download.call_args.kwargs["revision"], "a" * 40)
        self.assertNotIn("*.py", download.call_args.kwargs["allow_patterns"])
        self.assertNotIn("*.gguf", download.call_args.kwargs["allow_patterns"])

    def test_inspection_pins_both_metadata_downloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(json.dumps(model_config()), encoding="utf-8")
            (root / "tokenizer_config.json").write_text(
                json.dumps({"chat_template": "{{ tools }} tool_call"}), encoding="utf-8"
            )
            hub = Hub(root)
            with patch.object(hub.api, "model_info", return_value=model_info()), patch(
                "huggingface_hub.hf_hub_download",
                side_effect=lambda repo, filename, **kwargs: str(root / filename),
            ) as download:
                candidate = hub.inspect("mlx-community/test", hardware(), 4096)
            self.assertEqual(candidate.revision, "a" * 40)
            self.assertEqual(len(download.call_args_list), 2)
            for call in download.call_args_list:
                self.assertEqual(call.kwargs["revision"], "a" * 40)

    def test_separate_chat_template_is_loaded_from_same_revision(self):
        info = model_info()
        info.siblings.append(SimpleNamespace(rfilename="chat_template.jinja", size=128))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(json.dumps(model_config()), encoding="utf-8")
            (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (root / "chat_template.jinja").write_text("{{ tools }} tool_call", encoding="utf-8")
            hub = Hub(root)
            with patch.object(hub.api, "model_info", return_value=info), patch(
                "huggingface_hub.hf_hub_download",
                side_effect=lambda repo, filename, **kwargs: str(root / filename),
            ) as download:
                candidate = hub.inspect(info.id, hardware(), 4096)
            self.assertEqual(candidate.fit, "plausible")
            self.assertTrue(candidate.has_tool_template)
            self.assertEqual(download.call_args.kwargs["revision"], info.sha)
            self.assertEqual(download.call_args.args[1], "chat_template.jinja")

    def test_discovery_uses_bounded_text_only_query(self):
        hub = Hub(Path("cache"))
        with patch.object(hub.api, "list_models", autospec=True, return_value=[]) as listing:
            self.assertEqual(hub.discover("Qwen", 8, hardware(), 4096), [])
        self.assertEqual(listing.call_args.kwargs["pipeline_tag"], "text-generation")
        self.assertEqual(listing.call_args.kwargs["limit"], 8)


class ContextTests(unittest.TestCase):
    def test_task_brief_has_checkable_fields_without_inference(self):
        text = brief("Fix login", "auth.py", "No dependencies", "Tests pass", "Patch")
        self.assertIn("Task: Fix login\n", text)
        self.assertIn("Done when: Tests pass", text)

    def test_context_file_ranges_and_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scope").mkdir()
            (root / "scope" / "test.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
            (root / "outside.txt").write_text("private", encoding="utf-8")
            text = read_context(root / "scope", ["test.txt::2-3"])
            self.assertIn("two\nthree", text)
            self.assertNotIn("\none\n", text)
            for spec in (str(Path("..") / "outside.txt"), "test.txt::0-1", "test.txt::2-9", "test.txt::x"):
                with self.subTest(spec=spec), self.assertRaises(HarnessError):
                    read_context(root / "scope", [spec])

    def test_oversized_context_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "large.txt").write_text("a" * (256 * 1024 + 1), encoding="utf-8")
            with self.assertRaises(HarnessError):
                read_context(root, ["large.txt"])

    def test_external_estimate_includes_tools_and_unicode_bytes(self):
        config = configuration()
        messages = messages_for("Explain \u4e2d\u6587")
        count, label = count_tokens(config, messages)
        with_tools, _ = count_tokens(config, messages, [{"description": "abc" * 100}])
        self.assertGreater(with_tools, count)
        self.assertIn("NOT exact", label)

    def test_exact_context_threshold_and_output_limit(self):
        config = configuration(context_tokens=4096, max_output=512)
        with patch("dedicatedharness.prompts.count_tokens", return_value=(3520, "exact")):
            self.assertEqual(admit(config, messages_for("x"), 512)["output_limit"], 512)
        with patch("dedicatedharness.prompts.count_tokens", return_value=(3521, "exact")):
            with self.assertRaisesRegex(HarnessError, "Nothing was sent"):
                admit(config, messages_for("x"), 512)
        for output in (0, 513):
            with self.assertRaises(HarnessError):
                admit(config, messages_for("x"), output)

    def test_mlx_count_uses_pinned_local_tokenizer_no_remote_code(self):
        config = configuration(backend="mlx", runtime_python=__file__, snapshot="snapshot")
        with patch("dedicatedharness.prompts.subprocess.run", return_value=SimpleNamespace(
            returncode=0, stdout="42\n",
        )) as run:
            count, label = count_tokens(config, messages_for("test"), [{"type": "function"}])
        self.assertEqual(count, 42)
        self.assertIn("exact", label)
        script = run.call_args.args[0][-1]
        self.assertIn("local_files_only=True", script)
        self.assertIn("trust_remote_code=False", script)
        self.assertEqual(
            json.loads(run.call_args.kwargs["input"])["tools"], [{"type": "function"}]
        )


class SystemTests(unittest.TestCase):
    def test_runtime_hardware_guards(self):
        require_apple(hardware())
        for machine in (
            hardware(translated=True), hardware(machine="x86_64"),
            hardware(system="Windows"), hardware(macos="13.5"),
        ):
            with self.subTest(machine=machine), self.assertRaises(HarnessError):
                require_apple(machine)

    def test_detect_m5_memory_with_mocked_macos(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "dedicatedharness.system.platform.system", return_value="Darwin"
        ), patch("dedicatedharness.system.platform.machine", return_value="arm64"), patch(
            "dedicatedharness.system.platform.mac_ver", return_value=("26.0", "", "")
        ), patch("dedicatedharness.system.command_output", side_effect=[
            str(24 * GIB), "Apple M5",
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
            "Pages free: 100.\nPages speculative: 20.\n",
        ]), patch("dedicatedharness.system.subprocess.run", return_value=SimpleNamespace(
            stdout="0\n"
        )):
            result = detect_hardware(Path(temporary))
        self.assertEqual(result.memory_bytes, 24 * GIB)
        self.assertEqual(result.available_bytes, 120 * 16384)
        self.assertTrue(result.apple_silicon)

    def test_config_persistent_atomic_and_no_overwrite_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            with self.assertRaises(HarnessError):
                load_config(home)
            save_config(home, configuration())
            self.assertEqual(load_config(home)["model"], "test")
            save_config(home, configuration(model="second"))
            self.assertEqual(load_config(home)["model"], "second")
            self.assertEqual(list(home.glob("*.tmp")), [])
            out = home / "export.json"
            write_json(out, {"hello": "world"})
            with self.assertRaises(FileExistsError):
                write_json(out, {"other": True})
            self.assertEqual(json.loads(out.read_text())["hello"], "world")

    def test_invalid_config_budgets_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            for override in (
                {"context_tokens": -1}, {"max_output": "128"},
                {"context_tokens": 128, "max_output": 128},
                {"backend": "cloud"}, {"model": None}, {"backend": "mlx"},
            ):
                save_config(home, configuration(**override))
                with self.assertRaises(HarnessError):
                    load_config(home)

    def test_managed_runtime_is_never_installed_on_windows(self):
        with patch("dedicatedharness.runtime.detect_hardware", return_value=hardware(
            system="Windows",
        )), patch("dedicatedharness.runtime.venv.EnvBuilder") as builder:
            with self.assertRaises(HarnessError):
                prepare_runtime(Path("unused"), True)
        builder.assert_not_called()

    def test_setup_refuses_bad_fit_before_install_or_download(self):
        args = parser_for_cli().parse_args([
            "setup", "--yes", "--repo", "mlx-community/test-4bit",
        ])
        candidate = assess(
            model_info(), model_config(), {"chat_template": "hello"},
            hardware(available_bytes=0), 4096,
        )
        with tempfile.TemporaryDirectory() as temporary, patch(
            "dedicatedharness.cli.detect_hardware", return_value=hardware()
        ), patch("dedicatedharness.cli.Hub") as hub, patch(
            "dedicatedharness.cli.prepare_runtime"
        ) as install, patch(
            "dedicatedharness.cli.model_manager.require_stopped"
        ), contextlib.redirect_stdout(io.StringIO()):
            hub.return_value.inspect.return_value = candidate
            with self.assertRaisesRegex(HarnessError, "memory-pressure"):
                setup(args, Path(temporary))
            hub.return_value.download.assert_not_called()
            install.assert_not_called()

    def test_setup_persists_pinned_revision_without_modifying_other_configs(self):
        args = parser_for_cli().parse_args([
            "setup", "--yes", "--install-runtime", "--repo", "mlx-community/test-4bit",
        ])
        candidate = assess(
            model_info(), model_config(), {"chat_template": "hello"}, hardware(), 4096,
        )
        with tempfile.TemporaryDirectory() as temporary, patch(
            "dedicatedharness.cli.detect_hardware", return_value=hardware()
        ), patch("dedicatedharness.cli.Hub") as hub, patch(
            "dedicatedharness.cli.prepare_runtime", return_value=(Path("python"), "0.test")
        ), patch("dedicatedharness.cli.model_manager.verify_model"), patch(
            "dedicatedharness.cli.model_manager.record_owned"
        ), patch("dedicatedharness.cli.model_manager.require_stopped"
        ), contextlib.redirect_stdout(io.StringIO()):
            home = Path(temporary)
            hub.return_value.inspect.return_value = candidate
            hub.return_value.download.return_value = home / "snapshot"
            setup(args, home)
            config = load_config(home)
            self.assertEqual(config["revision"], "a" * 40)
            self.assertFalse(config["tools_verified"])
            self.assertEqual(config["context_tokens"], 4096)
            self.assertEqual(set(home.iterdir()), {home / "config.json", home / "activity.lock"})

    def test_managed_serve_verifies_inference_and_cleans_up_its_child(self):
        help_text = (
            "--decode-concurrency --prompt-concurrency --prompt-cache-size --allowed-origins"
        )
        with tempfile.TemporaryDirectory() as temporary, patch(
            "dedicatedharness.runtime.detect_hardware", return_value=hardware()
        ), patch("dedicatedharness.runtime.socket.socket"), patch(
            "dedicatedharness.runtime.subprocess.check_output", return_value=help_text
        ), patch("dedicatedharness.runtime.subprocess.Popen") as popen, patch(
            "dedicatedharness.runtime.LocalClient"
        ) as client, contextlib.redirect_stdout(io.StringIO()):
            home = Path(temporary)
            snapshot = home / "snapshot"
            snapshot.mkdir()
            config = configuration(
                backend="mlx", runtime_python=__file__, snapshot=str(snapshot),
                model=str(snapshot),
            )
            process = popen.return_value
            process.poll.side_effect = [None, None]
            process.wait.side_effect = [KeyboardInterrupt, 0]
            client.return_value.base_url = config["base_url"]
            client.return_value.models.return_value = [str(snapshot)]
            with self.assertRaises(KeyboardInterrupt):
                serve(config, home, startup_timeout=5)
            process.terminate.assert_called_once()
            client.return_value.chat.assert_called_once()
            command = popen.call_args.args[0]
            self.assertNotIn("--trust-remote-code", command)
            for flag in ("--decode-concurrency", "--prompt-concurrency", "--prompt-cache-size"):
                self.assertEqual(command[command.index(flag) + 1], "1")
            self.assertEqual(popen.call_args.kwargs["env"]["HF_HUB_OFFLINE"], "1")


class FakeServer(BaseHTTPRequestHandler):
    requests_seen = []

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"data": [{"id": "test"}]}).encode())

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.requests_seen.append(payload)
        task = payload["messages"][-1]["content"]
        if task == "redirect":
            self.send_response(302)
            self.send_header("Location", "https://example.com/v1/chat/completions")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if task == "error":
            self.wfile.write(b'data: {"error":{"message":"private prompt"}}\n\n')
            self.wfile.flush()
            return
        if task == "tool":
            packets = [
                {"choices": [{"delta": {"tool_calls": [{
                    "index": 0, "id": "call_1", "function": {
                        "name": "read_", "arguments": "{",
                    },
                }]}}]},
                {"choices": [{"delta": {"tool_calls": [{
                    "index": 0, "function": {"name": "probe", "arguments": "}"},
                }]}, "finish_reason": "tool_calls"}]},
            ]
        elif payload.get("tools") and payload["messages"][-1]["role"] != "tool":
            packets = [{"choices": [{"delta": {"tool_calls": [{
                "index": 0, "id": "probe_1",
                "function": {"name": "read_probe", "arguments": "{}"},
            }]}, "finish_reason": "tool_calls"}]}]
        else:
            text = task if payload["messages"][-1]["role"] == "tool" else "Hello \u4e16\u754c"
            packets = [
                {"choices": [{"delta": {"content": text}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 17, "completion_tokens": 3}},
            ]
        for packet in packets:
            self.wfile.write(("data: " + json.dumps(packet, ensure_ascii=False) + "\n\n").encode())
        if task != "truncated":
            self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class LocalHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def test_only_numeric_loopback_urls_are_permitted(self):
        for url in (
            "https://127.0.0.1:1234/v1", "http://example.com:1234/v1",
            "http://192.168.1.1:1234/v1", "http://localhost:1234/v1",
            "http://127.0.0.1:1234/v1?foo=x", "http://user@127.0.0.1:1234/v1",
            "http://127.0.0.1/v1", "http://127.0.0.1:99999/v1",
            "http://127.0.0.1:1234/api",
        ):
            with self.subTest(url=url), self.assertRaises(HarnessError):
                local_url(url)
        self.assertEqual(local_url("http://[::1]:1234/v1/"), "http://[::1]:1234/v1")

    def test_real_stream_handles_usage_only_packet_unicode_and_no_proxy(self):
        with patch.dict(os.environ, {"HTTP_PROXY": "http://invalid.example:1"}):
            client = LocalClient(self.url)
            self.assertEqual(client.models(), ["test"])
            chunks = []
            result = client.chat("test", messages_for("hi"), 32, on_text=chunks.append)
        self.assertEqual(result.text, "Hello \u4e16\u754c")
        self.assertEqual("".join(chunks), result.text)
        self.assertEqual(result.usage["prompt_tokens"], 17)
        self.assertIsNotNone(result.first_visible_seconds)
        self.assertNotIn("text", result.metrics())
        self.assertNotIn("Hello", json.dumps(result.metrics()))

    def test_streamed_tool_call_fragments_are_assembled(self):
        result = LocalClient(self.url).chat("test", messages_for("tool"), 128)
        self.assertEqual(result.tool_calls[0]["function"], {
            "name": "read_probe", "arguments": "{}",
        })
        self.assertIsNone(result.first_visible_seconds)

    def test_explicit_sampling_is_sent_to_the_local_endpoint(self):
        LocalClient(self.url).chat(
            "test", messages_for("hi"), 32, sampling={"temperature": 0.2, "top_p": 0.9, "seed": 5},
        )
        self.assertEqual(FakeServer.requests_seen[-1]["temperature"], 0.2)
        self.assertEqual(FakeServer.requests_seen[-1]["top_p"], 0.9)
        self.assertEqual(FakeServer.requests_seen[-1]["seed"], 5)

    def test_redirect_truncation_and_stream_errors_are_explicit(self):
        for task, message in (
            ("redirect", "redirects"), ("truncated", "Incomplete"), ("error", "streaming error"),
        ):
            with self.subTest(task=task), self.assertRaisesRegex(HarnessError, message):
                LocalClient(self.url).chat("test", messages_for(task), 32)

    def test_cli_connect_query_and_benchmark_end_to_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            home = str(root / "home")
            prefix = ["--home", home]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(prefix + ["connect", "--url", self.url]), 0)
                self.assertEqual(main(prefix + ["ask", "Explain tests"]), 0)
                report_path = root / "report.json"
                self.assertEqual(main(prefix + [
                    "bench", "Explain tests", "--runs", "2", "--save", str(report_path),
                ]), 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(len(report["runs"]), 2)
            self.assertNotIn("Explain tests", report_path.read_text())
            self.assertNotIn("Hello", report_path.read_text())
            self.assertIsNone(report["runs"][0]["pure_decode_tokens_per_second"])

    def test_admission_blocks_http_and_dry_run_sends_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            save_config(home, configuration(base_url=self.url, context_tokens=900))
            before = len(FakeServer.requests_seen)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--home", str(home), "ask", "x" * 1000]), 1)
                self.assertEqual(main(["--home", str(home), "ask", "x", "--dry-run"]), 0)
            self.assertEqual(len(FakeServer.requests_seen), before)

    def test_real_tool_roundtrip_enables_exports_without_selecting_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            config = configuration(base_url=self.url)
            save_config(home, config)
            with contextlib.redirect_stdout(io.StringIO()):
                check_tools(config, home, timeout=5)
            self.assertTrue(load_config(home)["tools_verified"])
            for target in ("openclaw", "opencode"):
                exported = export_config(config, target)
                self.assertNotIn("agents", exported)
                self.assertNotIn("model", exported)
                self.assertNotIn("fallbacks", json.dumps(exported))
                self.assertIn(self.url, json.dumps(exported))

    def test_failed_tool_verification_invalidates_previous_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            config = configuration(base_url=self.url, tools_verified=True)
            with patch("dedicatedharness.cli.LocalClient") as client:
                client.return_value.chat.return_value = Result(
                    "I called it", 0.1, 0.01, None, "stop", [],
                )
                with self.assertRaises(HarnessError):
                    check_tools(config, home, timeout=5)
            self.assertFalse(load_config(home)["tools_verified"])
            with self.assertRaises(HarnessError):
                export_config(config, "opencode")


if __name__ == "__main__":
    unittest.main()
