from __future__ import annotations

import contextlib
import io
import json
import socket
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import test_harness as harness_tests

from dedicatedharness.cli import main
from dedicatedharness.local import validate_sampling
from dedicatedharness.model_manager import (
    cleanup_previous, evidence, inventory, record_owned, require_stopped, upgrade_allowed,
)
from dedicatedharness.models import Hub
from dedicatedharness.projects import ProjectStore
from dedicatedharness.quality import (
    add_case, execute, initialize, profile, register_command,
    report, run_check, start_trial,
)
from dedicatedharness.repository import inspect, suggest
from dedicatedharness.system import HarnessError, load_config, save_config
from test_harness import configuration


class GitQualityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "repo"
        self.root.mkdir()
        self.home = self.base / "private"
        self.git("init", "--initial-branch=main")
        (self.root / ".gitignore").write_text("__pycache__/\n*.log\n", encoding="utf-8")
        (self.root / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
        (self.root / "grader.py").write_text(
            "import runpy\nassert runpy.run_path('source.py')['VALUE'] == 2\n", encoding="utf-8"
        )
        self.git("add", ".")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                 "-c", "commit.gpgsign=false", "commit", "-m", "Fixture baseline")
        self.store = ProjectStore(self.home, self.root)
        self.store.initialize("Fixture", "Improve correctness", "Preserve trusted tests")
        initialize(self.store)
        register_command(self.store, "tests", [sys.executable, "grader.py"], 5, ["grader.py"])
        save_config(self.home, configuration(repo="mlx-community/test", revision="a" * 40))

    def git(self, *arguments):
        return subprocess.check_output(
            ["git", "-c", f"core.hooksPath={self.base / 'no-hooks'}", "-C", str(self.root),
             *arguments], stderr=subprocess.PIPE, text=True,
        )

    def candidate(self, *, change=True):
        task = self.store.new("Set VALUE to 2", "Trusted grader passes", ["source.py"], ["tests"])
        self.store.checkpoint(None, decisions="Set VALUE to 2 without changing grader")
        self.store.approve(None, "Only change VALUE")
        if change:
            (self.root / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
        self.store.checkpoint(None, changes="Changed VALUE in source.py")
        self.store.ready(None)
        return task

    def test_scoped_fingerprint_detects_untracked_staged_and_deleted_changes(self):
        first = inspect(self.root, required=True)
        (self.root / "output.log").write_text("ignored", encoding="utf-8")
        self.assertEqual(inspect(self.root).fingerprint, first.fingerprint)
        (self.root / "new.py").write_text("print('new')", encoding="utf-8")
        second = inspect(self.root)
        self.assertNotEqual(first.fingerprint, second.fingerprint)
        self.git("add", "new.py")
        self.assertNotEqual(second.fingerprint, inspect(self.root).fingerprint)
        (self.root / "source.py").unlink()
        self.assertIn("source.py", [item["path"] for item in inspect(self.root).changes])

    def test_subproject_scope_ignores_sibling_changes(self):
        child = self.root / "child"
        child.mkdir()
        (child / "module.py").write_text("VALUE=1", encoding="utf-8")
        first = inspect(child.resolve())
        (self.root / "source.py").write_text("VALUE=99", encoding="utf-8")
        self.assertEqual(first.fingerprint, inspect(child.resolve()).fingerprint)

    def test_git_drift_invalidates_executed_checks_and_completed_task(self):
        self.candidate()
        self.assertTrue(run_check(self.store, "tests", None)["success"])
        self.assertEqual(self.store.finish(None).phase, "done")
        (self.root / "source.py").write_text("VALUE = 3\n", encoding="utf-8")
        task = self.store.snapshot()[1]
        self.assertEqual(task.phase, "investigate")
        self.assertEqual(task.checks["tests"].status, "pending")
        self.assertFalse(task.approval_scope)

    def test_branch_change_revokes_approval(self):
        self.candidate()
        self.git("switch", "-c", "other")
        task = self.store.snapshot()[1]
        self.assertFalse(task.approval_scope)
        self.assertEqual(task.repository["branch"], "other")

    def test_manual_pass_is_refused_and_actual_success_is_bound_to_snapshot(self):
        self.candidate()
        with self.assertRaisesRegex(HarnessError, "manual passing"):
            self.store.validate(None, "tests", "passed", "Trust me")
        result = run_check(self.store, "tests", None)
        self.assertTrue(result["success"])
        task = self.store.snapshot()[1]
        self.assertEqual(task.checks["tests"].run_id, result["id"])
        self.assertEqual(result["fingerprint"], task.repository["fingerprint"])
        self.assertEqual(self.store.finish(None).phase, "done")

    def test_failing_command_does_not_accept_candidate(self):
        self.candidate(change=False)
        result = run_check(self.store, "tests", None)
        self.assertFalse(result["success"])
        self.assertEqual(self.store.snapshot()[1].checks["tests"].status, "failed")
        with self.assertRaises(HarnessError):
            self.store.finish(None)

    def test_protected_test_modification_is_not_a_valid_grade(self):
        self.candidate()
        (self.root / "grader.py").write_text("print('always pass')\n", encoding="utf-8")
        # A new implementation revision must still respect the frozen grader.
        self.store.checkpoint(None, changes="Implementation and grader changed")
        self.store.ready(None)
        with self.assertRaisesRegex(HarnessError, "Trusted graders changed"):
            run_check(self.store, "tests", None)

    def test_tests_that_modify_the_repository_are_not_reported_passed(self):
        (self.root / "grader.py").write_text(
            "from pathlib import Path\nPath('unexpected.py').write_text('changed')\n",
            encoding="utf-8",
        )
        register_command(self.store, "tests", [sys.executable, "grader.py"], 5, ["grader.py"])
        self.candidate()
        result = run_check(self.store, "tests", None)
        self.assertFalse(result["success"])
        self.assertNotEqual(self.store.snapshot()[1].checks["tests"].status, "passed")

    def test_reapproving_command_invalidates_previous_evidence(self):
        self.candidate()
        run_check(self.store, "tests", None)
        register_command(self.store, "tests", [sys.executable, "grader.py"], 10, ["grader.py"])
        with self.assertRaises(HarnessError):
            self.store.finish(None)
        self.assertEqual(self.store.snapshot()[1].checks["tests"].status, "pending")

    def test_context_suggestions_and_registry_are_local(self):
        self.assertEqual(suggest(self.root, "source value")[0]["file"], "source.py")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["--home", str(self.home), "project", "list"]), 0)
        self.assertEqual(json.loads(out.getvalue())["projects"][0]["project"]["name"], "Fixture")

    def test_first_submitted_trial_uses_executed_protected_results(self):
        add_case(self.store, "value-two", "Set VALUE to 2", "Grader passes", ["tests"], ["source.py"])
        start_trial(self.store, "value-two", configuration(repo="mlx-community/test", revision="a" * 40))
        self.store.checkpoint(None, decisions="Change only VALUE")
        self.store.approve(None, "Change VALUE")
        (self.root / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
        self.store.checkpoint(None, changes="Set VALUE to 2")
        self.store.ready(None)
        self.assertTrue(run_check(self.store, "tests", None)["success"])
        summary = report(self.store)
        group = summary["profiles"][0]
        self.assertEqual(group["passed"], 1)
        self.assertEqual(group["tasks"], 1)
        self.assertFalse(group["target_observed"])
        self.assertIn("not", summary["limitations"].lower())

    def test_changed_case_graders_require_a_new_case_version(self):
        add_case(self.store, "value-two", "Set VALUE", "Grader passes", ["tests"], [])
        register_command(self.store, "tests", [sys.executable, "grader.py"], 10, ["grader.py"])
        with self.assertRaisesRegex(HarnessError, "versioned case"):
            start_trial(self.store, "value-two", configuration())
        self.assertFalse(self.store.list_tasks())

    def test_failed_first_submission_cannot_be_flipped_by_repair(self):
        add_case(self.store, "value-two", "Set VALUE to 2", "Grader passes", ["tests"], [])
        start_trial(self.store, "value-two", configuration(repo="mlx-community/test", revision="a" * 40))
        self.store.checkpoint(None, decisions="Change VALUE")
        self.store.approve(None, "Change VALUE")
        (self.root / "source.py").write_text("VALUE = 3\n", encoding="utf-8")
        self.store.checkpoint(None, changes="First candidate")
        self.store.ready(None)
        self.assertFalse(run_check(self.store, "tests", None)["success"])
        (self.root / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
        self.store.checkpoint(None, changes="Repaired candidate")
        self.store.ready(None)
        self.assertTrue(run_check(self.store, "tests", None)["success"])
        group = report(self.store)["profiles"][0]
        self.assertEqual(group["passed"], 0)
        self.assertEqual(group["tasks"], 1)

    def test_95_percent_threshold_has_minimum_sample_and_confidence_interval(self):
        identity = profile(configuration())
        with self.store.connection() as database:
            for index in range(100):
                trial = {
                    "profile": identity, "case": f"case-{index}",
                    "status": "passed" if index < 95 else "failed",
                }
                database.execute("INSERT INTO quality_trials VALUES (?, ?)",
                                 (str(index), json.dumps(trial)))
            duplicate = {"profile": identity, "case": "case-99", "status": "passed"}
            database.execute("INSERT INTO quality_trials VALUES (?, ?)", ("retry", json.dumps(duplicate)))
        group = report(self.store)["profiles"][0]
        self.assertEqual(group["success_rate"], 0.95)
        self.assertEqual(group["tasks"], 100)
        self.assertTrue(group["target_observed"])
        self.assertLess(group["wilson_95_interval"][0], 0.95)
        self.assertGreater(group["wilson_95_interval"][1], 0.95)

    def test_pending_trials_count_against_the_target(self):
        with self.store.connection() as database:
            for index in range(100):
                trial = {"profile": profile(configuration()), "case": str(index),
                         "status": "passed" if index < 99 else "pending"}
                database.execute("INSERT INTO quality_trials VALUES (?, ?)", (str(index), json.dumps(trial)))
        group = report(self.store)["profiles"][0]
        self.assertEqual(group["pending"], 1)
        self.assertFalse(group["target_observed"])

    def test_profile_changes_do_not_receive_old_evaluation_credit(self):
        add_case(self.store, "value-two", "Change VALUE", "Grader passes", ["tests"], [])
        start_trial(self.store, "value-two", configuration(repo="mlx-community/test", revision="a" * 40))
        self.store.checkpoint(None, decisions="Change VALUE")
        self.store.approve(None, "Change VALUE")
        (self.root / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
        self.store.checkpoint(None, changes="Candidate")
        self.store.ready(None)
        save_config(self.home, configuration(model="different"))
        with self.assertRaisesRegex(HarnessError, "profile changed"):
            run_check(self.store, "tests", None)


class ModelLifecycleTests(unittest.TestCase):
    def candidate(self):
        return harness_tests.ModelTests().candidate()

    @contextlib.contextmanager
    def managed(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "dedicatedharness.cli.detect_hardware", return_value=harness_tests.hardware()
        ), patch("dedicatedharness.cli.model_manager.require_stopped"), patch(
            "dedicatedharness.cli.prepare_runtime", return_value=(Path("python"), "0.test")
        ), patch("dedicatedharness.cli.Hub") as hub, contextlib.redirect_stdout(
            io.StringIO()
        ), contextlib.redirect_stderr(io.StringIO()):
            home = Path(temporary)
            old = home / "hub" / "old"
            old.mkdir(parents=True)
            save_config(home, configuration(
                backend="mlx", repo="old", revision="a" * 40, model=str(old),
                runtime_python="python", runtime_version="0.test", snapshot=str(old),
            ))
            candidate = replace(self.candidate(), repo="new", revision="b" * 40)
            hub.return_value.inspect.return_value = candidate
            hub.return_value.download.return_value = home / "hub" / "new"
            yield home, candidate, hub.return_value

    def test_failed_setup_preserves_active_configuration(self):
        with self.managed() as (home, candidate, hub), patch(
            "dedicatedharness.cli.model_manager.verify_model", side_effect=HarnessError("load failed")
        ):
            before = (home / "config.json").read_bytes()
            self.assertEqual(main(["--home", str(home), "setup", "--repo", candidate.repo, "--yes"]), 1)
            self.assertEqual((home / "config.json").read_bytes(), before)
            self.assertIn(candidate.repo + "@" + candidate.revision, inventory(home))

    def test_staged_candidate_does_not_replace_active_configuration(self):
        with self.managed() as (home, candidate, hub), patch(
            "dedicatedharness.cli.model_manager.verify_model"
        ) as verify, patch("dedicatedharness.cli.model_manager.evidence", return_value={}):
            before = (home / "config.json").read_bytes()
            self.assertEqual(main([
                "--home", str(home), "upgrade", "--repo", candidate.repo, "--stage", "--yes",
            ]), 0)
            verify.assert_called_once()
            self.assertEqual((home / "config.json").read_bytes(), before)

    def test_unmeasured_upgrade_never_downloads_or_deletes_the_current_model(self):
        with self.managed() as (home, candidate, hub), patch(
            "dedicatedharness.cli.model_manager.evidence", return_value={}
        ), patch("dedicatedharness.cli.model_manager.cleanup_previous") as cleanup:
            before = (home / "config.json").read_bytes()
            self.assertEqual(main([
                "--home", str(home), "upgrade", "--repo", candidate.repo, "--yes",
            ]), 0)
            hub.download.assert_not_called()
            cleanup.assert_not_called()
            self.assertEqual((home / "config.json").read_bytes(), before)

    def test_measured_upgrade_verifies_before_switching_and_cleans_up_afterwards(self):
        with self.managed() as (home, candidate, hub):
            current = load_config(home)
            cases = [str(index) for index in range(100)]
            base = {
                "case_ids": cases, "case_digests": {case: "grader" for case in cases},
                "case_baselines": {case: "baseline" for case in cases},
            }
            scores = {
                current["repo"] + "@" + current["revision"]: {
                    **base, "profile": profile(current), "success_rate": 0.94,
                },
                candidate.repo + "@" + candidate.revision: {
                    **base, "profile": profile(current | {"repo": candidate.repo}),
                    "success_rate": 0.96, "target_observed": True,
                },
            }
            def verify(*_):
                self.assertEqual(load_config(home)["repo"], "old")

            def cleanup(_, previous, active):
                self.assertEqual(previous["repo"], "old")
                self.assertEqual(load_config(home)["revision"], candidate.revision)
                self.assertEqual(active["revision"], candidate.revision)
                return "Owned previous revision cleaned"

            with patch("dedicatedharness.cli.model_manager.evidence", return_value=scores), patch(
                "dedicatedharness.cli.model_manager.verify_model", side_effect=verify
            ), patch("dedicatedharness.cli.model_manager.cleanup_previous", side_effect=cleanup) as cleaned:
                self.assertEqual(main([
                    "--home", str(home), "upgrade", "--repo", candidate.repo, "--yes",
                ]), 0)
                cleaned.assert_called_once()
                hub.inspect.assert_any_call(
                    candidate.repo, harness_tests.hardware(), current["context_tokens"],
                    revision=candidate.revision,
                )

    def test_existing_server_is_never_killed_or_reconfigured(self):
        with tempfile.TemporaryDirectory() as temporary, socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            with self.assertRaisesRegex(HarnessError, "Stop the listener"):
                require_stopped(Path(temporary), f"http://127.0.0.1:{listener.getsockname()[1]}/v1")

    def test_recommendation_refreshes_popular_and_recent_candidates_and_uses_measured_results(self):
        old = replace(self.candidate(), repo="mlx-community/Coder-old", modified_at="2025-01-01")
        new = replace(old, repo="mlx-community/Coder-new", revision="b" * 40, modified_at="2026-10-30")
        too_large = replace(new, repo="mlx-community/Coder-big", fit="too-large")
        hub = Hub(Path("cache"))
        with patch.object(hub, "discover", return_value=[old, new, too_large]) as discovery:
            selected, _ = hub.recommend("coding", harness_tests.hardware(), 4096)
            self.assertEqual(selected.repo, new.repo)
            selected, _ = hub.recommend(
                "coding", harness_tests.hardware(), 4096,
                {old.repo + "@" + old.revision: {"target_observed": True, "success_rate": 0.97}},
            )
            self.assertEqual(selected.repo, old.repo)
        self.assertTrue(any(call.kwargs.get("sort") == "last_modified" for call in discovery.call_args_list))

    def test_upgrade_requires_comparable_observed_improvement(self):
        candidate = self.candidate()
        current = {"repo": "old", "revision": "old"}
        new_key = candidate.repo + "@" + candidate.revision
        scores = {
            "old@old": {"success_rate": 0.94, "case_ids": ["a", "b"]},
            new_key: {"success_rate": 0.95, "case_ids": ["a", "b"], "target_observed": True},
        }
        for score in scores.values():
            score["profile"] = profile(current)
            score["case_digests"] = {"a": "grader-a", "b": "grader-b"}
            score["case_baselines"] = {"a": "baseline-a", "b": "baseline-b"}
        self.assertTrue(upgrade_allowed(current, candidate, scores))
        scores[new_key]["case_ids"] = ["different"]
        self.assertFalse(upgrade_allowed(current, candidate, scores))
        self.assertFalse(upgrade_allowed(current, candidate, {}))
        scores[new_key]["case_ids"] = ["a", "b"]
        scores[new_key]["case_digests"] = {"a": "changed-grader", "b": "grader-b"}
        self.assertFalse(upgrade_allowed(current, candidate, scores))

    def test_cleanup_refuses_unowned_models_and_uses_hub_revision_cleanup_for_owned(self):
        candidate = self.candidate()
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            old = {"repo": candidate.repo, "revision": candidate.revision}
            new = {"repo": "other", "revision": "new"}
            with patch("huggingface_hub.scan_cache_dir") as scan:
                self.assertIn("not recorded", cleanup_previous(home, old, new))
                scan.assert_not_called()
                record_owned(home, candidate, home / "hub" / "snapshot")
                cache = scan.return_value
                cache.repos = [SimpleNamespace(
                    repo_type="model", repo_id=candidate.repo,
                    revisions=[SimpleNamespace(commit_hash=candidate.revision)],
                )]
                self.assertIn("Removed only", cleanup_previous(home, old, new))
                cache.delete_revisions.assert_called_once_with(candidate.revision)
                cache.delete_revisions.return_value.execute.assert_called_once()

    def test_sampling_validation_is_not_a_training_or_accuracy_claim(self):
        validate_sampling({"temperature": 0.2, "top_p": 0.9, "seed": 3})
        for value in ({"temperature": float("nan")}, {"top_p": 0}, {"seed": "3"}, {"evil": "x"}):
            with self.assertRaises(HarnessError):
                validate_sampling(value)

    def test_sampling_profile_change_invalidates_tool_verification(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            home = Path(temporary)
            save_config(home, configuration(tools_verified=True))
            self.assertEqual(main([
                "--home", str(home), "quality", "profile", "--temperature", "0.2",
                "--top-p", "0.9", "--seed", "5",
            ]), 0)
            config = load_config(home)
            self.assertFalse(config["tools_verified"])
            self.assertEqual(config["sampling"], {"temperature": 0.2, "top_p": 0.9, "seed": 5})

    def test_global_model_discovery_does_not_require_a_project(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(evidence(root / "private", root), {})

    def test_approved_runner_reports_timeout_and_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertTrue(execute([sys.executable, "-c", "import time; time.sleep(20)"], root, 1)["timed_out"])
            self.assertEqual(execute([sys.executable, "-c", "raise SystemExit(4)"], root, 2)["exit_code"], 4)
            flood = execute(
                [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'x'*(9*1024*1024))"],
                root, 5,
            )
            self.assertTrue(flood["output_limited"])
            self.assertLessEqual(len(flood["output_tail"]), 8192)


if __name__ == "__main__":
    unittest.main()
