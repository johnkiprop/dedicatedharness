from __future__ import annotations

import contextlib
import io
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from dedicatedharness.cli import main
from dedicatedharness.projects import (
    ProjectStore, context_for, handoff, text,
)
from dedicatedharness.system import HarnessError, save_config
from test_harness import FakeServer, configuration


class ProjectStateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "source"
        self.root.mkdir()
        (self.root / "example.py").write_text("first\nsecond\nthird\n", encoding="utf-8")
        self.home = self.base / "private"
        self.store = ProjectStore(self.home, self.root)
        self.store.initialize("Example", "Understand and improve the app", "No new dependencies")
        self.task = self.store.new(
            "Investigate heating", "Emulator and phone validation pass",
            ["example.py::1-2"], ["emulator", "phone"],
        )

    def plan(self):
        return self.store.checkpoint(
            None, findings="Polling causes repeated work", decisions="Reduce polling frequency",
        )

    def implementation(self):
        self.plan()
        self.store.approve(None, "Reduce polling only; preserve public API")
        return self.store.checkpoint(None, changes="Changed polling interval in example.py")

    def validation(self):
        self.implementation()
        return self.store.ready(None)

    def test_project_state_survives_reopen_and_stays_outside_repository(self):
        reopened = ProjectStore(self.home, self.root)
        project, task = reopened.snapshot()
        self.assertEqual(task.id, self.task.id)
        self.assertEqual(project.name, "Example")
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["example.py"])
        self.assertEqual(task.phase, "investigate")
        self.assertEqual(task.checks["phone"].status, "pending")
        self.assertEqual(task.approval_scope, "")
        self.assertTrue(reopened.path.is_relative_to(self.home))
        with self.assertRaises(HarnessError):
            reopened.initialize("Overwrite", "Different goal", "")
        self.assertEqual(reopened.snapshot()[0].goal, "Understand and improve the app")

    def test_different_roots_and_homes_do_not_share_context(self):
        other = self.base / "other"
        other.mkdir()
        for store in (
            ProjectStore(self.home, other), ProjectStore(self.base / "other-home", self.root),
        ):
            with self.assertRaisesRegex(HarnessError, "not initialized"):
                store.snapshot()

    def test_task_selection_preserves_both_tasks(self):
        second = self.store.new("Fix login", "Login tests pass", [], ["tests"])
        self.assertEqual(self.store.snapshot()[1].id, second.id)
        self.store.use(self.task.id)
        self.assertEqual(self.store.snapshot()[1].goal, "Investigate heating")
        self.assertEqual(len(self.store.list_tasks()), 2)
        with self.assertRaises(HarnessError):
            self.store.use("does-not-exist")

    def test_approval_is_explicit_and_requires_a_plan(self):
        with self.assertRaisesRegex(HarnessError, "proposed decision"):
            self.store.approve(None, "Go ahead")
        with self.assertRaisesRegex(HarnessError, "explicitly"):
            self.store.checkpoint(None, changes="I changed the code")
        self.plan()
        approved = self.store.approve(None, "Reduce polling only")
        self.assertEqual(approved.phase, "implement")
        self.assertEqual(approved.approval_scope, "Reduce polling only")
        with self.assertRaises(HarnessError):
            self.store.approve(None, "Different scope")

    def test_changed_plan_revokes_approval_and_evidence_atomically(self):
        self.validation()
        self.store.validate(None, "emulator", "passed", "Emulator regression passed")
        updated = self.store.checkpoint(None, decisions="Use a different algorithm")
        self.assertEqual(updated.phase, "investigate")
        self.assertEqual(updated.approval_scope, "")
        self.assertEqual(updated.checks["emulator"].status, "pending")
        self.assertEqual(updated.checks["emulator"].evidence, "")
        self.assertIn("request explicit approval", updated.next_action)
        with self.assertRaises(HarnessError):
            self.store.ready(None)

    def test_scope_and_implementation_cannot_be_changed_in_the_same_checkpoint(self):
        self.implementation()
        before = self.store.snapshot()[1]
        with self.assertRaises(HarnessError):
            self.store.checkpoint(None, decisions="New scope", changes="New implementation")
        after = self.store.snapshot()[1]
        self.assertEqual(before.revision, after.revision)
        self.assertEqual(before.decisions, after.decisions)

    def test_goal_file_and_project_constraint_changes_revoke_approval(self):
        for section in (
            {"goal": "Different goal"}, {"done_when": "A new condition"},
            {"files": ["example.py::2-3"]},
        ):
            with self.subTest(section=section):
                self.plan()
                self.store.approve(None, "Current plan")
                changed = self.store.checkpoint(None, **section)
                self.assertEqual(changed.phase, "investigate")
                self.assertFalse(changed.approval_scope)
        self.plan()
        self.store.approve(None, "Current plan")
        self.store.update_project(None, None, "No polling")
        self.assertFalse(self.store.snapshot()[1].approval_scope)
        self.assertEqual(self.store.snapshot()[0].constraints, "No polling")

    def test_project_rename_does_not_revoke_approval(self):
        self.plan()
        self.store.approve(None, "Current plan")
        self.store.update_project("New name", None, None)
        self.assertEqual(self.store.snapshot()[1].phase, "implement")

    def test_emulator_success_never_means_phone_success_or_deployment(self):
        self.validation()
        result = self.store.validate(None, "emulator", "passed", "Emulator run 123 passed")
        self.assertEqual(result.checks["emulator"].status, "passed")
        self.assertEqual(result.checks["phone"].status, "pending")
        with self.assertRaisesRegex(HarnessError, "phone"):
            self.store.finish(None)
        project, task = self.store.snapshot()
        prompt = context_for(project, task)
        self.assertIn("not mean deployed", prompt)
        self.assertIn('"phone": {"status": "pending"', prompt)

    def test_validation_requires_evidence_and_expected_check(self):
        self.validation()
        for name, status, evidence in (
            ("emulator", "passed", ""), ("emulator", "failed", ""),
            ("emulator", "blocked", ""), ("unknown", "passed", "Evidence"),
        ):
            with self.subTest(name=name, status=status), self.assertRaises(HarnessError):
                self.store.validate(None, name, status, evidence)
        self.store.validate(None, "phone", "blocked", "Phone not connected")
        self.assertEqual(self.store.snapshot()[1].checks["phone"].status, "blocked")

    def test_implementation_update_invalidates_checks_but_keeps_audit(self):
        self.validation()
        self.store.validate(None, "emulator", "passed", "Old emulator evidence")
        updated = self.store.checkpoint(None, changes="A second implementation checkpoint")
        self.assertEqual(updated.change_revision, 2)
        self.assertEqual(updated.phase, "implement")
        self.assertTrue(all(check.status == "pending" for check in updated.checks.values()))
        audit = self.store.history(None, 20)
        self.assertIn("Old emulator evidence", json.dumps(audit))
        self.assertNotIn("Old emulator evidence", json.dumps(
            {name: check.evidence for name, check in updated.checks.items()}
        ))

    def test_explicit_change_always_invalidates_checks_even_for_identical_summary(self):
        self.validation()
        self.store.validate(None, "emulator", "passed", "Emulator passed")
        previous = self.store.snapshot()[1]
        updated = self.store.checkpoint(None, changes=previous.changes)
        self.assertEqual(updated.change_revision, previous.change_revision + 1)
        self.assertEqual(updated.checks["emulator"].status, "pending")

    def test_finish_requires_all_checks_for_current_implementation(self):
        self.validation()
        self.store.validate(None, "emulator", "passed", "Emulator passed")
        self.store.validate(None, "phone", "passed", "Installed build tested on phone")
        task = self.store.finish(None)
        self.assertEqual(task.phase, "done")
        with self.assertRaisesRegex(HarnessError, "immutable"):
            self.store.checkpoint(None, findings="Unrelated new work")

    def test_remember_is_opt_in_latest_only_and_never_promotes_model_claims(self):
        self.store.remember(
            self.task.id, self.task.revision, "What should change?", "Run less often",
        )
        task = self.store.snapshot()[1]
        self.assertEqual(task.last_exchange.answer, "Run less often")
        self.assertFalse(task.approval_scope)
        self.assertEqual(task.findings, "")
        self.store.remember(task.id, task.revision, "Why?", "Because polling repeats work")
        task = self.store.snapshot()[1]
        self.assertEqual(task.last_exchange.question, "Why?")
        self.assertNotIn("Run less often", json.dumps(self.store.history(None, 20)))
        self.assertNotIn("Because polling repeats work", json.dumps(self.store.history(None, 20)))
        forgotten = self.store.forget(None)
        self.assertIsNone(forgotten.last_exchange)

    def test_checkpoint_compacts_and_replaces_sections_without_silent_append(self):
        self.store.remember(self.task.id, self.task.revision, "Question", "Answer")
        first = self.store.checkpoint(None, findings="Finding one")
        self.assertIsNone(first.last_exchange)
        second = self.store.checkpoint(None, findings="Finding two")
        self.assertEqual(second.findings, "Finding two")
        self.assertIn("Finding one", json.dumps(self.store.history(None, 10)))

    def test_explicit_next_action_survives_plan_invalidation(self):
        task = self.store.checkpoint(
            None, decisions="Compare two approaches", next_action="Explain their tradeoffs",
        )
        self.assertEqual(task.next_action, "Explain their tradeoffs")
        self.assertEqual(task.phase, "investigate")

    def test_stale_query_cannot_overwrite_a_newer_checkpoint(self):
        snapshot = self.store.snapshot()[1]
        self.store.checkpoint(None, findings="New information arrived")
        with self.assertRaisesRegex(HarnessError, "changed during inference"):
            self.store.remember(snapshot.id, snapshot.revision, "Old question", "Old answer")
        self.assertEqual(self.store.snapshot()[1].findings, "New information arrived")
        self.assertIsNone(self.store.snapshot()[1].last_exchange)

    def test_handoff_respects_phase_and_does_not_embed_file_contents(self):
        project, task = self.store.snapshot()
        for target in ("opencode", "openclaw"):
            prompt = handoff(project, task, target)
            self.assertIn("Do not edit code or self-approve", prompt)
            self.assertIn("example.py::1-2", prompt)
            self.assertNotIn("first\nsecond", prompt)
            self.assertIn("configured local model", prompt)
        self.plan()
        self.store.approve(None, "Reduce polling only")
        project, task = self.store.snapshot()
        self.assertIn("explicit approved scope", handoff(project, task, "opencode"))
        with self.assertRaisesRegex(HarnessError, "silently truncated"):
            handoff(project, task, "opencode", max_bytes=100)

    def test_file_boundaries_size_limits_and_invalid_inputs(self):
        outside = self.base / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        for specs in (
            [str(Path("..") / "outside.txt")], ["example.py::0-2"], ["example.py::2-9"],
        ):
            with self.subTest(specs=specs), self.assertRaises(HarnessError):
                self.store.checkpoint(None, files=specs)
        for checks in ([], ["phone", "phone"], ["Phone test"], ["test"] * 9):
            with self.assertRaises(HarnessError):
                self.store.new("Goal", "Done", [], checks)
        with self.assertRaises(HarnessError):
            text("\u4e2d" * 334, "Unicode summary")
        with self.assertRaises(HarnessError):
            self.store.checkpoint(None)
        for limit in (0, 101):
            with self.assertRaises(HarnessError):
                self.store.history(None, limit)


class ProjectCLIAndHTTPTests(unittest.TestCase):
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

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "root"
        self.root.mkdir()
        (self.root / "source.py").write_text("selected source content\n", encoding="utf-8")
        self.home = self.base / "home"
        self.prefix = ["--home", str(self.home)]

    def cli(self, *arguments, expected=0):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            result = main([*self.prefix, *arguments])
        self.assertEqual(result, expected, err.getvalue())
        return out.getvalue(), err.getvalue()

    def create_task(self):
        self.cli("project", "init", "--root", str(self.root), "--goal", "Improve app reliability")
        output, _ = self.cli(
            "task", "new", "Investigate heating", "--root", str(self.root),
            "--done", "Emulator and phone checks pass", "--file", "source.py",
            "--check", "emulator", "--check", "phone",
        )
        return json.loads(output)["task"]

    def test_complete_engineering_lifecycle_and_safe_handoff_output(self):
        self.create_task()
        common = ["--root", str(self.root)]
        self.cli("task", "checkpoint", *common, "--finding", "Polling repeats work",
                 "--decision", "Reduce polling")
        self.cli("task", "approve", *common, "--scope", "Reduce polling only")
        destination = self.base / "handoff.txt"
        self.cli("task", "handoff", "opencode", *common, "--out", str(destination))
        self.assertIn("Reduce polling only", destination.read_text(encoding="utf-8"))
        self.cli("task", "handoff", "opencode", *common, "--out", str(destination), expected=1)
        self.cli("task", "checkpoint", *common, "--change", "Updated polling interval")
        self.cli("task", "ready", *common)
        self.cli("task", "validate", "emulator", *common, "--status", "passed",
                 "--evidence", "Emulator test passed")
        self.cli("task", "finish", *common, expected=1)
        self.cli("task", "validate", "phone", *common, "--status", "passed",
                 "--evidence", "Phone build installed and verified")
        output, _ = self.cli("task", "finish", *common)
        self.assertEqual(json.loads(output)["task"]["phase"], "done")
        self.assertFalse((self.home / "config.json").exists())

    def test_followup_loads_state_selected_files_and_only_opted_in_last_exchange(self):
        initial = self.create_task()
        save_config(self.home, configuration(base_url=self.url, context_tokens=12000))
        self.cli("ask", "What caused the heating?", "--project", "--root", str(self.root))
        store = ProjectStore(self.home, self.root)
        self.assertIsNone(store.snapshot()[1].last_exchange)
        self.cli("ask", "What should we do?", "--project", "--remember", "--root", str(self.root))
        task = store.snapshot()[1]
        self.assertEqual(task.last_exchange.question, "What should we do?")
        self.assertFalse(task.approval_scope)
        self.cli("ask", "okay proceed", "--project", "--root", str(self.root))
        request = FakeServer.requests_seen[-1]["messages"][-1]["content"]
        self.assertIn(initial["id"], request)
        self.assertIn("What should we do?", request)
        self.assertIn("selected source content", request)
        self.assertIn('"phone": {"status": "pending"', request)
        self.assertEqual(store.snapshot()[1].phase, "investigate")

    def test_project_admission_dryrun_and_invalid_flags_send_nothing(self):
        self.create_task()
        save_config(self.home, configuration(base_url=self.url, context_tokens=500))
        before = len(FakeServer.requests_seen)
        self.cli("ask", "Why?", "--project", "--root", str(self.root), expected=1)
        self.cli("ask", "Why?", "--remember", "--root", str(self.root), expected=1)
        save_config(self.home, configuration(base_url=self.url, context_tokens=12000))
        self.cli("ask", "Why?", "--project", "--remember", "--dry-run",
                 "--root", str(self.root))
        self.assertEqual(len(FakeServer.requests_seen), before)
        self.assertIsNone(ProjectStore(self.home, self.root).snapshot()[1].last_exchange)

    def test_missing_project_and_removed_selected_file_are_explicit_failures(self):
        save_config(self.home, configuration(base_url=self.url, context_tokens=12000))
        self.cli("ask", "Why?", "--project", "--root", str(self.root), expected=1)
        self.create_task()
        (self.root / "source.py").unlink()
        before = len(FakeServer.requests_seen)
        self.cli("ask", "Why?", "--project", "--root", str(self.root), expected=1)
        self.assertEqual(len(FakeServer.requests_seen), before)

    def test_piped_question_is_remembered_exactly_without_checkpoint_or_code(self):
        self.create_task()
        save_config(self.home, configuration(base_url=self.url, context_tokens=12000))
        question = "Task: Explain polling.\nContext: This is part of the task brief."
        with patch("dedicatedharness.cli.sys.stdin", io.StringIO(question)):
            self.cli("ask", "--project", "--remember", "--root", str(self.root))
        exchange = ProjectStore(self.home, self.root).snapshot()[1].last_exchange
        self.assertEqual(exchange.question, question)
        self.assertNotIn("selected source content", exchange.question)
        self.assertNotIn("Project/task checkpoint", exchange.question)

    def test_explicit_task_selection_and_project_bench_do_not_change_active_task(self):
        first = self.create_task()
        second, _ = self.cli(
            "task", "new", "Different task goal", "--root", str(self.root),
            "--done", "Other checks pass",
        )
        second_id = json.loads(second)["task"]["id"]
        save_config(self.home, configuration(base_url=self.url, context_tokens=12000))
        arguments = [
            "--project", "--root", str(self.root), "--task-id", first["id"],
        ]
        self.cli("ask", "What remains?", *arguments)
        request = FakeServer.requests_seen[-1]["messages"][-1]["content"]
        self.assertIn(first["id"], request)
        self.assertNotIn("Different task goal", request)
        output, _ = self.cli("bench", "Explain the state", *arguments, "--runs", "2")
        report = json.loads(output)
        self.assertEqual(report["runs"][0]["project_task_id"], first["id"])
        store = ProjectStore(self.home, self.root)
        self.assertEqual(store.snapshot()[1].id, second_id)
        self.assertIsNone(store.snapshot(first["id"])[1].last_exchange)
        self.assertEqual(store.snapshot(first["id"])[1].checks["emulator"].status, "pending")


if __name__ == "__main__":
    unittest.main()
