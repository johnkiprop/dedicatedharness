from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import uuid
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

from .prompts import read_context
from .repository import inspect as inspect_repository
from .system import HarnessError

Phase = Literal["investigate", "implement", "validate", "done"]
CheckStatus = Literal["pending", "passed", "failed", "blocked"]
MAX_TEXT_BYTES = 1000
MAX_EXCHANGE_BYTES = 12 * 1024


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def text(value: str, label: str, *, required: bool = True, limit: int = MAX_TEXT_BYTES) -> str:
    if not isinstance(value, str):
        raise HarnessError(f"{label} must be text.")
    value = value.strip()
    if required and not value:
        raise HarnessError(f"{label} must not be empty.")
    if len(value.encode("utf-8")) > limit:
        raise HarnessError(f"{label} exceeds {limit} UTF-8 bytes; write a compact summary.")
    return value


@dataclass
class Project:
    name: str
    goal: str
    constraints: str
    root: str


@dataclass
class Validation:
    status: CheckStatus = "pending"
    evidence: str = ""
    change_revision: int = 0
    recorded_at: str = ""
    run_id: str = ""


@dataclass
class Exchange:
    question: str
    answer: str
    recorded_at: str


@dataclass
class Task:
    id: str
    goal: str
    done_when: str
    files: list[str]
    checks: dict[str, Validation]
    phase: Phase = "investigate"
    findings: str = ""
    decisions: str = ""
    changes: str = ""
    next_action: str = "Investigate and explain before proposing changes."
    approval_scope: str = ""
    change_revision: int = 0
    revision: int = 1
    last_exchange: Exchange | None = None
    updated_at: str = field(default_factory=now)
    repository: dict[str, Any] = field(default_factory=dict)
    trial_id: str = ""


def task_from_json(document: str) -> Task:
    value = json.loads(document)
    if not isinstance(value, dict):
        raise HarnessError("Invalid task checkpoint.")
    checks = value.get("checks")
    if not isinstance(checks, dict) or not checks:
        raise HarnessError("Invalid task validation inventory.")
    try:
        value["checks"] = {name: Validation(**check) for name, check in checks.items()}
        if value.get("last_exchange") is not None:
            value["last_exchange"] = Exchange(**value["last_exchange"])
        task = Task(**value)
    except TypeError as error:
        raise HarnessError("Invalid task schema.") from error
    if task.phase not in ("investigate", "implement", "validate", "done"):
        raise HarnessError("Invalid task phase.")
    if any(check.status not in ("pending", "passed", "failed", "blocked")
           for check in task.checks.values()):
        raise HarnessError("Invalid validation status.")
    return task


class ProjectStore:
    def __init__(self, home: Path, root: Path):
        root = root.expanduser().resolve(strict=True)
        if not root.is_dir():
            raise HarnessError("Project --root must be an existing directory.")
        self.root = root
        self.home = home.resolve()
        if home.resolve().is_relative_to(root):
            raise HarnessError("Keep --home outside the project root to avoid tracking the state database itself.")
        key = hashlib.sha256(os.path.normcase(str(root)).encode("utf-8")).hexdigest()
        self.path = home / "projects" / key / "state.sqlite3"

    @contextmanager
    def connection(self, *, create: bool = False) -> Iterator[sqlite3.Connection]:
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        elif not self.path.exists():
            raise HarnessError("Project is not initialized for this root; run 'dh project init'.")
        with closing(sqlite3.connect(self.path, timeout=5)) as database:
            database.execute("BEGIN IMMEDIATE")
            with database:
                if create:
                    database.execute(
                        "CREATE TABLE IF NOT EXISTS project "
                        "(id INTEGER PRIMARY KEY CHECK (id = 1), document TEXT NOT NULL, active TEXT)"
                    )
                    database.execute(
                        "CREATE TABLE IF NOT EXISTS tasks "
                        "(id TEXT PRIMARY KEY, document TEXT NOT NULL)"
                    )
                    database.execute(
                        "CREATE TABLE IF NOT EXISTS events "
                        "(sequence INTEGER PRIMARY KEY, task_id TEXT, kind TEXT NOT NULL, "
                        "document TEXT NOT NULL, recorded_at TEXT NOT NULL)"
                    )
                    version = database.execute("PRAGMA user_version").fetchone()[0]
                    if version not in (0, 1):
                        raise HarnessError("Unsupported project database version.")
                    database.execute("PRAGMA user_version = 1")
                elif database.execute("PRAGMA user_version").fetchone()[0] != 1:
                    raise HarnessError("Unsupported project database version.")
                yield database

    def profile(self, database: sqlite3.Connection) -> Project:
        row = database.execute("SELECT document FROM project WHERE id = 1").fetchone()
        if row is None:
            raise HarnessError("Project initialization is incomplete; run 'dh project init'.")
        try:
            project = Project(**json.loads(row[0]))
        except TypeError as error:
            raise HarnessError("Invalid project schema.") from error
        if os.path.normcase(project.root) != os.path.normcase(str(self.root)):
            raise HarnessError("Project root does not match stored state.")
        return project

    def selected(self, database: sqlite3.Connection, task_id: str | None) -> Task:
        self.profile(database)
        if task_id is None:
            task_id = database.execute("SELECT active FROM project WHERE id = 1").fetchone()[0]
        row = database.execute(
            "SELECT document FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if row is None:
            raise HarnessError("No matching task. Run 'dh task new' or select an existing task.")
        task = task_from_json(row[0])
        repository = inspect_repository(self.root)
        if repository is not None:
            current = repository.summary()
            old = task.repository
            if old.get("fingerprint") != current["fingerprint"]:
                if old:
                    if old.get("head") != current["head"] or old.get("branch") != current["branch"]:
                        self.reset_plan(task)
                    else:
                        self.reset_checks(task)
                        if task.phase == "done":
                            self.reset_plan(task)
                        elif task.phase == "validate":
                            task.phase = "implement"
                    task.change_revision += 1
                    task.last_exchange = None
                    task.next_action = "Git changed. Inspect the diff and record the current implementation before validation."
                task.repository = current
                task.revision += 1
                self.save(database, task, "git-snapshot-updated")
        return task

    def save(self, database: sqlite3.Connection, task: Task, kind: str) -> None:
        task.updated_at = now()
        database.execute(
            "INSERT INTO tasks (id, document) VALUES (?, ?) "
            "ON CONFLICT(id) DO UPDATE SET document = excluded.document",
            (task.id, json.dumps(asdict(task))),
        )
        snapshot = asdict(task)
        # Audit checkpoint state, never accumulate conversation transcripts.
        snapshot.pop("last_exchange")
        database.execute(
            "INSERT INTO events (task_id, kind, document, recorded_at) VALUES (?, ?, ?, ?)",
            (task.id, kind, json.dumps(snapshot), task.updated_at),
        )

    def initialize(self, name: str, goal: str, constraints: str) -> Project:
        project = Project(
            text(name, "Project name", limit=120), text(goal, "Project goal"),
            text(constraints, "Project constraints", required=False), str(self.root),
        )
        with self.connection(create=True) as database:
            if database.execute("SELECT 1 FROM project").fetchone():
                raise HarnessError("Project already initialized; use 'dh project update'.")
            database.execute(
                "INSERT INTO project (id, document) VALUES (1, ?)",
                (json.dumps(asdict(project)),),
            )
        return project

    def update_project(
        self, name: str | None, goal: str | None, constraints: str | None,
    ) -> Project:
        if all(value is None for value in (name, goal, constraints)):
            raise HarnessError("Supply --name, --goal, or --constraints.")
        with self.connection() as database:
            project = self.profile(database)
            changed_plan = False
            for key, value in (("name", name), ("goal", goal), ("constraints", constraints)):
                if value is not None:
                    checked = text(value, key, required=key != "constraints",
                                   limit=120 if key == "name" else MAX_TEXT_BYTES)
                    changed_plan |= key != "name" and checked != getattr(project, key)
                    setattr(project, key, checked)
            database.execute(
                "UPDATE project SET document = ? WHERE id = 1", (json.dumps(asdict(project)),)
            )
            if changed_plan:
                rows = database.execute("SELECT document FROM tasks").fetchall()
                for row in rows:
                    task = task_from_json(row[0])
                    if task.phase != "done":
                        self.reset_plan(task)
                        task.revision += 1
                        self.save(database, task, "project-plan-updated")
        return project

    def file_specs(self, specs: list[str]) -> list[str]:
        if len(specs) > 12:
            raise HarnessError("A task may reference at most 12 files.")
        normalized: list[str] = []
        for spec in specs:
            text(spec, "File reference", limit=500)
            name, separator, bounds = spec.partition("::")
            candidate = (self.root / name).resolve(strict=True)
            if not candidate.is_relative_to(self.root):
                raise HarnessError("Task files must stay inside the project root.")
            reference = str(candidate.relative_to(self.root))
            if separator:
                reference += "::" + bounds
            normalized.append(reference)
        normalized = list(dict.fromkeys(normalized))
        read_context(self.root, normalized)
        return normalized

    def new(self, goal: str, done_when: str, files: list[str], checks: list[str]) -> Task:
        goal = text(goal, "Task goal")
        done_when = text(done_when, "Completion condition")
        if not checks or len(checks) > 8 or len(set(checks)) != len(checks):
            raise HarnessError("Choose 1-8 distinct required validation checks.")
        if any(not re.fullmatch(r"[a-z][a-z0-9-]{0,47}", name) for name in checks):
            raise HarnessError("Check names use lowercase letters, digits, and hyphens.")
        task = Task(
            uuid.uuid4().hex[:12], goal, done_when, self.file_specs(files),
            {name: Validation() for name in checks},
        )
        with self.connection() as database:
            self.profile(database)
            if database.execute("SELECT 1 FROM tasks WHERE id = ?", (task.id,)).fetchone():
                raise HarnessError("Task identifier collision; retry task creation.")
            self.save(database, task, "created")
            database.execute("UPDATE project SET active = ? WHERE id = 1", (task.id,))
            task = self.selected(database, task.id)
        return task

    def snapshot(self, task_id: str | None = None) -> tuple[Project, Task]:
        with self.connection() as database:
            return self.profile(database), self.selected(database, task_id)

    def show_project(self) -> dict[str, Any]:
        with self.connection() as database:
            project = self.profile(database)
            active = database.execute("SELECT active FROM project WHERE id = 1").fetchone()[0]
            repository = inspect_repository(self.root)
            return {"project": asdict(project), "active_task": active, "state_file": str(self.path),
                    "repository": repository.summary() if repository else None}

    def list_tasks(self) -> list[dict[str, Any]]:
        with self.connection() as database:
            self.profile(database)
            rows = database.execute(
                "SELECT document FROM tasks ORDER BY rowid DESC LIMIT 100"
            ).fetchall()
            return [
                {"id": task.id, "goal": task.goal, "phase": task.phase,
                 "updated_at": task.updated_at}
                for task in (task_from_json(row[0]) for row in rows)
            ]

    def use(self, task_id: str) -> Task:
        with self.connection() as database:
            task = self.selected(database, task_id)
            database.execute("UPDATE project SET active = ? WHERE id = 1", (task.id,))
            return task

    @staticmethod
    def reset_checks(task: Task) -> None:
        task.checks = {
            name: Validation(change_revision=task.change_revision) for name in task.checks
        }

    @classmethod
    def reset_plan(cls, task: Task) -> None:
        task.approval_scope = ""
        task.phase = "investigate"
        task.last_exchange = None
        task.next_action = "Investigate the updated plan and request explicit approval."
        cls.reset_checks(task)

    def checkpoint(self, task_id: str | None, **sections: Any) -> Task:
        allowed = {"goal", "done_when", "findings", "decisions", "changes", "next_action", "files"}
        if not sections or not set(sections) <= allowed:
            raise HarnessError("Supply at least one checkpoint section.")
        with self.connection() as database:
            task = self.selected(database, task_id)
            if task.phase == "done":
                raise HarnessError("Completed tasks are immutable; start a new task.")
            values: dict[str, Any] = {}
            for key, value in sections.items():
                values[key] = (
                    self.file_specs(value) if key == "files"
                    else text(value, key, required=key in ("goal", "done_when"))
                )
            plan_changed = any(
                key in values and values[key] != getattr(task, key)
                for key in ("goal", "done_when", "decisions", "files")
            )
            code_recorded = "changes" in values
            if code_recorded and (
                plan_changed or not task.approval_scope
                or task.phase not in ("implement", "validate")
            ):
                raise HarnessError(
                    "Record/revise the plan first, then explicitly 'dh task approve --scope ...' "
                    "before recording implementation changes."
                )
            for key, value in values.items():
                setattr(task, key, value)
            if plan_changed:
                self.reset_plan(task)
                if "next_action" in values:
                    task.next_action = values["next_action"]
            if code_recorded:
                task.change_revision += 1
                self.reset_checks(task)
                task.phase = "implement"
                if "next_action" not in values:
                    task.next_action = "Review the implementation, then run dh task ready."
            task.last_exchange = None
            task.revision += 1
            self.save(database, task, "checkpoint")
            return task

    def approve(self, task_id: str | None, scope: str) -> Task:
        scope = text(scope, "Approval scope")
        with self.connection() as database:
            task = self.selected(database, task_id)
            if task.phase != "investigate":
                raise HarnessError("Only an investigation can be approved; revise the plan first.")
            if not task.decisions:
                raise HarnessError("Record a proposed decision/plan before approving it.")
            task.approval_scope = scope
            task.phase = "implement"
            task.next_action = "Implement only the approved scope, then record the changes."
            task.last_exchange = None
            task.revision += 1
            self.save(database, task, "approved")
            return task

    def ready(self, task_id: str | None) -> Task:
        with self.connection() as database:
            task = self.selected(database, task_id)
            if task.phase != "implement" or not task.approval_scope or not task.changes:
                raise HarnessError("Validation requires an approved task with recorded implementation.")
            task.phase = "validate"
            task.next_action = "Run required checks; keep device validation pending until reported."
            task.last_exchange = None
            task.revision += 1
            self.save(database, task, "ready-for-validation")
            return task

    def validate(
        self, task_id: str | None, name: str, status: CheckStatus, evidence: str,
    ) -> Task:
        if status not in ("pending", "passed", "failed", "blocked"):
            raise HarnessError("Unknown validation status.")
        evidence = text(evidence, "Validation evidence", required=status != "pending")
        with self.connection() as database:
            task = self.selected(database, task_id)
            from .quality import enabled
            if status == "passed" and enabled(database):
                raise HarnessError("Verified mode requires 'dh quality run CHECK'; manual passing claims are not accepted.")
            if task.phase != "validate":
                raise HarnessError("Run 'dh task ready' after recording implementation.")
            if name not in task.checks:
                raise HarnessError("Unknown check; use a required check from this task.")
            task.checks[name] = Validation(status, evidence, task.change_revision, now())
            task.last_exchange = None
            task.revision += 1
            self.save(database, task, "validation-reported")
            return task

    def finish(self, task_id: str | None) -> Task:
        with self.connection() as database:
            task = self.selected(database, task_id)
            if task.phase != "validate" or not task.approval_scope or not task.changes:
                raise HarnessError("Completion requires an approved implementation in validation.")
            pending = [
                name for name, check in task.checks.items()
                if check.status != "passed" or check.change_revision != task.change_revision
            ]
            if pending:
                raise HarnessError("Cannot finish; required checks not passed: " + ", ".join(pending))
            from .quality import verify_completion
            verify_completion(database, task)
            task.phase = "done"
            task.next_action = "Completed according to reported evidence; no new work is authorized."
            task.last_exchange = None
            task.revision += 1
            self.save(database, task, "completed")
            return task

    def remember(self, task_id: str, revision: int, question: str, answer: str) -> None:
        question = text(question, "Remembered question", limit=4096)
        answer = text(answer, "Remembered answer", limit=8192)
        if len((question + answer).encode("utf-8")) > MAX_EXCHANGE_BYTES:
            raise HarnessError("Exchange exceeds the local continuity limit; record a compact checkpoint.")
        with self.connection() as database:
            task = self.selected(database, task_id)
            if task.revision != revision:
                raise HarnessError(
                    "Task changed during inference; answer was produced but not remembered. "
                    "Review the latest checkpoint before continuing."
                )
            task.last_exchange = Exchange(question, answer, now())
            task.revision += 1
            self.save(database, task, "latest-exchange-replaced")

    def forget(self, task_id: str | None) -> Task:
        with self.connection() as database:
            task = self.selected(database, task_id)
            task.last_exchange = None
            task.revision += 1
            self.save(database, task, "latest-exchange-forgotten")
            return task

    def history(self, task_id: str | None, limit: int) -> list[dict[str, Any]]:
        if not 1 <= limit <= 100:
            raise HarnessError("History --limit must be between 1 and 100.")
        with self.connection() as database:
            task = self.selected(database, task_id)
            rows = database.execute(
                "SELECT sequence, kind, document, recorded_at FROM events "
                "WHERE task_id = ? ORDER BY sequence DESC LIMIT ?", (task.id, limit),
            ).fetchall()
            return [
                {"sequence": row[0], "kind": row[1], "checkpoint": json.loads(row[2]),
                 "recorded_at": row[3]} for row in rows
            ]


def context_for(project: Project, task: Task) -> str:
    return (
        "Project/task checkpoint (user-maintained claims, not independently verified):\n"
        + json.dumps({"project": asdict(project), "task": asdict(task)}, ensure_ascii=False)
        + "\nValidation evidence is user-reported, not executed by Dedicated Harness. "
        "Recorded code changes do not mean deployed or verified on a device. "
        "Any remembered answer is unverified model output, not approval or test evidence."
    )


def handoff(project: Project, task: Task, target: str, max_bytes: int = 8192) -> str:
    from .quality import CODING_CONTRACT
    if target not in ("opencode", "openclaw"):
        raise HarnessError("Unsupported handoff target.")
    if not 1 <= max_bytes <= 32768:
        raise HarnessError("Handoff --max-bytes must be between 1 and 32768.")
    mode = {
        "investigate": "Investigate and propose only. Do not edit code or self-approve.",
        "implement": "Implement only the explicit approved scope; ask before expanding it.",
        "validate": "Verify the recorded implementation. Report failures; do not expand scope.",
        "done": "Task is closed. Explain its state only; new work requires a new task.",
    }[task.phase]
    result = (
        f"Dedicated Harness handoff for {target}\n"
        f"Work in the project root: {project.root}\n"
        f"Workflow: {mode}\n"
        "Use your existing tool permissions and the configured local model. "
        "This handoff grants no additional execution permissions.\n"
        "Inspect current code before relying on checkpoint claims. "
        "Read recorded file references as needed; their contents are not embedded here.\n\n"
        + CODING_CONTRACT + "\n\n"
        + context_for(project, task)
        + "\n\nReturn concise findings, changed files, checks actually run with evidence, "
        "and the next action. Keep pending device validation distinct from emulator results. "
        "Ask the user to record a compact dh task checkpoint; do not approve your own work "
        "or invent validation evidence.\n"
    )
    if len(result.encode("utf-8")) > max_bytes:
        raise HarnessError(
            "Handoff exceeds its byte budget; compact checkpoint sections or forget the "
            "last exchange. Nothing was silently truncated."
        )
    return result
