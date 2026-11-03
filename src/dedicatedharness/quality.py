from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from .projects import ProjectStore, Task, Validation, now, text
from .repository import inspect
from .system import HarnessError, load_config

CONTRACT_VERSION = "coding-contract-v1"
CODING_CONTRACT = (
    "Engineering contract: inspect the current implementation and relevant tests first. "
    "Make the smallest scoped fix; reuse existing APIs and helpers. "
    "Add a regression test for the requested behavior and preserve existing trusted tests. "
    "Do not weaken tests, skip checks, or claim success from generated text. "
    "Report actual validation results and unresolved assumptions. "
    "Use only explicitly approved validation commands. "
    "If the local model cannot solve the task reliably, stop and report the blocker."
)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def harness_digest() -> str:
    directory = Path(__file__).parent
    return digest({
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.glob("*.py"))
    })


def profile(config: dict[str, Any], execution: dict[str, Any] | None = None) -> dict[str, Any]:
    data = {key: config.get(key) for key in (
        "backend", "model", "repo", "revision", "runtime_version", "base_url",
        "context_tokens", "max_output", "sampling",
    )}
    data["contract"] = CONTRACT_VERSION
    if execution is not None:
        data["execution"] = execution
    return {"id": digest(data), "configuration": data}


def bind_execution(
    store: ProjectStore, task_id: str, config: dict[str, Any], execution: dict[str, Any], run_id: str,
) -> None:
    with store.connection() as database:
        task = store.selected(database, task_id)
        if not task.trial_id:
            return
        row = database.execute("SELECT document FROM quality_trials WHERE id = ?", (task.trial_id,)).fetchone()
        if not row:
            raise HarnessError("Missing evaluation trial.")
        trial = json.loads(row[0])
        existing = trial["profile"]["configuration"].get("execution")
        if existing is not None:
            if existing != execution:
                raise HarnessError("Trial execution harness changed; start a fresh evaluation trial.")
            if trial["status"] == "pending" and trial.get("execution_run_id") != run_id:
                raise HarnessError("Continue the original supervised trial run; do not reset its attempt budget.")
        elif trial["status"] != "pending" or trial["first_fingerprint"] or trial["results"]:
            raise HarnessError("Cannot relabel an already submitted external trial as supervised.")
        if profile(config)["id"] != profile({
            key: value for key, value in trial["profile"]["configuration"].items()
            if key not in ("execution", "contract")
        })["id"]:
            raise HarnessError("Trial model configuration differs from the supervised run.")
        trial["profile"] = profile(config, execution)
        trial.setdefault("execution_run_id", run_id)
        database.execute("UPDATE quality_trials SET document = ? WHERE id = ?",
                         (json.dumps(trial), task.trial_id))


def enabled(database: sqlite3.Connection) -> bool:
    exists = database.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'quality_settings'"
    ).fetchone()
    return bool(exists and database.execute("SELECT 1 FROM quality_settings").fetchone())


def require_enabled(database: sqlite3.Connection) -> None:
    if not enabled(database):
        raise HarnessError("Run 'dh quality init' for this project first.")


def initialize(store: ProjectStore) -> None:
    inspect(store.root, required=True)
    with store.connection() as database:
        store.profile(database)
        database.execute("CREATE TABLE IF NOT EXISTS quality_settings (document TEXT NOT NULL)")
        database.execute(
            "CREATE TABLE IF NOT EXISTS quality_commands (name TEXT PRIMARY KEY, document TEXT NOT NULL)"
        )
        database.execute(
            "CREATE TABLE IF NOT EXISTS quality_runs (id TEXT PRIMARY KEY, document TEXT NOT NULL)"
        )
        database.execute(
            "CREATE TABLE IF NOT EXISTS quality_cases (id TEXT PRIMARY KEY, document TEXT NOT NULL)"
        )
        database.execute(
            "CREATE TABLE IF NOT EXISTS quality_trials (id TEXT PRIMARY KEY, document TEXT NOT NULL)"
        )
        if not enabled(database):
            database.execute(
                "INSERT INTO quality_settings VALUES (?)",
                (json.dumps({"target": 0.95, "minimum_tasks": 100}),),
            )
            for row in database.execute("SELECT document FROM tasks").fetchall():
                from .projects import task_from_json
                task = task_from_json(row[0])
                store.reset_checks(task)
                task.phase = "validate" if task.approval_scope and task.changes else "investigate"
                task.revision += 1
                store.save(database, task, "verified-mode-enabled")


def protected_files(root: Path, paths: list[str]) -> dict[str, str]:
    repository = inspect(root, required=True)
    assert repository is not None
    chosen: dict[str, str] = {}
    total = 0
    for spec in paths:
        base = (root / spec).resolve(strict=True)
        if not base.is_relative_to(root):
            raise HarnessError("Protected tests must be inside the project root.")
        for name in repository.files:
            path = root / name
            if (path == base or path.is_relative_to(base)) and path.is_file():
                if path.is_symlink() or path.stat().st_size > 1024 * 1024:
                    raise HarnessError("Protected graders must be ordinary files no larger than 1 MiB.")
                if name not in chosen:
                    total += path.stat().st_size
                if total > 32 * 1024 * 1024 or len(chosen) >= 2000:
                    raise HarnessError("Protected graders exceed 32 MiB or 2,000 files.")
                chosen[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if not any((root / name) == base or (root / name).is_relative_to(base) for name in chosen):
            raise HarnessError(f"No non-ignored grader files found for --protect {spec}.")
    if len(chosen) > 2000:
        raise HarnessError("Protected grader inventory exceeds 2,000 files.")
    return chosen


def register_command(
    store: ProjectStore, name: str, argv: list[str], timeout: int, protects: list[str],
) -> dict[str, Any]:
    import re
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,47}", name):
        raise HarnessError("Command names use lowercase letters, digits, and hyphens.")
    if (not isinstance(argv, list) or not argv or len(argv) > 64
            or any(not isinstance(part, str) or "\0" in part for part in argv)):
        raise HarnessError("--argv-json must be a nonempty string array of at most 64 arguments.")
    if not 1 <= timeout <= 1800:
        raise HarnessError("Validation timeout must be between 1 and 1,800 seconds.")
    requested = Path(argv[0])
    executable = (
        shutil.which(str(store.root / requested))
        if not requested.is_absolute() and requested.parent != Path(".") else shutil.which(argv[0])
    )
    if not executable:
        raise HarnessError("Approved executable not found on PATH.")
    command = {
        "name": name, "argv": [str(Path(executable).resolve()), *argv[1:]], "timeout": timeout,
        "protected": protected_files(store.root, protects),
    }
    command["digest"] = digest(command)
    with store.connection() as database:
        require_enabled(database)
        database.execute(
            "INSERT INTO quality_commands VALUES (?, ?) "
            "ON CONFLICT(name) DO UPDATE SET document = excluded.document",
            (name, json.dumps(command)),
        )
        from .projects import task_from_json
        for row in database.execute("SELECT document FROM tasks").fetchall():
            task = task_from_json(row[0])
            if name in task.checks:
                task.checks[name] = Validation(change_revision=task.change_revision)
                if task.phase == "done":
                    task.phase = "validate"
                task.revision += 1
                store.save(database, task, "validation-command-approved")
    return command


def stop_owned_process(process: subprocess.Popen) -> None:
    if os.name == "nt":
        if process.poll() is not None:
            return
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       capture_output=True, timeout=15, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=15)


def execute(argv: list[str], root: Path, timeout: int | float) -> dict[str, Any]:
    start = time.monotonic()
    timed_out = False
    output_limited = False
    with tempfile.TemporaryFile() as output:
        options = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            if os.name == "nt" else {"start_new_session": True}
        )
        process = subprocess.Popen(
            argv, cwd=root, stdin=subprocess.DEVNULL, stdout=output,
            stderr=subprocess.STDOUT, shell=False, **options,
        )
        try:
            while process.poll() is None:
                timed_out = time.monotonic() - start > timeout
                output_limited = os.fstat(output.fileno()).st_size > 8 * 1024 * 1024
                if timed_out or output_limited:
                    break
                time.sleep(0.05)
        finally:
            stop_owned_process(process)
        size = output.seek(0, 2)
        output_limited = output_limited or size > 8 * 1024 * 1024
        output.seek(max(0, size - 8192))
        tail = output.read().decode("utf-8", errors="replace")
    return {
        "exit_code": process.returncode, "timed_out": timed_out,
        "output_limited": output_limited,
        "elapsed_seconds": time.monotonic() - start,
        "output_tail": tail, "output_bytes": size,
    }


def verify_graders(root: Path, command: dict[str, Any]) -> None:
    changed = [
        path for path, expected in command["protected"].items()
        if not (root / path).is_file()
        or (root / path).is_symlink()
        or (root / path).stat().st_size > 1024 * 1024
        or hashlib.sha256((root / path).read_bytes()).hexdigest() != expected
    ]
    if changed:
        raise HarnessError("Trusted graders changed; review and explicitly reapprove the command/test baseline.")


def run_check(
    store: ProjectStore, name: str, task_id: str | None, *, timeout_limit: float | None = None,
    execution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _, task = store.snapshot(task_id)
    if task.phase != "validate" or not task.approval_scope or name not in task.checks:
        raise HarnessError("Select an approved task in validation with this required check.")
    with store.connection() as database:
        require_enabled(database)
        row = database.execute("SELECT document FROM quality_commands WHERE name = ?", (name,)).fetchone()
        if not row:
            raise HarnessError("No explicitly approved command for this check.")
        command = json.loads(row[0])
        trial_row = database.execute(
            "SELECT document FROM quality_trials WHERE id = ?", (task.trial_id,)
        ).fetchone() if task.trial_id else None
    if trial_row:
        trial = json.loads(trial_row[0])
        if trial["profile"]["configuration"].get("execution") != execution:
            raise HarnessError("This trial requires its bound execution supervisor; do not submit manual checks.")
        if trial["profile"]["id"] != profile(load_config(store.home), execution)["id"]:
            raise HarnessError("Model/prompt profile changed during this trial; do not attribute results to the old profile.")
    before = inspect(store.root, required=True)
    assert before is not None
    verify_graders(store.root, command)
    if timeout_limit is not None and timeout_limit <= 0:
        raise HarnessError("Supervised validation deadline reached; no check was started.")
    timeout = min(command["timeout"], timeout_limit) if timeout_limit is not None else command["timeout"]
    outcome = execute(command["argv"], store.root, timeout)
    after = inspect(store.root, required=True)
    assert after is not None
    success = (
        outcome["exit_code"] == 0 and not outcome["timed_out"] and not outcome["output_limited"]
        and before.fingerprint == after.fingerprint
    )
    result = {
        "id": uuid.uuid4().hex, "task_id": task.id, "check": name,
        "command_digest": command["digest"], "fingerprint": before.fingerprint,
        "success": success, "protected_graders": bool(command["protected"]),
        "recorded_at": now(), **outcome,
    }
    if execution is not None:
        result["execution"] = execution
    if trial_row and json.loads(trial_row[0])["profile"]["id"] != profile(load_config(store.home), execution)["id"]:
        result["success"] = False
        result["profile_changed"] = True
    with store.connection() as database:
        current = store.selected(database, task.id)
        latest = database.execute("SELECT document FROM quality_commands WHERE name = ?", (name,)).fetchone()
        stale = (
            current.revision != task.revision or current.phase != "validate"
            or json.loads(latest[0])["digest"] != command["digest"]
        )
        if stale:
            result["success"] = False
            result["stale"] = True
        database.execute("INSERT INTO quality_runs VALUES (?, ?)", (result["id"], json.dumps(result)))
        if not stale:
            current.checks[name] = Validation(
                "passed" if result["success"] else "failed",
                f"Executed run {result['id']}; exit {outcome['exit_code']}; timeout={outcome['timed_out']}",
                current.change_revision, now(), result["id"],
            )
            current.last_exchange = None
            current.revision += 1
            store.save(database, current, "validation-executed")
        if current.trial_id:
            record_trial_result(database, current.trial_id, result)
    return result


def verify_completion(database: sqlite3.Connection, task: Task) -> None:
    if not enabled(database):
        return
    for name, check in task.checks.items():
        run = database.execute("SELECT document FROM quality_runs WHERE id = ?", (check.run_id,)).fetchone()
        command = database.execute("SELECT document FROM quality_commands WHERE name = ?", (name,)).fetchone()
        if not run or not command:
            raise HarnessError(f"Required check {name} has no executed, approved validation.")
        record = json.loads(run[0])
        if (not record["success"] or record["task_id"] != task.id
                or record["fingerprint"] != task.repository.get("fingerprint")
                or record["command_digest"] != json.loads(command[0])["digest"]):
            raise HarnessError(f"Required check {name} is stale; run it again.")


def add_case(
    store: ProjectStore, case_id: str, goal: str, done: str, checks: list[str],
    files: list[str],
) -> dict[str, Any]:
    case = {
        "id": text(case_id, "Case ID", limit=120), "goal": text(goal, "Case goal"),
        "done": text(done, "Case completion"), "checks": checks,
        "files": store.file_specs(files),
    }
    if not checks or len(checks) != len(set(checks)) or len(checks) > 8:
        raise HarnessError("Cases require 1-8 distinct checks.")
    with store.connection() as database:
        require_enabled(database)
        case["command_digests"] = {}
        for name in checks:
            row = database.execute("SELECT document FROM quality_commands WHERE name = ?", (name,)).fetchone()
            if not row or not json.loads(row[0])["protected"]:
                raise HarnessError("Held-out cases require approved commands with protected grader files.")
            case["command_digests"][name] = json.loads(row[0])["digest"]
        database.execute("INSERT INTO quality_cases VALUES (?, ?)", (case["id"], json.dumps(case)))
    return case


def start_trial(store: ProjectStore, case_id: str, config: dict[str, Any]) -> dict[str, Any]:
    with store.connection() as database:
        require_enabled(database)
        row = database.execute("SELECT document FROM quality_cases WHERE id = ?", (case_id,)).fetchone()
        if not row:
            raise HarnessError("Register the user-designated held-out case first.")
        case = json.loads(row[0])
        for name, expected in case["command_digests"].items():
            command = database.execute(
                "SELECT document FROM quality_commands WHERE name = ?", (name,),
            ).fetchone()
            if not command or json.loads(command[0])["digest"] != expected:
                raise HarnessError("Case graders changed; register a new versioned case before starting a trial.")
    task = store.new(case["goal"], case["done"], case["files"], case["checks"])
    identity = profile(config)
    trial = {
        "id": uuid.uuid4().hex, "case": case_id, "task_id": task.id, "profile": identity,
        "status": "pending", "checks": case["checks"], "results": {},
        "first_fingerprint": "", "started_at": now(),
        "baseline_fingerprint": task.repository.get("fingerprint"),
        "command_digests": case["command_digests"],
    }
    with store.connection() as database:
        current = store.selected(database, task.id)
        current.trial_id = trial["id"]
        current.revision += 1
        store.save(database, current, "evaluation-started")
        database.execute("INSERT INTO quality_trials VALUES (?, ?)", (trial["id"], json.dumps(trial)))
    return trial


def record_trial_result(database: sqlite3.Connection, trial_id: str, result: dict[str, Any]) -> None:
    row = database.execute("SELECT document FROM quality_trials WHERE id = ?", (trial_id,)).fetchone()
    if not row:
        raise HarnessError("Missing evaluation trial.")
    trial = json.loads(row[0])
    if trial["status"] != "pending":
        return
    # First submitted candidate only. A later repair never flips a failed trial.
    if not trial["first_fingerprint"]:
        trial["first_fingerprint"] = result["fingerprint"]
    if (not result["success"] or not result["protected_graders"]
            or trial["first_fingerprint"] != result["fingerprint"]
            or result["fingerprint"] == trial["baseline_fingerprint"]
            or result["command_digest"] != trial["command_digests"].get(result["check"])):
        trial["status"] = "failed"
    if result["check"] not in trial["results"]:
        trial["results"][result["check"]] = result["id"]
    if trial["status"] == "pending" and set(trial["results"]) == set(trial["checks"]):
        trial["status"] = "passed"
    database.execute("UPDATE quality_trials SET document = ? WHERE id = ?", (json.dumps(trial), trial_id))


def report(store: ProjectStore) -> dict[str, Any]:
    with store.connection() as database:
        require_enabled(database)
        trials = [json.loads(row[0]) for row in database.execute(
            "SELECT document FROM quality_trials ORDER BY rowid"
        )]
    grouped: dict[str, dict[str, Any]] = {}
    seen = set()
    for trial in trials:
        key = (trial["profile"]["id"], trial["case"])
        if key in seen:
            continue
        seen.add(key)
        group = grouped.setdefault(key[0], {
            "profile": trial["profile"], "tasks": 0, "passed": 0, "pending": 0,
            "case_ids": [],
            "case_digests": {}, "case_baselines": {},
        })
        # Pending first trials remain in the denominator to prevent cherry-picking.
        group["tasks"] += 1
        group["case_ids"].append(trial["case"])
        group["case_digests"][trial["case"]] = digest(trial.get("command_digests", {}))
        group["case_baselines"][trial["case"]] = trial.get("baseline_fingerprint")
        group["passed"] += trial["status"] == "passed"
        group["pending"] += trial["status"] == "pending"
    for group in grouped.values():
        group["case_ids"].sort()
        n, passed = group["tasks"], group["passed"]
        rate = passed / n
        z = 1.959963984540054
        center = (rate + z * z / (2 * n)) / (1 + z * z / n)
        margin = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        group.update({
            "success_rate": rate, "wilson_95_interval": [center - margin, center + margin],
            "target_observed": n >= 100 and group["pending"] == 0 and rate >= 0.95,
        })
    return {
        "metric": "First submitted candidate passing all registered protected checks, per user-designated held-out case",
        "target": 0.95, "minimum_distinct_tasks": 100, "profiles": list(grouped.values()),
        "limitations": (
            "Not a correctness guarantee, test coverage, or model-generation pass@1. "
            "External-agent internal retries/model choice are not independently observed; supervised profiles "
            "include observed engine/harness provenance and remain separate. Training-data contamination is unknown. "
            "Use independent representative tasks and review grader quality."
        ),
    }
