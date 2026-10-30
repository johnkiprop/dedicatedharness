from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

from .projects import ProjectStore, handoff
from .system import HarnessError, as_json


def root_option(parser: argparse.ArgumentParser, *, task_id: bool = False) -> None:
    parser.add_argument("--root", type=Path, default=Path.cwd())
    if task_id:
        parser.add_argument("--id", help="Task ID; defaults to this project's active task")


def register(commands: argparse._SubParsersAction) -> None:
    project = commands.add_parser("project", help="Maintain local project context without inference")
    project_commands = project.add_subparsers(dest="project_command", required=True)
    project_commands.add_parser("list", help="List projects initialized under this --home")
    init = project_commands.add_parser("init", help="Initialize private state for one checkout")
    root_option(init)
    init.add_argument("--name")
    init.add_argument("--goal", required=True)
    init.add_argument("--constraints", default="Keep changes scoped and verify before claiming success.")
    show = project_commands.add_parser("show", help="Show project context and active task")
    root_option(show)
    update = project_commands.add_parser("update", help="Update context; changed scope revokes open approvals")
    root_option(update)
    update.add_argument("--name")
    update.add_argument("--goal")
    update.add_argument("--constraints")

    task = commands.add_parser("task", help="Maintain bounded engineering checkpoints and agent handoffs")
    task_commands = task.add_subparsers(dest="task_command", required=True)
    new = task_commands.add_parser("new", help="Start a task in investigation mode")
    root_option(new)
    new.add_argument("goal")
    new.add_argument("--done", required=True, help="Checkable completion condition")
    new.add_argument("--file", action="append", default=[], metavar="PATH[::START-END]")
    new.add_argument("--check", action="append", help="Required check name; repeatable; default: tests")
    listing = task_commands.add_parser("list", help="List the latest 100 tasks")
    root_option(listing)
    use = task_commands.add_parser("use", help="Resume a task without replaying its transcript")
    root_option(use)
    use.add_argument("task_id")
    show_task = task_commands.add_parser("show", help="Show current state; validation is user-reported")
    root_option(show_task, task_id=True)

    checkpoint = task_commands.add_parser("checkpoint", help="Replace selected compact state sections")
    root_option(checkpoint, task_id=True)
    checkpoint.add_argument("--goal")
    checkpoint.add_argument("--done", dest="done_when")
    checkpoint.add_argument("--finding", dest="findings")
    checkpoint.add_argument("--decision", dest="decisions")
    checkpoint.add_argument("--change", dest="changes")
    checkpoint.add_argument("--next", dest="next_action")
    files = checkpoint.add_mutually_exclusive_group()
    files.add_argument("--file", action="append", metavar="PATH[::START-END]")
    files.add_argument("--clear-files", action="store_true")

    approve = task_commands.add_parser("approve", help="Explicitly approve a recorded implementation plan")
    root_option(approve, task_id=True)
    approve.add_argument("--scope", required=True)
    ready = task_commands.add_parser("ready", help="Move recorded implementation to validation")
    root_option(ready, task_id=True)
    validate = task_commands.add_parser("validate", help="Record evidence without executing a check")
    root_option(validate, task_id=True)
    validate.add_argument("name")
    validate.add_argument("--status", choices=("pending", "passed", "failed", "blocked"), required=True)
    validate.add_argument("--evidence", default="")
    finish = task_commands.add_parser("finish", help="Close only when all required checks are reported passed")
    root_option(finish, task_id=True)
    forget = task_commands.add_parser("forget", help="Forget the optional latest exchange, keeping checkpoint state")
    root_option(forget, task_id=True)
    history = task_commands.add_parser("history", help="Inspect checkpoint audit, not a chat transcript")
    root_option(history, task_id=True)
    history.add_argument("--limit", type=int, default=10)

    handing = task_commands.add_parser("handoff", help="Generate a bounded prompt; never launch another agent")
    root_option(handing, task_id=True)
    handing.add_argument("target", choices=("opencode", "openclaw"))
    handing.add_argument("--max-bytes", type=int, default=8192)
    handing.add_argument("--out", type=Path, help="Write a new UTF-8 prompt file without overwriting")


def run(args: argparse.Namespace, home: Path) -> None:
    if args.command == "project" and args.project_command == "list":
        projects = []
        for path in sorted((home / "projects").glob("*/state.sqlite3")):
            with closing(sqlite3.connect(path)) as database:
                row = database.execute("SELECT document, active FROM project WHERE id = 1").fetchone()
                if row is None:
                    raise HarnessError(f"Incomplete project database: {path}")
                checkpoint = database.execute(
                    "SELECT document FROM tasks WHERE id = ?", (row[1],),
                ).fetchone()
                state = json.loads(checkpoint[0]) if checkpoint else None
                projects.append({
                    "project": json.loads(row[0]), "active_task": row[1],
                    "active_checkpoint": {
                        key: state[key] for key in ("goal", "phase", "updated_at")
                    } if state else None,
                })
        print(as_json({"projects": projects, "phase_status": "Last recorded; select/show a task to refresh Git state."}))
        return
    store = ProjectStore(home, args.root)
    if args.command == "project":
        if args.project_command == "init":
            result = asdict(store.initialize(
                args.name or store.root.name, args.goal, args.constraints,
            ))
        elif args.project_command == "update":
            result = asdict(store.update_project(args.name, args.goal, args.constraints))
        else:
            result = store.show_project()
        print(as_json(result))
        return
    action = args.task_command
    task_id = getattr(args, "id", None)
    if action == "new":
        selected = store.new(args.goal, args.done, args.file, args.check or ["tests"])
    elif action == "list":
        print(as_json({"tasks": store.list_tasks(), "limit": 100}))
        return
    elif action == "use":
        selected = store.use(args.task_id)
    elif action == "show":
        _, selected = store.snapshot(task_id)
    elif action == "checkpoint":
        sections = {
            key: value for key in (
                "goal", "done_when", "findings", "decisions", "changes", "next_action",
            ) if (value := getattr(args, key)) is not None
        }
        if args.file is not None or args.clear_files:
            sections["files"] = [] if args.clear_files else args.file
        selected = store.checkpoint(task_id, **sections)
    elif action == "approve":
        selected = store.approve(task_id, args.scope)
    elif action == "ready":
        selected = store.ready(task_id)
    elif action == "validate":
        selected = store.validate(task_id, args.name, args.status, args.evidence)
    elif action == "finish":
        selected = store.finish(task_id)
    elif action == "forget":
        selected = store.forget(task_id)
    elif action == "history":
        print(as_json({"events": store.history(task_id, args.limit), "limit": args.limit}))
        return
    elif action == "handoff":
        project, selected = store.snapshot(task_id)
        prompt = handoff(project, selected, args.target, args.max_bytes)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x", encoding="utf-8") as destination:
                destination.write(prompt)
            print(f"Wrote {args.out}; review before passing it to your configured local agent.")
        else:
            print(prompt, end="")
        return
    else:
        raise HarnessError("Unknown task operation.")
    from .quality import enabled
    with store.connection() as database:
        verified = enabled(database)
    print(as_json({
        "task": asdict(selected),
        "evidence_source": (
            "Passing checks require executed quality runs; findings, approvals, and other evidence are user-reported."
            if verified else "user-reported; no checks, code edits, or deployments executed by dh"
        ),
    }))
