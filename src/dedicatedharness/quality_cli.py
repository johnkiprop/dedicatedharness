from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import quality
from .projects import ProjectStore
from .repository import suggest
from .system import HarnessError, as_json, load_config, save_config


def register(commands: argparse._SubParsersAction) -> None:
    parent = commands.add_parser("quality", help="Git-bound validation and measured coding outcomes")
    actions = parent.add_subparsers(dest="quality_command", required=True)
    init = actions.add_parser("init", help="Require executed checks for this project's acceptance")
    init.add_argument("--root", type=Path, default=Path.cwd())
    command = actions.add_parser("command", help="Explicitly approve a fixed validation command")
    command.add_argument("name")
    command.add_argument("--root", type=Path, default=Path.cwd())
    command.add_argument("--argv-json", required=True, help='Example: ["python","-m","unittest","discover"]')
    command.add_argument("--timeout", type=int, default=180)
    command.add_argument("--protect", action="append", default=[], help="Existing trusted grader file/directory")
    run = actions.add_parser("run", help="Execute only a previously approved command")
    run.add_argument("name")
    run.add_argument("--root", type=Path, default=Path.cwd())
    run.add_argument("--id", help="Task ID; defaults to active")
    context = actions.add_parser("context", help="Suggest source/test paths without another model call")
    context.add_argument("query")
    context.add_argument("--root", type=Path, default=Path.cwd())
    context.add_argument("--limit", type=int, default=8)
    case = actions.add_parser("case", help="Register one immutable user-designated held-out coding case")
    case.add_argument("case_id")
    case.add_argument("--root", type=Path, default=Path.cwd())
    case.add_argument("--goal", required=True)
    case.add_argument("--done", required=True)
    case.add_argument("--check", action="append", required=True)
    case.add_argument("--file", action="append", default=[])
    trial = actions.add_parser("trial", help="Start a fresh case task before submitting an implementation")
    trial.add_argument("case_id")
    trial.add_argument("--root", type=Path, default=Path.cwd())
    report = actions.add_parser("report", help="Report observed first-submission outcomes, not a guarantee")
    report.add_argument("--root", type=Path, default=Path.cwd())
    sampling = actions.add_parser("profile", help="Set explicit inference settings, not training weights")
    sampling.add_argument("--temperature", type=float, default=0)
    sampling.add_argument("--top-p", type=float, default=1)
    sampling.add_argument("--seed", type=int, default=0)


def run(args: argparse.Namespace, home: Path) -> None:
    action = args.quality_command
    if action == "profile":
        from .local import validate_sampling
        configuration = load_config(home)
        settings = {"temperature": args.temperature, "top_p": args.top_p, "seed": args.seed}
        validate_sampling(settings)
        configuration["sampling"] = settings
        configuration["tools_verified"] = False
        configuration.pop("tools_verified_at", None)
        save_config(home, configuration)
        print(as_json(quality.profile(configuration)))
        return
    if action == "context":
        print(as_json({
            "suggestions": suggest(args.root.resolve(), args.query, args.limit),
            "method": "Path keywords and Git changes; not semantic ranking or a quality score",
            "next": "Review and explicitly select relevant source AND tests with task checkpoint --file.",
        }))
        return
    store = ProjectStore(home, args.root)
    if action == "init":
        quality.initialize(store)
        result = {"verified_mode": True, "target": 0.95, "minimum_distinct_tasks": 100}
    elif action == "command":
        result = quality.register_command(
            store, args.name, json.loads(args.argv_json), args.timeout, args.protect,
        )
    elif action == "run":
        result = quality.run_check(store, args.name, args.id)
        print(as_json(result))
        if not result["success"]:
            raise HarnessError("Validation failed or became stale; task is not accepted.")
        return
    elif action == "case":
        result = quality.add_case(store, args.case_id, args.goal, args.done, args.check, args.file)
    elif action == "trial":
        result = quality.start_trial(store, args.case_id, load_config(home))
    else:
        result = quality.report(store)
    print(as_json(result))
