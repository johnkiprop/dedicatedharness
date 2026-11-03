from __future__ import annotations

import argparse
import json
import secrets
import sqlite3
import statistics
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from httpx import HTTPError

from .local import LocalClient, Result, local_url
from .models import Hub
from . import model_manager
from . import profiles, supervisor
from .project_cli import register as register_project_commands, run as run_project_command
from .projects import ProjectStore, context_for, text
from .quality_cli import register as register_quality_commands, run as run_quality_command
from .prompts import admit, brief, messages_for, read_context
from .runtime import prepare_runtime, serve
from .system import (
    GIB, HarnessError, as_json, default_home, detect_hardware, load_config,
    require_apple, save_config, write_json,
)


def positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def budget_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--context", type=positive, default=4096, help="Total context budget")
    parser.add_argument("--output", type=positive, default=512, help="Maximum output tokens")


def query_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("task", nargs="?")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--file", action="append", default=[], metavar="PATH[::START-END]",
        help="Explicit UTF-8 context inside --root; repeatable",
    )
    parser.add_argument("--output", type=positive, help="Output limit within configured budget")
    parser.add_argument("--timeout", type=positive, default=120)
    parser.add_argument("--project", action="store_true", help="Include the active task's compact project state")
    parser.add_argument("--task-id", help="With --project, use a specific task instead of the active one")


def parser_for_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dh", description="Hardware-aware local models and bounded coding supervision for Mac engineers."
    )
    parser.add_argument("--home", type=Path, default=default_home())
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Inspect hardware and configured runtime")

    models = commands.add_parser("models", help="Discover and budget Hugging Face candidates")
    models.add_argument("--search", default="Qwen")
    selection = models.add_mutually_exclusive_group()
    selection.add_argument("--repo", help="Inspect one explicit repository instead of searching")
    models.add_argument("--limit", type=positive, default=8)
    models.add_argument("--context", type=positive, default=4096)
    models.add_argument("--memory-gb", type=positive, help="Preview only: hypothetical RAM in GiB")
    models.add_argument("--json", action="store_true")
    selection.add_argument("--recommend", choices=("coding", "reasoning"))
    models.add_argument("--root", type=Path, default=Path.cwd(), help="Project whose evaluation evidence to consult")

    setup = commands.add_parser("setup", help="Review and install one pinned MLX model")
    setup.add_argument("--repo", help="Explicit override; otherwise discover a purpose-specific candidate")
    setup.add_argument("--revision", help="Pin an explicit --repo to an immutable Hub revision")
    setup.add_argument("--purpose", choices=("coding", "reasoning"), default="coding")
    setup.add_argument("--root", type=Path, default=Path.cwd())
    setup.add_argument("--install-runtime", action="store_true", help="Install/update isolated MLX")
    setup.add_argument("--port", type=positive, default=8080)
    setup.add_argument("--yes", action="store_true", help="Approve displayed model/runtime changes")
    budget_options(setup)
    upgrade = commands.add_parser("upgrade", help="Stage new models; promote only measured improvements")
    upgrade.add_argument("--purpose", choices=("coding", "reasoning"), default="coding")
    upgrade.add_argument("--repo", help="Inspect a specific candidate")
    upgrade.add_argument("--root", type=Path, default=Path.cwd())
    upgrade.add_argument("--stage", action="store_true", help="Download/verify without changing the active model")
    upgrade.add_argument("--yes", action="store_true")
    upgrade.add_argument("--keep-old", action="store_true")
    upgrade.add_argument("--profile", help="With --stage, retain the candidate as a named evaluation profile")

    connect = commands.add_parser("connect", help="Connect to an existing local OpenAI-compatible API")
    connect.add_argument("--url", default="http://127.0.0.1:1234/v1")
    connect.add_argument("--model", help="Must match /v1/models; required when several are listed")
    budget_options(connect)

    serving = commands.add_parser("serve", help="Run the managed MLX server in the foreground")
    serving.add_argument("--startup-timeout", type=positive, default=180)

    guide = commands.add_parser("guide", help="Build a plain-English task brief without inference")
    guide.add_argument("--task")
    guide.add_argument("--context", default="")
    guide.add_argument("--constraints", default="Keep changes scoped. Do not add dependencies.")
    guide.add_argument("--done", default="State a checkable completion condition.")
    guide.add_argument("--output", default="Answer directly and state any uncertainty.")

    ask = commands.add_parser("ask", help="Stream one scoped query; do not retain a transcript")
    query_options(ask)
    ask.add_argument("--dry-run", action="store_true", help="Show admission counts without inference")
    ask.add_argument(
        "--remember", action="store_true",
        help="With --project, retain only the latest successful question/answer locally",
    )

    bench = commands.add_parser("bench", help="Measure the same scoped query across repeated calls")
    query_options(bench)
    bench.add_argument("--runs", type=positive, default=3)
    bench.add_argument("--save", type=Path, help="Opt-in metadata-only report; never overwrite")

    check = commands.add_parser("check-tools", help="Verify structured tool request and result roundtrip")
    check.add_argument("--timeout", type=positive, default=120)

    export = commands.add_parser("export", help="Emit agent configuration without changing defaults")
    export.add_argument("target", choices=("opencode", "openclaw"))
    export.add_argument("--out", type=Path, help="Write a new file, refusing to overwrite")
    register_project_commands(commands)
    register_quality_commands(commands)
    profiles.register(commands)
    supervisor.register(commands)
    return parser


def print_json(value: Any) -> None:
    print(as_json(value))


def confirm(message: str, approved: bool) -> None:
    if approved:
        return
    if not sys.stdin.isatty():
        raise HarnessError(f"{message} Rerun interactively or pass --yes after reviewing.")
    if input(f"{message} [y/N] ").strip().lower() not in ("y", "yes"):
        raise HarnessError("Cancelled; configuration unchanged.")


def validate_budgets(context: int, output: int) -> None:
    if output >= context:
        raise HarnessError("--output must be smaller than --context.")


def setup(args: argparse.Namespace, home: Path) -> None:
    validate_budgets(args.context, args.output)
    if getattr(args, "revision", None) and not args.repo:
        raise HarnessError("--revision requires an explicit --repo.")
    url = local_url(f"http://127.0.0.1:{args.port}/v1")
    model_manager.require_stopped(home, url)
    previous = load_config(home) if (home / "config.json").exists() else {}
    hardware = detect_hardware(home / "hub")
    require_apple(hardware)
    hub = Hub(home / "hub")
    candidate = getattr(args, "selected_candidate", None)
    if candidate is not None:
        candidate = hub.inspect(
            candidate.repo, hardware, args.context, revision=candidate.revision,
        )
    if candidate is None:
        candidate = (
            hub.inspect(args.repo, hardware, args.context, revision=getattr(args, "revision", None))
            if args.repo else
            hub.recommend(args.purpose, hardware, args.context,
                          model_manager.evidence(home, args.root))[0]
        )
    print("Recommendation is provisional unless supported by project evaluation evidence; not a global-best claim.")
    print_json(candidate.to_dict())
    if candidate.fit != "plausible":
        raise HarnessError(f"Installation refused: {candidate.fit}. See reasons above.")
    if candidate.license == "unknown":
        raise HarnessError("Installation refused until the repository declares a reviewable license.")
    print(
        f"Runtime: isolated mlx-lm; install/update={args.install_runtime}. "
        f"Cache: {home / 'hub'}. Full artifact size is charged conservatively, "
        "even for cached files. No prompt is sent to Hugging Face."
    )
    confirm("Download this revision, prepare runtime, and replace only dh's configuration?", args.yes)
    python, version = prepare_runtime(home, args.install_runtime)
    if getattr(args, "expected_runtime_version", version) != version:
        raise HarnessError("Runtime differs from the evaluated replacement profile; evaluate again before upgrading.")
    snapshot = hub.download(candidate).resolve()
    model_manager.record_owned(home, candidate, snapshot)
    model_manager.verify_model(python, snapshot)
    model_manager.require_stopped(home, url)
    configuration = {
        "version": 1, "backend": "mlx", "base_url": url,
        "model": str(snapshot), "repo": candidate.repo, "revision": candidate.revision,
        "license": candidate.license, "snapshot": str(snapshot),
        "runtime_python": str(python), "runtime_version": version,
        "context_tokens": args.context, "max_output": args.output,
        "tools_verified": False,
    }
    settings = getattr(args, "selected_sampling", previous.get("sampling"))
    if settings is not None:
        configuration["sampling"] = settings
    save_config(home, configuration)
    print("Pinned model and runtime configured. Run 'dh serve', then 'dh ask' in another terminal.")
    print("Offline load/generation passed; coding quality, speed, and tool compatibility need separate evaluation.")


def upgrade(args: argparse.Namespace, home: Path) -> None:
    profile_name = getattr(args, "profile", None)
    if profile_name:
        if not args.stage:
            raise HarnessError("--profile requires --stage; named candidates are not automatic promotions.")
        if profiles.profile_path(home, profile_name).exists():
            raise HarnessError("That profile already exists; choose a new name before staging.")
    config = load_config(home)
    if config["backend"] != "mlx":
        raise HarnessError("Automatic upgrades manage only harness-owned MLX models, not LM Studio/external servers.")
    model_manager.require_stopped(home)
    hardware = detect_hardware(home / "hub")
    require_apple(hardware)
    hub = Hub(home / "hub")
    measured = model_manager.evidence(home, args.root)
    candidate = (
        hub.inspect(args.repo, hardware, config["context_tokens"]) if args.repo else
        hub.recommend(args.purpose, hardware, config["context_tokens"],
                      {} if args.stage else measured)[0]
    )
    print_json(candidate.to_dict())
    if candidate.fit != "plausible" or candidate.license == "unknown":
        raise HarnessError("Candidate is not safely eligible for installation.")
    if candidate.repo == config.get("repo") and candidate.revision == config.get("revision"):
        if profile_name:
            raise HarnessError(
                f"Candidate is already active; no staged profile was created. Use 'dh profile save {profile_name}'."
            )
        print("Current model is the preferred eligible candidate; no files or configuration changed.")
        return
    if args.stage:
        confirm("Download and verify this staged model without changing the active model?", args.yes)
        python, runtime_version = prepare_runtime(home, False)
        snapshot = hub.download(candidate).resolve()
        model_manager.record_owned(home, candidate, snapshot)
        model_manager.verify_model(python, snapshot)
        if profile_name:
            profiles.save_profile(home, profile_name, {
                **config, "model": str(snapshot), "snapshot": str(snapshot),
                "repo": candidate.repo, "revision": candidate.revision, "license": candidate.license,
                "runtime_python": str(python), "runtime_version": runtime_version,
            })
            print(f"Candidate retained as '{profile_name}'; use 'dh profile use {profile_name}' for explicit evaluation.")
        print(
            "Candidate staged. Active model retained; no old model removed. "
            f"For explicit evaluation, use setup --repo {candidate.repo} "
            f"--revision {candidate.revision} --context {config['context_tokens']} "
            f"--output {config['max_output']}, then serve and verify tools."
        )
        return
    if not model_manager.upgrade_allowed(config, candidate, measured):
        print(
            "No demonstrated improvement on the same held-out cases. Active model retained. "
            "Use --stage to prepare a candidate, then evaluate it before promotion."
        )
        return
    tested = measured[candidate.repo + "@" + candidate.revision]["profile"]["configuration"]
    print(
        "Observed improvement on matching cases, graders, baselines, and inference settings; "
        "not proof of general model superiority. "
        + ("Previous model retained." if args.keep_old else
           "After successful verification, only the previous harness-owned revision will be cleaned up.")
    )
    chosen = argparse.Namespace(
        context=tested["context_tokens"], output=tested["max_output"],
        port=int(config["base_url"].split(":")[-1].split("/")[0]),
        repo=candidate.repo, selected_candidate=candidate, root=args.root, purpose=args.purpose,
        install_runtime=False, yes=args.yes, selected_sampling=tested.get("sampling") or {},
        expected_runtime_version=tested["runtime_version"],
    )
    setup(chosen, home)
    if not args.keep_old:
        print(model_manager.cleanup_previous(home, config, load_config(home)))


def connect(args: argparse.Namespace, home: Path) -> None:
    validate_budgets(args.context, args.output)
    model_manager.require_stopped(home)
    client = LocalClient(args.url)
    models = client.models()
    model = args.model
    if model is None:
        if len(models) != 1:
            print_json({"available_models": models})
            raise HarnessError("Several models are exposed; select one using --model.")
        model = models[0]
    if model not in models:
        raise HarnessError("Selected --model is not exposed by the local /v1/models API.")
    save_config(home, {
        "version": 1, "backend": "external", "base_url": client.base_url,
        "model": model, "context_tokens": args.context, "max_output": args.output,
        "tools_verified": False,
    })
    print(
        f"Connected to {model}. Set the server's context limit to at least "
        f"{args.context}; external token admission uses an estimate, not its tokenizer. "
        "Inference/tool use are not yet verified."
    )


def prepare_query(
    args: argparse.Namespace, config: dict[str, Any], home: Path,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    task = args.task
    if task is None:
        if sys.stdin.isatty():
            raise HarnessError("Provide a task argument or pipe a guided brief on stdin.")
        task = sys.stdin.read(256 * 1024 + 1)
        if len(task) > 256 * 1024:
            raise HarnessError("Piped task exceeds 256 KiB; provide a smaller brief.")
    if (args.task_id or getattr(args, "remember", False)) and not args.project:
        raise HarnessError("--task-id and --remember require --project.")
    sections: list[str] = []
    files = args.file
    if args.project:
        store = ProjectStore(home, args.root)
        project, checkpoint = store.snapshot(args.task_id)
        sections.append(context_for(project, checkpoint))
        files = list(dict.fromkeys([*checkpoint.files, *args.file]))
        # Pin the selected checkpoint through inference, even if another task is activated.
        args.project_task_id = checkpoint.id
        args.project_task_revision = checkpoint.revision
    if getattr(args, "remember", False):
        text(task, "Remembered question", limit=4096)
    args.question_text = task
    selected_context = read_context(args.root, files)
    if selected_context:
        sections.append(selected_context)
    context = "\n\n".join(sections)
    messages = messages_for(task, context)
    output = args.output or config["max_output"]
    admission = admit(config, messages, output)
    if args.project:
        admission.update({
            "project_task_id": args.project_task_id,
            "project_checkpoint_revision": args.project_task_revision,
        })
    return messages, output, admission


def metrics_for(result: Result, admission: dict[str, Any]) -> dict[str, Any]:
    return {
        **admission, **result.metrics(),
        "pure_decode_tokens_per_second": None,
        "note": "Elapsed includes prefill/generation/network. Decode speed is not inferred.",
    }


def check_tools(config: dict[str, Any], home: Path, timeout: int) -> None:
    config["tools_verified"] = False
    config.pop("tools_verified_at", None)
    save_config(home, config)
    nonce = "DH_PROBE_" + secrets.token_hex(8)
    tools = [{
        "type": "function",
        "function": {
            "name": "read_probe",
            "description": "Return the contents of the harmless in-memory probe.",
            "parameters": {
                "type": "object", "properties": {}, "additionalProperties": False,
            },
        },
    }]
    messages = messages_for(
        "Call read_probe with no arguments, then return its result exactly. "
        "You cannot know the result until the tool responds."
    )
    admit(config, messages, config["max_output"], tools)
    client = LocalClient(config["base_url"], timeout=timeout)
    first = client.chat(
        config["model"], messages, config["max_output"], tools=tools, sampling=config.get("sampling"),
    )
    if first.finish_reason == "length" or len(first.tool_calls) != 1:
        raise HarnessError("Tool check failed: expected exactly one complete structured call.")
    call = first.tool_calls[0]
    try:
        arguments = json.loads(call["function"]["arguments"])
    except json.JSONDecodeError as error:
        raise HarnessError("Tool check failed: malformed arguments.") from error
    if call["function"]["name"] != "read_probe" or arguments != {} or not call["id"]:
        raise HarnessError("Tool check failed: unexpected function, arguments, or missing call id.")
    messages.extend([
        {"role": "assistant", "content": first.text, "tool_calls": first.tool_calls},
        {"role": "tool", "content": nonce, "tool_call_id": call["id"]},
    ])
    admit(config, messages, config["max_output"], tools)
    second = client.chat(
        config["model"], messages, config["max_output"], tools=tools, sampling=config.get("sampling"),
    )
    if second.text.strip() != nonce or second.tool_calls or second.finish_reason == "length":
        raise HarnessError("Tool check failed: model did not return the exact tool result.")
    config["tools_verified"] = True
    config["tools_verified_at"] = datetime.now(timezone.utc).isoformat()
    save_config(home, config)
    print("Structured tool roundtrip passed. No filesystem or shell tool was executed.")


def export_config(config: dict[str, Any], target: str) -> dict[str, Any]:
    url = local_url(config["base_url"])
    if not config.get("tools_verified"):
        raise HarnessError(
            "Run 'dh check-tools' before exporting a coding-agent provider. "
            "A chat response alone does not establish tool support."
        )
    model = config["model"]
    if target == "opencode":
        return {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "dedicatedharness": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Dedicated Harness (local only)",
                    "options": {"baseURL": url},
                    "models": {model: {
                        "name": config.get("repo", model), "tool_call": True,
                        "limit": {
                            "context": config["context_tokens"],
                            "output": config["max_output"],
                        },
                    }},
                },
            },
        }
    return {
        "models": {
            "mode": "merge",
            "providers": {
                "dedicatedharness": {
                    "baseUrl": url, "apiKey": "local-placeholder-not-a-secret",
                    "api": "openai-completions",
                    "models": [{
                        "id": model, "name": config.get("repo", model),
                        "input": ["text"], "reasoning": False,
                        "contextWindow": config["context_tokens"],
                        "maxTokens": config["max_output"],
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    }],
                },
            },
        },
    }


def run(args: argparse.Namespace) -> None:
    from contextlib import nullcontext
    from .activity import activity_lock
    mutating = (
        args.command in ("setup", "upgrade", "connect", "check-tools")
        or args.command == "profile" and args.profile_command != "list"
        or args.command == "quality" and args.quality_command == "profile"
    )
    with activity_lock(args.home.expanduser().resolve()) if mutating else nullcontext():
        _run(args)


def _run(args: argparse.Namespace) -> None:
    home = args.home.expanduser().resolve()
    if args.command == "doctor":
        hardware = detect_hardware(home / "hub")
        report: dict[str, Any] = {
            "hardware": json.loads(as_json(hardware)),
            "managed_mlx_eligible": hardware.apple_silicon,
            "config_file": str(home / "config.json"),
            "configured": (home / "config.json").exists(),
        }
        if report["configured"]:
            configuration = load_config(home)
            local_url(configuration["base_url"])
            report["connection"] = {
                key: configuration[key]
                for key in ("backend", "base_url", "model", "context_tokens", "max_output")
            }
        print_json(report)
    elif args.command == "models":
        if args.limit > 30:
            raise HarnessError("--limit is capped at 30 to bound discovery requests.")
        hardware = detect_hardware(home / "hub")
        if args.memory_gb:
            hardware = replace(
                hardware, memory_bytes=args.memory_gb * GIB, available_bytes=None,
                notes=(*hardware.notes, "Hypothetical memory override; preview only."),
            )
        hub = Hub(home / "hub")
        recommendation = None
        if args.recommend:
            recommendation, candidates = hub.recommend(
                args.recommend, hardware, args.context, model_manager.evidence(home, args.root),
            )
        else:
            candidates = (
                [hub.inspect(args.repo, hardware, args.context)] if args.repo
                else hub.discover(args.search, args.limit, hardware, args.context)
            )
        candidates.sort(key=lambda candidate: (
            candidate.fit != "plausible", candidate.weight_bytes,
        ))
        if not candidates:
            raise HarnessError("No matching candidates; change --search or inspect --repo.")
        if args.json:
            print_json({
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "preview_only": bool(args.memory_gb) or not hardware.apple_silicon,
                "context_tokens": args.context,
                "ranking": "Plausible first, then smaller weights; not a quality ranking",
                "candidates": [candidate.to_dict() for candidate in candidates],
                "recommendation": recommendation.to_dict() if recommendation else None,
                "recommendation_basis": "Project outcomes when available; otherwise provisional purpose/recency heuristic",
            })
        else:
            if recommendation:
                print(f"Preferred candidate: {recommendation.repo} (provisional without measured evidence)")
            print("Fit screening, NOT a quality/speed ranking. Preview" if args.memory_gb
                  else "Fit screening, NOT a quality/speed ranking.")
            for candidate in candidates:
                print(
                    f"{candidate.fit:16} {candidate.weight_bytes / GIB:5.2f} GiB weights  "
                    f"{candidate.repo}"
                )
                for reason in candidate.reasons:
                    print(f"  {reason}")
    elif args.command == "setup":
        setup(args, home)
    elif args.command == "connect":
        connect(args, home)
    elif args.command == "upgrade":
        upgrade(args, home)
    elif args.command == "guide":
        task = args.task
        if task is None:
            if not sys.stdin.isatty():
                raise HarnessError("Use --task in non-interactive mode.")
            task = input("What should be done? ").strip()
            args.context = input("Relevant files/facts (optional): ").strip()
            args.constraints = input("Constraints (optional): ").strip()
            args.done = input("How will you verify completion? ").strip()
            args.output = input("Desired output (optional): ").strip()
        if not task.strip():
            raise HarnessError("Task must not be empty.")
        print(brief(task, args.context, args.constraints, args.done, args.output))
    elif args.command in ("project", "task"):
        run_project_command(args, home)
    elif args.command == "quality":
        run_quality_command(args, home)
    elif args.command == "profile":
        profiles.run(args, home)
    elif args.command in ("run", "runs"):
        supervisor.run(args, home)
    else:
        config = load_config(home)
        if args.command == "serve":
            serve(config, home, args.startup_timeout)
        elif args.command == "check-tools":
            check_tools(config, home, args.timeout)
        elif args.command == "export":
            result = export_config(config, args.target)
            if args.out:
                write_json(args.out, result)
                print(f"Wrote {args.out}; review and merge manually. No defaults were changed.")
            else:
                print_json(result)
        elif args.command in ("ask", "bench"):
            messages, output, admission = prepare_query(args, config, home)
            if args.command == "ask" and args.dry_run:
                print_json(admission)
                return
            client = LocalClient(config["base_url"], timeout=args.timeout)
            if args.command == "ask":
                print(as_json(admission), file=sys.stderr)
                result = client.chat(
                    config["model"], messages, output,
                    on_text=lambda text: print(text, end="", flush=True),
                    sampling=config.get("sampling"),
                )
                print()
                if result.tool_calls:
                    raise HarnessError("Unexpected tool calls; this command has no tool executor.")
                print(as_json(metrics_for(result, admission)), file=sys.stderr)
                if result.finish_reason == "length":
                    raise HarnessError("Response hit its output limit and may be incomplete.")
                if args.remember:
                    ProjectStore(home, args.root).remember(
                        args.project_task_id, args.project_task_revision,
                        args.question_text, result.text,
                    )
                    print("Latest exchange remembered locally; no task facts were auto-approved.",
                          file=sys.stderr)
            else:
                if args.runs > 10:
                    raise HarnessError("--runs is capped at 10.")
                if args.save and args.save.exists():
                    raise HarnessError("Report exists; choose a new --save path.")
                results = [
                    client.chat(config["model"], messages, output, sampling=config.get("sampling"))
                    for _ in range(args.runs)
                ]
                if any(result.tool_calls for result in results):
                    raise HarnessError("Benchmark returned unexpected tool calls.")
                report = {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "model": config["model"], "base_url": config["base_url"],
                    "first_run_state": "unknown: may include model load; not a cold-load measurement",
                    "same_prompt_reused": True,
                    "task_quality": "not scored; inspect real task outcomes separately",
                    "median_elapsed_seconds": statistics.median(
                        result.elapsed_seconds for result in results
                    ),
                    "runs": [metrics_for(result, admission) for result in results],
                }
                if args.save:
                    write_json(args.save, report)
                print_json(report)


def main(argv: list[str] | None = None) -> int:
    try:
        run(parser_for_cli().parse_args(argv))
        return 0
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130
    except (
        HarnessError, OSError, ValueError, subprocess.SubprocessError,
        ImportError, HTTPError, sqlite3.Error,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
