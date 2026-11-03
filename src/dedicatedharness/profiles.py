from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .activity import activity_lock
from .local import LocalClient, local_url, validate_sampling
from .models import assess
from .system import (
    HarnessError, as_json, detect_hardware, load_config, require_apple,
    save_config, validate_config, write_json,
)


def profile_path(home: Path, name: str) -> Path:
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,47}", name):
        raise HarnessError("Profile names use lowercase letters, digits, and hyphens.")
    return home / "profiles" / f"{name}.json"


def read_profile(home: Path, name: str) -> dict[str, Any]:
    path = profile_path(home, name)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 64 * 1024:
        raise HarnessError(f"Missing or invalid saved profile: {name}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("version") != 1:
        raise HarnessError(f"Unsupported saved profile: {name}")
    configuration = validate_config(document.get("configuration"))
    local_url(configuration["base_url"])
    validate_sampling(configuration.get("sampling", {}))
    return configuration


def save_profile(home: Path, name: str, config: dict[str, Any] | None = None) -> None:
    with activity_lock(home):
        configuration = dict(validate_config(config if config is not None else load_config(home)))
        local_url(configuration["base_url"])
        validate_sampling(configuration.get("sampling", {}))
        configuration["tools_verified"] = False
        configuration.pop("tools_verified_at", None)
        write_json(profile_path(home, name), {"version": 1, "configuration": configuration})


def references(home: Path) -> set[tuple[str | None, str | None]]:
    result = set()
    for path in sorted((home / "profiles").glob("*.json")):
        config = read_profile(home, path.stem)
        result.add((config.get("repo"), config.get("revision")))
    return result


def activate(home: Path, name: str) -> dict[str, Any]:
    from .model_manager import inventory, require_stopped, verify_model
    from .runtime import prepare_runtime

    with activity_lock(home):
        config = dict(read_profile(home, name))
        require_stopped(home, config["base_url"] if config["backend"] == "mlx" else None)
        if config["backend"] == "mlx":
            hardware = detect_hardware(home / "hub")
            require_apple(hardware)
            key = str(config.get("repo")) + "@" + str(config.get("revision"))
            owned = inventory(home).get(key)
            snapshot = Path(config["snapshot"]).resolve(strict=True)
            if (
                not owned or snapshot != Path(owned["snapshot"]).resolve()
                or not snapshot.is_relative_to((home / "hub").resolve())
                or config["model"] != str(snapshot)
            ):
                raise HarnessError("Profile does not identify a retained harness-owned snapshot.")
            files = list(snapshot.rglob("*"))
            if len(files) > 4096 or any(
                not path.resolve().is_relative_to((home / "hub").resolve()) for path in files
            ):
                raise HarnessError("Invalid cached snapshot inventory.")
            metadata = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
            tokenizer = json.loads((snapshot / "tokenizer_config.json").read_text(encoding="utf-8"))
            if not isinstance(metadata, dict) or not isinstance(tokenizer, dict):
                raise HarnessError("Cached model metadata must contain JSON objects.")
            template = snapshot / "chat_template.jinja"
            if not tokenizer.get("chat_template") and template.is_file():
                tokenizer["chat_template"] = template.read_text(encoding="utf-8")
            info = SimpleNamespace(
                id=config["repo"], sha=config["revision"], pipeline_tag="text-generation",
                card_data={"license": config.get("license", "unknown")},
                siblings=[SimpleNamespace(rfilename=str(path.relative_to(snapshot)),
                                          size=path.stat().st_size)
                          for path in files if path.is_file()],
            )
            candidate = assess(info, metadata, tokenizer, hardware, config["context_tokens"], cached=True)
            if candidate.fit != "plausible" or candidate.license == "unknown":
                raise HarnessError(f"Saved profile is not currently hardware eligible: {candidate.fit}.")
            python, version = prepare_runtime(home, False)
            if str(python) != config["runtime_python"] or version != config.get("runtime_version"):
                raise HarnessError("Saved profile's runtime changed; re-stage and evaluate it.")
            verify_model(python, snapshot)
            require_stopped(home, config["base_url"])
        elif config["model"] not in LocalClient(config["base_url"]).models():
            raise HarnessError("Saved external model is not exposed by its local server.")
        config["tools_verified"] = False
        config.pop("tools_verified_at", None)
        save_config(home, config)
        return config


def register(commands: argparse._SubParsersAction) -> None:
    parent = commands.add_parser("profile", help="Save and safely switch named local model configurations")
    actions = parent.add_subparsers(dest="profile_command", required=True)
    actions.add_parser("list", help="List retained model profiles")
    save = actions.add_parser("save", help="Save the current profile without overwriting an existing name")
    save.add_argument("name")
    use = actions.add_parser("use", help="Revalidate and activate a saved profile; never start a second model")
    use.add_argument("name")
    use.add_argument("--yes", action="store_true")


def run(args: argparse.Namespace, home: Path) -> None:
    if args.profile_command == "list":
        print(as_json([
            {"name": path.stem, "configuration": read_profile(home, path.stem)}
            for path in sorted((home / "profiles").glob("*.json"))
        ]))
    elif args.profile_command == "save":
        save_profile(home, args.name)
        print(f"Saved {args.name}; its model revision will be retained during upgrades.")
    else:
        from .cli import confirm
        config = read_profile(home, args.name)
        print(as_json(config))
        confirm("Revalidate and activate this profile without deleting the current model?", args.yes)
        activate(home, args.name)
        print("Profile activated; start its local server and rerun 'dh check-tools'.")
