from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .local import local_url
from .models import Candidate
from .quality import execute, profile, report
from .projects import ProjectStore
from .system import HarnessError, load_config, write_json


def require_stopped(home: Path, target: str | None = None) -> None:
    urls = {target} if target else set()
    if (home / "config.json").exists():
        current = load_config(home)
        if current["backend"] == "mlx":
            urls.add(current["base_url"])
    for url in urls:
        address = urlsplit(local_url(url))
        try:
            connection = socket.create_connection((address.hostname, address.port), timeout=1)
        except ConnectionRefusedError:
            continue
        except OSError as error:
            raise HarnessError(f"Cannot confirm the model endpoint is stopped: {error}") from error
        connection.close()
        raise HarnessError(
            f"Stop the listener at {url} before loading/replacing a model. "
            "The harness will not kill an existing server or load a second model beside it."
        )


def evidence(home: Path, root: Path) -> dict[str, dict[str, Any]]:
    root = root.expanduser().resolve(strict=True)
    if home.resolve().is_relative_to(root):
        return {}  # This scope cannot contain an initialized private project store.
    store = ProjectStore(home, root)
    if not store.path.exists():
        return {}
    with store.connection() as database:
        from .quality import enabled
        if not enabled(database):
            return {}
    result: dict[str, dict[str, Any]] = {}
    for group in report(store)["profiles"]:
        configuration = group["profile"]["configuration"]
        if configuration.get("repo") and configuration.get("revision"):
            key = configuration["repo"] + "@" + configuration["revision"]
            previous = result.get(key)
            if previous is None or group["tasks"] > previous["tasks"]:
                result[key] = group
    return result


def inventory(home: Path) -> dict[str, Any]:
    path = home / "owned-models.json"
    if not path.exists():
        return {}
    owned = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(owned, dict) or any(
        not isinstance(entry, dict)
        or any(not isinstance(entry.get(field), str) or not entry[field]
               for field in ("repo", "revision", "snapshot"))
        or key != entry["repo"] + "@" + entry["revision"]
        for key, entry in owned.items()
    ):
        raise HarnessError("Invalid owned-model inventory; refusing model cleanup.")
    return owned


def record_owned(home: Path, candidate: Candidate, snapshot: Path) -> None:
    if not snapshot.resolve().is_relative_to((home / "hub").resolve()):
        raise HarnessError("Refusing to claim ownership outside the dedicated model cache.")
    owned = inventory(home)
    owned[candidate.repo + "@" + candidate.revision] = {
        "repo": candidate.repo, "revision": candidate.revision, "snapshot": str(snapshot.resolve()),
    }
    write_json(home / "owned-models.json", owned, overwrite=True)


def verify_model(python: Path, snapshot: Path) -> None:
    script = (
        "from mlx_lm import load,generate; from mlx_lm.sample_utils import make_sampler; "
        "import sys; m,t=load(sys.argv[1],tokenizer_config={'trust_remote_code':False}); "
        "p=t.apply_chat_template([{'role':'user','content':'Reply briefly: ready.'}],"
        "tokenize=False,add_generation_prompt=True); "
        "s=generate(m,t,prompt=p,max_tokens=64,sampler=make_sampler(0)); "
        "assert s.strip(), 'No visible generation'"
    )
    # Pass environment directly, without changing global interpreter state.
    result = execute(
        [str(python), "-c",
         "import os; os.environ['HF_HUB_OFFLINE']='1'; "
         "os.environ['TRANSFORMERS_OFFLINE']='1'; " + script, str(snapshot)],
        snapshot, 180,
    )
    if result["exit_code"] or result["timed_out"] or result["output_limited"]:
        raise HarnessError(
            "Candidate failed offline load/generation verification; active configuration is unchanged. "
            "Downloaded candidate remains staged for inspection."
        )


def cleanup_previous(home: Path, old: dict[str, Any], new: dict[str, Any]) -> str:
    from huggingface_hub import scan_cache_dir

    old_key = str(old.get("repo")) + "@" + str(old.get("revision"))
    new_key = str(new.get("repo")) + "@" + str(new.get("revision"))
    owned = inventory(home)
    if old_key == new_key or old_key not in owned:
        return "Previous model retained (same revision or not recorded as harness-owned)."
    entry = owned[old_key]
    if not Path(entry["snapshot"]).resolve().is_relative_to((home / "hub").resolve()):
        raise HarnessError("Cleanup refused: ownership path is outside the dedicated cache.")
    cache = scan_cache_dir(home / "hub")
    matching = [
        repo for repo in cache.repos for revision in repo.revisions
        if revision.commit_hash == entry["revision"]
    ]
    if any(repo.repo_type != "model" or repo.repo_id != entry["repo"] for repo in matching):
        return "Previous revision retained because its commit is shared with another cached repository."
    if matching:
        cache.delete_revisions(entry["revision"]).execute()
    owned.pop(old_key)
    write_json(home / "owned-models.json", owned, overwrite=True)
    return "Removed only the previous harness-owned revision using Hugging Face cache cleanup."


def upgrade_allowed(
    current: dict[str, Any], candidate: Candidate, measured: dict[str, dict[str, Any]],
) -> bool:
    previous = measured.get(str(current.get("repo")) + "@" + str(current.get("revision")))
    replacement = measured.get(candidate.repo + "@" + candidate.revision)
    if not previous or not replacement:
        return False
    before = previous["profile"]["configuration"]
    after = replacement["profile"]["configuration"]
    active = profile(current)["configuration"]
    comparable = ("backend", "runtime_version", "context_tokens", "max_output", "sampling", "contract")
    return bool(
        replacement["target_observed"]
        and previous.get("case_ids") == replacement.get("case_ids")
        and previous.get("case_digests") == replacement.get("case_digests")
        and previous.get("case_baselines") == replacement.get("case_baselines")
        and all(before.get(key) == after.get(key) == active.get(key) for key in comparable)
        and replacement["success_rate"] > previous["success_rate"]
    )
