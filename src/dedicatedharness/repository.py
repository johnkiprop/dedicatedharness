from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .system import HarnessError


@dataclass(frozen=True)
class Repository:
    fingerprint: str
    head: str
    branch: str
    changes: list[dict[str, str]]
    files: list[str]

    def summary(self) -> dict:
        return {
            "fingerprint": self.fingerprint, "head": self.head, "branch": self.branch,
            "changes": self.changes[:30], "changes_total": len(self.changes),
            "tracked_and_untracked_files": len(self.files),
        }


def git(root: Path, *arguments: str, allow_failure: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "-C", str(root), *arguments],
        capture_output=True, timeout=30, check=False,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )
    if result.returncode and not allow_failure:
        raise HarnessError(f"Git inspection failed: {result.stderr.decode(errors='replace').strip()}")
    if len(result.stdout) > 8 * 1024 * 1024:
        raise HarnessError("Git inventory exceeds 8 MiB; narrow the project root.")
    return result


def inspect(root: Path, *, required: bool = False) -> Repository | None:
    root = root.resolve()
    try:
        location = git(root, "rev-parse", "--show-toplevel", allow_failure=True)
    except FileNotFoundError as error:
        if required:
            raise HarnessError("Git is required for verified quality mode.") from error
        return None
    if location.returncode:
        if b"not a git repository" not in location.stderr.lower():
            raise HarnessError("Cannot inspect repository; check Git permissions and configuration.")
        if required:
            raise HarnessError("Initialize Git before enabling verified quality mode.")
        return None
    top = Path(os.fsdecode(location.stdout.strip())).resolve()
    head_result = git(root, "rev-parse", "--verify", "HEAD", allow_failure=True)
    head = head_result.stdout.decode().strip() if not head_result.returncode else "unborn"
    branch_result = git(root, "symbolic-ref", "--quiet", "--short", "HEAD", allow_failure=True)
    branch = branch_result.stdout.decode().strip() if not branch_result.returncode else "detached"
    listing = git(root, "ls-files", "-z", "--full-name", "--cached", "--others",
                  "--exclude-standard", "--", ".").stdout
    names = sorted(set(os.fsdecode(item) for item in listing.split(b"\0") if item))
    if len(names) > 20000:
        raise HarnessError("Repository scope exceeds 20,000 files; choose a narrower root.")
    digest = hashlib.sha256()
    digest.update(json.dumps([head, branch, len(names)]).encode())
    files: list[str] = []
    total = 0
    for name in names:
        path = top / name
        # Do not dereference repository symlinks into other directories.
        if not path.parent.resolve().is_relative_to(root):
            raise HarnessError("Repository inventory contains a path outside the selected root.")
        relative = str(path.relative_to(root))
        files.append(relative)
        digest.update(os.fsencode(relative) + b"\0")
        if not path.exists() and not path.is_symlink():
            digest.update(b"deleted")
            continue
        attributes = path.lstat()
        digest.update(str(stat.S_IMODE(attributes.st_mode)).encode())
        if path.is_symlink():
            digest.update(b"link\0" + os.fsencode(os.readlink(path)))
        elif path.is_file():
            if attributes.st_size > 128 * 1024 * 1024:
                raise HarnessError(f"Tracked file exceeds 128 MiB: {relative}")
            total += attributes.st_size
            if total > 512 * 1024 * 1024:
                raise HarnessError("Repository fingerprint exceeds 512 MiB; narrow the root or ignore artifacts.")
            file_digest = hashlib.sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(65536), b""):
                    file_digest.update(block)
            digest.update(b"file\0" + file_digest.digest())
        else:
            raise HarnessError(f"Unsupported repository entry (including submodule): {relative}")
        digest.update(b"\0")
    raw_status = git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--", ".").stdout
    digest.update(raw_status)
    digest.update(git(root, "ls-files", "--stage", "-z", "--", ".").stdout)
    changes = []
    entries = iter(item for item in raw_status.split(b"\0") if item)
    for entry in entries:
        code = entry[:2].decode("ascii")
        entry_path = top / os.fsdecode(entry[3:])
        change = {"status": code, "path": str(entry_path.relative_to(root))}
        if "R" in code or "C" in code:
            change["from"] = os.fsdecode(next(entries))
        changes.append(change)
    return Repository(digest.hexdigest(), head, branch, changes, files)


def suggest(root: Path, query: str, limit: int = 8) -> list[dict]:
    if not 1 <= limit <= 12:
        raise HarnessError("Context suggestion limit must be between 1 and 12.")
    repository = inspect(root, required=True)
    assert repository is not None
    words = {word.lower() for word in re.findall(r"[A-Za-z_][A-Za-z_0-9]{2,}", query)}
    changed = {item["path"] for item in repository.changes}
    scored = []
    for name in repository.files:
        path = Path(name)
        lowered = name.lower()
        if path.suffix.lower() not in {
            ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".kt", ".java",
            ".swift", ".c", ".cpp", ".h", ".cs", ".json", ".toml", ".yaml", ".yml",
        } or any(part in lowered for part in (".env", "secret", "credential", "package-lock")):
            continue
        matches = sorted(word for word in words if word in lowered)
        score = 3 * len(matches) + (2 if name in changed else 0)
        if score:
            scored.append({"file": name, "score": score, "reason": {
                "path_matches": matches, "git_changed": name in changed,
                "test_candidate": "test" in lowered or "spec" in lowered,
            }})
    scored.sort(key=lambda item: (-item["score"], item["file"]))
    return scored[:limit]
