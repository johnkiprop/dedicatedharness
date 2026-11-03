from __future__ import annotations

import json
import os
import secrets
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

from .agent_gateway import Gateway
from .quality import CODING_CONTRACT, stop_owned_process
from .system import HarnessError, write_json

AGENT = "dedicatedharness"
MODEL = {"providerID": AGENT, "modelID": "local"}


def engine_config(
    config: dict[str, Any], gateway: Gateway, url: str, root: Path,
    edits: list[str], protected: list[str], steps: int,
) -> dict[str, Any]:
    edits_rule = {"*": "deny"}
    def patterns(name: str) -> list[str]:
        path = root / name
        return list(dict.fromkeys([name, Path(name).as_posix(), str(path), path.as_posix()]))

    for name in edits:
        path = root / name
        for pattern in patterns(name):
            edits_rule[pattern] = "allow"
            if path.is_dir():
                edits_rule[pattern.rstrip("\\/") + "/*"] = "allow"
    for name in [".git", ".opencode", *protected]:
        for pattern in patterns(name):
            edits_rule[pattern] = "deny"
            edits_rule[pattern.rstrip("\\/") + "/*"] = "deny"
    for pattern in ("*.env", "*.env.*", "*/.git/*", "*/.opencode/*", "*\\.git\\*", "*\\.opencode\\*"):
        edits_rule[pattern] = "deny"
    permission = {
        "*": "deny", "read": {"*": "allow", "*.env": "deny", "*.env.*": "deny",
                             "*/.git/*": "deny", "*credentials*": "deny", "*.pem": "deny"},
        "glob": "allow", "grep": "allow", "edit": edits_rule,
        "external_directory": "deny", "doom_loop": "deny",
    }
    return {
        "$schema": "https://opencode.ai/config.json",
        "enabled_providers": [AGENT], "model": AGENT + "/local", "small_model": AGENT + "/local",
        "default_agent": AGENT, "share": "disabled", "autoupdate": False, "snapshot": False,
        "plugin": [], "mcp": {}, "lsp": False, "formatter": False, "instructions": [],
        "compaction": {"auto": False, "prune": False},
        "provider": {AGENT: {
            "npm": "@ai-sdk/openai-compatible", "name": "Bounded local worker",
            "options": {"baseURL": url, "apiKey": gateway.key, "timeout": 600000},
            "models": {"local": {
                "id": config["model"], "name": config.get("repo", config["model"]),
                "tool_call": True, "reasoning": False,
                "limit": {"context": config["context_tokens"], "output": config["max_output"]},
            }},
        }},
        "permission": permission,
        "agent": {
            **{name: {"disable": True, "options": {}, "permission": {}} for name in (
                "build", "plan", "general", "explore", "compaction", "title", "summary",
            )},
            AGENT: {
                "description": "One bounded local coding worker", "mode": "primary",
                "options": {},
                "model": AGENT + "/local", "steps": steps, "permission": permission,
                "prompt": (
                    CODING_CONTRACT
                    + " Read applicable AGENTS.md instructions before editing. "
                    "Read only relevant files, using small line ranges and targeted search. "
                    "Tool outputs may be compacted; retrieve omitted source when needed. "
                    "Use file tools only. Shell, network, skills and subagents are disabled. "
                    "The supervisor runs trusted checks after your turn. "
                    "Do not edit outside the listed editable paths. "
                    "Stop if scope is insufficient; report changed paths and unresolved issues."
                ),
            },
        },
    }


def environment(directory: Path, configuration: dict[str, Any], password: str) -> dict[str, str]:
    result = {
        key: value for key, value in os.environ.items()
        if key.upper() in {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP",
                           "LANG", "LC_ALL", "OPENCODE_GIT_BASH_PATH"}
    }
    for variable, child in (
        ("HOME", "home"), ("USERPROFILE", "home"), ("APPDATA", "appdata"),
        ("LOCALAPPDATA", "localappdata"), ("XDG_CONFIG_HOME", "config"),
        ("XDG_DATA_HOME", "data"), ("XDG_CACHE_HOME", "cache"), ("XDG_STATE_HOME", "state"),
    ):
        path = directory / child
        path.mkdir(parents=True, exist_ok=True)
        result[variable] = str(path)
    catalog = directory / "models.json"
    write_json(catalog, {})
    result.update({
        "OPENCODE_CONFIG_CONTENT": json.dumps(configuration),
        "OPENCODE_CONFIG_DIR": str(directory / "config"),
        "OPENCODE_SERVER_USERNAME": AGENT, "OPENCODE_SERVER_PASSWORD": password,
        "OPENCODE_MODELS_PATH": str(catalog),
        "OPENCODE_CLIENT": AGENT, "OPENCODE_PURE": "true",
        "OPENCODE_DISABLE_PROJECT_CONFIG": "true",
        "OPENCODE_DISABLE_EXTERNAL_SKILLS": "true",
        "OPENCODE_DISABLE_CLAUDE_CODE": "true",
        "OPENCODE_DISABLE_DEFAULT_PLUGINS": "true",
        "OPENCODE_DISABLE_MODELS_FETCH": "true",
        "OPENCODE_DISABLE_AUTOUPDATE": "true",
        "OPENCODE_DISABLE_AUTOCOMPACT": "true",
        "OPENCODE_DISABLE_LSP_DOWNLOAD": "true",
        "OPENCODE_EXPERIMENTAL_DISABLE_FILEWATCHER": "true",
        "DO_NOT_TRACK": "1",
    })
    return result


class Engine:
    def __init__(
        self, binary: str, directory: Path, root: Path, configuration: dict[str, Any],
        deadline: float,
    ):
        self.binary = binary
        self.directory = directory
        self.root = root
        self.configuration = configuration
        self.deadline = deadline
        self.process: subprocess.Popen | None = None
        self.client: httpx.Client | None = None
        self.log = None
        self.session = ""
        self.version = ""
        self.ready = False

    def request(self, method: str, route: str, body: dict[str, Any] | None = None) -> Any:
        if self.client is None:
            raise HarnessError("OpenCode engine is not running.")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise HarnessError("Supervised wall-clock deadline reached.")
        with self.client.stream(
            method, route, json=body, params={"directory": str(self.root)},
            timeout=min(15, remaining),
        ) as response:
            if not response.is_success:
                raise HarnessError(f"OpenCode returned HTTP {response.status_code} for {route}.")
            data = bytearray()
            for block in response.iter_bytes():
                data.extend(block)
                if len(data) > 8 * 1024 * 1024:
                    raise HarnessError("OpenCode response exceeded its 8 MiB limit.")
            return json.loads(data) if data else None

    def __enter__(self) -> Engine:
        try:
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=False)
            password = secrets.token_urlsafe(32)
            env = environment(self.directory, self.configuration, password)
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            self.log = (self.directory / "engine.log").open("w+b")
            options = (
                {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                if os.name == "nt" else {"start_new_session": True}
            )
            self.process = subprocess.Popen(
                [self.binary, "--pure", "serve", "--hostname", "127.0.0.1", "--port", str(port)],
                cwd=self.root, env=env, stdin=subprocess.DEVNULL,
                stdout=self.log, stderr=subprocess.STDOUT, shell=False, **options,
            )
            self.client = httpx.Client(
                base_url=f"http://127.0.0.1:{port}", auth=(AGENT, password),
                trust_env=False, follow_redirects=False,
            )
            startup = min(self.deadline, time.monotonic() + 45)
            while time.monotonic() < startup:
                if self.process.poll() is not None:
                    self.log.seek(0)
                    detail = self.log.read(2048).decode("utf-8", errors="replace")
                    raise HarnessError(f"OpenCode startup failed: {detail}")
                try:
                    health = self.request("GET", "/global/health")
                except httpx.TransportError:
                    time.sleep(0.1)
                    continue
                if not isinstance(health, dict) or health.get("healthy") is not True:
                    raise HarnessError("Unsupported OpenCode health response.")
                self.version = str(health.get("version", ""))
                self.ready = True
                if not self.version.startswith("1."):
                    raise HarnessError("This adapter requires OpenCode 1.x; validated with 1.19.4, not OpenCode 2.")
                break
            else:
                raise HarnessError("OpenCode did not become ready within the startup deadline.")
            actual = self.request("GET", "/config")
            for key, expected in self.configuration.items():
                if key != "$schema" and actual.get(key) != expected:
                    raise HarnessError(f"OpenCode configuration isolation failed for {key}; nothing was sent.")
            path = self.request("GET", "/path")
            if Path(path["directory"]).resolve() != self.root:
                raise HarnessError("OpenCode workspace differs from the approved project root.")
            if Path(path["worktree"]).resolve() != self.root:
                raise HarnessError("Supervised execution requires the Git checkout root, not a nested project root.")
            providers = self.request("GET", "/config/providers")
            if {provider["id"] for provider in providers["providers"]} != {AGENT}:
                raise HarnessError("OpenCode has an unexpected active provider; nothing was sent.")
            return self
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        try:
            if self.ready and self.client is not None and self.process is not None and self.process.poll() is None:
                if self.session:
                    self.client.post(f"/session/{self.session}/abort",
                                     params={"directory": str(self.root)}, timeout=3).raise_for_status()
                self.client.post("/instance/dispose",
                                 params={"directory": str(self.root)}, timeout=3).raise_for_status()
        finally:
            self.ready = False
            if self.process is not None:
                stop_owned_process(self.process)
                self.process = None
            if self.client is not None:
                self.client.close()
                self.client = None
            if self.log is not None:
                self.log.close()
                self.log = None

    def __exit__(self, *_arguments) -> None:
        self.close()

    def observe(self) -> dict[str, Any] | None:
        if self.process is None or self.process.poll() is not None:
            raise HarnessError("Owned OpenCode process exited before its turn completed.")
        if self.log is not None and os.fstat(self.log.fileno()).st_size > 8 * 1024 * 1024:
            raise HarnessError("OpenCode process output exceeded its 8 MiB limit.")
        total = count = 0
        for directory, folders, files in os.walk(self.directory, followlinks=False):
            count += len(folders) + len(files)
            if count > 4096:
                raise HarnessError("OpenCode private state exceeded 4,096 entries.")
            for name in files:
                try:
                    total += (Path(directory) / name).stat().st_size
                except FileNotFoundError:
                    continue  # OpenCode removes temporary locks while it is running.
            if total > 64 * 1024 * 1024:
                raise HarnessError("OpenCode private state exceeded 64 MiB.")
        messages = self.request("GET", f"/session/{self.session}/message")
        if not isinstance(messages, list) or len(messages) > 128:
            raise HarnessError("Unsupported or oversized OpenCode session history.")
        assistants = [message for message in messages if message["info"]["role"] == "assistant"]
        counts: dict[str, int] = {}
        tool_errors = 0
        for message in assistants:
            info = message["info"]
            if (info.get("providerID"), info.get("modelID")) != (AGENT, "local"):
                raise HarnessError("Observed OpenCode session used an unexpected model/provider.")
            if info.get("error"):
                raise HarnessError("OpenCode turn failed: " + str(info["error"].get("name", "agent error")))
            for part in message["parts"]:
                if part.get("type") == "tool":
                    name = part["tool"]
                    from .agent_gateway import TOOLS
                    if name not in TOOLS:
                        raise HarnessError("Observed a tool outside the execution allowlist.")
                    counts[name] = counts.get(name, 0) + 1
                    state = part.get("state", {}).get("status")
                    tool_errors += int(state == "error")
                    if state not in ("completed", "error"):
                        return None
        if not assistants:
            return None
        last = assistants[-1]["info"]
        if not last.get("time", {}).get("completed") or last.get("finish") not in ("stop", "end_turn"):
            return None
        status = self.request("GET", "/session/status").get(self.session, {"type": "idle"})
        if status.get("type") != "idle":
            return None
        if self.request("GET", f"/session/{self.session}/children"):
            raise HarnessError("Unexpected subagent session; supervised runs permit one worker only.")
        return {
            "messages": len(messages), "assistant_messages": len(assistants),
            "tools": counts, "tool_errors": tool_errors, "terminal": True,
        }

    def start(self, prompt: str) -> str:
        created = self.request("POST", "/session", {"title": "Dedicated Harness supervised task"})
        session = created.get("id")
        if not isinstance(session, str) or not session.startswith("ses_") or not session.replace("_", "").isalnum():
            raise HarnessError("OpenCode returned an invalid session identifier.")
        self.session = session
        self.request("POST", f"/session/{session}/prompt_async", {
            "model": MODEL, "agent": AGENT,
            "parts": [{"type": "text", "text": prompt}],
        })
        return session


def executable(value: str) -> str:
    result = shutil.which(value)
    if not result:
        raise HarnessError(
            "OpenCode is missing. Install OpenCode 1.x (adapter validated with 1.19.4), "
            "or select its executable with --opencode PATH. No engine was installed automatically."
        )
    return str(Path(result).resolve())
