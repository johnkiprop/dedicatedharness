from __future__ import annotations

import copy
import hmac
import json
import secrets
import threading
import time
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator

from .local import LocalClient, Result, validate_sampling
from .prompts import admit
from .system import HarnessError

TOOLS = frozenset({"read", "glob", "grep", "edit", "write", "apply_patch", "invalid"})
MAX_BODY = 2 * 1024 * 1024


def compact(messages: list[dict[str, Any]], *, clear_old: bool = False) -> tuple[list[dict[str, Any]], int]:
    result = copy.deepcopy(messages)
    tools = [index for index, message in enumerate(result) if message.get("role") == "tool"]
    changed = 0
    for index in tools:
        message = result[index]
        content = message.get("content")
        if not isinstance(content, str):
            raise HarnessError("The supervised worker supports text-only tool results.")
        if clear_old and index != tools[-1]:
            replacement = "[Earlier tool output cleared. Re-read the source with a narrow line range if needed.]"
        elif len(content.encode("utf-8")) > 3072:
            data = content.encode("utf-8")
            replacement = (
                data[:1536].decode("utf-8", errors="ignore")
                + "\n[Tool output compacted. Use narrow read ranges or search to retrieve omitted content.]\n"
                + data[-1024:].decode("utf-8", errors="ignore")
            )
        else:
            continue
        message["content"] = replacement
        changed += 1
    return result, changed


class Gateway:
    def __init__(
        self, config: dict[str, Any], deadline: float, requests: int, budget: int,
        *, calls: list[dict[str, Any]] | None = None,
        root: Path | None = None, edits: list[str] | None = None, protected: list[str] | None = None,
        guard: Callable[[], None] | None = None,
    ):
        self.config = dict(config)
        validate_sampling(self.config.get("sampling", {}))
        self.deadline = deadline
        self.max_requests = requests
        self.budget = budget
        self.calls = list(calls or [])
        self.key = secrets.token_urlsafe(32)
        self.fatal = ""
        self.lock = threading.Lock()
        self.records_lock = threading.Lock()
        self.stopped = threading.Event()
        self.root = root
        self.edits = [root / name for name in edits or []] if root else []
        self.protected = {root / name for name in protected or []} if root else set()
        self.guard = guard

    def snapshot(self) -> list[dict[str, Any]]:
        with self.records_lock:
            return copy.deepcopy(self.calls)

    def record(self, call: dict[str, Any], **fields: Any) -> None:
        with self.records_lock:
            call.update(fields)

    def validate_path(self, name: Any, *, editing: bool) -> None:
        if self.root is None:
            return
        if not isinstance(name, str) or not name or "\0" in name:
            raise HarnessError("File tools require an explicit ordinary workspace path.")
        path = (self.root / name).resolve()
        if (
            not path.is_relative_to(self.root)
            or any(part in (".git", ".opencode") for part in path.relative_to(self.root).parts)
        ):
            raise HarnessError("Local model requested a file outside the permitted workspace.")
        if editing and (path in self.protected or not any(
            path == allowed or allowed.is_dir() and path.is_relative_to(allowed) for allowed in self.edits
        )):
            raise HarnessError("Local model requested an edit outside approved paths or to a protected grader.")

    def validate_tool(self, name: str, arguments: dict[str, Any]) -> None:
        if name in ("read", "edit", "write"):
            self.validate_path(arguments.get("filePath"), editing=name != "read")
        elif name in ("glob", "grep") and arguments.get("path") is not None:
            self.validate_path(arguments["path"], editing=False)
        elif name == "apply_patch":
            patch = arguments.get("patchText")
            if not isinstance(patch, str):
                raise HarnessError("Patch tool requires text.")
            files = []
            for line in patch.splitlines():
                for prefix in ("*** Add File: ", "*** Update File: ", "*** Delete File: ", "*** Move to: "):
                    if line.strip().startswith(prefix):
                        files.append(line.strip()[len(prefix):])
            if not files:
                raise HarnessError("Patch tool did not identify scoped files.")
            for filename in files:
                self.validate_path(filename, editing=True)

    def generate(self, payload: dict[str, Any]) -> Result:
        if not self.lock.acquire(blocking=False):
            raise HarnessError("Concurrent inference is disabled for this Mac worker.")
        call: dict[str, Any] = {"status": "refused"}
        try:
            if self.stopped.is_set() or time.monotonic() >= self.deadline:
                raise HarnessError("Supervised inference deadline reached.")
            if len(self.calls) >= self.max_requests:
                raise HarnessError("Supervised model-request limit reached.")
            with self.records_lock:
                self.calls.append(call)
            if self.guard:
                self.guard()
            if payload.get("model") != self.config["model"]:
                raise HarnessError("OpenCode requested a model other than the pinned local model.")
            messages = payload.get("messages")
            tools = payload.get("tools") or []
            if (
                not isinstance(messages, list) or not messages
                or any(not isinstance(message, dict) for message in messages)
                or not isinstance(tools, list)
                or any(not isinstance(tool, dict) or tool.get("type") != "function"
                       or not isinstance(tool.get("function"), dict) for tool in tools)
            ):
                raise HarnessError("Unsupported OpenAI-compatible agent request.")
            names = [tool["function"].get("name") for tool in tools]
            if any(not isinstance(name, str) or name not in TOOLS for name in names):
                raise HarnessError("OpenCode exposed a tool outside the supervised tool allowlist.")
            messages, clipped = compact(messages)
            try:
                admission = admit(self.config, messages, self.config["max_output"], tools)
            except HarnessError as error:
                if not str(error).startswith("Context budget exceeded:"):
                    raise
                messages, cleared = compact(messages, clear_old=True)
                clipped += cleared
                admission = admit(self.config, messages, self.config["max_output"], tools)
            cost = admission["input_budget_count"] + admission["output_limit"]
            used = sum(item.get("admitted_units", 0) for item in self.calls)
            if used + cost > self.budget:
                raise HarnessError("Total supervised admission budget reached; narrow the task.")
            self.record(call, **admission, admitted_units=cost, compacted_results=clipped, status="started")
            remaining = self.deadline - time.monotonic()
            if remaining <= 0 or self.stopped.is_set():
                raise HarnessError("Supervised inference deadline reached.")
            result = LocalClient(self.config["base_url"], timeout=remaining).chat(
                self.config["model"], messages, self.config["max_output"],
                tools=tools or None,
                sampling={"temperature": 0, "top_p": 1, **self.config.get("sampling", {})},
            )
            if self.guard:
                self.guard()
            metrics = result.metrics()
            if result.usage:
                metrics["usage"] = {key: value for key, value in result.usage.items()
                                    if key in ("prompt_tokens", "completion_tokens", "total_tokens")
                                    and type(value) is int and value >= 0}
            self.record(call, **metrics, status="completed")
            if result.finish_reason not in ("stop", "tool_calls"):
                raise HarnessError("Local response was incomplete or reached its output limit.")
            if len(result.tool_calls) > 16:
                raise HarnessError("Local response exceeded 16 tool calls.")
            ids = set()
            for tool in result.tool_calls:
                try:
                    arguments = json.loads(tool["function"]["arguments"])
                except ValueError as error:
                    raise HarnessError("Local model returned malformed tool arguments.") from error
                if (
                    not tool["id"] or tool["id"] in ids
                    or tool["function"]["name"] not in names
                    or not isinstance(arguments, dict)
                ):
                    raise HarnessError("Local model returned an invalid or unauthorized tool call.")
                ids.add(tool["id"])
                self.validate_tool(tool["function"]["name"], arguments)
            self.record(call, tools=[tool["function"]["name"] for tool in result.tool_calls])
            return result
        except (HarnessError, OSError, ValueError, TimeoutError) as error:
            self.record(call, status="failed")
            self.fatal = str(error) if isinstance(error, HarnessError) else "Local inference transport failed."
            raise HarnessError(self.fatal) from error
        finally:
            self.lock.release()

    @contextmanager
    def serve(self) -> Iterator[str]:
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(max(0.1, min(10, gateway.deadline - time.monotonic())))

            def log_message(self, *_arguments):
                pass

            def reply(self, status: int, value: Any) -> None:
                body = json.dumps(value).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def authorized(self) -> bool:
                return hmac.compare_digest(
                    self.headers.get("Authorization", ""), "Bearer " + gateway.key,
                )

            def do_GET(self):
                if not self.authorized():
                    self.reply(401, {"error": {"message": "Authentication required."}})
                elif self.path == "/v1/models":
                    self.reply(200, {"object": "list", "data": [
                        {"id": gateway.config["model"], "object": "model", "owned_by": "local"},
                    ]})
                else:
                    self.reply(404, {"error": {"message": "Unsupported route."}})

            def do_POST(self):
                if not self.authorized():
                    self.reply(401, {"error": {"message": "Authentication required."}})
                    return
                if self.path != "/v1/chat/completions":
                    self.reply(404, {"error": {"message": "Unsupported route."}})
                    return
                try:
                    self.connection.settimeout(max(0.1, min(10, gateway.deadline - time.monotonic())))
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= MAX_BODY or self.headers.get("Transfer-Encoding"):
                        raise HarnessError("Agent request exceeds its 2 MiB limit or has unsupported framing.")
                    raw = self.rfile.read(length)
                    if len(raw) != length:
                        raise HarnessError("Incomplete agent request body.")
                    payload = json.loads(raw)
                    if not isinstance(payload, dict) or payload.get("stream") is not True:
                        raise HarnessError("The supervised worker requires streaming chat completions.")
                    result = gateway.generate(payload)
                except (HarnessError, OSError, ValueError) as error:
                    gateway.fatal = str(error) if isinstance(error, HarnessError) else "Invalid agent request."
                    try:
                        self.reply(400, {"error": {"message": gateway.fatal, "type": "harness_limit"}})
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                if gateway.stopped.is_set():
                    return
                packet = {
                    "id": "chatcmpl-dh-" + uuid.uuid4().hex,
                    "object": "chat.completion.chunk", "created": int(time.time()),
                    "model": gateway.config["model"],
                }
                delta: dict[str, Any] = {"role": "assistant", "content": result.text}
                if result.tool_calls:
                    delta["tool_calls"] = [
                        {"index": index, **call} for index, call in enumerate(result.tool_calls)
                    ]
                frames = [
                    {**packet, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                    {**packet, "choices": [{"index": 0, "delta": {}, "finish_reason": result.finish_reason}]},
                ]
                if result.usage:
                    frames.append({**packet, "choices": [], "usage": result.usage})
                body = "".join("data: " + json.dumps(frame) + "\n\n" for frame in frames)
                body += "data: [DONE]\n\n"
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(body.encode("utf-8"))))
                    self.end_headers()
                    self.wfile.write(body.encode("utf-8"))
                except (BrokenPipeError, ConnectionResetError):
                    gateway.stopped.set()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = False
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1"
        finally:
            self.stopped.set()
            server.shutdown()
            server.server_close()
            worker.join()
