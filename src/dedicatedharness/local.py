from __future__ import annotations

import ipaddress
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Callable

from .system import HarnessError


def validate_sampling(settings: dict[str, Any]) -> None:
    if not isinstance(settings, dict) or set(settings) - {"temperature", "top_p", "seed"}:
        raise HarnessError("Unsupported sampling settings.")
    for key, low, high in (("temperature", 0, 2), ("top_p", 0.000001, 1)):
        value = settings.get(key, 0 if key == "temperature" else 1)
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise HarnessError(f"Sampling {key} must be finite and between {low} and {high}.")
    if "seed" in settings and type(settings["seed"]) is not int:
        raise HarnessError("Sampling seed must be an integer.")


def local_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "http" or parsed.username or parsed.password
        or parsed.query or parsed.fragment
    ):
        raise HarnessError("Use an http:// loopback URL without credentials/query/fragment.")
    try:
        host = ipaddress.ip_address(parsed.hostname or "")
        port = parsed.port
    except ValueError as error:
        raise HarnessError(
            "Use a numeric loopback address, e.g. http://127.0.0.1:1234/v1; "
            "DNS names are not accepted for local-only inference."
        ) from error
    if not host.is_loopback or port is None or not 1 <= port <= 65535:
        raise HarnessError("Inference is restricted to numeric loopback addresses with a port.")
    if parsed.path.rstrip("/") != "/v1":
        raise HarnessError("OpenAI-compatible base URL must end in /v1.")
    return url.rstrip("/")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HarnessError("Inference redirects are refused to preserve local-only routing.")


@dataclass
class Result:
    text: str
    elapsed_seconds: float
    first_visible_seconds: float | None
    usage: dict[str, Any] | None
    finish_reason: str | None
    tool_calls: list[dict[str, Any]]

    def metrics(self) -> dict[str, Any]:
        return {
            key: value for key, value in asdict(self).items()
            if key not in ("text", "tool_calls")
        }


class LocalClient:
    def __init__(self, base_url: str, timeout: float = 120):
        self.base_url = local_url(base_url)
        self.timeout = timeout
        # Never route loopback requests through HTTP_PROXY/HTTPS_PROXY.
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), NoRedirect()
        )

    def request(self, route: str, payload: dict[str, Any] | None = None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + route, data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            return self.opener.open(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            # The response may echo a private prompt. Never print its body.
            raise HarnessError(
                f"Local server returned HTTP {error.code} for {route}; check server logs."
            ) from error
        except urllib.error.URLError as error:
            raise HarnessError(
                f"Local server unavailable at {self.base_url}: {error.reason}. "
                "Start 'dh serve' or the LM Studio local server."
            ) from error

    def models(self) -> list[str]:
        with self.request("/models") as response:
            value = json.load(response)
        if not isinstance(value, dict) or not isinstance(value.get("data"), list):
            raise HarnessError("Local /models response must contain a data array.")
        models = [
            item["id"] for item in value["data"]
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]
        if not models:
            raise HarnessError("Local server exposes no models; load one and retry.")
        return models

    def chat(
        self, model: str, messages: list[dict[str, Any]], max_tokens: int,
        on_text: Callable[[str], None] | None = None,
        tools: list[dict[str, Any]] | None = None,
        sampling: dict[str, Any] | None = None,
    ) -> Result:
        payload = {
            "model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
        if sampling is not None:
            validate_sampling(sampling)
            payload.update(sampling)
        start = time.perf_counter()
        first = None
        chunks: list[str] = []
        usage = finish = None
        done = False
        calls: dict[int, dict[str, Any]] = {}
        with self.request("/chat/completions", payload) as response:
            event: list[str] = []
            received = 0
            while raw := response.readline(1024 * 1024 + 1):
                received += len(raw)
                if len(raw) > 1024 * 1024 or received > 8 * 1024 * 1024:
                    raise HarnessError("Local inference stream exceeded its bounded response size.")
                if time.perf_counter() - start > self.timeout:
                    raise HarnessError("Local generation exceeded its wall-clock deadline.")
                line = raw.decode("utf-8").rstrip("\r\n")
                if line.startswith("data:"):
                    event.append(line[5:].lstrip())
                    continue
                if line or not event:
                    continue
                content = "\n".join(event)
                event.clear()
                if content == "[DONE]":
                    done = True
                    break
                packet = json.loads(content)
                if packet.get("error"):
                    raise HarnessError("Local server reported a streaming error; check server logs.")
                if isinstance(packet.get("usage"), dict):
                    usage = packet["usage"]
                for choice in packet.get("choices", []):
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
                    delta = choice.get("delta", {})
                    text = delta.get("content")
                    if text:
                        if not isinstance(text, str):
                            raise HarnessError("Non-text content returned by text-only backend.")
                        if first is None:
                            first = time.perf_counter() - start
                        chunks.append(text)
                        if on_text:
                            on_text(text)
                    for call in delta.get("tool_calls", []):
                        index = call["index"]
                        merged = calls.setdefault(index, {
                            "id": "", "type": "function",
                            "function": {"name": "", "arguments": ""},
                        })
                        if call.get("id"):
                            merged["id"] = call["id"]
                        for key in ("name", "arguments"):
                            merged["function"][key] += call.get("function", {}).get(key, "")
        if not done:
            raise HarnessError("Incomplete event stream: server closed before [DONE].")
        if not chunks and not calls:
            raise HarnessError("Model returned no visible text or structured tool calls.")
        return Result(
            "".join(chunks), time.perf_counter() - start, first,
            usage, finish, [calls[index] for index in sorted(calls)],
        )
