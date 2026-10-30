from __future__ import annotations

import os
import socket
import subprocess
import time
import urllib.parse
import venv
from pathlib import Path
from typing import Any

from .local import LocalClient
from .system import HarnessError, detect_hardware, require_apple


def runtime_python(home: Path) -> Path:
    return home / "runtime" / "bin" / "python"


def prepare_runtime(home: Path, install: bool) -> tuple[Path, str]:
    require_apple(detect_hardware(home / "hub"))
    python = runtime_python(home)
    if install:
        venv.EnvBuilder(with_pip=True).create(home / "runtime")
        subprocess.run(
            [str(python), "-m", "pip", "install", "--upgrade", "mlx-lm>=0.26.3,<1"],
            check=True,
        )
    if not python.is_file():
        raise HarnessError("MLX runtime is missing; pass --install-runtime to create it.")
    version = subprocess.check_output(
        [str(python), "-c",
         "from importlib.metadata import version; print(version('mlx-lm'))"],
        text=True, timeout=30,
    ).strip()
    return python, version


def serve(config: dict[str, Any], home: Path, startup_timeout: int) -> None:
    require_apple(detect_hardware(home / "hub"))
    if config["backend"] != "mlx":
        raise HarnessError("This connection is externally managed; start its local server.")
    python = Path(config["runtime_python"])
    snapshot = Path(config["snapshot"])
    if not python.is_file() or not snapshot.is_dir():
        raise HarnessError("Runtime or downloaded snapshot missing; rerun setup.")
    client = LocalClient(config["base_url"], timeout=5)
    parsed = urllib.parse.urlsplit(client.base_url)
    # Refuse to attach to an unrelated server occupying the selected port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", parsed.port))
        except OSError as error:
            raise HarnessError(f"Port {parsed.port} is occupied; no server was started.") from error
    environment = {
        **os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_CACHE": str(home / "hub"),
    }
    arguments = [
        str(python), "-m", "mlx_lm.server", "--model", str(snapshot),
        "--host", "127.0.0.1", "--port", str(parsed.port),
        "--max-tokens", str(config["max_output"]), "--log-level", "WARNING",
    ]
    help_text = subprocess.check_output(
        [str(python), "-m", "mlx_lm.server", "--help"],
        env=environment, text=True, timeout=60,
    )
    for flag, value in (
        ("--decode-concurrency", "1"), ("--prompt-concurrency", "1"),
        ("--prompt-cache-size", "1"), ("--allowed-origins", "http://127.0.0.1"),
    ):
        if flag in help_text:
            arguments.extend((flag, value))
        elif flag != "--allowed-origins":
            raise HarnessError(
                f"Installed MLX lacks {flag}; rerun setup --install-runtime."
            )
    process = subprocess.Popen(arguments, env=environment)
    try:
        deadline = time.monotonic() + startup_timeout
        last_error = "not yet reachable"
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise HarnessError(f"MLX server exited with status {process.returncode}.")
            try:
                models = client.models()
                if config["model"] not in models:
                    raise HarnessError("Started server does not expose the selected snapshot.")
                break
            except HarnessError as error:
                last_error = str(error)
                time.sleep(0.5)
        else:
            raise HarnessError(f"MLX server startup timed out: {last_error}")
        print("API ready. Verifying a real local generation...", flush=True)
        LocalClient(config["base_url"], timeout=startup_timeout).chat(
            config["model"], [{"role": "user", "content": "Reply with OK."}], 32,
        )
        print(
            f"Local generation verified at {config['base_url']}. "
            "Keep this terminal open; Ctrl-C stops only this server.",
            flush=True,
        )
        returncode = process.wait()
        if returncode:
            raise HarnessError(f"MLX server exited with status {returncode}.")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
