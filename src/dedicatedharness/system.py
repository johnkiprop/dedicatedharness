from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

GIB = 1024**3


class HarnessError(Exception):
    """An actionable user-facing failure."""


def command_output(arguments: list[str]) -> str:
    return subprocess.check_output(
        arguments, text=True, stderr=subprocess.PIPE, timeout=15
    ).strip()


def existing_parent(path: Path) -> Path:
    path = path.expanduser().resolve()
    while not path.exists():
        path = path.parent
    return path


@dataclass(frozen=True)
class Hardware:
    system: str
    machine: str
    chip: str
    memory_bytes: int | None
    available_bytes: int | None
    disk_free_bytes: int
    macos: str
    translated: bool
    notes: tuple[str, ...] = ()

    @property
    def apple_silicon(self) -> bool:
        return (
            self.system == "Darwin"
            and self.machine == "arm64"
            and not self.translated
        )


def detect_hardware(cache: Path) -> Hardware:
    system = platform.system()
    notes: list[str] = []
    memory = available = None
    chip = platform.processor() or platform.machine()
    translated = False
    if system == "Darwin":
        memory = int(command_output(["sysctl", "-n", "hw.memsize"]))
        chip = command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
        # An absent sysctl is normal on a native Intel Mac.
        translation = subprocess.run(
            ["sysctl", "-in", "sysctl.proc_translated"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        translated = translation.stdout.strip() == "1"
        try:
            vm = command_output(["vm_stat"])
            page_match = re.search(r"page size of (\d+) bytes", vm)
            counts = dict(re.findall(r"^(Pages [\w ]+):\s+(\d+)\.", vm, re.MULTILINE))
            if not page_match or "Pages free" not in counts:
                raise ValueError("unrecognized vm_stat output")
            # Do not count inactive/compressed pages as immediately free.
            available = int(page_match[1]) * (
                int(counts["Pages free"]) + int(counts.get("Pages speculative", "0"))
            )
            notes.append(
                "Available memory counts free/speculative pages only; "
                "reclaimable memory is deliberately excluded."
            )
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            notes.append(f"Available memory unavailable: {error}")
    else:
        notes.append(
            "Managed MLX requires native Apple Silicon macOS. "
            "Use --memory-gb for a recommendation preview, not installation."
        )
    return Hardware(
        system, platform.machine(), chip, memory, available,
        shutil.disk_usage(existing_parent(cache)).free,
        platform.mac_ver()[0], translated, tuple(notes),
    )


def require_apple(hardware: Hardware) -> None:
    if not hardware.apple_silicon:
        raise HarnessError(
            "Managed MLX requires native arm64 Python on Apple Silicon macOS. "
            "Intel/Rosetta/Windows can use an existing local endpoint instead."
        )
    version = tuple(int(part) for part in hardware.macos.split(".")[:2])
    if version < (14, 0):
        raise HarnessError("Managed MLX requires macOS 14 or newer.")


def default_home() -> Path:
    return Path(os.environ.get(
        "DEDICATEDHARNESS_HOME", str(Path.home() / ".dedicatedharness")
    )).expanduser().resolve()


def load_config(home: Path) -> dict[str, Any]:
    path = home / "config.json"
    if not path.exists():
        raise HarnessError("No configuration. Run 'dh setup' or 'dh connect' first.")
    return validate_config(json.loads(path.read_text(encoding="utf-8")), path)


def validate_config(value: Any, path: Path | str = "saved profile") -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("version") != 1:
        raise HarnessError(f"Unsupported configuration: {path}")
    required = ("backend", "base_url", "model", "context_tokens", "max_output")
    if any(key not in value for key in required):
        raise HarnessError(f"Incomplete configuration: {path}; run setup/connect again.")
    if value["backend"] not in ("mlx", "external"):
        raise HarnessError("Unsupported configured backend; run setup/connect again.")
    if any(not isinstance(value[key], str) or not value[key] for key in ("base_url", "model")):
        raise HarnessError("Configured URL and model must be nonempty strings.")
    if value["backend"] == "mlx" and any(
        not isinstance(value.get(key), str) or not value[key]
        for key in ("runtime_python", "snapshot")
    ):
        raise HarnessError("Managed runtime/snapshot configuration missing; run setup again.")
    for key in ("context_tokens", "max_output"):
        if type(value[key]) is not int or value[key] <= 0:
            raise HarnessError(f"Configuration {key} must be a positive integer.")
    if value["max_output"] >= value["context_tokens"]:
        raise HarnessError("Configured output budget must be smaller than context.")
    return value


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        with path.open("x", encoding="utf-8") as target:
            json.dump(value, target, indent=2)
            target.write("\n")
        return
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(value, target, indent=2)
            target.write("\n")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_config(home: Path, value: dict[str, Any]) -> None:
    from .activity import activity_lock
    with activity_lock(home):
        write_json(home / "config.json", value, overwrite=True)


def as_json(value: Any) -> str:
    if isinstance(value, Hardware):
        value = asdict(value)
    return json.dumps(value, indent=2)
