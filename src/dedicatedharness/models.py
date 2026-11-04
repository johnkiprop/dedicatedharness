from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .system import GIB, Hardware, HarnessError

# Unknown architectures remain visible but are never automatically installed.
ATTENTION_MODELS = frozenset({"qwen2", "qwen3", "qwen3_moe", "llama", "mistral"})
ARTIFACT_PATTERNS = [
    "*.safetensors", "*.json", "*.jinja", "*.model", "*.tiktoken",
    "merges.txt", "vocab.txt", "README.md", "LICENSE*",
]


@dataclass(frozen=True)
class Candidate:
    repo: str
    revision: str
    license: str
    weight_bytes: int
    download_bytes: int
    model_type: str
    context_limit: int | None
    kv_bytes: int | None
    required_bytes: int | None
    fit: str
    reasons: tuple[str, ...]
    has_tool_template: bool
    modified_at: str = ""
    downloads: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def selected_files(info: Any) -> list[Any]:
    from fnmatch import fnmatch

    return [
        item for item in info.siblings or []
        if any(fnmatch(item.rfilename, pattern) for pattern in ARTIFACT_PATTERNS)
    ]


def assess(
    info: Any, config: dict[str, Any], tokenizer: dict[str, Any],
    hardware: Hardware, context: int, *, cached: bool = False,
) -> Candidate:
    if context <= 0:
        raise HarnessError("Context must be positive.")
    reasons: list[str] = []
    files = selected_files(info)
    sizes_known = bool(files) and all(
        type(item.size) is int and item.size >= 0 for item in files
    )
    weights = [item for item in files if item.rfilename.endswith(".safetensors")]
    weight_bytes = sum(item.size or 0 for item in weights)
    download_bytes = sum(item.size or 0 for item in files)
    model_type = str(config.get("model_type", "unknown"))
    supported = (
        model_type in ATTENTION_MODELS
        and not config.get("vision_config")
        and info.pipeline_tag in (None, "text-generation")
        and bool(config.get("quantization"))
        and bool(weights)
    )
    if not supported:
        reasons.append("Not a vetted quantized text-only MLX architecture.")
    if not sizes_known:
        reasons.append("Artifact sizes unavailable; installation cannot be budgeted.")
    template = tokenizer.get("chat_template", "")
    template_text = json.dumps(template)
    if not template:
        supported = False
        reasons.append("No chat template found; automatic setup refused.")
    card = info.card_data
    license_name = card.get("license", "unknown") if card else "unknown"
    if isinstance(license_name, list):
        license_name = ", ".join(license_name)
    if license_name == "unknown":
        reasons.append("License is unknown; inspect the upstream license.")
    limit = config.get("max_position_embeddings")
    limit = limit if type(limit) is int and limit > 0 else None
    if limit is None:
        reasons.append("Context limit unknown.")
    elif context > limit:
        reasons.append(f"Requested context exceeds model limit {limit}.")

    kv_bytes = None
    layers = config.get("num_hidden_layers")
    kv_heads = config.get("num_key_value_heads")
    heads = config.get("num_attention_heads")
    head_dim = config.get("head_dim")
    hidden = config.get("hidden_size")
    if head_dim is None and type(hidden) is int and type(heads) is int and heads > 0:
        head_dim = hidden // heads
    dimensions = (layers, kv_heads, head_dim)
    if supported and all(type(value) is int and value > 0 for value in dimensions):
        # Full context, K and V, fp16/bf16, one sequence. Never infer from Q4 weights.
        kv_bytes = 2 * layers * kv_heads * head_dim * context * 2
    else:
        reasons.append("KV estimate unavailable; no memory-fit claim.")
    required = None
    fit = "uncertain"
    if not supported:
        fit = "unsupported"
    elif sizes_known and kv_bytes is not None and hardware.memory_bytes is not None:
        reserve = max(6 * GIB, int(hardware.memory_bytes * 0.25))
        scratch = max(2 * GIB, int(weight_bytes * 0.15))
        working_set = weight_bytes + kv_bytes + scratch
        required = working_set + reserve
        if limit is None:
            fit = "uncertain"
        elif context > limit or required > hardware.memory_bytes:
            fit = "too-large"
            reasons.append("Weights, context, scratch, and OS/apps reserve exceed budget.")
        elif hardware.available_bytes is None:
            fit = "plausible"
            reasons.append("Current available memory unknown; fit is not guaranteed.")
        elif working_set > hardware.available_bytes:
            fit = "memory-pressure"
            reasons.append(
                f"Estimated available memory {hardware.available_bytes / GIB:.2f} GiB "
                f"is below the model working-set estimate {working_set / GIB:.2f} GiB "
                "(weights + context + scratch); unload other models/close apps and retry."
            )
        else:
            fit = "plausible"
    if not cached and download_bytes + 2 * GIB > hardware.disk_free_bytes:
        fit = "disk-full"
        reasons.append("Insufficient disk space, including 2 GiB staging reserve.")
    reasons.append("Conservative estimate, not measured peak memory or speed.")
    return Candidate(
        info.id, info.sha, str(license_name), weight_bytes, download_bytes,
        model_type, limit, kv_bytes, required, fit, tuple(reasons),
        "tools" in template_text and "tool_call" in template_text,
        str(getattr(info, "last_modified", "") or ""),
        int(getattr(info, "downloads", 0) or 0),
    )


class Hub:
    def __init__(self, cache: Path):
        from huggingface_hub import HfApi

        self.cache = cache
        self.api = HfApi(
            endpoint="https://huggingface.co",
            library_name="dedicatedharness", library_version="0.1.0",
        )

    def inspect(
        self, repo: str, hardware: Hardware, context: int, revision: str | None = None,
    ) -> Candidate:
        from huggingface_hub import hf_hub_download

        info = self.api.model_info(repo, files_metadata=True, revision=revision)
        if info.gated:
            raise HarnessError(
                f"{repo} is gated. Review its terms and authenticate with 'hf auth login'."
            )
        if not info.sha:
            raise HarnessError(f"{repo}: missing immutable repository revision.")
        metadata: dict[str, dict[str, Any]] = {}
        for filename in ("config.json", "tokenizer_config.json"):
            path = hf_hub_download(
                repo, filename, revision=info.sha, cache_dir=self.cache,
                endpoint="https://huggingface.co",
            )
            value = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise HarnessError(f"{repo}/{filename} is not a JSON object.")
            metadata[filename] = value
        # Current Transformers saves chat templates separately from tokenizer_config.
        filenames = {item.rfilename for item in info.siblings or []}
        if (
            not metadata["tokenizer_config.json"].get("chat_template")
            and "chat_template.jinja" in filenames
        ):
            template_path = hf_hub_download(
                repo, "chat_template.jinja", revision=info.sha, cache_dir=self.cache,
                endpoint="https://huggingface.co",
            )
            metadata["tokenizer_config.json"]["chat_template"] = Path(
                template_path
            ).read_text(encoding="utf-8")
        return assess(
            info, metadata["config.json"], metadata["tokenizer_config.json"],
            hardware, context,
        )

    def discover(
        self, search: str, limit: int, hardware: Hardware, context: int,
        sort: str = "downloads",
    ) -> list[Candidate]:
        models = self.api.list_models(
            author="mlx-community", search=search, sort=sort,
            limit=limit, pipeline_tag="text-generation",
        )
        return [self.inspect(model.id, hardware, context) for model in models]

    def recommend(
        self, purpose: str, hardware: Hardware, context: int,
        measured: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[Candidate, list[Candidate]]:
        if purpose not in ("coding", "reasoning"):
            raise HarnessError("Choose coding or reasoning.")
        candidates: dict[str, Candidate] = {}
        searches = (
            ("Coder", "coding", "Qwen3-4B-Instruct") if purpose == "coding"
            else ("Qwen3", "Instruct")
        )
        for search in searches:
            for candidate in self.discover(search, 6, hardware, context):
                candidates[candidate.repo] = candidate
        # Also examine recent conversions rather than freezing discovery to old popular models.
        for candidate in self.discover(searches[0], 4, hardware, context, sort="last_modified"):
            candidates[candidate.repo] = candidate
        measured = measured or {}
        eligible = [
            item for item in candidates.values()
            if item.fit == "plausible" and item.license != "unknown"
            and (purpose != "coding" or item.has_tool_template)
        ]
        if not eligible:
            fits = ", ".join(
                f"{fit}={sum(item.fit == fit for item in candidates.values())}"
                for fit in sorted({item.fit for item in candidates.values()})
            )
            available = (
                f" Estimated available memory: {hardware.available_bytes / GIB:.2f} GiB."
                if hardware.available_bytes is not None else ""
            )
            raise HarnessError(
                "No compatible, safely budgeted recommendation. "
                f"Screened candidates: {len(candidates)} ({fits or 'none'}).{available} "
                "Candidates also need a declared license and, for coding, a tool template. "
                "Inspect 'dh models --repo OWNER/MODEL' for detailed fit reasons."
            )

        def rank(candidate: Candidate) -> tuple:
            evidence = measured.get(candidate.repo + "@" + candidate.revision, {})
            affinity = int(
                ("coder" in candidate.repo.lower() or "coding" in candidate.repo.lower())
                if purpose == "coding" else
                ("qwen3" in candidate.repo.lower() or "reason" in candidate.repo.lower())
            )
            return (
                bool(evidence.get("target_observed")), evidence.get("success_rate", 0),
                affinity, candidate.modified_at, candidate.downloads,
            )

        eligible.sort(key=rank, reverse=True)
        return eligible[0], list(candidates.values())

    def download(self, candidate: Candidate) -> Path:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(
            candidate.repo, revision=candidate.revision, cache_dir=self.cache,
            endpoint="https://huggingface.co", allow_patterns=ARTIFACT_PATTERNS,
            max_workers=4,
        ))
