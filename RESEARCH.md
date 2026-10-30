# Dedicated Harness: research and recommendations

Research date: 2026-10-30. These recommendations are based on the primary
sources linked below, not on measured performance on the user's Mac.
The development environment is Windows; Mac hardware and runtime behavior
have not yet been verified.

## Recommendation

Build a small, local-first companion CLI for existing coding agents. Do not
start by building a replacement for OpenCode, OpenClaw, or LangGraph.

Its useful differentiation should be:

1. Detect Mac hardware and explain which model/runtime combinations are
   plausible, including memory and storage headroom.
2. Discover compatible Hugging Face candidates, download a reviewed selection,
   and prepare an isolated runtime without changing existing agent defaults.
3. Benchmark the user's actual workload before recommending a default.
4. Guide requests into clear task briefs and bounded context.
5. Export reviewed integration configuration and report latency/token usage.

The implementation now also provides private per-checkout project/task state:
compact checkpoints, explicit scope approval, separate required validation
records, optional latest-exchange continuity, and bounded OpenCode/OpenClaw
handoff prompts. These records are user-maintained; they are not independent
proof that code was edited, checks were run, or a build was deployed.
See the README's personal-project workflow for commands and limits.

Keep all inference local by default. Access to Hugging Face for discovery
and downloads is separate from sending prompts to a hosted inference API.
Do not silently enable hosted fallback.

## What existing tools already provide

### OpenClaw

The current official documentation describes a llama.cpp provider plugin with
managed installation, hardware-aware recipes, verified downloads, loopback
serving, and a real tool-use activation check. Its catalog currently includes
Qwen3.5 4B/9B, Gemma 4 12B, Muse Glimmer 30B, and Qwen3.8 27B.
Those are product selection defaults, not independent quality rankings.

Check the installed OpenClaw version before relying on these features.
Do not rebuild this entire installation stack simply because earlier versions
were cumbersome. Use it as the first baseline for an OpenClaw-only workflow.

OpenClaw's Ollama integration specifically requires its native API, not an
arbitrary OpenAI-compatible endpoint. A generic integration exporter must not
apply one configuration to every backend.

### LM Studio

LM Studio already supports scripted model management, headless operation,
OpenAI-compatible APIs, and tool use. Its GUI is not itself evidence that
inference is slow. An unsuitable model, excessive context, cold loading,
memory pressure, or reasoning output can dominate latency.

Preserve the existing installation and benchmark it before replacing it.

### MLX LM and llama.cpp

MLX LM is an Apple Silicon inference option with Hugging Face integration,
streaming, prompt caching, and configurable prefill and cache controls.
llama.cpp supports Apple Silicon through Metal and supports GGUF quantization.

Neither is a universal speed winner. Model architecture, quantization,
runtime version, memory bandwidth, and prompt shape matter. An Intel Mac
must not be offered MLX as if it were Apple Silicon.

## Model selection: fit is not quality or speed

Do not call the most-downloaded model the "best."

Use two stages:

1. Eligibility: runtime and architecture support, license, artifact format,
   available memory, disk space, chat template, and tool parser.
2. Selection: measured task success, latency, context needs, and user preference.

Examples verified on Hugging Face during this research:

| Candidate | Appropriate experiment | Important limitation |
| --- | --- | --- |
| `mlx-community/Qwen3-4B-Instruct-2507-4bit` | Small text-only baseline for short explanations and guided queries | Not a claim of strong autonomous coding performance; weights alone are about 2.26 GB decimal |
| `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` | Established coding-focused baseline | Older model; compare against newer compatible candidates rather than treating downloads as quality |
| `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` | Coding candidate on a sufficiently large-memory Mac | All resident expert weights count toward memory, not just active parameters |
| `mlx-community/Qwen3-Coder-Next-4bit` | Larger-memory coding experiment | Upstream model has 80B total / 3B active parameters; not a 3B-sized download |
| `mlx-community/Qwen3.8-27B-4bit` | Separate multimodal-runtime experiment | Its card specifies `mlx-vlm`; do not assume `mlx-lm` compatibility from the `mlx` tag |

These are candidates, not an exhaustive current leaderboard or guaranteed
recommendations for an unknown Mac. The CLI should refresh discovery at
runtime and show the metadata timestamp.

### Memory policy

Calculate or conservatively estimate:

`weights + context/KV or recurrent state + runtime scratch + OS/apps reserve`

For conventional attention, KV memory depends on layers, KV heads, head
dimension, cache precision, context length, and concurrent sequences.
Hybrid/recurrent architectures need their own accounting. If required
metadata is missing, label fit uncertain rather than manufacturing precision.

Read installed memory, architecture, macOS version, current memory pressure,
and free space on the actual cache volume. Detect translated processes where
possible. Chip family alone is not enough.

Start with one loaded model and a modest context budget, then measure.
Do not modify kernel wired-memory limits, enable swap tricks, or promise
tokens/second from RAM size. Show separate memory-fit and performance status.

### Download policy

Use Hugging Face's maintained download/cache library rather than inventing a
downloader. Resolve a repository revision before installation and record it.
Download only the selected artifacts; do not fetch every GGUF quantization.
Account for staging, cache reuse, and free space.

Show repository, revision, license, artifact size, and runtime before consent.
Never enable `trust_remote_code` by default. Treat model cards as untrusted
metadata, not executable installation instructions.

## Agent and token efficiency

### Prefer one bounded worker

LangChain's current multi-agent documentation explicitly says a single agent
with appropriate tools and prompts can often solve a complex task.
Anthropic likewise recommends the simplest effective solution.

Default to one model call for questions and one bounded coding-agent session
for implementation. Add multiple workers only for genuinely independent
work or an evaluation that demonstrates a benefit.

On one memory-bandwidth-limited laptop, parallel agents can contend for the
same resources while multiplying context. Parallelism is not free throughput.

### Graphs are control flow, not token compression

A useful initial flow is:

`brief -> select relevant context -> execute -> deterministic check -> finish`

Route explicitly by command or task type before spending tokens on an LLM
router. Do not require planning, critique, and synthesis calls for every
small question.

Use LangGraph when checkpointing, durable resume, branching, human approval,
or bounded retries actually become requirements. A graph does not itself
reduce tokens; what each node receives and how often it runs determine cost.

### Plain English helps clarity, not magically cheaper tokenization

Use a guided brief with:

```text
Task: Fix the login redirect after token expiry.
Context: Read auth/session.py and tests/test_session.py.
Constraints: Keep the public API. Add no dependencies.
Done when: Existing tests and a new expiry regression test pass.
Output: Patch and a short explanation.
```

Use short, direct sentences and concrete acceptance criteria. Preserve exact
identifiers, errors, and meaningful code. Do not strip essential context just
to minimize prompt length.

There is no established requirement here to adopt a formal standardized
English dialect. The verified Karpathy source supports context engineering:
provide the right information, split control flows appropriately, choose
capable models, and verify outputs. It does not establish a universal
token-saving prompt grammar or a requirement to use LangGraph.

### Reduce repeated context

Keep stable instructions small. Load task-specific skills and tool schemas
only when needed. Use explicit file/range selection rather than entire
repositories or full command logs. Preserve a reusable prefix when supported,
and keep changing content after that prefix.

For long work, retain a small structured state: goal, decisions, changed
files, current evidence, and next action. Do not replay an unlimited transcript
or repeatedly summarize everything. Summarization itself costs tokens and
can lose important facts.

Initially reject over-budget requests with an explanation and guidance.
Do not silently truncate user requirements or claim a character heuristic is
an exact tokenizer count. Use the actual model tokenizer when available and
include chat-template and output overhead in admission checks.

## Measurements required before recommending a default

Use both short questions and realistic coding/tool tasks, not just "hello."
Measure:

| Measurement | Why it matters |
| --- | --- |
| Cold-load time | Separates model loading from inference |
| Time to first visible token | Exposes prefill and hidden reasoning delays |
| Prompt processing throughput, when exposed | Captures repeated-context cost |
| Decode throughput, when exposed | Captures generation speed |
| End-to-end latency | Measures the user's actual wait |
| Input, output, reasoning, and cached tokens when reported | Avoids conflating different kinds of work |
| Peak memory / memory pressure when available | Identifies oversizing and contention |
| Valid tool call and verified task outcome | Distinguishes fluent chat from useful engineering |

Run comparable prompt/output budgets and repeated warm trials. Report backend
counters as such; do not label output tokens divided by total wall time as
pure decode throughput. Show missing counters as unavailable.

Remote cached-input totals are not equivalent to fresh local prefill work,
API billing, or useful progress. Do not import personal session histories
into the project or upload them for benchmarking.

## Proposed first release

A Python CLI is a practical fit for Hugging Face and MLX, with an isolated
environment and an optional Apple-Silicon-only MLX dependency.

Suggested command surfaces:

- `doctor`: hardware, cache volume, installed runtimes, actionable problems.
- `models`: discover candidates and explain eligibility / uncertainty.
- `setup`: review, pin, download, and configure one selected model.
- `ask`: bounded one-shot query with explicit context and measured usage.
- `guide`: compile a task brief locally, without another model call.
- `bench`: compare repeatable workloads against a selected local endpoint.
- `export`: emit OpenCode/OpenClaw configuration for review without
  overwriting existing files.

Reuse a configured LM Studio or other local endpoint for a low-friction
starting path. Offer managed MLX separately on Apple Silicon.
Do not run generated shell commands or modify source files autonomously in
the first version; let the existing coding agents own execution permissions.

Bind managed servers to loopback. Avoid prompt logging by default. Store
only operational metadata and opt-in benchmark results. Do not enable
telemetry or cloud escalation implicitly.

The first release can validate portable discovery, configuration, budgeting,
and HTTP behavior on Windows. Actual Mac model loading, Metal acceleration,
memory behavior, and speed must be verified on a Mac before calling them
tested or recommending a performance winner.

## Implemented quality and model lifecycle policy

The companion now fingerprints Git state on task access, offers explicit
path-based context suggestions, and can execute user-approved validation
commands bound to repository snapshots and protected graders. Compact task
state and explicit context remain preferable to replaying entire transcripts.
This is not semantic retrieval, autonomous source editing, or a sandbox.

Following agent-evaluation guidance, reliability is measured through real
protected checks, not self-reported model confidence. The selected target is
all required checks for acceptance and at least 95% first-submitted-candidate
success over at least 100 independent user-designated cases, with a reported
confidence interval. External-agent retries/model selection are not observed:
the metric must not be relabeled model-generation pass@1. Good graders,
representative tasks, clean starting baselines, and human review are essential.
See Anthropic's evaluation discussion:
https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents

Sampling/prompt/context improvements do not train weights. MLX LM has actual
LoRA/fine-tuning interfaces, but these require suitable licensed data, separate
evaluation, compute, and explicit authorization. This project intentionally
does not train or claim that parameter settings make small local models equal
to stronger hosted models:
https://github.com/ml-explore/mlx-lm#fine-tuning

Model discovery includes current popular and recent conversions rather than a
frozen default. Hardware-compatible, unmeasured recommendations are provisional;
recency/download counts are not coding-quality evidence. Automatic replacement
requires improved observed results on matching cases, graders, Git baselines,
runtime, budgets, and sampling settings. Candidates are pinned, downloaded,
offline smoke-tested, and only then activated. Failed verification preserves
the active configuration. Cleanup is restricted to recorded owned revisions in
the dedicated cache, using Hugging Face's reference-aware cache API:
https://huggingface.co/docs/huggingface_hub/guides/manage-cache

Windows validation does not establish actual M5 model quality, loadability,
Metal behavior, or speed. The CLI enforces Mac load/generation verification at
installation; real coding trials and agent integration still need that hardware.

## Primary sources

Accessed 2026-10-30; online documentation and catalogs change.

1. OpenClaw local models:
   https://docs.openclaw.ai/gateway/local-models
2. OpenClaw managed llama.cpp provider, hardware-aware selection and tool check:
   https://docs.openclaw.ai/plugins/llama-cpp
3. OpenClaw native Ollama API warning:
   https://docs.openclaw.ai/providers/ollama
4. LM Studio APIs, headless operation, model management, and CLI:
   https://lmstudio.ai/docs/developer
5. MLX LM, supported interfaces, prompt caching and long-context controls:
   https://github.com/ml-explore/mlx-lm
6. llama.cpp, supported hardware and quantization:
   https://github.com/ggml-org/llama.cpp
7. Hugging Face versioned and filtered downloads:
   https://huggingface.co/docs/huggingface_hub/guides/download
8. OpenCode custom/local provider configuration:
   https://opencode.ai/docs/providers/
9. LangChain multi-agent tradeoffs and LangGraph custom workflows:
   https://docs.langchain.com/oss/python/langchain/multi-agent
10. Anthropic, effective context engineering:
    https://www.anthropic.com/research/effective-context-engineering-for-ai-agents
11. Anthropic, building effective agents (foundational article; its current
    page notes that tooling has changed since December 2024):
    https://www.anthropic.com/research/building-effective-agents
12. Karpathy's direct context-engineering statement:
    https://x.com/karpathy/status/1937902205765607626
13. Small text-only MLX candidate:
    https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit
14. Coding MLX candidates:
    https://huggingface.co/mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
    https://huggingface.co/mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
15. Qwen3-Coder-Next upstream architecture and total/active parameter counts:
    https://huggingface.co/Qwen/Qwen3-Coder-Next
16. Current multimodal MLX model's runtime instructions:
    https://huggingface.co/mlx-community/Qwen3.8-27B-4bit
