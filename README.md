# Dedicated Harness

[![CI](https://github.com/johnkiprop/dedicatedharness/actions/workflows/ci.yml/badge.svg)](https://github.com/johnkiprop/dedicatedharness/actions/workflows/ci.yml)

A local-first model manager and coding supervisor for Mac engineers using OpenCode, OpenClaw, and
LM Studio. Discover and budget Hugging Face models, prepare isolated MLX,
write focused prompts, retain compact project checkpoints, measure latency, and
export reviewed agent configuration and task handoffs. `dh run` supervises an
existing OpenCode engine with scoped file tools, bounded inference, independent
checks, and limited repairs. It does not rebuild OpenCode or launch a swarm.
OpenClaw integration remains provider exports and handoffs, not a supervised adapter.

Start with the [user guide](docs/USER_GUIDE.md), also available as a
[Word document](docs/USER_GUIDE.docx). It covers Mac setup, daily project work,
quality checks, supervised coding, saved-model rollback, and troubleshooting.

See [RESEARCH.md](RESEARCH.md) for the dated research, model caveats, architecture
recommendations, and primary sources. Start with one bounded worker, not a swarm.

## Install on your Mac

Requires Python 3.11+; managed MLX additionally needs native arm64 Python on
Apple Silicon and macOS 14+. Existing local endpoint connections are portable.
After the source has been pushed to GitHub, run these commands in Terminal
on the Mac. Authenticate with GitHub first if the repository is private.

```sh
git clone https://github.com/johnkiprop/dedicatedharness.git
cd dedicatedharness
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
dh doctor
```

Check `python3 --version` before installation; use a native arm64 Python 3.11+
interpreter if the system Python is older or runs through Rosetta. Do not copy
the Windows `.venv` to the Mac; create a fresh environment as shown above.

These are **Mac shell commands**. All following commands assume the environment
is activated. Model downloads and runtime installation require internet;
queries go only to a numeric loopback address. No cloud fallback is implemented.

Configuration, the Hugging Face cache, and the optional MLX runtime live under
`~/.dedicatedharness`. Use `dh --home PATH ...` or `DEDICATEDHARNESS_HOME` to
choose another location. Project/task state is kept in a separate SQLite
database per resolved checkout root under that location, not in your source
repository. No other application's configuration is changed.

## Your M5 / 24 GB starting path

Discover compatible coding/reasoning candidates and measure real work before
assuming a larger or newer model is better. Setup now discovers a current coding
candidate by default; unmeasured recommendations are **provisional, not claims
of global-best quality or speed**:

```sh
dh models --recommend coding
dh models --recommend reasoning
dh setup --purpose coding --install-runtime
dh serve
```

Run `dh serve` only after setup succeeds. Setup displays the license,
immutable revision, download size, fit reasoning,
and isolated-runtime action before asking permission. It downloads the pinned
revision and verifies offline loading plus short generation before saving the
active configuration. `--yes` explicitly
approves those actions for automation. It replaces only Dedicated Harness's
connection configuration. The previous configuration is preserved on failure.
An explicitly requested runtime installation/update can still modify that
isolated runtime even when a later setup step fails.

In another terminal with the environment activated:

```sh
dh ask "Explain when a single agent is better than several agents."
dh bench "Explain a race condition with one example." --runs 3
```

`serve` remains in the foreground, verifies actual generation, and terminates
its own child on exit. It disables Hub downloads at inference time, uses one
decode/prompt slot and one prompt-cache entry, and binds only to `127.0.0.1`.
It does not install a login service or detach a daemon.

Choose another model explicitly with `dh setup --repo OWNER/MODEL`; add
`--revision COMMIT` to install a specific revision. A conservative explicit
starting baseline is `mlx-community/Qwen3-4B-Instruct-2507-4bit`.
Live metadata screening during development found the 4B baseline and
`mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` plausible under a hypothetical
24 GiB / 4,096-token budget. The latter is a coding baseline, not a proven
quality winner. `Qwen3-Coder-30B-A3B-Instruct-4bit` exceeded that conservative
budget despite only 3B active parameters. Estimated available memory on your Mac
must still be checked before either smaller model is installed.

Only a conservative initial text-only architecture set is eligible:
`qwen2`, `qwen3`, `qwen3_moe`, `llama`, and `mistral`, with MLX quantization,
safetensors weights, a known context limit, and a chat template.
Other models are reported as unsupported/uncertain rather than guessed.
Multimodal and hybrid-state models need separate runtime/memory support.

`dh models --memory-gb 24` provides a hypothetical-memory preview on Windows
or another machine. It cannot authorize installation or prove Mac performance.

### Model updates without blind replacement

Stop `dh serve` before setting up, staging, or replacing a managed model.
The CLI refuses a live managed endpoint or occupied setup port; it does not
kill existing servers. Close LM Studio/other memory-heavy applications if the
available-memory check fails.

```sh
dh upgrade --purpose coding
dh upgrade --repo OWNER/MODEL --stage
dh upgrade --purpose coding --yes --keep-old
```

Discovery checks popular and recently modified `mlx-community` conversions,
then filters architectures, weights, full-context memory, disk, license, and
coding tool-template eligibility. Recommendations prefer measured project
outcomes; otherwise purpose-name affinity, modification time, and downloads
are only heuristics. This is a bounded search, not the whole Hugging Face catalog.
Unknown/new architectures remain unsupported until their runtime and memory
behavior are explicitly supported.

`upgrade` retains the active model unless the replacement has a higher observed
success rate, meets the quality target below, and was evaluated on identical
cases, protected graders, starting fingerprints, runtime, context/output budgets,
and sampling settings. These are observed workflow results, not statistically
proven intrinsic superiority. There is no background watcher: discovery runs
when you invoke these commands.

Without sufficient evidence, `--stage` downloads and smoke-tests a candidate
without changing the active connection. The printed pinned `setup` command
lets you explicitly switch for evaluation; this is a user-approved override,
not automatic promotion. Keep the same budgets/settings, restore each case's
original repository baseline, and use the same project/`--home` to compare
profiles. Restore the original pinned setup if the candidate is worse.

After an eligible replacement loads and generates successfully, configuration
switches atomically; only then can cleanup delete the previous manifest-owned
revision from `~/.dedicatedharness/hub`. Hugging Face's revision/refcount-aware
cleanup preserves referenced blobs. Unowned revisions, revisions shared with
another cached repository, and LM Studio/shared caches are not removed.
`--keep-old` disables cleanup. Exports must be reviewed again after model changes.

## Keep LM Studio if it works

Enable LM Studio's OpenAI-compatible local server and load a model, then:

```sh
dh connect --url http://127.0.0.1:1234/v1 --context 4096 --output 512
dh ask "Explain the tradeoff between latency and answer quality."
```

If several model IDs are listed, add `--model EXACT_ID`. Set LM Studio's actual
context capacity to at least the declared budget. `connect` verifies model
discovery, not generation speed or tool quality. Reconnecting clears any
previous tool-verification result.

For managed MLX, admission counts the downloaded model's real chat-template
tokens in the isolated runtime, without remote code. For external servers,
admission uses a deliberately conservative UTF-8/JSON byte estimate plus
template allowance. **That estimate is not a guaranteed bound or an exact
token count for an unknown model.** The server must also enforce its limits.

## Guided prompts and explicit context

```sh
dh guide
dh guide --task "Fix token expiry handling" --context "auth.py and its tests" \
  --done "Expiry regression test and existing tests pass" --output "Patch and short explanation"
dh guide --task "Explain the auth flow" --done "Describe entry and exit conditions" | dh ask
dh ask "Review this expiry condition" --root . --file 'auth.py::20-70' --dry-run
dh ask "Review this expiry condition" --root . --file 'auth.py::20-70'
```

`guide` is a deterministic brief builder, not another LLM call. With no
arguments it asks for the task, context, constraints, completion condition,
and desired output. File paths and resolved symlinks must stay inside `--root`.
Use `::START-END` to distinguish line ranges from Windows drive letters.
Individual files and combined selected context are limited to 256 KiB.
Large files must be split first; line selection does not bypass the file-size
limit. Only explicitly named UTF-8 files are read; secrets are not automatically
detected or redacted, so select context carefully.

By default, `ask` is one-shot: no transcript replay, no model-written shell
commands, no automatic file edits, no transcript storage, and no hidden summarization.
Over-budget requests are rejected without sending them. An output-limit finish
is reported as incomplete, not success.

## Continue work on your personal project

With `dh` available in your shell, run these commands from your application's
repository root. These commands do not require a model until `ask` is used.

```sh
dh project init --goal "Improve the app's reliability" \
  --constraints "Keep data local. Explain changes before implementation. Validate emulator and phone separately."
dh task new "Investigate battery heating" \
  --done "The cause is explained, the approved fix is implemented, and emulator and phone checks pass" \
  --check emulator --check phone
dh task handoff opencode --out investigate.prompt.txt
```

Review the handoff and pass its contents to your existing OpenCode session in
that repository, with the local provider configured. Use `openclaw` as the
handoff target if that is your execution agent instead. Handoffs are plain
text: they do not launch an agent, select a provider, run shell commands,
change permissions, or grant implementation approval.

Tasks start in **investigate** mode. After investigation, record a compact
finding and proposed plan; the following is an illustrative finding, not a
claim about your app:

```sh
dh task checkpoint --finding "Redundant polling appears to repeat work." \
  --decision "Reduce redundant wakeups while preserving the public API."
dh task approve --scope "Reduce redundant wakeups only; do not change unrelated features."
dh task handoff opencode --out implement.prompt.txt
```

Pass the reviewed implementation handoff to your execution agent. **After it
actually makes the changes**, record what changed and move to validation:

```sh
dh task checkpoint --change "Reduced redundant polling in the approved implementation." \
  --next "Run emulator regression checks; phone validation remains pending."
dh task ready
dh task handoff opencode --out validate.prompt.txt
```

Run the real checks with your existing agent/tooling before recording their
results. Supply actual command/build/device evidence rather than copying an
example as if it were a test result:

```sh
dh task validate emulator --status passed --evidence "Actual emulator command, build, and observed passing result"
dh task validate phone --status blocked --evidence "Phone validation has not been performed yet"
dh task show
```

`dh task finish` refuses to close this task while phone validation is pending,
blocked, or failed. Once you actually test the implementation on the phone,
record its evidence with `--status passed`, then finish. These are
**user-reported records in the default/manual workflow**. For executable,
repository-bound evidence, enable verified quality mode below. Neither mode
can establish real-device behavior from emulator output.

### Natural follow-ups without an unlimited transcript

Use the active task's checkpoint to anchor questions:

```sh
dh ask "What remains before we can call this finished?" --project
dh ask "Explain the proposed tradeoff." --project --remember
dh ask "Why not the other approach?" --project --remember
```

`--project` includes current project constraints, task state, approval scope,
reported checks, and any previously opted-in latest exchange. It also reads
only the task's explicitly recorded file selections plus any `--file` arguments.
It does not search the repository or acquire coding tools. Add file references
with `dh task checkpoint --file 'relative-file::START-END'`; references must
exist inside the project root. `--clear-files` removes recorded selections.
Source files are reread, not stored as a repository snapshot.

`--remember` is **opt-in local text storage**: only the latest successful
question/answer pair is retained for that task. It does not retain the injected
file context, replay the whole conversation, interpret "proceed" as approval,
or promote model claims to findings or passing tests. Questions are limited
to 4,096 UTF-8 bytes and answers to 8,192; overlong answers are displayed but
cannot be remembered and produce an explicit error. Concurrent task changes
also prevent saving a stale answer.

Use `dh task checkpoint` to deliberately replace selected compact sections
(`--finding`, `--decision`, `--change`, `--next`, `--goal`, `--done`, `--file`).
Unspecified sections are preserved. A checkpoint clears the optional latest
exchange; prior checkpoint states remain in the audit, not in every prompt.
This is manual compaction with no summarizer model call. `dh task forget`
clears the optional latest exchange without replacing the checkpoint.
Forgetting is logical deletion, not a secure-erasure guarantee.

The workflow is:

`investigate -> explicit approval -> implement -> validate -> done`

Changing the task goal, completion condition, decision, or file scope revokes
approval and resets required checks. Changing project goals/constraints with
`dh project update` does the same for all open tasks. Every `--change` records
a new implementation revision and invalidates earlier check results, even
when the summary text is unchanged. Git repositories are automatically
fingerprinted when a task is selected/read: HEAD, branch, index, tracked and
non-ignored untracked content are included. Working-tree drift invalidates
checks; branch/HEAD drift revokes approval. Drift on a selected closed task
reopens it conservatively. Otherwise, closed implementation checkpoints cannot
be edited. Ignored files and submodule contents are not verified; submodules
are rejected. There is no continuous filesystem watcher or atomic filesystem
snapshot. A phase/approval record is a workflow guard, not an external-agent sandbox.

```sh
dh project show
dh project list
dh task list
dh task use TASK_ID
dh task history --limit 10
dh task handoff openclaw
```

Task commands accept `--root PATH`; commands targeting an existing task also
accept `--id TASK_ID`. `ask`/`bench` accept `--project --task-id TASK_ID`.
The active task and state survive CLI restarts for the same checkout and
`--home`; another checkout/root has separate state. Moving or cloning a
repository does not automatically transfer its private checkpoints.

Project text is capped at 1,000 UTF-8 bytes per summary section, file selections
at twelve, and required check names at eight. Handoffs default to an 8,192-byte
budget and refuse oversized output without silent truncation; use a compact
checkpoint or explicitly raise `--max-bytes` (maximum 32,768). Queries still
pass the existing tokenizer/context admission checks. No transcript or project
state is uploaded by these state commands. Review exported handoffs before
sharing: they include your project descriptions, evidence, and optional last
exchange. Your execution agent's provider/privacy settings remain your responsibility.

## Executed quality gates and a measured 95% target

Initialize your project first. Git is required for verified mode; keep the
harness home outside the repository. Opting in invalidates old manual evidence
and requires executed results for **every** required check:

```sh
dh quality init
dh quality command tests --argv-json '["python","-m","unittest","discover","-s","tests"]' \
  --protect tests --timeout 180
dh quality context "Fix token expiry and add regression coverage"
```

Register your actual test/build/lint commands separately, using the same names
as task checks. `--argv-json` approves a fixed argument array, resolved executable,
timeout, and optional trusted grader baseline. Commands run directly, without
a shell; use executable/script paths appropriate to your project. This does
not make the commands safe: tests, package scripts, plugins, and imported source
can execute arbitrary code. Review them and use an isolated environment for
untrusted generated changes. Never register a model's unreviewed shell string.

Context suggestions use file-path keywords and Git changes, not embeddings or
semantic understanding. Review and explicitly select relevant source AND tests
with `dh task checkpoint --file ...`; files are not automatically injected.
Handoffs require scoped fixes, regression tests, preserved graders, and actual
evidence rather than model-written success claims.

After the approved implementation and `dh task checkpoint --change ...`:

```sh
dh task ready
dh quality run tests
dh task finish
```

Run all registered required checks before finishing. A nonzero exit, timeout,
output-limit cutoff, changed repository, modified grader, or stale task/command
cannot count as a pass. Output is polled against an 8 MiB cutoff, which can
overshoot between polls; only the last 8 KiB are retained in the private database.
Logs may contain sensitive project output. macOS cleans up the owned process
group; Windows timeout cleanup targets the owned PID tree but cannot reliably
reclaim descendants that already outlived their parent. This runner is not an
OS sandbox or a hard resource quota.

In verified mode, manually recording `--status passed` is refused. Phone checks
therefore need an approved command that actually verifies the required device
behavior; merely listing a connected phone is not functional validation.
Protected graders are hashed before execution. Intentional changes to existing
graders require review and re-registering the command, which invalidates previous
evidence. Adding new regression tests is allowed, but quality cases below retain
their original command/grader digests.

### Measure reliability, do not promise it

```sh
dh quality profile --temperature 0 --top-p 1 --seed 0
dh quality case expiry-regression --goal "Fix expiry handling" \
  --done "Independent expiry grader and regressions pass" --check tests
dh quality trial expiry-regression
```

Register independent, representative, user-designated held-out cases before
solving them. Start each trial from the original unsolved Git baseline, then
use the normal investigate/approve/implement/ready/run workflow with your
execution agent. Each case requires protected graders. After real trials:

```sh
dh quality report
```

The target is **all required checks pass for an accepted change**, plus
**at least 95% first-submitted-candidate success across at least 100 distinct
cases** for a declared model/prompt/sampling profile. The first failure is
permanent for that trial; repairs/retries do not replace the first outcome.
Pending first trials remain in the denominator and prevent target qualification.
The report includes a Wilson 95% interval: observing 95/100 passes does not mean
the true success rate is at least 95% with 95% confidence.

This is **not model-generation pass@1**. Ordinary external OpenCode/OpenClaw
sessions can retry or override settings without observation. `dh run` instead
observes its owned engine sessions and routes every model request through its
bounded local gateway. These execution profiles are accounted separately.
Neither mode certifies held-out independence, grader completeness,
training-data contamination, or industry-model equivalence. Provider exports do
not force an ordinary external agent's settings. Human review remains necessary.

`quality profile` changes inference settings used by `ask`, `bench`, and the
tool probe, and clears tool verification. It does not update model weights,
automatically search parameters, learn from private histories, or guarantee a
reproducible result. No fine-tuning, training-data collection, or automatic
weight training is implemented.

## Supervised coding on your Mac

Install **OpenCode 1.x** separately if needed; this adapter is validated against
1.19.4, not OpenCode 2. Keep the selected MLX or LM Studio server running.
For managed coding experiments, consider an 8,192-token context and 1,024-token
output budget instead of the smaller chat defaults; setup rechecks hardware fit.
External-server admission is a conservative byte estimate and may need a larger
declared/server context. Larger contexts cost memory and prefill time.

From an initialized Git checkout root, enable verified quality, register trusted
commands with `--protect`, and approve a task using the workflow above. Then use
this **instead of** the external implementation handoff and manual bookkeeping.
Replace these example paths with your approved source and a new regression test:

```sh
dh check-tools
dh run --allow-edit src/auth.py --allow-edit tests/test_auth_expiry_new.py
dh runs
```

The CLI starts its own authenticated loopback OpenCode server with separate
home/config/data/cache directories. Project/global OpenCode configuration,
plugins, external skills, MCP, LSP/formatters, sharing, and hosted providers are
disabled; resolved configuration and provider identity are checked before a
prompt. File-tool arguments are checked again by the harness. Shell commands,
web tools, subagents, edits outside the explicit paths, and edits to protected
graders are refused. This is **runtime policy, not an OS sandbox**: approved
checks execute project code, and concurrent writers can still cause drift.

One worker uses just-in-time source reads and search, not a repository dump.
A local inference gateway pins the model and sampling, serializes generation,
checks the actual messages/tool schemas against the context budget, visibly
compacts long/old tool results, and bounds aggregate admission. It never
silently removes the task or sends an over-budget prompt. Defaults are six
agent steps per turn, at most one repair, sixteen model requests, 600 seconds,
and an aggregate budget of 32,768 units for MLX or 131,072 for external servers.
This adds admitted input units and reserved output tokens. External input units
are estimated bytes, **not measured tokens**, so budgets are not directly
comparable between backends. Override with `--budget`. Cleanup can add time
after a deadline; an earlier budget or no-progress failure stops repairs.

Completion requires an observed terminal/idle engine session, Git-observed
changes, unchanged protected graders, and every required independent check
passing on the current fingerprint. The engine is stopped before final
acceptance. Text saying "done" is not evidence. Unregistered device checks stay
pending; emulator success cannot fabricate phone validation. Failed edits are
retained for review, never automatically reverted.

```sh
dh runs --id RUN_ID
dh run --resume RUN_ID
```

Resume uses compact task/check evidence and a fresh engine session. Repeat
the original nondefault scope/limit flags; model, engine version, limits,
approval, graders, task revision and Git fingerprint must still match. Consumed
budgets are not reset, and an exhausted run cannot be resumed. Model-changing
commands are locked out during a run.

Default persistence is compact operational metadata and bounded check-output
tails, which may contain project data. Raw OpenCode state is deleted after the
owned engine stops; `--keep-trace` explicitly retains source/prompts/transcripts
in the private harness home. Existing LM Studio/server logging is separate.
Supervised evaluation trials include actual engine version, harness checksum,
and limits; they are not pooled with declared external-agent profiles.

### Save, evaluate, and restore model profiles

```sh
dh profile save coding-stable
# Stop dh serve before staging or switching managed models.
dh upgrade --purpose coding --stage --profile coding-candidate
dh profile use coding-candidate
dh serve
# In another terminal: dh check-tools, then evaluate the candidate.
# Stop dh serve before restoring the old model:
dh profile use coding-stable
dh profile list
```

Named profiles retain pinned snapshots and sampling/context/runtime details.
Activation rechecks current hardware, owned cache contents, runtime version,
and offline generation, then clears tool verification. Cached restore does not
need Hub access or a second full download's disk budget. External profiles
verify that their exact model ID is exposed; they cannot attest which weights
an external application actually serves. Existing names are not overwritten.
Upgrade cleanup preserves every revision referenced by a saved profile.
An explicit candidate switch is permission to evaluate, **not evidence of superiority**.

## Verify tools, then export

```sh
dh check-tools
dh export opencode --out opencode.local.fragment.json
dh export openclaw --out openclaw.local.fragment.json
```

The check requires a structured request for a harmless in-memory tool and an
exact result roundtrip. Raw JSON in assistant text does not pass. It never
runs a filesystem or shell tool. Exports are refused until it passes.
A passing probe does not establish reliability for arbitrary coding tasks;
repeat it after changing model weights, server/runtime, or chat template.

Exports are provider **fragments**. Review and merge them into your existing
configuration; they neither select a default nor add a hosted fallback.
Output files are never overwritten. Both exports use the generic
OpenAI-compatible transport, not OpenClaw's native Ollama provider.
Configure LM Studio as local-only too; this CLI cannot prevent another process
from proxying a loopback request to the cloud.

MLX's server provides only basic security checks. Loopback is not
authentication: other local processes can reach it. Exported context/output
limits are client declarations, not a universal server-enforced admission
firewall; third-party agents bypass `dh ask`'s budget checks. Keep their tools
and permissions narrow.

## Read measurements honestly

`ask` streams text on stdout and prints admission/metrics on stderr.
`bench` repeats the same scoped prompt and reports elapsed time, time to first
visible token, completion status, and backend token counters when available.

```sh
dh bench "Explain this function" --root . --file 'example.py::1-30' \
  --runs 3 --save benchmark.json
```

Benchmarks are not quality scores. The first request may include loading;
it is not labeled a controlled cold-load measurement. Repeated prompts can
benefit from backend caching. Reasoning can delay the first visible token.
Pure prefill/decode speed, peak memory, and cold-load timing are not measured
by the portable OpenAI-compatible API and remain unavailable. Elapsed wall
time is never mislabeled as decode tokens/second.

Saved reports are opt-in, metadata-only, and never overwrite existing files.
They omit prompt and response text but include the model ID, local URL, and
backend usage fields. Requests use bounded socket timeouts and check elapsed
time between stream events; this is not a hard real-time cancellation guarantee.

## Fit policy and limits

The initial conventional-attention budget uses:

- Actual selected artifact sizes and full resident weights, including MoE experts.
- Full requested context with fp16/bf16 K/V for one sequence, independent of weight bits.
- Scratch reserve of at least 2 GiB and OS/apps reserve of at least 6 GiB / 25% RAM.
- Estimated available Mac memory from free + speculative + inactive pages.
  Inactive pages may need reclamation; purgeable/file-backed totals overlap
  other categories and are not added again. Active/wired/compressed pages
  and swap are excluded.
- Full download size plus 2 GiB disk reserve, even when some files are cached.

These are screening estimates, not measured peak usage. Close other memory-heavy
apps when current availability blocks setup. No wired-memory kernel settings
are modified. Downloads use Hugging Face's versioned cache and selected
non-code artifacts. Remote-code trust is never enabled.

Ordinary Hugging Face discovery is bounded (default eight candidates, maximum thirty),
currently scoped to `mlx-community`; explicit `--repo` supports other publishers.
Purpose recommendations make several bounded popular/recent searches.
Metadata/config downloads are not model-weight downloads. Discovery failures
are surfaced rather than silently replaced with a stale recommendation.

This first version intentionally has no LangGraph dependency, automatic swarm,
cloud inference, semantic repository indexing, or model-generated command executor.
Only explicitly registered validation commands are executed by the supervisor;
the model cannot invent shell commands. Ordinary exported-agent sessions retain
their own permissions; the owned `dh run` engine uses the restricted policy above.

## Development

GitHub Actions runs on pushes to `main`, pull requests, and manual dispatch.
CI checks Python 3.11 on Linux/macOS/Windows and Python 3.13 on Linux, runs the
portable regression suite, checks dependencies/compilation, and exercises the
installed CLI. A wheel is built only after all test jobs pass and retained as
an Actions artifact for seven days; nothing is published to PyPI automatically.
Actions are pinned to immutable commits with read-only repository permissions.
Linux/macOS Python 3.11 jobs also install OpenCode 1.19.4 and exercise its real
file tools against a fake loopback model, including permission refusal and
project-configuration isolation. CI does not download weights or validate real
MLX/Metal performance.

From the project directory:

```sh
python -m unittest discover -s tests -v
python -m compileall -q src tests
python -m pip check
```

Portable tests include a real loopback fake server, streaming and tool
roundtrips, no-send budget rejection, pinned downloads, memory screening,
project/task persistence, approval and evidence invalidation, bounded handoffs,
opt-in follow-up continuity, configuration persistence, provider exports,
real temporary Git repositories, executable quality gates, first-submission
accounting, supervised repairs/no-progress/deadlines, immutable execution
provenance, process-wide model locks, offline saved-profile restoration, and
staged/failed/measured model-upgrade paths. Native OpenCode protocol/file-tool
integration has also been exercised on Windows with a fake local model.
Actual MLX/Metal execution, M5 latency, local-model coding quality, and OpenClaw
session execution remain unverified; fake-model integration does not prove them.
