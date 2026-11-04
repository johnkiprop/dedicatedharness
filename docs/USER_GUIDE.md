# Dedicated Harness User Guide

Version 0.1.0 | Updated 3 November 2026

This guide covers the current CLI. Commands use the Mac Terminal.
The Word edition, `USER_GUIDE.docx`, contains the same guide.
The remaining-work section distinguishes proposals from shipped features.
Implementation diagrams are in the README's
[design section](../README.md#design-and-implementation).

## 1. What the harness does

Dedicated Harness prepares local models and helps you control engineering work.
It screens Hugging Face models against your hardware, installs a pinned MLX
model, builds focused prompts, keeps compact project checkpoints, and records
real validation results when verified quality mode is enabled.

Use `dh run` to supervise an existing OpenCode 1.x engine, or continue using
OpenCode/OpenClaw manually with reviewed handoffs.
`dh ask` answers questions; it does not edit your project. Handoffs are reviewed
text prompts, not automatic agent launches. LM Studio can remain your model server.

Keep the workflow simple:

1. Select one model that fits.
2. Select the relevant source and tests.
3. Investigate and approve a small plan.
4. Run one bounded worker or use a reviewed external-agent handoff.
5. Accept only current, independently executed checks.

## 2. Install on your Mac

You need Git and Python 3.11 or newer. Managed MLX also requires Apple Silicon,
native arm64 Python, and macOS 14 or newer. Do not copy a Windows virtual
environment to the Mac.

```sh
mkdir -p "$HOME/Projects"
git clone https://github.com/johnkiprop/dedicatedharness.git \
  "$HOME/Projects/dedicatedharness"
cd "$HOME/Projects/dedicatedharness"
python3 --version
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
dh doctor
```

Authenticate with GitHub first if the repository is private. If your Python is
too old or runs through Rosetta, select a suitable native interpreter before
creating `.venv`. Follow the error from `doctor` rather than bypassing fit checks.

In each new terminal, activate the CLI environment:

```sh
. "$HOME/Projects/dedicatedharness/.venv/bin/activate"
```

The default state directory is `~/.dedicatedharness`. It contains configuration,
model cache, optional runtime, project databases, and validation log tails.
Keep this directory outside your application's repository.
An alternative home is selected before the command:

```sh
dh --home "$HOME/.dedicatedharness-alt" doctor
```

Different homes have separate configuration and project state.

## 3. Choose your model server

### Option A: Let the harness manage MLX

Start with a small model and a 4,096-token context for chat on a 24 GB Mac.
For supervised coding, an 8,192-token context and 1,024-token output budget
are a useful starting experiment, subject to the same hardware screening.
Recommendations change with live metadata and available memory.
An unmeasured recommendation is provisional, not a proven best model.

```sh
dh models --recommend coding --context 4096
dh models --recommend reasoning --context 4096
dh setup --purpose coding --install-runtime --context 4096 --output 512
dh serve
```

Review the model license, revision, size, and fit before accepting setup.
Setup downloads the pinned revision and verifies offline loading and short
generation before changing the active connection. The explicit baseline below
is another option, not a claim that it beats current releases:

```sh
dh setup --repo mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --install-runtime --context 4096 --output 512
```

Do not run both setup examples in sequence unnecessarily. Stop `dh serve`
before another setup, staging operation, or upgrade. Run `dh serve` only after
setup succeeds. It runs in the foreground;
use Control-C to stop it. The CLI will not kill an existing model server.

In a second activated terminal:

```sh
dh ask "Explain one benefit and one cost of a single coding agent."
dh bench "Explain a race condition with one example." --runs 3
```

Benchmarks report latency and available backend counters, not coding accuracy.
First-token latency can include reasoning or model loading.

### Option B: Keep LM Studio

Load a model and enable LM Studio's OpenAI-compatible local server:

```sh
dh connect --url http://127.0.0.1:1234/v1 --context 4096 --output 512
dh ask "Explain a small, testable approach to fixing a bug."
```

If several models are exposed, add `--model EXACT_ID` using an ID from the
server. Configure the server's real context limit to match your declared budget.
Do not run `dh serve` for an external connection: LM Studio manages that server.

These are alternative active configurations. A later `connect` or successful
`setup` replaces the harness connection. Model downloads need internet;
inference uses numeric loopback URLs with no hosted fallback. A separate local
application could still proxy requests elsewhere, so review its settings.

### Existing LM Studio downloads: what counts against your Mac

| Model state | Storage and memory effect |
| --- | --- |
| Downloaded but unloaded | Uses disk space on its storage volume, not model-weight RAM merely because it is stored. The LM Studio application can still use memory. |
| Loaded or running in LM Studio | Resident weights and runtime/context buffers can use unified memory, reduce available headroom, and cause larger managed candidates to be rejected. |
| Also installed by managed MLX | May occupy another copy or a different conversion in the harness's separate cache. Files are not automatically shared across the applications. |

The harness checks free space where its cache is stored. LM Studio downloads
on that volume, or another APFS volume sharing its storage pool, can reduce the
reading; an independent external disk normally does not consume that space.
On macOS, memory availability is estimated from free, speculative, and inactive
pages, not installed RAM alone.
Inactive pages may need reclamation. Unload other local models before managed
discovery/setup if you want to free their memory; deleting downloaded files is
not necessary just to unload a model.

Managed recommendations inspect Hugging Face candidates, not your LM Studio
library. They do not automatically select, import, or deduplicate its models.
Managed MLX expects supported MLX artifacts; a GGUF download is not an
interchangeable MLX snapshot. The default managed cache is
`~/.dedicatedharness/hub`, and automatic cleanup never deletes LM Studio's
downloads.

To reuse existing weights without a harness-managed download, choose Option B:
load a model in LM Studio, start its local server, and run the `dh connect`
example above. It connects to that server; it does not copy weights, install
MLX, load/unload models, or start/stop LM Studio. Stop any managed `dh serve`
before changing the active connection. If still at the setup approval prompt,
answer `n` to decline the managed download before taking this alternative.
You do not need both model servers running.

In setup output, `download_bytes` estimates selected artifact storage, whereas
`required_bytes` budgets model weights, full-context KV, scratch, and OS/apps
reserves. The latter is not the download size, an actual memory measurement,
or an allocation of that entire amount at once. Total RAM must cover this
complete budget; when current availability is known, it must cover weights +
KV + scratch. The OS/apps reserve is not charged against available memory a
second time. A plausible fit still needs real generation, tool checks, and
task evaluation.

## 4. Connect OpenCode or OpenClaw

With the chosen model server running:

```sh
mkdir -p "$HOME/.dedicatedharness/exports"
dh check-tools
dh export opencode \
  --out "$HOME/.dedicatedharness/exports/opencode.local.json"
dh export openclaw \
  --out "$HOME/.dedicatedharness/exports/openclaw.local.json"
```

Use only the export for your chosen agent. Review and merge the fragment into
that agent's existing configuration. Then explicitly select the local provider
and model in the agent. Exports do not change defaults or launch a session.
Use a new filename if an export already exists; output files are not overwritten.

The tool probe checks one harmless structured tool roundtrip. It is not proof
that the model can reliably complete arbitrary coding tasks. Repeat it and
review new exports after changing the model, runtime, template, or sampling.
Provider fragments do not force the external agent's sampling parameters.

## 5. Ask focused questions

Use short, direct English with one task and a checkable completion condition:

```sh
dh guide --task "Fix token expiry handling" \
  --context "src/auth.py and tests/test_auth.py" \
  --constraints "Keep the public API. Add no dependencies." \
  --done "Expiry regression and existing tests pass" \
  --output "Small patch and a short explanation"
```

`guide` builds text without a model call. Run `dh guide` alone for guided inputs.
It does not attach source files. For an actual scoped query, use existing files:

```sh
cd "$HOME/Projects/my-app"
dh ask "Explain this expiry condition." \
  --file 'src/auth.py::20-70' --dry-run
dh ask "Explain this expiry condition." \
  --file 'src/auth.py::20-70'
```

Replace example paths and line ranges with your project's actual files.
`--dry-run` checks admission without sending a generation request; managed MLX
may load its tokenizer to count tokens. Source files are limited to 256 KiB
each and in total, even when selecting line ranges. Selected paths must stay
inside the project root. Over-budget requests are rejected, not silently cut.

Choose a smaller scope instead of attaching the whole repository. Preserve exact
identifiers, errors, and meaningful code. Secrets are not automatically redacted.
Managed MLX uses real chat-template token counts; external-server counting is
an explicitly approximate byte estimate.

## 6. Work on one project from investigation to completion

This example assumes a Git-backed application with `src/auth.py` and a `tests`
directory. Substitute your actual files, checks, and observed findings.
Do not copy an illustrative finding as evidence about your application.

```sh
cd "$HOME/Projects/my-app"
dh project init --goal "Improve authentication reliability" \
  --constraints "Keep data local. Approve scope before edits."
dh quality init
dh quality command tests \
  --argv-json '["python","-m","unittest","discover","-s","tests"]' \
  --protect tests --timeout 180
dh task new "Fix expired-session handling" \
  --done "Trusted checks and a new expiry regression test pass" \
  --file src/auth.py --check tests
dh quality context "Expiry handling source and regression tests"
```

Run `project init` once per checkout. `quality init` enables executed quality
gates and invalidates earlier manual evidence. Register commands that really
test your project. The example resolves `python` from your current PATH;
if the application has its own environment, register its actual interpreter,
such as `.venv/bin/python`, instead. The harness environment does not install
your application's dependencies.

For an Android project, a suitable command might instead be
`["./gradlew","testDebugUnitTest"]`, registered from the app root with its actual
trusted test directory. Register build/lint/device checks separately and include
each required name with another `--check`. Device presence is not a functional test.

Keep handoff files and reports outside the repository so generating them does
not invalidate Git-bound evidence:

```sh
mkdir -p "$HOME/.dedicatedharness/handoffs"
dh task handoff opencode \
  --out "$HOME/.dedicatedharness/handoffs/auth-investigate.txt"
```

Read this file and give its contents to your existing agent in the app checkout.
Use `openclaw` instead of `opencode` for that agent. Investigation does not
authorize code edits. After real investigation, record your actual finding and
reviewed plan. The following text is illustrative:

```sh
dh task checkpoint \
  --finding "The expiry boundary appears inconsistent." \
  --decision "Correct the boundary and add a separate regression test."
dh task approve --scope "Change expiry handling and add its regression only."
dh task handoff opencode \
  --out "$HOME/.dedicatedharness/handoffs/auth-implement.txt"
```

Give the reviewed implementation handoff to the agent. Existing protected graders
must remain unchanged; add new regression tests separately. Intentional edits
to an existing trusted grader need explicit review and command re-registration.
Only after the implementation actually exists:

```sh
dh task checkpoint --change "Corrected expiry handling; added a regression."
dh task ready
dh quality run tests
dh task finish
```

Do not record the example change until it describes real changes. Run every
required check. A nonzero exit, timeout, output cutoff, repository mutation,
changed grader, or stale task cannot produce an accepted pass.
Reviewed commands still execute project code: this is not an OS sandbox.

Git drift is checked when you select/read a task. Working-tree changes invalidate
checks; branch or HEAD changes revoke approval. Ignored files are excluded, and
submodules are unsupported. This is not a continuous filesystem watcher.

Without `quality init`, `task validate` stores user-reported results, not
independently executed evidence. In verified mode, manual passing claims are
refused. Emulator and real-device validation must remain separate.

## 7. Resume tomorrow without replaying everything

```sh
dh project list
dh project show
dh task list
dh task show
dh ask "What remains before this is finished?" --project
dh ask "Explain the remaining tradeoff." --project --remember
dh task history --limit 10
```

Use `dh task use TASK_ID` to select another task. Use `--root PATH` when outside
the app checkout, and `--id TASK_ID` for a specific existing task command.
For project-aware queries, the explicit selector is `--task-id TASK_ID`.

`--remember` stores only the latest successful question and answer locally.
It does not turn model statements into findings, approvals, or passing checks.
`dh task forget` clears that exchange logically, not as a secure erase.
A checkpoint clears the optional exchange while retaining compact audit history.
Project state is private to this resolved checkout and harness home; cloning
or moving the repository does not transfer it automatically.

## 8. Measure coding reliability honestly

Set inference settings before comparing profiles:

```sh
dh quality profile --temperature 0 --top-p 1 --seed 0
dh quality case expiry-case-v1 --goal "Fix expiry handling" \
  --done "Independent expiry grader and regressions pass" --check tests
dh quality trial expiry-case-v1
```

Register independent, representative cases before solving them. Each needs
protected graders. A trial creates a task; follow the investigation, approval,
implementation, and executed-check workflow. Restore the original unsolved Git
baseline before another model's trial. Do this only in a dedicated evaluation
checkout after preserving your work; no destructive reset command is required
by this guide. A fresh clone has separate private state, so compare models within
the same evaluation checkout and harness home.

```sh
dh quality report
```

The target is all required checks for an accepted change, and at least 95%
first-submitted-candidate success across at least 100 distinct cases for a
declared model/prompt/sampling profile. The first failure cannot be replaced by
a successful repair. Pending first trials remain in the denominator and prevent
qualification. The report includes a Wilson 95% confidence interval.

This is not model-generation pass@1, code coverage, or a correctness guarantee.
Ordinary external agents can retry or choose other providers without observation.
Supervised `dh run` trials record actual engine version, harness checksum, and
limits in a separate execution profile. Their model requests are observed and
pinned through the local gateway; they are not pooled with manual sessions.
A 95/100 observation does not establish a true 95% rate at 95% confidence.
Sampling changes do not train weights or make a small model equal to a stronger
hosted model. No automatic fine-tuning or parameter search is implemented.

## 9. Stage upgrades safely

Save your working profile before experimenting, then stop the managed server.
Run discovery from the app checkout to consult its evaluation evidence:

```sh
dh models --recommend coding
dh profile save coding-stable
dh upgrade --purpose coding --keep-old
```

Automatic promotion requires better observed results on matching cases,
graders, starting fingerprints, runtime, budgets, and sampling settings, plus
the quality target. A newer release or more downloads alone cannot replace
your active model.

To prepare an unmeasured candidate, replace `OWNER/MODEL` with its real Hub ID:

```sh
dh upgrade --repo OWNER/MODEL --stage
dh upgrade --purpose coding --stage --profile coding-candidate
```

These are alternative staging examples, not commands to run consecutively.
Staging downloads and smoke-tests a candidate without activating it. A named
candidate can then be activated for explicit evaluation:

```sh
dh profile use coding-candidate
dh serve
```

In another terminal, repeat `dh check-tools` and run representative tasks.
Switching for evaluation is an explicit override, not proof the model is better.
After stopping the managed server, restore the retained working configuration:

```sh
dh profile use coding-stable
dh profile list
```

Saved names cannot be overwritten. Activation rechecks actual hardware,
retained owned snapshot contents, the runtime version, and offline generation.
It clears tool verification and leaves the current configuration unchanged on
failure. Cached restoration needs no Hub access and does not charge a second
download's disk budget. External profiles check their exact exposed model ID;
the harness cannot attest an external application's underlying weights.

Eligible promotion changes configuration only after successful verification.
Without `--keep-old`, cleanup can remove only the previous manifest-owned
revision from the dedicated cache. It does not remove LM Studio/shared caches
or unowned models. Revisions referenced by any saved profile are retained.
Failed verification retains the active configuration.
An explicitly requested runtime installation/update is a separate modification.
There is no background upgrade watcher.

## 10. Common problems

| Problem | What to do |
| --- | --- |
| `dh` is not found | Activate the harness environment in this terminal. |
| No compatible recommendation | Inspect `dh models` reasons. Close heavy apps or select a smaller supported model. Do not bypass the budget. |
| Very low available memory on a 24 GB Mac | Update the harness and run `dh doctor`. Availability estimates free + speculative + inactive pages, not just free pages; inactive pages may need reclamation. If still insufficient, unload LM Studio/other models and close heavy apps. |
| Already have models in LM Studio | Reuse a loaded model via Option B's `dh connect`, or keep its files and unload it before managed MLX setup. The harness does not scan/deduplicate that library or delete its downloads. |
| Endpoint unavailable | Start `dh serve`, or start/load LM Studio for an external connection. |
| Setup says a listener is active | Stop that server yourself. The harness will not kill it. |
| Prompt exceeds context | Select fewer files or smaller ranges. Reserve output space. |
| Protected grader changed | Review the change and re-register the command. Old case versions retain their original grader digests. |
| Checks became stale | Inspect Git drift, record the real implementation checkpoint, use `task ready`, and run checks again. |
| Export or handoff already exists | Choose a new filename; generated files are not overwritten. |
| No 95% result | Collect representative independent trials. Do not count latency benchmarks as quality scores. |
| Project state is missing | Use the original resolved checkout and harness home. Another clone has separate state. |
| Another operation is active | Wait for the supervised run to end before changing model or sampling. |
| Run refused after a model switch | Start the selected server and repeat `dh check-tools`. |
| Run cannot resume | Restore the exact approved scope/profile/workspace and repeat original flags, or review and authorize a new run. Exhausted budgets cannot be reset by resume. |
| OpenCode configuration isolation failed | Use the supported 1.x engine. Managed configuration must not override the restricted settings; no prompt is sent after a failed preflight. |

Use `dh --help`, `dh task --help`, and `dh quality --help` for command syntax.
For application-specific test commands, use that application's documentation.

## 11. Supervise coding instead of handing off manually

Use this route from the Git checkout root after section 6's investigation,
trusted-command registration, and explicit task approval, but before its manual
implementation handoff. Install OpenCode 1.x separately if needed; the adapter
is validated with 1.19.4 and rejects OpenCode 2. Keep the selected MLX or
LM Studio server running. The CLI starts and stops its own OpenCode server;
it never reconfigures your usual OpenCode instance.

The following paths are examples. Select your approved source and a new test
file rather than granting edits to existing protected graders:

```sh
dh check-tools
dh run --allow-edit src/auth.py --allow-edit tests/test_auth_expiry_new.py
dh runs
```

Without `--allow-edit`, only the task's selected files are editable. A literal
directory grants its descendants, except protected graders. New file paths are
allowed inside the checkout; wildcard arguments and checkout-wide edit grants
are refused. Existing changes are preserved; the harness does not reset Git.

The worker receives the goal, approved plan, constraints, selected paths, and
path suggestions. It reads code/tests just in time. A local gateway pins the
exact model and sampling, serializes generation, admits actual prompt/tool
schemas, and compacts tool-result text with an explicit omission marker.
It never silently removes the goal or sends an over-budget prompt. An 8,192-token
managed context is a useful experiment; external byte-estimated admission may
need a larger declared/server context. Increase context only after fit screening.

Default limits are six steps per turn, at most one repair, sixteen model
requests, 600 seconds, and an aggregate admission budget of 32,768 units for
MLX or 131,072 for external servers. Override with `--budget`. This adds input
admission units and reserved output tokens; external input units are estimated
bytes, not measured tokens, so budgets are not directly comparable between
backends. All limits apply across repairs:
one may stop the run before another full repair is available. Engine shutdown
and filesystem cleanup can extend elapsed time after the deadline.

The owned engine has separate home/config/data/cache, authenticated numeric
loopback endpoints, one local provider, and no global/project configuration,
external plugins/skills, MCP, LSP/formatters, sharing, web tools, model-generated
shell commands, or subagents. File paths and protected graders are enforced by
the gateway and OpenCode policy. This is not an OS sandbox. Previously approved
checks execute project code, and other writers can still cause drift.

The supervisor waits for a real terminal/idle engine state, records Git-observed
changes, executes trusted checks, and gives a compact failure tail to a fresh
repair session if budgets permit. It stops on no progress or stale approval,
graders, profile, or Git state. It stops its engine before final acceptance.
The first failed checked candidate remains failed in evaluation reports.
Required device checks without approved commands remain pending, never invented.

```sh
dh runs --id RUN_ID
dh run --resume RUN_ID
```

Replace `RUN_ID` with the printed ID and repeat original nondefault
`--allow-edit` and budget flags when resuming. Resume starts a fresh session from
compact state, not the old full transcript. It requires the same model, engine,
scope, commands, task revision, Git fingerprint, and limits. Consumed budgets
remain consumed. Exhausted runs require review before a newly authorized run.
Model/setup/profile/sampling changes are locked out while supervision is active.

Default storage is compact operational metadata plus bounded check-output
tails, which may contain private project data. Raw OpenCode state is deleted
after shutdown. `--keep-trace` explicitly retains source/prompts/transcripts in
the harness home and prints its location in the run record. LM Studio/other
server logging remains that application's separate responsibility.

OpenClaw still uses provider exports and handoffs; no OpenClaw session
supervisor is claimed. No weight training or automatic cloud escalation occurs.

## 12. Remaining enhancements

These are proposals, not shipped commands. Prioritize measured local workloads.

| Priority | Enhancement | Intended benefit |
| --- | --- | --- |
| 1 | Mac calibration and first-run wizard | Measure warm latency, actual memory pressure, and usable context on your machine; guide runtime/server setup. |
| 2 | Syntax/symbol-aware context and OpenClaw session adapter | Improve retrieval and observe another engine without copying its entire implementation. |
| 3 | Isolated evaluation baselines | Automate clean paired cases and independent application/E2E grading rather than treating a passing unit test as complete correctness. |
| 4 | More architectures and measured harness tuning | Add correct memory/runtime support and compare one change at a time on held-out tasks. |

Do not default to a multi-agent swarm or add a graph framework only for branding.
Add bounded routing or independent workers only after measurements show a
quality or cost benefit on your Mac.

## 13. Updating the CLI

From the harness checkout, with a clean working tree:

```sh
cd "$HOME/Projects/dedicatedharness"
git pull --ff-only
. .venv/bin/activate
python -m pip install -e .
dh doctor
```

Preserve any local edits before updating. This updates the CLI, not model
weights automatically. GitHub Actions checks portable tests, real OpenCode
protocol/file tools with a fake local model on Linux/macOS, and wheel packaging.
It does not establish actual M5 model speed, Metal behavior, or agent accuracy.
Use [README.md](../README.md) for detailed limits and
[RESEARCH.md](../RESEARCH.md) for the architecture rationale and sources.
