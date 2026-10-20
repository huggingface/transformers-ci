# Model integration contracts in PR CI

The runner, dependency selector, schemas, summary renderer and fleet locks were recovered from Transformers [PR #49532](https://github.com/huggingface/transformers/pull/49532),
commit `aec34847b4456e24e89b3c2c7f145189611b8a04`. They now live in
`transformersci.contracts`. The reviewed baseline files were copied byte for byte into the consolidated Hub
models, alongside framework usage. Active locks pin the combined snapshots.

Install `transformers-ci[contracts,otel]` plus the framework and its runtime dependencies.
Runnable Markdown execution also requires the doc-builder revision installed in
`.github/workflows/pr-ci_contracts.yml`.

```bash
# Control fleet; select fixtures from the lock, using each default variant.
run-model-contracts --fixtures-only --fleet control
# The 23 prototype models, BF16.
run-model-contracts --fixtures-only --fleet prototypes --variant bfloat16
# Selection by changed model, configuration and processor dependencies.
git diff --name-only origin/main...HEAD > /tmp/mic-changed-files.txt
run-model-contracts --fixtures-only --fleet prototypes --variant bfloat16 \
  --changed-files /tmp/mic-changed-files.txt --output /tmp/results.jsonl 2>/tmp/run.log
report-model-contracts /tmp/results.jsonl --log /tmp/run.log
```

The dependency selector was already present in the deployed `tmic` invocation.
The migration restores it locally and fixes an empty selection incorrectly failing
when the control's `--target` was supplied. Documentation-only changes do not need
Hub downloads. Model changes include processor dependencies: a GPT-2 tokenizer
change can select Llama and SmolVLM; shared library or dependency changes select all.
Push and merge-group events run the entire fast fleet.

Baselines check exact outputs, numeric shapes and tolerances, required outputs,
finite values, named gradient checks and boolean assertions. A missing baseline or baseline identity mismatch fails. Locked file hashes verify contract code and fixture bytes independently in the
combined snapshot. Baseline identity checks use stable target, framework, variant
and platform identities. Original source revisions remain provenance after the
content-preserving relocation; baseline values and tolerances were preserved.
Runtime version differences are recorded in each result.

Use `--lock PATH` for a framework-owned fleet; the baseline directory is read from
`integrations/<framework>/expectations/` in the pinned MIC snapshot. Legacy
split-repo locks retain their `expectations/` directory beside the lock. `--expectations PATH` overrides it. `--record`
writes `.candidate.json` files only after fresh-process reproduction. Review values
and provenance before replacing an approved baseline; CI never records or approves.

## Observational behavior and telemetry

The command exits nonzero on an assertion or execution failure. Its CI step has
`continue-on-error: true`, so the GitHub job remains green while full failure text
appears in the job log, summary and artifacts. Installation and summary errors
also remain observational. The job remains outside the final CI gate.

`configure-ci-otel` sets the usual PR, run, repository and job resource attributes.
Each selected target/variant gets a timed span named
`tests/contracts::<contract>::<target>[<variant>]`, tagged with `pytest.nodeid`
and `pytest.span_type=test`. Existing dashboard ingestion can consume these spans.
Case attributes include `mic.contract`, `mic.contract_revision`, `mic.target`,
`mic.revision`, `mic.kind`, `mic.variant`, `mic.framework`, `mic.selected_by`,
`mic.platform`, `mic.baseline`, `mic.baseline_repo`, `mic.baseline_revision`,
`mic.baseline_path`, and `mic.status` when available. Assertion and
execution failures carry ERROR status and exception details. Expected failures
and unsupported cases retain their MIC status attributes; they are not promoted
to ERROR unless execution itself fails. Setup errors emit a `tests/contracts::runner`
failure. Providers flush on exit; without an endpoint tracing is a no-op.

The two command invocations share the `fast-integration` job identity. The log
formatter accepts OTEL wrapper diagnostics as well as JSON records. Summary rendering
combines selection reports from both fleets and produces fleet-specific repro commands.

## Follow-up: dedicated MIC failures page

After these spans are observed in production, add a dedicated dashboard page
filtered to the `fast-integration` job and `mic.*` attributes. Show failures by
contract, target, variant, platform, PR/commit, selection reason and baseline,
with the assertion text, trace and CI run links. Keep expected failures distinct.
This page is a follow-up; this change establishes its telemetry inputs.

## Hub ownership and consolidated snapshots

The active fleet uses 33 `huggingface-mic/<name>-mic` model repositories.
Each contains the slim weights, `build.json`, `integration.json`, framework usage
and `integrations/<framework>/expectations/<platform>/<target>[.<variant>].json`.
All 373 copied reviewed reference files retained their original bytes. Weathernext2
has no reviewed baseline yet; its expectations directory documents that gap.

The migrated locks keep historical fixture target names with an explicit `target`
field. Contract selection keys and known-failure names also stay stable across
repository renames. `.mic-repo.json` defines the contract's file inventory, so
adding a baseline cannot accidentally invalidate the weight hash or hide a
contract source change. Contract and fixture roles pin the combined snapshot.

The 24 old split fixture repositories were deleted after their replacements were
verified. Land the consolidated runner before switching Transformers to these
locks. The integration repo renames retain their original Git history and Hub
redirects. Full usage and slim-model descriptions remain on the umbrella cards.
