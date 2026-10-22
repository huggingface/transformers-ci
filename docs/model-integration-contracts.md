# Model integration contracts in PR CI

MICs run small model fixtures through executable usage examples and compare their
outputs with reviewed expectations. The runner, dependency selector, schemas and
summary renderer live in `transformersci.contracts`.

## Shared fleet

The two fleets live in the
[`huggingface-mic/transformers-fleet`](https://huggingface.co/datasets/huggingface-mic/transformers-fleet)
Hub dataset. `fleet-lock.json` defines the control fleet; `prototype-fleet.lock`
defines the BF16 prototype fleet. There are no runner fleet copies in the
Transformers or transformers-ci source trees.

Transformers keeps one file, `tests/contracts.lock`, containing the immutable
dataset commit SHA. During the framework pin rollout, CI ignores this file,
resolves dataset `main` once per job, and passes that immutable commit explicitly
to both invocations. It works with checkouts that have no pin. The CLI reads the
local pin by default; outside a checkout, it resolves dataset `main`. Use
`--fleet-revision` to override the pin for reproduction.
Results, OTEL spans and reproduction commands carry the dataset revision. A later
fleet update therefore does not change how an earlier run is reproduced.

```bash
pip install 'transformers-ci[contracts,otel]'
run-model-contracts --fixtures-only --fleet control
run-model-contracts --fixtures-only --fleet prototypes --variant bfloat16
# Reproduce a recorded job:
run-model-contracts --fixtures-only --fleet prototypes --variant bfloat16 \
  --fleet-revision <dataset-commit>
# Select MICs affected by a framework change:
git diff --name-only origin/main...HEAD > /tmp/mic-changed-files.txt
run-model-contracts --fixtures-only --fleet prototypes --variant bfloat16 \
  --changed-files /tmp/mic-changed-files.txt --diff-base origin/main \
  --output /tmp/results.jsonl 2>/tmp/run.log
report-model-contracts /tmp/results.jsonl --log /tmp/run.log
```

Install the framework and its runtime dependencies as well. Runnable Markdown
execution requires the doc-builder revision installed by
`.github/workflows/pr-ci_contracts.yml`. `--lock PATH` remains available for an
explicit custom fleet; it cannot be combined with `--fleet` or `--fleet-revision`.

## Selection and baselines

Model changes select MICs that use that model or its processors. Additive Auto
registrations and model imports under `TYPE_CHECKING` select only the registered
models when `--diff-base` is supplied. A new model without a MIC selects none of
the existing MICs. Changes to existing registrations, shared runtime code,
dependencies, or unavailable diffs retain full-fleet selection. Push and
merge-group events run both entire fleets.

Fleet entries pin immutable `huggingface-mic/<name>-mic` model snapshots containing
weights, `integration.json`, usage, and reviewed expectations under
`integrations/<framework>/expectations/<platform>/<target>[.<variant>].json`.
Contract and fixture hashes verify executable code and weights independently.

Checks cover exact outputs, numeric shapes and tolerances, required outputs,
finite values, named gradient checks and boolean assertions. Missing references
and identity mismatches fail. Known failures and excluded outputs remain scoped
in the fleet files. Runtime version differences are recorded in the results.
`--record` writes candidate files after fresh-process reproduction; it never
approves them. Review measured changes before updating model expectations and
fleet pins in the dataset using `tmic export --execute`, then adopt the resulting
dataset SHA with `tmic pin --transformers-checkout PATH --revision SHA`. `--expectations PATH` can override baseline lookup for
local review.

## Observational behavior and telemetry

Assertion and execution failures return a nonzero exit code. The reusable CI
workflow keeps the job observational: errors appear in logs, summaries, artifacts
and OTEL without blocking the PR. Setup errors are reported too.

Each case has a timed span named
`tests/contracts::<contract>::<target>[<variant>]`. Attributes identify the model,
variant, platform, random/overfit weights, selection reason, model and baseline
pins, plus `mic.fleet_repository` and `mic.fleet_revision`. Assertion and execution
failures carry ERROR status with their full details. Expected failures and
unsupported cases retain their MIC status. Without an endpoint, tracing is a no-op.

The dashboard's MIC page reads the failure index and compact per-run records.
It provides branch, variant, platform, failure-type and time-window filters,
model links, internal PR/trace/run links, assertion details and daily failure bars.
The global rate divides failed checks by recorded MIC checks for the selected
window and branch scope, across all variants and platforms. The exporter uses a
small offline provenance index for historical labels; it does not fetch fleets
or model metadata over public egress.

With **Sticky failures only** enabled, the historical rate instead uses observed
checks for matching cases whose latest check still fails, before the group limit.
Its label changes to **Sticky failure rate**. The separate **Failing now** metric
continues to describe the latest state across all cases in the branch scope.
