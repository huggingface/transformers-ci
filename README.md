# transformers-ci

This project contains CI packages used for the `transformers` project.


## OpenTelemetry


`transformersci.otel`, contains the OpenTelemetry support for `pytest` tests,
that can be used from the main `transformers` repository.

The main pieces are:

- `transformersci.otel.cli`: configures OTEL env vars and launches pytest
- `transformersci.otel.resource_plugin`: optional per-test CPU, RSS, and CUDA memory sampling
- `transformersci.otel.trace_exporter`: converts recent Tempo traces into Prometheus metrics

## CLI Entry Points

Installing the package exposes:

- `configure-ci-otel`
- `pytest-trace-exporter`
- `assign-reviewers`

Example:

```sh
configure-ci-otel \
  --job local_smoke \
  --service-name transformers-tests \
  --protocol grpc \
  --otlp-endpoint http://localhost:5317 \
  -- python3 -m pytest tests/test_cli.py -q
```

## Pytest Plugin

The resource plugin is registered through `pytest11` and stays inert unless resource collection is explicitly enabled with one of:

- `--resource-metrics-file <path>`
- `PYTEST_RESOURCE_METRICS_FILE=<path>`
- `TRANSFORMERS_TEST_RESOURCE_METRICS_FILE=<path>`

Example:

```sh
configure-ci-otel \
  --job resource_demo \
  --service-name pytest-observability-demo \
  --protocol grpc \
  --otlp-endpoint http://localhost:5317 \
  -- \
  python3 -m pytest tests/test_demo_workload.py -q \
  --resource-metrics-file dashboard/data/pytest-resource-metrics.jsonl
```

## GitHub Actions Usage

To use `transformers-ci` in a GitHub Actions workflow:

1. **Add repository secrets:**
   - `OTEL_EXPORTER_OTLP_ENDPOINT` - The OTLP endpoint URL (e.g., `https://transformers-ci-traces.lor-e.huggingface.cool`)
   - `OTEL_EXPORTER_OTLP_TOKEN` - Raw bearer token, without the `Authorization=Bearer ` prefix

   `configure-ci-otel` can now set `OTEL_TRACES_EXPORTER`,
   `OTEL_EXPORTER_OTLP_PROTOCOL`, `OTEL_EXPORTER_OTLP_ENDPOINT`, and
   `OTEL_EXPORTER_OTLP_HEADERS` for you from CLI flags.
   Use the traces collector host here, not the Grafana UI host:
   `transformers-ci-traces.lor-e.huggingface.cool`, not `transformers-ci.huggingface.cool`.

2. **Update your workflow:**

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install transformers-ci
        run: pip install transformers-ci[otel]

      - name: Run tests with OpenTelemetry tracing
        run: >-
          configure-ci-otel
          --service-name transformers-tests
          --protocol http
          --otlp-endpoint "${OTEL_EXPORTER_OTLP_ENDPOINT}"
          --token "${OTEL_EXPORTER_OTLP_TOKEN}"
          -- pytest tests/ -v
        env:
          OTEL_EXPORTER_OTLP_ENDPOINT: ${{ secrets.OTEL_EXPORTER_OTLP_ENDPOINT }}
          OTEL_EXPORTER_OTLP_TOKEN: ${{ secrets.OTEL_EXPORTER_OTLP_TOKEN }}
```

### Mirroring to a staging backend

To send every span to a second (staging) backend in addition to the primary,
add `--staging-endpoint` (and `--staging-token` if staging uses its own auth).
This attaches a second span processor inside the pytest run, so each span is
exported to both backends; a flaky staging box does not affect the primary
export. Staging reuses the primary `--protocol` unless you override it with
`--staging-protocol` (e.g. prod over `http` but the stage box only speaks
`grpc`).

```yaml
      - name: Run tests with OpenTelemetry tracing (mirrored to staging)
        run: >-
          configure-ci-otel
          --service-name transformers-tests
          --protocol http
          --otlp-endpoint "${OTEL_EXPORTER_OTLP_ENDPOINT}"
          --token "${OTEL_EXPORTER_OTLP_TOKEN}"
          --staging-endpoint "${OTEL_STAGING_OTLP_ENDPOINT}"
          --staging-protocol grpc
          --staging-token "${OTEL_STAGING_OTLP_TOKEN}"
          -- pytest tests/ -v
        env:
          OTEL_EXPORTER_OTLP_ENDPOINT: ${{ secrets.OTEL_EXPORTER_OTLP_ENDPOINT }}
          OTEL_EXPORTER_OTLP_TOKEN: ${{ secrets.OTEL_EXPORTER_OTLP_TOKEN }}
          OTEL_STAGING_OTLP_ENDPOINT: ${{ secrets.OTEL_STAGING_OTLP_ENDPOINT }}
          OTEL_STAGING_OTLP_TOKEN: ${{ secrets.OTEL_STAGING_OTLP_TOKEN }}
```

The endpoint/token/protocol can also come from the environment instead of flags
via `TRANSFORMERS_TEST_OTEL_STAGING_ENDPOINT`,
`TRANSFORMERS_TEST_OTEL_STAGING_TOKEN`, and
`TRANSFORMERS_TEST_OTEL_STAGING_PROTOCOL`. If `--staging-token` is omitted,
staging falls back to the primary token; if `--staging-protocol` is omitted, it
falls back to the primary protocol.

### Local Testing (No Endpoint Required)

To test the instrumentation locally without sending traces to a remote endpoint:

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests with local resource metrics collection (no OTLP endpoint needed)
configure-ci-otel --force-export-traces --job local_test -- \
  pytest tests/test_demo_workload.py -v \
  --resource-metrics-file /tmp/pytest-metrics.jsonl
```

This writes per-test CPU, RSS, and CUDA metrics to a local JSONL file without requiring an OTLP endpoint.

## Reviewer assignment

`.github/workflows/assign-reviewers.yml` is a reusable workflow that requests up to two reviewers
on a pull request, ranked by how many lines the PR changes in the files they own. Call it from the
repository being reviewed:

```yaml
on:
  pull_request_target:
    branches: [main]
    types: [ready_for_review]

jobs:
  assign_reviewers:
    permissions:
      contents: read
      pull-requests: write
    uses: huggingface/transformers-ci/.github/workflows/assign-reviewers.yml@main
```

Only the resolution logic lives here (`transformersci.reviewers`). The ownership data stays in the
repository being reviewed, at `.github/scripts/codeowners_for_review_action` — who owns what is
that repo's decision. Point `codeowners_file` elsewhere if it lives somewhere else. A file
resolves to its owners in this order, most specific first:

1. `# Reviewers: @login` in the leading comment block of a model's `modular_*.py`/`modeling_*.py`
2. a path rule in the codeowners file
3. the model's modality — its section in `docs/source/en/_toctree.yml`, matched by an
   `@@modality/<slug> @login` rule
4. the `*` catch-all

Each reviewer is requested in a separate call, and anyone who is not a collaborator is skipped
with a `::warning::` so the next-ranked owner takes the slot — one stale codeowners entry costs
one reviewer, not all of them.

Call it from `pull_request_target`: requesting a review needs a write token, which a fork's
`pull_request` run does not get. The checkout is the PR's base and nothing from the head is
executed; the head is only read file-by-file through the API, as text, to see models and
`# Reviewers:` tags the PR adds, and every login is gated on the collaborator check. The
codeowners file is never read from the head, and `pr-ci-security-gate.yml` can block untrusted
PRs from touching it at all — list it in that workflow's `protected_paths` input, which is opt-in
and leaves the gate unchanged for callers that do not set it.

## Dashboard

An initial Grafana dashboard implementation lives under [`dashboard/`](dashboard/README.md).

MIC execution and baselines: [runner and telemetry](docs/model-integration-contracts.md),
[rollout audit](docs/mic-rollout-audit-2026-10-20.md).

Dashboard GitHub sign-in, user profiles and CLI authorization: [setup and API](docs/dashboard-auth.md).

## @askserge in Slack and on failure pages

See the [infra handoff and rollout guide](docs/askserge-deployment.md) for prerequisites,
deployment order, acceptance checks, and disablement.

Mention `@askserge look at the traceback: <traceback or dashboard test link>` in a
channel containing the bot, or send `look at the traceback: ...` directly to it.
The error page's **Fix it!** and the WDYT dialog's **Deeper investigation** enter
the same workflow. Both prepare a fix, then ask before creating a PR.

The dashboard resolves linked failures through its own repository-filtered Tempo
reader; it never fetches a user-supplied hostname. Serge runs its existing task
pipeline (prior work, reproduction, investigation, normalization and configured
GPU verification). Each investigation gets a private DM thread and a native Slack
agent session. Phase updates and the proposed fix stay in that thread. Sessions
show processing during work, suspended for approval, and closed on completion.
Processing status is refreshed every 20 minutes and failed status updates retry. **Create PR**
rechecks repository write access and the candidate commit; **No PR** declines
publication. You can also reply **yes** or **no** in the investigation’s thread. Verification may create a `serge/slack-*` candidate branch before
approval. No PR is opened automatically. Approval expires 24 hours after launch;
candidate branches are retained. Missing test IDs or disabled verification are
reported without claiming the fix was verified.

Other commands work as mentions, DMs or `/askserge` commands:

- `pr 123`: current-head GitHub check/status counts and a dashboard link.
- `rerun 123`: rerun failed CI through the existing permission and rate-limit gates.
  If active runs need cancellation, review and confirm in the dashboard.
- `investigate 123`: a PR-focused Serge draft review, with a link to its result.
- `help`, `dashboard`, `serge`, `result <id>`.

Mention/DM results are delivered privately. Slash commands immediately return a
private receipt; `result <id>` retrieves the saved result. Interactive traceback
investigations also send their phase updates by DM. Command receipts are retained
for seven days and deduplicate Slack retries. At most four command handlers run
concurrently; users can start 20 commands/hour, with Serge additionally limiting
investigations to five/hour and one running job per GitHub account. The first
version uses operator-verified account mappings rather than self-service linking.

### Configure the existing Slack app

Use [the app manifest](deploy/slack-app-manifest.yaml) as the settings reference
for the existing **askserge** app. Preserve its other features. The manifest enables
`agent_view`, the Messages tab, and suggested prompts. Reinstall the app after
adding scopes. Slack’s Stop control is deliberately not enabled: backend worker
cancellation is not implemented for these sessions yet. It adds
`assistant:write`, `app_mentions:read`, `chat:write`, `im:write`, `im:history`, and `commands`, subscribes
to `app_mention` and `message.im`, and configures signed requests at:

- `https://transformers-ci.huggingface.cool/slack/events`
- `https://transformers-ci.huggingface.cool/slack/actions`
- `https://transformers-ci.huggingface.cool/slack/commands`

Reinstall the app after changing scopes and invite it to the desired channels.
Put its signing secret and bot token in the existing dashboard secret via the
normal secret manager, under `slack-signing-secret` and `slack-bot-token`.
Do not put credentials in Helm values or Git.

```yaml
traceExporter:
  replicas: 1
  runStore:
    enabled: true
  slack:
    enabled: true
    publicUrl: https://transformers-ci.huggingface.cool
    sergePublicUrl: https://serge.huggingface.tech
    appId: A_REPLACE_ME
    botUserId: U_REPLACE_ME
    teamIds: [T_REPLACE_ME]
    actors:
      "T_REPLACE_ME:U_REQUESTER": verified-github-login
    proxy: "" # HTTPS CONNECT proxy if outbound access requires one
```

Verify each mapping with the account holder. Never derive GitHub identity from a
Slack display name. Keep mappings current when a GitHub login changes. Dashboard
users need exactly one matching Slack account. Existing GitHub sign-in and CSRF
checks protect dashboard buttons; Serge independently checks repository access.
The dashboard needs HTTPS access to `slack.com` (directly or through the configured
proxy), in addition to its existing Serge connection. The bot token belongs to
the configured workspace. The Slack routes are disabled unless configured.

Deploy matching Serge web **and task-runner** images containing deferred PR
support before setting `DASHBOARD_INTERACTIVE_TASKS_ENABLED=1` in Serge's
`envVars`. An older runner ignores fields it does not recognize and cannot honor
the approval gate. Retain the existing `TASK_API_ENABLED`, per-repository task
opt-in, dashboard service token/repository allowlist, and ITF verification
configuration. Then deploy and enable the dashboard Slack settings. This change
does not alter production values or install the app.

Active investigation IDs and phase cursors survive exporter restarts. A lost
Serge dispatch response is reconciled by its deterministic task ID. Queued command
workers themselves are not resumed after a crash: check the saved receipt and
Serge before submitting again. Phase delivery can repeat after a crash between
sending a DM and recording its cursor; PR approval is idempotent across retries.

Validation: `pytest -q tests/test_slack_bot.py tests/test_rerun_failed.py
 tests/test_wdyt.py` and `helm lint deploy/helm`; see Serge's task tests for the
publication gate. Smoke-test one real traceback, its DMs, both decision buttons,
and both dashboard entry points after provisioning the app and deploying.
