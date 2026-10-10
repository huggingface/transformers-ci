# Dashboard

This directory is the initial implementation of the Grafana dashboard for pytest observability.

It includes:

- `docker-compose.yml`: Grafana, Tempo, and Prometheus
- `tempo.yaml`: Tempo config — OTLP receivers and local trace storage backend
- `prometheus.yml`: scrape config for the trace exporter
- `grafana-datasources.yaml`: provisioned Tempo and Prometheus data sources
- `grafana-dashboard.yaml`: dashboard provisioning
- `pytest-observability-dashboard.json`: overview dashboard
- `pytest-observability-job-dashboard.json`: per-job drill-down
- `pytest-test-health-dashboard.json`: test health dashboard
- `pytest-test-dashboard.json`: per-test view (metadata + embedded Tempo trace view when failed)
- `data/`: shared local data directory for resource metrics

## PR diff panel

The Branch panel has a **Patch | Tests** switch next to **Re-run failed tests**.
Tests is the default view; choosing Patch replaces the panels below Branch with
a single diff panel and loads the iframe on demand. The selection is kept in
the URL. The Tests button animates while the selected PR has queued or running
jobs, using the `pytest_run_job_active` metric (polled every 30 seconds).

The PR dashboard embeds `/patch-view` in an iframe and passes the selected PR and
repository. The trace exporter serves the unified diff and a sidebar of discussion,
review summaries, and inline review threads. File filters hide tests, generated
modular outputs, copyright headers, and Python comment lines; comment filters hide
bots, resolved threads, and command comments. **Show the line(s)** highlights the
current code or displays saved context for an outdated comment. Icon controls
refresh, collapse/expand, and download the full patch. The comments sidebar starts
level with the patch filters and scrolls independently. Separate progress bars
track diff and comment loading.

GitHub responses are cached for one minute. Diffs and individual API responses
are limited to 4 MiB; comment pagination stops at 500 entries per category and
shows a truncation notice. Resolved status uses GitHub GraphQL and requires the
dashboard's GitHub token. Only configured public repositories can be requested.

To iterate locally with Grafana listening on port 3301:

```bash
PYTHONPATH=src .venv/bin/python deploy/scripts/preview-patch-view.py
```

Open `http://localhost:8765/patch-view?pr=43451` for the panel alone. The preview
reloads UI assets on each request. To use downloaded snapshots, pass
`--fixture-dir /path/to/snapshots`, with `<PR>/pr.diff` and `<PR>/comments.json`
inside that directory.

## Architecture

Traces are ingested and stored by a single Grafana Tempo service (no separate
OTEL collector and no external trace database). The flow is:

```
test/job instrumentation
  -> tempo (OTLP receiver)   (host ports 5317 gRPC / 5318 HTTP -> internal 4317/4318)
  -> tempo local storage     (internal, persisted to the tempo-data volume)
  -> tempo query API         (host port 3200)
  -> pytest-trace-exporter   (polls /api/search + /api/traces, exports Prometheus metrics)
  -> Grafana                 (host port 3000; Tempo datasource for trace views)
```

The host-facing OTLP endpoints (5317 gRPC / 5318 HTTP) are unchanged from the
previous otelcol-fronted setup, so emitters need no reconfiguration.

Image versions are pinned. Trace data survives a `tempo` container restart
because it lives in the `tempo-data` named volume.

## Public Data Publisher

The `ci-data-publisher` sidecar publishes the CI telemetry to a **public HF
bucket** (`hf://buckets/huggingface/transformers-ci-telemetry`) on an hourly
cron, so anyone can build apps on top of the CI test data without access to
this stack.

Each cycle reads the last `PUBLISH_WINDOW` (default 48h) of traces from Tempo,
shapes them into daily-partitioned Parquet, writes the raw trace JSON alongside,
refreshes a data card + manifest, and `hf sync`s the staging volume to the
bucket:

```
daily/<YYYY-MM-DD>/test_rows.parquet     one row per (trace_id, test_nodeid)
daily/<YYYY-MM-DD>/run_rollups.parquet   one row per (run_id, test_job)
daily/<YYYY-MM-DD>/traces/<trace_id>.json  raw Jaeger-shaped trace
current_view.json                        manifest (schema, partitions, totals)
README.md                                data card
```

Source of truth is **Tempo** (raw rows), not Prometheus. The publisher reuses
the exporter's Tempo client and `extract_trace_rows`; `model`/`gpu` are derived
from the nodeid/job. Published exception messages and stacktraces are **full and
untruncated** — this bucket is public, so the privacy review (no secrets in
failure text) applies.

Config (env, all overridable via `dashboard/.env`):

- `HF_TOKEN` — **write-scoped** token for the bucket. A **secret**: not written
  by `deploy.sh` (it appends an empty `HF_TOKEN=` placeholder to fill in) and
  never committed.
- `HF_BUCKET_URI` — `hf://buckets/...` destination (default above).
- `PUBLISH_WINDOW` (default `48h`), `PUBLISH_LIMIT` (default `5000`).

Run one cycle locally without syncing (writes Parquet under `./out`):

```sh
PUBLISH_STAGING_DIR=./out \
PYTEST_TRACE_EXPORTER_TEMPO_URL=http://localhost:3200 \
  python -m transformersci.publish.main --dry-run
```

Logs surface via `docker compose -f dashboard/docker-compose.yml logs ci-data-publisher`.

## Start The Stack

From the package root:

```sh
docker compose -f dashboard/docker-compose.yml up -d
```

## Stop The Stack

```sh
docker compose -f dashboard/docker-compose.yml down
```

## Services

- Grafana: `http://localhost:3000`
- Prometheus: `http://localhost:9091`
- Tempo API: `http://localhost:3200`
- OTLP gRPC endpoint: `http://localhost:5317`
- OTLP HTTP endpoint: `http://localhost:5318`

## Run One Traced Pytest Job

```sh
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:5317 \
OTEL_SERVICE_NAME=pytest-observability-demo \
configure-ci-otel --job grafana_demo -- \
  python3 -m pytest tests/test_cli.py -q
```

## Run One Traced Job With CPU And Memory Sampling

```sh
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:5317 \
OTEL_SERVICE_NAME=pytest-observability-demo \
configure-ci-otel --job resource_demo -- \
  python3 -m pytest tests/test_demo_workload.py -q \
  --resource-metrics-file dashboard/data/pytest-resource-metrics.jsonl
```

## Build An Average Across Multiple Runs

Reuse the same `--job` across repeated runs:

```sh
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:5317 \
OTEL_SERVICE_NAME=pytest-observability-demo \
configure-ci-otel --job repeat_avg -- \
  python3 -m pytest tests/test_demo_workload.py -q
```

## Grafana

Open:

- Dashboard: `http://localhost:3000/d/pytest-observability/pytest-observability`
- Explore: `http://localhost:3000/explore?orgId=1`

The main panels are:

- `Slowest Tests By Average Duration`
- `Highest Avg CPU Time`
- `Highest Avg Peak RSS`
- `Aggregated Traces`

The PR dashboard can enrich its top panel from the GitHub API. For higher rate
limits, export `PYTEST_GITHUB_TOKEN` before starting the stack. Optional knobs:
`PYTEST_GITHUB_API_URL` and `PYTEST_TRACE_EXPORTER_GITHUB_CACHE_SECONDS`.

### GitHub PR Integration

The trace exporter is exposed on bare paths (`/failure`, `/badge`, `/summary`),
identical across the compose and kube deployments. The exporter serves a small
live SVG badge and JSON summary for a PR from its cached metrics payload:

```md
[![CI](https://transformers-ci.huggingface.cool/badge/pr?pr=46767)](https://transformers-ci.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=46767)
```

```sh
curl -fsS "https://transformers-ci.huggingface.cool/summary/pr?pr=46767"
```

(The old `/exporter/*` prefix still works on the compose deployment as a
backward-compatible alias.)

## Tempo

- API: `http://localhost:3200`

Browse traces through Grafana's **Explore** view (Tempo datasource) rather than
a standalone UI. Useful resource/span attributes to search on:

- `transformers.test.run.id`
- `transformers.test.job`
- `transformers.test.job.run`
- `vcs.change.id`

A failing test's trace is one click away: the per-test dashboard embeds a Tempo
trace view and links to the full waterfall in Explore.

## Validate The Stack

After `docker compose -f dashboard/docker-compose.yml up -d`:

```sh
# Tempo ready:
curl -fsS http://localhost:3200/ready   # expect: ready

# Send a test trace (uses the same OTLP host ports tests use):
./dashboard/sample-run.sh

# Confirm Tempo has the demo service after ingest (TraceQL search):
curl -s "http://localhost:3200/api/search?q=%7B%20resource.service.name%3D%22pytest-observability-demo%22%20%7D&limit=5" | jq '.traces | length'

# Confirm the pytest trace exporter can query Tempo:
docker compose -f dashboard/docker-compose.yml logs pytest-trace-exporter | tail
```

To verify persistence, restart Tempo and confirm old traces are still queryable:

```sh
docker compose -f dashboard/docker-compose.yml restart tempo
# wait ~15s for /ready, then re-run the TraceQL search above; previously
# ingested traces are still returned because they live in the tempo-data volume.
```

## Rollback

The previous setup used Jaeger v2 + OpenSearch. To roll back without
redeploying from scratch:

1. `git revert` the Tempo migration commit (or check out the pre-migration
   `docker-compose.yml` and restore `jaeger.yaml` / `otelcol.yaml`).
2. `docker compose -f dashboard/docker-compose.yml up -d`. The previous
   `opensearch-data` volume, if it still exists on the host, is reattached and
   any spans it held become queryable again.
3. The `tempo-data` volume is left in place and can be removed with
   `docker volume rm dashboard_tempo-data` once rollback is confirmed stable.

Note that the two backends do **not** share data — traces written to Tempo are
not migrated into OpenSearch on rollback. Treat the rollback as a fresh window
of trace history.
