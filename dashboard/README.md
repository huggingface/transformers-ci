# Dashboard

This directory is the initial implementation of the Grafana dashboard for pytest observability.

It includes:

- `docker-compose.yml`: Grafana, Jaeger, OpenSearch, Prometheus, and the OTEL collector
- `jaeger.yaml`: Jaeger v2 config — OTLP receivers and OpenSearch storage backend
- `otelcol.yaml`: OTLP receiver and Jaeger exporter config
- `prometheus.yml`: scrape config for the trace exporter
- `grafana-datasources.yaml`: provisioned Jaeger and Prometheus data sources
- `grafana-dashboard.yaml`: dashboard provisioning
- `pytest-observability-dashboard.json`: overview dashboard
- `pytest-observability-job-dashboard.json`: per-job drill-down
- `pytest-test-dashboard.json`: per-test view (metadata + stacktrace when failed)
- `data/`: shared local data directory for resource metrics

## Architecture

Trace storage is backed by an external OpenSearch service, not Jaeger's
embedded Badger store. The flow is:

```
test/job instrumentation
  -> otelcol           (host ports 5317 gRPC / 5318 HTTP)
  -> jaeger (v2 OTLP)  (internal 4317)
  -> opensearch        (internal 9200, persisted to opensearch-data volume)
  -> jaeger query/UI   (host port 16687, internal 16686)
  -> Grafana           (host port 3000)
```

Image versions are pinned. Trace data survives a `jaeger` container restart
because it lives in the `opensearch-data` named volume.

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
- Jaeger: `http://localhost:16687`
- OTLP gRPC collector endpoint: `http://localhost:5317`
- OTLP HTTP collector endpoint: `http://localhost:5318`

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

## Jaeger

- `http://localhost:16687`

Useful tags:

- `transformers.test.run.id`
- `transformers.test.job`
- `transformers.test.job.run`
- `vcs.change.id`

## Validate The Stack

After `docker compose -f dashboard/docker-compose.yml up -d`:

```sh
# OpenSearch healthy (status green or yellow):
curl -s http://localhost:9200/_cluster/health | jq .

# Jaeger UI reachable:
curl -fI http://localhost:16687/ | head -1   # expect HTTP/1.1 200

# Send a test trace (uses the same otelcol path tests use):
./dashboard/sample-run.sh

# Confirm a Jaeger index was created in OpenSearch:
curl -s http://localhost:9200/_cat/indices/jaeger-main-*?v

# Confirm the pytest trace exporter can query Jaeger:
docker compose -f dashboard/docker-compose.yml logs pytest-trace-exporter | tail
```

To verify persistence, restart Jaeger and confirm old traces are still queryable:

```sh
docker compose -f dashboard/docker-compose.yml restart jaeger
# wait ~10s, then re-open the Jaeger UI; previously ingested traces are still listed.
```

## Rollback

The previous setup used `jaegertracing/all-in-one` with a Badger volume.
To roll back without redeploying from scratch:

1. `git revert` the OpenSearch migration commit (or check out the pre-migration
   `docker-compose.yml` and delete `jaeger.yaml`).
2. `docker compose -f dashboard/docker-compose.yml up -d`. The previous
   `jaeger-data` volume, if it still exists on the host, is reattached and any
   spans it held become queryable again.
3. The `opensearch-data` volume is left in place and can be removed with
   `docker volume rm dashboard_opensearch-data` once rollback is confirmed
   stable.

Note that the two backends do **not** share data — traces written to OpenSearch
are not migrated back into Badger on rollback. Treat the rollback as a fresh
window of trace history.
