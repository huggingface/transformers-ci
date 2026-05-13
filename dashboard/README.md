# Dashboard

This directory is the initial implementation of the Grafana dashboard for pytest observability.

It includes:

- `docker-compose.yml`: Grafana, Jaeger, Prometheus, and the OTEL collector
- `otelcol.yaml`: OTLP receiver and Jaeger exporter config
- `prometheus.yml`: scrape config for the trace exporter
- `grafana-datasources.yaml`: provisioned Jaeger and Prometheus data sources
- `grafana-dashboard.yaml`: dashboard provisioning
- `pytest-observability-dashboard.json`: overview dashboard
- `pytest-observability-suite-dashboard.json`: per-suite drill-down
- `pytest-traceback-dashboard.json`: per-trace stacktrace view
- `data/`: shared local data directory for resource metrics

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
configure-ci-otel --suite grafana_demo -- \
  python3 -m pytest tests/test_cli.py -q
```

## Run One Traced Suite With CPU And Memory Sampling

```sh
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:5317 \
OTEL_SERVICE_NAME=pytest-observability-demo \
configure-ci-otel --suite resource_demo -- \
  python3 -m pytest tests/test_demo_workload.py -q \
  --resource-metrics-file dashboard/data/pytest-resource-metrics.jsonl
```

## Build An Average Across Multiple Runs

Reuse the same `--suite` across repeated runs:

```sh
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:5317 \
OTEL_SERVICE_NAME=pytest-observability-demo \
configure-ci-otel --suite repeat_avg -- \
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

## Jaeger

- `http://localhost:16687`

Useful tags:

- `transformers.test.run.id`
- `transformers.test.suite`
- `transformers.test.suite.run`
- `vcs.change.id`
