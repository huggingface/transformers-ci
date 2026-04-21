# transformers-ci

This project contains CI packages used for the `transformers` project.


## OpenTelemetry


`transformersci.otel`, contains the OpenTelemetry support for `pytest` tests,
that can be used from the main `transformers` repository.

The main pieces are:

- `transformersci.otel.cli`: configures OTEL env vars and launches pytest
- `transformersci.otel.resource_plugin`: optional per-test CPU, RSS, and CUDA memory sampling
- `transformersci.otel.trace_exporter`: converts recent Jaeger traces into Prometheus metrics

## CLI Entry Points

Installing the package exposes:

- `configure-ci-otel`
- `pytest-trace-exporter`

Example:

```sh
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:5317 \
OTEL_SERVICE_NAME=transformers-tests \
configure-ci-otel --job-name local_smoke -- python3 -m pytest tests/test_cli.py -q
```

## Pytest Plugin

The resource plugin is registered through `pytest11` and stays inert unless resource collection is explicitly enabled with one of:

- `--resource-metrics-file <path>`
- `PYTEST_RESOURCE_METRICS_FILE=<path>`
- `TRANSFORMERS_TEST_RESOURCE_METRICS_FILE=<path>`

Example:

```sh
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:5317 \
OTEL_SERVICE_NAME=pytest-observability-demo \
configure-ci-otel --job-name resource_demo -- \
  python3 -m pytest tests/test_demo_workload.py -q \
  --resource-metrics-file dashboard/data/pytest-resource-metrics.jsonl
```

## Dashboard

An initial Grafana dashboard implementation lives under [`dashboard/`](dashboard/README.md).
