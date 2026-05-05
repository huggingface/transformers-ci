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

## GitHub Actions Usage

To use `transformers-ci` in a GitHub Actions workflow:

1. **Add repository secrets:**
   - `OTEL_EXPORTER_OTLP_ENDPOINT` - The OTLP endpoint URL (e.g., `https://transformers-ci-traces.lor-e.huggingface.cool`)
   - `OTEL_EXPORTER_OTLP_HEADERS` - API key header in Bearer format (e.g., `Authorization=Bearer <your-api-key>`)

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
        run: configure-ci-otel -- pytest tests/ -v
        env:
          OTEL_EXPORTER_OTLP_ENDPOINT: ${{ secrets.OTEL_EXPORTER_OTLP_ENDPOINT }}
          OTEL_EXPORTER_OTLP_HEADERS: ${{ secrets.OTEL_EXPORTER_OTLP_HEADERS }}
```

### Local Testing (No Endpoint Required)

To test the instrumentation locally without sending traces to a remote endpoint:

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests with local resource metrics collection (no OTLP endpoint needed)
configure-ci-otel --force-export-traces --suite local_test -- \
  pytest tests/test_demo_workload.py -v \
  --resource-metrics-file /tmp/pytest-metrics.jsonl
```

This writes per-test CPU, RSS, and CUDA metrics to a local JSONL file without requiring an OTLP endpoint.

## Dashboard

An initial Grafana dashboard implementation lives under [`dashboard/`](dashboard/README.md).
