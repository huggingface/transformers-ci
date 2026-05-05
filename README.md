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
configure-ci-otel \
  --suite local_smoke \
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
  --suite resource_demo \
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
   `transformers-ci-traces.lor-e.huggingface.cool`, not `transformers-ci.lor-e.huggingface.cool`.

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
