# Benchmark pages

`/benchmarks` replaces the browsing experience of the transformers-perf Space
inside the CI dashboard. It reads the same dataset,
`hf-internal-testing/transformers-perf`, and uses the analysis from
`transformers-perf` pinned in the `benchmarks` optional dependency. The Space and
its publishing job are left in place until the dashboard rollout is verified.

The overview links to `/benchmarks/main`, `/benchmarks/pr-<number>` and
`/benchmarks/benchmark-changes`. Daily charts show generation, multimodal
processors and encoders; comparison pages include all paired runs, confidence
intervals, verdicts, counters and failed scenarios. GET forms select the metric,
model and group. Python renders HTML, SVG and detail tables; these pages require
no JavaScript, frontend build or Space iframe.

## Data flow

The `benchmark-refresh` container periodically resolves one immutable Hub
revision and downloads only `data/*/*.parquet` and `processors/*/*.parquet`.
It reuses the pinned `tperf.dashboard.build` analysis, then atomically replaces
`/benchmarks/snapshot.json`. Unchanged dataset revisions skip the rebuild.
GitHub PR titles are best effort, as in the Space; a GitHub rate limit does not
hide benchmark results.

The exporter reads that snapshot through a read-only mount and stays
stdlib-only. It never contacts the Hub in the HTTP request path. A failed refresh
retains the last complete snapshot. Pages flag results more than 48 hours old;
without any snapshot they return a friendly 503. Unknown pages return 404.
A shared 4 GiB emptyDir contains the snapshot and Hub cache; it is rebuilt on pod
replacement. Refresh errors appear in the `benchmark-refresh` container logs.

## Local preview

Use Python 3.11 or newer for the refresher:

```bash
python -m pip install -e '.[benchmarks,test]'
python -m transformersci.benchmarks.refresh --once --path /tmp/benchmarks/snapshot.json
TCI_BENCHMARK_SNAPSHOT=/tmp/benchmarks/snapshot.json python -m transformersci.otel.trace_exporter
```

Open the exporter's `/benchmarks` path. The refresher uses standard Hub
credentials (`HF_TOKEN` or the local login) and the Hub cache. `--dataset`,
`--interval` and `--path` override defaults. Tests with the extra installed:

```bash
python -m pytest tests/test_benchmarks.py tests/test_dashboard_loader.py tests/test_dashboard_auth.py
```

## Rollout

Enable the refresher in the deployment values and pin `traceExporter.sourceRevision`
to the commit containing these changes:

```yaml
traceExporter:
  benchmarks:
    enabled: true
    dataset: hf-internal-testing/transformers-perf
    refreshSeconds: 900
    egress:
      enabled: true  # dedicated allowlisted proxy for restricted clusters
    proxy: ""  # optional alternative HTTP CONNECT proxy
```

The optional `benchmark-egress` proxy admits only the exporter pod and permits
HTTPS to the package, source and Hub download hosts. It has a separate network
policy, with private address ranges excluded from public egress. This does not
change Serge’s shared proxy allowlist. The deployment uses the existing pinned
Serge tinyproxy image.

The refresher installs the `benchmarks` extra at startup, following the existing
publisher's dependency-install pattern. Its network path must permit PyPI,
GitHub/codeload (the pinned package archive and PR titles), huggingface.co and
Hub file-download hosts. With a proxy, its allowlist must cover these hosts;
setting the proxy URL alone does not grant access. The existing configured
`secrets.keys.hfToken` is mounted only in the refresher; the browser receives
neither the token nor the raw dataset. Xet is disabled in this container to use
ordinary HTTPS downloads through that proxy.

The Helm change adds `/benchmarks` routing to the exporter and a refresh sidecar;
the exporter rolls once. Use the normal dashboard deploy plan, preflight and
post-deploy validation. Verify the sidecar has built a snapshot, then open the
overview, daily page and a PR page, including filtering with JavaScript disabled.
The Grafana search customization measures menu width and switches to wrapped
navigation plus a separate search row when needed. Check desktop and mobile
widths and ensure Benchmarks immediately precedes Profile.
