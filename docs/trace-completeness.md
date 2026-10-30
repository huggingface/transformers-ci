# Trace completeness and failure visibility

The exporter reads whole traces from Tempo before shaping test rows. A missing
job or an unexpectedly green job can therefore indicate incomplete telemetry,
rather than a passing test run. Compare workflow outcomes, emitted span counts,
trace contents, and exported test rows to locate the loss.

## Read-path limits

Successful ingestion does not guarantee a trace can be read back. Tempo's
querier-to-frontend gRPC path and the HTTP client's timeout impose separate
limits. Check trace-by-ID errors and
`pytest_trace_exporter_trace_fetch_errors_total` before changing ingestion limits.
Relevant Tempo settings include `server.grpc_server_max_send_msg_size` and
`querier.frontend_worker.grpc_client_config.max_send_msg_size`.

Increasing read limits also increases parsing memory and render time. Size memory
for concurrent trace fetches plus cached data, and measure render duration before
choosing a refresh interval. Configuration changes must reach the running process;
updating a ConfigMap alone may not restart it. Deployment-specific limits and
incident measurements belong in the operator's runbooks.

## Source-side reduction

`transformersci.otel.span_pruning` suppresses redundant phase and fixture spans
while retaining test protocol spans, session spans, outcomes, and exceptions.
This reduces trace size without multiplying traces or changing test attribution.
It trades away per-phase and fixture timing in the trace waterfall.

Validate the actual emitted spans with the installed pytest plugin: hook wrapping
and plugin registration order matter. Check that a failing test still has its
status and traceback on the retained protocol span. Verify a fresh large trace
can be read and shaped completely before reducing backend limits.

## Failure outcomes must survive incomplete test rows

Summing failing test spans alone is insufficient: losing the only error span can
make a failing job appear green. Preserve session-level failure evidence when
per-test rows are absent or incomplete. A partial trace may undercount failures;
it must not erase a known failed session outcome.

Regression coverage lives in `tests/test_span_pruning.py` and
`tests/test_trace_exporter.py`, including real pytest subprocesses and incomplete
trace scenarios. For a deployment check, compare the workflow's collected tests
and outcome with exported rows and verify trace-read error metrics remain stable.
