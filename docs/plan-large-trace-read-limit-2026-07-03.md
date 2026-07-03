# Plan: large shard traces exceed Tempo's read-path limit (tests_torch vanishes)

**Date:** 2026-07-03
**Status:** immediate mitigation shipped; source-side reduction pending

## Symptom

`tests_torch` (and, for large runs, whichever job produces the biggest shard
traces) is missing from the **Jobs** table on the by-run dashboard and from the
`/run` drill-down, even though the job ran and its traces are in Tempo.

Reproduced on run `28653280321:1`: GitHub ran all 8 `tests_torch` shards
(success); Tempo search/tag-values list `tests_torch`; but the Prometheus
`pytest_run_job_*` series and the persisted run store (`28653280321_1.json.gz`,
47,276 rows) both contain the same 14 jobs with **no `tests_torch`**.

## Root cause

The exporter turns a trace into rows by fetching it whole via
`GET /api/traces/{id}`. A `tests_torch` shard trace is **~36 MB / ~100k spans**
(post span-id-collision-fix — before the fix duplicate span_ids collapsed a
shard to ~6k spans, which stayed under the limit). Tempo's querier assembles the
whole trace and sends it to the query-frontend in **one gRPC message**, capped
by `server.grpc_server_max_send_msg_size` = **16 MiB** (the Tempo default). So:

```
GET /api/traces/{id}
→ HTTP 500  "response larger than the max (36329178 vs 16777216)"
```

`_fetch_trace_with_settled` catches the error and returns `None`, so the trace
is **never shaped**. Both the Prometheus roll-up (`extract_run_rollup_metrics`)
and the run store (`persist_settled_runs`) consume the shaped `extracted` set,
so the whole job disappears from both, identically — while the job's tiny setup
traces (1752 B) fetch fine but carry no test rows.

Ingestion was already raised to 64 MiB (`max_bytes_per_trace`,
`max_recv_msg_size_mib`, `max_request_body_size`); only the **read/gRPC-send**
path was left at 16 MiB.

**Blast radius:** ~27.6k trace-by-id HTTP 500s / 24h (~17% of all reads) — it
drops the biggest job from *every* large run, not one PR.

## Shipped mitigation (this change)

Reading a giant trace back turned out to require raising limits at **three**
layers — each one uncovered the next:

1. **Tempo read limit** — `deploy/helm/templates/tempo.yaml`. The 16 MiB cap was
   NOT `server.grpc_server_max_send_msg_size` (raised that too, needed for the
   frontend's receive side) but **`querier.frontend_worker.grpc_client_config.max_send_msg_size`**
   — the querier→frontend hop, whose send side defaults to 16 MiB while its recv
   already defaults to 100 MiB. Raised both to 64 MiB. Verified: the 36 MB trace
   that 500'd now returns 200. Requires a `tempo-0` restart (ConfigMap change
   alone does NOT roll a StatefulSet pod).
2. **Exporter HTTP timeout** — `values.yaml` `config.httpTimeout` 10s → 45s. A
   40–65 MB trace takes ~10s+ to assemble in-cluster; at 10s the fetch timed out
   right at the edge, so the trace would still be dropped (now as `timeout`).
3. **Exporter memory** — `env/private.yaml` limit 2Gi → 4Gi. Parsing a 40–65 MB
   trace costs several hundred MB; up to `fetchConcurrency` (4) parse at once on
   top of the ~1.4 GiB caches. At 2Gi a big run OOM-killed the exporter.
4. **Render cadence** — `values.yaml` `config.traceRefetchSeconds` 300s → 900s.
   Once the exporter actually downloaded the big traces, render jumped ~10s →
   ~300–330s, at/above the 300s refetch interval → the whole window re-fetches
   every cycle and render spirals. 900s sits above render, below lookback/2.
5. **Visibility** — `pytest_trace_exporter_trace_fetch_errors_total{reason=...}`
   (exporter) + a "Trace read failures — dropped jobs (too_large)" panel on the
   CI Health dashboard, cross-checked against
   `tempo_request_duration_seconds_count{route="api_traces_traceid",status_code=~"5.."}`.

### Caveat — the mitigation is a moving ceiling (this is the point)

Every layer above is a band-aid, and layer 4 shows the cost is already biting:
downloading + parsing 40–65 MB traces makes the exporter's render ~30× slower
(metrics now refresh every ~5 min) and holds far more RAM. tests_torch will keep
growing; at 60–100 MB we hit the limits again *and* Tempo/exporter RSS pressure
(Tempo already OOM-restarted at the 8Gi→16Gi bump). **The render-time blowup is
the signal that the durable fix below is now required, not optional.**

## Source-side reduction (pending — pick one)

Goal: no single trace should approach the read limit.

- **A. Smaller trace grain.** Stop grouping an entire shard (~100k spans) under
  one `trace_id`. One trace per xdist worker (or per test-file batch) keeps each
  trace ≪16 MiB and parallelizes reads. Cost: the "one trace = one shard" view
  in Grafana/Tempo splits; the exporter's per-run grouping already keys on
  `run.id` + `test.job`, so roll-ups are unaffected, but the trace/span view
  changes. **Recommended.**
- **B. Fewer spans per test.** pytest emits ~4 spans/test; a shard runs ~thousands
  of tests. Dropping redundant child spans (keep the test span + failure event)
  cuts trace size several-fold with no schema split. Complementary to A.
- **C. Don't fetch whole traces.** Shape rows from paged TraceQL span search
  instead of `GET /api/traces/{id}`, so the exporter never holds a multi-MB blob.
  Biggest rework; also fixes exporter RSS. Consider if A/B prove insufficient.

## Verification

- After deploy: `curl .../api/traces/<big tests_torch id>` returns 200.
- `pytest_trace_exporter_trace_fetch_errors_total{reason="too_large"}` stops
  climbing; the CI Health panel goes flat.
- A fresh large run shows `tests_torch` in the Jobs table and `/run` table.
