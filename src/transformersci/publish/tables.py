"""Shape Tempo traces into the published dataset's tables.

Two tables, both daily-partitioned (partition = UTC day of the trace's start):

* ``test_rows``    — one row per (trace_id, test_nodeid)
* ``run_rollups``  — one row per (run_id, test_job)

Per the "include everything" publishing decision the per-test rows carry the
*full*, untruncated ``exception_message`` and ``exception_stacktrace`` (pulled
via :func:`extract_failure_details`, not the metric-capped
:func:`extract_exception_info`). The raw trace JSON is published alongside the
Parquet (see :func:`write_raw_traces`) so consumers can re-derive anything.

Pure-python row building lives here with no hard dependency on ``pyarrow`` —
only :func:`write_parquet` imports it, lazily, so the shaping logic stays unit
-testable without the heavy dep.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..otel.trace_exporter import (
    extract_failure_details,
    extract_trace_rows,
    fetch_github_commit_message_cached,
    repository_from_pr_url,
)

SCHEMA_VERSION = 1

# test_rows columns, in published order. Kept explicit (and additive-only across
# schema versions) so the Parquet schema is stable for downstream consumers.
TEST_ROW_COLUMNS = [
    "ts",
    "date",
    "service_name",
    "provider",
    "pr",
    "run_id",
    "trace_id",
    "test_job",
    "test_nodeid",
    "test_module",
    "test_class",
    "test_function",
    "test_line",
    "model",
    "gpu",
    "status_code",
    "duration_seconds",
    "exception_type",
    "exception_message",
    "exception_stacktrace",
]

RUN_ROLLUP_COLUMNS = [
    "date",
    "service_name",
    "provider",
    "pr",
    "run_id",
    "test_job",
    "total_tests",
    "passed_tests",
    "failed_tests",
    "duration_seconds",
    "start_time",
    "end_time",
    "job_count",
    "commit_sha",
    "commit_message",
]


def model_from_nodeid(test_nodeid: str) -> str:
    """Derive the model under test from a ``tests/models/<model>/...`` nodeid.

    Returns "" for tests that don't live under ``tests/models`` (the dataset's
    convention for "not a per-model test").
    """
    path = (test_nodeid or "").split("::", 1)[0]
    parts = path.split("/")
    if len(parts) >= 3 and parts[0] == "tests" and parts[1] == "models":
        return parts[2]
    return ""


def gpu_from_job(test_job: str) -> str:
    """Derive the GPU dimension (single/multi) from the job name.

    The data model has no dedicated GPU attribute (see the trace schema notes);
    CI job names encode it, e.g. ``single-gpu`` / ``multi-gpu``.
    """
    job = (test_job or "").lower()
    if "multi" in job:
        return "multi"
    if "single" in job:
        return "single"
    return ""


def _micros_to_epoch_seconds(micros: object) -> int:
    try:
        return int(float(str(micros)) / 1_000_000)
    except (TypeError, ValueError):
        return 0


def _epoch_to_date(epoch_seconds: int) -> str:
    if not epoch_seconds:
        return "unknown"
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).strftime("%Y-%m-%d")


def shape_trace_rows(trace: dict) -> list[dict]:
    """Build the per-test rows for a single trace (deduped by test_nodeid).

    Pulled out of :func:`build_test_rows` so the publisher can shape one trace
    at a time and never hold the whole window of (large) traces in memory.
    """
    trace_info, rows = extract_trace_rows(trace)
    if not rows:
        return []
    ts = _micros_to_epoch_seconds(trace_info.get("start_time", 0))
    date = _epoch_to_date(ts)

    # nodeid -> full (untruncated) exception message + stacktrace
    failures = {d["test_nodeid"]: d for d in extract_failure_details(trace)}

    by_node: dict[str, dict] = {}
    for row in rows:
        nodeid = str(row.get("test_nodeid", ""))
        detail = failures.get(nodeid, {})
        by_node[nodeid] = {
            "ts": ts,
            "date": date,
            "service_name": str(row.get("service_name", "unknown")),
            "provider": str(row.get("provider", "unknown")),
            "pr": str(row.get("pr", "none")),
            "run_id": str(row.get("run_id", "")),
            "trace_id": str(row.get("trace_id", "")),
            "test_job": str(row.get("test_job", "unknown")),
            "test_nodeid": nodeid,
            "test_module": str(row.get("test_module", "")),
            "test_class": str(row.get("test_class", "")),
            "test_function": str(row.get("test_function", "")),
            "test_line": str(row.get("test_line", "")),
            "model": model_from_nodeid(nodeid),
            "gpu": gpu_from_job(str(row.get("test_job", ""))),
            "status_code": str(row.get("status_code", "UNSET")),
            "duration_seconds": float(row.get("duration_seconds", 0.0)),
            "exception_type": str(
                detail.get("exception_type", row.get("exception_type", ""))
            ),
            # Full, untruncated message + traceback (the metric path caps
            # the stacktrace at 4000 chars; we publish everything).
            "exception_message": str(detail.get("exception_message", "")),
            "exception_stacktrace": str(
                detail.get("exception_stacktrace", "")
                or row.get("exception_stacktrace", "")
            ),
        }
    return list(by_node.values())


def build_test_rows(traces: list[dict]) -> list[dict]:
    """Build deduped per-test rows from Jaeger-shaped traces.

    Deduped by (trace_id, test_nodeid); later traces win, which keeps the most
    complete copy when the same settled trace appears across overlapping
    windows. (The streaming publisher uses :func:`shape_trace_rows` directly.)
    """
    by_key: dict[tuple[str, str], dict] = {}
    for trace in traces:
        for record in shape_trace_rows(trace):
            by_key[(record["trace_id"], record["test_nodeid"])] = record
    return list(by_key.values())


class RunRollupAccumulator:
    """Incrementally aggregate traces into one rollup row per (run_id, test_job).

    Folding traces in one at a time (``add``) keeps the publisher's memory flat
    regardless of window size — only the small aggregates are retained, not the
    traces. ``job_count`` (distinct jobs that contributed tests to the run) is
    resolved at the end in :meth:`rows`.
    """

    def __init__(self, commit_message_fetcher=None) -> None:
        # (service, provider, pr, run_id, test_job) -> aggregate
        self._job_agg: dict[tuple[str, str, str, str, str], dict] = {}
        self._run_jobs: dict[tuple[str, str, str, str], set[str]] = {}
        # (service, provider, pr, run_id) -> {"commit_sha", "repository"}; commit
        # metadata is run-level so it's resolved to a message once in `rows`.
        self._run_commit: dict[tuple[str, str, str, str], dict[str, str]] = {}
        self._commit_message_fetcher = (
            commit_message_fetcher or fetch_github_commit_message_cached
        )

    def add(self, trace: dict) -> None:
        trace_info, rows = extract_trace_rows(trace)
        if not rows:
            return
        service = str(trace_info.get("service_name", "unknown"))
        provider = str(trace_info.get("provider", "unknown"))
        pr = str(trace_info.get("pr", "none"))
        run_id = str(trace_info.get("run_id", trace_info.get("trace_id", "")))
        job = str(trace_info.get("test_job", "unknown"))
        start = _micros_to_epoch_seconds(trace_info.get("start_time", 0))
        end = _micros_to_epoch_seconds(trace_info.get("end_time", 0))

        total = len(rows)
        failed = sum(1 for r in rows if str(r.get("status_code")) == "ERROR")
        duration = sum(float(r.get("duration_seconds", 0.0)) for r in rows)

        run_key = (service, provider, pr, run_id)
        self._run_jobs.setdefault(run_key, set()).add(job)

        commit_sha = str(trace_info.get("commit_sha", ""))
        if commit_sha:
            existing = self._run_commit.get(run_key)
            if existing is None or not existing.get("commit_sha"):
                repository = str(trace_info.get("repository", ""))
                if not repository:
                    repository = repository_from_pr_url(
                        str(trace_info.get("pr_url", ""))
                    )
                self._run_commit[run_key] = {
                    "commit_sha": commit_sha,
                    "repository": repository,
                }

        key = (service, provider, pr, run_id, job)
        agg = self._job_agg.get(key)
        if agg is None:
            agg = {
                "date": _epoch_to_date(start),
                "service_name": service,
                "provider": provider,
                "pr": pr,
                "run_id": run_id,
                "test_job": job,
                "total_tests": 0,
                "failed_tests": 0,
                "duration_seconds": 0.0,
                "start_time": start,
                "end_time": end,
            }
            self._job_agg[key] = agg
        agg["total_tests"] += total
        agg["failed_tests"] += failed
        agg["duration_seconds"] += duration
        if start and (agg["start_time"] == 0 or start < agg["start_time"]):
            agg["start_time"] = start
            agg["date"] = _epoch_to_date(start)
        agg["end_time"] = max(agg["end_time"], end)

    def rows(self) -> list[dict]:
        # Resolve each run's commit subject once (cached + deduped by run) so the
        # GitHub lookup doesn't repeat per job row.
        commit_messages: dict[tuple[str, str, str, str], str] = {}
        for run_key, commit in self._run_commit.items():
            sha = commit.get("commit_sha", "")
            repository = commit.get("repository", "")
            commit_messages[run_key] = (
                self._commit_message_fetcher(repository, sha)
                if sha and repository
                else ""
            )

        rollups: list[dict] = []
        for agg in self._job_agg.values():
            agg = dict(agg)
            agg["passed_tests"] = agg["total_tests"] - agg["failed_tests"]
            run_key = (agg["service_name"], agg["provider"], agg["pr"], agg["run_id"])
            agg["job_count"] = len(self._run_jobs.get(run_key, set()))
            commit = self._run_commit.get(run_key, {})
            agg["commit_sha"] = commit.get("commit_sha", "")
            agg["commit_message"] = commit_messages.get(run_key, "")
            rollups.append({col: agg[col] for col in RUN_ROLLUP_COLUMNS})
        return rollups


def build_run_rollups(traces: list[dict]) -> list[dict]:
    """Aggregate traces into one row per (run_id, test_job).

    ``job_count`` is the number of distinct jobs that contributed tests to the
    *run* (carried onto every job row of that run), mirroring the exporter's
    run-level rollup semantics. (The streaming publisher uses
    :class:`RunRollupAccumulator` directly.)
    """
    acc = RunRollupAccumulator()
    for trace in traces:
        acc.add(trace)
    return acc.rows()


def group_by_day(rows: list[dict]) -> dict[str, list[dict]]:
    """Bucket rows by their ``date`` column."""
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(str(row.get("date", "unknown")), []).append(row)
    return buckets


def group_traces_by_day(traces: list[dict]) -> dict[str, list[tuple[str, dict]]]:
    """Bucket raw traces by UTC day -> list of (trace_id, trace).

    Uses the same start-time/day derivation as the test rows so the raw JSON
    lands in the same partition as the rows derived from it. Traces with no
    test spans are dropped (nothing to publish for them).
    """
    buckets: dict[str, list[tuple[str, dict]]] = {}
    for trace in traces:
        trace_info, rows = extract_trace_rows(trace)
        if not rows:
            continue
        date = _epoch_to_date(_micros_to_epoch_seconds(trace_info.get("start_time", 0)))
        trace_id = str(trace_info.get("trace_id") or trace.get("traceID") or "")
        if not trace_id:
            continue
        buckets.setdefault(date, []).append((trace_id, trace))
    return buckets


def write_parquet(rows: list[dict], columns: list[str], path) -> int:
    """Write ``rows`` to ``path`` as Parquet with an explicit column order.

    Returns the number of rows written. Imports ``pyarrow`` lazily so this
    module can be imported (and its pure shaping logic tested) without it.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({col: [row.get(col) for row in rows] for col in columns})
    pq.write_table(table, str(path))
    return len(rows)
