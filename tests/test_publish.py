from __future__ import annotations

import json

import pytest

from transformersci.publish import main, manifest, self_metrics, tables, tempo_window


# --- trace fixtures (mirrors tests/test_trace_exporter.py's shape) -----------


def _tag(key: str, value: str) -> dict[str, str]:
    return {"key": key, "value": value}


def _test_span(
    *,
    nodeid: str,
    start_time: int,
    duration: int,
    status_code: str = "UNSET",
    exception_type: str | None = None,
    exception_message: str = "",
    exception_stacktrace: str = "",
) -> dict:
    logs: list[dict[str, object]] = []
    if exception_type is not None:
        logs.append(
            {
                "fields": [
                    _tag("event", "exception"),
                    _tag("exception.type", exception_type),
                    _tag("exception.message", exception_message),
                    _tag("exception.stacktrace", exception_stacktrace),
                ]
            }
        )
    return {
        "duration": duration,
        "logs": logs,
        "operationName": nodeid,
        "processID": "p0",
        "startTime": start_time,
        "tags": [
            _tag("pytest.nodeid", nodeid),
            _tag("pytest.span_type", "test"),
            _tag("otel.status_code", status_code),
        ],
    }


def _trace(
    *,
    trace_id: str,
    run_id: str,
    job: str,
    spans: list[dict],
    commit_sha: str = "",
    repository: str = "",
) -> dict:
    tags = [
        _tag("transformers.test.provider", "github_actions"),
        _tag("transformers.test.run.id", run_id),
        _tag("transformers.test.job", job),
        _tag("vcs.change.id", "4321"),
    ]
    if commit_sha:
        tags.append(_tag("vcs.ref.head.revision", commit_sha))
    if repository:
        tags.append(_tag("vcs.repository.name", repository))
    return {
        "processes": {
            "p0": {
                "serviceName": "transformers-tests",
                "tags": tags,
            }
        },
        "spans": spans,
        "traceID": trace_id,
    }


def _sample_traces() -> list[dict]:
    return [
        _trace(
            trace_id="trace-a",
            run_id="run-1",
            job="single-gpu",
            spans=[
                _test_span(
                    nodeid="tests/models/bert/test_modeling_bert.py::T::test_ok",
                    start_time=1_000_000,
                    duration=2_000_000,
                ),
                _test_span(
                    nodeid="tests/models/bert/test_modeling_bert.py::T::test_boom",
                    start_time=4_000_000,
                    duration=1_000_000,
                    status_code="ERROR",
                    exception_type="AssertionError",
                    exception_message="assert 1 == 2",
                    exception_stacktrace="Traceback...\nE   assert 1 == 2",
                ),
            ],
        ),
        _trace(
            trace_id="trace-b",
            run_id="run-1",
            job="multi-gpu",
            spans=[
                _test_span(
                    nodeid="tests/test_cli.py::test_plain",
                    start_time=2_000_000,
                    duration=500_000,
                ),
            ],
        ),
    ]


def _sample_traces_with_commit() -> list[dict]:
    """Two job traces for one run, tagged with a head commit SHA + repo."""
    traces = _sample_traces()
    for trace in traces:
        trace["processes"]["p0"]["tags"].extend(
            [
                _tag("vcs.ref.head.revision", "cafef00d1234"),
                _tag("vcs.repository.name", "huggingface/transformers"),
            ]
        )
    return traces


# --- pure helpers ------------------------------------------------------------


def test_model_from_nodeid():
    assert (
        tables.model_from_nodeid("tests/models/bert/test_modeling_bert.py::T::t")
        == "bert"
    )
    assert tables.model_from_nodeid("tests/test_cli.py::t") == ""
    assert tables.model_from_nodeid("") == ""


def test_gpu_from_job():
    assert tables.gpu_from_job("multi-gpu") == "multi"
    assert tables.gpu_from_job("single-gpu") == "single"
    assert tables.gpu_from_job("lint") == ""


# --- row building ------------------------------------------------------------


def test_build_test_rows_shape_and_exceptions():
    rows = tables.build_test_rows(_sample_traces())
    assert len(rows) == 3
    by_node = {r["test_nodeid"]: r for r in rows}

    ok = by_node["tests/models/bert/test_modeling_bert.py::T::test_ok"]
    assert ok["model"] == "bert"
    assert ok["gpu"] == "single"
    assert ok["status_code"] == "UNSET"
    assert ok["date"] == "1970-01-01"  # start_time 1_000_000 micros

    boom = by_node["tests/models/bert/test_modeling_bert.py::T::test_boom"]
    assert boom["status_code"] == "ERROR"
    assert boom["exception_type"] == "AssertionError"
    # Full, untruncated message + stacktrace are published.
    assert boom["exception_message"] == "assert 1 == 2"
    assert "assert 1 == 2" in boom["exception_stacktrace"]

    plain = by_node["tests/test_cli.py::test_plain"]
    assert plain["model"] == ""
    assert plain["gpu"] == "multi"


def test_build_test_rows_dedups_by_trace_and_node():
    traces = _sample_traces()
    # Same trace twice should not duplicate rows.
    rows = tables.build_test_rows(traces + [traces[0]])
    assert len(rows) == 3


def test_build_run_rollups():
    rollups = tables.build_run_rollups(_sample_traces())
    # One run (run-1), two jobs -> two rollup rows.
    assert len(rollups) == 2
    by_job = {r["test_job"]: r for r in rollups}

    single = by_job["single-gpu"]
    assert single["total_tests"] == 2
    assert single["failed_tests"] == 1
    assert single["passed_tests"] == 1
    assert single["job_count"] == 2  # both jobs in the run
    assert single["run_id"] == "run-1"

    multi = by_job["multi-gpu"]
    assert multi["total_tests"] == 1
    assert multi["failed_tests"] == 0
    assert multi["job_count"] == 2

    # No commit SHA on these fixtures -> commit columns stay empty (no fetch).
    assert single["commit_sha"] == ""
    assert single["commit_message"] == ""


def test_run_rollups_carry_commit_message_resolved_once_per_run():
    calls: list[tuple[str, str]] = []

    def commit_fetcher(repository: str, sha: str) -> str:
        calls.append((repository, sha))
        return "Drop the legacy shim"

    acc = tables.RunRollupAccumulator(commit_message_fetcher=commit_fetcher)
    for trace in _sample_traces_with_commit():
        acc.add(trace)
    rollups = acc.rows()

    # Two job rows for the one run, both carrying the same run-level commit data.
    assert {r["test_job"] for r in rollups} == {"single-gpu", "multi-gpu"}
    for row in rollups:
        assert row["commit_sha"] == "cafef00d1234"
        assert row["commit_message"] == "Drop the legacy shim"
    # One run -> a single GitHub lookup despite multiple job traces.
    assert calls == [("huggingface/transformers", "cafef00d1234")]


def test_group_by_day():
    rows = tables.build_test_rows(_sample_traces())
    days = tables.group_by_day(rows)
    assert set(days) == {"1970-01-01"}
    assert len(days["1970-01-01"]) == 3


def test_group_traces_by_day_skips_empty():
    traces = _sample_traces()
    traces.append(_trace(trace_id="empty", run_id="r", job="j", spans=[]))
    grouped = tables.group_traces_by_day(traces)
    ids = {tid for items in grouped.values() for tid, _ in items}
    assert ids == {"trace-a", "trace-b"}  # the empty trace is dropped


# --- parquet + cycle (need pyarrow) ------------------------------------------


def test_write_parquet_roundtrip(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    rows = tables.build_test_rows(_sample_traces())
    path = tmp_path / "test_rows.parquet"
    n = tables.write_parquet(rows, tables.TEST_ROW_COLUMNS, path)
    assert n == 3
    table = pq.read_table(str(path))
    assert table.num_rows == 3
    assert table.column_names == tables.TEST_ROW_COLUMNS


def test_run_cycle_writes_partitions(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    monkeypatch.setattr(main, "iter_window_traces", lambda: iter(_sample_traces()))

    manifest_result = main.run_cycle(str(tmp_path), main.DEFAULT_BUCKET_URI)

    day_dir = tmp_path / "daily" / "1970-01-01"
    assert (day_dir / "test_rows.parquet").is_file()
    assert (day_dir / "run_rollups.parquet").is_file()
    assert (day_dir / "traces" / "trace-a.json").is_file()
    assert (day_dir / "traces" / "trace-b.json").is_file()
    assert (tmp_path / "README.md").is_file()

    view = json.loads((tmp_path / "current_view.json").read_text())
    assert view == manifest_result
    assert view["schema_version"] == tables.SCHEMA_VERSION
    assert view["partition_count"] == 1
    assert view["total_test_rows"] == 3
    assert view["partitions"][0]["traces"] == 2

    # Raw trace JSON is faithful.
    raw = json.loads((day_dir / "traces" / "trace-a.json").read_text())
    assert raw["traceID"] == "trace-a"


def test_main_dry_run_does_not_sync(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    monkeypatch.setattr(main, "iter_window_traces", lambda: iter(_sample_traces()))
    called = {"sync": False}
    monkeypatch.setattr(
        main, "sync_to_bucket", lambda *a, **k: called.__setitem__("sync", True)
    )

    rc = main.main(["--staging-dir", str(tmp_path), "--dry-run", "--sync"])
    assert rc == 0
    assert called["sync"] is False  # --dry-run overrides --sync


def test_build_manifest_empty(tmp_path):
    m = manifest.build_manifest(str(tmp_path))
    assert m["partition_count"] == 0
    assert m["partitions"] == []


def test_prune_old_trace_partitions(tmp_path):
    from datetime import datetime, timedelta, timezone

    today = datetime.now(tz=timezone.utc).date()
    old = (today - timedelta(days=5)).isoformat()
    recent = today.isoformat()
    daily = tmp_path / "daily"
    for day in (old, recent):
        traces = daily / day / "traces"
        traces.mkdir(parents=True)
        (traces / "a.json").write_text("{}", encoding="utf-8")
        (daily / day / "test_rows.parquet").write_text("x", encoding="utf-8")

    pruned = main.prune_old_trace_partitions(str(tmp_path), keep_days=2)

    assert pruned == 1
    # Old partition: raw traces gone, Parquet (and the partition dir) kept.
    assert not (daily / old / "traces").exists()
    assert (daily / old / "test_rows.parquet").exists()
    # Recent partition fully intact.
    assert (daily / recent / "traces" / "a.json").exists()


def test_main_prunes_old_traces_after_sync(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    monkeypatch.setattr(main, "iter_window_traces", lambda: iter(_sample_traces()))
    monkeypatch.setattr(main, "sync_to_bucket", lambda *a, **k: None)
    seen = {}
    monkeypatch.setattr(
        main,
        "prune_old_trace_partitions",
        lambda staging, keep: seen.update(staging=staging, keep=keep) or 0,
    )
    monkeypatch.setenv("PUBLISH_TRACE_RETENTION_DAYS", "3")

    rc = main.main(["--staging-dir", str(tmp_path), "--sync"])
    assert rc == 0
    assert seen == {"staging": str(tmp_path), "keep": 3}


# --- publisher search resilience ---------------------------------------------


def test_search_retries_then_succeeds(monkeypatch):
    monkeypatch.setenv("PUBLISH_SEARCH_RETRIES", "3")
    monkeypatch.setattr(tempo_window.time, "sleep", lambda *_: None)  # no real backoff
    calls = {"n": 0}

    def flaky(base_url, service_name, start, end, limit):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("timed out")
        return ["t1", "t2"]

    monkeypatch.setattr(tempo_window, "search_trace_ids", flaky)
    ids = tempo_window.window_trace_ids(
        base_url="http://tempo:3200",
        service_names=["svc"],
        window_seconds=3600,
        limit=10,
    )
    assert ids == ["t1", "t2"]
    assert calls["n"] == 3  # failed twice, succeeded on the third


def test_search_reraises_after_exhausting_retries(monkeypatch):
    monkeypatch.setenv("PUBLISH_SEARCH_RETRIES", "2")
    monkeypatch.setattr(tempo_window.time, "sleep", lambda *_: None)

    def always_fail(*a, **k):
        raise TimeoutError("timed out")

    monkeypatch.setattr(tempo_window, "search_trace_ids", always_fail)
    with pytest.raises(TimeoutError):
        tempo_window.window_trace_ids(
            base_url="http://tempo:3200",
            service_names=["svc"],
            window_seconds=3600,
            limit=10,
        )


# --- publisher self-metrics --------------------------------------------------


def test_render_self_metrics_emits_known_skips_unknown():
    body = self_metrics.render_self_metrics(
        {
            "ci_publisher_traces_published": 12,
            "ci_publisher_traces_per_second": 1.5,
            "not_a_real_metric": 99,  # unknown keys are dropped
        }
    )
    assert "# TYPE ci_publisher_traces_published gauge" in body
    assert "ci_publisher_traces_published 12" in body
    assert "ci_publisher_traces_per_second 1.5" in body
    assert "not_a_real_metric" not in body
    assert body.endswith("\n")


def test_render_self_metrics_full_precision_no_scientific():
    # A unix timestamp must render at full precision, not %g's 6-sig-fig
    # "1.78117e+09" (which would break time() - last_run on the dashboard).
    body = self_metrics.render_self_metrics(
        {"ci_publisher_last_run_timestamp_seconds": 1781170123.456}
    )
    assert "ci_publisher_last_run_timestamp_seconds 1781170123.456" in body
    assert "e+" not in body
    # Whole numbers render as plain integers, not floats.
    assert "ci_publisher_peak_rss_bytes 27205632" in self_metrics.render_self_metrics(
        {"ci_publisher_peak_rss_bytes": 27205632}
    )


def test_write_self_metrics_atomic_and_world_readable(tmp_path):
    target = tmp_path / "ci_publisher.prom"
    self_metrics.write_self_metrics({"ci_publisher_up": 1}, path=target)
    assert "ci_publisher_up 1" in target.read_text()
    # No half-written .tmp siblings left behind.
    assert [p.name for p in tmp_path.iterdir()] == ["ci_publisher.prom"]
    # node-exporter (unprivileged) must be able to read it.
    assert (target.stat().st_mode & 0o777) == 0o644


def test_write_self_metrics_best_effort_on_bad_path(tmp_path):
    # A missing directory must not raise — metrics never break a publish.
    self_metrics.write_self_metrics(
        {"ci_publisher_up": 1}, path=tmp_path / "missing" / "x.prom"
    )


def test_main_writes_self_metrics(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    monkeypatch.setattr(main, "iter_window_traces", lambda: iter(_sample_traces()))
    metrics_file = tmp_path / "ci_publisher.prom"
    monkeypatch.setenv("PUBLISH_METRICS_FILE", str(metrics_file))

    rc = main.main(["--staging-dir", str(tmp_path), "--dry-run"])
    assert rc == 0

    body = metrics_file.read_text()
    assert "ci_publisher_last_run_success 1" in body
    assert "ci_publisher_traces_published 2" in body
    assert "ci_publisher_test_rows_published 3" in body
    assert "ci_publisher_dataset_bytes " in body
    assert "ci_publisher_peak_rss_bytes " in body
