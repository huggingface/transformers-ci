from __future__ import annotations

import io
import json
import urllib.parse
from unittest.mock import patch

import pytest

from transformersci.otel import trace_exporter


def _shaped_iter(*traces: dict):
    """Mimic :func:`_iter_window_shaped` for render tests.

    Shapes each raw (Jaeger) trace and yields ``(trace_info, rows, is_new=True)``
    — the contract the render now consumes — so a test can drive the render with
    fixed traces without standing up the Tempo search/fetch path.
    """
    shaped = [(*trace_exporter.extract_trace_rows(t), True) for t in traces]

    def _gen(*_args, **_kwargs):
        # Reset carryover so per-test emission is deterministic across tests.
        trace_exporter._previous_new_ids = set()
        return iter(list(shaped))

    return _gen


def make_tag(key: str, value: str) -> dict[str, str]:
    return {"key": key, "value": value}


def make_test_span(
    *,
    process_id: str,
    nodeid: str,
    start_time: int,
    duration: int,
    status_code: str = "UNSET",
    exception_type: str | None = None,
) -> dict:
    logs: list[dict[str, object]] = []
    if exception_type is not None:
        logs.append(
            {
                "fields": [
                    make_tag("event", "exception"),
                    make_tag("exception.type", exception_type),
                    make_tag(
                        "exception.stacktrace",
                        f"Traceback for {nodeid}",
                    ),
                ]
            }
        )

    return {
        "duration": duration,
        "logs": logs,
        "operationName": nodeid,
        "processID": process_id,
        "startTime": start_time,
        "tags": [
            make_tag("pytest.nodeid", nodeid),
            make_tag("pytest.span_type", "test"),
            make_tag("otel.status_code", status_code),
        ],
    }


def make_trace(
    *,
    trace_id: str,
    run_id: str,
    job: str,
    spans: list[dict],
    pr: str = "4321",
    pr_url: str = "https://github.com/huggingface/transformers/pull/4321",
    provider: str = "github_actions",
    repository: str = "huggingface/transformers",
    service_name: str = "transformers-tests",
    job_tag_key: str = "transformers.test.job",
    commit_sha: str = "",
) -> dict:
    tags = [
        make_tag("transformers.test.provider", provider),
        make_tag("transformers.test.run.id", run_id),
        make_tag(job_tag_key, job),
        make_tag("vcs.change.id", pr),
        make_tag("vcs.change.url", pr_url),
        make_tag("vcs.repository.name", repository),
    ]
    if commit_sha:
        tags.append(make_tag("vcs.ref.head.revision", commit_sha))
    return {
        "processes": {
            "pytest-process": {
                "serviceName": service_name,
                "tags": tags,
            }
        },
        "spans": spans,
        "traceID": trace_id,
    }


def workflow_split_across_three_jobs() -> list[dict]:
    process_id = "pytest-process"
    return [
        make_trace(
            trace_id="trace-torch",
            run_id="12345:2",
            job="tests_torch",
            spans=[
                make_test_span(
                    process_id=process_id,
                    nodeid="tests/test_torch.py::TestTorch::test_pass",
                    start_time=1_000_000,
                    duration=2_000_000,
                ),
                make_test_span(
                    process_id=process_id,
                    nodeid="tests/test_torch.py::TestTorch::test_fail",
                    start_time=4_000_000,
                    duration=1_000_000,
                    status_code="ERROR",
                    exception_type="AssertionError",
                ),
            ],
        ),
        make_trace(
            trace_id="trace-tf",
            run_id="12345:2",
            job="tests_tf",
            spans=[
                make_test_span(
                    process_id=process_id,
                    nodeid="tests/test_tf.py::TestTf::test_pass",
                    start_time=2_000_000,
                    duration=3_000_000,
                )
            ],
        ),
        make_trace(
            trace_id="trace-flax",
            run_id="12345:2",
            job="tests_flax",
            spans=[
                make_test_span(
                    process_id=process_id,
                    nodeid="tests/test_flax.py::TestFlax::test_pass",
                    start_time=3_000_000,
                    duration=1_500_000,
                )
            ],
        ),
    ]


def metric_lines(metrics: list[str], metric_name: str) -> list[str]:
    prefix = f"{metric_name}{{"
    return [line for line in metrics if line.startswith(prefix)]


def test_extract_per_run_metrics_aggregates_job_traces_into_one_run() -> None:
    metrics = trace_exporter.extract_per_run_metrics(workflow_split_across_three_jobs())

    run_start_lines = metric_lines(metrics, "pytest_run_start_time_seconds")
    assert len(run_start_lines) == 1
    assert 'run_id="12345:2"' in run_start_lines[0]
    # Run-level totals are values on their own metrics now, NOT labels on the
    # start-time series (see comment in extract_run_rollup_metrics). The
    # start-time series carries only the stable run identity.
    for mutable_label in (
        "job_count=",
        "jobs=",
        "trace_count=",
        "total_tests=",
        "failed_tests=",
        "total_duration_seconds=",
        "failure_rate_percent=",
        "success_rate_percent=",
    ):
        assert mutable_label not in run_start_lines[0]
    assert run_start_lines[0].endswith(" 1.000000")

    job_count_lines = metric_lines(metrics, "pytest_run_job_count")
    assert len(job_count_lines) == 1
    assert 'run_id="12345:2"' in job_count_lines[0]
    assert job_count_lines[0].endswith(" 3")

    runner_execution_lines = metric_lines(
        trace_exporter.extract_ci_runner_execution_metrics(
            workflow_split_across_three_jobs()
        ),
        "pytest_ci_runner_execution_info",
    )
    assert len(runner_execution_lines) == 3
    assert any('trace_id="trace-torch"' in line for line in runner_execution_lines)
    assert any('trace_id="trace-tf"' in line for line in runner_execution_lines)
    assert any('trace_id="trace-flax"' in line for line in runner_execution_lines)

    run_end_lines = metric_lines(metrics, "pytest_run_end_time_seconds")
    assert len(run_end_lines) == 1
    assert 'run_id="12345:2"' in run_end_lines[0]
    assert run_end_lines[0].endswith(" 5.000000")

    total_test_lines = metric_lines(metrics, "pytest_run_total_tests")
    assert len(total_test_lines) == 1
    assert total_test_lines[0].endswith(" 4")

    failed_test_lines = metric_lines(metrics, "pytest_run_failed_tests")
    assert len(failed_test_lines) == 1
    assert failed_test_lines[0].endswith(" 1")

    job_member_lines = metric_lines(metrics, "pytest_run_job_member_info")
    assert len(job_member_lines) == 3
    assert any('test_job="tests_torch"' in line for line in job_member_lines)
    assert any('test_job="tests_tf"' in line for line in job_member_lines)
    assert any('test_job="tests_flax"' in line for line in job_member_lines)

    job_total_lines = metric_lines(metrics, "pytest_run_job_total_tests")
    assert len(job_total_lines) == 3
    assert any(
        'test_job="tests_torch"' in line and line.endswith(" 2")
        for line in job_total_lines
    )

    job_passed_lines = metric_lines(metrics, "pytest_run_job_passed_tests")
    assert len(job_passed_lines) == 3
    assert any(
        'test_job="tests_torch"' in line and line.endswith(" 1")
        for line in job_passed_lines
    )

    job_failed_lines = metric_lines(metrics, "pytest_run_job_failed_tests")
    assert len(job_failed_lines) == 3
    assert any(
        'test_job="tests_torch"' in line and line.endswith(" 1")
        for line in job_failed_lines
    )

    job_duration_lines = metric_lines(metrics, "pytest_run_job_duration_seconds")
    assert len(job_duration_lines) == 3
    assert any(
        'test_job="tests_torch"' in line and line.endswith(" 3.000000")
        for line in job_duration_lines
    )

    duration_lines = metric_lines(metrics, "pytest_test_duration_seconds")
    assert len(duration_lines) == 4
    # run_id / trace_id / pr are intentionally NOT labels here: keying the
    # per-test metric by run blew Prometheus cardinality up to ~2M series and
    # OOM-killed it. The series is now keyed by the test (with test_job kept),
    # and per-run drill-down is served from the trace via the /run endpoint.
    assert all("run_id=" not in line for line in duration_lines)
    assert all("trace_id=" not in line for line in duration_lines)
    assert all('pr="' not in line for line in duration_lines)
    assert any('test_job="tests_torch"' in line for line in duration_lines)
    assert any('test_job="tests_tf"' in line for line in duration_lines)
    assert any('test_job="tests_flax"' in line for line in duration_lines)


def test_per_test_duration_dedups_same_test_across_runs() -> None:
    """The per-test metric is no longer keyed by run, so the same test seen in
    two runs within the window must collapse to ONE series (else Prometheus
    rejects the duplicate sample), keeping the most-recently-started run's
    value."""
    process_id = "pytest-process"
    older = make_trace(
        trace_id="trace-old",
        run_id="run-1",
        job="tests_torch",
        spans=[
            make_test_span(
                process_id=process_id,
                nodeid="tests/test_torch.py::T::test_x",
                start_time=1_000_000,
                duration=2_000_000,
            )
        ],
    )
    newer = make_trace(
        trace_id="trace-new",
        run_id="run-2",
        job="tests_torch",
        spans=[
            make_test_span(
                process_id=process_id,
                nodeid="tests/test_torch.py::T::test_x",
                start_time=9_000_000,
                duration=5_000_000,
            )
        ],
    )
    lines = metric_lines(
        trace_exporter.extract_per_test_duration_metrics([older, newer]),
        "pytest_test_duration_seconds",
    )
    assert len(lines) == 1
    assert "run_id=" not in lines[0] and "trace_id=" not in lines[0]
    assert lines[0].endswith(" 5.000000000")  # newer run wins


def test_render_run_html_sorts_filters_and_links() -> None:
    rows = [
        {
            "test_nodeid": "tests/test_a.py::A::test_slow",
            "test_job": "tests_torch",
            "status_code": "OK",
            "trace_id": "tr1",
            "pr": "4321",
            "duration_seconds": 9.5,
        },
        {
            "test_nodeid": "tests/test_a.py::A::test_boom",
            "test_job": "tests_torch",
            "status_code": "ERROR",
            "trace_id": "tr1",
            "pr": "4321",
            "duration_seconds": 0.2,
        },
        {
            "test_nodeid": "tests/test_b.py::B::test_other",
            "test_job": "tests_tf",
            "status_code": "OK",
            "trace_id": "tr2",
            "pr": "4321",
            "duration_seconds": 1.0,
        },
    ]
    html_out = trace_exporter.render_run_html("123:1", rows)
    # sorted by duration desc
    assert (
        html_out.index("test_slow")
        < html_out.index("test_other")
        < html_out.index("test_boom")
    )
    assert "FAIL" in html_out
    # link to the per-test page carries the context the page now needs via URL
    assert "/d/pytest-test/test" in html_out
    assert "var-trace_id=tr1" in html_out
    assert "var-run_id=123%3A1" in html_out
    assert "var-pr=4321" in html_out
    assert "var-gh_run_id=123" in html_out  # leading digits of run id

    # job filter
    only_tf = trace_exporter.render_run_html("123:1", rows, job="tests_tf")
    assert "test_other" in only_tf and "test_slow" not in only_tf
    # status filter (Failing); ".+" sentinel means no filter
    failing = trace_exporter.render_run_html("123:1", rows, status="ERROR")
    assert "test_boom" in failing and "test_slow" not in failing
    allrows = trace_exporter.render_run_html("123:1", rows, status=".+")
    assert "test_slow" in allrows and "test_boom" in allrows


def test_gather_run_test_rows_from_membership(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run in the in-memory membership map is reconstructed from the trace
    cache without any Tempo network call."""
    trace = make_trace(
        trace_id="trace-mem",
        run_id="run-mem",
        job="tests_torch",
        spans=[
            make_test_span(
                process_id="pytest-process",
                nodeid="tests/test_torch.py::T::test_x",
                start_time=1_000_000,
                duration=2_000_000,
            )
        ],
    )
    monkeypatch.setitem(trace_exporter._run_members, "run-mem", {"trace-mem"})
    monkeypatch.setitem(trace_exporter._trace_cache, "trace-mem", trace)
    # Any Tempo fetch would be a bug for a cached run.
    monkeypatch.setattr(
        trace_exporter, "get_trace", lambda *a, **k: pytest.fail("should not fetch")
    )
    rows = trace_exporter.gather_run_test_rows("run-mem", base_url="http://unused")
    assert len(rows) == 1
    assert rows[0]["test_nodeid"] == "tests/test_torch.py::T::test_x"


def test_run_store_roundtrip_and_prune(tmp_path) -> None:
    d = str(tmp_path)
    rows = [
        {
            "test_nodeid": "t.py::A::test_x",
            "test_job": "tests_torch",
            "status_code": "ERROR",
            "duration_seconds": 1.5,
            "trace_id": "tr1",
            "pr": "42",
            "extra": "dropped",  # non-slim field should not be stored
        }
    ]
    trace_exporter.persist_run_rows("99:1", rows, directory=d)
    loaded = trace_exporter.load_run_rows("99:1", directory=d)
    assert loaded is not None and len(loaded) == 1
    assert loaded[0]["test_nodeid"] == "t.py::A::test_x"
    assert loaded[0]["status_code"] == "ERROR"
    assert "extra" not in loaded[0]  # only slim fields persisted
    assert trace_exporter.load_run_rows("nope", directory=d) is None
    # prune removes nothing when fresh, everything when max age is 0
    assert trace_exporter.prune_run_store(3600, directory=d) == 0
    assert trace_exporter.prune_run_store(0, directory=d) == 1
    assert trace_exporter.load_run_rows("99:1", directory=d) is None


def test_persist_run_rows_merges_shards_across_renders(tmp_path) -> None:
    """A big run's shards rotate through the window over many renders; persist
    must UNION them (keyed by trace_id+test_nodeid), not overwrite — otherwise
    the store is capped at one render's partial view."""
    d = str(tmp_path)
    shard1 = [
        {
            "test_nodeid": "a",
            "test_job": "j",
            "status_code": "OK",
            "duration_seconds": 1.0,
            "trace_id": "shard1",
            "pr": "1",
        },
    ]
    shard2 = [
        {
            "test_nodeid": "b",
            "test_job": "j",
            "status_code": "ERROR",
            "duration_seconds": 2.0,
            "trace_id": "shard2",
            "pr": "1",
        },
    ]
    trace_exporter.persist_run_rows("7:1", shard1, directory=d)
    trace_exporter.persist_run_rows("7:1", shard2, directory=d)  # later render
    loaded = trace_exporter.load_run_rows("7:1", directory=d)
    nodeids = sorted(r["test_nodeid"] for r in loaded)
    assert nodeids == ["a", "b"]  # both shards accumulated
    # re-persisting an already-seen shard does not duplicate
    trace_exporter.persist_run_rows("7:1", shard1, directory=d)
    assert len(trace_exporter.load_run_rows("7:1", directory=d)) == 2


def test_persist_settled_runs_groups_by_run(tmp_path) -> None:
    d = str(tmp_path)
    extracted = [
        (
            {"run_id": "100:1"},
            [
                {
                    "test_nodeid": "a",
                    "test_job": "j1",
                    "status_code": "OK",
                    "duration_seconds": 1.0,
                    "trace_id": "t1",
                    "pr": "1",
                }
            ],
        ),
        (
            {"run_id": "100:1"},
            [
                {
                    "test_nodeid": "b",
                    "test_job": "j2",
                    "status_code": "ERROR",
                    "duration_seconds": 2.0,
                    "trace_id": "t2",
                    "pr": "1",
                }
            ],
        ),
    ]
    trace_exporter.persist_settled_runs(extracted, directory=d)
    loaded = trace_exporter.load_run_rows("100:1", directory=d)
    assert loaded is not None and len(loaded) == 2  # both traces' rows merged


def test_gather_run_test_rows_prefers_store(tmp_path, monkeypatch) -> None:
    d = str(tmp_path)
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_RUN_STORE", d)
    monkeypatch.setattr(trace_exporter, "_run_rows_cache", trace_exporter.OrderedDict())
    trace_exporter.persist_run_rows(
        "55:1",
        [
            {
                "test_nodeid": "x",
                "test_job": "j",
                "status_code": "OK",
                "duration_seconds": 1.0,
                "trace_id": "t",
                "pr": "1",
            }
        ],
        directory=d,
    )
    monkeypatch.setattr(
        trace_exporter,
        "_search_run_trace_ids",
        lambda *a, **k: pytest.fail("should not search when persisted"),
    )
    rows = trace_exporter.gather_run_test_rows("55:1", base_url="http://unused")
    assert len(rows) == 1 and rows[0]["test_nodeid"] == "x"


def test_run_store_disabled_is_noop(tmp_path) -> None:
    assert trace_exporter.load_run_rows("1:1", directory="") is None
    trace_exporter.persist_run_rows("1:1", [{"test_nodeid": "x"}], directory="")
    assert trace_exporter.prune_run_store(0, directory="") == 0


def test_extract_pr_last_failure_metrics_links_failure_back_to_run() -> None:
    metrics = trace_exporter.extract_pr_last_failure_metrics(
        workflow_split_across_three_jobs()
    )

    failure_lines = metric_lines(metrics, "pytest_pr_last_failure_info")
    assert len(failure_lines) == 1
    assert 'pr="4321"' in failure_lines[0]
    assert 'run_id="12345:2"' in failure_lines[0]
    assert 'trace_id="trace-torch"' in failure_lines[0]
    assert 'test_job="tests_torch"' in failure_lines[0]


def test_extract_trace_rows_falls_back_to_branch_name_when_no_pr() -> None:
    """Push events to main carry no vcs.change.id; the branch name from
    vcs.ref.head.name takes its place so main-branch data doesn't collapse into
    a single pr="none" bucket."""
    trace = {
        "processes": {
            "pytest-process": {
                "serviceName": "transformers-tests",
                "tags": [
                    make_tag("transformers.test.provider", "github_actions"),
                    make_tag("transformers.test.run.id", "run-main"),
                    make_tag("transformers.test.job", "tests_torch"),
                    make_tag("vcs.ref.head.name", "main"),
                    make_tag("vcs.repository.name", "huggingface/transformers"),
                ],
            }
        },
        "spans": [
            make_test_span(
                process_id="pytest-process",
                nodeid="tests/test_main.py::TestMain::test_one",
                start_time=1_000_000,
                duration=1_000_000,
            )
        ],
        "traceID": "trace-main",
    }
    trace_info, rows = trace_exporter.extract_trace_rows(trace)
    assert trace_info["pr"] == "main"
    assert rows[0]["pr"] == "main"


def test_extract_trace_rows_falls_back_to_legacy_suite_tag() -> None:
    """A trace still using the legacy `transformers.test.suite` process tag is
    surfaced as `test_job` in the resulting rows, so dashboards keep working
    while emitters migrate to the new attribute name."""
    legacy_trace = make_trace(
        trace_id="trace-legacy",
        run_id="run-legacy",
        job="legacy_job_name",
        job_tag_key="transformers.test.suite",
        spans=[
            make_test_span(
                process_id="pytest-process",
                nodeid="tests/test_legacy.py::TestLegacy::test_one",
                start_time=1_000_000,
                duration=1_000_000,
            )
        ],
    )
    trace_info, rows = trace_exporter.extract_trace_rows(legacy_trace)
    assert trace_info["test_job"] == "legacy_job_name"
    assert rows[0]["test_job"] == "legacy_job_name"


class FakeResponse(io.StringIO):
    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


def test_fetch_github_pr_info_uses_github_api_response() -> None:
    pr_payload = FakeResponse(
        '{"html_url": "https://github.com/huggingface/transformers/pull/4321", '
        '"state": "open", "title": "Fix dashboard metadata", '
        '"user": {"login": "octocat"}, '
        '"head": {"sha": "deadbeefcafebabe1234567890abcdef00000000"}, '
        '"created_at": "2024-01-02T03:04:05Z", '
        '"requested_reviewers": [{"login": "alice"}, {"login": "bob"}]}'
    )
    reviews_payload = FakeResponse(
        '[{"user": {"login": "carol"}}, {"user": {"login": "alice"}}]'
    )
    with patch.dict(
        "os.environ",
        {
            "PYTEST_GITHUB_API_URL": "https://api.github.example",
            "PYTEST_GITHUB_TOKEN": "secret-token",
        },
        clear=False,
    ):
        with patch(
            "transformersci.otel.trace_exporter.urlopen",
            side_effect=[pr_payload, reviews_payload],
        ) as mocked_urlopen:
            metadata = trace_exporter.fetch_github_pr_info(
                "huggingface/transformers", "4321"
            )

    first_request = mocked_urlopen.call_args_list[0].args[0]
    second_request = mocked_urlopen.call_args_list[1].args[0]
    assert first_request.full_url == (
        "https://api.github.example/repos/huggingface/transformers/pulls/4321"
    )
    assert second_request.full_url == (
        "https://api.github.example/repos/huggingface/transformers/pulls/4321/reviews"
    )
    assert first_request.get_header("Authorization") == "Bearer secret-token"
    assert metadata == {
        "author": "octocat",
        "commit_sha": "deadbeefcafebabe1234567890abcdef00000000",
        "created_at": "2024-01-02T03:04:05Z",
        "html_url": "https://github.com/huggingface/transformers/pull/4321",
        "merged": "false",
        "reviewers": "carol,alice,bob",
        "state": "open",
        "title": "Fix dashboard metadata",
    }


def test_fetch_github_pr_info_expands_emoji_shortcodes_in_title() -> None:
    pr_payload = FakeResponse(
        '{"html_url": "https://github.com/huggingface/transformers/pull/4321", '
        '"state": "open", "title": ":rotating_light: Fix flaky test :fire:", '
        '"user": {"login": "octocat"}, '
        '"head": {"sha": "deadbeef"}, "created_at": "2024-01-02T03:04:05Z"}'
    )
    reviews_payload = FakeResponse("[]")
    with patch(
        "transformersci.otel.trace_exporter.urlopen",
        side_effect=[pr_payload, reviews_payload],
    ):
        metadata = trace_exporter.fetch_github_pr_info(
            "huggingface/transformers", "4321"
        )

    # Grafana renders GitHub shortcodes verbatim, so the exporter expands them
    # to real glyphs at the source.
    assert metadata["title"] == "🚨 Fix flaky test 🔥"


def test_fetch_github_pr_info_carries_merged_flag() -> None:
    pr_payload = FakeResponse(
        '{"html_url": "https://github.com/huggingface/transformers/pull/4321", '
        '"state": "closed", "merged": true, "title": "Done", '
        '"user": {"login": "octocat"}, "head": {"sha": "deadbeef"}}'
    )
    reviews_payload = FakeResponse("[]")
    with patch(
        "transformersci.otel.trace_exporter.urlopen",
        side_effect=[pr_payload, reviews_payload],
    ):
        metadata = trace_exporter.fetch_github_pr_info(
            "huggingface/transformers", "4321"
        )
    assert metadata["state"] == "closed"
    assert metadata["merged"] == "true"


def test_fetch_github_commit_message_expands_emoji_shortcodes() -> None:
    payload = FakeResponse(
        '{"commit": {"message": ":bug: Fix the flaky test\\n\\nLong body."}}'
    )
    with patch(
        "transformersci.otel.trace_exporter.urlopen",
        side_effect=[payload],
    ):
        message = trace_exporter.fetch_github_commit_message(
            "huggingface/transformers", "cafef00d1234"
        )

    assert message == "🐛 Fix the flaky test"


def test_emojize_is_a_noop_without_shortcodes_or_dependency() -> None:
    assert trace_exporter._emojize("plain title with no codes") == (
        "plain title with no codes"
    )
    assert trace_exporter._emojize("") == ""
    # Unknown shortcodes are left untouched rather than dropped.
    assert trace_exporter._emojize(":not_a_real_emoji:") == ":not_a_real_emoji:"


def test_extract_pr_info_metrics_fetches_metadata_once_per_pr() -> None:
    calls: list[tuple[str, str]] = []

    def metadata_fetcher(repository: str, pr: str) -> dict[str, str]:
        calls.append((repository, pr))
        return {
            "author": "octocat",
            "commit_sha": "deadbeefcafebabe1234567890abcdef00000000",
            "created_at": "2024-01-02T03:04:05Z",
            "html_url": "https://github.com/huggingface/transformers/pull/4321",
            "reviewers": "alice,bob",
            "state": "open",
            "title": "Fix dashboard metadata",
        }

    metrics = trace_exporter.extract_pr_info_metrics(
        workflow_split_across_three_jobs(),
        _metadata_fetcher=metadata_fetcher,
    )

    info_lines = metric_lines(metrics, "pytest_pr_info")
    assert len(info_lines) == 1
    assert calls == [("huggingface/transformers", "4321")]
    assert 'author="octocat"' in info_lines[0]
    assert 'commit_sha="deadbeefcafebabe1234567890abcdef00000000"' in info_lines[0]
    assert (
        'html_url="https://github.com/huggingface/transformers/pull/4321"'
        in info_lines[0]
    )
    assert 'repository="huggingface/transformers"' in info_lines[0]
    assert 'reviewers="alice,bob"' in info_lines[0]
    assert 'state="open"' in info_lines[0]
    assert 'title="Fix dashboard metadata"' in info_lines[0]

    created_lines = metric_lines(metrics, "pytest_pr_created_at_seconds")
    assert len(created_lines) == 1
    assert 'pr="4321"' in created_lines[0]
    # 2024-01-02T03:04:05Z == 1704164645
    assert created_lines[0].endswith(" 1704164645")

    # Numeric state gauge: one series per PR, open -> 1 (closed -> 0). Lets the
    # dashboard last_over_time() the current state without a multi-valued label.
    state_lines = metric_lines(metrics, "pytest_pr_state")
    assert len(state_lines) == 1
    assert 'pr="4321"' in state_lines[0]
    assert state_lines[0].endswith(" 1")


def test_extract_pr_info_metrics_state_gauge_closed_is_zero() -> None:
    # Closed without merging -> 0 (abandoned).
    metrics = trace_exporter.extract_pr_info_metrics(
        workflow_split_across_three_jobs(),
        _metadata_fetcher=lambda repo, pr: {"state": "closed", "merged": "false"},
    )
    state_lines = metric_lines(metrics, "pytest_pr_state")
    assert len(state_lines) == 1
    assert state_lines[0].endswith(" 0")


def test_extract_pr_info_metrics_state_gauge_merged_is_two() -> None:
    # Closed *and* merged -> 2, distinguishing it from an abandoned close.
    metrics = trace_exporter.extract_pr_info_metrics(
        workflow_split_across_three_jobs(),
        _metadata_fetcher=lambda repo, pr: {"state": "closed", "merged": "true"},
    )
    state_lines = metric_lines(metrics, "pytest_pr_state")
    assert len(state_lines) == 1
    assert state_lines[0].endswith(" 2")


def test_extract_pr_info_metrics_state_gauge_omitted_when_unknown() -> None:
    # Unknown state (GitHub lookup failed) -> no gauge, so the column stays blank
    # rather than implying open/closed.
    metrics = trace_exporter.extract_pr_info_metrics(
        workflow_split_across_three_jobs(),
        _metadata_fetcher=lambda repo, pr: {"state": ""},
    )
    assert metric_lines(metrics, "pytest_pr_state") == []


def test_extract_pr_info_metrics_defaults_commit_sha_to_main() -> None:
    def metadata_fetcher(repository: str, pr: str) -> dict[str, str]:
        return {
            "author": "octocat",
            "commit_sha": "",
            "html_url": "https://github.com/huggingface/transformers/pull/4321",
            "state": "open",
            "title": "Fix dashboard metadata",
        }

    metrics = trace_exporter.extract_pr_info_metrics(
        workflow_split_across_three_jobs(),
        _metadata_fetcher=metadata_fetcher,
    )

    info_lines = metric_lines(metrics, "pytest_pr_info")
    assert len(info_lines) == 1
    assert 'commit_sha="main"' in info_lines[0]


def test_extract_trace_rows_promotes_commit_sha_from_head_revision() -> None:
    trace = make_trace(
        trace_id="trace-main",
        run_id="run-main",
        job="tests_torch",
        commit_sha="cafef00d1234",
        spans=[
            make_test_span(
                process_id="pytest-process",
                nodeid="tests/test_main.py::TestMain::test_one",
                start_time=1_000_000,
                duration=1_000_000,
            )
        ],
    )
    trace_info, _rows = trace_exporter.extract_trace_rows(trace)
    assert trace_info["commit_sha"] == "cafef00d1234"


def test_fetch_github_commit_message_returns_subject_line() -> None:
    payload = FakeResponse(
        '{"commit": {"message": "Fix the flaky test\\n\\nLong body here."}}'
    )
    with patch.dict(
        "os.environ",
        {"PYTEST_GITHUB_API_URL": "https://api.github.example"},
        clear=False,
    ):
        with patch(
            "transformersci.otel.trace_exporter.urlopen",
            side_effect=[payload],
        ) as mocked_urlopen:
            message = trace_exporter.fetch_github_commit_message(
                "huggingface/transformers", "cafef00d1234"
            )

    request = mocked_urlopen.call_args_list[0].args[0]
    assert request.full_url == (
        "https://api.github.example/repos/huggingface/transformers/commits/cafef00d1234"
    )
    assert message == "Fix the flaky test"


def test_fetch_github_commit_message_returns_empty_on_error() -> None:
    with patch(
        "transformersci.otel.trace_exporter.urlopen",
        side_effect=OSError("boom"),
    ):
        assert (
            trace_exporter.fetch_github_commit_message(
                "huggingface/transformers", "abc"
            )
            == ""
        )


def test_extract_run_info_metrics_resolves_commit_message_once_per_run() -> None:
    calls: list[tuple[str, str]] = []

    def commit_fetcher(repository: str, sha: str) -> str:
        calls.append((repository, sha))
        return "Bump version to 5.0"

    traces = [
        make_trace(
            trace_id="trace-a",
            run_id="run-main",
            job="tests_torch",
            pr="main",
            commit_sha="cafef00d1234",
            spans=[
                make_test_span(
                    process_id="pytest-process",
                    nodeid="tests/test_main.py::TestMain::test_one",
                    start_time=1_000_000,
                    duration=1_000_000,
                )
            ],
        ),
        make_trace(
            trace_id="trace-b",
            run_id="run-main",
            job="tests_tf",
            pr="main",
            commit_sha="cafef00d1234",
            spans=[
                make_test_span(
                    process_id="pytest-process",
                    nodeid="tests/test_main.py::TestMain::test_two",
                    start_time=2_000_000,
                    duration=1_000_000,
                )
            ],
        ),
    ]

    metrics = trace_exporter.extract_run_info_metrics(
        traces, _commit_fetcher=commit_fetcher
    )
    info_lines = metric_lines(metrics, "pytest_run_info")
    assert len(info_lines) == 1
    # One run identity -> one GitHub lookup, even across multiple job traces.
    assert calls == [("huggingface/transformers", "cafef00d1234")]
    assert 'commit_message="Bump version to 5.0"' in info_lines[0]
    assert 'commit_sha="cafef00d1234"' in info_lines[0]
    assert 'run_id="run-main"' in info_lines[0]
    assert (
        'html_url="https://github.com/huggingface/transformers/commit/cafef00d1234"'
        in info_lines[0]
    )


def test_extract_run_info_metrics_skips_github_when_no_commit_sha() -> None:
    def commit_fetcher(repository: str, sha: str) -> str:
        raise AssertionError("should not fetch without a commit sha")

    metrics = trace_exporter.extract_run_info_metrics(
        workflow_split_across_three_jobs(),
        _commit_fetcher=commit_fetcher,
    )
    info_lines = metric_lines(metrics, "pytest_run_info")
    assert len(info_lines) == 1
    assert 'commit_message=""' in info_lines[0]
    assert 'commit_sha=""' in info_lines[0]


def test_extract_run_info_metrics_falls_back_to_short_sha_when_no_message() -> None:
    """When GitHub yields no message (no token / lookup failure), the Commit
    column should still show the short SHA, and html_url still links the commit."""

    def commit_fetcher(repository: str, sha: str) -> str:
        return ""  # e.g. unauthenticated + rate-limited

    trace = make_trace(
        trace_id="trace-main",
        run_id="run-main",
        job="tests_torch",
        pr="main",
        commit_sha="cafef00d1234deadbeef",
        spans=[
            make_test_span(
                process_id="pytest-process",
                nodeid="tests/test_main.py::TestMain::test_one",
                start_time=1_000_000,
                duration=1_000_000,
            )
        ],
    )
    metrics = trace_exporter.extract_run_info_metrics(
        [trace], _commit_fetcher=commit_fetcher
    )
    info_lines = metric_lines(metrics, "pytest_run_info")
    assert len(info_lines) == 1
    # Short SHA (first 12 chars) used as the display message.
    assert 'commit_message="cafef00d1234"' in info_lines[0]
    assert 'commit_sha="cafef00d1234deadbeef"' in info_lines[0]
    assert (
        'html_url="https://github.com/huggingface/transformers/commit/'
        'cafef00d1234deadbeef"' in info_lines[0]
    )


def test_extract_test_line_returns_first_test_file_line_number() -> None:
    nodeid = "tests/pipelines/test_x.py::TestX::test_one"
    stacktrace = (
        "self = <tests.pipelines.test_x.TestX testMethod=test_one>\n\n"
        "    def test_one(self):\n"
        ">       call_thing()\n\n"
        "tests/pipelines/test_x.py:145: \n"
        "_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n"
        "src/transformers/foo.py:1002: in call_thing\n"
        '    raise ValueError("boom")\n'
    )
    assert trace_exporter.extract_test_line(stacktrace, nodeid) == "145"


def test_extract_test_line_returns_empty_when_no_match() -> None:
    nodeid = "tests/test_z.py::test_nope"
    stacktrace = "some other stacktrace that does not reference the test file"
    assert trace_exporter.extract_test_line(stacktrace, nodeid) == ""


def test_extract_pr_info_metrics_prefers_repository_backed_trace() -> None:
    process_id = "pytest-process"
    traces = [
        make_trace(
            trace_id="trace-without-repo",
            run_id="run-1",
            job="tests_a",
            spans=[
                make_test_span(
                    process_id=process_id,
                    nodeid="tests/test_a.py::TestA::test_one",
                    start_time=1_000_000,
                    duration=1_000_000,
                )
            ],
            pr="45983",
            pr_url="",
            repository="",
        ),
        make_trace(
            trace_id="trace-with-repo",
            run_id="run-2",
            job="tests_b",
            spans=[
                make_test_span(
                    process_id=process_id,
                    nodeid="tests/test_b.py::TestB::test_two",
                    start_time=2_000_000,
                    duration=1_000_000,
                )
            ],
            pr="45983",
            pr_url="https://github.com/huggingface/transformers/pull/45983",
            repository="huggingface/transformers",
        ),
    ]

    metrics = trace_exporter.extract_pr_info_metrics(
        traces,
        _metadata_fetcher=lambda repository, pr: {
            "author": "octocat",
            "html_url": f"https://github.com/{repository}/pull/{pr}",
            "state": "open",
            "title": "Existing PR sample",
        },
    )

    info_lines = metric_lines(metrics, "pytest_pr_info")
    assert len(info_lines) == 1
    assert 'repository="huggingface/transformers"' in info_lines[0]
    assert (
        'html_url="https://github.com/huggingface/transformers/pull/45983"'
        in info_lines[0]
    )


# ---------------------------------------------------------------------------
# Tempo client + OTLP-shape adapter
# ---------------------------------------------------------------------------


def otlp_attr(key: str, value: str) -> dict:
    return {"key": key, "value": {"stringValue": value}}


def make_otlp_trace(
    *,
    nodeid: str,
    start_nano: int,
    end_nano: int,
    status_code: str = "STATUS_CODE_UNSET",
    exception_type: str | None = None,
    service_name: str = "pytest-observability-demo",
) -> dict:
    """Build a Tempo /api/traces/<id> payload (OTLP JSON) for one test span."""
    events = []
    if exception_type is not None:
        events.append(
            {
                "name": "exception",
                "attributes": [
                    otlp_attr("exception.type", exception_type),
                    otlp_attr("exception.stacktrace", f"Traceback for {nodeid}"),
                ],
            }
        )
    return {
        "batches": [
            {
                "resource": {
                    "attributes": [
                        otlp_attr("service.name", service_name),
                        otlp_attr("transformers.test.run.id", "run-1"),
                        otlp_attr("transformers.test.job", "tests_torch"),
                        otlp_attr("transformers.test.provider", "github_actions"),
                        otlp_attr("vcs.change.id", "4321"),
                        otlp_attr(
                            "vcs.change.url",
                            "https://github.com/huggingface/transformers/pull/4321",
                        ),
                        otlp_attr("vcs.repository.name", "huggingface/transformers"),
                    ]
                },
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "name": nodeid,
                                "startTimeUnixNano": str(start_nano),
                                "endTimeUnixNano": str(end_nano),
                                "status": {"code": status_code},
                                "attributes": [
                                    otlp_attr("pytest.nodeid", nodeid),
                                    otlp_attr("pytest.span_type", "test"),
                                ],
                                "events": events,
                            }
                        ]
                    }
                ],
            }
        ]
    }


def test_parse_lookback_seconds_handles_units() -> None:
    assert trace_exporter.parse_lookback_seconds("1h") == 3600
    assert trace_exporter.parse_lookback_seconds("30m") == 1800
    assert trace_exporter.parse_lookback_seconds("45s") == 45
    assert trace_exporter.parse_lookback_seconds("2d") == 172800
    # Falls back to the default (1h) for unparseable input.
    assert trace_exporter.parse_lookback_seconds("garbage") == 3600


def test_tempo_trace_to_jaeger_maps_spans_status_and_events() -> None:
    payload = make_otlp_trace(
        nodeid="tests/test_torch.py::TestTorch::test_fail",
        start_nano=4_000_000_000,  # 4_000_000 micros
        end_nano=5_000_000_000,  # duration 1_000_000 micros
        status_code="STATUS_CODE_ERROR",
        exception_type="AssertionError",
    )
    trace = trace_exporter.tempo_trace_to_jaeger("trace-torch", payload)

    assert trace["traceID"] == "trace-torch"
    assert len(trace["spans"]) == 1
    span = trace["spans"][0]
    assert span["operationName"] == "tests/test_torch.py::TestTorch::test_fail"
    assert span["startTime"] == 4_000_000
    assert span["duration"] == 1_000_000

    span_tags = trace_exporter.tag_map(span["tags"])
    # status.code (no explicit otel.status_code attr) is synthesized into a tag.
    assert span_tags["otel.status_code"] == "ERROR"
    assert span_tags["pytest.span_type"] == "test"

    # The exception event becomes a Jaeger-style log that extract_exception_info reads.
    exc_type, stacktrace = trace_exporter.extract_exception_info(span)
    assert exc_type == "AssertionError"
    assert "Traceback for" in stacktrace

    process = trace["processes"][span["processID"]]
    assert process["serviceName"] == "pytest-observability-demo"
    process_tags = trace_exporter.tag_map(process["tags"])
    assert process_tags["transformers.test.run.id"] == "run-1"
    assert process_tags["vcs.change.id"] == "4321"


def test_adapted_tempo_trace_flows_through_extract_trace_rows() -> None:
    payload = make_otlp_trace(
        nodeid="tests/test_torch.py::TestTorch::test_fail",
        start_nano=4_000_000_000,
        end_nano=5_000_000_000,
        status_code="STATUS_CODE_ERROR",
        exception_type="AssertionError",
    )
    trace = trace_exporter.tempo_trace_to_jaeger("trace-torch", payload)
    trace_info, rows = trace_exporter.extract_trace_rows(trace)

    assert trace_info["run_id"] == "run-1"
    assert trace_info["test_job"] == "tests_torch"
    assert len(rows) == 1
    assert rows[0]["status_code"] == "ERROR"
    assert rows[0]["exception_type"] == "AssertionError"
    assert rows[0]["test_function"] == "test_fail"


def test_extract_trace_rows_does_not_retain_stacktrace_in_metric_row() -> None:
    # The capped stacktrace is consumed only to derive test_line during shaping;
    # no metric emits it, so it must NOT be retained in the row (keeps the kept
    # rows small under high failure volume — the exporter must stay under ~1G).
    payload = make_otlp_trace(
        nodeid="tests/test_torch.py::TestTorch::test_fail",
        start_nano=4_000_000_000,
        end_nano=5_000_000_000,
        status_code="STATUS_CODE_ERROR",
        exception_type="AssertionError",
    )
    trace = trace_exporter.tempo_trace_to_jaeger("trace-torch", payload)
    _info, rows = trace_exporter.extract_trace_rows(trace)
    assert rows and rows[0]["exception_type"] == "AssertionError"
    assert "exception_stacktrace" not in rows[0]


def test_render_emits_exporter_self_metrics(monkeypatch) -> None:
    # Speed / throughput / memory self-metrics for the stack-health dashboard.
    payload = make_otlp_trace(
        nodeid="tests/test_torch.py::TestTorch::test_one",
        start_nano=1_000_000_000,
        end_nano=2_000_000_000,
        status_code="STATUS_CODE_OK",
    )
    trace = trace_exporter.tempo_trace_to_jaeger("trace-torch", payload)
    monkeypatch.setattr(trace_exporter, "_iter_window_shaped", _shaped_iter(trace))

    out = trace_exporter._render_metrics_uncached()

    assert "pytest_trace_exporter_render_duration_seconds " in out
    assert "pytest_trace_exporter_traces_processed_total " in out
    assert "pytest_trace_exporter_trace_count 1" in out
    # RSS is Linux-only (/proc); assert it only where the helper can read it.
    if trace_exporter._process_resident_bytes() is not None:
        assert "pytest_trace_exporter_process_resident_bytes " in out


def test_exporter_self_metrics_include_http_request_counters() -> None:
    trace_exporter._http_requests_total.clear()
    trace_exporter._http_request_duration_seconds_total.clear()
    trace_exporter._http_response_bytes_total.clear()

    trace_exporter._observe_http_request("/badge/pr", 200, 0.012, 512, "miss")
    trace_exporter._observe_http_request("/badge/pr", 200, 0.003, 512, "hit")
    lines = "\n".join(trace_exporter._exporter_self_metric_lines(0.1))

    assert (
        'pytest_trace_exporter_http_requests_total{route="/badge/pr",'
        'status="200",cache="miss"} 1'
    ) in lines
    assert (
        'pytest_trace_exporter_http_requests_total{route="/badge/pr",'
        'status="200",cache="hit"} 1'
    ) in lines
    assert (
        'pytest_trace_exporter_http_request_duration_seconds_total{route="/badge/pr",'
        'status="200",cache="miss"} 0.012000'
    ) in lines
    assert (
        'pytest_trace_exporter_http_response_bytes_total{route="/badge/pr",'
        'status="200",cache="hit"} 512'
    ) in lines


def test_render_emits_configured_limits_for_pressure(monkeypatch) -> None:
    # Dashboards compute exporter "pressure" against the live config, so the
    # limit and soft-memory ceiling are emitted as gauges (not hardcoded).
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_LIMIT", "321")
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_MEM_SOFT_MB", "1740")
    monkeypatch.setattr(trace_exporter, "_iter_window_shaped", _shaped_iter())

    out = trace_exporter._render_metrics_uncached()

    assert "pytest_trace_exporter_limit 321" in out
    assert f"pytest_trace_exporter_mem_soft_bytes {1740 * 1024 * 1024}" in out


def test_render_emits_self_metrics_on_empty_window(monkeypatch) -> None:
    # An idle exporter (no traces in the window) is still healthy: its own
    # health metrics must stay populated so the dashboard shows "idle", not
    # "no data".
    monkeypatch.setattr(trace_exporter, "_iter_window_shaped", _shaped_iter())
    out = trace_exporter._render_metrics_uncached()
    assert "pytest_trace_exporter_up 1" in out
    assert "pytest_trace_exporter_render_duration_seconds " in out
    assert "pytest_trace_exporter_traces_processed_total " in out


def test_render_metrics_serves_cached_payload(monkeypatch, tmp_path) -> None:
    # Background-render model: render_metrics serves the last payload the
    # refresher published to disk and never renders inline, so a scrape can't
    # block on the multi-second Tempo fetch (which had been timing out scrapes).
    payload_file = tmp_path / "payload.prom"
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_PAYLOAD_FILE", str(payload_file))
    assert (
        "pytest_trace_exporter_up 1" in trace_exporter.render_metrics()
    )  # warming: no file yet
    payload = make_otlp_trace(
        nodeid="tests/test_torch.py::TestTorch::test_one",
        start_nano=1_000_000_000,
        end_nano=2_000_000_000,
        status_code="STATUS_CODE_OK",
    )
    trace = trace_exporter.tempo_trace_to_jaeger("trace-torch", payload)
    monkeypatch.setattr(trace_exporter, "_iter_window_shaped", _shaped_iter(trace))
    trace_exporter._refresh_cache_once()
    out = trace_exporter.render_metrics()
    assert "pytest_trace_exporter_render_duration_seconds " in out
    assert "pytest_trace_exporter_last_render_timestamp_seconds " in out
    assert "pytest_test_duration_seconds{" in out


def test_refresh_cache_once_publishes_atomically(monkeypatch, tmp_path) -> None:
    # The published file must always be a complete payload, never a half-written
    # temp file, and no .tmp siblings may be left behind.
    payload_file = tmp_path / "payload.prom"
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_PAYLOAD_FILE", str(payload_file))
    monkeypatch.setattr(trace_exporter, "_iter_window_shaped", _shaped_iter())
    trace_exporter._refresh_cache_once()
    body = payload_file.read_text(encoding="utf-8")
    assert body.endswith("\n")
    assert "pytest_trace_exporter_up 1" in body
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != payload_file.name]
    assert leftovers == []


def test_streamed_payload_matches_full_render(monkeypatch, tmp_path) -> None:
    # The streaming writer must produce the same metric data as the whole-string
    # render — streaming is purely a memory optimisation. Self-metric lines that
    # vary per render (timing, RSS, cumulative counter) are filtered before
    # comparison so two separate renders can be compared.
    volatile = (
        "pytest_trace_exporter_render_duration_seconds",
        "pytest_trace_exporter_last_render_timestamp_seconds",
        "pytest_trace_exporter_traces_processed_total",
        "pytest_trace_exporter_process_resident_bytes",
    )

    def stable(text: str) -> list[str]:
        return [ln for ln in text.splitlines() if not ln.startswith(volatile)]

    payload = make_otlp_trace(
        nodeid="tests/test_torch.py::TestTorch::test_one",
        start_nano=1_000_000_000,
        end_nano=2_000_000_000,
        status_code="STATUS_CODE_OK",
    )
    trace = trace_exporter.tempo_trace_to_jaeger("trace-torch", payload)
    monkeypatch.setattr(trace_exporter, "_iter_window_shaped", _shaped_iter(trace))
    expected = trace_exporter._render_metrics_uncached()

    monkeypatch.setattr(trace_exporter, "_iter_window_shaped", _shaped_iter(trace))
    out = tmp_path / "payload.prom"
    trace_exporter._write_payload_atomic(out)
    streamed = out.read_text(encoding="utf-8")

    assert streamed.endswith("\n")
    assert stable(streamed) == stable(expected)
    assert "pytest_test_duration_seconds{" in streamed


def test_memory_guard_drops_cache_over_soft_limit(monkeypatch) -> None:
    trace_exporter._trace_cache.clear()
    trace_exporter._trace_cache_sizes.clear()
    trace_exporter._trace_cache_bytes = 0
    trace_exporter._trace_cache["t0"] = {"traceID": "t0"}
    trace_exporter._trace_cache_sizes["t0"] = 100
    trace_exporter._trace_cache_bytes = 100
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_MEM_SOFT_MB", "500")

    # Under the soft limit -> cache untouched.
    monkeypatch.setattr(
        trace_exporter, "_process_resident_bytes", lambda: 100 * 1024 * 1024
    )
    trace_exporter._relieve_memory_pressure()
    assert len(trace_exporter._trace_cache) == 1

    # Over the soft limit -> cache dropped so the next render can't grow past it.
    monkeypatch.setattr(
        trace_exporter, "_process_resident_bytes", lambda: 600 * 1024 * 1024
    )
    trace_exporter._relieve_memory_pressure()
    assert len(trace_exporter._trace_cache) == 0
    assert trace_exporter._trace_cache_bytes == 0


def test_limit_malloc_arenas_is_safe_everywhere(monkeypatch) -> None:
    # Must never raise — it's a best-effort glibc tweak that no-ops elsewhere
    # (e.g. macOS), so the feature works without any launcher-set env.
    trace_exporter._limit_malloc_arenas()
    monkeypatch.setenv("MALLOC_ARENA_MAX", "1")
    trace_exporter._limit_malloc_arenas()
    monkeypatch.setenv("MALLOC_ARENA_MAX", "0")  # disabled
    trace_exporter._limit_malloc_arenas()


def test_memory_guard_disabled_when_zero(monkeypatch) -> None:
    trace_exporter._trace_cache.clear()
    trace_exporter._trace_cache["t0"] = {"traceID": "t0"}
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_MEM_SOFT_MB", "0")
    monkeypatch.setattr(
        trace_exporter, "_process_resident_bytes", lambda: 999 * 1024 * 1024
    )
    trace_exporter._relieve_memory_pressure()
    assert len(trace_exporter._trace_cache) == 1  # disabled: never drops
    trace_exporter._trace_cache.clear()


def test_payload_survives_crash_and_serves_last_good(monkeypatch, tmp_path) -> None:
    # Simulate a crash: the next render raises *after* a good payload was already
    # published. The on-disk file from before the crash must still be served
    # intact (it is derived/cacheable data, but a stale-complete payload beats a
    # torn one), and the failed render must not corrupt it.
    payload_file = tmp_path / "payload.prom"
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_PAYLOAD_FILE", str(payload_file))
    payload = make_otlp_trace(
        nodeid="tests/test_torch.py::TestTorch::test_one",
        start_nano=1_000_000_000,
        end_nano=2_000_000_000,
        status_code="STATUS_CODE_OK",
    )
    trace = trace_exporter.tempo_trace_to_jaeger("trace-torch", payload)
    monkeypatch.setattr(trace_exporter, "_iter_window_shaped", _shaped_iter(trace))
    trace_exporter._refresh_cache_once()
    good = payload_file.read_text(encoding="utf-8")
    assert "pytest_test_duration_seconds{" in good

    def boom(*_a, **_k):
        raise RuntimeError("render crashed")

    # Crash mid-stream (after the writer has opened its temp file): the streaming
    # writer must clean up the temp file and leave the previous payload intact.
    monkeypatch.setattr(trace_exporter, "extract_per_test_duration_metrics", boom)
    with pytest.raises(RuntimeError):
        trace_exporter._refresh_cache_once()
    # Last-good payload still intact, no torn body, no leftover temp files.
    assert payload_file.read_text(encoding="utf-8") == good
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != payload_file.name]
    assert leftovers == []


class _UrlResponse(io.StringIO):
    def __enter__(self) -> "_UrlResponse":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


def _tempo_urlopen(search_ids: list[str], trace_payload: dict):
    """Fake urlopen routing /api/search vs /api/traces/<id> for fetch_traces."""

    def opener(url, timeout=None):
        if "/api/search" in url:
            body = {"traces": [{"traceID": tid} for tid in search_ids]}
        else:
            body = trace_payload
        return _UrlResponse(json.dumps(body))

    return opener


def test_fetch_traces_searches_then_fetches_each_trace() -> None:
    trace_exporter._trace_cache.clear()
    payload = make_otlp_trace(
        nodeid="tests/test_torch.py::TestTorch::test_one",
        start_nano=1_000_000_000,
        end_nano=2_000_000_000,
        status_code="STATUS_CODE_OK",
    )
    with patch.dict(
        "os.environ",
        {
            "PYTEST_TRACE_EXPORTER_TEMPO_URL": "http://tempo:3200",
            "PYTEST_TRACE_EXPORTER_SERVICE_NAME": "pytest-observability-demo",
        },
        clear=False,
    ):
        with patch(
            "transformersci.otel.trace_exporter.urlopen",
            side_effect=_tempo_urlopen(["trace-torch"], payload),
        ):
            traces = trace_exporter.fetch_traces()

    assert len(traces) == 1
    assert traces[0]["traceID"] == "trace-torch"
    assert traces[0]["spans"][0]["operationName"].endswith("test_one")


def test_fetch_traces_caches_settled_traces() -> None:
    trace_exporter._trace_cache.clear()
    trace_exporter._trace_growth.clear()
    payload = make_otlp_trace(
        nodeid="tests/test_torch.py::TestTorch::test_one",
        start_nano=1_000_000_000,
        end_nano=2_000_000_000,
        status_code="STATUS_CODE_OK",
    )
    opener = _tempo_urlopen(["trace-torch"], payload)
    # settle=0: a trace settles once its span count is observed UNCHANGED across
    # two renders (count-quiescence). The first render only records the count, so
    # the trace is re-fetched on the second render — and memoized then, served
    # from cache on the third.
    with patch.dict(
        "os.environ",
        {
            "PYTEST_TRACE_EXPORTER_TRACE_SETTLE_SECONDS": "0",
            # Isolate the pure count-quiescence path; the reverify window is
            # exercised separately in test_trace_reverify_window_*.
            "PYTEST_TRACE_EXPORTER_TRACE_REVERIFY_SECONDS": "0",
        },
        clear=False,
    ):
        with patch(
            "transformersci.otel.trace_exporter.urlopen", side_effect=opener
        ) as mocked:
            trace_exporter.fetch_traces()
            first_calls = mocked.call_count
            trace_exporter.fetch_traces()
            second_calls = mocked.call_count
            trace_exporter.fetch_traces()
            third_calls = mocked.call_count

    # Renders 1 & 2 each do 1 search + 1 trace fetch (the count must be seen
    # stable twice before it's trusted as complete); render 3 is search-only,
    # the now-settled trace served from cache.
    assert first_calls == 2
    assert second_calls == 4
    assert third_calls == 5


def test_iter_traces_skips_failed_fetches() -> None:
    # A slow/failing trace must drop out (None) without sinking the others, so
    # the streaming render still gets every healthy trace. Also exercises the
    # bounded-window fan-out across more ids than the default concurrency.
    trace_exporter._trace_cache.clear()
    payload = make_otlp_trace(
        nodeid="tests/test_torch.py::TestTorch::test_one",
        start_nano=1_000_000_000,
        end_nano=2_000_000_000,
        status_code="STATUS_CODE_OK",
    )
    ids = [f"ok-{i}" for i in range(20)] + ["boom"]

    def opener(url, timeout=None):
        if "/api/search" in url:
            return _UrlResponse(json.dumps({"traces": [{"traceID": t} for t in ids]}))
        if "boom" in url:
            raise OSError("connection reset by peer")
        return _UrlResponse(json.dumps(payload))

    with patch.dict(
        "os.environ",
        {"PYTEST_TRACE_EXPORTER_TEMPO_URL": "http://tempo:3200"},
        clear=False,
    ):
        with patch("transformersci.otel.trace_exporter.urlopen", side_effect=opener):
            traces = list(trace_exporter.iter_traces())

    # The failing fetch is dropped; all 20 healthy traces still come through.
    assert len(traces) == 20
    assert {t["traceID"] for t in traces} == {f"ok-{i}" for i in range(20)}


def _windowed_search_opener(timestamps: dict[str, int]):
    """Fake urlopen for /api/search that honours the [start,end) window + limit.

    Models Tempo: returns the ids whose timestamp falls in the requested window,
    most-recent first, capped at ``limit`` — exactly the behaviour that buries a
    sharded run's older traces under newer ones at real volume.
    """

    def opener(url, timeout=None):
        assert "/api/search" in url
        params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        start = int(params["start"][0])
        end = int(params["end"][0])
        limit = int(params["limit"][0])
        in_window = [tid for tid, ts in timestamps.items() if start <= ts < end]
        in_window.sort(key=lambda tid: timestamps[tid], reverse=True)
        chosen = in_window[:limit]
        return _UrlResponse(json.dumps({"traces": [{"traceID": t} for t in chosen]}))

    return opener


def test_search_all_trace_ids_enumerates_every_shard_beyond_limit() -> None:
    # Models the production bug: 8 shard traces of one run plus noise, but a
    # single search only ever returns the newest ``limit`` ids. Bisection must
    # surface ALL of them so no shard is silently dropped.
    shards = {f"shard-{i}": 2_000 + i for i in range(8)}
    noise = {f"noise-{i}": 2_100 + i for i in range(40)}
    timestamps = {**shards, **noise}
    with patch(
        "transformersci.otel.trace_exporter.urlopen",
        side_effect=_windowed_search_opener(timestamps),
    ):
        ids, truncated = trace_exporter.search_all_trace_ids(
            "http://tempo:3200", "svc", 2_000, 2_200, limit=4
        )

    assert set(ids) == set(timestamps)  # every shard AND every noise trace
    assert set(shards).issubset(set(ids))
    assert truncated is False


def test_search_all_trace_ids_flags_truncation_when_unsplittable() -> None:
    # More than ``limit`` traces sharing one timestamp can't be split below a
    # 1-second slice; enumeration returns what it can and flags truncation
    # rather than silently claiming completeness.
    timestamps = {f"t{i}": 5_000 for i in range(6)}
    with patch(
        "transformersci.otel.trace_exporter.urlopen",
        side_effect=_windowed_search_opener(timestamps),
    ):
        ids, truncated = trace_exporter.search_all_trace_ids(
            "http://tempo:3200", "svc", 5_000, 5_001, limit=3
        )

    assert truncated is True
    assert len(ids) <= 3


def _reset_window_state() -> None:
    trace_exporter._shaped_cache.clear()
    trace_exporter._shaped_cache_sizes.clear()
    trace_exporter._shaped_meta.clear()
    trace_exporter._shaped_cache_bytes = 0
    trace_exporter._trace_cache.clear()
    trace_exporter._trace_growth.clear()
    trace_exporter._previous_new_ids = set()
    trace_exporter._run_members.clear()
    trace_exporter._run_last_growth.clear()


def test_window_shaped_caches_settled_trace_and_skips_refetch() -> None:
    # A settled trace is shaped once and served from the shaped cache thereafter,
    # so the complete window costs no repeat Tempo fetches each render.
    _reset_window_state()
    payload = make_otlp_trace(
        nodeid="tests/test_torch.py::TestTorch::test_one",
        start_nano=1_000_000_000,
        end_nano=2_000_000_000,
        status_code="STATUS_CODE_OK",
    )
    calls = {"search": 0, "traces": 0}

    def opener(url, timeout=None):
        if "/api/search" in url:
            calls["search"] += 1
            return _UrlResponse(json.dumps({"traces": [{"traceID": "trace-torch"}]}))
        calls["traces"] += 1
        return _UrlResponse(json.dumps(payload))

    with patch("transformersci.otel.trace_exporter.urlopen", side_effect=opener):
        first = list(trace_exporter._iter_window_shaped("http://tempo:3200"))
        second = list(trace_exporter._iter_window_shaped("http://tempo:3200"))

    assert [is_new for *_rest, is_new in first] == [True]
    assert [is_new for *_rest, is_new in second] == [False]
    assert calls["traces"] == 1  # fetched once, cached for the second render
    _reset_window_state()


def test_window_defers_new_fetches_beyond_cap(monkeypatch) -> None:
    # New traces beyond the per-render cap are deferred (not fetched this cycle),
    # bounding the cold-start/restart burst; the rest land on later renders.
    _reset_window_state()
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_MAX_NEW_FETCH_PER_RENDER", "2")
    ids = [f"trace-{i}" for i in range(5)]
    payload = make_otlp_trace(
        nodeid="tests/test_torch.py::TestTorch::test_one",
        start_nano=1_000_000_000,
        end_nano=2_000_000_000,
        status_code="STATUS_CODE_OK",
    )

    def opener(url, timeout=None):
        if "/api/search" in url:
            return _UrlResponse(json.dumps({"traces": [{"traceID": t} for t in ids]}))
        return _UrlResponse(json.dumps(payload))

    with patch("transformersci.otel.trace_exporter.urlopen", side_effect=opener):
        yielded = list(trace_exporter._iter_window_shaped("http://tempo:3200"))

    assert sum(1 for *_r, is_new in yielded if is_new) == 2
    assert trace_exporter._last_enumeration_deferred == 3
    _reset_window_state()


def test_per_test_emitted_for_new_traces_then_retires_after_carryover(
    monkeypatch,
) -> None:
    # Per-test (high-cardinality) series are emitted for a freshly-seen trace and
    # for one carryover render, then stop — Prometheus' last_over_time retains
    # them, so the payload stays small without losing the data.
    _reset_window_state()
    payload = make_otlp_trace(
        nodeid="tests/test_torch.py::TestTorch::test_one",
        start_nano=1_000_000_000,
        end_nano=2_000_000_000,
        status_code="STATUS_CODE_OK",
    )
    with patch(
        "transformersci.otel.trace_exporter.urlopen",
        side_effect=_tempo_urlopen(["trace-torch"], payload),
    ):
        first = trace_exporter._render_metrics_uncached()
        second = trace_exporter._render_metrics_uncached()
        third = trace_exporter._render_metrics_uncached()

    marker = "pytest_test_duration_seconds{"
    assert marker in first  # freshly seen
    assert marker in second  # one-render carryover
    assert marker not in third  # retired; retained by Prometheus
    _reset_window_state()


def _with_run_id(payload: dict, run_id: str) -> dict:
    """Clone an OTLP payload, overriding its run id (used to vary noise traces)."""
    clone = json.loads(json.dumps(payload))
    for attr in clone["batches"][0]["resource"]["attributes"]:
        if attr["key"] == "transformers.test.run.id":
            attr["value"] = {"stringValue": run_id}
    return clone


def test_render_rolls_up_every_shard_of_a_sharded_run(monkeypatch) -> None:
    # The headline fix: a run sharded across 8 traces, buried under noise so a
    # single capped search can never see them all at once. Complete enumeration
    # must let the run roll-up count all 8 shards, not the ~3 the old top-N
    # snapshot captured.
    _reset_window_state()
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_LIMIT", "3")
    monkeypatch.setattr(trace_exporter, "_run_settle_seconds", lambda: 0.0)

    base = int(__import__("time").time()) - 300  # recent but settled (>120s old)
    payloads: dict[str, dict] = {}
    timestamps: dict[str, int] = {}
    # 8 shard traces — same run-1 / tests_torch, one distinct test each.
    for shard in range(8):
        tid = f"shard-{shard}"
        payloads[tid] = make_otlp_trace(
            nodeid=f"tests/test_torch.py::TestTorch::test_{shard}",
            start_nano=(base + shard) * 1_000_000_000,
            end_nano=(base + shard + 1) * 1_000_000_000,
            status_code="STATUS_CODE_OK",
        )
        timestamps[tid] = base + shard
    # Noise: newer traces from *other* runs that crowd out the shards in any
    # single top-3 search.
    for n in range(20):
        tid = f"noise-{n}"
        payloads[tid] = _with_run_id(
            make_otlp_trace(
                nodeid=f"tests/test_other.py::test_{n}",
                start_nano=(base + 50 + n) * 1_000_000_000,
                end_nano=(base + 51 + n) * 1_000_000_000,
                status_code="STATUS_CODE_OK",
            ),
            run_id=f"noise-run-{n}",
        )
        timestamps[tid] = base + 50 + n

    def opener(url, timeout=None):
        if "/api/search" in url:
            return _windowed_search_opener(timestamps)(url, timeout)
        tid = urllib.parse.unquote(url.rsplit("/", 1)[-1])
        return _UrlResponse(json.dumps(payloads[tid]))

    with patch("transformersci.otel.trace_exporter.urlopen", side_effect=opener):
        out = trace_exporter._render_metrics_uncached()

    torch_job = next(
        (
            value
            for labels, value in trace_exporter._iter_metric_samples(
                "pytest_run_job_total_tests", source=out
            )
            if labels.get("run_id") == "run-1"
            and labels.get("test_job") == "tests_torch"
        ),
        None,
    )
    assert torch_job == 8.0  # all 8 shards counted, not a top-N fraction
    _reset_window_state()


def test_trace_cache_is_bounded_lru(monkeypatch) -> None:
    # The settled-trace cache must not grow without bound (it holds full
    # multi-MB traces; unbounded growth crept the exporter RSS toward its cap).
    trace_exporter._trace_cache.clear()
    trace_exporter._trace_growth.clear()
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_TRACE_CACHE_MAX", "3")
    # Settle timing is exercised elsewhere; here force every trace settled so the
    # focus stays on the LRU bound. Same payload shape for every id.
    monkeypatch.setattr(trace_exporter, "_trace_is_settled", lambda *a, **k: True)
    payload = make_otlp_trace(
        nodeid="tests/test_torch.py::TestTorch::test_one",
        start_nano=1_000_000_000,
        end_nano=2_000_000_000,
        status_code="STATUS_CODE_OK",
    )
    with patch(
        "transformersci.otel.trace_exporter.urlopen",
        side_effect=lambda url, timeout=None: _UrlResponse(json.dumps(payload)),
    ):
        for i in range(6):
            trace_exporter.get_trace(f"t{i}", "http://tempo:3200")
    try:
        assert len(trace_exporter._trace_cache) == 3
        # Oldest three evicted; the three most-recently inserted remain.
        assert set(trace_exporter._trace_cache) == {"t3", "t4", "t5"}
    finally:
        trace_exporter._trace_cache.clear()


def _trace_with_spans(n: int) -> dict:
    return {"spans": [{"spanID": f"s{i}"} for i in range(n)]}


def test_trace_is_settled_requires_a_prior_observation() -> None:
    trace_exporter._trace_growth.clear()
    # First sight never settles — we cannot yet know the count has stopped
    # growing, even with a zero settle window.
    assert (
        trace_exporter._trace_is_settled(
            "t", _trace_with_spans(100), now=1000.0, settle_seconds=0.0
        )
        is False
    )
    trace_exporter._trace_growth.clear()


def test_trace_settles_once_count_holds_for_the_settle_window() -> None:
    trace_exporter._trace_growth.clear()
    trace = _trace_with_spans(100)
    settle = 120.0
    # Recorded at t=0; still 100 spans at t=60 (<settle); settled at t=121.
    assert trace_exporter._trace_is_settled("t", trace, 0.0, settle) is False
    assert trace_exporter._trace_is_settled("t", trace, 60.0, settle) is False
    assert trace_exporter._trace_is_settled("t", trace, 121.0, settle) is True
    trace_exporter._trace_growth.clear()


def test_trace_not_settled_while_still_growing() -> None:
    # THE regression. A sharded trace is fed by pytest-xdist worker PROCESSES
    # that flush spans at staggered exit times, so it keeps growing for minutes
    # with gaps longer than the settle window. Each new span resets the
    # quiescence clock, so it must NOT be declared settled mid-run — the old
    # newest-span-age heuristic froze such traces at a partial count (~half the
    # tests). It settles only once growth stops, with the COMPLETE count.
    trace_exporter._trace_growth.clear()
    settle = 120.0
    sid = "shard"
    # Bursts at t=0/130/260 — each gap exceeds settle, yet none settles.
    assert (
        trace_exporter._trace_is_settled(sid, _trace_with_spans(3000), 0.0, settle)
        is False
    )
    assert (
        trace_exporter._trace_is_settled(sid, _trace_with_spans(7000), 130.0, settle)
        is False
    )
    assert (
        trace_exporter._trace_is_settled(sid, _trace_with_spans(12477), 260.0, settle)
        is False
    )
    # Growth stops; the final count then holds for the settle window -> settled.
    assert (
        trace_exporter._trace_is_settled(sid, _trace_with_spans(12477), 390.0, settle)
        is True
    )
    trace_exporter._trace_growth.clear()


def test_trace_reverify_window_delays_freeze_past_apparent_quiescence() -> None:
    # The reverify window keeps a count-quiescent trace UNSETTLED (and therefore
    # still re-read) for `reverify_seconds` beyond the settle window, so it is not
    # frozen the instant the count first looks steady.
    trace_exporter._trace_growth.clear()
    settle, reverify = 120.0, 180.0
    trace = _trace_with_spans(100)
    sid = "reverify"
    # Recorded at t=0; steady through settle (t=121) but NOT yet frozen — the
    # reverify window (total 300s) is still open.
    assert trace_exporter._trace_is_settled(sid, trace, 0.0, settle, reverify) is False
    assert (
        trace_exporter._trace_is_settled(sid, trace, 121.0, settle, reverify) is False
    )
    assert (
        trace_exporter._trace_is_settled(sid, trace, 299.0, settle, reverify) is False
    )
    # Only once the full settle+reverify window has held does it settle.
    assert trace_exporter._trace_is_settled(sid, trace, 301.0, settle, reverify) is True
    trace_exporter._trace_growth.clear()


def test_trace_reverify_catches_out_of_order_late_span() -> None:
    # THE fix. Tempo can make a trace's spans queryable OUT OF ORDER, so a failing
    # test's ERROR span may surface after the count first looks quiescent. Without
    # the reverify window the trace would freeze at t~121 missing that span, and
    # the job's failed-count would stay 0 (green dashboard, red GitHub — PR 46259).
    # With reverify, the trace is still being re-read when the late span arrives:
    # the count grows, quiescence re-opens, and the frozen snapshot includes it.
    trace_exporter._trace_growth.clear()
    settle, reverify = 120.0, 180.0
    sid = "late-error"
    # Count looks steady at 100 through the settle window...
    assert (
        trace_exporter._trace_is_settled(
            sid, _trace_with_spans(100), 0.0, settle, reverify
        )
        is False
    )
    assert (
        trace_exporter._trace_is_settled(
            sid, _trace_with_spans(100), 121.0, settle, reverify
        )
        is False
    )
    # ...then a late span becomes queryable at t=200 (still inside the reverify
    # window): the count grows, resetting the quiescence clock.
    assert (
        trace_exporter._trace_is_settled(
            sid, _trace_with_spans(101), 200.0, settle, reverify
        )
        is False
    )
    # It must not freeze until the NEW count holds for the full window (t>=500).
    assert (
        trace_exporter._trace_is_settled(
            sid, _trace_with_spans(101), 480.0, settle, reverify
        )
        is False
    )
    assert (
        trace_exporter._trace_is_settled(
            sid, _trace_with_spans(101), 501.0, settle, reverify
        )
        is True
    )
    trace_exporter._trace_growth.clear()


def test_trace_growth_tracker_is_bounded() -> None:
    # Trace ids that age out before settling must not leak unboundedly.
    trace_exporter._trace_growth.clear()
    original = trace_exporter._TRACE_GROWTH_MAX
    trace_exporter._TRACE_GROWTH_MAX = 4
    try:
        for i in range(10):
            trace_exporter._trace_is_settled(f"t{i}", _trace_with_spans(1), 0.0, 120.0)
        assert len(trace_exporter._trace_growth) == 4
        # The most-recently-seen ids are the ones kept.
        assert set(trace_exporter._trace_growth) == {"t6", "t7", "t8", "t9"}
    finally:
        trace_exporter._TRACE_GROWTH_MAX = original
        trace_exporter._trace_growth.clear()


def test_window_serves_unsettled_trace_from_cache_between_refetches(
    monkeypatch,
) -> None:
    # The bounded-cadence redesign: an in-flight (unsettled) trace's shape is
    # cached and served between renders, and re-fetched at most once per the
    # re-fetch interval — NOT every render. This is what keeps render duration
    # flat while a large sharded run is still settling.
    _reset_window_state()
    # Never settles (count keeps growing); re-fetch only every 100s.
    monkeypatch.setattr(trace_exporter, "_trace_is_settled", lambda *a, **k: False)
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_TRACE_REFETCH_SECONDS", "100")
    payload = make_otlp_trace(
        nodeid="tests/test_torch.py::TestTorch::test_one",
        start_nano=1_000_000_000,
        end_nano=2_000_000_000,
        status_code="STATUS_CODE_OK",
    )
    calls = {"traces": 0}

    def opener(url, timeout=None):
        if "/api/search" in url:
            return _UrlResponse(json.dumps({"traces": [{"traceID": "trace-torch"}]}))
        calls["traces"] += 1
        return _UrlResponse(json.dumps(payload))

    clock = {"now": 1_000.0}
    monkeypatch.setattr(trace_exporter.time, "time", lambda: clock["now"])
    with patch("transformersci.otel.trace_exporter.urlopen", side_effect=opener):
        first = list(trace_exporter._iter_window_shaped("http://tempo:3200"))
        clock["now"] += 10  # within the re-fetch interval
        second = list(trace_exporter._iter_window_shaped("http://tempo:3200"))
        clock["now"] += 100  # past the interval -> due for a re-read
        third = list(trace_exporter._iter_window_shaped("http://tempo:3200"))

    # Every render still yields the trace (the window stays complete)...
    assert [is_new for *_r, is_new in first] == [True]
    assert [is_new for *_r, is_new in second] == [False]  # served from cache
    assert [is_new for *_r, is_new in third] == [True]  # re-fetched after interval
    # ...but it was fetched only twice (render 1 and render 3), not three times.
    assert calls["traces"] == 2
    _reset_window_state()


def test_window_serves_cached_shape_when_refetch_fails(monkeypatch) -> None:
    # A due-for-refetch in-flight trace whose re-fetch fails must still be served
    # from its prior cached shape, not dropped from the window.
    _reset_window_state()
    monkeypatch.setattr(trace_exporter, "_trace_is_settled", lambda *a, **k: False)
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_TRACE_REFETCH_SECONDS", "0")
    payload = make_otlp_trace(
        nodeid="tests/test_torch.py::TestTorch::test_one",
        start_nano=1_000_000_000,
        end_nano=2_000_000_000,
        status_code="STATUS_CODE_OK",
    )
    state = {"fail": False}

    def opener(url, timeout=None):
        if "/api/search" in url:
            return _UrlResponse(json.dumps({"traces": [{"traceID": "trace-torch"}]}))
        if state["fail"]:
            raise OSError("connection reset by peer")
        return _UrlResponse(json.dumps(payload))

    with patch("transformersci.otel.trace_exporter.urlopen", side_effect=opener):
        first = list(trace_exporter._iter_window_shaped("http://tempo:3200"))
        state["fail"] = True
        second = list(trace_exporter._iter_window_shaped("http://tempo:3200"))

    assert [is_new for *_r, is_new in first] == [True]
    # Re-fetch failed, but the prior shape is still served (stale, not new).
    assert [is_new for *_r, is_new in second] == [False]
    assert len(second) == 1
    _reset_window_state()


def test_trace_cache_entry_bytes_counts_resident_overhead() -> None:
    # The cap is a memory budget, so an entry's cost must exceed its raw
    # serialized size (parsed dicts are several times larger in RAM).
    trace = {"data": [{"k": "v" * 100} for _ in range(20)]}
    serialized = len(json.dumps(trace, ensure_ascii=False, separators=(",", ":")))
    assert trace_exporter._trace_cache_entry_bytes(trace) == (
        serialized * trace_exporter._TRACE_RAM_OVERHEAD
    )


def test_trace_cache_is_bounded_by_bytes(monkeypatch) -> None:
    trace_exporter._trace_cache.clear()
    trace_exporter._trace_cache_sizes.clear()
    trace_exporter._trace_cache_bytes = 0
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_TRACE_CACHE_MAX", "10")
    # The byte budget counts ~4x serialized size (resident-memory estimate), so
    # the cap is scaled accordingly to keep this exercising partial eviction.
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_TRACE_CACHE_MAX_BYTES", "4800")
    payload = make_otlp_trace(
        nodeid="tests/test_torch.py::TestTorch::test_one",
        start_nano=1_000_000_000,
        end_nano=2_000_000_000,
        status_code="STATUS_CODE_OK",
    )

    def opener(url, timeout=None):
        payload["resourceSpans"][0]["resource"]["attributes"].append(
            {
                "key": "large.attr",
                "value": {"stringValue": "x" * 900},
            }
        )
        return _UrlResponse(json.dumps(payload))

    with patch("transformersci.otel.trace_exporter.urlopen", side_effect=opener):
        for i in range(3):
            trace_exporter.get_trace(f"t{i}", "http://tempo:3200")

    try:
        assert len(trace_exporter._trace_cache) < 3
        assert trace_exporter._trace_cache_bytes <= 4800
    finally:
        trace_exporter._trace_cache.clear()
        trace_exporter._trace_cache_sizes.clear()
        trace_exporter._trace_cache_bytes = 0


def test_last_failure_metrics_omit_stacktrace_but_keep_pointers() -> None:
    metrics = trace_exporter.extract_average_metrics(workflow_split_across_three_jobs())
    pointer_lines = metric_lines(metrics, "pytest_test_last_failure_info")
    assert len(pointer_lines) == 1
    assert "stacktrace=" not in pointer_lines[0]
    assert 'exception_type="AssertionError"' in pointer_lines[0]
    assert 'trace_id="trace-torch"' in pointer_lines[0]

    # The per-test aggregate gauges were dropped (no dashboard consumed them and
    # at real scale they were ~75% of the payload / RSS). Only the tiny
    # last_failure_info pointer is emitted now. Guard against regressing that.
    body = "\n".join(metrics)
    assert "pytest_test_average_duration_seconds" not in body
    assert "pytest_test_run_count" not in body
    assert "pytest_test_failure_count" not in body

    pr_metrics = trace_exporter.extract_pr_last_failure_metrics(
        workflow_split_across_three_jobs()
    )
    pr_lines = metric_lines(pr_metrics, "pytest_pr_last_failure_info")
    assert len(pr_lines) == 1
    assert "stacktrace=" not in pr_lines[0]
    assert 'trace_id="trace-torch"' in pr_lines[0]


# ---------------------------------------------------------------------------
# /failure traceback page
# ---------------------------------------------------------------------------


def _trace_with_exception(message: str, stacktrace: str) -> dict:
    return {
        "traceID": "trace-fail",
        "processes": {"p0": {"serviceName": "demo", "tags": []}},
        "spans": [
            {
                "operationName": "tests/test_x.py::TestX::test_boom",
                "processID": "p0",
                "startTime": 1_000_000,
                "duration": 1_000_000,
                "tags": [
                    make_tag("pytest.nodeid", "tests/test_x.py::TestX::test_boom"),
                    make_tag("pytest.span_type", "test"),
                    make_tag("otel.status_code", "ERROR"),
                ],
                "logs": [
                    {
                        "fields": [
                            make_tag("event", "exception"),
                            make_tag("exception.type", "AssertionError"),
                            make_tag("exception.message", message),
                            make_tag("exception.stacktrace", stacktrace),
                        ]
                    }
                ],
            }
        ],
    }


def test_extract_failure_details_returns_untruncated_message_and_stacktrace() -> None:
    long_stack = "Traceback\n" + ("frame line\n" * 600)  # >4000 chars
    trace = _trace_with_exception("assert 3 == 4\n  +  where 3 = f()", long_stack)
    details = trace_exporter.extract_failure_details(trace)
    assert len(details) == 1
    d = details[0]
    assert d["test_nodeid"] == "tests/test_x.py::TestX::test_boom"
    assert d["exception_type"] == "AssertionError"
    assert d["exception_message"] == "assert 3 == 4\n  +  where 3 = f()"
    # Not truncated (extract_exception_info caps at 4000, this must not).
    assert d["exception_stacktrace"] == long_stack
    assert "(truncated)" not in d["exception_stacktrace"]


def test_extract_failure_details_filters_by_nodeid() -> None:
    trace = _trace_with_exception("boom", "stack")
    assert trace_exporter.extract_failure_details(
        trace, "tests/test_x.py::TestX::test_boom"
    )
    assert (
        trace_exporter.extract_failure_details(trace, "tests/other.py::test_nope") == []
    )


def test_render_failure_html_escapes_and_includes_both_blocks() -> None:
    details = [
        {
            "test_nodeid": "tests/test_x.py::test_boom",
            "exception_type": "ValueError",
            "exception_message": "bad <value> & stuff",
            "exception_stacktrace": "line1\nline2 <tag>",
        }
    ]
    page = trace_exporter.render_failure_html("abc123", details)
    assert "<pre" in page
    assert "ValueError" in page
    # HTML-escaped, not raw, to avoid breaking the page / XSS.
    assert "bad &lt;value&gt; &amp; stuff" in page
    assert "line2 &lt;tag&gt;" in page
    assert "<value>" not in page


def test_render_failure_html_handles_missing_trace() -> None:
    assert "No trace selected" in trace_exporter.render_failure_html("", [])
    assert "No failing test span" in trace_exporter.render_failure_html("abc", [])


def test_github_test_url_points_at_file_and_line() -> None:
    nodeid = "tests/pipelines/test_pipelines_depth_estimation.py::DepthEstimationPipelineTests::test_multiprocess"
    stacktrace = (
        "self = <...>\n"
        "tests/pipelines/test_pipelines_depth_estimation.py:142: in test_multiprocess\n"
        "    raise ValueError('boom')\n"
    )
    url = trace_exporter.github_test_url(
        "huggingface/transformers", "abc123", nodeid, stacktrace
    )
    assert url == (
        "https://github.com/huggingface/transformers/blob/abc123/"
        "tests/pipelines/test_pipelines_depth_estimation.py#L142"
    )


def test_github_test_url_falls_back_to_main_and_omits_missing_line() -> None:
    nodeid = "tests/test_x.py::test_one"
    url = trace_exporter.github_test_url("org/repo", "", nodeid, "no line info here")
    assert url == "https://github.com/org/repo/blob/main/tests/test_x.py"


def test_github_test_url_empty_without_repository() -> None:
    assert trace_exporter.github_test_url("", "abc", "tests/test_x.py::t", "x") == ""


def test_annotate_github_links_uses_commit_sha_from_metadata() -> None:
    trace = make_trace(
        trace_id="trace-fail",
        run_id="run-1",
        job="tests_x",
        pr="45983",
        repository="huggingface/transformers",
        spans=[
            make_test_span(
                process_id="pytest-process",
                nodeid="tests/test_x.py::TestX::test_boom",
                start_time=1_000_000,
                duration=1_000_000,
                status_code="ERROR",
                exception_type="AssertionError",
            )
        ],
    )
    # make_test_span writes stacktrace "Traceback for <nodeid>" (no file:line),
    # so the URL resolves to the file without an #L anchor.
    details = trace_exporter.extract_failure_details(trace)
    trace_exporter.annotate_github_links(
        trace,
        details,
        _metadata_fetcher=lambda repo, pr: {"commit_sha": "deadbeef"},
    )
    assert details[0]["github_url"] == (
        "https://github.com/huggingface/transformers/blob/deadbeef/tests/test_x.py"
    )


def test_render_failure_html_linkifies_nodeid_when_url_present() -> None:
    details = [
        {
            "test_nodeid": "tests/test_x.py::test_boom",
            "exception_type": "ValueError",
            "exception_message": "boom",
            "exception_stacktrace": "stack",
            "github_url": "https://github.com/org/repo/blob/abc/tests/test_x.py#L9",
        }
    ]
    page = trace_exporter.render_failure_html("t", details)
    assert '<a href="https://github.com/org/repo/blob/abc/tests/test_x.py#L9"' in page
    assert 'target="_blank"' in page


# ---------------------------------------------------------------------------
# Run roll-up: complete-set computation + settle gating (no decay / no churn)
# ---------------------------------------------------------------------------


def test_run_rollup_over_complete_set_counts_all_jobs() -> None:
    rollup = trace_exporter.extract_run_rollup_metrics(
        workflow_split_across_three_jobs()
    )
    start = metric_lines(rollup, "pytest_run_start_time_seconds")
    assert len(start) == 1
    # Totals live on their own value metrics keyed by run identity.
    assert metric_lines(rollup, "pytest_run_total_tests")[0].endswith(" 4")
    assert metric_lines(rollup, "pytest_run_failed_tests")[0].endswith(" 1")
    # The split functions together equal the legacy combined emitter.
    combined = trace_exporter.extract_per_run_metrics(
        workflow_split_across_three_jobs()
    )
    assert metric_lines(combined, "pytest_run_failed_tests")[0].endswith(" 1")
    assert metric_lines(
        combined, "pytest_test_duration_seconds"
    )  # per-test still there


def test_run_membership_settle_gating() -> None:
    trace_exporter._run_members.clear()
    trace_exporter._run_last_growth.clear()
    extracted = trace_exporter._precompute_trace_rows(
        workflow_split_across_three_jobs()
    )
    trace_exporter.record_run_membership(extracted, now=1000.0)
    # Just ingested -> not settled -> nothing emitted yet (avoids churn).
    assert (
        trace_exporter.settled_runs_complete_extracted(extracted, 1000.0, 120.0) == []
    )
    # After a quiet period the run is considered complete and is emitted.
    complete = trace_exporter.settled_runs_complete_extracted(extracted, 1200.0, 120.0)
    assert len(complete) == 3


def test_run_rollup_resists_lookback_window_decay() -> None:
    """The exact bug: a run's failing job trace ages out of the window, but the
    roll-up must still report the failure (computed over the complete set)."""
    trace_exporter._run_members.clear()
    trace_exporter._run_last_growth.clear()
    trace_exporter._trace_cache.clear()

    failing = make_trace(
        trace_id="t-fail",
        run_id="run-x",
        job="job_fail",
        spans=[
            make_test_span(
                process_id="pytest-process",
                nodeid="tests/a.py::test_bad",
                start_time=1_000_000,
                duration=1_000_000,
                status_code="ERROR",
                exception_type="AssertionError",
            )
        ],
    )
    passing = make_trace(
        trace_id="t-pass",
        run_id="run-x",
        job="job_pass",
        spans=[
            make_test_span(
                process_id="pytest-process",
                nodeid="tests/b.py::test_ok",
                start_time=2_000_000,
                duration=1_000_000,
            )
        ],
    )

    # Scrape 1: both job traces visible; both settle into the trace cache.
    trace_exporter.record_run_membership(
        trace_exporter._precompute_trace_rows([failing, passing]), now=1000.0
    )
    trace_exporter._trace_cache["t-fail"] = failing
    trace_exporter._trace_cache["t-pass"] = passing

    # Scrape 2 (much later): the failing trace has aged out of the window.
    window2 = trace_exporter._precompute_trace_rows([passing])
    trace_exporter.record_run_membership(window2, now=2000.0)  # no new trace
    complete = trace_exporter.settled_runs_complete_extracted(window2, 2000.0, 120.0)

    rollup = trace_exporter.extract_run_rollup_metrics(_extracted=complete)
    assert metric_lines(rollup, "pytest_run_failed_tests")[0].endswith(" 1")
    assert metric_lines(rollup, "pytest_run_total_tests")[0].endswith(" 2")

    # Without the complete-set fix (window-only) the failure would vanish.
    window_only = trace_exporter.extract_run_rollup_metrics(_extracted=window2)
    assert metric_lines(window_only, "pytest_run_failed_tests")[0].endswith(" 0")


# ---------------------------------------------------------------------------
# Badge / summary on-demand PR fallback
# ---------------------------------------------------------------------------


def test_pr_badge_uses_payload_fast_path_without_querying_tempo(monkeypatch) -> None:
    """When the live payload already has the PR, the badge reads it directly and
    never falls back to a Tempo search."""
    trace_exporter._pr_summary_cache.clear()
    monkeypatch.delenv("PYTEST_TRACE_EXPORTER_PROMETHEUS_URL", raising=False)
    payload = "\n".join(
        trace_exporter.extract_run_rollup_metrics(workflow_split_across_three_jobs())
    )
    monkeypatch.setattr(trace_exporter, "render_metrics", lambda: payload)

    def _must_not_search(*args, **kwargs):
        raise AssertionError("Tempo search must not run when the payload has the PR")

    monkeypatch.setattr(trace_exporter, "search_trace_ids", _must_not_search)

    svg = trace_exporter.render_pr_badge_svg("4321").decode()
    assert "1 failed" in svg
    assert "4 tests" in svg
    assert "3 jobs" in svg
    assert "no data" not in svg


def test_pr_badge_falls_back_to_tempo_when_payload_has_no_pr(monkeypatch) -> None:
    """A PR aged out of the render window is no longer in the payload; the badge
    falls back to a per-PR Tempo search and still reports its latest run."""
    trace_exporter._pr_summary_cache.clear()
    monkeypatch.delenv("PYTEST_TRACE_EXPORTER_PROMETHEUS_URL", raising=False)
    monkeypatch.setattr(trace_exporter, "render_metrics", lambda: "")  # payload miss

    traces = workflow_split_across_three_jobs()
    by_id = {trace["traceID"]: trace for trace in traces}
    monkeypatch.setattr(
        trace_exporter,
        "search_trace_ids",
        lambda *a, **k: list(by_id),
    )
    monkeypatch.setattr(
        trace_exporter, "get_trace", lambda tid, base_url=None: by_id.get(tid)
    )

    svg = trace_exporter.render_pr_badge_svg("4321").decode()
    assert "1 failed" in svg
    assert "4 tests" in svg
    assert "3 jobs" in svg
    assert "no data" not in svg


def test_pr_badge_scopes_fallback_search_to_requested_pr(monkeypatch) -> None:
    """The fallback search is filtered to the requested PR via vcs.change.id, and
    an empty result renders a graceful 'no data' badge rather than erroring."""
    trace_exporter._pr_summary_cache.clear()
    monkeypatch.delenv("PYTEST_TRACE_EXPORTER_PROMETHEUS_URL", raising=False)
    monkeypatch.setattr(trace_exporter, "render_metrics", lambda: "")
    captured: dict[str, str] = {}

    def _search(base_url, service_name, start, end, limit, extra_selector=""):
        captured["selector"] = extra_selector
        return []

    monkeypatch.setattr(trace_exporter, "search_trace_ids", _search)

    svg = trace_exporter.render_pr_badge_svg("46180").decode()
    assert 'vcs.change.id = "46180"' in captured["selector"]
    assert "no data" in svg


def test_pr_fallback_result_is_memoized_per_pr(monkeypatch) -> None:
    """A miss (including 'no data') is cached, so a hammered badge does not
    re-search Tempo on every hit."""
    trace_exporter._pr_summary_cache.clear()
    monkeypatch.delenv("PYTEST_TRACE_EXPORTER_PROMETHEUS_URL", raising=False)
    monkeypatch.setattr(trace_exporter, "render_metrics", lambda: "")
    calls = {"n": 0}

    def _search(*a, **k):
        calls["n"] += 1
        return []

    monkeypatch.setattr(trace_exporter, "search_trace_ids", _search)

    trace_exporter.render_pr_badge_svg("999")
    trace_exporter.render_pr_badge_svg("999")
    assert calls["n"] == 1


def test_public_response_cache_stores_rendered_badge_bytes(monkeypatch) -> None:
    trace_exporter._public_response_cache.clear()
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_PUBLIC_RESPONSE_CACHE_SECONDS", "60")

    assert trace_exporter._public_response_cache_get("badge:4321") is None
    trace_exporter._public_response_cache_put("badge:4321", b"<svg/>")
    assert trace_exporter._public_response_cache_get("badge:4321") == b"<svg/>"


def test_public_response_cache_can_be_disabled(monkeypatch) -> None:
    trace_exporter._public_response_cache.clear()
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_PUBLIC_RESPONSE_CACHE_SECONDS", "0")

    trace_exporter._public_response_cache_put("badge:4321", b"<svg/>")

    assert trace_exporter._public_response_cache_get("badge:4321") is None
    assert trace_exporter._public_cache_control_header() == "no-store"


def test_pr_badge_color_matches_main_dashboard_failure_rate_thresholds() -> None:
    assert trace_exporter._badge_failure_color(0, 100) == "green"
    assert trace_exporter._badge_failure_color(1, 19_801) == "94B45F"
    assert trace_exporter._badge_failure_color(1, 100) == "orange"
    assert trace_exporter._badge_failure_color(10, 100) == "red"


def test_badge_fill_prefixes_hex_but_leaves_css_names_bare() -> None:
    # Hex tokens need a leading '#'; bare CSS names must not get one (that would
    # produce the invalid '#green' and the fill would fall back to black).
    assert trace_exporter._badge_fill("94B45F") == "#94B45F"
    assert trace_exporter._badge_fill("9f9f9f") == "#9f9f9f"
    assert trace_exporter._badge_fill(trace_exporter.BADGE_MERGED_COLOR).startswith("#")
    assert trace_exporter._badge_fill("green") == "green"
    assert trace_exporter._badge_fill("orange") == "orange"
    assert trace_exporter._badge_fill("red") == "red"


def _passing_pr_payload(pr: str, state_value: str | None) -> str:
    lines = [
        f'pytest_run_start_time_seconds{{service_name="s",provider="p",pr="{pr}",run_id="r1"}} 100',
        f'pytest_run_total_tests{{service_name="s",provider="p",pr="{pr}",run_id="r1"}} 10',
        f'pytest_run_failed_tests{{service_name="s",provider="p",pr="{pr}",run_id="r1"}} 0',
        f'pytest_run_job_count{{service_name="s",provider="p",pr="{pr}",run_id="r1"}} 2',
    ]
    if state_value is not None:
        lines.append(
            f'pytest_pr_state{{pr="{pr}",repository="x",service_name="s"}} {state_value}'
        )
    return "\n".join(lines)


def test_pr_badge_open_passing_run_is_green(monkeypatch) -> None:
    trace_exporter._pr_summary_cache.clear()
    monkeypatch.delenv("PYTEST_TRACE_EXPORTER_PROMETHEUS_URL", raising=False)
    monkeypatch.setattr(
        trace_exporter, "render_metrics", lambda: _passing_pr_payload("4321", "1")
    )
    svg = trace_exporter.render_pr_badge_svg("4321").decode()
    assert 'fill="green"' in svg
    assert "0 failed" in svg
    assert "merged" not in svg


def test_pr_badge_merged_passing_run_is_merged_blue(monkeypatch) -> None:
    trace_exporter._pr_summary_cache.clear()
    monkeypatch.delenv("PYTEST_TRACE_EXPORTER_PROMETHEUS_URL", raising=False)
    monkeypatch.setattr(
        trace_exporter, "render_metrics", lambda: _passing_pr_payload("4321", "2")
    )
    svg = trace_exporter.render_pr_badge_svg("4321").decode()
    assert f'fill="#{trace_exporter.BADGE_MERGED_COLOR}"' in svg
    assert "merged" in svg
    assert 'fill="green"' not in svg


def test_pr_badge_closed_unmerged_passing_run_is_grey(monkeypatch) -> None:
    trace_exporter._pr_summary_cache.clear()
    monkeypatch.delenv("PYTEST_TRACE_EXPORTER_PROMETHEUS_URL", raising=False)
    monkeypatch.setattr(
        trace_exporter, "render_metrics", lambda: _passing_pr_payload("4321", "0")
    )
    svg = trace_exporter.render_pr_badge_svg("4321").decode()
    assert f'fill="#{trace_exporter.BADGE_CLOSED_COLOR}"' in svg
    assert "closed" in svg
    assert "merged" not in svg
    assert 'fill="green"' not in svg


def test_pr_badge_unknown_state_passing_run_stays_green(monkeypatch) -> None:
    trace_exporter._pr_summary_cache.clear()
    monkeypatch.delenv("PYTEST_TRACE_EXPORTER_PROMETHEUS_URL", raising=False)
    monkeypatch.setattr(
        trace_exporter, "render_metrics", lambda: _passing_pr_payload("4321", None)
    )
    svg = trace_exporter.render_pr_badge_svg("4321").decode()
    assert 'fill="green"' in svg
    assert "merged" not in svg


def test_pr_badge_uses_prometheus_rollups_before_tempo(monkeypatch) -> None:
    """When the live payload misses, query Prometheus rollups before doing the
    expensive Tempo search+full-trace fetch fallback."""
    trace_exporter._pr_summary_cache.clear()
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_PROMETHEUS_URL", "http://prometheus:9090")
    monkeypatch.setattr(trace_exporter, "render_metrics", lambda: "")
    captured: dict[str, str] = {}

    def _query(url):
        captured["url"] = url
        labels = {
            "pr": "4321",
            "provider": "github_actions",
            "run_id": "newer",
            "service_name": "pytest-observability-demo",
        }
        values = {
            "pytest_run_start_time_seconds": "200",
            "pytest_run_end_time_seconds": "260",
            "pytest_run_total_tests": "4",
            "pytest_run_failed_tests": "1",
            "pytest_run_duration_seconds": "12.5",
            "pytest_run_job_count": "3",
        }
        return {
            "status": "success",
            "data": {
                "result": [
                    {"metric": {"__name__": name, **labels}, "value": [0, value]}
                    for name, value in values.items()
                ]
            },
        }

    def _must_not_search(*args, **kwargs):
        raise AssertionError("Tempo search must not run when Prometheus has rollups")

    monkeypatch.setattr(trace_exporter, "_http_get_json", _query)
    monkeypatch.setattr(trace_exporter, "search_trace_ids", _must_not_search)

    svg = trace_exporter.render_pr_badge_svg("4321").decode()
    assert "/api/v1/query?" in captured["url"]
    assert "last_over_time" in captured["url"]
    assert "1 failed" in svg
    assert "4 tests" in svg
    assert "3 jobs" in svg


def test_pr_badge_prefers_prometheus_job_rollups_for_latest_run(monkeypatch) -> None:
    """The PR dashboard derives latest-run failures from job rollups, so the
    badge should not underreport when run-level and job-level rollups diverge."""
    trace_exporter._pr_summary_cache.clear()
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_PROMETHEUS_URL", "http://prometheus:9090")
    monkeypatch.setattr(trace_exporter, "render_metrics", lambda: "")

    def _query(url):
        labels = {
            "pr": "45638",
            "provider": "github_actions",
            "run_id": "latest",
            "service_name": "pytest-observability",
        }
        rows = [
            ("pytest_run_start_time_seconds", labels, "200"),
            ("pytest_run_end_time_seconds", labels, "260"),
            ("pytest_run_total_tests", labels, "4"),
            ("pytest_run_failed_tests", labels, "0"),
            ("pytest_run_duration_seconds", labels, "12.5"),
            ("pytest_run_job_count", labels, "2"),
            (
                "pytest_run_job_total_tests",
                {**labels, "test_job": "tests_non_model"},
                "3",
            ),
            (
                "pytest_run_job_failed_tests",
                {**labels, "test_job": "tests_non_model"},
                "1",
            ),
            (
                "pytest_run_job_total_tests",
                {**labels, "test_job": "tests_torch"},
                "1",
            ),
            (
                "pytest_run_job_failed_tests",
                {**labels, "test_job": "tests_torch"},
                "0",
            ),
        ]
        return {
            "status": "success",
            "data": {
                "result": [
                    {"metric": {"__name__": name, **metric_labels}, "value": [0, value]}
                    for name, metric_labels, value in rows
                ]
            },
        }

    def _must_not_search(*args, **kwargs):
        raise AssertionError("Tempo search must not run when Prometheus has rollups")

    monkeypatch.setattr(trace_exporter, "_http_get_json", _query)
    monkeypatch.setattr(trace_exporter, "search_trace_ids", _must_not_search)

    svg = trace_exporter.render_pr_badge_svg("45638").decode()
    assert "1 failed" in svg
    assert "4 tests" in svg
    assert "2 jobs" in svg
