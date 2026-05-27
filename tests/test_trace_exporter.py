from __future__ import annotations

import io
import json
from unittest.mock import patch

from transformersci.otel import trace_exporter


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
) -> dict:
    return {
        "processes": {
            "pytest-process": {
                "serviceName": service_name,
                "tags": [
                    make_tag("transformers.test.provider", provider),
                    make_tag("transformers.test.run.id", run_id),
                    make_tag(job_tag_key, job),
                    make_tag("vcs.change.id", pr),
                    make_tag("vcs.change.url", pr_url),
                    make_tag("vcs.repository.name", repository),
                ],
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
    assert 'job_count="3"' in run_start_lines[0]
    assert 'jobs="tests_flax,tests_tf,tests_torch"' in run_start_lines[0]
    assert 'trace_count="3"' in run_start_lines[0]
    assert 'total_tests="4"' in run_start_lines[0]
    assert 'failed_tests="1"' in run_start_lines[0]
    assert run_start_lines[0].endswith(" 1.000000")

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

    duration_lines = metric_lines(metrics, "pytest_test_duration_seconds")
    assert len(duration_lines) == 4
    assert all('run_id="12345:2"' in line for line in duration_lines)
    assert any('trace_id="trace-torch"' in line for line in duration_lines)
    assert any('trace_id="trace-tf"' in line for line in duration_lines)
    assert any('trace_id="trace-flax"' in line for line in duration_lines)


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
        "reviewers": "carol,alice,bob",
        "state": "open",
        "title": "Fix dashboard metadata",
    }


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
    # Old timestamps so the trace is "settled" and gets memoized.
    payload = make_otlp_trace(
        nodeid="tests/test_torch.py::TestTorch::test_one",
        start_nano=1_000_000_000,
        end_nano=2_000_000_000,
        status_code="STATUS_CODE_OK",
    )
    opener = _tempo_urlopen(["trace-torch"], payload)
    with patch.dict(
        "os.environ",
        {"PYTEST_TRACE_EXPORTER_TRACE_SETTLE_SECONDS": "60"},
        clear=False,
    ):
        with patch(
            "transformersci.otel.trace_exporter.urlopen", side_effect=opener
        ) as mocked:
            trace_exporter.fetch_traces()
            first_calls = mocked.call_count
            trace_exporter.fetch_traces()
            second_calls = mocked.call_count

    # First fetch: 1 search + 1 trace fetch = 2. Second fetch: only the search,
    # because the settled trace is served from the in-memory cache.
    assert first_calls == 2
    assert second_calls == 3


def test_last_failure_metrics_omit_stacktrace_but_keep_pointers() -> None:
    metrics = trace_exporter.extract_average_metrics(workflow_split_across_three_jobs())
    pointer_lines = metric_lines(metrics, "pytest_test_last_failure_info")
    assert len(pointer_lines) == 1
    assert "stacktrace=" not in pointer_lines[0]
    assert 'exception_type="AssertionError"' in pointer_lines[0]
    assert 'trace_id="trace-torch"' in pointer_lines[0]

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
