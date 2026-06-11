from __future__ import annotations

import io
import json
from unittest.mock import patch

import pytest

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

    # Numeric state gauge: one series per PR, open -> 1 (closed -> 0). Lets the
    # dashboard last_over_time() the current state without a multi-valued label.
    state_lines = metric_lines(metrics, "pytest_pr_state")
    assert len(state_lines) == 1
    assert 'pr="4321"' in state_lines[0]
    assert state_lines[0].endswith(" 1")


def test_extract_pr_info_metrics_state_gauge_closed_is_zero() -> None:
    metrics = trace_exporter.extract_pr_info_metrics(
        workflow_split_across_three_jobs(),
        _metadata_fetcher=lambda repo, pr: {"state": "closed"},
    )
    state_lines = metric_lines(metrics, "pytest_pr_state")
    assert len(state_lines) == 1
    assert state_lines[0].endswith(" 0")


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
    monkeypatch.setattr(trace_exporter, "iter_traces", lambda *a, **k: iter([trace]))

    out = trace_exporter._render_metrics_uncached()

    assert "pytest_trace_exporter_render_duration_seconds " in out
    assert "pytest_trace_exporter_traces_processed_total " in out
    assert "pytest_trace_exporter_trace_count 1" in out
    # RSS is Linux-only (/proc); assert it only where the helper can read it.
    if trace_exporter._process_resident_bytes() is not None:
        assert "pytest_trace_exporter_process_resident_bytes " in out


def test_render_emits_self_metrics_on_empty_window(monkeypatch) -> None:
    # An idle exporter (no traces in the window) is still healthy: its own
    # health metrics must stay populated so the dashboard shows "idle", not
    # "no data".
    monkeypatch.setattr(trace_exporter, "iter_traces", lambda *a, **k: iter([]))
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
    monkeypatch.setattr(trace_exporter, "iter_traces", lambda *a, **k: iter([trace]))
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
    monkeypatch.setattr(trace_exporter, "iter_traces", lambda *a, **k: iter([]))
    trace_exporter._refresh_cache_once()
    body = payload_file.read_text(encoding="utf-8")
    assert body.endswith("\n")
    assert "pytest_trace_exporter_up 1" in body
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != payload_file.name]
    assert leftovers == []


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
    monkeypatch.setattr(trace_exporter, "iter_traces", lambda *a, **k: iter([trace]))
    trace_exporter._refresh_cache_once()
    good = payload_file.read_text(encoding="utf-8")
    assert "pytest_test_duration_seconds{" in good

    def boom(*_a, **_k):
        raise RuntimeError("render crashed")

    monkeypatch.setattr(trace_exporter, "_render_metrics_uncached", boom)
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


def test_trace_cache_is_bounded_lru(monkeypatch) -> None:
    # The settled-trace cache must not grow without bound (it holds full
    # multi-MB traces; unbounded growth crept the exporter RSS toward its cap).
    trace_exporter._trace_cache.clear()
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_TRACE_CACHE_MAX", "3")
    # Old timestamps -> "settled" -> memoized; same payload shape for every id.
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


def test_trace_cache_is_bounded_by_bytes(monkeypatch) -> None:
    trace_exporter._trace_cache.clear()
    trace_exporter._trace_cache_sizes.clear()
    trace_exporter._trace_cache_bytes = 0
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_TRACE_CACHE_MAX", "10")
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_TRACE_CACHE_MAX_BYTES", "1200")
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
        assert trace_exporter._trace_cache_bytes <= 1200
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
