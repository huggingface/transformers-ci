from __future__ import annotations

import io
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
    suite: str,
    spans: list[dict],
    pr: str = "4321",
    pr_url: str = "https://github.com/huggingface/transformers/pull/4321",
    provider: str = "github_actions",
    repository: str = "huggingface/transformers",
    service_name: str = "transformers-tests",
) -> dict:
    return {
        "processes": {
            "pytest-process": {
                "serviceName": service_name,
                "tags": [
                    make_tag("transformers.test.provider", provider),
                    make_tag("transformers.test.run.id", run_id),
                    make_tag("transformers.test.suite", suite),
                    make_tag("vcs.change.id", pr),
                    make_tag("vcs.change.url", pr_url),
                    make_tag("vcs.repository.name", repository),
                ],
            }
        },
        "spans": spans,
        "traceID": trace_id,
    }


def workflow_split_across_three_suites() -> list[dict]:
    process_id = "pytest-process"
    return [
        make_trace(
            trace_id="trace-torch",
            run_id="12345:2",
            suite="tests_torch",
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
            suite="tests_tf",
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
            suite="tests_flax",
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


def test_extract_per_run_metrics_aggregates_suite_traces_into_one_run() -> None:
    metrics = trace_exporter.extract_per_run_metrics(
        workflow_split_across_three_suites()
    )

    run_start_lines = metric_lines(metrics, "pytest_run_start_time_seconds")
    assert len(run_start_lines) == 1
    assert 'run_id="12345:2"' in run_start_lines[0]
    assert 'suite_count="3"' in run_start_lines[0]
    assert 'suites="tests_flax,tests_tf,tests_torch"' in run_start_lines[0]
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

    suite_member_lines = metric_lines(metrics, "pytest_run_suite_member_info")
    assert len(suite_member_lines) == 3
    assert any('test_suite="tests_torch"' in line for line in suite_member_lines)
    assert any('test_suite="tests_tf"' in line for line in suite_member_lines)
    assert any('test_suite="tests_flax"' in line for line in suite_member_lines)

    duration_lines = metric_lines(metrics, "pytest_test_duration_seconds")
    assert len(duration_lines) == 4
    assert all('run_id="12345:2"' in line for line in duration_lines)
    assert any('trace_id="trace-torch"' in line for line in duration_lines)
    assert any('trace_id="trace-tf"' in line for line in duration_lines)
    assert any('trace_id="trace-flax"' in line for line in duration_lines)


def test_extract_pr_last_failure_metrics_links_failure_back_to_run() -> None:
    metrics = trace_exporter.extract_pr_last_failure_metrics(
        workflow_split_across_three_suites()
    )

    failure_lines = metric_lines(metrics, "pytest_pr_last_failure_info")
    assert len(failure_lines) == 1
    assert 'pr="4321"' in failure_lines[0]
    assert 'run_id="12345:2"' in failure_lines[0]
    assert 'trace_id="trace-torch"' in failure_lines[0]
    assert 'test_suite="tests_torch"' in failure_lines[0]


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
        workflow_split_across_three_suites(),
        _metadata_fetcher=metadata_fetcher,
    )

    info_lines = metric_lines(metrics, "pytest_pr_info")
    assert len(info_lines) == 1
    assert calls == [("huggingface/transformers", "4321")]
    assert 'author="octocat"' in info_lines[0]
    assert 'commit_sha="deadbeefcafebabe1234567890abcdef00000000"' in info_lines[0]
    assert 'html_url="https://github.com/huggingface/transformers/pull/4321"' in info_lines[0]
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
        workflow_split_across_three_suites(),
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
        "    raise ValueError(\"boom\")\n"
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
            suite="tests_a",
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
            suite="tests_b",
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
    assert 'html_url="https://github.com/huggingface/transformers/pull/45983"' in info_lines[0]
