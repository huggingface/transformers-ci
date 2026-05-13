from __future__ import annotations

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
    provider: str = "github_actions",
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
