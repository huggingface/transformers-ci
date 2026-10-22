import gzip
import json
import threading
from collections import OrderedDict
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from transformersci.otel import mic_page, trace_exporter

NODE = "tests/contracts::mic-demo::mic-demo-slim[bfloat16]"


def vector(labels, value=1):
    return {"metric": labels, "value": [1000, str(value)]}


def failure(run="123:1", trace="a" * 32, node=NODE):
    return vector(
        {
            "run_id": run,
            "trace_id": trace,
            "test_nodeid": node,
            "pr": "42",
            "exception_type": "MICAssertionError",
        }
    )


def queries(failures):
    def query(expression):
        if "pytest_run_start_time_seconds" in expression:
            return [
                vector({"run_id": "123:1"}, 1000),
                vector({"run_id": "124:1"}, 2000),
            ]
        if "pytest_run_job_" in expression:
            return []
        if "pytest_pr_info" in expression:
            return [vector({"pr": "42", "title": "Fix demo attention"})]
        assert 'test_job="fast-integration"' in expression
        return failures

    return query


def detail(**extra):
    return {
        "test_job": "fast-integration",
        "trace_id": "a" * 32,
        "test_nodeid": NODE,
        **extra,
    }


def test_rows_join_case_metadata_and_link_pinned_baseline():
    calls = []
    row = detail(
        **{
            "mic.target": "mic-demo-slim",
            "mic.contract": "huggingface-mic/demo-mic",
            "mic.variant": "bfloat16",
            "mic.platform": "cpu-x86_64-bf16",
            "mic.status": "fail",
            "mic.failure": "logits: max delta 0.01 over tolerance",
            "mic.selected_by": "models/demo",
            "mic.baseline_repo": "huggingface-mic/demo-mic",
            "mic.baseline_revision": "b" * 40,
            "mic.baseline_path": "integrations/transformers/expectations/cpu-x86_64-bf16/demo.json",
        }
    )

    def load(run):
        calls.append(run)
        return [row]

    data = mic_page.build_rows(queries([failure(), failure()]), load)
    assert calls == ["123:1"]
    assert data["total"] == 1
    case = data["rows"][0]
    assert case["failure"].startswith("logits:")
    assert case["platform"] == "cpu-x86_64-bf16"
    assert case["selected_by"] == "models/demo"
    assert f"/blob/{'b' * 40}/integrations/" in case["baseline_url"]
    assert case["run_url"].startswith("/d/pytest-observability-by-run/")
    assert "var-run_id=123%3A1" in case["run_url"]
    assert case["trace_url"].startswith("/d/pytest-test/")
    assert "var-trace_id=" in case["trace_url"]
    assert case["pr_title"] == "Fix demo attention"
    assert "var-pr=42" in case["pr_url"]


def test_older_rows_work_without_mic_metadata_and_newest_runs_sort_first():
    data = mic_page.build_rows(queries([failure(), failure("124:1")]), lambda run: None)
    assert [r["run_id"] for r in data["rows"]] == ["124:1"]
    assert data["rows"][0]["fails"] == 2
    assert data["rows"][0]["target"] == "mic-demo-slim"
    assert data["rows"][0]["variant"] == "bfloat16"
    assert data["rows"][0]["baseline_url"] == ""


@pytest.mark.parametrize("status", ["pass", "xfail", "unsupported"])
def test_expected_or_unsupported_cases_are_excluded(status):
    data = mic_page.build_rows(
        queries([failure()]), lambda run: [detail(**{"mic.status": status})]
    )
    assert data["rows"] == []


def test_malformed_ids_and_other_tests_are_excluded():
    data = mic_page.build_rows(
        queries([failure("../etc/passwd"), failure(node="tests/models/demo.py::test")]),
        lambda run: pytest.fail("unexpected load"),
    )
    assert not data["rows"]
    assert mic_page.parse_days("7] malicious") == 7


def test_trace_extraction_and_persistence_keep_mic_details_only(tmp_path):
    def tags(values):
        return [{"key": key, "value": value} for key, value in values.items()]

    trace = {
        "traceID": "a" * 32,
        "processes": {
            "p": {
                "serviceName": "pytest-observability",
                "tags": tags(
                    {
                        "transformers.test.run.id": "123:1",
                        "transformers.test.job": "fast-integration",
                        "vcs.repository.name": "huggingface/transformers",
                        "vcs.ref.head.revision": "c" * 40,
                    }
                ),
            }
        },
        "spans": [
            {
                "operationName": NODE,
                "processID": "p",
                "startTime": 1000,
                "duration": 500,
                "tags": tags(
                    {
                        "pytest.nodeid": NODE,
                        "pytest.span_type": "test",
                        "otel.status_code": "ERROR",
                        "mic.target": "mic-demo-slim",
                        "mic.status": "fail",
                        "mic.platform": "cpu-x86_64-bf16",
                    }
                ),
                "logs": [
                    {
                        "fields": tags(
                            {
                                "event": "exception",
                                "exception.type": "MICAssertionError",
                                "exception.stacktrace": "logits: over tolerance",
                            }
                        )
                    }
                ],
            }
        ],
    }
    _, rows = trace_exporter.extract_trace_rows(trace)
    trace_exporter.persist_run_rows("123:1", rows, directory=str(tmp_path))
    saved = trace_exporter.load_run_rows("123:1", directory=str(tmp_path))[0]
    mic_saved = trace_exporter.load_run_rows("123:1", directory=str(tmp_path / "mic"))
    assert mic_saved == [saved]
    assert saved["mic.failure"] == "logits: over tolerance"
    assert saved["mic.platform"] == "cpu-x86_64-bf16"
    assert saved["mic.commit_sha"] == "c" * 40
    trace["processes"]["p"]["tags"] = tags({"transformers.test.job": "tests_torch"})
    _, rows = trace_exporter.extract_trace_rows(trace)
    assert not any(key.startswith("mic.") for key in rows[0])


def test_http_page_data_caching_and_unavailable_state(monkeypatch):
    calls = []

    def query(expression):
        calls.append(expression)
        return []

    monkeypatch.setattr(trace_exporter, "_prometheus_vector", query)
    monkeypatch.setattr(trace_exporter, "_swr_cache", OrderedDict())
    monkeypatch.setattr(trace_exporter, "_swr_inflight", {})
    server = ThreadingHTTPServer(("127.0.0.1", 0), trace_exporter.MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        assert b"MIC failures" in urlopen(base + "/mic").read()
        assert (
            urlopen(base + "/mic/app.js")
            .headers["Content-Type"]
            .startswith("text/javascript")
        )
        assert json.load(urlopen(base + "/mic/data?days=1"))["rows"] == []
        urlopen(base + "/mic/data?days=1").read()
        assert len(calls) == 4
        urlopen(base + "/mic/data?days=1&failure_type=Runtime%20traceback").read()
        assert len(calls) == 8
        urlopen(base + "/mic/data?days=1&failure_type=Runtime%20traceback").read()
        assert len(calls) == 8
        assert json.load(urlopen(base + "/mic/data?days=1&group=pr"))["group"] == "pr"
        assert len(calls) == 12
        assert json.load(urlopen(base + "/mic/data?days=1"))["group"] == "model"
        assert len(calls) == 12

        def broken(expression):
            raise RuntimeError("private internal URL")

        monkeypatch.setattr(trace_exporter, "_prometheus_vector", broken)
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/mic/data?days=30")
        assert error.value.code == 502
        assert b"private internal URL" not in error.value.read()
    finally:
        server.shutdown()
        server.server_close()


def test_mic_cache_write_failure_does_not_drop_main_run_rows(tmp_path, monkeypatch):
    original = trace_exporter.os.makedirs

    def makedirs(path, **kwargs):
        if str(path).endswith("/mic"):
            raise PermissionError("MIC cache is not writable")
        return original(path, **kwargs)

    monkeypatch.setattr(trace_exporter.os, "makedirs", makedirs)
    trace_exporter.persist_run_rows("123:1", [detail()], directory=str(tmp_path))
    assert len(trace_exporter.load_run_rows("123:1", directory=str(tmp_path))) == 1


def test_recovers_assertion_and_model_provenance_for_older_rows():
    def trace_rows(trace):
        assert trace == "a" * 32
        return [
            detail(
                **{
                    "run_id": "123:1",
                    "mic.failure": "logits: max delta 0.01 over tolerance",
                    "mic.fixture_repo": "huggingface-mic/demo-mic",
                    "mic.revision": "b" * 40,
                }
            )
        ]

    def build(pin):
        assert pin == ("huggingface-mic/demo-mic", "b" * 40)
        return {"source_model": "creator/Large-Demo-24B", "weights": "overfit"}

    row = mic_page.build_rows(
        queries([failure()]),
        lambda run: None,
        trace_rows=trace_rows,
        build_metadata=build,
    )["rows"][0]
    assert row["failure"].startswith("logits:")
    assert row["weights"] == "overfit"
    assert row["source_model_url"] == "https://huggingface.co/creator/Large-Demo-24B"
    assert row["fixture_repo_url"] == "https://huggingface.co/huggingface-mic/demo-mic"


def test_variant_and_platform_filters_apply_before_row_limit(monkeypatch):
    monkeypatch.setattr(mic_page, "LIMIT", 1)
    newer = failure("124:1", node="tests/contracts::demo::demo[default]")
    records = {
        "123:1": [detail(**{"mic.platform": "cpu", "mic.variant": "bfloat16"})],
        "124:1": [
            {
                **detail(),
                "test_nodeid": newer["metric"]["test_nodeid"],
                "mic.platform": "cuda",
                "mic.variant": "default",
            }
        ],
    }
    data = mic_page.build_rows(
        queries([failure(), newer]), records.get, variant="bfloat16", platform="cpu"
    )
    assert [row["run_id"] for row in data["rows"]] == ["123:1"]
    assert data["options"] == {
        "failure_type": ["Baseline assertion"],
        "variant": ["bfloat16", "default"],
        "platform": ["cpu", "cuda"],
    }


def test_model_limit_keeps_all_error_cases_of_each_selected_model(monkeypatch):
    monkeypatch.setattr(mic_page, "LIMIT", 1)
    other = failure(node="tests/contracts::other::other[default]")
    older = failure()
    newer = failure("124:1", node="tests/contracts::demo::demo[eager]")

    def record(item, repo):
        return detail(
            test_nodeid=item["metric"]["test_nodeid"], **{"mic.fixture_repo": repo}
        )

    records = {
        "123:1": [record(other, "creator/other"), record(older, "creator/demo")],
        "124:1": [record(newer, "creator/demo")],
    }
    data = mic_page.build_rows(queries([other, older, newer]), records.get)
    assert data["models"] == 2
    assert data["cases"] == 3
    assert len(data["rows"]) == 2
    assert {row["fixture_repo"] for row in data["rows"]} == {"creator/demo"}


def test_pr_group_limit_keeps_all_models_and_includes_merge_queue_cases(monkeypatch):
    monkeypatch.setattr(mic_page, "LIMIT", 1)
    newest = failure("124:1")
    queued = failure(node="tests/contracts::other::other[default]")
    queued["metric"]["pr"] = "gh-readonly-queue/main/pr-42-abcdef"
    unrelated = failure(node="tests/contracts::unrelated::unrelated[default]")
    unrelated["metric"]["pr"] = "43"

    def record(item, repo):
        return detail(
            test_nodeid=item["metric"]["test_nodeid"], **{"mic.fixture_repo": repo}
        )

    records = {
        "123:1": [
            record(queued, "creator/other"),
            record(unrelated, "creator/unrelated"),
        ],
        "124:1": [record(newest, "creator/demo")],
    }
    data = mic_page.build_rows(
        queries([queued, unrelated, newest]), records.get, group="pr"
    )
    assert data["group"] == "pr"
    assert data["groups"] == 2
    assert data["cases"] == 3
    assert {row["pr_number"] for row in data["rows"]} == {"42"}
    assert {row["fixture_repo"] for row in data["rows"]} == {
        "creator/demo",
        "creator/other",
    }


def test_invalid_group_falls_back_to_model():
    assert (
        mic_page.build_rows(queries([]), lambda run: None, group="invalid")["group"]
        == "model"
    )


@pytest.mark.parametrize(
    "scope,matcher",
    [("main", ',pr="main"'), ("prs", ',pr=~"[0-9]+|gh-readonly-queue/.*"')],
)
def test_scope_is_applied_to_failure_query(scope, matcher):
    expressions = []
    mic_page.build_rows(
        lambda expression: expressions.append(expression) or [],
        lambda run: None,
        scope=scope,
    )
    assert matcher in expressions[0]


def test_merge_queue_rows_link_to_the_original_pr():
    row = failure()
    row["metric"]["pr"] = "gh-readonly-queue/main/pr-42-abcdef"
    result = mic_page.build_rows(queries([row]), lambda run: None)["rows"][0]
    assert result["pr_number"] == "42"
    assert result["pr_title"] == "Fix demo attention"
    assert result["pr_url"].endswith("var-pr=42")


def test_daily_strip_counts_distinct_runs_and_keeps_latest_failure():
    day = 86400
    today = 100 * day

    def query(expression):
        if "pytest_run_start_time_seconds" in expression:
            return [
                vector({"run_id": "123:1"}, today - day),
                vector({"run_id": "124:1"}, today + 60),
            ]
        if "pytest_run_job_" in expression:
            return []
        if "pytest_pr_info" in expression:
            return []
        return [failure(), failure(), failure("124:1")]

    data = mic_page.build_rows(query, lambda run: None, now=today + 3600, days=7)
    assert len(data["rows"]) == 1
    row = data["rows"][0]
    assert row["days"] == [0, 0, 0, 0, 0, 1, 1]
    assert row["fails"] == 2
    assert row["run_id"] == "124:1"
    assert data["first_day"] == 94 * day


@pytest.mark.parametrize("scope,other_branch", [("main", "42"), ("all", "42")])
def test_daily_history_observes_passes_without_clearing_a_different_branch(
    scope, other_branch
):
    today = 100 * 86400
    old = failure()
    old["metric"]["pr"] = "main"
    records = {
        "123:1": [
            detail(
                **{"mic.fixture_repo": "huggingface-mic/demo-mic", "mic.status": "fail"}
            )
        ],
        "124:1": [
            detail(
                **{"mic.fixture_repo": "huggingface-mic/demo-mic", "mic.status": "pass"}
            )
        ],
        "125:1": [
            detail(
                **{
                    "mic.fixture_repo": "huggingface-mic/demo-mic",
                    "mic.status": "error",
                }
            )
        ],
    }

    def query(expression):
        if "pytest_run_start_time_seconds" in expression:
            return [
                vector({"run_id": f"{run}:1"}, today + run) for run in (123, 124, 125)
            ]
        if "max by (run_id, pr)" in expression:
            assert ('pr="main"' in expression) == (scope == "main")
            return [
                vector({"run_id": "124:1", "pr": "main"}),
                vector({"run_id": "125:1", "pr": other_branch}),
            ]
        if "pytest_run_job_" in expression or "pytest_pr_info" in expression:
            return []
        return [old, old]

    row = mic_page.build_rows(query, records.get, now=today + 200, scope=scope)["rows"][
        0
    ]
    assert row["daily"][-1] == {"pass": 1, "fail": 1, "skipped": 0}
    assert row["latest"]["run_id"] == "124:1"
    assert row["latest"]["status"] == "pass"
    assert row["sticky"] is False
    assert "var-pr=main" in row["latest"]["run_url"]
    assert (
        mic_page.build_rows(
            query, records.get, now=today + 200, scope=scope, sticky_only=True
        )["rows"]
        == []
    )


def test_missing_case_records_do_not_infer_recovery_from_a_green_job():
    def query(expression):
        if "max by (run_id, pr)" in expression:
            return [vector({"run_id": "124:1", "pr": "42"})]
        return queries([failure()])(expression)

    row = mic_page.build_rows(query, lambda run: None, sticky_only=True)["rows"][0]
    assert row["sticky"] is True
    assert row["latest"]["run_id"] == "123:1"
    assert not any(day["pass"] for day in row["daily"])


@pytest.mark.parametrize("complete", [True, False])
def test_setup_error_clears_only_after_observing_a_complete_successful_job(complete):
    setup = failure(node="tests/contracts::runner")

    def query(expression):
        if "max by (run_id, pr)" in expression:
            return [vector({"run_id": "124:1", "pr": "42"}, 1 if complete else 2)]
        return queries([setup])(expression)

    def load(run):
        return [detail(**{"mic.status": "pass"})] if run == "124:1" else None

    row = mic_page.build_rows(query, load)["rows"][0]
    assert row["sticky"] is not complete
    assert row["latest"]["status"] == ("pass" if complete else "fail")


def test_contract_link_uses_the_contract_revision_instead_of_fixture_revision():
    row = detail(
        **{
            "mic.contract": "huggingface-mic/demo-mic",
            "mic.contract_revision": "c" * 40,
            "mic.revision": "d" * 40,
        }
    )
    result = mic_page.build_rows(queries([failure()]), lambda run: [row])["rows"][0]
    assert (
        result["contract_url"]
        == f"https://huggingface.co/huggingface-mic/demo-mic/blob/{'c' * 40}/integrations/transformers/usage.md"
    )


def test_multi_select_filters_combine_values_with_or_and_dimensions_with_and():
    other = failure("124:1", node="tests/contracts::demo::demo[default]")
    other["metric"]["exception_type"] = "MICExecutionError"
    records = {
        "123:1": [detail(**{"mic.variant": "bfloat16", "mic.platform": "cpu"})],
        "124:1": [
            detail(
                test_nodeid=other["metric"]["test_nodeid"],
                **{
                    "mic.variant": "default",
                    "mic.platform": "cuda",
                    "mic.status": "error",
                },
            )
        ],
    }
    rows = mic_page.build_rows(
        queries([failure(), other]),
        records.get,
        variant=["default", "bfloat16"],
        platform=["cpu", "cuda"],
        failure_type=["Runtime traceback", "Baseline assertion"],
    )["rows"]
    assert len(rows) == 2
    rows = mic_page.build_rows(
        queries([failure(), other]),
        records.get,
        variant=["default", "bfloat16"],
        platform=["cpu"],
    )["rows"]
    assert [row["run_id"] for row in rows] == ["123:1"]


@pytest.mark.parametrize(
    "status,exception,target,message,expected",
    [
        (
            "fail",
            "MICAssertionError",
            "demo",
            "logits: max delta 0.01",
            "Baseline assertion",
        ),
        (
            "error",
            "MICExecutionError",
            "demo",
            "Traceback: generate failed",
            "Runtime traceback",
        ),
        (
            "fail",
            "MICExecutionError",
            "demo",
            "Older execution failure",
            "Runtime traceback",
        ),
        ("error", "MICRunnerError", "runner", "Manifest invalid", "Setup error"),
        (
            "fail",
            "MICAssertionError",
            "demo",
            "no reviewed baseline at path",
            "Missing baseline",
        ),
        (
            "fail",
            "MICAssertionError",
            "demo",
            "listed in known_failures but passes; remove it",
            "Unexpected pass",
        ),
    ],
)
def test_failure_kind_distinguishes_traceback_from_baseline(
    status, exception, target, message, expected
):
    assert mic_page.failure_kind(status, exception, target, message) == expected


def test_recovered_details_survive_writes_from_older_main_rows(tmp_path):
    row = detail(**{"mic.failure": "logits: over tolerance", "mic.weights": "random"})
    directory = str(tmp_path / "mic")
    trace_exporter._persist_mic_rows("123:1", [row], directory)
    trace_exporter._persist_mic_rows("123:1", [detail()], directory)
    stored = trace_exporter.load_run_rows("123:1", directory=directory)[0]
    assert stored["mic.failure"] == "logits: over tolerance"
    assert stored["mic.weights"] == "random"


def test_fixture_provenance_is_available_without_public_network(monkeypatch):
    monkeypatch.setattr(
        trace_exporter,
        "_http_get_json",
        lambda *a, **k: pytest.fail("public network lookup"),
    )
    for pin in set(mic_page.fixture_catalog().values()):
        metadata = trace_exporter._mic_build_metadata(pin)
        assert metadata["source_model"]
        assert metadata["weights"] in {"random", "overfit", "unknown"}
    assert trace_exporter._mic_build_metadata(("unknown/model", "f" * 40)) == {}


def test_failure_type_filter_precedes_grouping_and_row_limit(monkeypatch):
    monkeypatch.setattr(mic_page, "LIMIT", 1)
    older = failure()
    newer = failure("124:1")
    newer["metric"]["exception_type"] = "MICExecutionError"
    records = {
        "123:1": [detail(**{"mic.status": "fail", "mic.failure": "logits: mismatch"})],
        "124:1": [
            detail(
                **{
                    "mic.status": "error",
                    "mic.failure": "RuntimeError: generation failed",
                }
            )
        ],
    }
    for kind, run in [("Baseline assertion", "123:1"), ("Runtime traceback", "124:1")]:
        data = mic_page.build_rows(
            queries([older, newer]), records.get, failure_type=kind
        )
        assert [row["run_id"] for row in data["rows"]] == [run]
        assert data["total"] == data["cases"] == data["rows"][0]["fails"] == 1
        assert data["options"]["failure_type"] == [
            "Baseline assertion",
            "Runtime traceback",
        ]
    assert (
        mic_page.build_rows(
            queries([older, newer]), records.get, failure_type="Missing baseline"
        )["rows"]
        == []
    )


@pytest.mark.parametrize(
    "scope,matcher",
    [
        ("all", None),
        ("main", 'pr="main"'),
        ("prs", 'pr=~"[0-9]+|gh-readonly-queue/.*"'),
    ],
)
def test_failure_rate_uses_weighted_check_totals(scope, matcher):
    def query(expression):
        assert "[14d]" in expression
        assert 'test_job="fast-integration"' in expression
        assert "max by (run_id, hardware)" in expression
        if matcher:
            assert matcher in expression
        values = [10, 90] if "_total_tests" in expression else [10, 0]
        return [
            vector({"run_id": f"{i + 1}:1"}, value) for i, value in enumerate(values)
        ]

    assert mic_page.failure_summary(query, days=14, scope=scope) == {
        "total": 100,
        "failed": 10,
        "rate": 10.0,
    }


def test_failure_rate_with_no_recorded_checks_is_unknown():
    assert mic_page.failure_summary(lambda query: [], days=7, scope="all") == {
        "total": 0,
        "failed": 0,
        "rate": None,
    }


def test_failure_rate_is_independent_of_table_filters():
    def query(expression):
        if "pytest_run_job_" in expression:
            return [
                vector({"run_id": "123:1"}, 10 if "_total_tests" in expression else 2)
            ]
        return queries([failure()])(expression)

    result = mic_page.build_rows(
        query,
        lambda run: [detail()],
        variant="absent",
        platform="absent",
        failure_type="Runtime traceback",
    )
    assert result["rows"] == []
    assert result["summary"] == {"total": 10, "failed": 2, "rate": 20.0}


def test_http_data_is_gzipped_for_clients_that_accept_it(monkeypatch):
    monkeypatch.setattr(trace_exporter, "_prometheus_vector", lambda expression: [])
    monkeypatch.setattr(trace_exporter, "_swr_cache", OrderedDict())
    monkeypatch.setattr(trace_exporter, "_swr_inflight", {})
    server = ThreadingHTTPServer(("127.0.0.1", 0), trace_exporter.MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/mic/data?days=7"
        response = urlopen(Request(url, headers={"Accept-Encoding": "gzip, br"}))
        assert response.headers["Content-Encoding"] == "gzip"
        assert response.headers["Vary"] == "Accept-Encoding"
        assert json.loads(gzip.decompress(response.read()))["rows"] == []
        plain = urlopen(url)
        assert plain.headers["Content-Encoding"] is None
        assert plain.headers["X-Cache-State"] == "hit"
        assert json.load(plain)["days"] == 7
    finally:
        server.shutdown()
        server.server_close()


def test_start_times_are_read_only_for_failing_and_history_runs():
    expressions = []

    def query(expression):
        expressions.append(expression)
        if "max by (run_id, pr)" in expression:
            return [vector({"run_id": "124:1", "pr": "42"})]
        return queries([failure()])(expression)

    mic_page.build_rows(query, lambda run: None)
    (starts,) = [e for e in expressions if "pytest_run_start_time_seconds" in e]
    assert 'run_id=~"123:1|124:1"' in starts


def test_start_times_fall_back_to_all_runs_above_the_matcher_limit(monkeypatch):
    monkeypatch.setattr(mic_page, "MAX_RUN_MATCHER", 1)
    expressions = []
    mic_page.run_starts(lambda e: expressions.append(e) or [], {"1:1", "2:1"}, days=7)
    assert expressions == [
        "max by (run_id) (last_over_time(pytest_run_start_time_seconds[7d]))"
    ]
    assert mic_page.run_starts(lambda e: pytest.fail(e), set(), days=7) == {}


def test_renamed_pr_shows_its_newest_title_without_a_month_long_subquery():
    expressions = []

    def query(expression):
        expressions.append(expression)
        if "ci_github_run_title_info" in expression:
            return [vector({"pr": "43", "title": "From the run"})]
        if "[1h]" in expression:
            return [vector({"pr": "42", "title": "Renamed"})]
        if "[6h:]" in expression:
            return [vector({"pr": "44", "title": "Second rename"})]
        if "[6h]" in expression:
            return [
                vector({"pr": "44", "title": "First rename"}),
                vector({"pr": "44", "title": "Second rename"}),
            ]
        return [
            vector({"pr": "42", "title": "Original"}),
            vector({"pr": "42", "title": "Renamed"}),
            vector({"pr": "44", "title": "Original"}),
        ]

    failures = [
        ({"pr": pr}, 1) for pr in ("42", "gh-readonly-queue/main/pr-43-ab", "44")
    ]
    assert mic_page.pr_titles(query, failures, days=30) == {
        "42": "Renamed",
        "43": "From the run",
        "44": "Second rename",
    }
    # Only the tie inside the 6h window needs per-sample timestamps, over 6h.
    subqueries = [e for e in expressions if "timestamp(" in e]
    assert len(subqueries) == 1
    assert subqueries[0].endswith('pr=~"44",title!=""}))[6h:])))')
    assert not any("[30d:" in e for e in expressions)


def test_title_lookup_is_skipped_without_prs_and_never_fails_the_page():
    assert (
        mic_page.pr_titles(lambda e: pytest.fail(e), [({"pr": "main"}, 1)], days=7)
        == {}
    )

    def broken(expression):
        raise RuntimeError("down")

    assert mic_page.pr_titles(broken, [({"pr": "42"}, 1)], days=7) == {}


def test_each_run_is_loaded_once_for_rows_and_history():
    calls = []

    def query(expression):
        if "max by (run_id, pr)" in expression:
            return [
                vector({"run_id": "123:1", "pr": "42"}),
                vector({"run_id": "124:1", "pr": "42"}),
            ]
        return queries([failure()])(expression)

    def load(run):
        calls.append(run)
        return [detail(**{"mic.status": "fail"})] if run == "123:1" else None

    mic_page.build_rows(query, load)
    assert sorted(calls) == ["123:1", "124:1"]


def test_traces_without_mic_details_are_not_fetched_again(monkeypatch, tmp_path):
    fetched = []
    traces = {"a" * 32: {"spans": []}, "b" * 32: None}

    def fetch(trace_id):
        fetched.append(trace_id)
        return traces[trace_id]

    monkeypatch.setattr(trace_exporter, "fetch_failure_trace", fetch)
    monkeypatch.setattr(trace_exporter, "extract_trace_rows", lambda trace: ({}, []))
    monkeypatch.setattr(trace_exporter, "_run_store_dir", lambda: "")
    monkeypatch.setattr(trace_exporter, "_mic_trace_misses", {})
    for _ in range(2):
        assert trace_exporter._mic_trace_rows("a" * 32) == []
        assert trace_exporter._mic_trace_rows("b" * 32) == []
    # The empty trace is remembered; the failed fetch is retried.
    assert fetched == ["a" * 32, "b" * 32, "b" * 32]
