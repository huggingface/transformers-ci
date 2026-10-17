from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import urlopen

from transformersci.otel import project_health, trace_exporter

DAY = 86400
NOW = 100 * DAY + 12 * 3600  # noon UTC on day 100
GPU_TEST = "tests/models/bert/test_modeling_bert.py::BertModelTest::test_fa2"
CPU_TEST = "tests/utils/test_hub_utils.py::HubTests::test_cached"


def vector(*samples: tuple[dict[str, str], float]) -> list[dict]:
    return [{"metric": labels, "value": [NOW, str(value)]} for labels, value in samples]


def failure(
    nodeid: str, job: str, run_id: str, trace_id: str = "t", exc: str = "AssertionError"
):
    return (
        {
            "test_nodeid": nodeid,
            "test_job": job,
            "run_id": run_id,
            "trace_id": trace_id,
            "exception_type": exc,
        },
        1,
    )


def run(run_id: str, start: float):
    return ({"run_id": run_id}, start)


def job_run(run_id: str, job: str, total: float):
    return ({"run_id": run_id, "test_job": job}, total)


def rows_for(failures, starts, job_runs=(), job_max=(), days=7):
    return project_health.failure_rows(
        vector(*failures),
        vector(*starts),
        vector(*job_runs),
        vector(*job_max),
        now=NOW,
        days=days,
    )


def test_one_row_per_test_and_job_counting_failing_runs_and_days() -> None:
    rows = rows_for(
        [
            failure(GPU_TEST, "run_models_gpu", "r1", "a" * 32),
            failure(GPU_TEST, "run_models_gpu", "r2", "b" * 32, "RuntimeError"),
            failure(GPU_TEST, "run_models_gpu", "r3", "c" * 32),
        ],
        [run("r1", 98 * DAY + 10), run("r2", 99 * DAY + 10), run("r3", 99 * DAY + 20)],
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["fails"] == 3
    assert row["bad_days"] == 2
    assert row["days"] == [0, 0, 0, 0, 1, 2, 0]
    # The latest failing run supplies the signal, trace and run.
    assert (row["trace_id"], row["run_id"], row["signal"]) == (
        "c" * 32,
        "r3",
        "AssertionError",
    )
    assert (row["runner"], row["model"], row["test"]) == (
        "GPU",
        "bert",
        "BertModelTest::test_fa2",
    )
    assert row["sticky"] is True


def test_runner_comes_from_the_job_name() -> None:
    rows = rows_for(
        [
            failure(CPU_TEST, "tests_torch", "r1"),
            failure(CPU_TEST, "tests_fsdp_ci", "r1"),
            failure(CPU_TEST, "tests_tensor_parallel_ci", "r1"),
        ],
        [run("r1", 99 * DAY)],
    )
    assert {row["job"]: row["runner"] for row in rows} == {
        "tests_torch": "CPU",
        "tests_fsdp_ci": "GPU",
        "tests_tensor_parallel_ci": "GPU",
    }
    assert rows[0]["model"] == "utils"


def test_failures_without_a_run_start_are_dropped_like_main_health() -> None:
    assert (
        rows_for([failure(GPU_TEST, "run_models_gpu", "orphan")], [run("r1", 99 * DAY)])
        == []
    )


def test_one_day_of_failures_is_not_sticky() -> None:
    rows = rows_for(
        [
            failure(GPU_TEST, "run_models_gpu", "r1"),
            failure(GPU_TEST, "run_models_gpu", "r2"),
        ],
        [run("r1", 99 * DAY + 1), run("r2", 99 * DAY + 2)],
    )
    assert rows[0]["sticky"] is False


def test_four_full_runs_after_the_last_failure_clear_it() -> None:
    failures = [
        failure(GPU_TEST, "run_models_gpu", "f1"),
        failure(GPU_TEST, "run_models_gpu", "f2"),
    ]
    starts = [run("f1", 95 * DAY), run("f2", 96 * DAY)]
    later = [run(f"p{i}", 97 * DAY + i) for i in range(4)]
    job_runs = [
        job_run("f1", "run_models_gpu", 100),
        job_run("f2", "run_models_gpu", 100),
    ]
    job_runs += [job_run(f"p{i}", "run_models_gpu", 100) for i in range(4)]
    job_max = [({"test_job": "run_models_gpu"}, 100)]

    cleared = rows_for(failures, starts + later, job_runs, job_max)[0]
    assert (cleared["sticky"], cleared["cleared"]) == (False, True)

    # Three later runs are not enough.
    three = rows_for(failures, starts + later[:3], job_runs[:5], job_max)[0]
    assert (three["sticky"], three["cleared"]) == (True, False)


def test_partial_runs_do_not_clear_a_failure() -> None:
    failures = [
        failure(GPU_TEST, "run_models_gpu", "f1"),
        failure(GPU_TEST, "run_models_gpu", "f2"),
    ]
    starts = [run("f1", 95 * DAY), run("f2", 96 * DAY)] + [
        run(f"p{i}", 97 * DAY + i) for i in range(4)
    ]
    # Under half the job's largest run: targeted reruns, not full runs.
    job_runs = [
        job_run("f1", "run_models_gpu", 100),
        job_run("f2", "run_models_gpu", 100),
    ]
    job_runs += [job_run(f"p{i}", "run_models_gpu", 49) for i in range(4)]
    job_max = [({"test_job": "run_models_gpu"}, 100)]
    assert rows_for(failures, starts, job_runs, job_max)[0]["sticky"] is True


def test_a_failure_in_one_of_the_last_four_full_runs_stays_sticky() -> None:
    failures = [
        failure(GPU_TEST, "run_models_gpu", "f1"),
        failure(GPU_TEST, "run_models_gpu", "p2"),
    ]
    starts = [run("f1", 95 * DAY)] + [run(f"p{i}", 97 * DAY + i) for i in range(4)]
    job_runs = [
        job_run(run_id, "run_models_gpu", 100)
        for run_id in ("f1", "p0", "p1", "p2", "p3")
    ]
    job_max = [({"test_job": "run_models_gpu"}, 100)]
    assert rows_for(failures, starts, job_runs, job_max)[0]["sticky"] is True


def test_rows_sort_by_bad_days_then_recency() -> None:
    other = "tests/models/gpt2/test_modeling_gpt2.py::GPT2Test::test_x"
    third = "tests/models/t5/test_modeling_t5.py::T5Test::test_y"
    rows = rows_for(
        [
            failure(GPU_TEST, "run_models_gpu", "old"),
            failure(other, "run_models_gpu", "a"),
            failure(other, "run_models_gpu", "b"),
            failure(third, "run_models_gpu", "new"),
        ],
        [
            run("old", 95 * DAY),
            run("a", 97 * DAY),
            run("b", 98 * DAY),
            run("new", 99 * DAY),
        ],
    )
    assert [row["model"] for row in rows] == ["gpt2", "t5", "bert"]


def test_model_name() -> None:
    assert (
        project_health.model_name("tests/models/qwen2_moe/test_x.py::A::b")
        == "qwen2_moe"
    )
    assert (
        project_health.model_name("tests/generation/test_utils.py::A::b")
        == "generation"
    )
    assert (
        project_health.model_name("tests/test_tokenization_common.py::A::b")
        == "test_tokenization_common"
    )


def test_parse_days_only_accepts_the_offered_windows() -> None:
    assert [project_health.parse_days(raw) for raw in ("7", "30", "9", "", "x")] == [
        7,
        30,
        14,
        14,
        14,
    ]


def test_trend_counts_a_run_once_and_buckets_it_by_start_day() -> None:
    replies = {
        "starts": vector(
            run("g1", 99 * DAY),
            run("g2", 99 * DAY + 5),
            run("c1", 100 * DAY),
            run("old", 80 * DAY),
        ),
        # sum by run_id already merged single- and multi-GPU.
        "gpu_total": vector(
            ({"run_id": "g1"}, 22000),
            ({"run_id": "g2"}, 22000),
            ({"run_id": "old"}, 22000),
        ),
        "gpu_failed": vector(
            ({"run_id": "g1"}, 3), ({"run_id": "g2"}, 0), ({"run_id": "old"}, 9)
        ),
        "cpu_total": vector(
            ({"run_id": "c1"}, 199000), ({"run_id": "nostart"}, 199000)
        ),
        "cpu_failed": vector(({"run_id": "c1"}, 0)),
    }
    trend = project_health.merge_trend(replies, now=NOW, days=7)
    gpu, cpu = trend["series"]["gpu"], trend["series"]["cpu"]
    assert trend["first_day"] == 94 * DAY
    assert (gpu["runs"], gpu["failing"], gpu["share"]) == (2, 1, 0.5)
    assert gpu["days"][5] == {"runs": 2, "failing": 1, "tests": 44000.0, "failed": 3.0}
    assert gpu["test_rate"] == 3 / 44000
    assert (cpu["runs"], cpu["failing"], cpu["share"]) == (1, 0, 0.0)
    assert cpu["days"][6]["runs"] == 1


def test_trend_with_no_runs_has_no_share() -> None:
    trend = project_health.merge_trend({}, now=NOW, days=7)
    assert trend["series"]["gpu"]["share"] is None
    assert trend["series"]["cpu"]["test_rate"] is None


def test_merge_selectors_match_how_each_runner_reports_merges() -> None:
    queries = project_health.trend_queries(7)
    assert 'ci_event="push",test_job="run_models_gpu"' in queries["gpu_total"]
    # CPU merge runs carry ci_event="none", never "push".
    assert 'hardware="cpu",ci_event="none"' in queries["cpu_failed"]
    assert all("[7d]" in query for query in queries.values())


def test_rows_endpoint_serves_and_caches_one_answer_per_window(monkeypatch) -> None:
    calls: list[str] = []

    def prometheus(query: str) -> list[dict]:
        calls.append(query)
        if "pytest_test_last_failure_info" in query:
            return vector(failure(GPU_TEST, "run_models_gpu", "r1", "a" * 32))
        if "pytest_run_start_time_seconds" in query:
            return vector(run("r1", 99 * DAY))
        return []

    monkeypatch.setattr(trace_exporter, "_prometheus_vector", prometheus)
    monkeypatch.setattr(trace_exporter, "_project_health_cache", {})
    server = ThreadingHTTPServer(("127.0.0.1", 0), trace_exporter.MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        body = json.loads(urlopen(f"{base}/project-health/rows?days=7").read())
        assert body["days"] == 7
        assert [row["nodeid"] for row in body["rows"]] == [GPU_TEST]
        first = len(calls)
        urlopen(f"{base}/project-health/rows?days=7").read()
        assert len(calls) == first
        urlopen(f"{base}/project-health/rows?days=30").read()
        assert len(calls) > first
        page = urlopen(f"{base}/project-health").read().decode()
        assert "/project-health/app.js" in page
        assert (
            urlopen(f"{base}/project-health/app.js")
            .headers["Content-Type"]
            .startswith("text/javascript")
        )
    finally:
        server.shutdown()


def test_rows_endpoint_reports_prometheus_failure(monkeypatch) -> None:
    def broken(query: str) -> list[dict]:
        raise RuntimeError("down")

    monkeypatch.setattr(trace_exporter, "_prometheus_vector", broken)
    monkeypatch.setattr(trace_exporter, "_project_health_cache", {})
    server = ThreadingHTTPServer(("127.0.0.1", 0), trace_exporter.MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        try:
            urlopen(
                f"http://127.0.0.1:{server.server_port}/project-health/trend?days=7"
            )
        except Exception as error:
            assert getattr(error, "code", None) == 502
        else:
            raise AssertionError("expected a 502")
    finally:
        server.shutdown()
