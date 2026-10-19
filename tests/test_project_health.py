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
    assert gpu["days"][5] == {
        "runs": 2,
        "failing": 1,
        "tests": 44000.0,
        "failed": 3.0,
        "rate": (3 / 22000 + 0) / 2,
    }
    assert gpu["test_rate"] == 3 / 44000
    # Day and window rates are the mean of each run's own rate.
    assert gpu["days"][5]["rate"] == (3 / 22000 + 0) / 2
    assert gpu["rate"] == (3 / 22000) / 2
    assert gpu["days"][0]["rate"] is None
    assert (cpu["runs"], cpu["failing"], cpu["share"]) == (1, 0, 0.0)
    assert cpu["days"][6]["runs"] == 1


def test_trend_with_no_runs_has_no_share() -> None:
    trend = project_health.merge_trend({}, now=NOW, days=7)
    assert trend["series"]["gpu"]["share"] is None
    assert trend["series"]["gpu"]["rate"] is None
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
        assert "<title>Fix failures</title>" in page and 'id="chart-events"' not in page
        assert (
            urlopen(f"{base}/project-health/app.js")
            .headers["Content-Type"]
            .startswith("text/javascript")
        )
        # Monitoring: the stat tiles and the two charts, on the same data.
        monitoring = urlopen(f"{base}/project-health/monitoring").read().decode()
        assert "/project-health/monitoring.js" in monitoring
        assert 'id="chart-events"' in monitoring and 'id="rows"' not in monitoring
        script = urlopen(f"{base}/project-health/monitoring.js")
        assert script.headers["Content-Type"].startswith("text/javascript")
        assert b"loadTrend();" in script.read()
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


def matrix(*series: tuple[str, str, dict[int, float]]) -> list[dict]:
    """(nodeid, job, {day: seconds}) -> a range-query result, one point per day end."""
    return [
        {
            "metric": {"test_nodeid": nodeid, "test_job": job},
            "values": [
                [(day + 1) * DAY - 1, str(value)] for day, value in points.items()
            ],
        }
        for nodeid, job, points in series
    ]


def test_slow_rows_rank_by_average_and_mark_the_days_a_test_ranked(monkeypatch) -> None:
    monkeypatch.setattr(project_health, "SLOW_TOP", 2)
    slow = (
        "tests/models/llama4/test_modeling_llama4.py::Llama4IntegrationTest::test_fp32"
    )
    steady = "tests/models/swin/test_modeling_swin.py::SwinModelTest::test_compile"
    quick = "tests/models/bert/test_modeling_bert.py::BertModelTest::test_x"
    rows = project_health.slow_rows(
        {
            "GPU": matrix(
                (slow, "run_models_gpu", {99: 4000.0, 100: 5000.0}),
                (
                    steady,
                    "run_models_gpu",
                    {96: 300.0, 97: 300.0, 98: 300.0, 100: 300.0},
                ),
                (quick, "run_models_gpu", {96: 100.0, 97: 900.0, 100: 50.0}),
            ),
            "CPU": [],
        },
        now=NOW,
        days=7,
    )
    assert [row["nodeid"] for row in rows] == [slow, quick]  # top 2 by average
    first = rows[0]
    assert (first["category"], first["runner"], first["signal"]) == (
        "slow",
        "GPU",
        "75.0 min avg",
    )
    assert first["days"] == [0, 0, 0, 0, 0, 1, 1]
    assert first["durations"][5:] == [4000.0, 5000.0]
    # quick ranked in the top 2 on days 96 and 97 only (day 100: slow and steady win).
    assert rows[1]["days"] == [0, 0, 1, 1, 0, 0, 0]
    assert rows[1]["fails"] == 2
    # Sticky: slowest on 2+ days, one of them among the last two.
    assert (first["sticky"], rows[1]["sticky"]) == (True, False)


def test_build_rows_adds_slow_rows_from_two_range_queries() -> None:
    asked = []

    def query_range(expr, start, end, step):
        asked.append((expr, start, end, step))
        return []

    project_health.build_rows(lambda expr: [], now=NOW, days=7, query_range=query_range)
    assert len(asked) == 2
    assert all(step == DAY for *_, step in asked)
    assert asked[0][1] == 95 * DAY - 1 and asked[0][2] == 101 * DAY - 1
    assert 'test_job=~"' in asked[0][0] and 'test_job!~"' in asked[1][0]


def test_events_keep_the_window_and_name_each_kind() -> None:
    now = 100 * DAY + 3600
    replies = {
        "model": vector(
            ({"model": "beam", "pr": "49664"}, (99 * DAY + 10) * 1000),
            ({"model": "old", "pr": "1"}, (80 * DAY) * 1000),
        ),
        "dependency": vector(
            (
                {"dependency": "python", "change": "→ 3.11", "pr": "none"},
                (98 * DAY) * 1000,
            ),
        ),
    }
    events = project_health.project_events(replies, now=now, days=7)
    assert events == [
        {"kind": "dependency", "at": 98 * DAY, "pr": "", "title": "python → 3.11"},
        {"kind": "model", "at": 99 * DAY + 10, "pr": "49664", "title": "beam"},
    ]


def test_activity_query_counts_every_series_per_utc_day() -> None:
    query = project_health.activity_query(first_day=20377, days=2)
    assert query.count("issueCount") == 8
    assert (
        'prs_opened_0: search(query: "repo:huggingface/transformers is:pr created:2025-10-16"'
        in query
    )
    assert (
        'prs_merged_1: search(query: "repo:huggingface/transformers is:pr merged:2025-10-17"'
        in query
    )
    data = {"prs_opened_0": {"issueCount": 25}, "issues_opened_1": {"issueCount": 7}}
    assert project_health.activity_counts(data, 2) == [
        {
            "prs_opened": 25,
            "prs_merged": None,
            "issues_opened": None,
            "issues_closed": None,
        },
        {
            "prs_opened": None,
            "prs_merged": None,
            "issues_opened": 7,
            "issues_closed": None,
        },
    ]


def test_events_survive_github_being_down() -> None:
    def graphql(query: str) -> dict:
        raise RuntimeError("rate limited")

    body = project_health.build_events(lambda expr: [], graphql, now=100 * DAY, days=7)
    assert body["counts"] is None
    assert body["counts_error"] == "rate limited"
    assert body["events"] == []
    assert body["first_day"] == 94 * DAY


def test_events_endpoint_caches_github_counts(monkeypatch) -> None:
    calls: list[str] = []

    def graphql(query: str) -> dict:
        calls.append(query)
        return {"prs_opened_0": {"issueCount": 3}}

    monkeypatch.setattr(trace_exporter, "_prometheus_vector", lambda expr: [])
    monkeypatch.setattr(trace_exporter, "_github_graphql", graphql)
    monkeypatch.setattr(trace_exporter, "_project_health_cache", {})
    server = ThreadingHTTPServer(("127.0.0.1", 0), trace_exporter.MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        body = json.loads(urlopen(f"{base}/project-health/events?days=7").read())
        assert body["days"] == 7 and len(body["counts"]) == 7
        assert body["counts"][0]["prs_opened"] == 3
        urlopen(f"{base}/project-health/events?days=7").read()
        assert len(calls) == 1
    finally:
        server.shutdown()


def test_rows_built_before_the_fix_index_are_not_cached_for_long(monkeypatch) -> None:
    """Right after a restart the fix index is still building: those rows must
    not hold an empty Probable fix column for the full cache TTL."""
    from transformersci.otel import probable_fix

    def prometheus(query: str) -> list[dict]:
        if "pytest_test_last_failure_info" in query:
            return vector(failure(GPU_TEST, "run_models_gpu", "r1", "a" * 32))
        if "pytest_run_start_time_seconds" in query:
            return vector(run("r1", 99 * DAY))
        return []

    monkeypatch.setattr(trace_exporter, "_prometheus_vector", prometheus)
    monkeypatch.setattr(trace_exporter, "_project_health_cache", {})
    monkeypatch.setattr(trace_exporter, "_PROJECT_HEALTH_UNINDEXED_TTL_SECONDS", 0.0)
    monkeypatch.setitem(probable_fix._index, "built_at", 0.0)
    monkeypatch.setitem(probable_fix._index, "by_file", {})
    server = ThreadingHTTPServer(("127.0.0.1", 0), trace_exporter.MetricsHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/project-health/rows?days=7"
        body = json.loads(urlopen(url).read())
        assert body["fix_index_at"] == 0.0 and "fix" not in body["rows"][0]
        path = GPU_TEST.split("::")[0]
        topic = probable_fix.title_topic(path)
        pr = {
            "number": 7,
            "title": f"Fix {topic}",
            "url": "https://github.com/huggingface/transformers/pull/7",
            "state": "open",
            "merged_at": None,
        }
        monkeypatch.setitem(probable_fix._index, "by_file", {path: [pr]})
        monkeypatch.setitem(probable_fix._index, "built_at", 1.0)
        body = json.loads(urlopen(url).read())
        assert body["rows"][0]["fix"]["number"] == 7
    finally:
        server.shutdown()


def test_message_key_buckets_failures_that_fail_the_same_way() -> None:
    key = project_health.message_key
    # Real messages from main (2026-10-19).
    assert (
        key(
            "CUDA error: device-side assert triggered\nSearch for `cudaErrorAssert' in https://docs.nvidia.com/cuda/"
        )
        == "CUDA error: device-side assert triggered"
    )
    lists = [
        "Lists differ: [\" and the food is delicious. I'm so glad [24 chars]re'\"] != ['and the food']",
        'Lists differ: [" and I\'m"] != [" and I\'ll"]',
    ]
    assert key(lists[0]) == key(lists[1]) == "Lists differ: ['…'] != ['…']"
    assert key("'a b' != \"a c\"") == key("'x y' != 'x z'") == "'…' != '…'"
    device_map = "The device_map provided does not give any device for the following parameters: "
    assert key(device_map + "model.a.weight, model.b.weight, " * 4) == key(
        device_map + "model.visual.patch_embed.proj.weight, " * 3
    )
    assert (
        key(
            "Expected all tensors to be on the same device, but got mat2 is on cuda:1, different from other tensors on cuda:0"
        )
        == "Expected all tensors to be on the same device, but got mat2 is on cuda:N, different from other tensors on cuda:N"
    )
    # A short quoted name tells causes apart, and colour codes go.
    assert key("No module named 'flash_attn'") != key("No module named 'kernels'")
    assert key("\x1b[96mThis is step 2/3\x1b[0m") == "This is step …/…"
    assert key("") == ""
