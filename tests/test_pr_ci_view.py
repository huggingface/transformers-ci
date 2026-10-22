from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import urlopen

import pytest

from transformersci.otel import pr_ci_view, trace_exporter


def test_examples_job_uses_github_status() -> None:
    raw_jobs = [
        {
            "name": "pr-ci / examples_torch / examples_torch [shard 0/1]",
            "status": "completed",
            "conclusion": "success",
            "html_url": "https://github.com/huggingface/transformers/actions/runs/1/job/2",
        }
    ]
    rows = [
        {
            "test_job": "examples_torch",
            "hardware": "cpu",
            "status_code": "OK",
        }
    ]
    assert pr_ci_view.summarize_jobs(raw_jobs, rows, "cpu") == [
        {
            "name": "examples_torch",
            "hardware": "cpu",
            "status": "success",
            "pass": 1,
            "fail": None,
            "started_at": "",
            "url": "https://github.com/huggingface/transformers/actions/runs/1/job/2",
        }
    ]


def test_running_shard_keeps_job_active_after_another_shard_fails() -> None:
    jobs = pr_ci_view.summarize_jobs(
        [
            {
                "name": "pr-ci / tests_torch / tests_torch [shard 0/2]",
                "conclusion": "failure",
            },
            {
                "name": "pr-ci / tests_torch / tests_torch [shard 1/2]",
                "status": "in_progress",
            },
        ],
        [],
        "cpu",
    )
    assert jobs[0]["status"] == "in_progress"


def test_telemetry_jobs_use_completed_run_result_when_github_jobs_are_missing() -> None:
    rows = [
        {"test_job": "tests_torch", "hardware": "cpu", "status_code": "OK"},
        {"test_job": "tests_generate", "hardware": "cpu", "status_code": "ERROR"},
    ]
    jobs = pr_ci_view.summarize_jobs([], rows, "cpu", "success")
    assert {job["name"]: job["status"] for job in jobs} == {
        "tests_generate": "failure",
        "tests_torch": "success",
    }
    assert (
        pr_ci_view.summarize_jobs([], rows[:1], "cpu", "failure")[0]["status"]
        == "recorded"
    )


def test_source_lookup_includes_active_run_without_test_telemetry(monkeypatch) -> None:
    monkeypatch.setattr(
        trace_exporter, "prometheus_base_url", lambda: "http://prometheus"
    )

    def query(_url: str, **_kwargs) -> dict:
        return {
            "status": "success",
            "data": {
                "result": [
                    {
                        "metric": {"run_id": "100:1", "ci_event": "none"},
                        "value": [0, "100"],
                    }
                ]
            },
        }

    monkeypatch.setattr(trace_exporter, "_http_get_json", query)

    def api(path: str) -> dict:
        if path == "pulls/49607":
            return {"head": {"sha": "a" * 40}}
        if path.startswith("actions/workflows/pr-ci-caller.yml/runs?"):
            return {
                "workflow_runs": [
                    {
                        "id": 200,
                        "run_attempt": 1,
                        "path": ".github/workflows/pr-ci-caller.yml",
                        "event": "pull_request",
                        "head_sha": "a" * 40,
                        "status": "in_progress",
                        "created_at": "2026-10-14T12:00:00Z",
                    }
                ]
            }
        return {
            "path": ".github/workflows/pr-ci-caller.yml@main",
            "event": "pull_request",
            "status": "completed",
            "created_at": "2026-10-14T11:00:00Z",
        }

    monkeypatch.setattr(
        trace_exporter,
        "_rerun_github_api",
        api,
    )
    sources = object.__new__(trace_exporter.MetricsHandler)._pr_ci_sources("49607")
    assert sources["cpu"]["source_run"]["run_id"] == "100:1"
    assert pr_ci_view.newest_run(sources["cpu"])["run_id"] == "200:1"
    assert pr_ci_view.newest_run(sources["cpu"])["status"] == "in_progress"


def test_source_lookup_selects_latest_valid_run_per_lane(monkeypatch) -> None:
    monkeypatch.setattr(
        trace_exporter, "prometheus_base_url", lambda: "http://prometheus"
    )
    monkeypatch.setattr(
        trace_exporter,
        "_http_get_json",
        lambda *args, **kwargs: {
            "status": "success",
            "data": {
                "result": [
                    {
                        "metric": {"run_id": "101:1", "ci_event": "none"},
                        "value": [0, "100"],
                    },
                    {
                        "metric": {"run_id": "102:1", "ci_event": "pr-comment"},
                        "value": [0, "110"],
                    },
                    {
                        "metric": {"run_id": "103:1", "ci_event": "none"},
                        "value": [0, "120"],
                    },
                ]
            },
        },
    )
    runs = {
        "101": {
            "path": ".github/workflows/pr-ci-caller.yml@main",
            "event": "pull_request",
        },
        "102": {
            "path": ".github/workflows/self-comment-ci.yml@main",
            "event": "issue_comment",
        },
        "103": {
            "path": ".github/workflows/rerun-failed-tests.yml@main",
            "event": "workflow_dispatch",
        },
    }
    monkeypatch.setattr(
        trace_exporter, "_rerun_github_api", lambda path: runs[path.rsplit("/", 1)[1]]
    )
    sources = object.__new__(trace_exporter.MetricsHandler)._pr_ci_sources("49599")
    assert sources["cpu"]["source_run"]["run_id"] == "101:1"
    assert sources["gpu"]["source_run"]["run_id"] == "102:1"


def test_run_slow_comparison_uses_the_comment_report() -> None:
    body = """## CI (Nvidia + AMD)
<!-- nvidia-section-start -->
### Nvidia CI
[Workflow Run](https://github.com/huggingface/transformers/actions/runs/123)
| RUN | [a](https://github.com/huggingface/transformers/commit/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) |
| PR | [b](https://github.com/huggingface/transformers/commit/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb) |
| main | [c](https://github.com/huggingface/transformers/commit/cccccccccccccccccccccccccccccccccccccccc) |
❌ **[1 new failed tests from this PR](https://huggingface.co/datasets/example/report/new_failures_with_bad_commit.json)**
- [vit](https://github.com/huggingface/transformers/actions/runs/123/job/1):
    tests/models/vit/test_vit.py::test_regression (✅ ⟹ ❌)
<!-- nvidia-section-end -->
<!-- amd-section-start -->
### AMD CI
[Workflow Run](https://github.com/huggingface/transformers/actions/runs/456)
✅ No failing test specific to this PR
<!-- amd-section-end -->"""
    comment = {"body": body, "updated_at": "2026-10-14T12:00:00Z"}
    assert pr_ci_view.matching_comment([comment], "123:1") == comment
    assert pr_ci_view.matching_comment([comment], "999:1") is None
    assert pr_ci_view.latest_platform_comment([comment], "amd") == comment

    nvidia = pr_ci_view.comparison(body, "nvidia")
    assert nvidia["state"] == "complete"
    assert nvidia["run_id"] == "123"
    assert nvidia["commits"]["main"] == "c" * 40
    assert nvidia["report_url"].endswith("/new_failures_with_bad_commit.json")
    assert [item["nodeid"] for item in nvidia["specific"]] == [
        "tests/models/vit/test_vit.py::test_regression"
    ]
    assert pr_ci_view.comparison(body, "amd")["specific"] == []

    rows = [
        {
            "test_nodeid": "tests/models/vit/test_vit.py::test_regression",
            "hardware": "single-gpu",
            "test_job": "run_models_gpu",
            "status_code": "ERROR",
        },
        {
            "test_nodeid": "tests/models/vit/test_vit.py::test_regression",
            "hardware": "multi-gpu",
            "test_job": "run_models_gpu",
            "status_code": "ERROR",
        },
        {
            "test_nodeid": "tests/models/vit/test_vit.py::test_existing",
            "hardware": "single-gpu",
            "test_job": "run_models_gpu",
            "status_code": "ERROR",
        },
    ]
    assert [item["specific"] for item in pr_ci_view.failures(rows, nvidia)] == [
        True,
        True,
        False,
    ]


def test_combined_view_endpoint_serves_both_lanes(monkeypatch) -> None:
    cpu = {
        "run_id": "100:1",
        "url": "https://github.com/huggingface/transformers/actions/runs/100",
        "status": "completed",
        "conclusion": "success",
        "created_at": "2026-10-14T10:00:00Z",
    }
    gpu = {
        "run_id": "123:1",
        "url": "https://github.com/huggingface/transformers/actions/runs/123",
        "status": "completed",
        "conclusion": "failure",
        "created_at": "2026-10-14T11:00:00Z",
    }
    sources = {
        "cpu": {"source_run": cpu, "active_runs": [], "notes": []},
        "gpu": {"source_run": gpu, "active_runs": [], "notes": []},
    }
    monkeypatch.setattr(
        trace_exporter.MetricsHandler, "_pr_ci_sources", lambda self, pr: sources
    )
    monkeypatch.setattr(
        trace_exporter,
        "gather_run_test_rows",
        lambda run_id: (
            [
                {
                    "test_nodeid": "tests/models/vit/test_vit.py::test_regression",
                    "test_job": "run_models_gpu",
                    "hardware": "single-gpu",
                    "status_code": "ERROR",
                    "pr": "49599",
                }
            ]
            if run_id == "123:1"
            else []
        ),
    )
    comment_body = """<!-- nvidia-section-start -->
[Workflow Run](https://github.com/huggingface/transformers/actions/runs/123)
✅ No failing test specific to this PR
<!-- nvidia-section-end -->
<!-- amd-section-start -->
[Workflow Run](https://github.com/huggingface/transformers/actions/runs/456)
✅ No failing test specific to this PR
<!-- amd-section-end -->"""

    def api(path: str, **_kwargs) -> object:
        if path.startswith("issues/"):
            return [{"body": comment_body, "updated_at": "2026-10-14T12:00:00Z"}]
        if path.startswith("actions/runs/100/jobs"):
            return {
                "jobs": [
                    {
                        "name": "pr-ci / tests_torch / tests_torch [shard 0/1]",
                        "status": "completed",
                        "conclusion": "success",
                    }
                ]
            }
        if path.startswith("actions/runs/123/jobs"):
            return {
                "jobs": [
                    {
                        "name": "Model CI / run_models_gpu (aws-g5-4xlarge-cache, 0) / run_models_gpu (models/vit)",
                        "status": "completed",
                        "conclusion": "failure",
                    }
                ]
            }
        if path == "actions/runs/456":
            return {
                "id": 456,
                "path": ".github/workflows/self-comment-ci-amd.yml@main",
                "event": "issue_comment",
                "status": "completed",
                "conclusion": "success",
            }
        if path.startswith("actions/runs/456/jobs"):
            return {
                "jobs": [
                    {
                        "name": "Model CI / run_models_gpu (2gpu, 0) / run_models_gpu (models/vit)",
                        "status": "completed",
                        "conclusion": "success",
                    }
                ]
            }
        raise AssertionError(path)

    monkeypatch.setattr(trace_exporter, "_rerun_github_api", api)
    server = ThreadingHTTPServer(("127.0.0.1", 0), trace_exporter.MetricsHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(f"{base}/pr-ci-view?pr=49599", timeout=5) as response:
            page = response.read()
            assert b"Latest Run Jobs" in page
            assert b'data-group="test_model"' in page
            assert b"Hide non-PR-specific" in page
        with urlopen(f"{base}/pr-ci-view/app.js", timeout=5) as response:
            assert b"Only PR-specific" not in response.read()
        with urlopen(f"{base}/pr-ci-view/data?pr=49599", timeout=5) as response:
            payload = json.load(response)
    finally:
        server.shutdown()
        server.server_close()
    assert [lane["id"] for lane in payload["lanes"]] == ["cpu", "gpu", "amd"]
    assert payload["lanes"][0]["jobs"][0]["name"] == "tests_torch"
    assert payload["lanes"][1]["comparison"]["state"] == "complete"
    assert payload["lanes"][1]["failures"][0]["specific"] is False
    assert payload["lanes"][1]["failures"][0]["model"] == "vit"
    assert payload["lanes"][2]["jobs"][0]["hardware"] == "amd-2gpu"


def test_stale_values_are_served_at_once_and_rebuilt_in_the_background(
    monkeypatch,
) -> None:
    trace_exporter._swr_cache.clear()
    clock = [1000.0]
    monkeypatch.setattr(trace_exporter.time, "monotonic", lambda: clock[0])
    builds: list[int] = []
    rebuilt = threading.Event()

    def build() -> bytes:
        builds.append(1)
        if len(builds) > 1:
            rebuilt.set()
        return f"v{len(builds)}".encode()

    fetch = lambda: trace_exporter._swr_fetch(  # noqa: E731
        "k", build, fresh_seconds=30, stale_seconds=900
    )
    assert fetch() == (b"v1", "miss", 0.0)
    clock[0] += 10
    assert fetch()[:2] == (b"v1", "hit")
    clock[0] += 60  # past fresh, within stale: old value now, rebuild behind it
    assert fetch()[:2] == (b"v1", "stale")
    assert rebuilt.wait(5)
    for _ in range(100):  # the rebuild stores just after building
        if trace_exporter._swr_cache["k"][1] == b"v2":
            break
        threading.Event().wait(0.01)
    assert fetch()[:2] == (b"v2", "hit")
    clock[0] += 1000  # past stale: built in the request again
    assert fetch() == (b"v3", "miss", 0.0)
    trace_exporter._swr_cache.clear()


def test_concurrent_misses_share_one_build() -> None:
    trace_exporter._swr_cache.clear()
    gate, builds = threading.Event(), []

    def build() -> bytes:
        builds.append(1)
        gate.wait(5)
        return b"shared"

    results: list = []
    threads = [
        threading.Thread(
            target=lambda: results.append(
                trace_exporter._swr_fetch(
                    "shared", build, fresh_seconds=30, stale_seconds=900
                )[0]
            )
        )
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    threading.Event().wait(0.2)
    gate.set()
    for thread in threads:
        thread.join(5)
    assert results == [b"shared"] * 4 and len(builds) == 1
    trace_exporter._swr_cache.clear()


def test_job_pages_after_the_first_are_fetched_together_in_order(
    monkeypatch,
) -> None:
    trace_exporter._completed_jobs_cache.clear()
    calls: list[str] = []
    together = threading.Barrier(3, timeout=5)

    def api(path: str, **_kwargs) -> object:
        calls.append(path)
        assert "per_page=30&" in path
        page = int(path.rsplit("page=", 1)[1])
        if page > 1:
            together.wait()  # pages 2-4 must be in flight at the same time
        size = 30 if page < 4 else 10
        return {
            "total_count": 100,
            "jobs": [{"name": f"p{page}-{n}"} for n in range(size)],
        }

    monkeypatch.setattr(trace_exporter, "_rerun_github_api", api)
    jobs, notes = trace_exporter._pr_ci_run_jobs("77:1", completed=True)
    assert len(jobs) == 100 and notes == []
    assert [job["name"] for job in jobs[::30]] == ["p1-0", "p2-0", "p3-0", "p4-0"]
    assert len(calls) == 4
    # A finished attempt's jobs never change: the second read costs no GitHub call.
    assert trace_exporter._pr_ci_run_jobs("77:1", completed=True)[0] == jobs
    assert len(calls) == 4
    # A rerun is a new job list under the same run id: read again.
    trace_exporter._pr_ci_run_jobs("77:2", completed=True)
    assert len(calls) == 8
    trace_exporter._pr_ci_run_jobs("88", completed=False)
    trace_exporter._pr_ci_run_jobs("88", completed=False)
    assert len(calls) == 16  # a running run is read again
    trace_exporter._completed_jobs_cache.clear()


def test_job_reads_stop_at_the_cap(monkeypatch) -> None:
    trace_exporter._completed_jobs_cache.clear()
    pages: list[int] = []

    def api(path: str, **_kwargs) -> object:
        pages.append(int(path.rsplit("page=", 1)[1]))
        return {"total_count": 1200, "jobs": [{"name": "j"}] * 30}

    monkeypatch.setattr(trace_exporter, "_rerun_github_api", api)
    jobs, notes = trace_exporter._pr_ci_run_jobs("55", completed=True)
    assert len(jobs) == 500 and max(pages) == 17
    assert notes == ["Showing the first 500 GitHub jobs."]
    assert "55" in trace_exporter._completed_jobs_cache
    trace_exporter._completed_jobs_cache.clear()


def test_a_failed_job_page_keeps_the_pages_before_it(monkeypatch) -> None:
    trace_exporter._completed_jobs_cache.clear()

    def api(path: str, **_kwargs) -> object:
        page = int(path.rsplit("page=", 1)[1])
        if page == 3:
            raise OSError("timeout")
        return {"total_count": 120, "jobs": [{"name": f"p{page}"}] * 30}

    monkeypatch.setattr(trace_exporter, "_rerun_github_api", api)
    jobs, notes = trace_exporter._pr_ci_run_jobs("99", completed=True)
    assert len(jobs) == 60
    assert notes == ["GitHub job details are temporarily unavailable."]
    assert "99" not in trace_exporter._completed_jobs_cache  # retried next time


def test_the_runs_stage_answers_before_jobs_and_failures(monkeypatch) -> None:
    trace_exporter._swr_cache.clear()
    run = {
        "run_id": "100:1",
        "url": "https://github.com/huggingface/transformers/actions/runs/100",
        "status": "in_progress",
        "conclusion": "",
        "created_at": "2026-10-22T10:00:00Z",
    }
    monkeypatch.setattr(
        trace_exporter.MetricsHandler,
        "_pr_ci_sources",
        lambda self, pr: {
            "cpu": {"source_run": run, "active_runs": [], "notes": []},
            "gpu": {"source_run": None, "active_runs": [], "notes": []},
        },
    )

    def no_api(path: str, **_kwargs) -> object:
        raise AssertionError(f"the runs stage read {path}")

    monkeypatch.setattr(trace_exporter, "_rerun_github_api", no_api)
    server = ThreadingHTTPServer(("127.0.0.1", 0), trace_exporter.MetricsHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = (
            f"http://127.0.0.1:{server.server_port}/pr-ci-view/data?pr=49700&stage=runs"
        )
        with urlopen(url, timeout=5) as response:
            assert response.headers["X-Cache-State"] == "miss"
            payload = json.load(response)
        with urlopen(url, timeout=5) as response:
            assert response.headers["X-Cache-State"] == "hit"
    finally:
        server.shutdown()
        server.server_close()
        trace_exporter._swr_cache.clear()
    assert payload["pending"] is True
    assert [lane["id"] for lane in payload["lanes"]] == ["cpu", "gpu"]
    assert payload["lanes"][0]["run"]["run_id"] == "100:1"
    assert payload["lanes"][0]["jobs"] == [] and payload["lanes"][1]["run"] is None


def test_a_value_built_during_a_failure_is_rebuilt_on_the_next_request() -> None:
    trace_exporter._swr_cache.clear()
    built = iter([b"unavailable", b"good"])
    done = threading.Event()

    def build() -> bytes:
        value = next(built)
        if value == b"good":
            done.set()
        return value

    fetch = lambda: trace_exporter._swr_fetch(  # noqa: E731
        "k", build, fresh_seconds=30, stale_seconds=900, keep=lambda v: v == b"good"
    )
    assert fetch()[:2] == (b"unavailable", "miss")
    assert fetch()[:2] == (b"unavailable", "stale")  # served, rebuilt behind it
    assert done.wait(5)
    for _ in range(100):
        if trace_exporter._swr_cache["k"][1] == b"good":
            break
        threading.Event().wait(0.01)
    assert fetch()[:2] == (b"good", "hit")
    trace_exporter._swr_cache.clear()


def _view_sources(gpu_run_id: str = "123:1") -> dict:
    def run(run_id: str, conclusion: str) -> dict:
        return {
            "run_id": run_id,
            "url": f"https://github.com/huggingface/transformers/actions/runs/{run_id.split(':')[0]}",
            "status": "completed",
            "conclusion": conclusion,
            "created_at": "2026-10-14T10:00:00Z",
        }

    return {
        "cpu": {"source_run": run("100:1", "failure"), "active_runs": [], "notes": []},
        "gpu": {
            "source_run": run(gpu_run_id, "success"),
            "active_runs": [],
            "notes": [],
        },
    }


def _nvidia_comment(run: str) -> dict:
    body = (
        "<!-- nvidia-section-start -->\n"
        f"[Workflow Run](https://github.com/huggingface/transformers/actions/runs/{run})\n"
        "✅ No failing test specific to this PR\n<!-- nvidia-section-end -->"
    )
    return {"body": body, "updated_at": "2026-10-14T12:00:00Z"}


def _payload(monkeypatch, api, rows=lambda run_id: []) -> dict:
    trace_exporter._swr_cache.clear()
    trace_exporter._completed_jobs_cache.clear()
    monkeypatch.setattr(trace_exporter, "_rerun_github_api", api)
    monkeypatch.setattr(trace_exporter, "gather_run_test_rows", rows)
    handler = object.__new__(trace_exporter.MetricsHandler)
    try:
        return json.loads(handler._pr_ci_view_payload("49599"))
    finally:
        trace_exporter._swr_cache.clear()
        trace_exporter._completed_jobs_cache.clear()


def test_nvidia_jobs_are_read_while_the_comments_load(monkeypatch) -> None:
    monkeypatch.setattr(
        trace_exporter.MetricsHandler,
        "_pr_ci_sources",
        lambda self, pr: _view_sources(),
    )
    gpu_jobs_requested = threading.Event()

    def api(path: str, **_kwargs) -> object:
        if path.startswith("issues/"):
            # Before: the Nvidia jobs were only requested after this returned.
            assert gpu_jobs_requested.wait(5)
            return [_nvidia_comment("123")]
        if path.startswith("actions/runs/123/jobs"):
            gpu_jobs_requested.set()
        return {"jobs": []}

    payload = _payload(monkeypatch, api)
    assert [lane["run"]["run_id"] for lane in payload["lanes"]] == ["100:1", "123:1"]


def test_a_newer_nvidia_run_in_the_comments_replaces_the_early_reads(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        trace_exporter.MetricsHandler,
        "_pr_ci_sources",
        lambda self, pr: _view_sources(),
    )
    paths: list[str] = []

    def api(path: str, **_kwargs) -> object:
        paths.append(path)
        if path.startswith("issues/"):
            return [_nvidia_comment("124")]
        if path == "actions/runs/124":
            return {
                "id": 124,
                "run_attempt": 1,
                "path": ".github/workflows/self-comment-ci.yml@main",
                "event": "issue_comment",
                "status": "completed",
                "conclusion": "failure",
                "html_url": "https://github.com/huggingface/transformers/actions/runs/124",
                "created_at": "2026-10-14T13:00:00Z",
            }
        if path.startswith("actions/runs/124/jobs"):
            return {
                "jobs": [
                    {
                        "name": "Model CI / run_models_gpu (aws-g5-4xlarge-cache, 0) / run_models_gpu (models/vit)",
                        "status": "completed",
                        "conclusion": "failure",
                    }
                ]
            }
        return {"jobs": []}

    payload = _payload(monkeypatch, api)
    gpu = payload["lanes"][1]
    assert gpu["run"]["run_id"] == "124:1"
    assert [job["name"] for job in gpu["jobs"]] == ["run_models_gpu"]  # run 124's
    assert any(path.startswith("actions/runs/124/jobs") for path in paths)


def test_full_view_starts_the_queue_history_the_page_asks_for_next(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        trace_exporter.MetricsHandler,
        "_pr_ci_sources",
        lambda self, pr: _view_sources(),
    )
    requested: list[tuple[str, list[str]]] = []
    started = threading.Event()

    def history(pr: str, nodeids: list[str]) -> tuple[bytes, str]:
        requested.append((pr, nodeids))
        started.set()
        return b"{}", "miss"

    monkeypatch.setattr(trace_exporter, "_queue_history", history)

    def rows(run_id: str) -> list[dict]:
        if run_id != "100:1":
            return []
        return [
            {
                "test_nodeid": nodeid,
                "test_job": "tests_torch",
                "hardware": "cpu",
                "status_code": "ERROR",
                "pr": "49599",
            }
            for nodeid in (
                "tests/b.py::test_b",
                "tests/a.py::test_a",
                "tests/b.py::test_b",
            )
        ]

    _payload(
        monkeypatch,
        lambda path, **_kwargs: [] if path.startswith("issues/") else {"jobs": []},
        rows,
    )
    assert started.wait(5)
    # Exactly the request pr_ci_view.js makes: unique CPU failures, sorted.
    assert requested == [("49599", ["tests/a.py::test_a", "tests/b.py::test_b"])]


def test_no_queue_history_is_started_without_cpu_failures(monkeypatch) -> None:
    monkeypatch.setattr(
        trace_exporter,
        "_queue_history",
        lambda pr, nodeids: pytest.fail("no failures, no Tempo search"),
    )
    trace_exporter._prefetch_queue_history("49599", [{"id": "cpu", "failures": []}])
    trace_exporter._prefetch_queue_history(
        "49599", [{"id": "gpu", "failures": [{"nodeid": "tests/a.py::test_a"}]}]
    )


def test_a_lane_reads_the_pages_its_last_run_needed_together_with_page_one(
    monkeypatch,
) -> None:
    trace_exporter._completed_jobs_cache.clear()
    monkeypatch.setattr(trace_exporter, "_job_pages_seen", {})
    sizes = {"7": 100, "8": 100, "9": 10}
    calls: list[tuple[str, int]] = []
    barrier: list[threading.Barrier] = []

    def api(path: str, **_kwargs) -> object:
        run = path.split("/")[2]
        number = int(path.rsplit("page=", 1)[1])
        calls.append((run, number))
        if barrier:
            barrier[0].wait()
        start = (number - 1) * 30
        count = max(0, min(30, sizes[run] - start))
        return {
            "total_count": sizes[run],
            "jobs": [{"name": f"{run}-{start + n}"} for n in range(count)],
        }

    monkeypatch.setattr(trace_exporter, "_rerun_github_api", api)
    jobs, _ = trace_exporter._pr_ci_run_jobs("7:1", completed=False, lane="cpu")
    assert len(jobs) == 100 and calls[0] == ("7", 1)
    assert trace_exporter._job_pages_seen == {"cpu": 4}
    # Next cpu run: pages 1-4 must all be in flight at once.
    calls.clear()
    barrier.append(threading.Barrier(4, timeout=5))
    jobs, notes = trace_exporter._pr_ci_run_jobs("8:1", completed=False, lane="cpu")
    barrier.clear()
    assert sorted(calls) == [("8", n) for n in range(1, 5)]
    assert [job["name"] for job in jobs] == [
        f"8-{n}" for n in range(100)
    ] and notes == []
    # A smaller run wastes the guessed pages but returns only its own jobs.
    jobs, _ = trace_exporter._pr_ci_run_jobs("9:1", completed=False, lane="cpu")
    assert [job["name"] for job in jobs] == [f"9-{n}" for n in range(10)]
    assert trace_exporter._job_pages_seen == {"cpu": 1}
    # Other lanes keep their own hint.
    calls.clear()
    trace_exporter._pr_ci_run_jobs("8:1", completed=False, lane="gpu")
    assert calls[0] == ("8", 1) and len(calls) == 4


def test_guessed_pages_without_a_count_stop_at_the_first_short_page(
    monkeypatch,
) -> None:
    monkeypatch.setattr(trace_exporter, "_job_pages_seen", {"cpu": 4})
    calls: list[int] = []

    def api(path: str, **_kwargs) -> object:
        number = int(path.rsplit("page=", 1)[1])
        calls.append(number)
        return {"jobs": [{"name": f"p{number}"}] * (30 if number < 4 else 5)}

    monkeypatch.setattr(trace_exporter, "_rerun_github_api", api)
    jobs, notes = trace_exporter._pr_ci_run_jobs("5:1", completed=False, lane="cpu")
    assert len(jobs) == 95 and notes == [] and sorted(calls) == [1, 2, 3, 4]
