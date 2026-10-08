from __future__ import annotations

from transformersci.otel import rerun_failed


def _series(run_id: str, started: int, event: str) -> dict:
    return {
        "metric": {"run_id": run_id, "ci_event": event},
        "value": [0, str(started)],
    }


def test_snapshot_uses_latest_completed_per_lane_and_keeps_active_runs() -> None:
    starts = [
        _series("110:1", 110, "none"),
        _series("120:1", 120, "none"),
        _series("210:1", 210, "pr-comment"),
        _series("220:1", 220, "pr-comment"),
    ]

    def query(expr: str) -> list[dict]:
        if "pytest_run_info" in expr:
            return [
                {"metric": {"run_id": "110:1", "commit_sha": "a" * 40}},
                {"metric": {"run_id": "210:1", "commit_sha": "b" * 40}},
            ]
        return starts

    def get_run(run_id: str) -> dict:
        gpu = run_id.startswith("2")
        return {
            "path": ".github/workflows/"
            + ("self-comment-ci.yml" if gpu else "pr-ci-caller.yml")
            + "@main",
            "event": "issue_comment" if gpu else "pull_request",
            "pull_requests": [] if gpu else [{"number": 42}],
            "status": "in_progress" if run_id in {"120", "220"} else "completed",
        }

    def rows(run_id: str) -> list[dict]:
        return [
            {
                "pr": "42",
                "status_code": "ERROR",
                "test_nodeid": "tests/models/bert/test_modeling_bert.py::TestBert::test_x[fp16]",
                "test_job": "tests_torch" if run_id == "110:1" else "run_models_gpu",
                "hardware": "cpu" if run_id == "110:1" else "single-gpu",
            },
            {
                "pr": "42",
                "status_code": "ERROR",
                "test_nodeid": "tests_torch::oom_killed",
                "test_job": "tests_torch",
                "hardware": "cpu",
            },
        ]

    result = rerun_failed.snapshot("42", query=query, get_run=get_run, get_rows=rows)
    cpu, gpu = result["lanes"]["cpu"], result["lanes"]["gpu"]
    assert cpu["source_run"]["run_id"] == "110:1"
    assert gpu["source_run"]["run_id"] == "210:1"
    assert cpu["source_run"]["commit"] == "a" * 40
    assert gpu["source_run"]["commit"] == "b" * 40
    assert [r["run_id"] for r in cpu["active_runs"]] == ["120:1"]
    assert [r["run_id"] for r in gpu["active_runs"]] == ["220:1"]
    assert cpu["tests"][0]["model"] == "bert"
    assert cpu["tests"][0]["eligible"] is True
    assert cpu["tests"][1]["eligible"] is False


def test_preview_has_no_dispatch_action() -> None:
    assert "Run selected tests (coming soon)" in rerun_failed.PAGE_HTML
    assert "<button disabled" in rerun_failed.PAGE_HTML
    assert "/rerun-failed/data?pr=" in rerun_failed.PAGE_HTML
