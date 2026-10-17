from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

from transformersci.otel import branch_runs, trace_exporter


def vector(*samples: tuple[dict[str, str], float]) -> list[dict]:
    return [{"metric": labels, "value": [0, str(value)]} for labels, value in samples]


def run(run_id: str, **labels: str) -> dict[str, str]:
    return {"run_id": run_id, "pr": "main", **labels}


def test_rows_join_per_run_newest_first_with_every_lane() -> None:
    replies = {
        "start": vector((run("a"), 100.0), (run("b"), 300.0)),
        "jobs": vector((run("a"), 3), (run("b"), 4)),
        "total": vector((run("b"), 358903)),
        "rate": vector((run("b"), 0.2)),
        "commit": vector((run("b", commit_message="MoE", html_url="u"), 1)),
        "lanes": vector(
            (run("b", hardware="single-gpu"), 1),
            (run("b", hardware="multi-gpu"), 1),
            (run("a", hardware="cpu"), 1),
        ),
        "active": vector((run("a"), 1)),
    }
    rows = branch_runs.branch_rows(replies)
    assert [row["run_id"] for row in rows] == ["b", "a"]
    assert rows[0]["lanes"] == ["multi-gpu", "single-gpu"]
    assert rows[0]["commit_message"] == "MoE" and rows[0]["total"] == 358903
    assert rows[0]["live"] is False and rows[1]["live"] is True
    assert rows[1]["lanes"] == ["cpu"] and rows[1]["total"] is None


def test_rows_keep_the_ten_newest() -> None:
    replies = {"start": vector(*((run(str(i)), float(i)) for i in range(15)))}
    rows = branch_runs.branch_rows(replies)
    assert [row["run_id"] for row in rows] == [str(i) for i in range(14, 4, -1)]


def test_every_query_keeps_branches_without_a_pr_only() -> None:
    for name, expr in branch_runs.QUERIES.items():
        assert 'pr!~"[0-9]' in expr, name
        assert 'pr=~"' not in expr, name


def test_pages_share_one_common_js(monkeypatch) -> None:
    calls: list[str] = []

    def prometheus(query: str) -> list[dict]:
        calls.append(query)
        return vector((run("a"), 100.0)) if "topk" in query else []

    monkeypatch.setattr(trace_exporter, "_prometheus_vector", prometheus)
    monkeypatch.setattr(trace_exporter, "_branch_runs_cache", {})
    server = ThreadingHTTPServer(("127.0.0.1", 0), trace_exporter.MetricsHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        shared = Path(trace_exporter.__file__).with_name("common.js").read_bytes()
        for page in ("pr-ci-view", "pr-table", "project-health", "branch-runs"):
            assert urlopen(f"{base}/{page}/common.js").read() == shared
            html = urlopen(f"{base}/{page}").read().decode()
            assert f'<script src="/{page}/common.js"></script>' in html, page
        body = json.loads(urlopen(f"{base}/branch-runs/rows").read())
        assert [row["run_id"] for row in body["rows"]] == ["a"]
        first = len(calls)
        urlopen(f"{base}/branch-runs/rows").read()
        assert len(calls) == first
    finally:
        server.shutdown()
