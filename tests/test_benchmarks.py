from __future__ import annotations

import json
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.request import urlopen

import pytest

from transformersci.benchmarks import page, refresh
from transformersci.otel import trace_exporter


def rows():
    common = {
        "success": True,
        "model_id": "org/model<script>",
        "series_key": "gen",
        "scenario": "sdpa",
        "attn_implementation": "sdpa",
        "compile_mode": None,
        "batch_size": 1,
        "sequence_length": 128,
        "num_tokens_to_generate": 32,
        "commit_id": "a" * 40,
        "gpu_name": "GPU",
        "gpu_index": 0,
        "source": "main",
        "instance_type": "host",
        "sm_clock_mhz": 1500,
    }
    daily = [
        dict(
            common,
            label="daily-generation",
            group_id=f"day-{day}",
            started_at=f"2026-11-{day:02}T03:00:00Z",
            e2e_ms_median=value,
        )
        for day, value in ((20, 100), (21, 110), (22, 120))
    ]
    ab = [
        dict(
            common,
            label="ab",
            group_id="ab",
            ab_id="paired",
            ab_side=side,
            ab_round=i,
            pr_number=123,
            gpu_index=(i + (side == "head")) % 2,
            started_at="2026-11-22T04:00:00Z",
            e2e_ms_median=100 if side == "base" else 110,
        )
        for i in range(4)
        for side in ("base", "head")
    ]
    proc = [
        {
            "label": "daily-processors",
            "source": "main",
            "name": "vision",
            "group_id": "day-22",
            "started_at": "2026-11-22T03:00:00Z",
            "success": True,
            "median_ms": 3,
            "commit_id": "b" * 40,
            "inputs": "images",
            "batch": 1,
            "component": "vision",
            "stage": "encoder",
            "model": "org/vision",
            "threads": 8,
            "gpu_name": "GPU",
            "syncs": 2,
            "sync_sources": '{"copy":2}',
            "output_sha256": "abc123",
        }
    ]
    return daily + ab, proc


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    build = pytest.importorskip("tperf.dashboard.build")
    monkeypatch.setattr(build, "pr_titles", lambda prs: {123: "A <test> PR"})
    data = {
        "dataset": refresh.DATASET,
        "revision": "c" * 40,
        "updated_at": time.time(),
        "pages": refresh.finite(build.build(*rows())),
    }
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(data, allow_nan=False))
    monkeypatch.setenv("TCI_BENCHMARK_SNAPSHOT", str(path))
    return path, data


def test_overview_and_comparison_use_upstream_analysis(snapshot):
    status, html = page.render("/benchmarks", {})
    assert status == 200
    assert "/benchmarks/pr-123" in html and "A &lt;test&gt; PR" in html
    assert html.index(">Benchmarks</a>") < html.index(">Profile</a>")
    status, html = page.render("/benchmarks/pr-123", {})
    assert status == 200
    assert "REGRESSION" in html and "10 %" in html and "4 paired rounds" in html
    assert "org/model&lt;script&gt;" in html and "<script>" not in html


def test_daily_python_filters_and_svg(snapshot):
    status, html = page.render("/benchmarks/main", {})
    assert status == 200 and "<svg" in html and "<script" not in html
    assert "120 ms" in html and "+20.00%" in html
    assert "Latest output hashes" in html and "abc123" in html
    _, filtered = page.render(
        "/benchmarks/main", {"model": ["org/vision"], "processor_metric": ["syncs"]}
    )
    assert "2 syncs" in filtered
    assert "120 ms" not in filtered
    _, empty = page.render("/benchmarks/main", {"model": ["unknown"]})
    assert "No daily series match" in empty


def test_missing_invalid_and_stale_snapshot(tmp_path, monkeypatch, snapshot):
    path, data = snapshot
    data["updated_at"] = 1
    path.write_text(json.dumps(data))
    assert "more than 48 hours old" in page.render("/benchmarks", {})[1]
    assert page.render("/benchmarks/pr-999", {})[0] == 404
    assert page.render("/benchmarks/../../etc/passwd", {})[0] == 404
    path.write_text("invalid")
    assert page.render("/benchmarks", {})[0] == 503
    monkeypatch.setenv("TCI_BENCHMARK_SNAPSHOT", str(tmp_path / "absent.json"))
    assert page.render("/benchmarks", {})[0] == 503


def test_http_route(snapshot):
    server = ThreadingHTTPServer(("127.0.0.1", 0), trace_exporter.MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(
            f"http://127.0.0.1:{server.server_port}/benchmarks/main?model=org%2Fvision"
        ) as response:
            assert response.status == 200
            html = response.read().decode()
            assert "org/vision" in html and "120 ms" not in html
            assert response.headers["Content-Type"].startswith("text/html")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_refresher_pins_revision_and_keeps_last_good_snapshot(tmp_path, monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    build = pytest.importorskip("tperf.dashboard.build")
    monkeypatch.setattr(build, "pr_titles", lambda prs: {})
    data, processors = rows()
    root = tmp_path / "hub"
    for folder, items in [("data", data), ("processors", processors)]:
        target = root / folder / "2026-11"
        target.mkdir(parents=True)
        # Arrow infers columns from the first row; use a union for the fixture.
        keys = set().union(*(r.keys() for r in items))
        pq.write_table(
            pa.Table.from_pylist([{k: r.get(k) for k in keys} for r in items]),
            target / "rows.parquet",
        )

    class API:
        def repo_info(self, repo, repo_type):
            assert repo == refresh.DATASET and repo_type == "dataset"
            return type("Info", (), {"sha": "immutable-revision"})()

        def list_repo_files(self, repo, repo_type, revision):
            return ["data/2026-11/rows.parquet"]

    monkeypatch.setattr(hub, "HfApi", API)
    calls = []

    def download(repo, **kw):
        calls.append(kw)
        return str(root)

    monkeypatch.setattr(hub, "snapshot_download", download)
    path = tmp_path / "snapshot.json"
    assert refresh.refresh(path)
    assert calls[0]["revision"] == "immutable-revision"
    original = path.read_bytes()
    assert not refresh.refresh(path)
    assert len(calls) == 1
    # A new revision whose download fails must not overwrite the good snapshot.
    old = json.loads(path.read_text())
    old["revision"] = "older"
    path.write_text(json.dumps(old))
    original = path.read_bytes()
    monkeypatch.setattr(
        hub,
        "snapshot_download",
        lambda *a, **k: (_ for _ in ()).throw(OSError("offline")),
    )
    with pytest.raises(OSError):
        refresh.refresh(path)
    assert path.read_bytes() == original


def test_unbounded_intervals_are_valid_json(snapshot):
    _, data = snapshot
    payload = refresh.finite(
        {"ci": [-float("inf"), float("inf")], "value": float("nan")}
    )
    assert json.loads(json.dumps(payload, allow_nan=False)) == {
        "ci": [None, None],
        "value": None,
    }
    comparison = data["pages"]["pr-123.html"]
    comparison["abs"][0]["scenarios"][0]["metrics"][0]["ci"] = [None, None]
    assert "Unbounded (insufficient rounds)" in page.comparisons(comparison)


def test_theme_defaults_to_grafana_dark_and_preserves_explicit_light(snapshot):
    _, html = page.render("/benchmarks", {})
    assert 'data-theme="dark"' in html
    assert "prefers-color-scheme" not in html
    _, html = page.render("/benchmarks", {"theme": ["light"]})
    assert 'data-theme="light"' in html
    assert "/benchmarks/main?theme=light" in html
    assert "/benchmarks/pr-123?theme=light" in html
    _, html = page.render("/benchmarks/main", {"theme": ["light"]})
    assert 'name="theme" value="light"' in html
    _, html = page.render("/benchmarks", {"theme": ["<script>"]})
    assert 'data-theme="dark"' in html


def test_memory_reports_are_copied_linked_and_served_sandboxed(tmp_path, monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    build = pytest.importorskip("tperf.dashboard.build")
    if "reports" not in build.build.__code__.co_varnames:
        pytest.skip("pinned transformers-perf predates memory reports")
    monkeypatch.setattr(build, "pr_titles", lambda prs: {})
    data, processors = rows()
    root = tmp_path / "hub"
    for folder, items in [("data", data), ("processors", processors)]:
        target = root / folder / "2026-11"
        target.mkdir(parents=True)
        keys = set().union(*(r.keys() for r in items))
        pq.write_table(
            pa.Table.from_pylist([{k: r.get(k) for k in keys} for r in items]),
            target / "rows.parquet",
        )
    report = root / "memory/2026-11/day-22"
    report.mkdir(parents=True)
    (report / "report.html").write_text("<!doctype html><p>peak 1.9 GB</p>")
    (report / "host-flamegraph.html").write_text("<script>flame()</script>")
    files = [
        "data/2026-11/rows.parquet",
        "memory/2026-11/day-22/report.html",
        "memory/2026-11/day-22/host-flamegraph.html",
        "memory/2026-11/not-a-daily-group/report.html",
    ]

    class API:
        def repo_info(self, repo, repo_type):
            return type("Info", (), {"sha": "rev"})()

        def list_repo_files(self, repo, repo_type, revision):
            assert revision == "rev"
            return files

    downloads = []
    monkeypatch.setattr(hub, "HfApi", API)
    monkeypatch.setattr(
        hub, "snapshot_download", lambda repo, **kw: downloads.append(kw) or str(root)
    )
    path = tmp_path / "benchmarks" / "snapshot.json"
    stale = path.parent / "memory" / "day-1"
    stale.mkdir(parents=True)
    assert refresh.refresh(path)
    # Only the linked report is fetched, the stale copy is gone
    assert downloads[1]["allow_patterns"] == [
        "memory/2026-11/day-22/report.html",
        "memory/2026-11/day-22/host-flamegraph.html",
    ]
    assert not stale.exists()
    assert (
        (path.parent / "memory/day-22/report.html").read_text().endswith("1.9 GB</p>")
    )

    monkeypatch.setenv("TCI_BENCHMARK_SNAPSHOT", str(path))
    status, html = page.render("/benchmarks/main", {})
    assert status == 200
    assert 'href="/benchmarks/memory/day-22/report.html"' in html
    assert "Memory report, 2026-11-22" in html and "<th>Memory</th>" in html
    assert page.memory_file("/benchmarks/memory/day-22/report.html") == (
        200,
        b"<!doctype html><p>peak 1.9 GB</p>",
    )
    assert (
        page.memory_file("/benchmarks/memory/day-22/../../snapshot.json.html")[0] == 404
    )
    assert page.memory_file("/benchmarks/memory/../snapshot/report.html")[0] == 404
    assert page.memory_file("/benchmarks/memory/day-21/report.html")[0] == 404

    server = ThreadingHTTPServer(("127.0.0.1", 0), trace_exporter.MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}/benchmarks/memory/day-22"
        with urlopen(f"{base}/host-flamegraph.html") as response:
            assert response.read() == b"<script>flame()</script>"
            assert response.headers["Content-Security-Policy"].startswith(
                "sandbox allow-scripts"
            )
            assert response.headers["Content-Type"].startswith("text/html")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
