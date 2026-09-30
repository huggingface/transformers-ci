from __future__ import annotations

import io
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

import pytest

from transformersci.otel import related_issues as ri
from transformersci.otel import trace_exporter as te

NODE = "tests/trainer/test_trainer.py::TrainerIntegrationTest::test_end_to_end_example"
TRACE = "e48b720b8543e9950d7766d80011338b"


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_RELORE_URL", "http://relore.test")
    ri._cache.clear()
    ri._inflight.clear()


def hit(number=1, **extra):
    return {
        "repo": ri.REPOSITORY,
        "number": number,
        "type": "issue",
        "title": "A matching failure",
        **extra,
    }


def mock_response(monkeypatch, payload, version=ri.WIRE_VERSION):
    def open_request(request, timeout):
        assert timeout == 8
        assert request.get_header("X-relore-client") == ri.WIRE_VERSION
        assert json.loads(request.data)["repos"] == [ri.REPOSITORY]
        response = io.BytesIO(json.dumps(payload).encode())
        response.headers = {"x-relore-version": version}
        return response

    monkeypatch.setattr(ri, "urlopen", open_request)


def test_query_preserves_exact_test_and_expands_error_context():
    query = ri.search_payload(
        NODE + "[a'b&c]",
        [
            {
                "exception_type": "RuntimeError",
                "exception_message": "dataset loading failed",
            }
        ],
    )
    assert NODE + "[a'b&c]" in query["query"]
    assert "RuntimeError: dataset loading failed" in query["query"]
    assert "tests" not in query  # no AND-filter hiding failures in other tests
    assert query["expand"] is True
    assert query["kind"] == "failure"


def test_dedup_ranking_limit_and_safe_links(monkeypatch):
    mock_response(
        monkeypatch,
        {
            "hits": [
                hit(
                    3,
                    type="pr",
                    url="javascript:alert(1)",
                    title="<script>unsafe</script>",
                ),
                hit(3),
                hit(7, repo="private/repository"),
                hit(-1),
                hit(True),
                *[hit(n) for n in range(10, 20)],
            ]
        },
    )
    hits = ri.search("http://relore.test", ri.search_payload(NODE, []))
    assert [h["number"] for h in hits] == [3, 10, 11, 12, 13]
    assert hits[0]["url"] == "https://github.com/huggingface/transformers/pull/3"
    # Upstream strings remain text; the browser uses textContent, never HTML.
    assert hits[0]["title"] == "<script>unsafe</script>"


@pytest.mark.parametrize(
    "payload,version",
    [
        ({"hits": []}, "0.3.18"),
        ({"hits": []}, None),
        ({"error": "broken"}, ri.WIRE_VERSION),
        ({"hits": [None]}, ri.WIRE_VERSION),
        ({"hits": [], "padding": "x" * ri.MAX_RESPONSE_BYTES}, ri.WIRE_VERSION),
    ],
)
def test_bad_upstream_is_unavailable_not_no_matches(monkeypatch, payload, version):
    mock_response(monkeypatch, payload, version)
    assert ri.lookup(TRACE, NODE, lambda: []) == {"status": "unavailable", "hits": []}


def test_timeout_and_short_failure_cache(monkeypatch):
    calls = []
    clock = [100.0]
    monkeypatch.setattr(ri.time, "monotonic", lambda: clock[0])

    def fail(*args):
        calls.append(1)
        raise TimeoutError()

    monkeypatch.setattr(ri, "search", fail)
    assert ri.lookup(TRACE, NODE, lambda: [])["status"] == "unavailable"
    ri.lookup(TRACE, NODE, lambda: [])
    assert len(calls) == 1
    clock[0] += 16
    ri.lookup(TRACE, NODE, lambda: [])
    assert len(calls) == 2
    assert not ri._inflight


def test_cache_preserves_trace_and_test_identity(monkeypatch):
    calls = []
    monkeypatch.setattr(ri, "search", lambda *args: calls.append(args) or [])

    def details():
        return [{"exception_type": "RuntimeError", "exception_message": "bad"}]

    assert ri.lookup(TRACE, NODE, details)["context"] == "failure"
    ri.lookup(TRACE, NODE, details)
    ri.lookup("f" * 32, NODE, details)
    ri.lookup(TRACE, NODE + "[other]", details)
    assert len(calls) == 3
    for n in range(260):
        ri.lookup("", str(n), lambda: [])
    assert len(ri._cache) == 256


def test_missing_trace_searches_by_test_only(monkeypatch):
    mock_response(monkeypatch, {"hits": []})
    result = ri.lookup(TRACE, NODE, lambda: [])
    assert result == {"status": "ok", "hits": [], "context": "test-only"}


def test_disabled_and_busy_do_not_call_upstreams(monkeypatch):
    def unexpected():
        pytest.fail("unexpected trace fetch")

    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_RELORE_URL", "")
    assert ri.lookup(TRACE, NODE, unexpected)["status"] == "unavailable"
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_RELORE_URL", "http://relore.test")
    ri._inflight.add(("http://relore.test", TRACE, NODE))
    assert ri.lookup(TRACE, NODE, unexpected)["status"] == "busy"
    ri._inflight.add(("http://relore.test", "other", NODE))
    assert ri.lookup("f" * 32, NODE, unexpected)["status"] == "busy"


@pytest.fixture
def server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), te.MetricsHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    worker.join()


def test_route_shell_validation_and_selected_trace(server, monkeypatch):
    fetched = []
    monkeypatch.setattr(
        te, "get_trace", lambda trace: fetched.append(trace) or {"ok": True}
    )
    monkeypatch.setattr(te, "trace_repository", lambda trace: ri.REPOSITORY)
    monkeypatch.setattr(
        te,
        "extract_failure_details",
        lambda trace, node: [
            {"exception_type": "RuntimeError", "exception_message": node},
        ],
    )
    mock_response(monkeypatch, {"hits": [hit()]})
    with urlopen(server + "/related-issues") as response:
        assert b"Searching related issues" in response.read()
    with urlopen(server + "/related-issues?view=summary") as response:
        summary = response.read().decode()
        assert 'class="tci-meta"' in summary
        assert summary.index("Reproduce locally") < summary.index(
            "Potential related issues"
        )
    assert fetched == []  # shell returns before any upstream work
    params = urlencode({"format": "json", "trace_id": TRACE, "test_nodeid": NODE})
    with urlopen(server + "/related-issues?" + params) as response:
        result = json.load(response)
        assert response.headers["Cache-Control"] == "no-store"
    assert fetched == [TRACE]
    assert result["context"] == "failure"
    assert result["hits"][0]["number"] == 1
    with pytest.raises(HTTPError) as err:
        urlopen(server + "/related-issues?format=json&trace_id=invalid&test_nodeid=x")
    assert err.value.code == 400


def test_foreign_trace_context_is_not_sent_to_relore(server, monkeypatch):
    monkeypatch.setattr(te, "get_trace", lambda trace: {"secret": "context"})
    monkeypatch.setattr(te, "trace_repository", lambda trace: "other/repository")
    calls = []
    monkeypatch.setattr(ri, "search", lambda url, payload: calls.append(payload) or [])
    params = urlencode({"format": "json", "trace_id": TRACE, "test_nodeid": NODE})
    with urlopen(server + "/related-issues?" + params) as response:
        assert json.load(response)["context"] == "test-only"
    assert calls[0]["query"] == NODE


def test_serge_actions_shell(server):
    with urlopen(server + "/serge-actions") as response:
        page = response.read().decode()
        assert response.headers["Cache-Control"] == "no-store"
    assert "__ACTIONS__" not in page
    for label in ("New issue", "Fix it!", "WDYT?"):
        assert label in page
    assert "b.disabled=true" in page
    assert "fetch('/api/user'" in page
