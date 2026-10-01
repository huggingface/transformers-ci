import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

import pytest

import transformersci.otel.trace_exporter as te

TRACE = "2412abaff2fe8bfe905ded4ecc48ad07"
NODE = "tests/models/glm4_moe/test_modeling_glm4_moe.py::Glm4MoeIntegrationTest::test_1_dynamic_cache"
RUNNER = "aws-g5-12xlarge-cache-use1-public-80-85ptq-runner-t2fnd"
SHA = "d6c1e71bd717bf092f8293f0c3c9bd4a5ac5401a"
REPO = "huggingface/transformers"


def tags(**values):
    return [
        {"key": key, "type": "string", "value": value} for key, value in values.items()
    ]


def make_trace(**extra):
    resource = {
        "vcs.repository.name": REPO,
        "cicd.pipeline.run.id": "36807556167:1",
        "transformers.test.job": "run_models_gpu",
        "vcs.ref.head.name": "main",
        "vcs.ref.head.revision": SHA,
        "transformers.test.runner.type": "aws-g5-12xlarge-cache",
        "transformers.test.runner.name": RUNNER,
        **extra,
    }
    # Spans are never read: the context comes from process tags only.
    return {
        "traceID": TRACE,
        "spans": None,
        "processes": {"p1": {"tags": tags(**resource)}},
    }


@pytest.fixture(autouse=True)
def clear_caches():
    with te._trace_cache_lock:
        te._trace_cache.clear()
    te._failure_processes.clear()
    yield
    te._failure_processes.clear()
    assert te._failure_fetches == {}


def test_failure_context_reads_process_tags_only():
    context = te.failure_context(make_trace(), NODE)
    assert context == {
        "repository": REPO,
        "run_id": "36807556167:1",
        "run_url": f"https://github.com/{REPO}/actions/runs/36807556167/attempts/1",
        "job_url": "",
        "pr": "main",
        "pr_url": "",
        "hardware": "gpu",
        "runner_type": "aws-g5-12xlarge-cache",
        "runner_name": RUNNER,
        "file_url": f"https://github.com/{REPO}/blob/{SHA}/tests/models/glm4_moe/test_modeling_glm4_moe.py",
    }
    assert te.failure_context(None, NODE) == {}


def test_failure_context_pr_and_pr_comment_revision():
    trace = make_trace(
        **{
            "vcs.change.id": "49220",
            "transformers.test.ci_event": "pr-comment",
            "service.version": "a" * 40,
            "transformers.test.hardware": "multi-gpu",
        }
    )
    context = te.failure_context(trace, NODE)
    assert context["pr"] == "49220"
    assert context["pr_url"] == f"https://github.com/{REPO}/pull/49220"
    assert context["hardware"] == "multi-gpu"
    assert f"/blob/{'a' * 40}/" in context["file_url"]


def test_failure_context_links_the_stamped_job():
    trace = make_trace(**{"cicd.pipeline.task.run.id": "110196197506"})
    assert te.failure_context(trace, NODE)["job_url"] == (
        f"https://github.com/{REPO}/actions/runs/36807556167/job/110196197506"
    )
    for job_id in ("", "unknown", "1/../x"):
        trace = make_trace(**{"cicd.pipeline.task.run.id": job_id})
        assert te.failure_context(trace, NODE)["job_url"] == ""


def test_render_failure_html_job_log_link():
    page = te.render_failure_html(
        TRACE,
        [
            {
                "test_nodeid": NODE,
                "exception_type": "FileNotFoundError",
                "exception_message": "missing",
                "exception_stacktrace": "",
                "github_url": "https://github.com/f",
                "job_log_url": "https://github.com/j?a=1&b=2",
                "run_url": "https://github.com/r",
            }
        ],
    )
    assert 'href="https://github.com/j?a=1&amp;b=2"' in page
    assert "GitHub job log ↗" in page and "GitHub run ↗" not in page
    page = te.render_failure_html(
        TRACE,
        [
            {
                "test_nodeid": NODE,
                "exception_type": "E",
                "exception_message": "",
                "exception_stacktrace": "",
                "job_log_url": "",
                "run_url": "https://github.com/r",
            }
        ],
    )
    assert 'href="https://github.com/r"' in page and "GitHub run ↗" in page


@pytest.fixture
def server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), te.MetricsHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    worker.join()


def test_context_endpoint_uses_memoized_trace(server, monkeypatch):
    monkeypatch.setattr(te, "get_trace", lambda trace_id: pytest.fail("fetched"))
    with te._trace_cache_lock:
        te._trace_cache[TRACE] = make_trace()
    query = urlencode({"trace_id": TRACE, "test_nodeid": NODE})
    with urlopen(server + "/failure/context?" + query) as response:
        assert json.load(response)["runner_type"] == "aws-g5-12xlarge-cache"
    with pytest.raises(HTTPError) as err:
        urlopen(server + "/failure/context?trace_id=bad&test_nodeid=x")
    assert err.value.code == 400


def test_context_endpoint_fetches_once_then_reuses_process_tags(server, monkeypatch):
    fetched = []
    monkeypatch.setattr(te, "get_trace", lambda t: fetched.append(t) or make_trace())
    query = urlencode({"trace_id": TRACE, "test_nodeid": NODE})
    for _ in range(2):
        with urlopen(server + "/failure/context?" + query) as response:
            assert json.load(response)["runner_name"] == RUNNER
    assert fetched == [TRACE]
    # /failure on the same page load reuses nothing heavy: it needs the spans,
    # but a later context request is served from the kept process tags.
    assert set(te._failure_processes[TRACE][1]) == {"processes"}


def test_concurrent_panels_share_one_fetch(monkeypatch):
    started, release = threading.Event(), threading.Event()
    fetched = []

    def slow_get(trace_id):
        fetched.append(trace_id)
        started.set()
        release.wait(5)
        return make_trace()

    monkeypatch.setattr(te, "get_trace", slow_get)
    results = []
    leader = threading.Thread(
        target=lambda: results.append(te.fetch_failure_trace(TRACE))
    )
    leader.start()
    assert started.wait(5)
    follower = threading.Thread(
        target=lambda: results.append(te.failure_context_trace(TRACE))
    )
    follower.start()
    release.set()
    leader.join(5)
    follower.join(5)
    assert fetched == [TRACE]
    assert len(results) == 2 and all(r["processes"] for r in results)


def test_failure_page_links_job_log(server, monkeypatch):
    monkeypatch.setattr(te, "get_trace", lambda t: make_trace())
    monkeypatch.setattr(
        te,
        "extract_failure_details",
        lambda trace, node: [
            {
                "test_nodeid": NODE,
                "exception_type": "FileNotFoundError",
                "exception_message": "m",
                "exception_stacktrace": "",
            }
        ],
    )
    monkeypatch.setattr(te, "annotate_github_links", lambda trace, details: None)
    query = urlencode({"trace_id": TRACE, "test_nodeid": NODE})
    with urlopen(server + "/failure?" + query) as response:
        page = response.read().decode()
    assert (
        "https://github.com/huggingface/transformers/actions/runs/36807556167/attempts/1"
        in page
    )
    assert "GitHub run ↗" in page


def test_summary_links_scoped_job_page_and_exact_job():
    from transformersci.otel import related_issues as ri

    page = ri.SUMMARY_HTML
    assert "fetch('/failure/context?'" in page
    assert "'var-hardware':hardware||'$__all'" in page
    assert "runner:info.job_url||info.run_url" in page
    # Only GitHub (new tab) and Grafana-internal (same window) targets are linked.
    assert "/^https:\\/\\/github\\.com\\//" in page and "/^\\/d\\//" in page
