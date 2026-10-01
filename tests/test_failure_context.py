import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlparse
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
    assert f"GitHub job log {te.GITHUB_MARK_SVG}</a>" in page
    assert "GitHub run" not in page and "↗" not in page
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
    assert 'href="https://github.com/r"' in page
    assert f"GitHub run {te.GITHUB_MARK_SVG}</a>" in page


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
    assert f"GitHub run {te.GITHUB_MARK_SVG}" in page


def test_summary_links_scoped_job_page_and_exact_job():
    from transformersci.otel import related_issues as ri

    page = ri.SUMMARY_HTML
    assert "fetch('/failure/context?'" in page
    assert "'/failure/job-page?'+new URLSearchParams" in page
    assert "runner:info.job_url||info.run_url" in page
    # Only GitHub (new tab) and Grafana-internal (same window) targets are linked.
    assert "/^https:\\/\\/github\\.com\\//" in page
    assert "/^\\/failure\\/job-page\\?/" in page


def run_rows():
    def row(nodeid, trace, hardware, duration, status="ERROR"):
        return {
            "test_nodeid": nodeid,
            "trace_id": trace,
            "hardware": hardware,
            "duration_seconds": duration,
            "status_code": status,
            "test_job": "run_models_gpu",
            "pr": "main",
        }

    return [
        row("tests/models/a/test_a.py::T::test_x", "1" * 32, "single-gpu", 50),
        row("tests/models/b/test_b.py::T::test_y", "2" * 32, "single-gpu", 40),
        row(NODE, "3" * 32, "single-gpu", 1),
        row(NODE, TRACE, "multi-gpu", 2),
    ]


def test_run_focus_row_is_shown_highlighted_and_opened():
    page = te.render_run_html(
        "36807556167:1", run_rows(), limit=1, focus=NODE, focus_trace=TRACE
    )
    # Below the top-1 cut, but still listed, and the multi-gpu (trace) row.
    assert page.count("id='focus'") == 1
    focus_row = page[page.index("id='focus'") :].split("</tr>")[0]
    assert TRACE in focus_row and "xGPU" in focus_row
    assert "(showing top 1)" in page
    assert "getElementById('focus')" in page and "b.click()" in page
    # No trace given: the first row with that node id, in table (duration) order.
    page = te.render_run_html("36807556167:1", run_rows(), focus=NODE)
    assert TRACE in page[page.index("id='focus'") :].split("</tr>")[0]
    page = te.render_run_html("1:1", run_rows(), focus=NODE, focus_trace="3" * 32)
    assert "3" * 32 in page[page.index("id='focus'") :].split("</tr>")[0]
    assert "id='focus'" not in te.render_run_html("1:1", run_rows(), focus="nope")


def test_run_focus_opens_its_group():
    rows = run_rows() * 0 + [
        {**run_rows()[0], "test_nodeid": f"tests/models/c/test_c.py::T::t{i}"}
        for i in range(te._RUN_GROUP_OPEN_MAX + 5)
    ]
    focused = {**rows[0], "test_nodeid": "tests/models/c/test_c.py::T::zz"}
    rows.append(focused)
    page = te.render_run_html(
        "1:1", rows, group="model", limit=2, focus=focused["test_nodeid"]
    )
    assert "<details open" in page and "id='focus'" in page


def test_job_page_redirect_scopes_hardware_and_focus(server, monkeypatch):
    monkeypatch.setattr(
        te,
        "get_trace",
        lambda t: make_trace(**{"transformers.test.hardware": "multi-gpu"}),
    )
    query = urlencode(
        {
            "trace_id": "",
            "latest_trace": TRACE,
            "test_nodeid": NODE,
            "job": "run_models_gpu",
            "run_id": "36807556167:1",
            "pr": "main",
            "status": "ERROR",
        }
    )
    status, location = get_redirect(server, "/failure/job-page?" + query)
    assert status == 302
    assert location.startswith(
        "/d/pytest-observability-by-job/pytest-observability-job?"
    )
    params = parse_qs(urlparse(location).query)
    assert params["var-hardware"] == ["multi-gpu"]
    assert params["var-focus"] == [NODE] and params["var-focus_trace"] == [TRACE]
    assert params["var-status_filter"] == ["ERROR"]
    assert params["var-job"] == ["run_models_gpu"]
    # No trace at all: still a Job page, every hardware.
    status, location = get_redirect(
        server, "/failure/job-page?" + urlencode({"job": "j", "run_id": "1:1"})
    )
    params = parse_qs(urlparse(location).query)
    assert status == 302 and params["var-hardware"] == ["$__all"]


def get_redirect(server, path):
    parsed = urlparse(server)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port)
    conn.request("GET", path)
    response = conn.getresponse()
    location = response.getheader("Location")
    conn.close()
    return response.status, location


def test_exception_message_line_first_line_capped():
    span = {
        "logs": [
            {
                "fields": tags(
                    event="exception",
                    **{"exception.message": "\n  boom: shard 40 missing\nsecond line"},
                )
            }
        ]
    }
    assert te.exception_message_line(span) == "boom: shard 40 missing"
    span["logs"][0]["fields"] = tags(
        event="exception", **{"exception.message": "x" * 300}
    )
    assert te.exception_message_line(span) == "x" * 200 + "…"
    assert te.exception_message_line({"logs": []}) == ""


def test_run_search_server_side_and_error_shown():
    rows = run_rows()
    rows[3]["exception_type"] = "FileNotFoundError"
    rows[3]["exception_message"] = "No such file: model-00040-of-00047.safetensors"
    page = te.render_run_html("1:1", rows, query={}, q="glm4 xgpu safetensors")
    assert page.count("<tr><td") + page.count("<tr id=") == 1
    assert "FileNotFoundError: No such file: model-00040" in page
    assert "id='q'" in page and 'value="glm4 xgpu safetensors"' in page
    # The search box sits outside the live-refreshed body.
    assert page.index("id='q'") < page.index("id='runbody'")
    page = te.render_run_html("1:1", rows, query={}, q="nothing-matches")
    assert "No test matches <b>nothing-matches</b> among 4" in page
    # Truncated lists tell the box that Enter can search every row.
    assert "data-truncated='1'" in te.render_run_html("1:1", rows, query={}, limit=2)
    assert te.run_row_matches(rows[0], "models/a test_x") is True
    assert te.run_row_matches(rows[0], "models/a xgpu") is False


def test_summary_github_links_carry_the_github_mark():
    from transformersci.otel import related_issues as ri

    assert "__GITHUB_MARK__" not in ri.SUMMARY_HTML
    assert ri.GITHUB_MARK_PATH in ri.SUMMARY_HTML and "↗" not in ri.SUMMARY_HTML
