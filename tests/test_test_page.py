import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

import pytest

from transformersci.otel import test_page as tp
from transformersci.otel import trace_exporter as te

NODE = "tests/models/gpt_oss/test_modeling_gpt_oss.py::GptOssModelTest::test_flash_attn_2_fp32_ln"
TRACE = "2cd65c867632d67c0cc01c2cfb05415b"


def tag(key, value):
    return {"key": key, "type": "string", "value": value}


def make_span(nodeid=NODE, status="ERROR", start=1_791_897_921_926_012, **extra):
    tags = [
        tag("pytest.span_type", "test"),
        tag("pytest.nodeid", nodeid),
        tag("otel.status_code", status),
    ] + [tag(k, v) for k, v in extra.items()]
    return {"startTime": start, "duration": 1_100_000, "tags": tags, "logs": []}


def test_span_facts_reads_this_runs_span_not_the_latest_run():
    trace = {
        "spans": [
            make_span(
                **{
                    "pytest.cuda_delta_bytes": "-93696",
                    "pytest.cuda_inherited_bytes": "17136128",
                    "pytest.cuda_retained_bytes": "17042432",
                }
            ),
            make_span(nodeid="tests/other.py::test_x"),
        ]
    }

    facts = te.test_span_facts(trace, NODE)

    assert facts == {
        "status": "ERROR",
        "duration_seconds": 1.1,
        "started_at": 1_791_897_921.926012,
        "cuda_inherited_bytes": 17136128.0,
        "cuda_retained_bytes": 17042432.0,
        "cuda_delta_bytes": -93696.0,
    }


def test_span_facts_prefers_a_failing_rerun_and_the_worker_duration():
    trace = {
        "spans": [
            make_span(
                status="ERROR",
                start=1_000_000,
                **{"pytest.worker_duration_seconds": "2.5"},
            ),
            make_span(status="OK", start=9_000_000),
        ]
    }

    facts = te.test_span_facts(trace, NODE)

    assert facts["status"] == "ERROR"
    assert facts["duration_seconds"] == 2.5
    # No CUDA tags: absent, never zero.
    assert "cuda_delta_bytes" not in facts
    assert te.test_span_facts(trace, "tests/missing.py::test_y") == {}


def test_render_escapes_everything_from_the_url_and_trace():
    evil = "tests/x.py::test_<script>alert(1)</script>"
    page = tp.render_test_page(
        {
            "test_nodeid": evil,
            "trace_id": TRACE,
            "status": "ERROR",
            "has_trace": True,
            "details": [
                {
                    "exception_type": "ValueError<b>",
                    "exception_message": "</pre><script>x</script>",
                    "exception_stacktrace": "line",
                }
            ],
        }
    )

    assert "<script>alert(1)</script>" not in page
    assert "<script>x</script>" not in page
    assert "ValueError&lt;b&gt;" in page
    # The JSON context block cannot close its own script tag.
    assert "</script>alert" not in page and "\\u003c/script>" in page


def test_render_full_failure_page():
    page = tp.render_test_page(
        {
            "test_nodeid": NODE,
            "trace_id": TRACE,
            "status": "ERROR",
            "job": "run_models_gpu",
            "pr": "main",
            "run_id": "38401932908:1",
            "module": "test_modeling_gpt_oss.py",
            "hardware": "multi-gpu",
            "runner_type": "aws-g5-12xlarge-cache",
            "run_url": "https://github.com/huggingface/transformers/actions/runs/38401932908/attempts/1",
            "duration_seconds": 1.1,
            "started_at": 1_791_897_921.9,
            "cuda_inherited_bytes": 17136128.0,
            "cuda_retained_bytes": 17042432.0,
            "cuda_delta_bytes": -93696.0,
            "has_trace": True,
            "details": [
                {
                    "exception_type": "ImportError",
                    "exception_message": "`kernels` is either not installed",
                    "exception_stacktrace": "self = <...>",
                }
            ],
        }
    )

    # Duration cell: date · relative · duration; the exact time is the tooltip.
    assert (
        "Oct 13<span class='sep'>·</span><span data-ts='1791897922' "
        "title='2026-10-13 13:25 UTC'>" in page
    )
    assert "<span class='sep'>·</span>1.1 s</div>" in page
    # Net change -91.5 KiB: no leak, readings in the tooltip.
    assert "<div class='v good'>No</div>" in page
    assert (
        "inherited 16.3 MiB · left for next test 16.3 MiB · net change -91.5 KiB"
        in page
    )
    assert "Serge Actions</div><iframe" in page
    for text in (
        "aws-g5-12xlarge-cache · xGPU",
        "pytest -svx &#x27;" + NODE + "&#x27;",
        "ImportError",
        "Potential related issues",
        "/serge-actions?",
    ):
        assert text in page, text
    # No Run/Job/PR buttons: in Grafana the dashboard's own link bar has them.
    assert "pytest-observability-run?" not in page
    # The Job chip opens the Job page in the top window (the page runs in an iframe).
    assert "/failure/job-page?" in page and "target='_top'" in page


def test_render_passed_and_missing_trace():
    passed = tp.render_test_page(
        {"test_nodeid": NODE, "trace_id": TRACE, "status": "OK", "has_trace": True}
    )
    assert "This test passed in this run." in passed
    # A passing test keeps its related issues.
    assert "Potential related issues" in passed

    missing = tp.render_test_page({"test_nodeid": NODE, "trace_id": TRACE})
    assert "Trace not found." in missing
    assert "No trace selected." in tp.render_test_page({"test_nodeid": NODE})


def test_memory_leak_cell():
    assert tp.memory_leak({"cuda_delta_bytes": 2 * 1024**2})[:2] == ("+2.0 MiB", "bad")
    assert tp.memory_leak({"cuda_delta_bytes": 0.0})[:2] == ("No", "good")
    # No reading (CPU job, older plugin) is unknown, never "No".
    assert tp.memory_leak({})[:2] == ("—", "unknown")
    page = tp.render_test_page({"test_nodeid": NODE, "cuda_delta_bytes": 4096.0})
    assert "<div class='v bad'>+4.0 KiB</div>" in page


def test_compact_ago():
    assert tp.compact_ago(10) == "just now"
    assert tp.compact_ago(16 * 3600 + 5) == "16h ago"
    assert tp.compact_ago(3 * 86400) == "3d ago"
    assert tp.compact_ago(200 * 86400) == "6mo ago"


def test_reproduce_command_and_formatting():
    assert tp.reproduce_command("tests/a.py::T::test_it's") == (
        "pytest -svx 'tests/a.py::T::test_it'\"'\"'s'"
    )
    assert tp.reproduce_command("utils/checkers.py::check_repo") == "make 'check_repo'"
    assert tp.format_duration(0.25) == "250 ms"
    assert tp.format_duration(125) == "2m 05s"
    assert tp.format_duration(None) == "—"
    assert tp.format_bytes(-93696, signed=True) == "-91.5 KiB"
    assert tp.format_bytes(0, signed=True) == "0 B"
    assert tp.format_bytes(3 * 1024**3) == "3.0 GiB"


@pytest.fixture
def server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), te.MetricsHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    worker.join()


def test_route_renders_from_one_trace_fetch(server, monkeypatch):
    fetched = []
    trace = {
        "spans": [make_span(**{"pytest.cuda_delta_bytes": "-93696"})],
        "processes": {
            "p1": {
                "tags": [
                    tag("vcs.repository.name", "huggingface/transformers"),
                    tag("transformers.test.job", "run_models_gpu"),
                    tag("cicd.pipeline.run.id", "38401932908:1"),
                ]
            }
        },
    }
    monkeypatch.setattr(
        te, "fetch_failure_trace", lambda trace_id: fetched.append(trace_id) or trace
    )

    with urlopen(
        server + "/test?" + urlencode({"latest_trace": TRACE, "test_nodeid": NODE})
    ) as response:
        page = response.read().decode()
        assert response.headers["Content-Type"].startswith("text/html")

    # latest_trace stands in for a missing trace_id, as on the Grafana page.
    assert fetched == [TRACE]
    assert "-91.5 KiB" in page
    assert "run_models_gpu" in page
    assert "This test failed" not in page


@pytest.mark.parametrize(
    "query",
    [
        {},
        {"test_nodeid": NODE, "trace_id": "not-hex"},
        {"test_nodeid": "a\nb"},
    ],
)
def test_route_rejects_bad_input(server, query):
    with pytest.raises(HTTPError) as err:
        urlopen(server + "/test?" + urlencode(query))
    assert err.value.code == 400
