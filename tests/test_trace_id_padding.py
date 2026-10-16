"""Tempo drops leading zeros from trace ids; the exporter pads them back."""

from __future__ import annotations

import threading
from http.server import ThreadingHTTPServer
from urllib.request import urlopen

from transformersci.otel import trace_exporter as te

SHORT = "72c31cd0db949c374ff0c4f27ea5594"  # 31 characters, as Tempo search returns it
FULL = "0" + SHORT


def test_normalize_trace_id():
    assert te.normalize_trace_id(SHORT) == FULL
    assert te.normalize_trace_id(FULL.upper()) == FULL
    assert te.normalize_trace_id(" " + FULL + " ") == FULL
    assert te.normalize_trace_id("") == ""
    assert te.normalize_trace_id("not-hex") == "not-hex"
    assert te.normalize_trace_id("a" * 33) == "a" * 33
    assert te.normalize_trace_id("bad") == "bad"  # too short to be a real id


def test_query_trace_ids_are_padded_and_other_params_kept():
    path = (
        f"/test?trace_id={SHORT}&test_nodeid=tests/a.py::T::test_x&focus_trace="
        + "1f" * 13
    )
    fixed = te._normalize_trace_query(path)
    assert (
        f"trace_id={FULL}" in fixed and "focus_trace=" + "000000" + "1f" * 13 in fixed
    )
    assert te.urlparse(fixed).path == "/test"
    assert te.parse_qs(te.urlparse(fixed).query)["test_nodeid"] == [
        "tests/a.py::T::test_x"
    ]
    assert te._normalize_trace_query("/test?test_nodeid=x") == "/test?test_nodeid=x"


def test_search_results_and_fetched_traces_carry_32_hex_ids(monkeypatch):
    assert te.tempo_trace_to_jaeger(SHORT, {"batches": []})["traceID"] == FULL


def test_test_page_accepts_a_trace_id_without_its_leading_zero(monkeypatch):
    fetched = []
    monkeypatch.setattr(
        te, "fetch_failure_trace", lambda tid: fetched.append(tid) or None
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), te.MetricsHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = (
            f"http://127.0.0.1:{server.server_address[1]}/test?trace_id={SHORT}"
            "&test_nodeid=tests/models/gpt_oss/test_modeling_gpt_oss.py::GptOssModelTest::test_flash_attn_2_fp32_ln"
        )
        with urlopen(url) as reply:
            assert reply.status == 200
    finally:
        server.shutdown()
        server.server_close()
    assert fetched == [FULL]
