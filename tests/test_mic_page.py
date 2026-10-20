import json
import threading
from collections import OrderedDict
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from transformersci.otel import mic_page, trace_exporter

NODE = "tests/contracts::mic-demo::mic-demo-slim[bfloat16]"


def vector(labels, value=1):
    return {"metric": labels, "value": [1000, str(value)]}


def failure(run="123:1", trace="a" * 32, node=NODE):
    return vector(
        {
            "run_id": run,
            "trace_id": trace,
            "test_nodeid": node,
            "pr": "42",
            "exception_type": "MICAssertionError",
        }
    )


def queries(failures):
    def query(expression):
        if "pytest_run_start_time_seconds" in expression:
            return [
                vector({"run_id": "123:1"}, 1000),
                vector({"run_id": "124:1"}, 2000),
            ]
        assert 'test_job="fast-integration"' in expression
        return failures

    return query


def detail(**extra):
    return {
        "test_job": "fast-integration",
        "trace_id": "a" * 32,
        "test_nodeid": NODE,
        **extra,
    }


def test_rows_join_case_metadata_and_link_pinned_baseline():
    calls = []
    row = detail(
        **{
            "mic.target": "mic-demo-slim",
            "mic.contract": "huggingface-mic/demo-mic",
            "mic.variant": "bfloat16",
            "mic.platform": "cpu-x86_64-bf16",
            "mic.status": "fail",
            "mic.failure": "logits: max delta 0.01 over tolerance",
            "mic.selected_by": "models/demo",
            "mic.baseline_repo": "huggingface-mic/demo-mic",
            "mic.baseline_revision": "b" * 40,
            "mic.baseline_path": "integrations/transformers/expectations/cpu-x86_64-bf16/demo.json",
        }
    )

    def load(run):
        calls.append(run)
        return [row]

    data = mic_page.build_rows(queries([failure(), failure()]), load)
    assert calls == ["123:1"]
    assert data["total"] == 1
    case = data["rows"][0]
    assert case["failure"].startswith("logits:")
    assert case["platform"] == "cpu-x86_64-bf16"
    assert case["selected_by"] == "models/demo"
    assert f"/blob/{'b' * 40}/integrations/" in case["baseline_url"]
    assert case["run_url"].endswith("/actions/runs/123")
    assert "trace_id=" in case["trace_url"]


def test_older_rows_work_without_mic_metadata_and_newest_runs_sort_first():
    data = mic_page.build_rows(queries([failure(), failure("124:1")]), lambda run: None)
    assert [r["run_id"] for r in data["rows"]] == ["124:1", "123:1"]
    assert data["rows"][0]["target"] == "mic-demo-slim"
    assert data["rows"][0]["variant"] == "bfloat16"
    assert data["rows"][0]["baseline_url"] == ""


@pytest.mark.parametrize("status", ["pass", "xfail", "unsupported"])
def test_expected_or_unsupported_cases_are_excluded(status):
    data = mic_page.build_rows(
        queries([failure()]), lambda run: [detail(**{"mic.status": status})]
    )
    assert data["rows"] == []


def test_malformed_ids_and_other_tests_are_excluded():
    data = mic_page.build_rows(
        queries([failure("../etc/passwd"), failure(node="tests/models/demo.py::test")]),
        lambda run: pytest.fail("unexpected load"),
    )
    assert not data["rows"]
    assert mic_page.parse_days("7] malicious") == 7


def test_trace_extraction_and_persistence_keep_mic_details_only(tmp_path):
    def tags(values):
        return [{"key": key, "value": value} for key, value in values.items()]

    trace = {
        "traceID": "a" * 32,
        "processes": {
            "p": {
                "serviceName": "pytest-observability",
                "tags": tags(
                    {
                        "transformers.test.run.id": "123:1",
                        "transformers.test.job": "fast-integration",
                        "vcs.repository.name": "huggingface/transformers",
                        "vcs.ref.head.revision": "c" * 40,
                    }
                ),
            }
        },
        "spans": [
            {
                "operationName": NODE,
                "processID": "p",
                "startTime": 1000,
                "duration": 500,
                "tags": tags(
                    {
                        "pytest.nodeid": NODE,
                        "pytest.span_type": "test",
                        "otel.status_code": "ERROR",
                        "mic.target": "mic-demo-slim",
                        "mic.status": "fail",
                        "mic.platform": "cpu-x86_64-bf16",
                    }
                ),
                "logs": [
                    {
                        "fields": tags(
                            {
                                "event": "exception",
                                "exception.type": "MICAssertionError",
                                "exception.stacktrace": "logits: over tolerance",
                            }
                        )
                    }
                ],
            }
        ],
    }
    _, rows = trace_exporter.extract_trace_rows(trace)
    trace_exporter.persist_run_rows("123:1", rows, directory=str(tmp_path))
    saved = trace_exporter.load_run_rows("123:1", directory=str(tmp_path))[0]
    mic_saved = trace_exporter.load_run_rows("123:1", directory=str(tmp_path / "mic"))
    assert mic_saved == [saved]
    assert saved["mic.failure"] == "logits: over tolerance"
    assert saved["mic.platform"] == "cpu-x86_64-bf16"
    assert saved["mic.commit_sha"] == "c" * 40
    trace["processes"]["p"]["tags"] = tags({"transformers.test.job": "tests_torch"})
    _, rows = trace_exporter.extract_trace_rows(trace)
    assert not any(key.startswith("mic.") for key in rows[0])


def test_http_page_data_caching_and_unavailable_state(monkeypatch):
    calls = []

    def query(expression):
        calls.append(expression)
        return []

    monkeypatch.setattr(trace_exporter, "_prometheus_vector", query)
    monkeypatch.setattr(trace_exporter, "_public_response_cache", OrderedDict())
    server = ThreadingHTTPServer(("127.0.0.1", 0), trace_exporter.MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        assert b"MIC failures" in urlopen(base + "/mic").read()
        assert (
            urlopen(base + "/mic/app.js")
            .headers["Content-Type"]
            .startswith("text/javascript")
        )
        assert json.load(urlopen(base + "/mic/data?days=1"))["rows"] == []
        urlopen(base + "/mic/data?days=1").read()
        assert len(calls) == 2

        def broken(expression):
            raise RuntimeError("private internal URL")

        monkeypatch.setattr(trace_exporter, "_prometheus_vector", broken)
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/mic/data?days=30")
        assert error.value.code == 502
        assert b"private internal URL" not in error.value.read()
    finally:
        server.shutdown()
        server.server_close()


def test_mic_cache_write_failure_does_not_drop_main_run_rows(tmp_path, monkeypatch):
    original = trace_exporter.os.makedirs

    def makedirs(path, **kwargs):
        if str(path).endswith("/mic"):
            raise PermissionError("MIC cache is not writable")
        return original(path, **kwargs)

    monkeypatch.setattr(trace_exporter.os, "makedirs", makedirs)
    trace_exporter.persist_run_rows("123:1", [detail()], directory=str(tmp_path))
    assert len(trace_exporter.load_run_rows("123:1", directory=str(tmp_path))) == 1
