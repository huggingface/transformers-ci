from __future__ import annotations

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from transformersci.otel import instrument


def test_is_configured() -> None:
    assert instrument.is_configured({}) is False
    assert instrument.is_configured({"OTEL_EXPORTER_OTLP_ENDPOINT": "x"}) is True
    assert instrument.is_configured({"OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "x"}) is True


def test_run_is_noop_when_unconfigured() -> None:
    run = instrument.run("check_code_quality", env={})
    assert isinstance(run, instrument._NoRun)
    # The no-op must support the exact same call shape as the real Run/Step.
    with run as r:
        with r.step("utils/checkers.py::ruff_check", attributes={"a": 1}) as step:
            step.set_attribute("b", 2)
            step.set_exit_code(1)


@pytest.fixture
def captured_spans(monkeypatch):
    mem = InMemorySpanExporter()

    class _ListExporter:
        def export(self, spans):
            return mem.export(spans)

        def force_flush(self, timeout_millis: int = 30000):
            return True

        def shutdown(self):
            mem.shutdown()

    monkeypatch.setattr(
        instrument, "build_primary_exporter", lambda env: _ListExporter()
    )
    monkeypatch.setattr(instrument, "build_staging_exporter", lambda env: None)
    return mem


def _emit(env):
    with instrument.run(
        "check_code_quality", attributes={"transformers.check.mode": "check"}, env=env
    ) as run:
        with run.step(
            "utils/checkers.py::ruff_check",
            attributes={"transformers.check.name": "ruff_check"},
        ) as step:
            step.set_exit_code(0)
        with run.step(
            "utils/checkers.py::copies",
            attributes={"transformers.check.name": "copies"},
        ) as step:
            step.set_exit_code(1)


def test_steps_are_tagged_as_test_spans(captured_spans) -> None:
    _emit({"OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4318"})
    spans = {s.name: s for s in captured_spans.get_finished_spans()}

    # Each checker span carries the pytest test-span contract the exporter requires.
    for nodeid in ("utils/checkers.py::ruff_check", "utils/checkers.py::copies"):
        span = spans[nodeid]
        assert span.attributes["pytest.span_type"] == "test"
        assert span.attributes["pytest.nodeid"] == nodeid

    assert spans["utils/checkers.py::ruff_check"].status.status_code.name == "OK"
    assert spans["utils/checkers.py::copies"].status.status_code.name == "ERROR"
    assert (
        spans["utils/checkers.py::copies"].attributes["transformers.check.exit_code"]
        == 1
    )


def test_root_span_is_not_a_test_span(captured_spans) -> None:
    _emit({"OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4318"})
    spans = {s.name: s for s in captured_spans.get_finished_spans()}
    root = spans["check_code_quality"]
    # The root is a container only — no pytest tags, so the exporter skips it.
    assert "pytest.span_type" not in root.attributes


def test_steps_join_traceparent(captured_spans) -> None:
    traceparent = "00-1234567890abcdef1234567890abcdef-fedcba0987654321-01"
    _emit(
        {
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4318",
            "TRACEPARENT": traceparent,
        }
    )
    trace_ids = {
        format(s.context.trace_id, "032x") for s in captured_spans.get_finished_spans()
    }
    assert trace_ids == {"1234567890abcdef1234567890abcdef"}
