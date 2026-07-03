from __future__ import annotations

import pytest

from transformersci.otel import span_pruning


def test_should_keep_predicate() -> None:
    keep = span_pruning._should_keep
    nid = "tests/models/foo/test_foo.py::TestFoo::test_bar"
    # protocol span: span_type=test AND name == nodeid
    assert keep(nid, {"pytest.span_type": "test", "pytest.nodeid": nid}) is True
    # session/run span
    assert keep("test run", {"pytest.span_type": "run"}) is True
    # phase spans: span_type=test but name != nodeid
    assert keep(f"{nid}::setup", {"pytest.span_type": "test", "pytest.nodeid": nid}) is False
    assert keep(f"{nid}::call", {"pytest.span_type": "test", "pytest.nodeid": nid}) is False
    assert keep(f"{nid}::teardown", {"pytest.span_type": "test", "pytest.nodeid": nid}) is False
    # fixture spans
    assert keep("client setup", {"pytest.span_type": "fixture"}) is False
    # unattributed "fixture teardown" start_span
    assert keep("fixture teardown", None) is False


@pytest.fixture
def in_memory_tracer():
    """A real SDK tracer wired to an in-memory exporter, installed as the tracer
    pytest-opentelemetry creates spans through, with span_pruning applied."""
    pytest.importorskip("pytest_opentelemetry")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from pytest_opentelemetry import instrumentation

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    saved_tracer = instrumentation.tracer
    span_pruning._installed = False
    instrumentation.tracer = provider.get_tracer("test")
    assert span_pruning.install() is True
    try:
        yield instrumentation.tracer, exporter
    finally:
        instrumentation.tracer = saved_tracer
        span_pruning._installed = False


def _emit_one_test(tracer, nodeid, *, fail: bool) -> None:
    """Reproduce exactly how pytest-opentelemetry emits a test's spans."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    item_attrs = {"pytest.span_type": "test", "pytest.nodeid": nodeid}
    # pytest_runtest_protocol -> the per-test span, made current for the protocol
    with tracer.start_as_current_span(nodeid, attributes=item_attrs):
        # pytest_runtest_setup
        with tracer.start_as_current_span(f"{nodeid}::setup", attributes=item_attrs):
            # pytest_fixture_setup
            with tracer.start_as_current_span(
                name="client setup", attributes={"pytest.span_type": "fixture"}
            ):
                pass
        # pytest_runtest_call
        with tracer.start_as_current_span(f"{nodeid}::call", attributes=item_attrs):
            if fail:
                # pytest_exception_interact records on the CURRENT span
                span = trace.get_current_span()
                try:
                    raise ValueError("boom")
                except ValueError as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, "boom"))
        # pytest_runtest_teardown
        with tracer.start_as_current_span(f"{nodeid}::teardown", attributes=item_attrs):
            pass
        # pytest_runtest_logreport fires after each phase span closes: sets status
        # on the current span (the protocol span) for the call outcome.
        if fail:
            trace.get_current_span().set_status(Status(StatusCode.ERROR))


def test_only_protocol_and_run_spans_are_exported(in_memory_tracer) -> None:
    tracer, exporter = in_memory_tracer
    # session/run span (start_span, span_type=run) — kept
    tracer.start_span("test run", attributes={"pytest.span_type": "run"}).end()
    # a fixture-teardown start_span with no attributes — dropped
    tracer.start_span("fixture teardown").end()
    _emit_one_test(tracer, "tests/test_a.py::test_pass", fail=False)

    spans = exporter.get_finished_spans()
    names = sorted(s.name for s in spans)
    # Only the run span and the single per-test protocol span survive.
    assert names == ["test run", "tests/test_a.py::test_pass"], names
    for s in spans:
        assert "::setup" not in s.name and "::call" not in s.name
        assert "::teardown" not in s.name and "setup" != s.name.split()[-1]


def test_failure_status_and_exception_land_on_protocol_span(in_memory_tracer) -> None:
    from opentelemetry.trace import StatusCode

    tracer, exporter = in_memory_tracer
    _emit_one_test(tracer, "tests/test_a.py::test_fail", fail=True)

    # exactly one exported span (the protocol span), and it carries the failure
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "tests/test_a.py::test_fail"
    assert span.status.status_code == StatusCode.ERROR
    # the exception event recorded during the (dropped) ::call phase is on it
    event_names = [e.name for e in span.events]
    assert "exception" in event_names, event_names
