from __future__ import annotations

import os

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
    assert (
        keep(f"{nid}::setup", {"pytest.span_type": "test", "pytest.nodeid": nid})
        is False
    )
    assert (
        keep(f"{nid}::call", {"pytest.span_type": "test", "pytest.nodeid": nid})
        is False
    )
    assert (
        keep(f"{nid}::teardown", {"pytest.span_type": "test", "pytest.nodeid": nid})
        is False
    )
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


# --- end-to-end: pruning must hold in a REAL pytest run, incl. xdist workers ---
#
# The tests above wire spans through one hand-installed tracer, so they can't
# catch a regression where the *real* plugin routes some spans around the wrap
# (the prod symptom that started this: ~2.76 spans/test instead of ~1, from an
# unpinned pytest-opentelemetry — see the [otel] pin in pyproject.toml and
# docs/plan-failure-visibility-regression-2026-07-15.md). These run an actual
# pytest session with the installed transformersci_otel plugin and assert only
# run + per-test protocol spans are exported — first single-process, then under
# xdist where each worker is a fresh process that must re-install the wrap.

# A conftest that attaches a file-writing exporter to the live tracer provider
# AFTER pytest-opentelemetry configured it (trylast), recording exactly the spans
# that survive the wrap — in the controller and in every xdist worker subprocess.
_PROBE_CONFTEST = """
import os
import pytest


class _FileExporter:
    def __init__(self, path):
        self._path = path

    def export(self, spans):
        from opentelemetry.sdk.trace.export import SpanExportResult

        with open(self._path, "a") as fh:
            for s in spans:
                fh.write(f"{s.attributes.get('pytest.span_type')}\\t{s.name}\\n")
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=30000):
        return True


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    from opentelemetry import trace
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    provider = trace.get_tracer_provider()
    if not hasattr(provider, "add_span_processor"):
        return
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    out = os.path.join(os.environ["PRUNE_PROBE_DIR"], f"spans-{worker}.tsv")
    provider.add_span_processor(SimpleSpanProcessor(_FileExporter(out)))
"""

# 6 passing tests (each pulling a fixture, so setup/call/teardown + fixture spans
# all fire) + 1 failing test — every span type pytest-opentelemetry emits.
_PROBE_TESTS = """
import pytest


@pytest.fixture
def client():
    yield "c"


@pytest.mark.parametrize("i", range(6))
def test_ok(i, client):
    assert client == "c"


def test_fail(client):
    assert client == "WRONG"
"""

_N_TESTS = 7  # 6 parametrized + 1 failing


def _run_and_collect_spans(tmp_path, extra_args):
    """Run a real, isolated pytest session with the installed plugin and return
    the (span_type, name) rows it actually exported. Uses a subprocess (not the
    pytester fixture) so xdist workers are genuine processes and the transformersci
    plugin loads via its installed pytest11 entry point exactly as in CI (we do
    NOT pass -p for it — that would double-register it and abort the run)."""
    import subprocess
    import sys

    pytest.importorskip("pytest_opentelemetry")
    (tmp_path / "conftest.py").write_text(_PROBE_CONFTEST)
    (tmp_path / "test_probe.py").write_text(_PROBE_TESTS)
    env = dict(os.environ, PRUNE_PROBE_DIR=str(tmp_path))
    env.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)  # entry-point load is the point
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            str(tmp_path / "test_probe.py"),
            *extra_args,
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        # The suite intentionally contains a failing test, so a non-zero exit is
        # expected; we assert on the exported spans, not the return code.
        check=False,
    )
    rows = []
    for f in sorted(tmp_path.glob("spans-*.tsv")):
        for line in f.read_text().splitlines():
            if line:
                span_type, _, name = line.partition("\t")
                rows.append((span_type, name))
    return rows


def _assert_only_run_and_protocol(rows):
    assert rows, "no spans were exported — the probe conftest did not attach"
    # No phase or fixture spans leaked through the wrap.
    for span_type, name in rows:
        assert span_type in ("run", "test"), (span_type, name)
        assert "::setup" not in name, name
        assert "::call" not in name, name
        assert "::teardown" not in name, name
    test_spans = [n for t, n in rows if t == "test"]
    assert len(test_spans) == _N_TESTS, rows
    assert any(t == "run" for t, _ in rows), rows


def test_real_run_prunes_to_run_and_protocol_only(tmp_path):
    rows = _run_and_collect_spans(tmp_path, [])
    _assert_only_run_and_protocol(rows)
    # ~1 span/test (protocol) + 1 run span — the whole point of pruning.
    assert len(rows) == _N_TESTS + 1, rows


def test_real_run_prunes_under_xdist_workers(tmp_path):
    pytest.importorskip("xdist")
    rows = _run_and_collect_spans(tmp_path, ["-n", "2"])
    # The wrap must survive into each worker process: still no phase/fixture spans,
    # still exactly one protocol span per test (run spans: 1 controller + workers).
    _assert_only_run_and_protocol(rows)
    run_spans = [n for t, n in rows if t == "run"]
    assert len(run_spans) >= 2, rows  # controller + at least one worker session span
