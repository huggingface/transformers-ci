"""A test that fails every rerun must not reach Tempo as a pass."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("pytest_opentelemetry")
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.trace import Status, StatusCode  # noqa: E402
from pytest_opentelemetry.instrumentation import (  # noqa: E402
    PerTestOpenTelemetryPlugin,
)

from transformersci.otel import rerun_status  # noqa: E402

tracer = TracerProvider().get_tracer("test")


def attempts(outcomes: list[str], fixed: bool) -> StatusCode:
    """Replay pytest's per-attempt hook order on one test span.

    For each attempt: ``pytest_runtest_logreport`` (pytest-opentelemetry's,
    optionally wrapped by ours), then ``pytest_exception_interact`` when the
    attempt raised, which pytest calls for ``rerun`` attempts too.
    """
    otel = PerTestOpenTelemetryPlugin()
    ours = rerun_status.RerunStatusPlugin()
    with tracer.start_as_current_span("tests/test_x.py::test_x") as span:
        for outcome in outcomes:
            report = SimpleNamespace(when="call", outcome=outcome)
            if fixed:
                wrapper = ours.pytest_runtest_logreport(report)
                next(wrapper)
            otel.pytest_runtest_logreport(report)
            if fixed:
                with pytest.raises(StopIteration):
                    next(wrapper)
            if outcome != "passed":
                span.set_status(Status(StatusCode.ERROR, "HfHubHTTPError"))
    return span.status.status_code


def test_without_the_fix_a_rerun_locks_the_span_at_ok():
    # The bug as seen in prod: six failing attempts, span says OK.
    assert attempts(["rerun"] * 5 + ["failed"], fixed=False) == StatusCode.OK


def test_a_test_that_fails_every_rerun_is_error():
    assert attempts(["rerun"] * 5 + ["failed"], fixed=True) == StatusCode.ERROR


def test_a_test_that_passes_on_a_rerun_is_ok():
    assert attempts(["rerun", "rerun", "passed"], fixed=True) == StatusCode.OK


def test_runs_without_reruns_are_unchanged():
    assert attempts(["failed"], fixed=True) == StatusCode.ERROR
    assert attempts(["passed"], fixed=True) == StatusCode.OK


def test_install_registers_once():
    registered = []
    manager = SimpleNamespace(
        has_plugin=lambda name: bool(registered),
        register=lambda plugin, name: registered.append(name),
    )
    config = SimpleNamespace(pluginmanager=manager)
    rerun_status.install(config)
    rerun_status.install(config)
    assert registered == [rerun_status.PLUGIN_NAME]
