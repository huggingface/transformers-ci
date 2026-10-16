"""Keep a test that fails every rerun from reaching Tempo as a pass.

pytest-rerunfailures reports each attempt it will retry with the outcome
``rerun``, and pytest-opentelemetry's ``pytest_runtest_logreport`` turns any
call report that is not ``failed`` into ``set_status(OK)``. The OTel SDK treats
OK as final and ignores every later ``set_status``, so the first retried attempt
locks the test span at OK: the exception events of every attempt are recorded,
the final failure's ERROR is dropped, and the dashboards (which count ERROR
spans) show the test as passing. Seen on transformers#49251 (run 38870824263):
two muse_spark tests failed 6 of 6 attempts and both spans said ``ok``.

The fix wraps that hook: after a ``rerun`` report the span goes back to UNSET,
so the attempt decides nothing. ``pytest_exception_interact`` then marks it
ERROR as usual, a later passing attempt can still turn it OK (OK may replace
ERROR, just not the other way round), and a final failure stays ERROR.
"""

from __future__ import annotations

from typing import Any

import pytest

PLUGIN_NAME = "transformersci-rerun-status"


def reset_status(span: Any) -> bool:
    """Put a recording SDK span back to UNSET. Returns whether it did."""
    try:
        from opentelemetry.trace import Status, StatusCode
    except ImportError:  # pragma: no cover
        return False
    if not span.is_recording() or not hasattr(span, "_status"):
        return False
    lock = getattr(span, "_lock", None)
    if lock is None:
        span._status = Status(StatusCode.UNSET)
    else:
        with lock:
            span._status = Status(StatusCode.UNSET)
    return True


class RerunStatusPlugin:
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_logreport(self, report: pytest.TestReport) -> Any:
        yield
        if report.when != "call" or report.outcome != "rerun":
            return
        try:
            from opentelemetry import trace
        except ImportError:  # pragma: no cover
            return
        reset_status(trace.get_current_span())


def install(config: pytest.Config) -> None:
    if not config.pluginmanager.has_plugin(PLUGIN_NAME):
        config.pluginmanager.register(RerunStatusPlugin(), PLUGIN_NAME)
