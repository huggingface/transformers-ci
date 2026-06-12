# Copyright 2026 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""High-level CI-step tracing for non-pytest tooling (e.g. ``utils/checkers.py``).

The pytest path gets its trace export from ``pytest-opentelemetry`` (which builds
the SDK ``TracerProvider`` from the ``OTEL_*`` env) plus :mod:`resource_plugin`
(the staging mirror). A plain Python QA runner has no such plugin, so this module
provides the same thing as a library any script can call:

    from transformersci.otel import instrument

    with instrument.run("check_code_quality") as run:
        for name in names:
            nodeid = f"utils/checkers.py::{name}"
            with run.step(nodeid) as step:
                rc = do_work(name)
                step.set_exit_code(rc)

Design notes:

- **Configured entirely by env.** :func:`run` reads the same ``OTEL_*`` /
  ``TRANSFORMERS_TEST_OTEL_*`` env that ``configure-ci-otel`` exports. The job
  name, PR, run id, repo, and ``service.name`` all ride in via
  ``OTEL_RESOURCE_ATTRIBUTES`` / ``OTEL_SERVICE_NAME``, which the SDK's
  ``Resource.create()`` merges automatically — so this module never needs to
  know the job and the spans land on the same dashboards as the pytest spans.
- **No-op unless wired up.** When no OTLP endpoint is configured, or the OTel
  SDK is not installed, :func:`run` returns a do-nothing object with the same
  shape, so the calling code path is identical whether or not tracing is on.
- **Test-pipeline span contract.** Each :meth:`Run.step` span is tagged so the
  trace exporter ingests it exactly like a pytest test span: span name equals
  the ``pytest.nodeid`` tag and ``pytest.span_type`` is ``"test"``. This module
  owns that contract; callers just pass a nodeid string and an exit code.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from contextlib import contextmanager

from ._export import build_primary_exporter, build_staging_exporter

# Tags that make a span look like a pytest test span to the trace exporter
# (see trace_exporter.extract_trace_rows: it keeps a span only when these are
# present and the span name == the nodeid).
_SPAN_TYPE_TAG = "pytest.span_type"
_NODEID_TAG = "pytest.nodeid"
_TEST_SPAN_TYPE = "test"

_INSTRUMENTATION_SCOPE = "transformersci.otel.instrument"

# Cap the captured checker output recorded on a failing span's exception event.
_MAX_OUTPUT_CHARS = 8000


def is_configured(env: Mapping[str, str] | None = None) -> bool:
    """Return True when an OTLP traces endpoint is configured in ``env``."""
    env = env if env is not None else os.environ
    return bool(
        env.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        or env.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    )


def run(name: str, attributes: Mapping[str, object] | None = None, env=None):
    """Start a CI run: build a tracer provider and open a root span.

    Returns a context manager. On a configured run it yields a :class:`Run`
    whose :meth:`Run.step` opens per-step child spans; on an unconfigured run
    (or when the SDK is missing) it yields a no-op of the same shape.
    """
    env = env if env is not None else os.environ
    if not is_configured(env):
        return _NoRun()

    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )
    except ImportError as error:  # pragma: no cover - SDK not installed
        print(
            f"OTEL INSTRUMENT SDK unavailable, not tracing {name!r}: {error!r}",
            file=sys.stderr,
            flush=True,
        )
        return _NoRun()

    # Resource.create() merges OTEL_SERVICE_NAME + OTEL_RESOURCE_ATTRIBUTES from
    # env — i.e. transformers.test.job / vcs.* / run id that configure-ci-otel
    # already exported. The job grouping is therefore inherited, not set here.
    provider = TracerProvider(resource=Resource.create())
    attached = 0
    for exporter in (build_primary_exporter(env), build_staging_exporter(env)):
        if exporter is not None:
            provider.add_span_processor(BatchSpanProcessor(exporter))
            attached += 1
    if attached == 0:  # pragma: no cover - endpoint set but no exporter built
        provider.shutdown()
        return _NoRun()

    tracer = provider.get_tracer(_INSTRUMENTATION_SCOPE)
    parent_context = None
    traceparent = env.get("TRACEPARENT")
    if traceparent:
        # Join the trace configure-ci-otel started so the root span shares its
        # trace id, mirroring how pytest-opentelemetry parents off TRACEPARENT.
        parent_context = TraceContextTextMapPropagator().extract(
            {"traceparent": traceparent}
        )
    return _Run(provider, tracer, name, attributes, parent_context)


class _Run:
    def __init__(self, provider, tracer, name, attributes, parent_context):
        self._provider = provider
        self._tracer = tracer
        self._name = name
        self._attributes = attributes
        self._parent_context = parent_context
        self._span_cm = None

    def __enter__(self) -> "_Run":
        self._span_cm = self._tracer.start_as_current_span(
            self._name, context=self._parent_context
        )
        span = self._span_cm.__enter__()
        for key, value in (self._attributes or {}).items():
            span.set_attribute(key, value)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            if self._span_cm is not None:
                self._span_cm.__exit__(exc_type, exc, tb)
        finally:
            # Flush before shutdown so the batch processor ships everything this
            # process produced; bounded by the per-exporter export timeout.
            self._provider.force_flush()
            self._provider.shutdown()
        return False

    @contextmanager
    def step(self, nodeid: str, attributes: Mapping[str, object] | None = None):
        """Open a child span for one step, tagged so the exporter ingests it."""
        with self._tracer.start_as_current_span(nodeid) as span:
            span.set_attribute(_NODEID_TAG, nodeid)
            span.set_attribute(_SPAN_TYPE_TAG, _TEST_SPAN_TYPE)
            for key, value in (attributes or {}).items():
                span.set_attribute(key, value)
            yield _Step(span)


class _Step:
    def __init__(self, span):
        self._span = span

    def set_attribute(self, key: str, value: object) -> None:
        self._span.set_attribute(key, value)

    def set_exit_code(
        self,
        returncode: int,
        *,
        command: str | None = None,
        output: str | None = None,
    ) -> None:
        """Record the step's exit code; non-zero marks the span ERROR.

        The exporter reads the synthesized ``otel.status_code`` tag to decide
        pass/fail, so a non-zero code surfaces this step as a failing "test".

        On failure, the ``command`` and captured ``output`` are recorded as an
        ``"exception"`` span event using the same ``exception.message`` /
        ``exception.stacktrace`` fields a pytest traceback uses. This lets the
        trace exporter's ``/failure`` page render the checker's output with no
        exporter-side changes (it already reads that event).
        """
        from opentelemetry.trace import Status, StatusCode

        self._span.set_attribute("transformers.check.exit_code", int(returncode))
        if returncode != 0:
            self._span.set_status(Status(StatusCode.ERROR))
            self._record_failure_event(command, output)
        else:
            self._span.set_status(Status(StatusCode.OK))

    def _record_failure_event(self, command: str | None, output: str | None) -> None:
        attributes: dict[str, str] = {"exception.type": "CheckFailed"}
        if command:
            attributes["exception.message"] = command
        if output:
            # Bound the payload so a noisy checker can't bloat the span; the
            # tail holds the actual errors (ruff summaries, diffs, tracebacks).
            attributes["exception.stacktrace"] = output[-_MAX_OUTPUT_CHARS:]
        self._span.add_event("exception", attributes)


class _NoRun:
    """Shape-compatible no-op returned when tracing is not configured."""

    def __enter__(self) -> "_NoRun":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    @contextmanager
    def step(self, nodeid: str, attributes: Mapping[str, object] | None = None):
        yield _NoStep()


class _NoStep:
    def set_attribute(self, key: str, value: object) -> None:
        pass

    def set_exit_code(
        self,
        returncode: int,
        *,
        command: str | None = None,
        output: str | None = None,
    ) -> None:
        pass
