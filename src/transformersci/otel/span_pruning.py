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
"""Emit one span per test — drop pytest-opentelemetry's phase and fixture spans.

**Why.** pytest-opentelemetry emits, for every test, a *protocol* span
(``name == nodeid``) PLUS ``::setup`` / ``::call`` / ``::teardown`` phase spans
and a span per fixture. The trace exporter reads **only** the protocol span
(``extract_trace_rows`` requires ``pytest.span_type == "test"`` AND
``operationName == nodeid``; ``extract_failure_details`` filters the same way).
The rest — ~83% of a shard trace's spans — exist solely for Tempo's waterfall
view. At transformers scale a single ``tests_torch`` shard trace reached
~50 MB / ~100k spans, over Tempo's read-path message limit, so
``GET /api/traces/{id}`` 500'd and the exporter silently dropped the whole job
from both the run store and the roll-ups (it vanished from the Jobs table). See
docs/plan-large-trace-read-limit-2026-07-03.md.

Dropping the unused spans shrinks a shard trace ~6x (≈8 MB) with **no** change to
the trace topology, the exporter, or Tempo cardinality. Correctness is preserved:
pass/fail status is set by ``pytest_runtest_logreport`` on the *current* span,
which — after each (now-absent) phase span would have closed — is the protocol
span; and with the ``::call`` span gone, ``pytest_exception_interact``'s
``trace.get_current_span()`` is the protocol span, so exception tracebacks record
there, exactly where ``/failure`` reads them. Only the per-phase/fixture timing
breakdown in the Tempo waterfall is lost.

**How.** We can't drop these by patching the plugin's hook methods: pluggy binds
hookimpls at plugin-registration time, before our ``pytest_configure`` runs. So
we wrap the module-global ``tracer`` the plugin creates every span through and
return a no-op span for the ones we don't keep, decided per call from the span
name + attributes. Mirrors :mod:`transformersci.otel.id_generator` (patch at
span-creation time, before the first span).
"""

from __future__ import annotations

from contextlib import contextmanager

_installed = False


def _should_keep(name: object, attributes: object) -> bool:
    """Keep only the per-test protocol span and the session/run span.

    Drops the ``::setup``/``::call``/``::teardown`` phase spans (``span_type ==
    "test"`` but ``name != nodeid``), the fixture spans (``span_type ==
    "fixture"``), and the unattributed "fixture teardown" span.
    """
    attrs = attributes if isinstance(attributes, dict) else {}
    span_type = attrs.get("pytest.span_type")
    if span_type == "run":
        return True
    if span_type == "test" and name == attrs.get("pytest.nodeid"):
        return True
    return False


def install() -> bool:
    """Wrap pytest-opentelemetry's tracer to suppress phase/fixture spans.

    Idempotent and best-effort: ``True`` once the wrap is in place, ``False`` if
    pytest-opentelemetry / the OTel SDK is not importable. Must run before the
    first span (we call it from ``pytest_configure``).
    """
    global _installed
    if _installed:
        return True
    try:
        from opentelemetry.trace import INVALID_SPAN
        from pytest_opentelemetry import instrumentation
    except Exception:  # pragma: no cover - plugin/SDK not installed
        return False

    # Fail SAFE: if the plugin's internals aren't the shape we expect (e.g. a
    # future version drops the module-global tracer), decline to patch rather than
    # risk mis-filtering. Traces stay large-but-complete and the Tempo/exporter
    # read-path headroom still covers them.
    tracer = getattr(instrumentation, "tracer", None)
    if tracer is None or not (
        callable(getattr(tracer, "start_as_current_span", None))
        and callable(getattr(tracer, "start_span", None))
    ):
        return False

    real_start_as_current_span = tracer.start_as_current_span
    real_start_span = tracer.start_span

    @contextmanager
    def _noop_current_span():
        # Yield a non-recording span WITHOUT making it current, so the enclosing
        # protocol span stays active and status/exceptions attach to it.
        yield INVALID_SPAN

    def start_as_current_span(*args, **kwargs):  # noqa: ANN002,ANN003
        name = args[0] if args else kwargs.get("name")
        if _should_keep(name, kwargs.get("attributes")):
            return real_start_as_current_span(*args, **kwargs)
        return _noop_current_span()

    def start_span(*args, **kwargs):  # noqa: ANN002,ANN003
        name = args[0] if args else kwargs.get("name")
        if _should_keep(name, kwargs.get("attributes")):
            return real_start_span(*args, **kwargs)
        return INVALID_SPAN

    tracer.start_as_current_span = start_as_current_span
    tracer.start_span = start_span
    _installed = True
    return True
