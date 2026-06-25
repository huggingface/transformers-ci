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
"""Make OpenTelemetry span/trace ids immune to test-code RNG seeding.

**The bug this fixes.** OpenTelemetry's default ``RandomIdGenerator`` draws span
and trace ids from the process-**global** ``random`` module
(``random.getrandbits(64|128)``). The transformers test suite calls
``transformers.set_seed()`` / ``random.seed()`` constantly for reproducibility,
and each call reseeds that same global RNG. So every span created after a
``set_seed(42)`` gets a *deterministic* id — identical across the 8 pytest-xdist
workers that share one trace_id (via ``TRACEPARENT``) AND identical across every
test that reuses the same seed. Tempo stores spans keyed by
``(trace_id, span_id)``, so those colliding spans silently overwrite each other:
no discard, no export failure, no error anywhere.

Measured in prod (one ``tests_torch`` shard, 8 workers, one trace_id):
**100,329 spans produced / 100% exported / Tempo received 100% — yet only 6,354
distinct span_ids survived** (~94% collision). That single defect accounted for
the long-standing "~⅔ of spans vanish in Tempo" mystery; nothing downstream
(collector, Tempo, exporter read path) was actually losing data.

**The fix.** Draw ids from a process-private ``random.Random`` seeded once from
``os.urandom`` — state that test code never touches. We can't inject this at
``TracerProvider`` construction (pytest-opentelemetry builds the provider), so we
monkeypatch ``RandomIdGenerator``'s two methods. Patching the *class* methods
fixes any already-constructed generator too, because the lookup happens at
span-creation time (after :func:`install` has run).
"""

from __future__ import annotations

import os
import random

_installed = False


def install() -> bool:
    """Patch ``RandomIdGenerator`` to use a seed-independent private RNG.

    Idempotent and best-effort: returns ``True`` once the patch is in place,
    ``False`` if the OTel SDK is not importable (nothing to patch). Must run
    before the first span is created (we call it from ``pytest_configure``,
    which precedes session start and any test span).
    """
    global _installed
    if _installed:
        return True
    try:
        from opentelemetry.sdk.trace import id_generator as _idgen
        from opentelemetry.trace import INVALID_SPAN_ID, INVALID_TRACE_ID
    except Exception:  # pragma: no cover - SDK not installed
        return False

    # Private RNG, seeded from the OS CSPRNG. NOT the global ``random`` module,
    # so test code calling ``random.seed()`` / ``set_seed()`` cannot make these
    # ids deterministic (and therefore cannot make them collide).
    rng = random.Random(os.urandom(32))

    def generate_span_id(self) -> int:  # noqa: ANN001 - matches OTel signature
        span_id = rng.getrandbits(64)
        while span_id == INVALID_SPAN_ID:
            span_id = rng.getrandbits(64)
        return span_id

    def generate_trace_id(self) -> int:  # noqa: ANN001 - matches OTel signature
        trace_id = rng.getrandbits(128)
        while trace_id == INVALID_TRACE_ID:
            trace_id = rng.getrandbits(128)
        return trace_id

    _idgen.RandomIdGenerator.generate_span_id = generate_span_id
    _idgen.RandomIdGenerator.generate_trace_id = generate_trace_id
    _installed = True
    return True
