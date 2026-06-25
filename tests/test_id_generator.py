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
"""The span/trace id generator must be immune to global RNG seeding.

This is the regression guard for the prod span-collision bug: transformers'
``set_seed()`` reseeds the global ``random`` module, and OTel's stock id
generator draws ids from it — so without the fix, every span created after the
same seed gets the SAME id and collides in Tempo.
"""

import random

import pytest

idgen_sdk = pytest.importorskip("opentelemetry.sdk.trace.id_generator")

from transformersci.otel import id_generator


def _fresh_generator():
    # New module-level _installed each test would be ideal; install() is
    # idempotent, so just ensure the patch is applied for this process.
    id_generator._installed = False
    assert id_generator.install() is True
    return idgen_sdk.RandomIdGenerator()


def test_span_ids_survive_identical_global_seed():
    """Same global seed before each id must NOT yield the same span id."""
    gen = _fresh_generator()

    random.seed(42)
    first = gen.generate_span_id()
    random.seed(42)
    second = gen.generate_span_id()

    assert first != second, "span ids became deterministic under a fixed global seed"


def test_trace_ids_survive_identical_global_seed():
    gen = _fresh_generator()

    random.seed(1234)
    first = gen.generate_trace_id()
    random.seed(1234)
    second = gen.generate_trace_id()

    assert first != second, "trace ids became deterministic under a fixed global seed"


def test_simulated_xdist_workers_do_not_collide():
    """Two 'workers' that seed identically (as xdist+set_seed do) get distinct ids.

    Mirrors the prod failure: 8 workers sharing one trace_id each ran tests that
    called set_seed(SEED), so their span streams were identical and collapsed in
    Tempo. With the fix each worker's generator has its own os.urandom state.
    """
    gen = _fresh_generator()

    worker_ids = []
    for _worker in range(8):
        random.seed(0)  # what set_seed() does at the start of a test
        worker_ids.append([gen.generate_span_id() for _ in range(50)])

    flat = [sid for w in worker_ids for sid in w]
    # No collisions across the 8 workers' 50 spans each.
    assert len(set(flat)) == len(flat)


def test_ids_are_well_formed():
    gen = _fresh_generator()
    span_id = gen.generate_span_id()
    trace_id = gen.generate_trace_id()
    assert 0 < span_id < 2**64
    assert 0 < trace_id < 2**128
