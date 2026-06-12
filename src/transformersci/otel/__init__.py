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
"""OpenTelemetry tooling for the transformers CI observability stack.

This package holds the producer- and consumer-side pieces that turn pytest runs
into dashboards:

- :mod:`cli` — the ``transformersci-otel`` wrapper that runs pytest with OTLP
  trace export configured (endpoints, CI resource attributes, trace context).
- :mod:`resource_plugin` — a pytest plugin that samples per-test resource usage
  and mirrors spans to a staging backend.
- :mod:`instrument` — a high-level helper that lets non-pytest tooling (e.g.
  ``utils/checkers.py``) emit one trace per run / one span per step from the
  same ``OTEL_*`` env, landing on the same dashboards as the pytest spans.
- :mod:`trace_exporter` — a long-running HTTP service that reads traces back out
  of Tempo and renders them as Prometheus metrics for Grafana.
- :mod:`debug_exporter` — an opt-in diagnostic that logs every OTLP span export.

See ``docs/design.md`` for a high-level overview of how these fit together.
"""

from __future__ import annotations

from .cli import main

__all__ = ["main"]
