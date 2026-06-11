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
"""Publish CI telemetry from the EC2 Tempo stack to a public HF bucket.

This package shapes the raw traces the observability stack already collects into
an explorable, daily-partitioned dataset (Parquet tables + the raw trace JSON)
and syncs it to the ``transformers-ci-telemetry`` HF bucket on a schedule, so
anyone can build apps on top of the CI test data without access to our stack.

Source of truth is Tempo (raw per-test spans), not Prometheus — we want raw
rows, not pre-aggregated series. The Tempo client functions are reused from
:mod:`transformersci.otel.trace_exporter`.
"""
