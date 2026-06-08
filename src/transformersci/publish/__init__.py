"""Publish CI telemetry from the EC2 Tempo stack to a public HF bucket.

This package shapes the raw traces the observability stack already collects into
an explorable, daily-partitioned dataset (Parquet tables + the raw trace JSON)
and syncs it to the ``transformers-ci-telemetry`` HF bucket on a schedule, so
anyone can build apps on top of the CI test data without access to our stack.

Source of truth is Tempo (raw per-test spans), not Prometheus — we want raw
rows, not pre-aggregated series. The Tempo client functions are reused from
:mod:`transformersci.otel.trace_exporter`.
"""
