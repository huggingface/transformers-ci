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
"""Windowed Tempo fetch for the public CI-data publisher.

Reuses the live exporter's Tempo client (``search_trace_ids`` + ``get_trace``,
which already adapts Tempo's OTLP-JSON into the Jaeger-shaped dict the rest of
the code understands) but over a much wider lookback than the 1h scrape window,
so each publish cycle captures whole runs and late-arriving traces.

A cycle re-derives entire day partitions from this window and overwrites them;
settled traces are immutable, so a day stabilises once it ages past the window
and is never rewritten again (see :mod:`transformersci.publish.tables`).
"""

from __future__ import annotations

import os
import time

from ..otel.trace_exporter import (
    DEFAULT_SERVICE_NAME,
    get_trace,
    parse_lookback_seconds,
    search_trace_ids,
    tempo_base_url,
)

# The live exporter caps its scrape at 200 traces over 1h. The publisher spans
# many more runs, so it gets its own, larger ceiling. The window was 48h but
# halved to 24h to cut the cost of the /api/search against a memory-pressured
# single-node Tempo (the search was timing out and aborting the cycle); 24h
# still comfortably covers whole runs and late-arriving traces.
DEFAULT_PUBLISH_WINDOW = "24h"
DEFAULT_PUBLISH_LIMIT = 5000
# The wide search is the heaviest single Tempo call and the first thing to fail
# when Tempo is slow. Retry it a few times with backoff so a transient blip
# doesn't lose the whole hourly cycle (a longer HTTP timeout is set via
# PYTEST_TRACE_EXPORTER_HTTP_TIMEOUT in the publisher's environment).
DEFAULT_SEARCH_RETRIES = 3
DEFAULT_SEARCH_RETRY_BACKOFF = 5.0


def publish_window_seconds() -> int:
    return parse_lookback_seconds(
        os.getenv("PUBLISH_WINDOW", DEFAULT_PUBLISH_WINDOW),
        default=DEFAULT_PUBLISH_WINDOW,
    )


def _search_retries() -> int:
    try:
        return max(
            1, int(os.getenv("PUBLISH_SEARCH_RETRIES", "") or DEFAULT_SEARCH_RETRIES)
        )
    except ValueError:
        return DEFAULT_SEARCH_RETRIES


def _search_with_retry(
    base_url: str, service_name: str, start: int, end: int, limit: int
) -> list[str]:
    """search_trace_ids with bounded retries + backoff.

    A search timeout/connection error used to propagate straight out of the
    cycle (unlike per-trace get_trace failures, which are skipped), so one slow
    Tempo moment lost the entire hourly publish. Retrying absorbs transient
    blips; if every attempt fails the error is re-raised so the cycle still
    reports failure rather than silently publishing a partial window.
    """
    attempts = _search_retries()
    for attempt in range(1, attempts + 1):
        try:
            return search_trace_ids(base_url, service_name, start, end, limit)
        except Exception as error:  # noqa: BLE001 — Tempo can fail many ways
            if attempt >= attempts:
                raise
            delay = DEFAULT_SEARCH_RETRY_BACKOFF * attempt
            print(
                f"[ci-data-publisher] search for {service_name!r} failed "
                f"(attempt {attempt}/{attempts}): {error}; retrying in {delay:.0f}s",
                flush=True,
            )
            time.sleep(delay)
    return []


def publish_limit() -> int:
    raw = os.getenv("PUBLISH_LIMIT", "")
    try:
        return int(raw) if raw else DEFAULT_PUBLISH_LIMIT
    except ValueError:
        return DEFAULT_PUBLISH_LIMIT


def publish_service_names() -> list[str]:
    """Service names to publish, in priority order.

    ``PUBLISH_SERVICE_NAMES`` (comma-separated) lets one publisher cover several
    emitters — e.g. real CI (``pytest-observability``) plus the demo/sample-run
    name (``pytest-observability-demo``). Falls back to the single exporter
    service name when unset.
    """
    raw = os.getenv("PUBLISH_SERVICE_NAMES", "")
    if raw.strip():
        return [s.strip() for s in raw.split(",") if s.strip()]
    return [os.getenv("PYTEST_TRACE_EXPORTER_SERVICE_NAME", DEFAULT_SERVICE_NAME)]


def window_trace_ids(
    *,
    base_url: str | None = None,
    service_names: list[str] | None = None,
    window_seconds: int | None = None,
    limit: int | None = None,
    now: int | None = None,
) -> list[str]:
    """Distinct trace_ids seen in the lookback window across all service names."""
    base_url = base_url or tempo_base_url()
    service_names = service_names or publish_service_names()
    window_seconds = window_seconds or publish_window_seconds()
    limit = limit or publish_limit()

    end = int(now if now is not None else time.time())
    start = end - window_seconds

    seen: set[str] = set()
    trace_ids: list[str] = []
    for service_name in service_names:
        for trace_id in _search_with_retry(base_url, service_name, start, end, limit):
            if trace_id not in seen:
                seen.add(trace_id)
                trace_ids.append(trace_id)
    return trace_ids


def iter_window_traces(
    *,
    base_url: str | None = None,
    service_names: list[str] | None = None,
    window_seconds: int | None = None,
    limit: int | None = None,
    now: int | None = None,
):
    """Yield settled traces in the window one at a time, Jaeger-shaped.

    Streaming (vs. returning a list) keeps the publisher's memory flat: a single
    CI trace can be many MB, so materialising a whole window of them at once is
    what OOM-killed the sidecar. Each yielded item is the dict produced by
    ``tempo_trace_to_jaeger`` (via ``get_trace``); traces that fail to fetch are
    skipped, matching the exporter's best-effort behaviour.
    """
    base_url = base_url or tempo_base_url()
    for trace_id in window_trace_ids(
        base_url=base_url,
        service_names=service_names,
        window_seconds=window_seconds,
        limit=limit,
        now=now,
    ):
        trace = get_trace(trace_id, base_url)
        if trace is not None:
            yield trace


def fetch_window(
    *,
    base_url: str | None = None,
    service_names: list[str] | None = None,
    window_seconds: int | None = None,
    limit: int | None = None,
    now: int | None = None,
) -> list[dict]:
    """Eager list form of :func:`iter_window_traces` (kept for callers/tests)."""
    return list(
        iter_window_traces(
            base_url=base_url,
            service_names=service_names,
            window_seconds=window_seconds,
            limit=limit,
            now=now,
        )
    )
