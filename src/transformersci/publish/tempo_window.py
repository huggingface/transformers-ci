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

# The live exporter caps its scrape at 200 traces over 1h. A 48h publish window
# spans many more runs, so the publisher gets its own, larger ceiling.
DEFAULT_PUBLISH_WINDOW = "48h"
DEFAULT_PUBLISH_LIMIT = 5000


def publish_window_seconds() -> int:
    return parse_lookback_seconds(
        os.getenv("PUBLISH_WINDOW", DEFAULT_PUBLISH_WINDOW),
        default=DEFAULT_PUBLISH_WINDOW,
    )


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


def fetch_window(
    *,
    base_url: str | None = None,
    service_names: list[str] | None = None,
    window_seconds: int | None = None,
    limit: int | None = None,
    now: int | None = None,
) -> list[dict]:
    """Return all settled traces seen in the lookback window, Jaeger-shaped.

    Searches every configured service name and merges the results (deduped by
    trace_id). Each element is the dict produced by ``tempo_trace_to_jaeger``
    (via ``get_trace``); pass it straight to ``extract_trace_rows`` /
    ``build_test_rows``. Traces that fail to fetch are skipped, matching the
    exporter's best-effort behaviour.
    """
    base_url = base_url or tempo_base_url()
    service_names = service_names or publish_service_names()
    window_seconds = window_seconds or publish_window_seconds()
    limit = limit or publish_limit()

    end = int(now if now is not None else time.time())
    start = end - window_seconds

    seen: set[str] = set()
    trace_ids: list[str] = []
    for service_name in service_names:
        for trace_id in search_trace_ids(base_url, service_name, start, end, limit):
            if trace_id not in seen:
                seen.add(trace_id)
                trace_ids.append(trace_id)

    traces: list[dict] = []
    for trace_id in trace_ids:
        trace = get_trace(trace_id, base_url)
        if trace is not None:
            traces.append(trace)
    return traces
