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
"""Consumer-side service: turn Tempo traces into Prometheus metrics for Grafana.

This is a long-running HTTP server. Prometheus scrapes ``/metrics``; the body is
derived entirely from traces stored in Tempo, so the exporter holds no durable
state of its own — it renders a fresh snapshot of the lookback window each cycle.
``/failure`` serves an HTML view of a single trace's failure details for
dashboard drill-down.

Pipeline (and roughly the order functions appear in the file):

1. Fetch & shape — :func:`search_trace_ids` / :func:`get_trace` / :func:`iter_traces`
   pull traces from Tempo and :func:`tempo_trace_to_jaeger` converts OTLP JSON
   into the Jaeger-shaped dicts the extractors expect. A dual-bounded (count +
   byte) LRU memoizes settled (immutable) traces so a replay can't grow RSS
   without limit.
2. GitHub enrichment — cached lookups of PR title/state/reviews and commit
   messages, used to label runs (rate-limited, so authenticated when a token is
   set).
3. Metric extraction — a family of ``extract_*`` functions that roll the shaped
   traces up into per-test durations, per-run summaries, PR info/state, averages,
   and resource metrics (the last read from the plugin's JSONL file).
4. Self-observability — :func:`_exporter_self_metric_lines` reports the
   exporter's own up/render-time/RSS/last-render-timestamp so its health shows on
   the CI dashboard even when CI is idle.
5. Render, publish & serve — :func:`_render_metrics_uncached` builds the body in a
   background thread (:func:`_refresh_loop`) so scrapes never block on Tempo;
   :func:`_refresh_cache_once` publishes it atomically to a disk file, and
   :class:`MetricsHandler` streams that file so the multi-MB payload stays off the
   Python heap and survives a restart. See ``docs/design.md`` for the rationale.
"""

from __future__ import annotations

import html
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import islice
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import fsum
from pathlib import Path
from urllib.parse import parse_qs, quote, urlencode, urlparse
from urllib.request import Request, urlopen

try:
    # Optional: only present with the ``otel`` extra. Used to turn GitHub-style
    # emoji shortcodes (":rotating_light:") embedded in PR titles and commit
    # subjects into real Unicode glyphs, since Grafana renders shortcodes
    # literally. Absent in the dependency-free core, where it degrades to a
    # no-op (see :func:`_emojize`).
    import emoji as _emoji
except ImportError:  # pragma: no cover - exercised only without the otel extra
    _emoji = None


DEFAULT_TEMPO_URL = "http://tempo:3200"
DEFAULT_LIMIT = 100
DEFAULT_LOOKBACK = "1h"
DEFAULT_PORT = 8000
DEFAULT_RESOURCE_METRICS_FILE = "/data/pytest-resource-metrics.jsonl"
# The rendered Prometheus payload is written here instead of being held in the
# Python heap. It lives on the persistent /data volume so the last complete
# payload survives a crash/restart and /metrics can serve it immediately while
# the background thread re-renders, rather than falling back to a bare warming
# payload. Publishing is atomic (temp file + os.replace), so a crash mid-write
# can never expose a torn body.
DEFAULT_PAYLOAD_FILE = "/data/pytest-metrics-payload.prom"
METRICS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
SVG_CONTENT_TYPE = "image/svg+xml; charset=utf-8"
JSON_CONTENT_TYPE = "application/json; charset=utf-8"
# Soft RSS ceiling (MiB). When a render is about to run and RSS is already above
# this, the reclaimable trace cache is dropped first so the render doesn't tip
# the process into the hard container limit (an OOM-kill). ~88% of the 640 MiB
# cap; set to 0 to disable.
DEFAULT_MEM_SOFT_MB = 560
DEFAULT_SERVICE_NAME = "pytest-observability-demo"
DEFAULT_CACHE_SECONDS = 10.0
DEFAULT_REFRESH_COOLDOWN_SECONDS = 60.0
DEFAULT_REFRESH_SLOW_SECONDS = 30.0
DEFAULT_GITHUB_API_URL = "https://api.github.com"
DEFAULT_GITHUB_CACHE_SECONDS = 300.0
# A CI trace counts as "settled" (immutable, safe to memoize) once its span
# count has held steady for this long — see :func:`_trace_is_settled`. NOT
# merely once its newest span is old: a sharded pytest-xdist job is fed by
# several worker processes that flush spans at staggered exit times, so its
# trace keeps growing for minutes while its newest-present span is already old.
# Keying settle off count-quiescence stops us memoizing a partial snapshot.
DEFAULT_TRACE_SETTLE_SECONDS = 120.0
# Each get_trace() is an independent, blocking Tempo round-trip. Keep this low:
# fetching several multi-MB traces in parallel can make small single-node Tempo
# instances spike hard enough to hit their container memory limit.
DEFAULT_FETCH_CONCURRENCY = 4
# The PR badge/summary endpoints answer for a *specific* PR, whose last run is
# often hours old — outside the live render's lookback window. So they fall back
# to a targeted Tempo search scoped to that one PR over this wider window,
# independent of PYTEST_TRACE_EXPORTER_LOOKBACK. The window is bounded by Tempo's
# query_frontend.search.max_duration (26h in this deployment) — a larger value is
# rejected by Tempo and the fallback degrades to "no data", so keep this under
# that ceiling. To persist badges beyond ~a day you must also raise Tempo's
# max_duration (or add windowed paging). The per-PR result is memoized for
# DEFAULT_BADGE_CACHE_SECONDS so a hammered badge does not pound Tempo.
DEFAULT_BADGE_LOOKBACK = "24h"
DEFAULT_BADGE_TRACE_LIMIT = 100
DEFAULT_BADGE_CACHE_SECONDS = 120.0
DEFAULT_BADGE_PROMETHEUS_LOOKBACK = "90d"
DEFAULT_PROMETHEUS_URL = ""


def env_int(name: str, default: int) -> int:
    value = os.getenv(name, "")
    try:
        return int(value) if value else default
    except ValueError:
        return default


def metric_labels(labels: dict[str, str]) -> str:
    segments = []
    for key, value in labels.items():
        escaped = value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
        segments.append(f'{key}="{escaped}"')
    return "{" + ",".join(segments) + "}"


def tag_map(items: list[dict]) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for item in items:
        key = item.get("key")
        value = item.get("value")
        if isinstance(key, str) and value is not None:
            mapped[key] = str(value)
    return mapped


def extract_exception_info(span: dict) -> tuple[str, str]:
    logs = span.get("logs", [])
    if not isinstance(logs, list):
        return "", ""
    for log in logs:
        if not isinstance(log, dict):
            continue
        fields = tag_map(log.get("fields", []))
        if fields.get("event") == "exception":
            stacktrace = fields.get("exception.stacktrace", "")
            if len(stacktrace) > 4000:
                stacktrace = stacktrace[:4000] + "\n... (truncated)"
            return fields.get("exception.type", ""), stacktrace
    return "", ""


def extract_failure_details(trace: dict, test_nodeid: str = "") -> list[dict[str, str]]:
    """Pull full (untruncated) exception info for failing test spans in a trace.

    Unlike ``extract_exception_info`` (which feeds metrics and caps the
    stacktrace), this returns the complete ``exception.message`` and
    ``exception.stacktrace`` for the ``/failure`` traceback page. Optionally
    filtered to a single ``test_nodeid``.
    """
    details: list[dict[str, str]] = []
    spans = trace.get("spans", [])
    if not isinstance(spans, list):
        return details
    for span in spans:
        if not isinstance(span, dict):
            continue
        tags = tag_map(span.get("tags", []))
        if tags.get("pytest.span_type") != "test":
            continue
        nodeid = tags.get("pytest.nodeid") or str(span.get("operationName", ""))
        if test_nodeid and nodeid != test_nodeid:
            continue
        logs = span.get("logs", [])
        fields: dict[str, str] = {}
        for log in logs if isinstance(logs, list) else []:
            if isinstance(log, dict):
                candidate = tag_map(log.get("fields", []))
                if candidate.get("event") == "exception":
                    fields = candidate
                    break
        if not fields:
            continue
        details.append(
            {
                "test_nodeid": nodeid,
                "exception_type": fields.get("exception.type", ""),
                "exception_message": fields.get("exception.message", ""),
                "exception_stacktrace": fields.get("exception.stacktrace", ""),
            }
        )
    return details


def render_failure_html(trace_id: str, details: list[dict[str, str]]) -> str:
    """Render a self-contained, dark-themed HTML page for a trace's failures.

    Pytest exception messages and tracebacks are multi-line and render as an
    unreadable wall in Tempo's span-event view. This serves them in monospace
    ``<pre>`` blocks with preserved line breaks instead.
    """
    esc = html.escape
    out = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>Traceback {esc(trace_id)}</title>",
        "<style>"
        "body{margin:0;padding:14px;background:#0b0c0e;color:#d8d9da;"
        "font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}"
        "h2{margin:0 0 2px;font:600 14px system-ui,sans-serif;color:#f2cc60}"
        ".nodeid{margin:0 0 10px;color:#8e9197;font:12px system-ui,sans-serif}"
        ".nodeid a{color:#6ab0ff;text-decoration:none}"
        ".nodeid a:hover{text-decoration:underline}"
        ".label{margin:14px 0 4px;color:#8e9197;font:600 11px system-ui,sans-serif;"
        "text-transform:uppercase;letter-spacing:.05em}"
        "pre{margin:0;white-space:pre-wrap;word-break:break-word;background:#141619;"
        "border:1px solid #24262b;border-radius:6px;padding:12px;overflow:auto}"
        ".msg{color:#ff8a80}"
        "</style></head><body>",
    ]
    if not details:
        out.append("<p style='color:#8e9197;font-family:system-ui'>")
        out.append(
            "No failing test span found for this trace."
            if trace_id
            else "No trace selected."
        )
        out.append("</p>")
    for detail in details:
        out.append(f"<h2>{esc(detail['exception_type'] or 'Failure')}</h2>")
        github_url = detail.get("github_url", "")
        if github_url:
            nodeid_html = (
                f'<a href="{esc(github_url)}" target="_blank" rel="noopener">'
                f"{esc(detail['test_nodeid'])} ↗</a>"
            )
        else:
            nodeid_html = esc(detail["test_nodeid"])
        out.append(f"<div class='nodeid'>{nodeid_html}</div>")
        if detail["exception_message"]:
            out.append("<div class='label'>Message</div>")
            out.append(f"<pre class='msg'>{esc(detail['exception_message'])}</pre>")
        if detail["exception_stacktrace"]:
            out.append("<div class='label'>Traceback</div>")
            out.append(f"<pre>{esc(detail['exception_stacktrace'])}</pre>")
    out.append("</body></html>")
    return "".join(out)


def extract_test_line(stacktrace: str, test_nodeid: str) -> str:
    """Pull the first source line in *stacktrace* that points at the test file.

    Pytest's exception output starts with `<test_file>:<lineno>:` at the top of
    the captured stacktrace, before any framework or library frames. Used to
    deep-link the failure panel to the offending line on GitHub.
    """
    if not stacktrace or not test_nodeid:
        return ""
    test_file = test_nodeid.split("::", 1)[0]
    if not test_file:
        return ""
    match = re.search(rf"{re.escape(test_file)}:(\d+)", stacktrace)
    return match.group(1) if match else ""


def github_test_url(
    repository: str, ref: str, test_nodeid: str, stacktrace: str
) -> str:
    """Build a GitHub blob link to the failing test's file and line.

    ``ref`` should be the PR head commit SHA when known (so line numbers match
    the code that actually ran), falling back to ``main``. The line is pulled
    from the traceback via :func:`extract_test_line`.
    """
    test_file = test_nodeid.split("::", 1)[0]
    if not repository or not test_file:
        return ""
    url = f"https://github.com/{repository}/blob/{ref or 'main'}/{test_file}"
    line = extract_test_line(stacktrace, test_nodeid)
    if line:
        url += f"#L{line}"
    return url


def annotate_github_links(
    trace: dict,
    details: list[dict[str, str]],
    *,
    _metadata_fetcher: Callable[[str, str], dict[str, str]] | None = None,
) -> None:
    """Add a ``github_url`` to each failure detail pointing at the test file/line.

    Resolves the repository from the trace's resource tags and the PR head
    commit SHA via the GitHub API (cached) so the linked line matches the code
    that ran; falls back to the ``main`` branch when there is no numeric PR.
    """
    trace_info, _ = extract_trace_rows(trace)
    repository = str(trace_info.get("repository") or "")
    if not repository:
        repository = repository_from_pr_url(str(trace_info.get("pr_url") or ""))
    pr = str(trace_info.get("pr") or "none")
    ref = "main"
    if repository and pr.isdigit():
        fetcher = _metadata_fetcher or fetch_github_pr_info_cached
        ref = fetcher(repository, pr).get("commit_sha") or "main"
    for detail in details:
        detail["github_url"] = github_test_url(
            repository, ref, detail["test_nodeid"], detail["exception_stacktrace"]
        )


def split_pytest_nodeid(nodeid: str) -> dict[str, str]:
    parts = nodeid.split("::")
    module_path = parts[0] if parts else ""
    module_name = os.path.basename(module_path)
    if len(parts) >= 3:
        class_name = parts[-2]
        function_name = parts[-1]
    elif len(parts) == 2:
        class_name = ""
        function_name = parts[-1]
    else:
        class_name = ""
        function_name = ""

    return {
        "test_class": class_name,
        "test_function": function_name,
        "test_module": module_name,
    }


def trace_start_time(trace: dict) -> int:
    spans = trace.get("spans", [])
    if not isinstance(spans, list):
        return 0
    return max(
        (int(span.get("startTime", 0)) for span in spans if isinstance(span, dict)),
        default=0,
    )


def tempo_base_url() -> str:
    # PYTEST_TRACE_EXPORTER_JAEGER_URL is accepted as a deprecated alias so an
    # existing deployment's env keeps working through the Tempo cutover.
    raw = os.getenv("PYTEST_TRACE_EXPORTER_TEMPO_URL") or os.getenv(
        "PYTEST_TRACE_EXPORTER_JAEGER_URL", DEFAULT_TEMPO_URL
    )
    return raw.rstrip("/")


_LOOKBACK_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_lookback_seconds(lookback: str, default: str = DEFAULT_LOOKBACK) -> int:
    """Convert a Go-style duration like ``1h``/``30m``/``2d`` to seconds.

    Tempo's search API wants an absolute ``start``/``end`` window in unix
    seconds, whereas the old Jaeger query took a relative ``lookback`` string;
    this bridges the two so the exporter keeps the same env knob.
    """
    for candidate in (lookback, default):
        text = (candidate or "").strip().lower()
        match = re.fullmatch(r"(\d+)([smhd])", text)
        if match:
            return int(match.group(1)) * _LOOKBACK_UNIT_SECONDS[match.group(2)]
    return 3600


def _otlp_scalar(value: object) -> str:
    """Flatten an OTLP ``AnyValue`` JSON object down to a string.

    OTLP attribute values are tagged unions (``{"stringValue": ...}``,
    ``{"intValue": "5"}``, ...). Jaeger flattened these to plain strings, which
    is the shape ``tag_map`` and the rest of the exporter already expect.
    """
    if not isinstance(value, dict):
        return str(value)
    for key in ("stringValue", "intValue", "doubleValue"):
        if key in value:
            return str(value[key])
    if "boolValue" in value:
        return "true" if value["boolValue"] else "false"
    if "arrayValue" in value:
        items = value["arrayValue"].get("values", [])
        return ",".join(_otlp_scalar(item) for item in items)
    return ""


def otlp_attributes_to_tags(attributes: object) -> list[dict[str, str]]:
    tags: list[dict[str, str]] = []
    if not isinstance(attributes, list):
        return tags
    for attribute in attributes:
        if not isinstance(attribute, dict):
            continue
        key = attribute.get("key")
        if isinstance(key, str):
            tags.append({"key": key, "value": _otlp_scalar(attribute.get("value"))})
    return tags


_OTLP_STATUS_CODE = {
    0: "UNSET",
    1: "OK",
    2: "ERROR",
    "STATUS_CODE_UNSET": "UNSET",
    "STATUS_CODE_OK": "OK",
    "STATUS_CODE_ERROR": "ERROR",
}


def tempo_trace_to_jaeger(trace_id: str, payload: dict) -> dict:
    """Adapt Tempo's OTLP-JSON trace into the Jaeger-shaped dict the rest of
    the exporter already knows how to aggregate.

    Tempo returns ``{"batches": [ResourceSpans, ...]}``. Each batch carries a
    ``resource`` (→ a Jaeger *process*) and ``scopeSpans``/``spans``. We map
    span attributes to Jaeger ``tags``, synthesize the ``otel.status_code`` tag
    from the OTLP ``status`` enum, and turn span ``events`` into Jaeger
    ``logs`` so ``extract_exception_info`` keeps working unchanged.
    """
    batches = payload.get("batches")
    if not isinstance(batches, list):
        batches = payload.get("resourceSpans", [])
    processes: dict[str, dict] = {}
    spans: list[dict] = []

    for index, batch in enumerate(batches if isinstance(batches, list) else []):
        if not isinstance(batch, dict):
            continue
        process_id = f"p{index}"
        resource = batch.get("resource", {})
        resource_attrs = (
            resource.get("attributes", []) if isinstance(resource, dict) else []
        )
        resource_tags = otlp_attributes_to_tags(resource_attrs)
        service_name = next(
            (tag["value"] for tag in resource_tags if tag["key"] == "service.name"),
            "unknown",
        )
        processes[process_id] = {"serviceName": service_name, "tags": resource_tags}

        scope_spans = batch.get("scopeSpans")
        if not isinstance(scope_spans, list):
            scope_spans = batch.get("instrumentationLibrarySpans", [])
        for scope_span in scope_spans if isinstance(scope_spans, list) else []:
            if not isinstance(scope_span, dict):
                continue
            for span in scope_span.get("spans", []):
                if not isinstance(span, dict):
                    continue
                start_nano = int(span.get("startTimeUnixNano", 0) or 0)
                end_nano = int(span.get("endTimeUnixNano", 0) or 0)
                tags = otlp_attributes_to_tags(span.get("attributes"))
                if not any(tag["key"] == "otel.status_code" for tag in tags):
                    status = span.get("status", {})
                    code = status.get("code") if isinstance(status, dict) else None
                    tags.append(
                        {
                            "key": "otel.status_code",
                            "value": _OTLP_STATUS_CODE.get(code, "UNSET"),
                        }
                    )
                logs = []
                for event in span.get("events", []) or []:
                    if not isinstance(event, dict):
                        continue
                    fields = [{"key": "event", "value": str(event.get("name", ""))}]
                    fields.extend(otlp_attributes_to_tags(event.get("attributes")))
                    logs.append({"fields": fields})
                spans.append(
                    {
                        "duration": max(0, end_nano - start_nano) // 1000,
                        "logs": logs,
                        "operationName": span.get("name", ""),
                        "processID": process_id,
                        "startTime": start_nano // 1000,
                        "tags": tags,
                    }
                )

    return {"processes": processes, "spans": spans, "traceID": trace_id}


# Tempo search/get_trace latency over real CI-volume blocks is usually several
# seconds. Keep a firm timeout so slow traces are retried on a later refresh
# instead of letting many blocked requests pile pressure onto Tempo.
DEFAULT_HTTP_TIMEOUT = 10.0


def _http_timeout() -> float:
    raw = os.getenv("PYTEST_TRACE_EXPORTER_HTTP_TIMEOUT", "")
    try:
        return max(1.0, float(raw)) if raw else DEFAULT_HTTP_TIMEOUT
    except ValueError:
        return DEFAULT_HTTP_TIMEOUT


def _http_get_json(url: str, timeout: float | None = None) -> object:
    if timeout is None:
        timeout = _http_timeout()
    with urlopen(url, timeout=timeout) as response:
        return json.load(response)


def search_trace_ids(
    base_url: str,
    service_name: str,
    start: int,
    end: int,
    limit: int,
    extra_selector: str = "",
) -> list[str]:
    selector = f'resource.service.name = "{service_name}"'
    if extra_selector:
        selector = f"{selector} && {extra_selector}"
    traceql = quote(f"{{ {selector} }}", safe="")
    search_url = (
        f"{base_url}/api/search?q={traceql}&start={start}&end={end}&limit={limit}"
    )
    payload = _http_get_json(search_url)
    if not isinstance(payload, dict):
        return []
    found = payload.get("traces", [])
    if not isinstance(found, list):
        return []
    trace_ids: list[str] = []
    for entry in found:
        if isinstance(entry, dict):
            trace_id = entry.get("traceID")
            if isinstance(trace_id, str) and trace_id:
                trace_ids.append(trace_id)
    return trace_ids


# Tempo's /api/search returns at most ``limit`` traces, ordered most-recent
# first. At real CI volume (≈1500 traces/hr) a single search over a 1h window
# only ever surfaces the newest ``limit`` (200) of them — so the traces of a
# large sharded run (e.g. tests_torch's 8 shard traces) get buried behind newer
# small traces and are NEVER fetched. That silently truncated every dashboard
# roll-up to whatever fraction of a run's shards happened to be recent enough
# (~3 of 8 → ~15k of ~40k tests). See docs/session-notes-exporter-2026-06-23.md.
#
# search_all_trace_ids fixes the enumeration: it adaptively bisects the time
# window until every sub-slice returns fewer than ``limit`` traces, so the union
# is the COMPLETE set of trace ids in the window regardless of volume. Search
# returns only ids (no span payloads), so the extra calls are cheap; the
# expensive per-trace fetch is still bounded elsewhere (the render fetches each
# trace at most once and caps new fetches per cycle).
DEFAULT_SEARCH_MAX_SLICES = 64


def search_all_trace_ids(
    base_url: str,
    service_name: str,
    start: int,
    end: int,
    limit: int,
    *,
    max_slices: int = DEFAULT_SEARCH_MAX_SLICES,
    extra_selector: str = "",
) -> tuple[list[str], bool]:
    """Completely enumerate the window's trace ids via adaptive time-bisection.

    Returns ``(trace_ids, truncated)``. A slice that comes back *full* (``>=
    limit``) means it holds more traces than one search can return, so it is
    split at its midpoint and each half is searched independently; this repeats
    until every slice is under-full. ``truncated`` is ``True`` if the slice
    budget (``max_slices``) ran out or a slice could not be split further (a
    burst of >``limit`` traces within a 1-second window) while still full — i.e.
    enumeration may be incomplete and the caller should surface that.
    """
    if limit <= 0 or end <= start:
        return [], False

    seen: set[str] = set()
    ordered: list[str] = []
    truncated = False
    # LIFO stack of [start, end) slices still to search.
    stack: list[tuple[int, int]] = [(start, end)]
    slices = 0

    def _record(ids: list[str]) -> None:
        for trace_id in ids:
            if trace_id not in seen:
                seen.add(trace_id)
                ordered.append(trace_id)

    while stack:
        slice_start, slice_end = stack.pop()
        if slices >= max_slices:
            # Out of budget: take whatever this slice yields and flag truncation
            # rather than searching unboundedly.
            _record(
                search_trace_ids(
                    base_url,
                    service_name,
                    slice_start,
                    slice_end,
                    limit,
                    extra_selector,
                )
            )
            truncated = True
            continue
        slices += 1
        ids = search_trace_ids(
            base_url, service_name, slice_start, slice_end, limit, extra_selector
        )
        if len(ids) < limit:
            _record(ids)
            continue
        # Slice is saturated — it hides more than ``limit`` traces. Bisect it and
        # re-search each half; the parent's (truncated) ids are discarded because
        # the children re-enumerate the same range completely.
        mid = (slice_start + slice_end) // 2
        if mid <= slice_start or mid >= slice_end:
            # Cannot split a 1-second slice further; keep what we have and flag.
            _record(ids)
            truncated = True
            continue
        stack.append((slice_start, mid))
        stack.append((mid, slice_end))

    return ordered, truncated


# LRU cache of settled (immutable) traces, bounded by both entry count and an
# approximate serialized byte budget. The cache holds full multi-MB Jaeger dicts,
# so count-only bounds are not enough when a replay/search window contains a few
# very large traces.
_trace_cache_lock = threading.Lock()
_trace_cache: "OrderedDict[str, dict]" = OrderedDict()
_trace_cache_sizes: dict[str, int] = {}
_trace_cache_bytes = 0
DEFAULT_TRACE_CACHE_MAX = 256  # just above the fetch LIMIT (default 200)
DEFAULT_TRACE_CACHE_MAX_BYTES = 128 * 1024 * 1024
# A parsed Jaeger trace dict occupies far more RAM than its compact JSON: nested
# dicts/lists/str objects each carry per-object overhead. Measured ~4x on real
# transformers-CI traces (0.89 MB serialized -> 3.54 MB resident). The cap is a
# *memory* budget, so scale the serialized size by this factor — otherwise a
# 128 MiB cap silently admitted ~520 MiB of resident traces and pinned the
# exporter against its 640 MiB container limit.
_TRACE_RAM_OVERHEAD = 4


def _trace_cache_max() -> int:
    return max(
        0, env_int("PYTEST_TRACE_EXPORTER_TRACE_CACHE_MAX", DEFAULT_TRACE_CACHE_MAX)
    )


def _trace_cache_max_bytes() -> int:
    return max(
        0,
        env_int(
            "PYTEST_TRACE_EXPORTER_TRACE_CACHE_MAX_BYTES",
            DEFAULT_TRACE_CACHE_MAX_BYTES,
        ),
    )


def _trace_cache_entry_bytes(trace: dict) -> int:
    """Estimate a trace's *resident* memory cost for the cache byte budget.

    json.dumps gives a cheap, deterministic size proxy; multiply by
    :data:`_TRACE_RAM_OVERHEAD` to approximate the in-RAM footprint the cache
    actually holds (a full getsizeof walk per insert would be far costlier).
    """
    try:
        serialized = len(
            json.dumps(trace, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
    except (TypeError, ValueError):
        return 0
    return serialized * _TRACE_RAM_OVERHEAD


def _trace_settle_seconds() -> float:
    raw = os.getenv("PYTEST_TRACE_EXPORTER_TRACE_SETTLE_SECONDS")
    if raw is None or raw == "":
        return DEFAULT_TRACE_SETTLE_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_TRACE_SETTLE_SECONDS


def _trace_refetch_interval() -> float:
    """Minimum wall-clock between re-fetches of one still-unsettled trace.

    The render path re-reads an in-flight (unsettled) trace at most this often,
    serving its last shaped rows from cache in between, so render duration stays
    flat regardless of how many large sharded runs are in flight. Defaults to the
    settle window — re-reading more often than that only burns Tempo I/O without
    settling sooner (settle needs the count to hold steady for a whole window).
    """
    raw = os.getenv("PYTEST_TRACE_EXPORTER_TRACE_REFETCH_SECONDS")
    if raw is None or raw == "":
        return _trace_settle_seconds()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return _trace_settle_seconds()


# Count-quiescence settle tracking. A trace is "settled" (safe to memoize as
# immutable) only once its span count has stopped growing for ``settle_seconds``
# of wall-clock — NOT merely because its newest span is old. Sharded CI jobs run
# pytest-xdist across several worker PROCESSES that each flush their spans at
# staggered exit times, so one shard's trace keeps growing for minutes and its
# "newest span" can already be old while more spans are still arriving. Keying
# settle off count-quiescence (rather than span age) is what stops the exporter
# from memoizing a partial snapshot and freezing it (which capped sharded jobs
# at ~half their tests). Maps trace_id -> (last_seen_span_count, wall-clock time
# the count last changed).
_trace_growth_lock = threading.Lock()
_trace_growth: "OrderedDict[str, tuple[int, float]]" = OrderedDict()
# Bound the tracker so trace_ids that age out of the window before ever settling
# (rare) can't leak unboundedly; far above any real in-flight window.
_TRACE_GROWTH_MAX = 8192


def _trace_is_settled(
    trace_id: str, trace: dict, now: float, settle_seconds: float
) -> bool:
    """Return True once ``trace_id``'s span count has held steady for
    ``settle_seconds``.

    Records the current span count against the wall-clock time it last changed;
    the trace counts as settled only after that count has been stable for the
    full ``settle_seconds`` window. The first sighting never settles (we need a
    prior observation to know the count is steady) — that also costs one extra
    fetch for genuinely historical traces, which the raw/shaped caches absorb
    thereafter. Being called more than once per render is harmless: an unchanged
    count never resets the clock.
    """
    spans = trace.get("spans", [])
    span_count = len(spans) if isinstance(spans, list) else 0
    with _trace_growth_lock:
        prev = _trace_growth.get(trace_id)
        if prev is None:
            _trace_growth[trace_id] = (span_count, now)
            _trace_growth.move_to_end(trace_id)
            while len(_trace_growth) > _TRACE_GROWTH_MAX:
                _trace_growth.popitem(last=False)
            return False
        last_count, last_change = prev
        if span_count != last_count:
            last_change = now
        _trace_growth[trace_id] = (span_count, last_change)
        _trace_growth.move_to_end(trace_id)
        settled = (now - last_change) >= settle_seconds
        if settled:
            _trace_growth.pop(trace_id, None)
        return settled


def _store_trace(trace_id: str, trace: dict) -> None:
    """Memoize a settled (immutable) trace in the bounded raw-trace LRU."""
    max_entries = _trace_cache_max()
    max_bytes = _trace_cache_max_bytes()
    if max_entries == 0 or max_bytes == 0:
        return
    entry_bytes = _trace_cache_entry_bytes(trace)
    with _trace_cache_lock:
        global _trace_cache_bytes
        if not _trace_cache:
            _trace_cache_sizes.clear()
            _trace_cache_bytes = 0
        previous_size = _trace_cache_sizes.pop(trace_id, 0)
        _trace_cache_bytes = max(0, _trace_cache_bytes - previous_size)
        _trace_cache[trace_id] = trace
        _trace_cache_sizes[trace_id] = entry_bytes
        _trace_cache_bytes += entry_bytes
        _trace_cache.move_to_end(trace_id)
        while len(_trace_cache) > max_entries or _trace_cache_bytes > max_bytes:
            evicted_id, _ = _trace_cache.popitem(last=False)
            evicted_size = _trace_cache_sizes.pop(evicted_id, 0)
            _trace_cache_bytes = max(0, _trace_cache_bytes - evicted_size)


def _fetch_trace_with_settled(
    trace_id: str, base_url: str | None, now: float, settle_seconds: float
) -> tuple[dict | None, bool]:
    """Fetch one trace (Jaeger-shaped) and report whether it has settled.

    A settled trace is served from the raw-trace cache on later calls (and
    reported settled); an in-flight trace is re-fetched from Tempo every call so
    its later spans are picked up. The settle decision (count-quiescence) is made
    exactly once per fetch here, so the raw-trace and shaped caches stay
    consistent. Returns ``(None, False)`` when the fetch fails (retried later).
    """
    with _trace_cache_lock:
        cached = _trace_cache.get(trace_id)
        if cached is not None:
            _trace_cache.move_to_end(trace_id)  # mark most-recently-used
    if cached is not None:
        return cached, True

    if base_url is None:
        base_url = tempo_base_url()
    try:
        payload = _http_get_json(f"{base_url}/api/traces/{quote(trace_id, safe='')}")
    except Exception:
        return None, False
    if not isinstance(payload, dict):
        return None, False

    trace = tempo_trace_to_jaeger(trace_id, payload)
    settled = _trace_is_settled(trace_id, trace, now, settle_seconds)
    if settled:
        _store_trace(trace_id, trace)
    return trace, settled


def get_trace(trace_id: str, base_url: str | None = None) -> dict | None:
    """Return one trace in the Jaeger-shaped dict, fetching from Tempo if needed.

    A trace is memoized (and thereafter served from cache) only once its span
    count has been stable for the settle window — see :func:`_trace_is_settled`.
    Until then every call re-fetches from Tempo so an in-flight run's later spans
    (e.g. the staggered xdist-worker flushes of a sharded job) are not lost. Used
    by the scrape loop (``fetch_traces``) and the ``/failure`` traceback view.
    """
    trace, _ = _fetch_trace_with_settled(
        trace_id, base_url, time.time(), _trace_settle_seconds()
    )
    return trace


def iter_traces(base_url: str | None = None) -> Iterator[dict]:
    """Yield the window's traces one at a time, fetched concurrently with a
    bounded number in flight.

    Streaming (rather than returning the whole list) is what keeps the exporter
    memory-flat: the caller shapes each trace into small rows and drops it, so
    peak residency is the in-flight window plus the small shaped rows — never
    every (multi-MB) trace at once. A bounded sliding window of `concurrency`
    outstanding fetches both parallelizes Tempo's I/O waits (get_trace is I/O
    bound and thread-safe) AND caps how many fetched-but-unconsumed traces
    buffer up while the caller shapes each one. A slow/failing trace drops to
    None instead of sinking the render; result order doesn't matter (metrics
    aggregate).
    """
    if base_url is None:
        base_url = tempo_base_url()
    limit = env_int("PYTEST_TRACE_EXPORTER_LIMIT", DEFAULT_LIMIT)
    lookback = os.getenv("PYTEST_TRACE_EXPORTER_LOOKBACK", DEFAULT_LOOKBACK)
    service_name = os.getenv("PYTEST_TRACE_EXPORTER_SERVICE_NAME", DEFAULT_SERVICE_NAME)

    end = int(time.time())
    start = end - parse_lookback_seconds(lookback)
    trace_ids, _ = search_all_trace_ids(base_url, service_name, start, end, limit)
    if not trace_ids:
        return

    workers = max(
        1,
        min(
            env_int(
                "PYTEST_TRACE_EXPORTER_FETCH_CONCURRENCY", DEFAULT_FETCH_CONCURRENCY
            ),
            len(trace_ids),
        ),
    )
    pending = iter(trace_ids)
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="trace-fetch"
    ) as pool:
        inflight = {
            pool.submit(get_trace, tid, base_url) for tid in islice(pending, workers)
        }
        while inflight:
            done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for future in done:
                # Refill the window before yielding so a fetch is always running.
                nxt = next(pending, None)
                if nxt is not None:
                    inflight.add(pool.submit(get_trace, nxt, base_url))
                try:
                    trace = future.result()
                except Exception:
                    trace = None
                if trace is not None:
                    yield trace


def fetch_traces() -> list[dict]:
    """Eager list of the window's traces. Kept for callers/tests; the metrics
    render streams via :func:`iter_traces` to avoid materialising them all."""
    return list(iter_traces())


# ---------------------------------------------------------------------------
# Shaped-window cache — lets the render assemble the COMPLETE window cheaply.
#
# With complete enumeration the window can hold ~1500 traces/hr; keeping that
# many raw multi-MB traces resident is infeasible (~5 GiB). But each trace's
# *shaped* result — the small (trace_info, rows) tuple the extractors consume —
# is orders of magnitude smaller. So we memoize the shaped result by trace_id.
# Each render then rebuilds the whole window from this cache and only fetches the
# traces it must this cycle, instead of re-fetching the entire window.
#
# We cache the shape of EVERY trace we fetch, settled or not — keyed by
# ``_shaped_meta[trace_id] = (settled, last_fetch_wall)``. A settled trace's
# shape is immutable and served forever. A still-in-flight (unsettled) trace's
# shape is served too, but re-fetched at most once per :func:`_trace_refetch_interval`
# (its rows refreshed as later xdist-worker spans arrive). Caching unsettled
# shapes — rather than re-fetching every in-flight trace each render — is what
# keeps render duration flat under load: an earlier version re-read every
# unsettled multi-MB trace every cycle and the render time spiralled (15s->120s+)
# as concurrent sharded runs piled up.
# ---------------------------------------------------------------------------

# A shaped trace: the (trace_info, rows) pair extract_trace_rows returns. Kept
# loose (tuple[dict, list]) at runtime to avoid PEP 604 unions in an eagerly
# evaluated alias; the precise element types live on extract_trace_rows.
ShapedEntry = tuple[dict, list]

_shaped_cache_lock = threading.Lock()
_shaped_cache: "OrderedDict[str, ShapedEntry]" = OrderedDict()
_shaped_cache_sizes: dict[str, int] = {}
_shaped_cache_bytes = 0
# Per-trace settle/freshness metadata, guarded by ``_shaped_cache_lock`` and kept
# in lockstep with ``_shaped_cache`` (evicted/cleared together). Maps trace_id ->
# (settled, last_fetch_wall_clock). ``settled`` traces are never re-fetched;
# unsettled ones are re-fetched only once their ``last_fetch`` is older than the
# re-fetch interval.
_shaped_meta: dict[str, tuple[bool, float]] = {}
# Sized to comfortably cover well over an hour of CI volume so a run's shards
# survive in the cache until the run settles, even when their completion spans
# longer than the lookback window.
DEFAULT_SHAPED_CACHE_MAX = 8192
DEFAULT_SHAPED_CACHE_MAX_BYTES = 384 * 1024 * 1024
# How many not-yet-seen traces a single render is allowed to fetch+shape. This
# bounds per-cycle Tempo load and the cold-start/restart burst (when the shaped
# cache is empty the whole window is "new"); deferred traces stay in the window
# and are picked up over the next few renders. A run only emits its roll-up once
# settled, by which point its deferred shards have been fetched, so the cap
# never truncates a settled run's totals.
DEFAULT_MAX_NEW_FETCH_PER_RENDER = 600

# Observability for the render/self-metrics: stats from the most recent
# enumeration pass (updated by :func:`_iter_window_shaped`).
_enumeration_lock = threading.Lock()
_last_enumeration_total = 0
_last_enumeration_truncated = False
_last_enumeration_deferred = 0
# Trace ids whose per-test rows were emitted as "new" on the PREVIOUS render.
# Re-emitting them once more (while they are now cheap cache hits) guarantees a
# freshly-seen trace's high-cardinality series appears in at least two published
# payloads — so a Prometheus scrape can't miss it in the gap between the render
# that first saw it and the next republish. Touched only by the single render
# thread.
_previous_new_ids: set[str] = set()


def _shaped_cache_max() -> int:
    return max(
        0, env_int("PYTEST_TRACE_EXPORTER_SHAPED_CACHE_MAX", DEFAULT_SHAPED_CACHE_MAX)
    )


def _shaped_cache_max_bytes() -> int:
    return max(
        0,
        env_int(
            "PYTEST_TRACE_EXPORTER_SHAPED_CACHE_MAX_BYTES",
            DEFAULT_SHAPED_CACHE_MAX_BYTES,
        ),
    )


def _max_new_fetch_per_render() -> int:
    return max(
        1,
        env_int(
            "PYTEST_TRACE_EXPORTER_MAX_NEW_FETCH_PER_RENDER",
            DEFAULT_MAX_NEW_FETCH_PER_RENDER,
        ),
    )


def _shaped_entry_bytes(entry: ShapedEntry) -> int:
    """Approximate resident bytes of a shaped (trace_info, rows) tuple.

    Mirrors :func:`_trace_cache_entry_bytes`: a cheap serialized-size proxy
    scaled by the parsed-object overhead factor, so the byte budget reflects the
    in-RAM footprint rather than the compact JSON size.
    """
    try:
        serialized = len(
            json.dumps(entry, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
    except (TypeError, ValueError):
        return 0
    return serialized * _TRACE_RAM_OVERHEAD


def _store_shaped(
    trace_id: str, entry: ShapedEntry, settled: bool, fetched_at: float
) -> None:
    """Memoize a shaped trace and record its settle/freshness metadata.

    Both settled and still-in-flight shapes are cached; ``settled`` and
    ``fetched_at`` (recorded in :data:`_shaped_meta`, kept in lockstep with the
    cache) let the render decide whether a trace may be served as-is or is due
    for a bounded re-fetch.
    """
    max_entries = _shaped_cache_max()
    max_bytes = _shaped_cache_max_bytes()
    if max_entries == 0 or max_bytes == 0:
        return
    entry_bytes = _shaped_entry_bytes(entry)
    with _shaped_cache_lock:
        global _shaped_cache_bytes
        if not _shaped_cache:
            _shaped_cache_sizes.clear()
            _shaped_meta.clear()
            _shaped_cache_bytes = 0
        previous_size = _shaped_cache_sizes.pop(trace_id, 0)
        _shaped_cache_bytes = max(0, _shaped_cache_bytes - previous_size)
        _shaped_cache[trace_id] = entry
        _shaped_cache_sizes[trace_id] = entry_bytes
        _shaped_meta[trace_id] = (settled, fetched_at)
        _shaped_cache_bytes += entry_bytes
        _shaped_cache.move_to_end(trace_id)
        while len(_shaped_cache) > max_entries or _shaped_cache_bytes > max_bytes:
            evicted_id, _ = _shaped_cache.popitem(last=False)
            evicted_size = _shaped_cache_sizes.pop(evicted_id, 0)
            _shaped_meta.pop(evicted_id, None)
            _shaped_cache_bytes = max(0, _shaped_cache_bytes - evicted_size)


def _fetch_and_shape(
    trace_id: str, base_url: str, settle_seconds: float, now: float
) -> ShapedEntry | None:
    """Fetch a trace, shape it, and memoize the shape with its settle state.

    Returns the shaped entry, or ``None`` if the fetch failed (the id is then
    retried — or its prior cached shape re-served — on a later render). The
    shape is cached whether or not the trace has settled; an unsettled trace's
    entry is refreshed on its next bounded re-fetch as later spans arrive.
    """
    trace, settled = _fetch_trace_with_settled(trace_id, base_url, now, settle_seconds)
    if trace is None:
        return None
    entry = extract_trace_rows(trace)
    _store_shaped(trace_id, entry, settled, now)
    return entry


def _iter_window_shaped(
    base_url: str | None = None,
) -> Iterator[tuple[dict[str, str | int], list[dict[str, str | float]], bool]]:
    """Yield ``(trace_info, rows, is_new)`` for the COMPLETE enumerated window.

    A trace is (re-)fetched this render only if it is not yet cached, or it is
    still in-flight (unsettled) and its last fetch is older than
    :func:`_trace_refetch_interval`. Everything else — settled shapes, and
    unsettled shapes fetched recently — is served straight from the shaped cache
    with ``is_new=False`` and costs no Tempo round-trip. This bounded re-fetch
    cadence is what keeps render duration flat under load: an in-flight sharded
    trace is re-read at most once per interval (picking up its later xdist-worker
    spans), not on every render. Fetches are concurrent and capped at
    :func:`_max_new_fetch_per_render`; any id past the cap is deferred — its
    prior cached shape, if any, is still served so the window stays complete.
    ``is_new`` (set for traces actually fetched this render) lets the caller emit
    the high-cardinality per-test series only when a trace's rows may have
    changed, keeping the payload small while the roll-ups see the whole window.
    """
    if base_url is None:
        base_url = tempo_base_url()
    limit = env_int("PYTEST_TRACE_EXPORTER_LIMIT", DEFAULT_LIMIT)
    lookback = os.getenv("PYTEST_TRACE_EXPORTER_LOOKBACK", DEFAULT_LOOKBACK)
    service_name = os.getenv("PYTEST_TRACE_EXPORTER_SERVICE_NAME", DEFAULT_SERVICE_NAME)
    settle_seconds = _trace_settle_seconds()
    refetch_interval = _trace_refetch_interval()
    now = time.time()

    end = int(now)
    start = end - parse_lookback_seconds(lookback)
    trace_ids, truncated = search_all_trace_ids(
        base_url, service_name, start, end, limit
    )

    # Partition the window. ``need_fetch`` = never seen, or in-flight and due for
    # a bounded re-read; ``have_entry`` = any id we hold a (possibly stale) shape
    # for, so a deferred or fetch-failed id can still be served from cache.
    need_fetch: list[str] = []
    have_entry: set[str] = set()
    with _shaped_cache_lock:
        for trace_id in trace_ids:
            entry = _shaped_cache.get(trace_id)
            if entry is None:
                need_fetch.append(trace_id)
                continue
            have_entry.add(trace_id)
            settled, last_fetch = _shaped_meta.get(trace_id, (False, 0.0))
            if not settled and (now - last_fetch) >= refetch_interval:
                need_fetch.append(trace_id)
            else:
                _shaped_cache.move_to_end(trace_id)

    max_new = _max_new_fetch_per_render()
    to_fetch = need_fetch[:max_new]
    to_fetch_set = set(to_fetch)
    deferred = len(need_fetch) - len(to_fetch)

    # Serve from cache every windowed trace we are NOT re-fetching this render —
    # settled, fresh-unsettled, and stale-but-deferred alike (a stale shape keeps
    # the window complete until a later render refreshes it).
    serve_cached = [
        tid for tid in trace_ids if tid in have_entry and tid not in to_fetch_set
    ]

    with _enumeration_lock:
        global _last_enumeration_total, _last_enumeration_truncated
        global _last_enumeration_deferred
        _last_enumeration_total = len(trace_ids)
        _last_enumeration_truncated = truncated
        _last_enumeration_deferred = deferred

    # Yield the cache hits first (no I/O), then stream the fetched ones.
    for trace_id in serve_cached:
        with _shaped_cache_lock:
            entry = _shaped_cache.get(trace_id)
        if entry is not None:
            yield entry[0], entry[1], False

    if not to_fetch:
        return

    workers = max(
        1,
        min(
            env_int(
                "PYTEST_TRACE_EXPORTER_FETCH_CONCURRENCY", DEFAULT_FETCH_CONCURRENCY
            ),
            len(to_fetch),
        ),
    )
    pending = iter(to_fetch)
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="trace-fetch"
    ) as pool:
        inflight = {
            pool.submit(_fetch_and_shape, tid, base_url, settle_seconds, now): tid
            for tid in islice(pending, workers)
        }
        while inflight:
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for future in done:
                trace_id = inflight.pop(future)
                nxt = next(pending, None)
                if nxt is not None:
                    inflight[
                        pool.submit(
                            _fetch_and_shape, nxt, base_url, settle_seconds, now
                        )
                    ] = nxt
                try:
                    entry = future.result()
                except Exception:
                    entry = None
                if entry is not None:
                    yield entry[0], entry[1], True
                elif trace_id in have_entry:
                    # Fetch failed but we hold a prior shape — serve it (stale,
                    # not new) rather than dropping the trace from the window.
                    with _shaped_cache_lock:
                        cached = _shaped_cache.get(trace_id)
                    if cached is not None:
                        yield cached[0], cached[1], False


def fetch_resource_records() -> list[dict[str, str | float | int]]:
    resource_metrics_file = Path(
        os.getenv("PYTEST_RESOURCE_METRICS_FILE", DEFAULT_RESOURCE_METRICS_FILE)
    )
    if not resource_metrics_file.exists():
        return []

    records: list[dict[str, str | float | int]] = []
    with resource_metrics_file.open(encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def github_pr_html_url(repository: str, pr: str) -> str:
    if not repository or not pr:
        return ""
    return f"https://github.com/{repository}/pull/{quote(pr, safe='')}"


def repository_from_pr_url(pr_url: str) -> str:
    parsed = urlparse(pr_url)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 4 and parts[2] == "pull":
        return f"{parts[0]}/{parts[1]}"
    return ""


def github_api_token() -> str | None:
    for env_name in ("PYTEST_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return None


def github_cache_ttl_seconds() -> float:
    raw = os.getenv("PYTEST_TRACE_EXPORTER_GITHUB_CACHE_SECONDS")
    if raw is None or raw == "":
        return DEFAULT_GITHUB_CACHE_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_GITHUB_CACHE_SECONDS


def parse_github_timestamp(value: str) -> float | None:
    """Parse a GitHub ISO-8601 timestamp (e.g. '2024-01-02T03:04:05Z') to epoch seconds."""
    if not value:
        return None
    cleaned = value.replace("Z", "+00:00")
    try:
        from datetime import datetime

        return datetime.fromisoformat(cleaned).timestamp()
    except ValueError:
        return None


def _github_api_get(api_url: str) -> object:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "transformersci-trace-exporter",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = github_api_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(api_url, headers=headers)
    with urlopen(request, timeout=5) as response:
        return json.load(response)


def fetch_github_pr_reviews(repository: str, pr: str) -> list[str]:
    """Return GitHub logins that have submitted a review on the PR.

    Used to enrich `pytest_pr_info` so the dashboard can show actual reviewers
    in addition to pending `requested_reviewers`.
    """
    api_base_url = os.getenv("PYTEST_GITHUB_API_URL", DEFAULT_GITHUB_API_URL).rstrip(
        "/"
    )
    api_url = (
        f"{api_base_url}/repos/{quote(repository, safe='/')}/pulls/"
        f"{quote(pr, safe='')}/reviews"
    )
    try:
        payload = _github_api_get(api_url)
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    logins: list[str] = []
    seen: set[str] = set()
    for review in payload:
        if not isinstance(review, dict):
            continue
        user = review.get("user")
        login = user.get("login") if isinstance(user, dict) else None
        if isinstance(login, str) and login and login not in seen:
            seen.add(login)
            logins.append(login)
    return logins


def _emojize(text: str) -> str:
    """Convert GitHub-style emoji shortcodes to Unicode glyphs.

    PR titles and commit subjects come straight from GitHub, which stores
    aliases like ``:rotating_light:``. Grafana has no shortcode parser and
    renders them verbatim, so we expand them here — at the source — meaning
    every panel and all stored history show the real 🚨. No-ops when the
    optional ``emoji`` dependency is missing or there's no shortcode to expand.
    """
    if not text or _emoji is None or ":" not in text:
        return text
    return _emoji.emojize(text, language="alias")


def fetch_github_pr_info(repository: str, pr: str) -> dict[str, str]:
    api_base_url = os.getenv("PYTEST_GITHUB_API_URL", DEFAULT_GITHUB_API_URL).rstrip(
        "/"
    )
    api_url = (
        f"{api_base_url}/repos/{quote(repository, safe='/')}/pulls/{quote(pr, safe='')}"
    )
    payload = _github_api_get(api_url)
    if not isinstance(payload, dict):
        raise ValueError("GitHub API returned a non-object payload")

    user = payload.get("user")
    author = user.get("login") if isinstance(user, dict) else ""
    html_url = payload.get("html_url")
    title = payload.get("title")
    state = payload.get("state")
    head = payload.get("head")
    commit_sha = head.get("sha") if isinstance(head, dict) else ""
    created_at = payload.get("created_at")

    pending: list[str] = []
    requested = payload.get("requested_reviewers")
    if isinstance(requested, list):
        for entry in requested:
            if isinstance(entry, dict):
                login = entry.get("login")
                if isinstance(login, str) and login:
                    pending.append(login)
    submitted = fetch_github_pr_reviews(repository, pr)
    seen: set[str] = set()
    merged: list[str] = []
    for login in submitted + pending:
        if login not in seen:
            seen.add(login)
            merged.append(login)
    reviewers = ",".join(merged)

    return {
        "author": author if isinstance(author, str) else "",
        "commit_sha": commit_sha if isinstance(commit_sha, str) else "",
        "created_at": created_at if isinstance(created_at, str) else "",
        "html_url": html_url
        if isinstance(html_url, str)
        else github_pr_html_url(repository, pr),
        "reviewers": reviewers,
        "state": state if isinstance(state, str) else "",
        "title": _emojize(title) if isinstance(title, str) else "",
    }


_pr_info_cache_lock = threading.Lock()
_cached_pr_info: dict[tuple[str, str], tuple[float, dict[str, str]]] = {}


def fetch_github_pr_info_cached(repository: str, pr: str) -> dict[str, str]:
    key = (repository, pr)
    ttl = github_cache_ttl_seconds()
    now = time.monotonic()
    with _pr_info_cache_lock:
        cached = _cached_pr_info.get(key)
        if cached is not None and ttl > 0:
            cached_at, payload = cached
            if now - cached_at < ttl:
                return dict(payload)

    fallback = {
        "author": "",
        "commit_sha": "",
        "created_at": "",
        "html_url": github_pr_html_url(repository, pr),
        "reviewers": "",
        "state": "",
        "title": "",
    }
    try:
        payload = fetch_github_pr_info(repository, pr)
    except Exception:
        payload = fallback

    with _pr_info_cache_lock:
        _cached_pr_info[key] = (now, dict(payload))
    return dict(payload)


def github_commit_html_url(repository: str, sha: str) -> str:
    if not repository or not sha:
        return ""
    return f"https://github.com/{repository}/commit/{quote(sha, safe='')}"


def fetch_github_commit_message(repository: str, sha: str) -> str:
    """Return the first line (subject) of a commit's message from GitHub.

    Mirrors :func:`fetch_github_pr_info` but hits the commits endpoint so the
    overview's main-branch run table can show a human-readable commit subject
    instead of a bare run id. Returns "" when the lookup fails.
    """
    api_base_url = os.getenv("PYTEST_GITHUB_API_URL", DEFAULT_GITHUB_API_URL).rstrip(
        "/"
    )
    api_url = (
        f"{api_base_url}/repos/{quote(repository, safe='/')}/commits/"
        f"{quote(sha, safe='')}"
    )
    try:
        payload = _github_api_get(api_url)
    except Exception:
        return ""
    if not isinstance(payload, dict):
        return ""
    commit = payload.get("commit")
    message = commit.get("message") if isinstance(commit, dict) else ""
    if not isinstance(message, str):
        return ""
    # Commit messages are multi-line (subject + body); the table only wants the
    # subject, so keep the first non-empty line.
    return _emojize(message.strip().splitlines()[0]) if message.strip() else ""


_commit_msg_cache_lock = threading.Lock()
_cached_commit_msg: dict[tuple[str, str], tuple[float, str]] = {}


def fetch_github_commit_message_cached(repository: str, sha: str) -> str:
    key = (repository, sha)
    ttl = github_cache_ttl_seconds()
    now = time.monotonic()
    with _commit_msg_cache_lock:
        cached = _cached_commit_msg.get(key)
        if cached is not None and ttl > 0:
            cached_at, message = cached
            if now - cached_at < ttl:
                return message

    message = fetch_github_commit_message(repository, sha)

    with _commit_msg_cache_lock:
        _cached_commit_msg[key] = (now, message)
    return message


def latest_trace(traces: list[dict]) -> dict | None:
    if not traces:
        return None
    return max(traces, key=trace_start_time)


def extract_trace_rows(
    trace: dict,
) -> tuple[dict[str, str | int], list[dict[str, str | float]]]:
    trace_id = trace.get("traceID")
    spans = trace.get("spans", [])
    processes = trace.get("processes", {})

    if (
        not isinstance(trace_id, str)
        or not isinstance(spans, list)
        or not isinstance(processes, dict)
    ):
        return {
            "end_time": 0,
            "latest_start_time": 0,
            "run_id": "unknown",
            "start_time": 0,
            "trace_id": "unknown",
        }, []

    process_run_id = ""
    process_job = ""
    process_provider = ""
    process_pr = ""
    process_pr_url = ""
    process_repository = ""
    process_commit_sha = ""
    service_name = ""
    end_time = 0
    start_time = 0
    latest_start_time = 0
    rows: list[dict[str, str | float]] = []

    for span in spans:
        if not isinstance(span, dict):
            continue

        span_start_time = int(span.get("startTime", 0))
        span_end_time = span_start_time + int(span.get("duration", 0))
        if start_time == 0 or span_start_time < start_time:
            start_time = span_start_time
        if span_start_time > latest_start_time:
            latest_start_time = span_start_time
        if span_end_time > end_time:
            end_time = span_end_time

        process = processes.get(span.get("processID"), {})
        process_tags = (
            tag_map(process.get("tags", [])) if isinstance(process, dict) else {}
        )
        service_name = (
            process.get("serviceName", service_name)
            if isinstance(process, dict)
            else service_name
        )
        process_run_id = process_tags.get(
            "transformers.test.run.id",
            process_tags.get("cicd.pipeline.run.id", process_run_id),
        )
        process_job = process_tags.get(
            "transformers.test.job",
            process_tags.get("transformers.test.suite", process_job),
        )
        process_provider = process_tags.get(
            "transformers.test.provider", process_provider
        )
        process_pr = process_tags.get("vcs.change.id", process_pr)
        # Push events (e.g. merges to main) carry no vcs.change.id, so fall back
        # to the branch name. Keeps main-branch runs from collapsing into a
        # single pr="none" bucket in the dashboards.
        if not process_pr:
            process_pr = process_tags.get("vcs.ref.head.name", process_pr)
        process_pr_url = process_tags.get("vcs.change.url", process_pr_url)
        process_repository = process_tags.get("vcs.repository.name", process_repository)
        # The head commit SHA (GITHUB_SHA) rides along on every span's process
        # tags. We promote it to run scope so the run-info metric can resolve a
        # commit message from GitHub for main-branch (push) runs.
        process_commit_sha = process_tags.get(
            "vcs.ref.head.revision", process_commit_sha
        )

        span_tags = tag_map(span.get("tags", []))
        nodeid = span_tags.get("pytest.nodeid")
        span_type = span_tags.get("pytest.span_type")
        operation_name = span.get("operationName")
        if nodeid is None or span_type != "test" or operation_name != nodeid:
            continue

        node_parts = split_pytest_nodeid(nodeid)
        exc_type, exc_stacktrace = extract_exception_info(span)
        rows.append(
            {
                "duration_seconds": int(span.get("duration", 0)) / 1_000_000,
                "exception_type": exc_type,
                # The capped stacktrace is consumed here for test_line only; no
                # metric emits it, so it is deliberately not retained in the row
                # (it would otherwise bloat the kept rows under high failure
                # volume — the exporter must stay under ~1G at high volume).
                "pr": process_pr or "none",
                "provider": process_provider or "unknown",
                "run_id": process_run_id or trace_id,
                "service_name": service_name or "unknown",
                "status_code": span_tags.get("otel.status_code", "UNSET"),
                "test_class": node_parts["test_class"],
                "test_function": node_parts["test_function"],
                "test_line": extract_test_line(exc_stacktrace, nodeid),
                "test_job": process_job or "unknown",
                "test_module": node_parts["test_module"],
                "test_nodeid": nodeid,
                "trace_id": trace_id,
            }
        )

    if not process_repository and process_pr_url:
        process_repository = repository_from_pr_url(process_pr_url)
    if not process_pr_url and process_repository and process_pr:
        process_pr_url = github_pr_html_url(process_repository, process_pr)

    return {
        "commit_sha": process_commit_sha,
        "end_time": end_time,
        "latest_start_time": latest_start_time,
        "pr": process_pr or "none",
        "pr_url": process_pr_url,
        "provider": process_provider or "unknown",
        "repository": process_repository,
        "run_id": process_run_id or trace_id,
        "service_name": service_name or "unknown",
        "start_time": start_time,
        "test_job": process_job or "unknown",
        "trace_id": trace_id,
    }, rows


def _precompute_trace_rows(
    traces: list[dict],
) -> list[tuple[dict[str, str | int], list[dict[str, str | float]]]]:
    return [extract_trace_rows(trace) for trace in traces]


def latest_trace_info_lines(trace_info: dict[str, str | int]) -> list[str]:
    """Emit the 'latest-trace' pointer markers from an already-shaped trace_info.

    The render works from shaped (trace_info, rows) entries — the raw trace for
    a cache hit is long gone — so this takes the shaped info directly rather
    than re-deriving it from a raw trace.
    """
    info_labels = {
        "pr": str(trace_info["pr"]),
        "run_id": str(trace_info["run_id"]),
        "test_job": str(trace_info["test_job"]),
        "provider": str(trace_info["provider"]),
        "service_name": str(trace_info["service_name"]),
        "trace_id": str(trace_info["trace_id"]),
    }
    return [
        "# HELP pytest_latest_trace_info Metadata for the latest pytest trace visible to the exporter.",
        "# TYPE pytest_latest_trace_info gauge",
        f"pytest_latest_trace_info{metric_labels(info_labels)} 1",
        f"pytest_latest_trace_start_time_seconds{metric_labels(info_labels)} {int(trace_info['latest_start_time']) / 1_000_000:.6f}",
    ]


def extract_latest_trace_metrics(trace: dict) -> list[str]:
    """Emit the small set of 'latest-trace' markers.

    Per-test duration is now emitted for every trace in the lookback via
    ``extract_per_run_metrics`` — this function only emits the pointer
    markers used by a few legacy panels.
    """
    trace_info, _ = extract_trace_rows(trace)
    info_labels = {
        "pr": str(trace_info["pr"]),
        "run_id": str(trace_info["run_id"]),
        "test_job": str(trace_info["test_job"]),
        "provider": str(trace_info["provider"]),
        "service_name": str(trace_info["service_name"]),
        "trace_id": str(trace_info["trace_id"]),
    }
    return [
        "# HELP pytest_latest_trace_info Metadata for the latest pytest trace visible to the exporter.",
        "# TYPE pytest_latest_trace_info gauge",
        f"pytest_latest_trace_info{metric_labels(info_labels)} 1",
        f"pytest_latest_trace_start_time_seconds{metric_labels(info_labels)} {int(trace_info['latest_start_time']) / 1_000_000:.6f}",
    ]


def extract_pr_last_failure_metrics(
    traces: list[dict],
    *,
    _extracted: list[tuple[dict[str, str | int], list[dict[str, str | float]]]]
    | None = None,
) -> list[str]:
    """Emit one ``pytest_pr_last_failure_info`` series per PR — only when the
    most recent *run* (across all of its job traces) actually has a failure.

    Rows from all traces sharing the same ``(pr, run_id)`` are merged together
    so a workflow run that splits across multiple jobs is treated as one
    run. If the latest run for a PR finished clean, no sample is emitted, so
    the dashboard's Last Error panel hides entirely instead of surfacing a
    stale failure from an earlier run.
    """
    extracted = _extracted if _extracted is not None else _precompute_trace_rows(traces)

    runs: dict[tuple[str, str], tuple[int, list[dict[str, str | float]]]] = {}
    for trace_info, rows in extracted:
        if not rows:
            continue
        pr = str(trace_info.get("pr", "none"))
        run_id = str(trace_info.get("run_id", ""))
        end_time = int(trace_info.get("end_time", 0) or 0)
        key = (pr, run_id)
        existing = runs.get(key)
        if existing is None:
            runs[key] = (end_time, list(rows))
        else:
            runs[key] = (max(existing[0], end_time), existing[1] + list(rows))

    latest_run_per_pr: dict[str, tuple[int, list[dict[str, str | float]]]] = {}
    for (pr, _run_id), (end_time, merged_rows) in runs.items():
        existing = latest_run_per_pr.get(pr)
        if existing is None or end_time > existing[0]:
            latest_run_per_pr[pr] = (end_time, merged_rows)

    per_pr: dict[str, tuple[int, dict[str, str]]] = {}
    for pr, (latest_time, rows) in latest_run_per_pr.items():
        for row in rows:
            if str(row["status_code"]) != "ERROR":
                continue
            existing = per_pr.get(pr)
            if existing is None or latest_time >= existing[0]:
                per_pr[pr] = (
                    latest_time,
                    {
                        "pr": pr,
                        "service_name": str(row["service_name"]),
                        "provider": str(row["provider"]),
                        "run_id": str(row["run_id"]),
                        "test_job": str(row["test_job"]),
                        "test_function": str(row["test_function"]),
                        "test_module": str(row["test_module"]),
                        "test_class": str(row["test_class"]),
                        "test_line": str(row.get("test_line", "")),
                        "test_nodeid": str(row["test_nodeid"]),
                        "exception_type": str(row.get("exception_type", ""))
                        or "unknown",
                        # The stacktrace itself is intentionally NOT a label —
                        # it lives in the trace in Tempo, which the dashboard
                        # links to by trace_id. Baking ≤4000-char stacktraces
                        # into labels exploded TSDB cardinality on every code
                        # edit (and caused the group_left 422s).
                        "trace_id": str(row["trace_id"]),
                    },
                )

    lines = [
        "# HELP pytest_pr_last_failure_info Metadata of the most recent failing run in a PR.",
        "# TYPE pytest_pr_last_failure_info gauge",
    ]
    for pr, (_, labels) in sorted(per_pr.items()):
        lines.append(f"pytest_pr_last_failure_info{metric_labels(labels)} 1")
    return lines


def extract_pr_info_metrics(
    traces: list[dict],
    *,
    _extracted: list[tuple[dict[str, str | int], list[dict[str, str | float]]]]
    | None = None,
    _metadata_fetcher: Callable[[str, str], dict[str, str]] | None = None,
) -> list[str]:
    """Emit one ``pytest_pr_info`` series per PR with GitHub metadata.

    The metric is intentionally small and PR-scoped so Grafana can render
    stable PR metadata without querying GitHub directly from the panel.
    """
    extracted = _extracted if _extracted is not None else _precompute_trace_rows(traces)
    metadata_fetcher = _metadata_fetcher or fetch_github_pr_info_cached
    lines = [
        "# HELP pytest_pr_info Metadata fetched for a pull request.",
        "# TYPE pytest_pr_info gauge",
    ]
    created_lines: list[str] = []
    state_lines: list[str] = []
    best_by_pr: dict[tuple[str, str], tuple[int, dict[str, str]]] = {}
    for trace_info, _rows in extracted:
        pr = str(trace_info.get("pr", "none"))
        if not pr.isdigit():
            continue
        service_name = str(trace_info.get("service_name", "unknown"))
        repository = str(trace_info.get("repository", ""))
        pr_url = str(trace_info.get("pr_url", ""))
        if not repository and pr_url:
            repository = repository_from_pr_url(pr_url)
        key = (service_name, pr)
        candidate = {
            "pr": pr,
            "pr_url": pr_url,
            "repository": repository,
            "service_name": service_name,
        }
        trace_score = int(trace_info.get("latest_start_time", 0) or 0)
        if repository:
            trace_score += 1
        if pr_url:
            trace_score += 1
        existing = best_by_pr.get(key)
        if existing is None or trace_score >= existing[0]:
            best_by_pr[key] = (trace_score, candidate)

    for (_service_name, _pr), (_score, candidate) in sorted(best_by_pr.items()):
        pr = candidate["pr"]
        repository = candidate["repository"]
        pr_url = candidate["pr_url"]
        service_name = candidate["service_name"]
        metadata = {
            "author": "",
            "commit_sha": "",
            "created_at": "",
            "html_url": pr_url or github_pr_html_url(repository, pr),
            "reviewers": "",
            "state": "",
            "title": "",
        }
        if repository:
            metadata.update(metadata_fetcher(repository, pr))

        labels = {
            "author": str(metadata.get("author", "")),
            "commit_sha": str(metadata.get("commit_sha", "")) or "main",
            "html_url": str(metadata.get("html_url", "")),
            "pr": pr,
            "repository": repository,
            "reviewers": str(metadata.get("reviewers", "")),
            "service_name": service_name,
            "state": str(metadata.get("state", "")),
            "title": str(metadata.get("title", "")),
        }
        lines.append(f"pytest_pr_info{metric_labels(labels)} 1")

        # Numeric state gauge: ONE series per PR whose VALUE is the state
        # (open=1, closed=0), so the dashboard can last_over_time() it to get the
        # current state without juggling a multi-valued `state` label. Skipped
        # when the state is unknown (GitHub lookup failed) so the column stays
        # blank rather than implying a state.
        state = str(metadata.get("state", "")).lower()
        if state in ("open", "closed"):
            state_labels = {
                "pr": pr,
                "repository": repository,
                "service_name": service_name,
            }
            state_lines.append(
                f"pytest_pr_state{metric_labels(state_labels)} "
                f"{1 if state == 'open' else 0}"
            )

        created_ts = parse_github_timestamp(str(metadata.get("created_at", "")))
        if created_ts is not None:
            created_labels = {
                "pr": pr,
                "repository": repository,
                "service_name": service_name,
            }
            created_lines.append(
                f"pytest_pr_created_at_seconds{metric_labels(created_labels)} {created_ts:.0f}"
            )

    if created_lines:
        lines.append(
            "# HELP pytest_pr_created_at_seconds Unix timestamp the PR was created at."
        )
        lines.append("# TYPE pytest_pr_created_at_seconds gauge")
        lines.extend(created_lines)
    if state_lines:
        lines.append("# HELP pytest_pr_state PR state as a value: open=1, closed=0.")
        lines.append("# TYPE pytest_pr_state gauge")
        lines.extend(state_lines)
    return lines


def extract_run_info_metrics(
    traces: list[dict] | None = None,
    *,
    _extracted: list[tuple[dict[str, str | int], list[dict[str, str | float]]]]
    | None = None,
    _commit_fetcher: Callable[[str, str], str] | None = None,
) -> list[str]:
    """Emit one ``pytest_run_info`` series per run carrying its commit subject.

    The run roll-up metrics are intentionally network-free and keyed only by the
    stable run identity, so commit metadata lives here instead. We resolve the
    head commit SHA (promoted from each run's trace tags) to a one-line commit
    subject via the GitHub API (cached), letting the overview's main-branch run
    table show a human-readable message column next to the run id.
    """
    extracted = (
        _extracted if _extracted is not None else _precompute_trace_rows(traces or [])
    )
    commit_fetcher = _commit_fetcher or fetch_github_commit_message_cached

    # One candidate per run identity, preferring the trace with both a commit SHA
    # and a repository (needed for the GitHub lookup) and the latest start time.
    best_by_run: dict[tuple[str, str, str, str], tuple[int, dict[str, str]]] = {}
    for trace_info, rows in extracted:
        if not rows:
            continue
        run_key = (
            str(trace_info.get("service_name", "unknown")),
            str(trace_info.get("provider", "unknown")),
            str(trace_info.get("pr", "none")),
            str(trace_info.get("run_id", trace_info.get("trace_id", "unknown"))),
        )
        repository = str(trace_info.get("repository", ""))
        pr_url = str(trace_info.get("pr_url", ""))
        if not repository and pr_url:
            repository = repository_from_pr_url(pr_url)
        candidate = {
            "commit_sha": str(trace_info.get("commit_sha", "")),
            "repository": repository,
        }
        score = int(trace_info.get("latest_start_time", 0) or 0)
        if candidate["commit_sha"]:
            score += 1
        if repository:
            score += 1
        existing = best_by_run.get(run_key)
        if existing is None or score >= existing[0]:
            best_by_run[run_key] = (score, candidate)

    lines = [
        "# HELP pytest_run_info Commit metadata for a pytest run.",
        "# TYPE pytest_run_info gauge",
    ]
    for (service_name, provider, pr, run_id), (_score, candidate) in sorted(
        best_by_run.items()
    ):
        commit_sha = candidate["commit_sha"]
        repository = candidate["repository"]
        commit_message = ""
        if repository and commit_sha:
            commit_message = commit_fetcher(repository, commit_sha)
        # Fall back to the short SHA when no message is available (no GitHub
        # token, lookup failure, or missing repository) so the dashboard's
        # Commit column always shows something clickable rather than rendering
        # empty. The html_url label still points at the commit either way.
        if not commit_message and commit_sha:
            commit_message = commit_sha[:12]
        labels = {
            "commit_message": commit_message,
            "commit_sha": commit_sha,
            "html_url": github_commit_html_url(repository, commit_sha),
            "pr": pr,
            "provider": provider,
            "run_id": run_id,
            "service_name": service_name,
        }
        lines.append(f"pytest_run_info{metric_labels(labels)} 1")
    return lines


def extract_per_test_duration_metrics(
    traces: list[dict] | None = None,
    *,
    _extracted: list[tuple[dict[str, str | int], list[dict[str, str | float]]]]
    | None = None,
) -> list[str]:
    """Emit one ``pytest_test_duration_seconds`` sample per test span.

    Window-based: each test span seen in the lookback is emitted. Per-test
    series never decay (a test's last status is correct and retained by
    Prometheus), so this stays on the in-window set.
    """
    extracted = (
        _extracted if _extracted is not None else _precompute_trace_rows(traces or [])
    )
    lines = [
        "# HELP pytest_test_duration_seconds Duration of each pytest test span, labeled with run_id and trace_id.",
        "# TYPE pytest_test_duration_seconds gauge",
    ]
    for _trace_info, rows in extracted:
        for row in rows:
            test_labels = {
                "pr": str(row["pr"]),
                "test_job": str(row["test_job"]),
                "provider": str(row["provider"]),
                "run_id": str(row["run_id"]),
                "service_name": str(row["service_name"]),
                "status_code": str(row["status_code"]),
                "test_class": str(row["test_class"]),
                "test_function": str(row["test_function"]),
                "test_module": str(row["test_module"]),
                "test_nodeid": str(row["test_nodeid"]),
                "trace_id": str(row["trace_id"]),
            }
            lines.append(
                f"pytest_test_duration_seconds{metric_labels(test_labels)} {float(row['duration_seconds']):.9f}"
            )
    return lines


def extract_ci_runner_execution_metrics(
    traces: list[dict] | None = None,
    *,
    _extracted: list[tuple[dict[str, str | int], list[dict[str, str | float]]]]
    | None = None,
) -> list[str]:
    """Emit one low-cardinality series per pytest trace / runner execution.

    GitHub Actions matrix shards share the same workflow run id and logical job
    name, but each shard produces a distinct trace. Counting this metric is much
    cheaper than scanning every per-test duration series and deduplicating back
    to trace ids in Grafana.
    """
    extracted = (
        _extracted if _extracted is not None else _precompute_trace_rows(traces or [])
    )
    lines = [
        "# HELP pytest_ci_runner_execution_info One observed CI runner execution that emitted a pytest trace.",
        "# TYPE pytest_ci_runner_execution_info gauge",
    ]
    for trace_info, rows in extracted:
        if not rows:
            continue
        labels = {
            "pr": str(trace_info.get("pr", "none")),
            "provider": str(trace_info.get("provider", "unknown")),
            "run_id": str(
                trace_info.get("run_id", trace_info.get("trace_id", "unknown"))
            ),
            "service_name": str(trace_info.get("service_name", "unknown")),
            "test_job": str(trace_info.get("test_job", "unknown")),
            "trace_id": str(trace_info.get("trace_id", "unknown")),
        }
        lines.append(f"pytest_ci_runner_execution_info{metric_labels(labels)} 1")
    return lines


def extract_run_rollup_metrics(
    traces: list[dict] | None = None,
    *,
    _extracted: list[tuple[dict[str, str | int], list[dict[str, str | float]]]]
    | None = None,
) -> list[str]:
    """Emit one roll-up series per workflow-level pytest run.

    The caller is expected to pass the run's *complete* trace set (see
    :func:`settled_runs_complete_extracted`), not just the traces currently in
    the lookback window — otherwise the roll-up decays to a partial value as a
    run's job traces age out of the window one by one (which froze a wrong
    "100% pass" into Prometheus). Feeds the PR "Past Runs" table and the
    overview run/PR panels.
    """
    extracted = (
        _extracted if _extracted is not None else _precompute_trace_rows(traces or [])
    )
    lines = [
        "# HELP pytest_run_start_time_seconds Start time (unix seconds) of a pytest run.",
        "# TYPE pytest_run_start_time_seconds gauge",
        "# HELP pytest_run_end_time_seconds End time (unix seconds) of a pytest run.",
        "# TYPE pytest_run_end_time_seconds gauge",
        "# HELP pytest_run_total_tests Number of tests recorded in a pytest run.",
        "# TYPE pytest_run_total_tests gauge",
        "# HELP pytest_run_failed_tests Number of failing tests in a pytest run.",
        "# TYPE pytest_run_failed_tests gauge",
        "# HELP pytest_run_duration_seconds Total duration (sum of test span durations) of a pytest run.",
        "# TYPE pytest_run_duration_seconds gauge",
        "# HELP pytest_run_job_count Number of distinct jobs that contributed tests to a pytest run.",
        "# TYPE pytest_run_job_count gauge",
        "# HELP pytest_run_job_member_info Whether a job contributed tests to a pytest run.",
        "# TYPE pytest_run_job_member_info gauge",
        "# HELP pytest_run_job_total_tests Number of tests recorded in one job of a pytest run.",
        "# TYPE pytest_run_job_total_tests gauge",
        "# HELP pytest_run_job_passed_tests Number of passing tests recorded in one job of a pytest run.",
        "# TYPE pytest_run_job_passed_tests gauge",
        "# HELP pytest_run_job_failed_tests Number of failing tests recorded in one job of a pytest run.",
        "# TYPE pytest_run_job_failed_tests gauge",
        "# HELP pytest_run_job_duration_seconds Total duration (sum of test span durations) for one job of a pytest run.",
        "# TYPE pytest_run_job_duration_seconds gauge",
    ]
    run_aggregates: dict[tuple[str, str, str, str], dict[str, object]] = {}
    job_aggregates: dict[tuple[str, str, str, str, str], dict[str, object]] = {}

    for trace_info, rows in extracted:
        if not rows:
            continue

        total = len(rows)
        failed = sum(1 for r in rows if str(r["status_code"]) == "ERROR")
        total_duration = fsum(float(r["duration_seconds"]) for r in rows)
        run_key = (
            str(trace_info.get("service_name", "unknown")),
            str(trace_info.get("provider", "unknown")),
            str(trace_info.get("pr", "none")),
            str(trace_info.get("run_id", trace_info.get("trace_id", "unknown"))),
        )
        if run_key not in run_aggregates:
            run_aggregates[run_key] = {
                "end_time": int(trace_info.get("end_time", 0) or 0),
                "failed": 0,
                "start_time": int(trace_info.get("start_time", 0) or 0),
                "job_names": set(),
                "total": 0,
                "total_duration": 0.0,
            }

        aggregate = run_aggregates[run_key]
        aggregate["failed"] = int(aggregate["failed"]) + failed
        aggregate["total"] = int(aggregate["total"]) + total
        aggregate["total_duration"] = (
            float(aggregate["total_duration"]) + total_duration
        )
        aggregate["end_time"] = max(
            int(aggregate["end_time"]), int(trace_info.get("end_time", 0) or 0)
        )

        trace_start_time = int(trace_info.get("start_time", 0) or 0)
        aggregate_start_time = int(aggregate["start_time"])
        if aggregate_start_time == 0 or (
            trace_start_time != 0 and trace_start_time < aggregate_start_time
        ):
            aggregate["start_time"] = trace_start_time

        job_names = aggregate["job_names"]
        assert isinstance(job_names, set)
        job_names.add(str(trace_info.get("test_job", "unknown")))

        job_key = run_key + (str(trace_info.get("test_job", "unknown")),)
        if job_key not in job_aggregates:
            job_aggregates[job_key] = {
                "failed": 0,
                "total": 0,
                "total_duration": 0.0,
            }
        job_aggregate = job_aggregates[job_key]
        job_aggregate["failed"] = int(job_aggregate["failed"]) + failed
        job_aggregate["total"] = int(job_aggregate["total"]) + total
        job_aggregate["total_duration"] = (
            float(job_aggregate["total_duration"]) + total_duration
        )

    for (service_name, provider, pr, run_id), aggregate in sorted(
        run_aggregates.items()
    ):
        job_names = sorted(str(job_name) for job_name in aggregate["job_names"])
        total = int(aggregate["total"])
        failed = int(aggregate["failed"])
        total_duration = float(aggregate["total_duration"])
        start_time_seconds = int(aggregate["start_time"]) / 1_000_000
        end_time_seconds = int(aggregate["end_time"]) / 1_000_000
        run_labels = {
            "pr": pr,
            "provider": provider,
            "run_id": run_id,
            "service_name": service_name,
        }
        # Every run-level rollup is emitted as a metric *value* keyed only by the
        # stable run identity. Earlier versions baked the mutable totals
        # (total_tests, failed_tests, job_count, ...) into labels on the
        # start-time metric. A label set IS the series identity in Prometheus, so
        # each re-aggregation while a run's job traces trickled in minted a brand
        # new series; last_over_time(...[range]) then resurrected every stale
        # snapshot and a single run showed up as many identical-run_id rows.
        # Keeping totals as values means the run stays one series whose value
        # simply updates as more of its traces arrive.
        lines.append(
            f"pytest_run_start_time_seconds{metric_labels(run_labels)} {start_time_seconds:.6f}"
        )
        lines.append(
            f"pytest_run_end_time_seconds{metric_labels(run_labels)} {end_time_seconds:.6f}"
        )
        lines.append(f"pytest_run_total_tests{metric_labels(run_labels)} {total}")
        lines.append(f"pytest_run_failed_tests{metric_labels(run_labels)} {failed}")
        lines.append(
            f"pytest_run_duration_seconds{metric_labels(run_labels)} {total_duration:.6f}"
        )
        lines.append(
            f"pytest_run_job_count{metric_labels(run_labels)} {len(job_names)}"
        )
        for job_name in job_names:
            job_labels = dict(run_labels)
            job_labels["test_job"] = job_name
            lines.append(f"pytest_run_job_member_info{metric_labels(job_labels)} 1")

    for (service_name, provider, pr, run_id, test_job), aggregate in sorted(
        job_aggregates.items()
    ):
        total = int(aggregate["total"])
        failed = int(aggregate["failed"])
        job_labels = {
            "pr": pr,
            "provider": provider,
            "run_id": run_id,
            "service_name": service_name,
            "test_job": test_job,
        }
        lines.append(f"pytest_run_job_total_tests{metric_labels(job_labels)} {total}")
        lines.append(
            f"pytest_run_job_passed_tests{metric_labels(job_labels)} {total - failed}"
        )
        lines.append(f"pytest_run_job_failed_tests{metric_labels(job_labels)} {failed}")
        lines.append(
            f"pytest_run_job_duration_seconds{metric_labels(job_labels)} {float(aggregate['total_duration']):.6f}"
        )
    return lines


def extract_per_run_metrics(
    traces: list[dict] | None = None,
    *,
    _extracted: list[tuple[dict[str, str | int], list[dict[str, str | float]]]]
    | None = None,
) -> list[str]:
    """Backward-compatible combination of per-test duration + run roll-ups.

    Computes both from the same (window) set. The live exporter no longer uses
    this — it feeds the roll-up its complete-set view (see
    :func:`settled_runs_complete_extracted`) — but it keeps the original
    single-call behavior for callers/tests that pass one trace list.
    """
    extracted = (
        _extracted if _extracted is not None else _precompute_trace_rows(traces or [])
    )
    return extract_per_test_duration_metrics(
        _extracted=extracted
    ) + extract_run_rollup_metrics(_extracted=extracted)


# ---------------------------------------------------------------------------
# Run membership tracking — lets run roll-ups be computed over a run's COMPLETE
# set of job traces even after some have aged out of the lookback window.
# ---------------------------------------------------------------------------

_run_state_lock = threading.Lock()
# run_id -> set of trace_ids ever seen for that run.
_run_members: dict[str, set[str]] = {}
# run_id -> monotonic time a new trace_id was last added (i.e. last job arrival).
_run_last_growth: dict[str, float] = {}
# Drop membership for runs untouched for this long, to bound memory.
RUN_MEMBERSHIP_TTL_SECONDS = 86400.0


def _run_settle_seconds() -> float:
    """A run is 'complete' once no new job trace has arrived for this long.

    Reuses the trace-settle window: by the time a trace has settled, the run it
    belongs to has had a quiet period, so its roll-up can be emitted as a
    single stable series instead of one churning series per ingestion step.
    """
    return _trace_settle_seconds()


def record_run_membership(
    extracted: list[tuple[dict[str, str | int], list[dict[str, str | float]]]],
    now: float,
) -> None:
    with _run_state_lock:
        for trace_info, _rows in extracted:
            run_id = str(trace_info.get("run_id", ""))
            trace_id = str(trace_info.get("trace_id", ""))
            if not run_id or not trace_id:
                continue
            members = _run_members.setdefault(run_id, set())
            if trace_id not in members:
                members.add(trace_id)
                _run_last_growth[run_id] = now
        # Evict long-idle runs so the maps don't grow without bound.
        stale = [
            run_id
            for run_id, grown in _run_last_growth.items()
            if now - grown > RUN_MEMBERSHIP_TTL_SECONDS
        ]
        for run_id in stale:
            _run_members.pop(run_id, None)
            _run_last_growth.pop(run_id, None)


def settled_runs_complete_extracted(
    window_extracted: list[tuple[dict[str, str | int], list[dict[str, str | float]]]],
    now: float,
    settle_seconds: float,
) -> list[tuple[dict[str, str | int], list[dict[str, str | float]]]]:
    """Return complete per-run extracted rows for runs that are settled.

    For every run with at least one trace in the current window that has not
    received a new trace for ``settle_seconds``, gather all of its member
    traces — the in-window ones plus any aged-out ones still cached — so the
    roll-up reflects the whole run. Aged-out members are resolved from the
    shaped cache (small per-entry, holds the whole window+) and, failing that,
    the raw trace cache. Runs still ingesting are skipped this cycle (so only
    one stable, complete series is ever emitted per run).
    """
    by_trace_id: dict[
        str, tuple[dict[str, str | int], list[dict[str, str | float]]]
    ] = {}
    active_runs: set[str] = set()
    for trace_info, rows in window_extracted:
        trace_id = str(trace_info.get("trace_id", ""))
        run_id = str(trace_info.get("run_id", ""))
        if trace_id:
            by_trace_id[trace_id] = (trace_info, rows)
        if run_id:
            active_runs.add(run_id)

    complete: list[tuple[dict[str, str | int], list[dict[str, str | float]]]] = []
    with _run_state_lock:
        for run_id in active_runs:
            if now - _run_last_growth.get(run_id, now) < settle_seconds:
                continue  # still ingesting — wait until it looks complete
            for trace_id in _run_members.get(run_id, set()):
                entry = by_trace_id.get(trace_id)
                if entry is not None:
                    complete.append(entry)
                    continue
                with _shaped_cache_lock:
                    shaped = _shaped_cache.get(trace_id)
                if shaped is not None:
                    complete.append(shaped)
                    continue
                cached = _trace_cache.get(trace_id)
                if cached is not None:
                    complete.append(extract_trace_rows(cached))
    return complete


def extract_average_metrics(
    traces: list[dict],
    *,
    _extracted: list[tuple[dict[str, str | int], list[dict[str, str | float]]]]
    | None = None,
) -> list[str]:
    extracted = _extracted if _extracted is not None else _precompute_trace_rows(traces)
    # Only pytest_test_last_failure_info is emitted here. The per-test
    # pytest_test_{average_duration_seconds,run_count,failure_count} gauges used
    # to be emitted too, but at real transformers-CI scale (~44k distinct tests
    # per window) they were ~133k series / ~50 MiB of the payload and NO
    # dashboard ever queried them — Grafana already derives the same numbers
    # from pytest_test_duration_seconds (count by status_code, avg by test) and
    # the pytest_run_* rollups. Dropping them keeps the payload (and exporter
    # RSS) bounded. The failure aggregation below is still needed for the
    # last-failure pointer.
    lines = [
        "# HELP pytest_test_last_failure_info Pointer to the most recent failing trace for this pytest test span.",
        "# TYPE pytest_test_last_failure_info gauge",
    ]
    aggregates: dict[tuple[str, str, str, str, str], dict] = {}

    for trace_info, rows in extracted:
        trace_start = int(trace_info.get("latest_start_time", 0) or 0)
        trace_id = str(trace_info.get("trace_id", "unknown"))
        for row in rows:
            key = (
                str(row["service_name"]),
                str(row["test_job"]),
                str(row["pr"]),
                str(row["provider"]),
                str(row["test_nodeid"]),
            )
            if key not in aggregates:
                aggregates[key] = {
                    "durations": [],
                    "failure_count": 0,
                    "last_failure_start_time": 0,
                    "last_failure_trace_id": "",
                    "last_failure_exception_type": "",
                    "test_class": str(row["test_class"]),
                    "test_function": str(row["test_function"]),
                    "test_module": str(row["test_module"]),
                }
            aggregates[key]["durations"].append(float(row["duration_seconds"]))
            if str(row["status_code"]) == "ERROR":
                aggregates[key]["failure_count"] += 1
                if trace_start >= aggregates[key]["last_failure_start_time"]:
                    aggregates[key]["last_failure_start_time"] = trace_start
                    aggregates[key]["last_failure_trace_id"] = trace_id
                    aggregates[key]["last_failure_exception_type"] = (
                        str(row.get("exception_type", "")) or "unknown"
                    )

    for (service_name, test_job, pr, provider, test_nodeid), aggregate in sorted(
        aggregates.items()
    ):
        # Only failing tests produce a line now, so skip the (expensive at ~44k
        # tests) label-building entirely for the passing majority.
        if int(aggregate["failure_count"]) <= 0:
            continue
        labels = {
            "pr": pr,
            "test_job": test_job,
            "provider": provider,
            "service_name": service_name,
            "test_class": str(aggregate["test_class"]),
            "test_function": str(aggregate["test_function"]),
            "test_module": str(aggregate["test_module"]),
            "test_nodeid": test_nodeid,
            "trace_id": str(aggregate["last_failure_trace_id"]),
            # No stacktrace label here — the trace_id pointer is enough for the
            # dashboard to deep-link into the Tempo trace view.
            "exception_type": str(aggregate["last_failure_exception_type"]),
        }
        lines.append(f"pytest_test_last_failure_info{metric_labels(labels)} 1")

    return lines


def extract_average_resource_metrics(
    records: list[dict[str, str | float | int]],
) -> list[str]:
    lines = [
        "# HELP pytest_test_average_cpu_time_seconds Average process CPU time delta across recorded test runs.",
        "# TYPE pytest_test_average_cpu_time_seconds gauge",
        "# HELP pytest_test_average_rss_peak_bytes Average peak RSS across recorded test runs.",
        "# TYPE pytest_test_average_rss_peak_bytes gauge",
        "# HELP pytest_test_average_rss_delta_bytes Average RSS delta across recorded test runs.",
        "# TYPE pytest_test_average_rss_delta_bytes gauge",
        "# HELP pytest_test_average_cuda_peak_allocated_bytes Average peak CUDA allocated bytes across recorded test runs.",
        "# TYPE pytest_test_average_cuda_peak_allocated_bytes gauge",
        "# HELP pytest_test_resource_run_count Number of recorded resource samples for a given test.",
        "# TYPE pytest_test_resource_run_count gauge",
    ]
    aggregates: dict[tuple[str, str, str, str, str], dict[str, str | list[float]]] = {}

    for record in records:
        service_name = str(record.get("service_name", "unknown"))
        test_job = str(record.get("test_job", record.get("test_suite", "unknown")))
        pr = str(record.get("pr", "none"))
        provider = str(record.get("provider", "unknown"))
        test_nodeid = str(record.get("test_nodeid", "unknown"))
        key = (service_name, test_job, pr, provider, test_nodeid)
        if key not in aggregates:
            aggregates[key] = {
                "cpu_time_seconds": [],
                "rss_delta_bytes": [],
                "rss_peak_bytes": [],
                "cuda_peak_allocated_bytes": [],
                "test_class": str(record.get("test_class", "")),
                "test_function": str(record.get("test_function", "")),
                "test_module": str(record.get("test_module", "")),
            }

        aggregate = aggregates[key]
        for metric_name in (
            "cpu_time_seconds",
            "rss_delta_bytes",
            "rss_peak_bytes",
            "cuda_peak_allocated_bytes",
        ):
            value = record.get(metric_name)
            metric_values = aggregate[metric_name]
            assert isinstance(metric_values, list)
            if isinstance(value, (int, float)):
                metric_values.append(float(value))

    for (service_name, test_job, pr, provider, test_nodeid), aggregate in sorted(
        aggregates.items()
    ):
        labels = {
            "pr": pr,
            "test_job": test_job,
            "provider": provider,
            "service_name": service_name,
            "test_class": str(aggregate["test_class"]),
            "test_function": str(aggregate["test_function"]),
            "test_module": str(aggregate["test_module"]),
            "test_nodeid": test_nodeid,
        }
        resource_count = len(aggregate["cpu_time_seconds"])  # type: ignore[arg-type]
        lines.append(
            f"pytest_test_resource_run_count{metric_labels(labels)} {resource_count}"
        )
        for metric_name, prom_name in (
            ("cpu_time_seconds", "pytest_test_average_cpu_time_seconds"),
            ("rss_peak_bytes", "pytest_test_average_rss_peak_bytes"),
            ("rss_delta_bytes", "pytest_test_average_rss_delta_bytes"),
            (
                "cuda_peak_allocated_bytes",
                "pytest_test_average_cuda_peak_allocated_bytes",
            ),
        ):
            metric_values = aggregate[metric_name]
            assert isinstance(metric_values, list)
            if not metric_values:
                continue
            lines.append(
                f"{prom_name}{metric_labels(labels)} {fsum(metric_values) / len(metric_values):.9f}"
            )

    return lines


# Cumulative traces fetched+shaped since process start; only mutated under the
# render cache lock (one render at a time), so a plain int is safe. Exposed as a
# counter so the dashboard can take rate() for throughput.
_traces_processed_total = 0


def _process_resident_bytes() -> int | None:
    """Current resident set size (RSS) of this process in bytes.

    Reads ``/proc/self/statm`` (Linux, stdlib, no deps). Returns None where it
    isn't available (e.g. a non-Linux dev box) so callers just omit the metric.
    """
    try:
        with open("/proc/self/statm", encoding="ascii") as statm:
            rss_pages = int(statm.read().split()[1])
        return rss_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def _exporter_self_metric_lines(render_seconds: float) -> list[str]:
    """Self-observability metrics for the exporter itself (speed, throughput,
    memory) — surfaced on the CI stack-health dashboard."""
    lines = [
        "# HELP pytest_trace_exporter_render_duration_seconds Wall-clock to fetch+shape the window on the last refresh.",
        "# TYPE pytest_trace_exporter_render_duration_seconds gauge",
        f"pytest_trace_exporter_render_duration_seconds {render_seconds:.6f}",
        "# HELP pytest_trace_exporter_traces_processed_total Traces fetched and shaped since start.",
        "# TYPE pytest_trace_exporter_traces_processed_total counter",
        f"pytest_trace_exporter_traces_processed_total {_traces_processed_total}",
        # Wall-clock time of this render, baked into the payload. Because the
        # payload is served from disk and survives a restart, a stuck/crashed
        # exporter keeps serving an old file with an old timestamp — alert on
        # time() - this gauge to catch a payload that has gone stale.
        "# HELP pytest_trace_exporter_last_render_timestamp_seconds Unix time this payload was rendered.",
        "# TYPE pytest_trace_exporter_last_render_timestamp_seconds gauge",
        f"pytest_trace_exporter_last_render_timestamp_seconds {time.time():.3f}",
        # Configured limits, emitted so dashboards compute exporter "pressure"
        # (window saturation, memory headroom) against the live config instead of
        # hardcoded constants that silently drift when the deployment changes.
        "# HELP pytest_trace_exporter_limit Configured max traces fetched per render (PYTEST_TRACE_EXPORTER_LIMIT).",
        "# TYPE pytest_trace_exporter_limit gauge",
        f"pytest_trace_exporter_limit {env_int('PYTEST_TRACE_EXPORTER_LIMIT', DEFAULT_LIMIT)}",
        "# HELP pytest_trace_exporter_mem_soft_bytes Soft RSS ceiling in bytes; cache is dropped above this (0 = disabled).",
        "# TYPE pytest_trace_exporter_mem_soft_bytes gauge",
        f"pytest_trace_exporter_mem_soft_bytes {_mem_soft_limit_bytes()}",
    ]
    rss = _process_resident_bytes()
    if rss is not None:
        lines.extend(
            [
                "# HELP pytest_trace_exporter_process_resident_bytes Resident memory of the exporter process.",
                "# TYPE pytest_trace_exporter_process_resident_bytes gauge",
                f"pytest_trace_exporter_process_resident_bytes {rss}",
            ]
        )
    return lines


def _iter_metric_lines() -> Iterator[str]:
    """Yield the Prometheus payload one line at a time (no trailing newline).

    Generating lines lazily lets the publisher (:func:`_write_payload_atomic`)
    stream them straight to disk, so the full multi-MB payload is never
    materialised in the heap as a joined string *and* its encoded bytes — the
    render's transient memory stays roughly flat regardless of how many test
    series the window produces. Each ``extract_*`` group is consumed and freed
    before the next, so peak is one group plus the shaped rows, not all lines at
    once.
    """
    global _traces_processed_total, _previous_new_ids
    started = time.monotonic()
    # The window is enumerated completely (every trace id, not just the newest
    # ``limit``) and assembled mostly from the shaped cache; only traces seen for
    # the first time are fetched this cycle. ``is_new`` marks those, so the
    # high-cardinality per-test series is emitted only for fresh traces (plus a
    # one-render carryover) while the roll-ups still aggregate the whole window —
    # Prometheus' last_over_time retains the rest, keeping the payload small.
    extracted: list[tuple[dict[str, str | int], list[dict[str, str | float]]]] = []
    latest_info: dict[str, str | int] | None = None
    latest_start = -1
    current_new_ids: set[str] = set()
    new_count = 0
    try:
        for trace_info, rows, is_new in _iter_window_shaped():
            extracted.append((trace_info, rows))
            trace_id = str(trace_info.get("trace_id", ""))
            if is_new:
                new_count += 1
                if trace_id:
                    current_new_ids.add(trace_id)
            start = int(trace_info.get("latest_start_time", 0) or 0)
            if start > latest_start:
                latest_start = start
                latest_info = trace_info
        resource_records = fetch_resource_records()
    except Exception as error:
        yield "# HELP pytest_trace_exporter_up Whether the exporter could query Tempo."
        yield "# TYPE pytest_trace_exporter_up gauge"
        yield "pytest_trace_exporter_up 0"
        yield "# HELP pytest_trace_exporter_last_error Last exporter error."
        yield "# TYPE pytest_trace_exporter_last_error gauge"
        yield f"pytest_trace_exporter_last_error{{message={json.dumps(str(error))}}} 1"
        # Emit self-metrics on the error path too, so the exporter's own health
        # panels (render time, RSS) stay populated rather than going "no data".
        yield from _exporter_self_metric_lines(time.monotonic() - started)
        return

    if not extracted:
        # Empty window (no CI activity) is still a healthy exporter — emit the
        # self-metrics so the dashboard shows it idling, not "no data".
        yield "# HELP pytest_trace_exporter_up Whether the exporter could query Tempo."
        yield "# TYPE pytest_trace_exporter_up gauge"
        yield "pytest_trace_exporter_up 1"
        yield from _exporter_self_metric_lines(time.monotonic() - started)
        return

    # Track which traces belong to which run, then build the run roll-ups over
    # each run's COMPLETE (settled) trace set rather than just the in-window
    # traces — otherwise a run's rollup decays to a wrong partial value as its
    # job traces age out of the lookback window one by one.
    now = time.monotonic()
    record_run_membership(extracted, now)
    rollup_extracted = settled_runs_complete_extracted(
        extracted, now, _run_settle_seconds()
    )

    # Per-test is emitted only for traces newly seen this render plus the
    # previous render's new ids (the one-render carryover), so the same
    # high-cardinality series is never emitted twice in one payload (duplicate
    # samples are rejected) yet always lands in at least two consecutive
    # payloads. Everything else aggregates the COMPLETE window.
    emit_per_test_ids = current_new_ids | _previous_new_ids
    per_test_extracted = [
        (info, rows)
        for info, rows in extracted
        if str(info.get("trace_id", "")) in emit_per_test_ids
    ]
    _previous_new_ids = current_new_ids

    with _enumeration_lock:
        enumeration_truncated = _last_enumeration_truncated
        enumeration_deferred = _last_enumeration_deferred

    yield "# HELP pytest_trace_exporter_up Whether the exporter could query Tempo."
    yield "# TYPE pytest_trace_exporter_up gauge"
    yield "pytest_trace_exporter_up 1"
    yield "# HELP pytest_trace_exporter_trace_count Number of traces aggregated from the window this render."
    yield "# TYPE pytest_trace_exporter_trace_count gauge"
    yield f"pytest_trace_exporter_trace_count {len(extracted)}"
    yield "# HELP pytest_trace_exporter_traces_deferred Window traces not yet fetched this render (picked up on a later render)."
    yield "# TYPE pytest_trace_exporter_traces_deferred gauge"
    yield f"pytest_trace_exporter_traces_deferred {enumeration_deferred}"
    yield "# HELP pytest_trace_exporter_enumeration_truncated 1 if window enumeration hit its slice budget (possibly incomplete)."
    yield "# TYPE pytest_trace_exporter_enumeration_truncated gauge"
    yield (
        "pytest_trace_exporter_enumeration_truncated "
        f"{1 if enumeration_truncated else 0}"
    )
    yield from extract_ci_runner_execution_metrics(_extracted=extracted)
    yield from extract_per_test_duration_metrics(_extracted=per_test_extracted)
    yield from extract_run_rollup_metrics(_extracted=rollup_extracted)
    yield from extract_run_info_metrics(_extracted=rollup_extracted)
    yield from extract_pr_info_metrics([], _extracted=extracted)
    yield from extract_pr_last_failure_metrics([], _extracted=extracted)
    yield from extract_average_metrics([], _extracted=extracted)
    yield from extract_average_resource_metrics(resource_records)

    _traces_processed_total += new_count
    yield from _exporter_self_metric_lines(time.monotonic() - started)

    if latest_info is not None:
        yield from latest_trace_info_lines(latest_info)


def _render_metrics_uncached() -> str:
    """Full payload as one string. Production publishing streams to disk via
    :func:`_write_payload_atomic`; this whole-body form is for callers/tests."""
    return "\n".join(_iter_metric_lines()) + "\n"


def _payload_path() -> Path:
    return Path(os.getenv("PYTEST_TRACE_EXPORTER_PAYLOAD_FILE", DEFAULT_PAYLOAD_FILE))


def _write_payload_atomic(path: Path) -> None:
    """Render and publish the payload to ``path`` atomically, streaming.

    Folds rendering and publishing: lines from :func:`_iter_metric_lines` are
    written to a sibling temp file as they are produced — so the full payload
    never exists in the heap as a joined string plus its encoded bytes — then
    fsync + ``os.replace`` publish it atomically. The temp file shares the
    directory so the replace is a same-filesystem rename; a crash mid-write
    leaves the previous complete payload intact (a reader sees old-or-new, never
    torn), and the trailing directory fsync makes the rename durable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for line in _iter_metric_lines():
                handle.write(line)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    # Persist the rename itself, not just the file contents. Best-effort: some
    # filesystems/platforms don't support directory fsync.
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def _cache_ttl_seconds() -> float:
    raw = os.getenv("PYTEST_TRACE_EXPORTER_CACHE_SECONDS")
    if raw is None or raw == "":
        return DEFAULT_CACHE_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_CACHE_SECONDS


def _refresh_cooldown_seconds() -> float:
    raw = os.getenv("PYTEST_TRACE_EXPORTER_REFRESH_COOLDOWN_SECONDS")
    if raw is None or raw == "":
        return DEFAULT_REFRESH_COOLDOWN_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_REFRESH_COOLDOWN_SECONDS


def _refresh_slow_seconds() -> float:
    raw = os.getenv("PYTEST_TRACE_EXPORTER_REFRESH_SLOW_SECONDS")
    if raw is None or raw == "":
        return DEFAULT_REFRESH_SLOW_SECONDS
    try:
        return max(1.0, float(raw))
    except ValueError:
        return DEFAULT_REFRESH_SLOW_SECONDS


_WARMING_PAYLOAD = (
    "# HELP pytest_trace_exporter_up Whether the exporter could query Tempo.\n"
    "# TYPE pytest_trace_exporter_up gauge\n"
    "pytest_trace_exporter_up 1\n"
)


def render_metrics() -> str:
    """Return the most recently rendered payload instantly (never renders inline).

    Rendering runs in a background thread (:func:`_refresh_loop`) because a single
    render does a multi-second Tempo search + fetch/shape of the whole window —
    doing that inside the scrape handler made scrapes exceed Prometheus's
    scrape_timeout (target went down with "context deadline exceeded", so nothing
    landed). The payload is published to disk (:func:`_refresh_cache_once`) and
    served from there, so it never sits in the Python heap between scrapes and
    survives a restart. Until the first render lands the endpoint reports
    ``up 1`` so the target stays healthy while warming.
    """
    try:
        return _payload_path().read_text(encoding="utf-8")
    except FileNotFoundError:
        return _WARMING_PAYLOAD


def _parse_metric_labels(raw: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    index = 0
    while index < len(raw):
        match = re.match(r'\s*([a-zA-Z_:][a-zA-Z0-9_:]*)="', raw[index:])
        if match is None:
            break
        key = match.group(1)
        index += match.end()
        value_chars = []
        while index < len(raw):
            char = raw[index]
            if char == "\\" and index + 1 < len(raw):
                escaped = raw[index + 1]
                value_chars.append("\n" if escaped == "n" else escaped)
                index += 2
                continue
            if char == '"':
                index += 1
                break
            value_chars.append(char)
            index += 1
        labels[key] = "".join(value_chars)
        if index < len(raw) and raw[index] == ",":
            index += 1
    return labels


def _iter_metric_samples(
    metric_name: str, source: str | None = None
) -> Iterator[tuple[dict[str, str], float]]:
    """Yield (labels, value) for every sample of ``metric_name``.

    Reads the published payload (``render_metrics()``) by default, or a caller-
    supplied Prometheus-text blob — used by the badge fallback to parse freshly
    rendered roll-up lines for a single PR without touching the global payload.
    """
    text = source if source is not None else render_metrics()
    pattern = re.compile(
        rf"^{re.escape(metric_name)}(?:{{([^}}]*)}})?\s+([-+0-9.eE]+)(?:\s|$)"
    )
    for line in text.splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        labels = _parse_metric_labels(match.group(1) or "")
        try:
            value = float(match.group(2))
        except ValueError:
            continue
        yield labels, value


def _latest_pr_run_summary_from_metrics(
    pr: str, text: str
) -> dict[str, str | float | int | None] | None:
    runs: dict[tuple[str, str, str, str], dict[str, str | float | int | None]] = {}
    metric_to_field = {
        "pytest_run_start_time_seconds": "started_at_seconds",
        "pytest_run_end_time_seconds": "ended_at_seconds",
        "pytest_run_total_tests": "total_tests",
        "pytest_run_failed_tests": "failed_tests",
        "pytest_run_duration_seconds": "duration_seconds",
        "pytest_run_job_count": "job_count",
    }
    pattern = re.compile(
        r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:{([^}]*)})?\s+([-+0-9.eE]+)(?:\s|$)"
    )
    for line in text.splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        field_name = metric_to_field.get(match.group(1))
        if field_name is None:
            continue
        labels = _parse_metric_labels(match.group(2) or "")
        if labels.get("pr") != pr:
            continue
        try:
            value = float(match.group(3))
        except ValueError:
            continue
        key = (
            labels.get("service_name", ""),
            labels.get("provider", ""),
            labels.get("pr", ""),
            labels.get("run_id", ""),
        )
        summary = runs.setdefault(
            key,
            {
                "duration_seconds": None,
                "ended_at_seconds": None,
                "failed_tests": None,
                "job_count": None,
                "pr": pr,
                "provider": labels.get("provider", ""),
                "run_id": labels.get("run_id", ""),
                "service_name": labels.get("service_name", ""),
                "started_at_seconds": None,
                "total_tests": None,
            },
        )
        summary[field_name] = value

    if not runs:
        return None
    return max(
        runs.values(), key=lambda item: float(item.get("started_at_seconds") or 0)
    )


def _latest_pr_run_summary(
    pr: str, source: str | None = None
) -> dict[str, str | float | int | None] | None:
    return _latest_pr_run_summary_from_metrics(
        pr, source if source is not None else render_metrics()
    )


def _format_badge_count(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{int(value):,}"


def _badge_failure_color(failed: int, total: int) -> str:
    if failed <= 0:
        return "green"
    if total <= 0:
        return "red"
    failure_rate = 100 * failed / total
    if failure_rate < 1:
        return "94B45F"
    if failure_rate < 10:
        return "orange"
    return "red"


# ---------------------------------------------------------------------------
# On-demand per-PR lookup (badge / summary fallback)
#
# A PR's last CI run is frequently older than the live render's lookback window,
# so the payload the badge normally reads holds nothing for it — and the badge
# would read "no data" even for a perfectly valid PR. When the payload misses, we
# first query the cheap Prometheus run rollups. If that is unavailable, we search
# Tempo directly, scoped to just that one PR over a wider window, re-aggregate its
# runs with the same roll-up code the render uses, and pick the latest run. The
# result (including a negative "no data") is memoized per PR so a hammered badge
# does not pound the backends.
# ---------------------------------------------------------------------------


def _badge_lookback_seconds() -> int:
    return parse_lookback_seconds(
        os.getenv("PYTEST_TRACE_EXPORTER_BADGE_LOOKBACK", DEFAULT_BADGE_LOOKBACK),
        DEFAULT_BADGE_LOOKBACK,
    )


def _badge_cache_seconds() -> float:
    raw = os.getenv("PYTEST_TRACE_EXPORTER_BADGE_CACHE_SECONDS", "")
    try:
        return max(0.0, float(raw)) if raw else DEFAULT_BADGE_CACHE_SECONDS
    except ValueError:
        return DEFAULT_BADGE_CACHE_SECONDS


def prometheus_base_url() -> str:
    return os.getenv(
        "PYTEST_TRACE_EXPORTER_PROMETHEUS_URL", DEFAULT_PROMETHEUS_URL
    ).rstrip("/")


def _badge_prometheus_lookback() -> str:
    return os.getenv(
        "PYTEST_TRACE_EXPORTER_BADGE_PROMETHEUS_LOOKBACK",
        DEFAULT_BADGE_PROMETHEUS_LOOKBACK,
    )


_pr_summary_cache_lock = threading.Lock()
# pr -> (expiry_monotonic, summary-or-None). None (a genuinely-unknown PR) is
# cached too, so a miss doesn't re-search Tempo on every single badge hit.
_pr_summary_cache: dict[
    str, tuple[float, dict[str, str | float | int | None] | None]
] = {}


def _pr_run_summary_from_prometheus(
    pr: str,
) -> dict[str, str | float | int | None] | None:
    """Read a PR's latest run from Prometheus rollups.

    This is much cheaper than the Tempo fallback: the badge fields are already
    persisted as low-cardinality series, while Tempo has to scan blocks and
    fetch full traces. Prefer job-level totals for the selected run because the
    PR dashboard uses those and they can be more complete than the run-level
    aggregate while a run is settling.
    """
    base_url = prometheus_base_url()
    if not base_url:
        return None
    metric_pattern = (
        "pytest_run_start_time_seconds|"
        "pytest_run_end_time_seconds|"
        "pytest_run_total_tests|"
        "pytest_run_failed_tests|"
        "pytest_run_duration_seconds|"
        "pytest_run_job_count|"
        "pytest_run_job_member_info|"
        "pytest_run_job_total_tests|"
        "pytest_run_job_failed_tests"
    )
    query = (
        f'last_over_time({{__name__=~"{metric_pattern}",pr="{pr}"}}'
        f"[{_badge_prometheus_lookback()}])"
    )
    url = f"{base_url}/api/v1/query?{urlencode({'query': query})}"
    try:
        payload = _http_get_json(url)
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("status") != "success":
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    result = data.get("result")
    if not isinstance(result, list):
        return None

    lines: list[str] = []
    job_values: dict[
        tuple[tuple[str, str, str, str], str], dict[str, float | bool]
    ] = {}
    for item in result:
        if not isinstance(item, dict):
            continue
        metric = item.get("metric")
        value = item.get("value")
        if (
            not isinstance(metric, dict)
            or not isinstance(value, list)
            or len(value) < 2
        ):
            continue
        metric_name = metric.get("__name__")
        if not isinstance(metric_name, str):
            continue
        labels = {str(k): str(v) for k, v in metric.items() if k != "__name__"}
        try:
            metric_value = float(value[1])
        except (TypeError, ValueError):
            continue
        lines.append(f"{metric_name}{metric_labels(labels)} {value[1]}")
        if metric_name not in {
            "pytest_run_job_member_info",
            "pytest_run_job_total_tests",
            "pytest_run_job_failed_tests",
        }:
            continue
        test_job = labels.get("test_job", "")
        if not test_job:
            continue
        run_key = (
            labels.get("service_name", ""),
            labels.get("provider", ""),
            labels.get("pr", ""),
            labels.get("run_id", ""),
        )
        job_key = (run_key, test_job)
        aggregate = job_values.setdefault(
            job_key, {"member": False, "total_tests": 0.0, "failed_tests": 0.0}
        )
        if metric_name == "pytest_run_job_member_info":
            aggregate["member"] = True
        elif metric_name == "pytest_run_job_total_tests":
            aggregate["total_tests"] = max(
                float(aggregate["total_tests"]), metric_value
            )
        elif metric_name == "pytest_run_job_failed_tests":
            aggregate["failed_tests"] = max(
                float(aggregate["failed_tests"]), metric_value
            )
    summary = _latest_pr_run_summary(pr, source="\n".join(lines)) if lines else None
    if summary is None:
        return None

    summary_key = (
        str(summary.get("service_name") or ""),
        str(summary.get("provider") or ""),
        str(summary.get("pr") or ""),
        str(summary.get("run_id") or ""),
    )
    matching_jobs = [
        aggregate
        for (run_key, _), aggregate in job_values.items()
        if run_key == summary_key
    ]
    if matching_jobs:
        summary["total_tests"] = sum(
            float(aggregate["total_tests"]) for aggregate in matching_jobs
        )
        summary["failed_tests"] = sum(
            float(aggregate["failed_tests"]) for aggregate in matching_jobs
        )
        summary["job_count"] = len(matching_jobs)
    return summary


def _pr_extracted_rows(
    pr: str,
) -> list[tuple[dict[str, str | int], list[dict[str, str | float]]]]:
    """Search Tempo for this PR's traces and shape them into (trace_info, rows).

    Streams exactly like :func:`iter_traces`: each fetched trace is shaped into
    its small row set and then dropped, so the multi-MB traces never pile up on
    the heap (a single PR can return the full search limit of large traces, and a
    materialise-then-aggregate approach would spike the process toward its OOM
    ceiling). Returns [] on any search/fetch failure, which the caller renders as
    "no data" rather than erroring the badge.
    """
    base_url = tempo_base_url()
    service_name = os.getenv("PYTEST_TRACE_EXPORTER_SERVICE_NAME", DEFAULT_SERVICE_NAME)
    limit = env_int(
        "PYTEST_TRACE_EXPORTER_BADGE_TRACE_LIMIT", DEFAULT_BADGE_TRACE_LIMIT
    )
    end = int(time.time())
    start = end - _badge_lookback_seconds()
    # Handlers validate `pr` as digits-only before we get here; quote() in the
    # search keeps the TraceQL well-formed regardless.
    selector = f'resource.vcs.change.id = "{pr}"'
    try:
        trace_ids = search_trace_ids(
            base_url, service_name, start, end, limit, extra_selector=selector
        )
    except Exception:
        return []
    if not trace_ids:
        return []

    workers = max(
        1,
        min(
            env_int(
                "PYTEST_TRACE_EXPORTER_FETCH_CONCURRENCY", DEFAULT_FETCH_CONCURRENCY
            ),
            len(trace_ids),
        ),
    )
    extracted: list[tuple[dict[str, str | int], list[dict[str, str | float]]]] = []
    pending = iter(trace_ids)
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="badge-fetch"
    ) as pool:
        inflight = {
            pool.submit(get_trace, tid, base_url) for tid in islice(pending, workers)
        }
        while inflight:
            done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for future in done:
                nxt = next(pending, None)
                if nxt is not None:
                    inflight.add(pool.submit(get_trace, nxt, base_url))
                try:
                    trace = future.result()
                except Exception:
                    trace = None
                if trace is not None:
                    # Shape into small rows; the full trace is then dropped.
                    extracted.append(extract_trace_rows(trace))
    return extracted


def _pr_run_summary_cached(pr: str) -> dict[str, str | float | int | None] | None:
    """Latest-run summary for a PR: payload, Prometheus, then memoized Tempo.

    The live payload is the cheap path — a PR that ran inside the render window
    is already there. Otherwise query persisted Prometheus rollups, falling back
    to a per-PR Tempo search when needed. Misses are memoized for
    :func:`_badge_cache_seconds`.
    """
    summary = _latest_pr_run_summary(pr)
    if summary is not None:
        return summary

    now = time.monotonic()
    with _pr_summary_cache_lock:
        hit = _pr_summary_cache.get(pr)
        if hit is not None and hit[0] > now:
            return hit[1]

    summary = _pr_run_summary_from_prometheus(pr)
    if summary is not None:
        with _pr_summary_cache_lock:
            _pr_summary_cache[pr] = (
                time.monotonic() + _badge_cache_seconds(),
                summary,
            )
        return summary

    extracted = _pr_extracted_rows(pr)
    summary = (
        _latest_pr_run_summary(
            pr,
            source="\n".join(extract_run_rollup_metrics(_extracted=extracted)),
        )
        if extracted
        else None
    )
    with _pr_summary_cache_lock:
        _pr_summary_cache[pr] = (time.monotonic() + _badge_cache_seconds(), summary)
    return summary


def render_pr_badge_svg(pr: str) -> bytes:
    summary = _pr_run_summary_cached(pr)
    if summary is None:
        message = "no data"
        color = "9f9f9f"
    else:
        failed = int(float(summary.get("failed_tests") or 0))
        total_value = int(float(summary.get("total_tests") or 0))
        total = _format_badge_count(summary.get("total_tests"))
        jobs = _format_badge_count(summary.get("job_count"))
        message = f"{failed} failed / {total} tests / {jobs} jobs"
        color = _badge_failure_color(failed, total_value)

    label = f"PR {pr} CI"
    label_width = max(70, 7 * len(label) + 10)
    message_width = max(120, 7 * len(message) + 10)
    width = label_width + message_width
    escaped_label = html.escape(label)
    escaped_message = html.escape(message)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="20" role="img" aria-label="{escaped_label}: {escaped_message}">
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r"><rect width="{width}" height="20" rx="3" fill="#fff"/></clipPath>
  <g clip-path="url(#r)">
    <rect width="{label_width}" height="20" fill="#555"/>
    <rect x="{label_width}" width="{message_width}" height="20" fill="#{color}"/>
    <rect width="{width}" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" text-rendering="geometricPrecision" font-size="11">
    <text x="{label_width / 2:.1f}" y="15" fill="#010101" fill-opacity=".3">{escaped_label}</text>
    <text x="{label_width / 2:.1f}" y="14">{escaped_label}</text>
    <text x="{label_width + message_width / 2:.1f}" y="15" fill="#010101" fill-opacity=".3">{escaped_message}</text>
    <text x="{label_width + message_width / 2:.1f}" y="14">{escaped_message}</text>
  </g>
</svg>
'''
    return svg.encode("utf-8")


def render_pr_summary_json(pr: str) -> bytes:
    summary = _pr_run_summary_cached(pr)
    payload = {"pr": pr, "available": summary is not None, "latest_run": summary}
    return json.dumps(payload, sort_keys=True).encode("utf-8") + b"\n"


def _mem_soft_limit_bytes() -> int:
    """Soft RSS ceiling in bytes (0 = disabled). Default ~88% of the 640 MiB
    container limit."""
    return (
        max(0, env_int("PYTEST_TRACE_EXPORTER_MEM_SOFT_MB", DEFAULT_MEM_SOFT_MB))
        * 1024
        * 1024
    )


def _relieve_memory_pressure() -> None:
    """Before a render, if RSS is over the soft limit, drop the reclaimable
    trace caches so the render reuses freed heap instead of stacking new
    allocations on a near-full process and tipping into the hard cgroup limit
    (an OOM-kill).

    Both the raw-trace cache and the (now larger) shaped-window cache are
    reclaimable; Tempo is the durable source, so a dropped entry is just
    re-fetched — bounded by the per-render fetch cap, and the run-settle gate
    means a run's roll-up still waits until its re-fetched shards are back.
    Best-effort — a pure safety valve under load, off when MEM_SOFT_MB <= 0.
    """
    soft = _mem_soft_limit_bytes()
    if soft <= 0:
        return
    rss = _process_resident_bytes()
    if rss is None or rss < soft:
        return
    global _trace_cache_bytes, _shaped_cache_bytes
    with _trace_cache_lock:
        dropped = len(_trace_cache)
        _trace_cache.clear()
        _trace_cache_sizes.clear()
        _trace_cache_bytes = 0
    with _shaped_cache_lock:
        dropped += len(_shaped_cache)
        _shaped_cache.clear()
        _shaped_cache_sizes.clear()
        _shaped_meta.clear()
        _shaped_cache_bytes = 0
    print(
        f"[pytest-trace-exporter] RSS {rss // (1024 * 1024)}MiB over soft limit "
        f"{soft // (1024 * 1024)}MiB; dropped {dropped} cached traces to cap growth",
        file=sys.stderr,
        flush=True,
    )


def _refresh_cache_once() -> None:
    _relieve_memory_pressure()
    _write_payload_atomic(_payload_path())


def _refresh_loop(interval: float) -> None:
    """Re-render the payload in the background: render, then sleep `interval`.

    Renders immediately on start (so the cache fills within one render) and never
    holds the lock during the slow render — only the quick payload swap is locked.
    """
    while True:
        started = time.monotonic()
        failed = False
        try:
            _refresh_cache_once()
        except Exception:  # never let the refresher thread die
            failed = True
        elapsed = time.monotonic() - started
        if failed or elapsed >= _refresh_slow_seconds():
            time.sleep(max(1.0, _refresh_cooldown_seconds()))
        else:
            time.sleep(max(1.0, interval))


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/badge/pr":
            self._serve_pr_badge(parse_qs(parsed.query))
            return
        if parsed.path == "/summary/pr":
            self._serve_pr_summary(parse_qs(parsed.query))
            return
        if parsed.path == "/failure":
            self._serve_failure(parse_qs(parsed.query))
            return
        if parsed.path in {"/metrics", "/"}:
            self._serve_metrics()
            return
        self.send_response(404)
        self.end_headers()

    def _serve_metrics(self) -> None:
        """Stream the published payload file straight to the socket.

        Opening the file first pins that inode: if the refresher os.replace()s a
        new payload in mid-response, this handler keeps streaming the complete
        file it opened (and Content-Length still matches), so a scrape can never
        see a torn body. Streaming also keeps the multi-MB payload off the heap —
        we never materialize it as a single Python string per request.
        """
        try:
            handle = open(_payload_path(), "rb")
        except FileNotFoundError:
            self._send(200, METRICS_CONTENT_TYPE, _WARMING_PAYLOAD.encode("utf-8"))
            return
        with handle:
            size = os.fstat(handle.fileno()).st_size
            self.send_response(200)
            self.send_header("Content-Type", METRICS_CONTENT_TYPE)
            self.send_header("Content-Length", str(size))
            self.end_headers()
            shutil.copyfileobj(handle, self.wfile, 64 * 1024)

    def _serve_pr_badge(self, params: dict[str, list[str]]) -> None:
        pr = (params.get("pr") or [""])[0].strip()
        if not pr.isdigit():
            self._send(400, SVG_CONTENT_TYPE, render_pr_badge_svg("unknown"))
            return
        self._send(200, SVG_CONTENT_TYPE, render_pr_badge_svg(pr))

    def _serve_pr_summary(self, params: dict[str, list[str]]) -> None:
        pr = (params.get("pr") or [""])[0].strip()
        if not pr.isdigit():
            self._send(400, JSON_CONTENT_TYPE, b'{"error":"missing numeric pr"}\n')
            return
        self._send(200, JSON_CONTENT_TYPE, render_pr_summary_json(pr))

    def _serve_failure(self, params: dict[str, list[str]]) -> None:
        trace_id = (params.get("trace_id") or [""])[0].strip()
        test_nodeid = (params.get("test_nodeid") or [""])[0].strip()
        trace = get_trace(trace_id) if trace_id else None
        details = extract_failure_details(trace, test_nodeid) if trace else []
        if trace and details:
            annotate_github_links(trace, details)
        self._send(
            200,
            "text/html; charset=utf-8",
            render_failure_html(trace_id, details).encode("utf-8"),
        )

    def _send(self, status: int, content_type: str, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


def _limit_malloc_arenas() -> None:
    """Cap glibc malloc arenas before threads start.

    Threaded use (the fetch pool + the per-request HTTP server) otherwise lets
    glibc spin up to ~8*nproc arenas, inflating RSS far above live heap. We set
    this in code via ``mallopt`` rather than relying on a ``MALLOC_ARENA_MAX``
    env from the launcher, so the protection holds however the exporter is run
    (bare ``python -m ...``, systemd, docker, ...). A ``MALLOC_ARENA_MAX`` env,
    if present, still applies (glibc honours it at startup) and is respected
    here as an override. No-op on non-glibc platforms (e.g. macOS).
    """
    try:
        max_arenas = int(os.getenv("MALLOC_ARENA_MAX", "") or 2)
    except ValueError:
        max_arenas = 2
    if max_arenas <= 0:
        return
    try:
        import ctypes

        m_arena_max = -8  # glibc mallopt M_ARENA_MAX
        ctypes.CDLL(None).mallopt(m_arena_max, max_arenas)
    except (OSError, AttributeError, ValueError):
        pass  # mallopt unavailable (not glibc) — fine, just skip


def main() -> None:
    # Bound malloc arenas before any worker threads are created.
    _limit_malloc_arenas()
    port = env_int("PYTEST_TRACE_EXPORTER_PORT", DEFAULT_PORT)
    # Render in the background so scrapes serve the cached payload instantly.
    refresher = threading.Thread(
        target=_refresh_loop, args=(_cache_ttl_seconds(),), daemon=True
    )
    refresher.start()
    server = ThreadingHTTPServer(("0.0.0.0", port), MetricsHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
