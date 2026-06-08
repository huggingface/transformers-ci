from __future__ import annotations

import html
import json
import os
import re
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import fsum
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse
from urllib.request import Request, urlopen


DEFAULT_TEMPO_URL = "http://tempo:3200"
DEFAULT_LIMIT = 200
DEFAULT_LOOKBACK = "1h"
DEFAULT_PORT = 8000
DEFAULT_RESOURCE_METRICS_FILE = "/data/pytest-resource-metrics.jsonl"
DEFAULT_SERVICE_NAME = "pytest-observability-demo"
DEFAULT_CACHE_SECONDS = 10.0
DEFAULT_GITHUB_API_URL = "https://api.github.com"
DEFAULT_GITHUB_CACHE_SECONDS = 300.0
# A completed CI trace is immutable, so once it is this many seconds old we
# parse it once and memoize the result instead of re-fetching it from Tempo on
# every scrape. Younger traces may still be receiving spans, so they are left
# uncached and refreshed each cycle.
DEFAULT_TRACE_SETTLE_SECONDS = 120.0


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


def _http_get_json(url: str, timeout: float = 5.0) -> object:
    with urlopen(url, timeout=timeout) as response:
        return json.load(response)


def search_trace_ids(
    base_url: str, service_name: str, start: int, end: int, limit: int
) -> list[str]:
    traceql = quote(f'{{ resource.service.name = "{service_name}" }}', safe="")
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


_trace_cache_lock = threading.Lock()
_trace_cache: dict[str, dict] = {}


def _trace_settle_seconds() -> float:
    raw = os.getenv("PYTEST_TRACE_EXPORTER_TRACE_SETTLE_SECONDS")
    if raw is None or raw == "":
        return DEFAULT_TRACE_SETTLE_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_TRACE_SETTLE_SECONDS


def get_trace(trace_id: str, base_url: str | None = None) -> dict | None:
    """Return one trace in the Jaeger-shaped dict, fetching from Tempo if needed.

    Completed traces are immutable, so once one is old enough to have settled it
    is memoized by ``trace_id`` and never re-fetched. Used both by the scrape
    loop (``fetch_traces``) and the ``/failure`` traceback view.
    """
    with _trace_cache_lock:
        cached = _trace_cache.get(trace_id)
    if cached is not None:
        return cached

    if base_url is None:
        base_url = tempo_base_url()
    try:
        payload = _http_get_json(f"{base_url}/api/traces/{quote(trace_id, safe='')}")
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    trace = tempo_trace_to_jaeger(trace_id, payload)
    # Only memoize traces old enough to be definitely complete, so an in-flight
    # CI run still picks up its later spans on the next scrape.
    latest_micros = trace_start_time(trace)
    if (
        latest_micros
        and time.time() - (latest_micros / 1_000_000) > _trace_settle_seconds()
    ):
        with _trace_cache_lock:
            _trace_cache[trace_id] = trace
    return trace


def fetch_traces() -> list[dict]:
    base_url = tempo_base_url()
    limit = env_int("PYTEST_TRACE_EXPORTER_LIMIT", DEFAULT_LIMIT)
    lookback = os.getenv("PYTEST_TRACE_EXPORTER_LOOKBACK", DEFAULT_LOOKBACK)
    service_name = os.getenv("PYTEST_TRACE_EXPORTER_SERVICE_NAME", DEFAULT_SERVICE_NAME)

    end = int(time.time())
    start = end - parse_lookback_seconds(lookback)
    trace_ids = search_trace_ids(base_url, service_name, start, end, limit)

    traces: list[dict] = []
    for trace_id in trace_ids:
        trace = get_trace(trace_id, base_url)
        if trace is not None:
            traces.append(trace)
    return traces


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
        "title": title if isinstance(title, str) else "",
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
    return message.strip().splitlines()[0] if message.strip() else ""


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
                "exception_stacktrace": exc_stacktrace,
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
    traces — the in-window ones plus any aged-out ones still in the trace cache
    — so the roll-up reflects the whole run. Runs still ingesting are skipped
    this cycle (so only one stable, complete series is ever emitted per run).
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
    lines = [
        "# HELP pytest_test_average_duration_seconds Average duration of pytest test spans across fetched traces.",
        "# TYPE pytest_test_average_duration_seconds gauge",
        "# HELP pytest_test_run_count Number of fetched traces that contained a given pytest test span.",
        "# TYPE pytest_test_run_count gauge",
        "# HELP pytest_test_failure_count Number of fetched traces where this pytest test span failed.",
        "# TYPE pytest_test_failure_count gauge",
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
        durations = aggregate["durations"]
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
        lines.append(
            f"pytest_test_average_duration_seconds{metric_labels(labels)} {fsum(durations) / len(durations):.9f}"
        )
        lines.append(f"pytest_test_run_count{metric_labels(labels)} {len(durations)}")
        failure_count = int(aggregate["failure_count"])
        lines.append(
            f"pytest_test_failure_count{metric_labels(labels)} {failure_count}"
        )
        if failure_count > 0:
            pointer_labels = dict(labels)
            pointer_labels["trace_id"] = str(aggregate["last_failure_trace_id"])
            pointer_labels["exception_type"] = str(
                aggregate["last_failure_exception_type"]
            )
            # No stacktrace label here either — the trace_id pointer is enough
            # for the dashboard to deep-link into the Tempo trace view.
            lines.append(
                f"pytest_test_last_failure_info{metric_labels(pointer_labels)} 1"
            )

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


def _render_metrics_uncached() -> str:
    try:
        traces = fetch_traces()
        resource_records = fetch_resource_records()
    except Exception as error:
        return (
            "# HELP pytest_trace_exporter_up Whether the exporter could query Tempo.\n"
            "# TYPE pytest_trace_exporter_up gauge\n"
            "pytest_trace_exporter_up 0\n"
            "# HELP pytest_trace_exporter_last_error Last exporter error.\n"
            "# TYPE pytest_trace_exporter_last_error gauge\n"
            f"pytest_trace_exporter_last_error{{message={json.dumps(str(error))}}} 1\n"
        )

    if not traces:
        return (
            "# HELP pytest_trace_exporter_up Whether the exporter could query Tempo.\n"
            "# TYPE pytest_trace_exporter_up gauge\n"
            "pytest_trace_exporter_up 1\n"
        )

    extracted = _precompute_trace_rows(traces)

    # Track which traces belong to which run, then build the run roll-ups over
    # each run's COMPLETE (settled) trace set rather than just the in-window
    # traces — otherwise a run's rollup decays to a wrong partial value as its
    # job traces age out of the lookback window one by one.
    now = time.monotonic()
    record_run_membership(extracted, now)
    rollup_extracted = settled_runs_complete_extracted(
        extracted, now, _run_settle_seconds()
    )

    rendered = [
        "# HELP pytest_trace_exporter_up Whether the exporter could query Tempo.",
        "# TYPE pytest_trace_exporter_up gauge",
        "pytest_trace_exporter_up 1",
        "# HELP pytest_trace_exporter_trace_count Number of traces fetched from Tempo for aggregation.",
        "# TYPE pytest_trace_exporter_trace_count gauge",
        f"pytest_trace_exporter_trace_count {len(traces)}",
    ]
    rendered.extend(extract_per_test_duration_metrics(_extracted=extracted))
    rendered.extend(extract_run_rollup_metrics(_extracted=rollup_extracted))
    rendered.extend(extract_run_info_metrics(traces, _extracted=rollup_extracted))
    rendered.extend(extract_pr_info_metrics(traces, _extracted=extracted))
    rendered.extend(extract_pr_last_failure_metrics(traces, _extracted=extracted))
    rendered.extend(extract_average_metrics(traces, _extracted=extracted))
    rendered.extend(extract_average_resource_metrics(resource_records))

    latest = latest_trace(traces)
    if latest is not None:
        rendered.extend(extract_latest_trace_metrics(latest))
    return "\n".join(rendered) + "\n"


_cache_lock = threading.Lock()
_cached_payload: tuple[float, str] | None = None


def _cache_ttl_seconds() -> float:
    raw = os.getenv("PYTEST_TRACE_EXPORTER_CACHE_SECONDS")
    if raw is None or raw == "":
        return DEFAULT_CACHE_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_CACHE_SECONDS


def render_metrics() -> str:
    global _cached_payload
    ttl = _cache_ttl_seconds()
    with _cache_lock:
        now = time.monotonic()
        if _cached_payload is not None and ttl > 0:
            cached_at, body = _cached_payload
            if now - cached_at < ttl:
                return body
        body = _render_metrics_uncached()
        _cached_payload = (now, body)
        return body


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/failure":
            self._serve_failure(parse_qs(parsed.query))
            return
        if parsed.path in {"/metrics", "/"}:
            self._send(
                200,
                "text/plain; version=0.0.4; charset=utf-8",
                render_metrics().encode("utf-8"),
            )
            return
        self.send_response(404)
        self.end_headers()

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


def main() -> None:
    port = env_int("PYTEST_TRACE_EXPORTER_PORT", DEFAULT_PORT)
    server = ThreadingHTTPServer(("0.0.0.0", port), MetricsHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
