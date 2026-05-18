from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import fsum
from pathlib import Path
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen


DEFAULT_JAEGER_URL = "http://jaeger:16686"
DEFAULT_LIMIT = 200
DEFAULT_LOOKBACK = "1h"
DEFAULT_PORT = 8000
DEFAULT_RESOURCE_METRICS_FILE = "/data/pytest-resource-metrics.jsonl"
DEFAULT_SERVICE_NAME = "pytest-observability-demo"
DEFAULT_CACHE_SECONDS = 10.0
DEFAULT_GITHUB_API_URL = "https://api.github.com"
DEFAULT_GITHUB_CACHE_SECONDS = 300.0


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
    return max((int(span.get("startTime", 0)) for span in spans if isinstance(span, dict)), default=0)


def fetch_traces() -> list[dict]:
    base_url = os.getenv("PYTEST_TRACE_EXPORTER_JAEGER_URL", DEFAULT_JAEGER_URL).rstrip("/")
    limit = env_int("PYTEST_TRACE_EXPORTER_LIMIT", DEFAULT_LIMIT)
    lookback = os.getenv("PYTEST_TRACE_EXPORTER_LOOKBACK", DEFAULT_LOOKBACK)
    service_name = os.getenv("PYTEST_TRACE_EXPORTER_SERVICE_NAME", DEFAULT_SERVICE_NAME)
    search_url = (
        f"{base_url}/api/traces?service={quote(service_name)}&limit={limit}&lookback={quote(lookback)}"
    )
    with urlopen(search_url, timeout=5) as response:
        payload = json.load(response)

    traces = payload.get("data", [])
    if not isinstance(traces, list):
        return []
    return [trace for trace in traces if isinstance(trace, dict)]


def fetch_resource_records() -> list[dict[str, str | float | int]]:
    resource_metrics_file = Path(os.getenv("PYTEST_RESOURCE_METRICS_FILE", DEFAULT_RESOURCE_METRICS_FILE))
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
    api_base_url = os.getenv("PYTEST_GITHUB_API_URL", DEFAULT_GITHUB_API_URL).rstrip("/")
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
    api_base_url = os.getenv("PYTEST_GITHUB_API_URL", DEFAULT_GITHUB_API_URL).rstrip("/")
    api_url = f"{api_base_url}/repos/{quote(repository, safe='/')}/pulls/{quote(pr, safe='')}"
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
        "html_url": html_url if isinstance(html_url, str) else github_pr_html_url(repository, pr),
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


def latest_trace(traces: list[dict]) -> dict | None:
    if not traces:
        return None
    return max(traces, key=trace_start_time)


def extract_trace_rows(trace: dict) -> tuple[dict[str, str | int], list[dict[str, str | float]]]:
    trace_id = trace.get("traceID")
    spans = trace.get("spans", [])
    processes = trace.get("processes", {})

    if not isinstance(trace_id, str) or not isinstance(spans, list) or not isinstance(processes, dict):
        return {
            "end_time": 0,
            "latest_start_time": 0,
            "run_id": "unknown",
            "start_time": 0,
            "trace_id": "unknown",
        }, []

    process_run_id = ""
    process_suite = ""
    process_provider = ""
    process_pr = ""
    process_pr_url = ""
    process_repository = ""
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
        process_tags = tag_map(process.get("tags", [])) if isinstance(process, dict) else {}
        service_name = process.get("serviceName", service_name) if isinstance(process, dict) else service_name
        process_run_id = process_tags.get(
            "transformers.test.run.id",
            process_tags.get("cicd.pipeline.run.id", process_run_id),
        )
        process_suite = process_tags.get(
            "transformers.test.suite",
            process_tags.get("transformers.test.job", process_suite),
        )
        process_provider = process_tags.get("transformers.test.provider", process_provider)
        process_pr = process_tags.get("vcs.change.id", process_pr)
        process_pr_url = process_tags.get("vcs.change.url", process_pr_url)
        process_repository = process_tags.get("vcs.repository.name", process_repository)

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
                "test_suite": process_suite or "unknown",
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
        "end_time": end_time,
        "latest_start_time": latest_start_time,
        "pr": process_pr or "none",
        "pr_url": process_pr_url,
        "provider": process_provider or "unknown",
        "repository": process_repository,
        "run_id": process_run_id or trace_id,
        "service_name": service_name or "unknown",
        "start_time": start_time,
        "test_suite": process_suite or "unknown",
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
        "test_suite": str(trace_info["test_suite"]),
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
    _extracted: list[tuple[dict[str, str | int], list[dict[str, str | float]]]] | None = None,
) -> list[str]:
    """Emit one ``pytest_pr_last_failure_info`` series per PR with a failure.

    Picks the most recent failing test across all runs of the PR. When the PR
    has no failures, no sample is emitted — used by the PR dashboard's Last
    Error panel with a repeat so the panel hides entirely when no data.
    """
    extracted = _extracted if _extracted is not None else _precompute_trace_rows(traces)
    per_pr: dict[str, tuple[int, dict[str, str]]] = {}
    for trace_info, rows in extracted:
        if not rows:
            continue
        latest_time = int(trace_info.get("end_time", 0) or 0)
        pr = str(trace_info.get("pr", "none"))
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
                        "test_suite": str(row["test_suite"]),
                        "test_function": str(row["test_function"]),
                        "test_module": str(row["test_module"]),
                        "test_class": str(row["test_class"]),
                        "test_line": str(row.get("test_line", "")),
                        "test_nodeid": str(row["test_nodeid"]),
                        "exception_type": str(row.get("exception_type", "")) or "unknown",
                        "stacktrace": str(row.get("exception_stacktrace", "")),
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
    _extracted: list[tuple[dict[str, str | int], list[dict[str, str | float]]]] | None = None,
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
        lines.append("# HELP pytest_pr_created_at_seconds Unix timestamp the PR was created at.")
        lines.append("# TYPE pytest_pr_created_at_seconds gauge")
        lines.extend(created_lines)
    return lines


def extract_per_run_metrics(
    traces: list[dict],
    *,
    _extracted: list[tuple[dict[str, str | int], list[dict[str, str | float]]]] | None = None,
) -> list[str]:
    """Emit metrics scoped to each workflow-level pytest run.

    Unlike ``extract_average_metrics`` which aggregates across the lookback
    window, this produces one sample per (run_id, trace_id, test_nodeid) for
    per-test duration, plus one roll-up per workflow run across every suite
    trace that shared the same run identifier. Feeds the PR dashboard (list of
    workflow runs) and the Run dashboard (list of tests in one run).
    """
    extracted = _extracted if _extracted is not None else _precompute_trace_rows(traces)
    lines = [
        "# HELP pytest_test_duration_seconds Duration of each pytest test span, labeled with run_id and trace_id.",
        "# TYPE pytest_test_duration_seconds gauge",
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
        "# HELP pytest_run_suite_member_info Whether a suite contributed tests to a pytest run.",
        "# TYPE pytest_run_suite_member_info gauge",
    ]
    run_aggregates: dict[tuple[str, str, str, str], dict[str, object]] = {}

    for trace_info, rows in extracted:
        if not rows:
            continue

        for row in rows:
            test_labels = {
                "pr": str(row["pr"]),
                "test_suite": str(row["test_suite"]),
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
                "suite_names": set(),
                "total": 0,
                "total_duration": 0.0,
                "trace_ids": set(),
            }

        aggregate = run_aggregates[run_key]
        aggregate["failed"] = int(aggregate["failed"]) + failed
        aggregate["total"] = int(aggregate["total"]) + total
        aggregate["total_duration"] = float(aggregate["total_duration"]) + total_duration
        aggregate["end_time"] = max(int(aggregate["end_time"]), int(trace_info.get("end_time", 0) or 0))

        trace_start_time = int(trace_info.get("start_time", 0) or 0)
        aggregate_start_time = int(aggregate["start_time"])
        if aggregate_start_time == 0 or (
            trace_start_time != 0 and trace_start_time < aggregate_start_time
        ):
            aggregate["start_time"] = trace_start_time

        suite_names = aggregate["suite_names"]
        trace_ids = aggregate["trace_ids"]
        assert isinstance(suite_names, set)
        assert isinstance(trace_ids, set)
        suite_names.add(str(trace_info.get("test_suite", "unknown")))
        trace_ids.add(str(trace_info.get("trace_id", "unknown")))

    for (service_name, provider, pr, run_id), aggregate in sorted(run_aggregates.items()):
        suite_names = sorted(str(suite_name) for suite_name in aggregate["suite_names"])
        trace_ids = sorted(str(trace_id) for trace_id in aggregate["trace_ids"])
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
        # Bake totals/failures/duration/rate as labels on the start-time metric
        # so a single Grafana query can drive the Runs table without any merge
        # or Grafana-side arithmetic.
        failure_rate_percent = (100.0 * failed / total) if total else 0.0
        start_labels = dict(run_labels)
        start_labels["suite_count"] = str(len(suite_names))
        start_labels["suites"] = ",".join(suite_names)
        start_labels["trace_count"] = str(len(trace_ids))
        start_labels["total_tests"] = str(total)
        start_labels["failed_tests"] = str(failed)
        start_labels["total_duration_seconds"] = f"{total_duration:.3f}"
        start_labels["failure_rate_percent"] = f"{failure_rate_percent:.2f}"
        lines.append(f"pytest_run_start_time_seconds{metric_labels(start_labels)} {start_time_seconds:.6f}")
        lines.append(f"pytest_run_end_time_seconds{metric_labels(run_labels)} {end_time_seconds:.6f}")
        lines.append(f"pytest_run_total_tests{metric_labels(run_labels)} {total}")
        lines.append(f"pytest_run_failed_tests{metric_labels(run_labels)} {failed}")
        lines.append(f"pytest_run_duration_seconds{metric_labels(run_labels)} {total_duration:.6f}")
        for suite_name in suite_names:
            suite_labels = dict(run_labels)
            suite_labels["test_suite"] = suite_name
            lines.append(f"pytest_run_suite_member_info{metric_labels(suite_labels)} 1")
    return lines


def extract_average_metrics(
    traces: list[dict],
    *,
    _extracted: list[tuple[dict[str, str | int], list[dict[str, str | float]]]] | None = None,
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
                str(row["test_suite"]),
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
                    "last_failure_stacktrace": "",
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
                    aggregates[key]["last_failure_exception_type"] = str(row.get("exception_type", "")) or "unknown"
                    aggregates[key]["last_failure_stacktrace"] = str(row.get("exception_stacktrace", ""))

    for (service_name, test_suite, pr, provider, test_nodeid), aggregate in sorted(aggregates.items()):
        durations = aggregate["durations"]
        labels = {
            "pr": pr,
            "test_suite": test_suite,
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
        lines.append(f"pytest_test_failure_count{metric_labels(labels)} {failure_count}")
        if failure_count > 0:
            pointer_labels = dict(labels)
            pointer_labels["trace_id"] = str(aggregate["last_failure_trace_id"])
            pointer_labels["exception_type"] = str(aggregate["last_failure_exception_type"])
            pointer_labels["stacktrace"] = str(aggregate["last_failure_stacktrace"])
            lines.append(f"pytest_test_last_failure_info{metric_labels(pointer_labels)} 1")

    return lines


def extract_average_resource_metrics(records: list[dict[str, str | float | int]]) -> list[str]:
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
        test_suite = str(record.get("test_suite", "unknown"))
        pr = str(record.get("pr", "none"))
        provider = str(record.get("provider", "unknown"))
        test_nodeid = str(record.get("test_nodeid", "unknown"))
        key = (service_name, test_suite, pr, provider, test_nodeid)
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
        for metric_name in ("cpu_time_seconds", "rss_delta_bytes", "rss_peak_bytes", "cuda_peak_allocated_bytes"):
            value = record.get(metric_name)
            metric_values = aggregate[metric_name]
            assert isinstance(metric_values, list)
            if isinstance(value, (int, float)):
                metric_values.append(float(value))

    for (service_name, test_suite, pr, provider, test_nodeid), aggregate in sorted(aggregates.items()):
        labels = {
            "pr": pr,
            "test_suite": test_suite,
            "provider": provider,
            "service_name": service_name,
            "test_class": str(aggregate["test_class"]),
            "test_function": str(aggregate["test_function"]),
            "test_module": str(aggregate["test_module"]),
            "test_nodeid": test_nodeid,
        }
        resource_count = len(aggregate["cpu_time_seconds"])  # type: ignore[arg-type]
        lines.append(f"pytest_test_resource_run_count{metric_labels(labels)} {resource_count}")
        for metric_name, prom_name in (
            ("cpu_time_seconds", "pytest_test_average_cpu_time_seconds"),
            ("rss_peak_bytes", "pytest_test_average_rss_peak_bytes"),
            ("rss_delta_bytes", "pytest_test_average_rss_delta_bytes"),
            ("cuda_peak_allocated_bytes", "pytest_test_average_cuda_peak_allocated_bytes"),
        ):
            metric_values = aggregate[metric_name]
            assert isinstance(metric_values, list)
            if not metric_values:
                continue
            lines.append(f"{prom_name}{metric_labels(labels)} {fsum(metric_values) / len(metric_values):.9f}")

    return lines


def _render_metrics_uncached() -> str:
    try:
        traces = fetch_traces()
        resource_records = fetch_resource_records()
    except Exception as error:
        return (
            "# HELP pytest_trace_exporter_up Whether the exporter could query Jaeger.\n"
            "# TYPE pytest_trace_exporter_up gauge\n"
            "pytest_trace_exporter_up 0\n"
            "# HELP pytest_trace_exporter_last_error Last exporter error.\n"
            "# TYPE pytest_trace_exporter_last_error gauge\n"
            f"pytest_trace_exporter_last_error{{message={json.dumps(str(error))}}} 1\n"
        )

    if not traces:
        return (
            "# HELP pytest_trace_exporter_up Whether the exporter could query Jaeger.\n"
            "# TYPE pytest_trace_exporter_up gauge\n"
            "pytest_trace_exporter_up 1\n"
        )

    extracted = _precompute_trace_rows(traces)

    rendered = [
        "# HELP pytest_trace_exporter_up Whether the exporter could query Jaeger.",
        "# TYPE pytest_trace_exporter_up gauge",
        "pytest_trace_exporter_up 1",
        "# HELP pytest_trace_exporter_trace_count Number of traces fetched from Jaeger for aggregation.",
        "# TYPE pytest_trace_exporter_trace_count gauge",
        f"pytest_trace_exporter_trace_count {len(traces)}",
    ]
    rendered.extend(extract_per_run_metrics(traces, _extracted=extracted))
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
        if self.path not in {"/metrics", "/"}:
            self.send_response(404)
            self.end_headers()
            return

        payload = render_metrics().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
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
