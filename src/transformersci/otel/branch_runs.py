"""Branch Runs: the 10 most recent runs on branches without a PR, such as main.

``GET /branch-runs`` is the page the Overview dashboard iframes and
``/branch-runs/rows`` its data. The queries are the Grafana table's, moved
here unchanged so the rows do not change with the move; only the lane column
reads the raw ``hardware`` labels instead of a bitmask, so the page can draw
them with the PR CI view's icons (``lanes.js``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

LIMIT = 10

Query = Callable[[str], list[dict[str, Any]]]

_NO_PR = 'pr!~"[0-9]+"'
_MEMBER = (
    "max by (service_name, provider, pr, run_id) (max without (instance) "
    f'(last_over_time(pytest_run_job_member_info{{test_job=~".*",{_NO_PR}}}[7d])))'
)


def _per_run(metric: str) -> str:
    """A run-level metric, kept only for runs that have job members."""
    return (
        f"max by (run_id, pr) (max without (instance) "
        f"(last_over_time({metric}{{{_NO_PR}}}[7d])) "
        f"* on (service_name, provider, pr, run_id) group_left() {_MEMBER})"
    )


def _job_sum(metric_job: str, metric_run: str) -> str:
    return (
        f"(sum by (service_name, provider, pr, run_id) (max without (instance) "
        f"(last_over_time({metric_job}{{{_NO_PR}}}[7d]))) "
        f"or sum by (service_name, provider, pr, run_id) (max without (instance) "
        f"(last_over_time({metric_run}{{{_NO_PR}}}[7d])))) "
        f"* on (service_name, provider, pr, run_id) group_left() {_MEMBER}"
    )


# The ten newest runs: from traces, else from GitHub for runs that have not
# exported a span yet.
_NEWEST = (
    f"topk({LIMIT}, {_per_run('pytest_run_start_time_seconds')} "
    "or max by (run_id, pr) ((ci_github_run_started_timestamp_seconds "
    "or ci_github_run_created_timestamp_seconds) * on (repository, run_id) "
    'group_left(pr) max by (repository, run_id, pr) (ci_github_run_info{pr!~"[0-9]*"})))'
)


def _newest(expr: str) -> str:
    return f"({expr}) and on (run_id, pr) ({_NEWEST} * 1000)"


_TOTAL = f"max by (run_id, pr) ({_job_sum('pytest_run_job_total_tests', 'pytest_run_total_tests')})"
_FAILED = f"max by (run_id, pr) ({_job_sum('pytest_run_job_failed_tests', 'pytest_run_failed_tests')})"

QUERIES = {
    "active": (
        "(max by (run_id, pr) ((max by (repository, run_id) (ci_github_run_status) < 3) "
        "* on (repository, run_id) group_left(pr) max by (repository, run_id, pr) "
        '(ci_github_run_info{pr!~"[0-9]*"})) * 0 + 1) '
        f"or max by (run_id, pr) (last_over_time(pytest_run_active{{{_NO_PR}}}[5m]) "
        'unless on (run_id) ci_github_run_info{pr!=""})'
    ),
    "start": (
        f"({_NEWEST} or max by (run_id, pr) "
        f"(last_over_time(pytest_run_active_start_time_seconds{{{_NO_PR}}}[5m])))"
    ),
    "jobs": _newest(
        "count by (run_id, pr) (max without (instance) "
        f"(last_over_time(pytest_run_job_member_info{{{_NO_PR}}}[7d])))"
    ),
    "total": _newest(_TOTAL),
    "commit": _newest(
        "(max by (run_id, pr, commit_message, html_url) (max without (instance) "
        f"(last_over_time(pytest_run_info{{{_NO_PR}}}[7d])) "
        f"* on (service_name, provider, pr, run_id) group_left() {_MEMBER})) "
        "or on (run_id, pr) max by (run_id, pr, commit_message, html_url) "
        '(ci_github_run_commit_info{pr!~"[0-9]*"})'
    ),
    "duration": _newest(_per_run("pytest_run_duration_seconds")),
    "rate": _newest(f"100 * {_FAILED} / {_TOTAL}"),
    "wall": _newest(_per_run("pytest_run_wall_seconds")),
    "lanes": _newest(
        "group by (run_id, pr, hardware) "
        f"(last_over_time(pytest_run_job_member_info{{{_NO_PR}}}[7d]))"
    ),
}


def _samples(result: list[dict[str, Any]]) -> list[tuple[dict[str, str], float]]:
    out = []
    for item in result:
        try:
            out.append((item.get("metric", {}), float(item["value"][1])))
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return out


def branch_rows(replies: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Join the replies per (run_id, pr), newest first, at most ``LIMIT``."""

    def by_run(name: str) -> dict[tuple[str, str], float]:
        return {
            (labels.get("run_id", ""), labels.get("pr", "")): value
            for labels, value in _samples(replies.get(name, []))
        }

    starts = by_run("start")
    values = {
        name: by_run(name) for name in ("jobs", "total", "duration", "rate", "wall")
    }
    active = by_run("active")
    commits = {
        (labels.get("run_id", ""), labels.get("pr", "")): labels
        for labels, _ in _samples(replies.get("commit", []))
    }
    lanes: dict[tuple[str, str], list[str]] = {}
    for labels, _ in _samples(replies.get("lanes", [])):
        key = (labels.get("run_id", ""), labels.get("pr", ""))
        if labels.get("hardware"):
            lanes.setdefault(key, []).append(labels["hardware"])
    rows = []
    for key, started in sorted(starts.items(), key=lambda item: -item[1])[:LIMIT]:
        run_id, branch = key
        commit = commits.get(key, {})
        rows.append(
            {
                "run_id": run_id,
                "branch": branch,
                "live": key in active,
                "commit_message": commit.get("commit_message", ""),
                "html_url": commit.get("html_url", ""),
                "started": started,
                "lanes": sorted(set(lanes.get(key, []))),
                **{name: values[name].get(key) for name in values},
            }
        )
    return rows


def build_rows(query: Query) -> dict[str, Any]:
    replies = {name: query(expr) for name, expr in QUERIES.items()}
    return {"rows": branch_rows(replies)}
