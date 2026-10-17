"""Branch Runs: the most recent runs on branches without a PR, such as main.

``GET /branch-runs`` is the page the Overview dashboard iframes and
``/branch-runs/rows`` its data. The queries are the Grafana table's, moved
here unchanged so the rows did not change with the move (the defaults below
reproduce them exactly); only the lane column reads the raw ``hardware``
labels instead of a bitmask, so the page can draw them with the shared lane
icons (``common.js``). The page picks the window, the row count and whether
to show ``main`` only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

DAYS_CHOICES = (7, 14, 21)
LIMIT_CHOICES = (10, 20, 30)

Query = Callable[[str], list[dict[str, Any]]]


def _choice(raw: str, choices: tuple[int, ...]) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return choices[0]
    return value if value in choices else choices[0]


def parse_params(params: dict[str, list[str]]) -> tuple[int, int, bool]:
    """(days, limit, main_only) from the query string, defaulting to the old panel."""
    first = lambda name: (params.get(name) or [""])[0]  # noqa: E731
    return (
        _choice(first("days"), DAYS_CHOICES),
        _choice(first("limit"), LIMIT_CHOICES),
        first("main") in {"1", "true"},
    )


def queries(days: int = 7, limit: int = 10, main_only: bool = False) -> dict[str, str]:
    # Traces label a branch run's pr with the branch; GitHub's metrics leave it
    # empty-or-branch, hence the two spellings of "not a PR number".
    no_pr = 'pr="main"' if main_only else 'pr!~"[0-9]+"'
    github = 'pr="main"' if main_only else 'pr!~"[0-9]*"'
    window = f"[{days}d]"
    member = (
        "max by (service_name, provider, pr, run_id) (max without (instance) "
        f'(last_over_time(pytest_run_job_member_info{{test_job=~".*",{no_pr}}}{window})))'
    )

    def per_run(metric: str) -> str:
        """A run-level metric, kept only for runs that have job members."""
        return (
            f"max by (run_id, pr) (max without (instance) "
            f"(last_over_time({metric}{{{no_pr}}}{window})) "
            f"* on (service_name, provider, pr, run_id) group_left() {member})"
        )

    def job_sum(metric_job: str, metric_run: str) -> str:
        return (
            f"max by (run_id, pr) ((sum by (service_name, provider, pr, run_id) "
            f"(max without (instance) (last_over_time({metric_job}{{{no_pr}}}{window}))) "
            f"or sum by (service_name, provider, pr, run_id) (max without (instance) "
            f"(last_over_time({metric_run}{{{no_pr}}}{window})))) "
            f"* on (service_name, provider, pr, run_id) group_left() {member})"
        )

    # The newest runs: from traces, else from GitHub for runs that have not
    # exported a span yet.
    newest = (
        f"topk({limit}, {per_run('pytest_run_start_time_seconds')} "
        "or max by (run_id, pr) ((ci_github_run_started_timestamp_seconds "
        "or ci_github_run_created_timestamp_seconds) * on (repository, run_id) "
        f"group_left(pr) max by (repository, run_id, pr) (ci_github_run_info{{{github}}})))"
    )

    def among_newest(expr: str) -> str:
        return f"({expr}) and on (run_id, pr) ({newest} * 1000)"

    total = job_sum("pytest_run_job_total_tests", "pytest_run_total_tests")
    failed = job_sum("pytest_run_job_failed_tests", "pytest_run_failed_tests")
    return {
        "active": (
            "(max by (run_id, pr) ((max by (repository, run_id) (ci_github_run_status) < 3) "
            "* on (repository, run_id) group_left(pr) max by (repository, run_id, pr) "
            f"(ci_github_run_info{{{github}}})) * 0 + 1) "
            f"or max by (run_id, pr) (last_over_time(pytest_run_active{{{no_pr}}}[5m]) "
            'unless on (run_id) ci_github_run_info{pr!=""})'
        ),
        "start": (
            f"({newest} or max by (run_id, pr) "
            f"(last_over_time(pytest_run_active_start_time_seconds{{{no_pr}}}[5m])))"
        ),
        "jobs": among_newest(
            "count by (run_id, pr) (max without (instance) "
            f"(last_over_time(pytest_run_job_member_info{{{no_pr}}}{window})))"
        ),
        "total": among_newest(total),
        "commit": among_newest(
            "(max by (run_id, pr, commit_message, html_url) (max without (instance) "
            f"(last_over_time(pytest_run_info{{{no_pr}}}{window})) "
            f"* on (service_name, provider, pr, run_id) group_left() {member})) "
            "or on (run_id, pr) max by (run_id, pr, commit_message, html_url) "
            f"(ci_github_run_commit_info{{{github}}})"
        ),
        "duration": among_newest(per_run("pytest_run_duration_seconds")),
        "rate": among_newest(f"100 * {failed} / {total}"),
        "wall": among_newest(per_run("pytest_run_wall_seconds")),
        "lanes": among_newest(
            "group by (run_id, pr, hardware) "
            f"(last_over_time(pytest_run_job_member_info{{{no_pr}}}{window}))"
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


def branch_rows(
    replies: dict[str, list[dict[str, Any]]], limit: int = 10
) -> list[dict[str, Any]]:
    """Join the replies per (run_id, pr), newest first, at most ``limit``."""

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
    for key, started in sorted(starts.items(), key=lambda item: -item[1])[:limit]:
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


def build_rows(
    query: Query, days: int = 7, limit: int = 10, main_only: bool = False
) -> dict[str, Any]:
    replies = {
        name: query(expr) for name, expr in queries(days, limit, main_only).items()
    }
    return {
        "days": days,
        "limit": limit,
        "main_only": main_only,
        "rows": branch_rows(replies, limit),
    }
