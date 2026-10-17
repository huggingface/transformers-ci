"""Project health: failing tests on ``main`` that people can pick up.

``GET /project-health`` is the page, ``/project-health/rows`` the table,
``/project-health/trend`` the merge-run chart above it and
``/project-health/events`` the models, dependency changes and PR/issue counts
above that. Everything is read from
Prometheus; this module owns the queries and turns their replies into rows, so
the rules (sticky, the day strip, which runs count as merges) are Python with
tests instead of PromQL in a dashboard.

The sticky rule is Main health's, reproduced exactly so the sticky set does not
change when Main health is replaced: a test is sticky when it failed on at
least two UTC days in the window and has not been cleared, where cleared means
its job has had four full runs (at least half the job's largest test count)
since the test last failed.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

DAYS_CHOICES = (7, 14, 30)
DEFAULT_DAYS = 14
DAY = 86400

# The failure metric carries no `hardware` label, so GPU vs CPU comes from the
# job name, as on Main health.
GPU_JOB_PATTERN = r".*gpu.*|tests_fsdp_ci|tests_tensor_parallel_ci"

# A sticky failure is cleared after this many full runs of its job since.
CLEARING_RUNS = 4
# A run counts as full when it ran at least this share of the job's largest run.
FULL_RUN_SHARE = 0.5
STICKY_MIN_DAYS = 2

# Merge runs. GPU merges carry ci_event="push"; CPU merges carry
# ci_event="none" (measured 2026-10-17: no CPU run on main has "push").
MERGE_SELECTORS = {
    "gpu": 'pr="main",ci_event="push",test_job="run_models_gpu"',
    "cpu": 'pr="main",hardware="cpu",ci_event="none"',
}

Query = Callable[[str], list[dict[str, Any]]]
# (expr, start, end, step) -> a Prometheus matrix result.
RangeQuery = Callable[[str, float, float, float], list[dict[str, Any]]]

# Slow: the N slowest tests on main per runner, ranked by their average daily
# duration. No threshold, only a ranking (plan, 2026-10-17).
SLOW_TOP = 20
# Each day's slowest tests per runner fetched from Prometheus; ranking happens
# over those, so a test must be among a day's slowest to count that day.
SLOW_DAILY_POOL = 60


def is_gpu_job(job: str) -> bool:
    return re.fullmatch(GPU_JOB_PATTERN, job) is not None


def parse_days(raw: str) -> int:
    try:
        days = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_DAYS
    return days if days in DAYS_CHOICES else DEFAULT_DAYS


def _samples(result: list[dict[str, Any]]) -> list[tuple[dict[str, str], float]]:
    samples = []
    for item in result:
        if not isinstance(item, dict):
            continue
        metric, value = item.get("metric"), item.get("value")
        if not isinstance(metric, dict) or not isinstance(value, list):
            continue
        try:
            samples.append(
                ({str(k): str(v) for k, v in metric.items()}, float(value[1]))
            )
        except (IndexError, TypeError, ValueError):
            continue
    return samples


def model_name(nodeid: str) -> str:
    """``tests/models/qwen2_moe/...`` -> ``qwen2_moe``; otherwise the test directory."""
    parts = nodeid.split("::", 1)[0].split("/")
    if len(parts) >= 3 and parts[0] == "tests" and parts[1] == "models":
        return parts[2]
    if len(parts) >= 2 and parts[0] == "tests":
        return parts[1] if len(parts) > 2 else parts[1].removesuffix(".py")
    return parts[0]


def failure_queries(days: int) -> dict[str, str]:
    window = f"{days}d"
    return {
        # One series per failing run; trace_id and exception_type belong to that run.
        "failures": (
            "max by (test_nodeid, test_job, run_id, trace_id, exception_type) "
            f'(last_over_time(pytest_test_last_failure_info{{pr="main"}}[{window}]))'
        ),
        "starts": (
            "max by (run_id) "
            f'(last_over_time(pytest_run_start_time_seconds{{pr="main"}}[{window}]))'
        ),
        "job_runs": (
            "max by (run_id, test_job) "
            f'(last_over_time(pytest_run_job_total_tests{{pr="main"}}[{window}]))'
        ),
        "job_max": (
            "max by (test_job) "
            f'(max_over_time(pytest_run_job_total_tests{{pr="main"}}[{window}]))'
        ),
    }


def _cleared_after(
    job_runs: list[tuple[dict[str, str], float]],
    job_max: list[tuple[dict[str, str], float]],
    starts: dict[str, float],
) -> dict[str, float]:
    """Per job, the start of the oldest of its last four full runs.

    A failure older than that has been followed by four full runs and is
    cleared. Jobs with fewer than four full runs in the window clear nothing,
    so they are absent.
    """
    largest = {labels.get("test_job", ""): value for labels, value in job_max}
    full: dict[str, list[float]] = defaultdict(list)
    for labels, total in job_runs:
        job, run_id = labels.get("test_job", ""), labels.get("run_id", "")
        if run_id not in starts or job not in largest:
            continue
        if total >= FULL_RUN_SHARE * largest[job]:
            full[job].append(starts[run_id])
    return {
        job: sorted(times, reverse=True)[CLEARING_RUNS - 1]
        for job, times in full.items()
        if len(times) >= CLEARING_RUNS
    }


def failure_rows(
    failures: list[dict[str, Any]],
    starts: list[dict[str, Any]],
    job_runs: list[dict[str, Any]],
    job_max: list[dict[str, Any]],
    *,
    now: float,
    days: int,
) -> list[dict[str, Any]]:
    """One row per (test, job) that failed on main in the window."""
    start_of = {labels.get("run_id", ""): value for labels, value in _samples(starts)}
    cleared_after = _cleared_after(_samples(job_runs), _samples(job_max), start_of)
    today = int(now // DAY)
    first_day = today - days + 1

    runs: dict[tuple[str, str], dict[str, tuple[float, dict[str, str]]]] = defaultdict(
        dict
    )
    for labels, _ in _samples(failures):
        run_id = labels.get("run_id", "")
        if run_id not in start_of:
            # Main health joins on the run's start time and drops what has none.
            continue
        key = (labels.get("test_nodeid", ""), labels.get("test_job", ""))
        previous = runs[key].get(run_id)
        if previous is None or labels.get("trace_id", "") > previous[1].get(
            "trace_id", ""
        ):
            runs[key][run_id] = (start_of[run_id], labels)

    rows = []
    for (nodeid, job), by_run in runs.items():
        if not nodeid or not job:
            continue
        times = sorted(start for start, _ in by_run.values())
        last_seen = times[-1]
        latest = max(by_run.values(), key=lambda item: item[0])[1]
        fail_days = {int(start // DAY) for start in times}
        strip = [0] * days
        for start in times:
            index = int(start // DAY) - first_day
            if 0 <= index < days:
                strip[index] += 1
        cleared = job in cleared_after and last_seen < cleared_after[job]
        rows.append(
            {
                "category": "failure",
                "nodeid": nodeid,
                "job": job,
                "runner": "GPU" if is_gpu_job(job) else "CPU",
                "model": model_name(nodeid),
                "test": nodeid.split("::", 1)[1] if "::" in nodeid else nodeid,
                "signal": latest.get("exception_type", "") or "failed",
                "trace_id": latest.get("trace_id", ""),
                "run_id": latest.get("run_id", ""),
                "fails": len(times),
                "bad_days": len(fail_days),
                "days": strip,
                "last_seen": last_seen,
                "sticky": len(fail_days) >= STICKY_MIN_DAYS and not cleared,
                "cleared": cleared,
            }
        )
    rows.sort(key=lambda row: (-row["bad_days"], -row["last_seen"], row["nodeid"]))
    return rows


def slow_queries() -> dict[str, str]:
    """Per runner, each day's slowest main tests (evaluated once per UTC day)."""
    duration = (
        "max by (test_nodeid, test_job) "
        '(max_over_time(pytest_main_test_duration_seconds{{pr="main",test_job{op}"{jobs}"}}[1d]))'
    )
    return {
        runner: f"topk({SLOW_DAILY_POOL}, {duration.format(op=op, jobs=GPU_JOB_PATTERN)})"
        for runner, op in (("GPU", "=~"), ("CPU", "!~"))
    }


def slow_rows(
    matrices: dict[str, list[dict[str, Any]]], *, now: float, days: int
) -> list[dict[str, Any]]:
    """One row per (test, job) among each runner's SLOW_TOP slowest on main."""
    first_day = int(now // DAY) - days + 1
    rows = []
    for runner, result in matrices.items():
        per_test: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
        for item in result:
            labels = item.get("metric") or {}
            key = (str(labels.get("test_nodeid", "")), str(labels.get("test_job", "")))
            if not key[0] or not key[1]:
                continue
            for point in item.get("values") or []:
                try:
                    ts, value = float(point[0]), float(point[1])
                except (IndexError, TypeError, ValueError):
                    continue
                index = int(ts // DAY) - first_day
                if 0 <= index < days:
                    per_test[key][index] = value
        # A day "ranks" a test when it is among that day's SLOW_TOP for its runner.
        ranked_days: dict[tuple[str, str], set[int]] = defaultdict(set)
        for index in range(days):
            day = sorted(
                (values[index], key)
                for key, values in per_test.items()
                if index in values
            )
            for _, key in day[-SLOW_TOP:]:
                ranked_days[key].add(index)
        averages = sorted(
            ((sum(v.values()) / len(v), key) for key, v in per_test.items() if v),
            reverse=True,
        )[:SLOW_TOP]
        for average, (nodeid, job) in averages:
            values = per_test[(nodeid, job)]
            strip = [
                1 if index in ranked_days[(nodeid, job)] else 0 for index in range(days)
            ]
            last_index = max(values)
            rows.append(
                {
                    "category": "slow",
                    "nodeid": nodeid,
                    "job": job,
                    "runner": runner,
                    "model": model_name(nodeid),
                    "test": nodeid.split("::", 1)[1] if "::" in nodeid else nodeid,
                    "signal": f"{average:.0f} s avg",
                    "seconds": average,
                    "durations": [values.get(index) for index in range(days)],
                    "trace_id": "",
                    "run_id": "",
                    "fails": sum(strip),
                    "bad_days": sum(strip),
                    "days": strip,
                    "last_seen": min(now, (first_day + last_index + 1) * DAY - 1),
                    "sticky": False,
                    "cleared": False,
                }
            )
    return rows


def build_rows(
    query: Query,
    *,
    now: float,
    days: int,
    query_range: RangeQuery | None = None,
) -> dict[str, Any]:
    replies = {name: query(expr) for name, expr in failure_queries(days).items()}
    rows = failure_rows(
        replies["failures"],
        replies["starts"],
        replies["job_runs"],
        replies["job_max"],
        now=now,
        days=days,
    )
    if query_range is not None:
        today = int(now // DAY)
        start = (today - days + 2) * DAY - 1  # end of the window's first UTC day
        end = (today + 1) * DAY - 1  # end of today
        matrices = {
            runner: query_range(expr, start, end, DAY)
            for runner, expr in slow_queries().items()
        }
        rows += slow_rows(matrices, now=now, days=days)
    return {
        "days": days,
        "generated_at": now,
        "first_day": (int(now // DAY) - days + 1) * DAY,
        "rows": rows,
    }


def trend_queries(days: int) -> dict[str, str]:
    window = f"{days}d"
    queries = {}
    for runner, selector in MERGE_SELECTORS.items():
        for kind in ("failed", "total"):
            queries[f"{runner}_{kind}"] = (
                "sum by (run_id) (max by (run_id, hardware, test_job) "
                f"(max_over_time(pytest_run_job_{kind}_tests{{{selector}}}[{window}])))"
            )
    queries["starts"] = failure_queries(days)["starts"]
    return queries


def merge_trend(
    replies: dict[str, list[dict[str, Any]]], *, now: float, days: int
) -> dict[str, Any]:
    """Per UTC day, the average share of failing tests in a merge run.

    Each run's rate is its failed tests over its tests (1 of 100 is 1%); a
    day's value is the mean of its runs' rates, so one huge run does not drown
    the others. A run is one run_id: a GPU run's single- and multi-GPU halves
    are summed into one merge. The two runners' rates are not on the same
    scale (a GPU merge run is ~11k tests per GPU type, a CPU one the full ~199k
    suite), so the page charts them separately. How many runs had any failure
    is kept alongside, for the tooltip.
    """
    start_of = {
        labels.get("run_id", ""): value
        for labels, value in _samples(replies.get("starts", []))
    }
    first_day = int(now // DAY) - days + 1
    series: dict[str, Any] = {}
    for runner in MERGE_SELECTORS:
        failed = {
            labels.get("run_id", ""): value
            for labels, value in _samples(replies.get(f"{runner}_failed", []))
        }
        buckets = [
            {"runs": 0, "failing": 0, "tests": 0.0, "failed": 0.0, "rate": None}
            for _ in range(days)
        ]
        rates: list[list[float]] = [[] for _ in range(days)]
        for labels, total in _samples(replies.get(f"{runner}_total", [])):
            run_id = labels.get("run_id", "")
            if total <= 0 or run_id not in start_of:
                continue
            index = int(start_of[run_id] // DAY) - first_day
            if not 0 <= index < days:
                continue
            bucket = buckets[index]
            bucket["runs"] += 1
            bucket["tests"] += total
            bucket["failed"] += failed.get(run_id, 0.0)
            if failed.get(run_id, 0.0) > 0:
                bucket["failing"] += 1
            rates[index].append(failed.get(run_id, 0.0) / total)
        for bucket, day_rates in zip(buckets, rates):
            if day_rates:
                bucket["rate"] = sum(day_rates) / len(day_rates)
        every_rate = [rate for day_rates in rates for rate in day_rates]
        runs = sum(bucket["runs"] for bucket in buckets)
        failing = sum(bucket["failing"] for bucket in buckets)
        tests = sum(bucket["tests"] for bucket in buckets)
        series[runner] = {
            "days": buckets,
            "runs": runs,
            "failing": failing,
            "share": failing / runs if runs else None,
            "rate": sum(every_rate) / len(every_rate) if every_rate else None,
            "test_rate": sum(b["failed"] for b in buckets) / tests if tests else None,
        }
    return {"days": days, "first_day": first_day * DAY, "series": series}


def build_trend(query: Query, *, now: float, days: int) -> dict[str, Any]:
    replies = {name: query(expr) for name, expr in trend_queries(days).items()}
    return merge_trend(replies, now=now, days=days)


# ---- events: what changed on main, next to how busy the repository was ----
# Both series are info gauges whose value is the merge time in epoch ms, so
# the window's max_over_time keeps every event that was exported in it.
EVENT_QUERIES = {
    "model": "max by (model, pr) (max_over_time(pytest_model_added_info[{window}]))",
    "dependency": (
        "max by (dependency, change, pr) "
        "(max_over_time(pytest_dependency_change_info[{window}]))"
    ),
}
ACTIVITY_REPOSITORY = "huggingface/transformers"
# One GitHub search per series and day; each is an issueCount, never a page.
ACTIVITY_SEARCHES = {
    "prs_opened": "is:pr created:{day}",
    "prs_merged": "is:pr merged:{day}",
    "issues_opened": "is:issue created:{day}",
}

# query -> the GraphQL reply's `data` object.
GraphQL = Callable[[str], dict[str, Any]]


def _utc_day(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(ts))


def activity_query(first_day: int, days: int) -> str:
    """One GraphQL query counting every series for every day of the window."""
    fields = []
    for index in range(days):
        day = _utc_day((first_day + index) * DAY)
        for name, search in ACTIVITY_SEARCHES.items():
            text = f"repo:{ACTIVITY_REPOSITORY} {search.format(day=day)}"
            fields.append(
                f'{name}_{index}: search(query: "{text}", type: ISSUE, first: 0) '
                "{ issueCount }"
            )
    return "{ " + " ".join(fields) + " }"


def activity_counts(data: dict[str, Any], days: int) -> list[dict[str, int | None]]:
    return [
        {
            name: (data.get(f"{name}_{index}") or {}).get("issueCount")
            for name in ACTIVITY_SEARCHES
        }
        for index in range(days)
    ]


def project_events(
    replies: dict[str, list[dict[str, Any]]], *, now: float, days: int
) -> list[dict[str, Any]]:
    """Models added and dependency changes merged inside the window, oldest first."""
    start = (int(now // DAY) - days + 1) * DAY
    events = []
    for kind, result in replies.items():
        for labels, value in _samples(result):
            at = value / 1000
            if not start <= at <= now:
                continue
            pr = labels.get("pr", "none")
            event: dict[str, Any] = {
                "kind": kind,
                "at": at,
                "pr": "" if pr == "none" else pr,
            }
            if kind == "model":
                event["title"] = labels.get("model", "")
            else:
                event["title"] = (
                    f"{labels.get('dependency', '')} {labels.get('change', '')}".strip()
                )
            events.append(event)
    return sorted(events, key=lambda event: (event["at"], event["title"]))


def build_events(
    query: Query, graphql: GraphQL | None, *, now: float, days: int
) -> dict[str, Any]:
    """Events from Prometheus plus per-day GitHub counts. Counts are best effort:
    without GitHub the chart still shows the events."""
    first_day = int(now // DAY) - days + 1
    replies = {
        kind: query(expr.format(window=f"{days}d"))
        for kind, expr in EVENT_QUERIES.items()
    }
    counts = None
    error = ""
    if graphql is not None:
        try:
            counts = activity_counts(graphql(activity_query(first_day, days)), days)
        except Exception as exc:
            error = str(exc) or type(exc).__name__
    return {
        "days": days,
        "first_day": first_day * DAY,
        "generated_at": now,
        "repository": ACTIVITY_REPOSITORY,
        "events": project_events(replies, now=now, days=days),
        "counts": counts,
        "counts_error": error,
    }
