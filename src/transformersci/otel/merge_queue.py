"""Merge-queue history for a PR's failing tests.

A merge-queue run tests ``main`` plus the PRs queued ahead of it, so a test that
fails there for several *other* PRs is not this PR's fault. For each failing
node id this reads the recent queue runs that ran it from Tempo and gives a
verdict: likely this PR, flaky, failing for other PRs, or broken on main.

Queue runs are found by their head branch, ``gh-readonly-queue/main/pr-<N>-<sha>``;
their ``ci_event`` is unset, so it cannot be used.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor

WINDOW = 20  # runs shown per test
LOOKBACK_DAYS = 3  # ~15-37 queue runs per weekday, so this fills 20 for most tests
SEARCH_LIMIT = 1000  # traces per search
SPANS_PER_TRACE = 100  # Tempo's maximum spans per span set
NODEIDS_PER_QUERY = 40  # stay under SPANS_PER_TRACE in a trace that ran them all
MAX_NODEIDS = 60
MAX_NODEID_LENGTH = 500

MIN_RUNS = 5  # history needed before "Likely this PR"
MIN_OTHER_PRS = 2  # other PRs that must fail before it is not this PR
MAIN_PRS = 3  # other PRs before a consistent failure reads as broken on main
BROKEN_RATE = 0.8  # fail share that reads as consistent rather than flaky
STREAK = 3  # newest runs all failing, across 2+ PRs: broke recently, not flaky

# transformers-ci#275 (merged 2026-10-16T08:43:40Z; PR CI installs the plugin
# from main) made a test that fails every rerun an ERROR span. Before it such a
# test was exported OK with its exception events, and so was one that passed on
# a retry, so an OK span with an exception means "failed at least once" there.
# After it, the same span means "passed after a rerun".
RERUN_STATUS_FIXED_AT = 1792140220

_SELECT = (
    "select(status, event.exception.type, event.exception.message, "
    "resource.vcs.change.id, resource.cicd.pipeline.run.id, span.pytest.nodeid)"
)


def valid_nodeid(nodeid: str) -> bool:
    return (
        0 < len(nodeid) <= MAX_NODEID_LENGTH and "::" in nodeid and nodeid.isprintable()
    )


def _traceql_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def queries(nodeids: Iterable[str], service_name: str, repository: str) -> list[str]:
    """TraceQL searches covering ``nodeids``, a few dozen ids per query."""
    ordered = sorted(set(nodeids))
    found = []
    for index in range(0, len(ordered), NODEIDS_PER_QUERY):
        chunk = ordered[index : index + NODEIDS_PER_QUERY]
        pattern = "|".join(re.escape(nodeid) for nodeid in chunk)
        found.append(
            "{ "
            f'resource.service.name = "{_traceql_string(service_name)}" && '
            f'resource.vcs.repository.name = "{_traceql_string(repository)}" && '
            'resource.vcs.ref.head.name =~ "gh-readonly-queue/.*" && '
            f'span.pytest.nodeid =~ "{_traceql_string(pattern)}"'
            " } | " + _SELECT
        )
    return found


def _attributes(span: dict) -> dict[str, str]:
    values = {}
    for item in span.get("attributes") or []:
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        if isinstance(value, dict) and value:
            values[str(item.get("key"))] = str(next(iter(value.values())))
    return values


def parse_search(payload: object) -> list[dict]:
    """One row per matched test span in a Tempo search response."""
    if not isinstance(payload, dict):
        return []
    rows = []
    for trace in payload.get("traces") or []:
        if not isinstance(trace, dict):
            continue
        sets = trace.get("spanSets") or [trace.get("spanSet") or {}]
        seen = set()
        for span_set in sets:
            for span in (span_set or {}).get("spans") or []:
                if not isinstance(span, dict) or span.get("spanID") in seen:
                    continue
                seen.add(span.get("spanID"))
                attributes = _attributes(span)
                nodeid = attributes.get("pytest.nodeid", "")
                run_id = attributes.get("cicd.pipeline.run.id", "")
                pr = attributes.get("vcs.change.id", "")
                if not (nodeid and run_id and pr.isdigit()):
                    continue
                try:
                    started = int(span.get("startTimeUnixNano") or 0) // 10**9
                except ValueError:
                    continue
                rows.append(
                    {
                        "nodeid": nodeid,
                        "run_id": run_id,
                        "pr": pr,
                        "status": attributes.get("status", "unset"),
                        "exception_type": attributes.get("exception.type", ""),
                        "exception_message": attributes.get("exception.message", "")[
                            :300
                        ],
                        "start": started,
                    }
                )
    return rows


def fetch_rows(
    nodeids: Iterable[str],
    search: Callable[[str, int, int], object],
    *,
    service_name: str,
    repository: str,
    now: float | None = None,
) -> tuple[list[dict], bool]:
    """Search the last ``LOOKBACK_DAYS`` of queue runs, one day per request.

    ``search(traceql, start, end)`` returns Tempo's JSON. Returns the rows and
    whether any search failed or came back full (so history may be incomplete).
    """
    end = int(now if now is not None else time.time())
    jobs = [
        (query, end - (day + 1) * 86400, end - day * 86400)
        for query in queries(nodeids, service_name, repository)
        for day in range(LOOKBACK_DAYS)
    ]
    incomplete = False

    def one(job: tuple[str, int, int]) -> list[dict]:
        nonlocal incomplete
        try:
            payload = search(*job)
        except Exception:
            incomplete = True
            return []
        traces = payload.get("traces") if isinstance(payload, dict) else None
        if isinstance(traces, list) and len(traces) >= SEARCH_LIMIT:
            incomplete = True
        return parse_search(payload)

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = [row for found in pool.map(one, jobs) for row in found]
    return rows, incomplete


def outcome(row: dict) -> str:
    """failed, rerun (passed after a failed attempt) or passed."""
    if row["status"] == "error":
        return "failed"
    if row["exception_type"]:
        return "rerun" if row["start"] >= RERUN_STATUS_FIXED_AT else "failed"
    return "passed"


def _streak(runs: list[dict]) -> int:
    count = 0
    for run in runs:
        if run["status"] == "passed":
            break
        count += 1
    return count


def verdict(runs: list[dict]) -> tuple[str, str]:
    """Verdict and its one-line reason for runs of other PRs, newest first.

    A rerun counts as a failure: the test failed an attempt in someone else's PR.
    """
    ran = len(runs)
    failed = [run for run in runs if run["status"] != "passed"]
    prs = {run["pr"] for run in failed}
    if not ran:
        return (
            "none",
            f"Not run in any merge-queue run in the last {LOOKBACK_DAYS} days.",
        )
    if len(prs) >= MIN_OTHER_PRS:
        share = len(failed) / ran
        recent = _streak(runs)
        recent_prs = {run["pr"] for run in runs[:recent]}
        if ran > recent >= STREAK and len(recent_prs) >= MIN_OTHER_PRS:
            return ("main" if len(recent_prs) >= MAIN_PRS else "others"), (
                f"Started failing {recent} merge-queue runs ago, across {len(recent_prs)} PRs "
                f"({', '.join('#' + pr for pr in sorted(recent_prs))}), and has failed in every "
                "run since; it passed before that. Not caused by this PR."
            )
        if share >= BROKEN_RATE and len(prs) >= MAIN_PRS:
            return "main", (
                f"Broken on main: fails in {len(failed)} of the last {ran} merge-queue runs, "
                f"across {len(prs)} other PRs. Not caused by this PR."
            )
        if share >= BROKEN_RATE:
            return "others", (
                f"Fails for other PRs too: {len(failed)} of the last {ran} merge-queue runs, "
                f"across {len(prs)} PRs ({', '.join('#' + pr for pr in sorted(prs))}). Not caused "
                "by this PR; too few PRs yet to call it broken on main."
            )
        return "flaky", (
            f"Fails intermittently for other PRs: {len(failed)} of the last {ran} merge-queue "
            f"runs, across {len(prs)} PRs. Not caused by this PR."
        )
    if not failed and ran >= MIN_RUNS:
        return "pr", (
            f"Passed in all {ran} recent merge-queue runs of other PRs. Likely caused by this PR."
        )
    if failed:
        return "unknown", (
            f"Failed in 1 other PR's merge-queue run (#{failed[0]['pr']}) of {ran}: "
            "not enough to tell."
        )
    return "unknown", (
        f"Only {ran} merge-queue run{'s' if ran != 1 else ''} so far: not enough to tell."
    )


def history(
    rows: Iterable[dict], nodeids: Iterable[str], pr: str, now: float | None = None
) -> dict[str, dict]:
    """Per node id: recent queue runs of other PRs, counts and verdict."""
    current = now if now is not None else time.time()
    by_test: dict[str, dict[str, dict]] = {}
    for row in rows:
        if row["pr"] == pr:
            continue  # this PR's own queue attempts are not evidence about others
        status = outcome(row)
        runs = by_test.setdefault(row["nodeid"], {})
        previous = runs.get(row["run_id"])
        # One entry per run; a failing span outranks a passing one from a retry job.
        if previous and (previous["status"] != "passed" or status == "passed"):
            continue
        runs[row["run_id"]] = {
            "pr": row["pr"],
            "run_id": row["run_id"],
            "status": status,
            "reported_ok": status == "failed" and row["status"] != "error",
            "exception_type": row["exception_type"],
            "exception_message": row["exception_message"],
            "age_seconds": max(0, int(current - row["start"])),
            "log_url": "https://github.com/huggingface/transformers/actions/runs/"
            + row["run_id"].split(":", 1)[0],
        }
    tests = {}
    for nodeid in nodeids:
        runs = sorted(
            by_test.get(nodeid, {}).values(), key=lambda run: run["age_seconds"]
        )[:WINDOW]
        kind, reason = verdict(runs)
        failed = [run for run in runs if run["status"] != "passed"]
        tests[nodeid] = {
            "runs": runs,
            "ran_count": len(runs),
            "fail_count": len(failed),
            "distinct_prs_failed": len({run["pr"] for run in failed}),
            "verdict": kind,
            "reason": reason,
        }
    return tests
