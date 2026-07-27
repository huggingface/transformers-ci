#!/usr/bin/env python
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

"""CPU-main failing-test selector → Serge auto-fix dispatcher.

This is the first, scriptable version of a "main branch CPU failure" workflow:

  1. Query Prometheus for branch-keyed main failure pointers.
  2. For each candidate, query the last 7 daily windows.
  3. Keep tests that still have failure evidence in every daily window.
  4. Select 3 candidates.
  5. In dry-run mode, print the exact Serge payloads. With ``--dispatch``, post
     one Serge task per selected test. The task report is shaped so Serge's
     reproduce/verify gate can parse the targeted node id: reproduce first on
     the baseline, then verify the candidate red-to-green before opening a PR.

The data path intentionally goes through Grafana's public Prometheus datasource
proxy, matching the dashboard, and uses Tempo only to enrich a selected task
with the captured exception stacktrace when a trace id is available.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from . import dashboard_failure_triage as dft
from .github_api import (
    create_issue,
    find_open_issue_by_marker,
    list_open_pulls,
    match_pr,
    update_issue_body,
)
from .serge_dispatch import (
    SergeDispatchError,
    build_task_payload,
    dispatch_to_serge,
    mint_serge_oidc_token,
    poll_serge_task,
)

DEFAULT_GRAFANA_URL = os.environ.get(
    "TRANSFORMERS_CI_GRAFANA_URL", "https://transformers-ci.lor-e.huggingface.cool"
)
DEFAULT_PROM_UID = os.environ.get("PROM_DATASOURCE_UID", "prometheus")
DEFAULT_REPO = "huggingface/transformers"
DEFAULT_CPU_JOB_RE = r".*[gG][pP][uU].*"

_STATE_SOURCE = "cpu-main-failure-triage"

_INSTRUCTION = (
    "Fix the single failing CPU test on the transformers main branch described "
    "in the report below. The report was selected from Prometheus because the "
    "test has a branch-keyed main failure pointer and failure evidence across "
    "the required daily windows.\n\n"
    "This task must use the same reproduce-first / verify-before-PR discipline "
    "as the slow-test flow: first reproduce the targeted node id on the baseline "
    "tree and bail without LLM work if it is already green; after producing a "
    "candidate patch, verify the same targeted node id red-to-green before "
    "opening a PR. Do not publish a PR for a patch that was not verified against "
    "the targeted test.\n\n"
    "Investigate the root cause and propose a minimal patch that makes the test "
    "pass without touching unrelated code. If the failure cannot be fixed safely, "
    "produce no patch.\n\n"
    "Ground yourself in the repository's own conventions before editing: read "
    "the root `AGENTS.md` / `CLAUDE.md` and `.ai/AGENTS.md`, then read the "
    "failing test file and the relevant model code. If the model directory has "
    "a `modular_<name>.py`, edit that source file rather than generated "
    "`modeling_*.py` files. Do not edit inside `# Copied from ...` blocks unless "
    "you intentionally break the copy link. Keep any "
    "`<!-- serge-cpu-main-fix:... -->` HTML comment from the report in the PR "
    "body."
)


class PrometheusQueryError(Exception):
    """Prometheus query failed or returned an unexpected response."""


def _quote_label_value(value: str) -> str:
    return json.dumps(value)


def _metric_selector(
    *,
    pr: str = "main",
    test_job: str | None = None,
    test_nodeid: str | None = None,
    exclude_job_regex: str | None = DEFAULT_CPU_JOB_RE,
) -> str:
    labels = [f"pr={_quote_label_value(pr)}"]
    if test_job is not None:
        labels.append(f"test_job={_quote_label_value(test_job)}")
    elif exclude_job_regex:
        labels.append(f"test_job!~{_quote_label_value(exclude_job_regex)}")
    if test_nodeid is not None:
        labels.append(f"test_nodeid={_quote_label_value(test_nodeid)}")
    return "pytest_test_last_failure_info{" + ",".join(labels) + "}"


def recent_failures_query(
    *,
    pr: str = "main",
    limit: int = 10,
    lookback: str = "90d",
    exclude_job_regex: str | None = DEFAULT_CPU_JOB_RE,
) -> str:
    selector = _metric_selector(
        pr=pr,
        exclude_job_regex=exclude_job_regex,
    )
    return (
        f"topk({limit}, topk by (test_job, test_nodeid) "
        f"(1, timestamp(max without (instance) ({selector})) * 1000))"
    )


def latest_failure_detail_query(
    candidate: dict,
    *,
    pr: str = "main",
    lookback: str = "90d",
) -> str:
    selector = _metric_selector(
        pr=pr,
        test_job=candidate["test_job"],
        test_nodeid=candidate["test_nodeid"],
        exclude_job_regex=None,
    )
    return (
        "topk(1, timestamp(max without (instance) "
        f"(last_over_time({selector}[{lookback}]))) * 1000)"
    )


def persistence_query(candidate: dict, *, pr: str = "main") -> str:
    selector = _metric_selector(
        pr=pr,
        test_job=candidate["test_job"],
        test_nodeid=candidate["test_nodeid"],
        exclude_job_regex=None,
    )
    return f"max by (test_job, test_nodeid) (last_over_time({selector}[24h]))"


def _proxy_url(
    grafana_url: str, datasource_uid: str, path: str, params: dict[str, object]
) -> str:
    return (
        f"{grafana_url.rstrip('/')}/api/datasources/proxy/uid/"
        f"{urllib.parse.quote(datasource_uid)}{path}?"
        f"{urllib.parse.urlencode(params)}"
    )


def _get_json(url: str, timeout: int) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise PrometheusQueryError(f"HTTP {e.code} querying Prometheus: {detail}")
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        raise PrometheusQueryError(f"could not query Prometheus: {e}")
    if payload.get("status") != "success":
        raise PrometheusQueryError(f"Prometheus query failed: {payload}")
    return payload


def prom_query(
    query: str,
    *,
    eval_time: float | None = None,
    grafana_url: str = DEFAULT_GRAFANA_URL,
    datasource_uid: str = DEFAULT_PROM_UID,
    timeout: int = 60,
) -> list[dict]:
    params: dict[str, object] = {"query": query}
    if eval_time is not None:
        params["time"] = int(eval_time)
    url = _proxy_url(grafana_url, datasource_uid, "/api/v1/query", params)
    return _get_json(url, timeout)["data"]["result"]


def _value_as_float(value: object) -> float:
    if isinstance(value, list) and len(value) >= 2:
        value = value[1]
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_recent_failures(series: list[dict]) -> list[dict]:
    out: list[dict] = []
    for item in series:
        metric = item.get("metric") or {}
        test_job = metric.get("test_job")
        test_nodeid = metric.get("test_nodeid")
        if not test_job or not test_nodeid:
            continue
        last_failed_ms = _value_as_float(item.get("value"))
        out.append(
            {
                "test_job": str(test_job),
                "test_nodeid": str(test_nodeid),
                "last_failed_ms": last_failed_ms,
            }
        )
    out.sort(key=lambda c: c["last_failed_ms"], reverse=True)
    return out


def daily_failure_count(range_series: list[dict]) -> int:
    stamps: set[int] = set()
    for item in range_series:
        for ts, value in item.get("values") or []:
            if _value_as_float(value) > 0:
                stamps.add(int(float(ts)))
    return len(stamps)


def has_failure_evidence(series: list[dict]) -> bool:
    for item in series:
        if _value_as_float(item.get("value")) > 0:
            return True
    return False


def failure_fingerprint(candidate: dict, failure: dict) -> str:
    basis = {
        "source": _STATE_SOURCE,
        "test_job": candidate.get("test_job") or "",
        "test_nodeid": candidate.get("test_nodeid") or "",
        "exception_type": failure.get("exception_type") or "",
        "message": dft._normalize_message(failure.get("exception_message") or ""),
    }
    raw = json.dumps(basis, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def fingerprint_marker(fingerprint: str) -> str:
    return f"<!-- serge-cpu-main-fix:sha256:{fingerprint} -->"


def task_branch_prefix(fingerprint: str) -> str:
    return f"serge/fix/cpu-main-{fingerprint[:12]}"


def run_key_for_today(now: datetime.datetime | None = None) -> str:
    now = now or datetime.datetime.now(datetime.timezone.utc)
    return now.date().isoformat()


def tracking_issue_marker(run_key: str) -> str:
    return f"<!-- serge-triage-run:{_STATE_SOURCE}:{run_key} -->"


def _md_cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def tracking_issue_title(run_key: str) -> str:
    return f"[serge] CPU main failing tests triage {run_key}"


def render_tracking_issue_body(
    rows: list[dict],
    run_key: str,
    *,
    statuses: dict[str, str] | None = None,
    task_urls: dict[str, str] | None = None,
    pr_numbers: dict[str, int | None] | None = None,
) -> str:
    statuses = statuses or {}
    task_urls = task_urls or {}
    pr_numbers = pr_numbers or {}
    lines = [
        tracking_issue_marker(run_key),
        "",
        f"Automated **CPU main failure triage** for `{run_key}`.",
        "",
        "This issue was generated by AI-assisted automation. Serge should reproduce "
        "each targeted test on the baseline first, then verify red-to-green before "
        "opening a PR.",
        "",
        "## Dispatched CPU failures",
        "",
        "| Job | Test | Error | Seen | Serge task | PR |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        fp = row["fingerprint"]
        candidate = row["candidate"]
        failure = row["failure"]
        task_url = task_urls.get(fp)
        task_cell = f"[task]({task_url})" if task_url else "—"
        pr = pr_numbers.get(fp)
        status = statuses.get(fp)
        if pr:
            pr_cell = f"#{pr}"
        elif status == "no_fix":
            pr_cell = "no fix"
        elif status == "error":
            pr_cell = "task failed"
        elif status:
            pr_cell = f"{status}"
        else:
            pr_cell = f"`{task_branch_prefix(fp)}` (pending)"
        cells = [
            f"`{candidate.get('test_job') or '?'}`",
            f"`{candidate.get('test_nodeid') or '?'}`",
            failure.get("exception_type") or candidate.get("exception_type") or "?",
            f"{candidate.get('daily_windows_seen')}/{candidate.get('required_daily_windows') or 7}",
            task_cell,
            pr_cell,
        ]
        lines.append("| " + " | ".join(_md_cell(c) for c in cells) + " |")
    lines += [
        "",
        f"_Generated {datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()}._",
    ]
    return "\n".join(lines)


def ensure_tracking_issue(
    repo: str, run_key: str, body: str, github_token: str | None
) -> int | None:
    if not github_token:
        return None
    marker = tracking_issue_marker(run_key)
    try:
        existing = find_open_issue_by_marker(repo, marker, github_token)
        if existing is not None:
            update_issue_body(repo, existing, body, github_token)
            return existing
        return create_issue(repo, tracking_issue_title(run_key), body, github_token)
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        reason = getattr(e, "reason", e)
        print(f"      warning: could not create/update tracking issue: {reason}")
        return None


def enrich_latest_failure(
    candidate: dict,
    *,
    pr: str,
    lookback: str,
    grafana_url: str,
    datasource_uid: str,
) -> dict:
    result = prom_query(
        latest_failure_detail_query(candidate, pr=pr, lookback=lookback),
        grafana_url=grafana_url,
        datasource_uid=datasource_uid,
    )
    if not result:
        return dict(candidate)
    metric = result[0].get("metric") or {}
    enriched = {**candidate, **{k: str(v) for k, v in metric.items() if v != ""}}
    enriched["last_failed_ms"] = _value_as_float(result[0].get("value"))
    return enriched


def collect_persistent_failures(
    *,
    grafana_url: str = DEFAULT_GRAFANA_URL,
    datasource_uid: str = DEFAULT_PROM_UID,
    pr: str = "main",
    limit: int = 10,
    select: int = 3,
    lookback: str = "90d",
    days: int = 7,
    end: float | None = None,
    exclude_job_regex: str | None = DEFAULT_CPU_JOB_RE,
) -> list[dict]:
    end = time.time() if end is None else end
    recent = parse_recent_failures(
        prom_query(
            recent_failures_query(
                pr=pr,
                limit=limit,
                lookback=lookback,
                exclude_job_regex=exclude_job_regex,
            ),
            grafana_url=grafana_url,
            datasource_uid=datasource_uid,
        )
    )
    persistent: list[dict] = []
    for candidate in recent:
        seen_days = 0
        for day in range(days):
            series = prom_query(
                persistence_query(candidate, pr=pr),
                eval_time=end - day * 86400,
                grafana_url=grafana_url,
                datasource_uid=datasource_uid,
            )
            if has_failure_evidence(series):
                seen_days += 1
        if seen_days >= days:
            enriched = enrich_latest_failure(
                candidate,
                pr=pr,
                lookback=lookback,
                grafana_url=grafana_url,
                datasource_uid=datasource_uid,
            )
            enriched["daily_windows_seen"] = seen_days
            enriched["required_daily_windows"] = days
            persistent.append(enriched)
    return persistent[:select]


def _fallback_failure(candidate: dict) -> dict:
    return {
        "nodeid": candidate.get("test_nodeid") or "",
        "exception_type": candidate.get("exception_type") or "",
        "exception_message": "",
        "exception_stacktrace": "",
    }


def fetch_failure_detail(candidate: dict, grafana_url: str, trace_chars: int) -> dict:
    trace_id = candidate.get("trace_id")
    nodeid = candidate.get("test_nodeid")
    if not trace_id:
        return _fallback_failure(candidate)
    try:
        failure = dft.fetch_trace_failure(
            trace_id, grafana_url, nodeid=nodeid, timeout=90
        )
    except dft.TraceFetchError as e:
        print(f"      warning: could not fetch trace {trace_id}: {e}", flush=True)
        return _fallback_failure(candidate)
    if failure is None:
        return _fallback_failure(candidate)
    failure["exception_stacktrace"] = dft._stacktrace_tail(
        failure.get("exception_stacktrace") or "", trace_chars
    )
    return failure


def render_context(
    candidate: dict,
    failure: dict,
    fingerprint: str,
    grafana_url: str,
    *,
    issue_number: int | None = None,
) -> str:
    nodeid = candidate.get("test_nodeid") or failure.get("nodeid") or "?"
    trace_id = candidate.get("trace_id")
    last_failed = ""
    if candidate.get("last_failed_ms"):
        dt = datetime.datetime.fromtimestamp(
            float(candidate["last_failed_ms"]) / 1000,
            datetime.timezone.utc,
        )
        last_failed = dt.isoformat()
    required_days = candidate.get("required_daily_windows") or 7
    lines = [
        fingerprint_marker(fingerprint),
        f"Serge task fingerprint: `{fingerprint}`.",
        "If you open or update a PR for this task, keep the HTML comment above "
        "in the PR body.",
    ]
    if issue_number is not None:
        lines.append(
            f"Also include the line `Relates to #{issue_number}` in the PR body "
            "so the PR links back to the CPU-main triage tracking issue."
        )
    lines += [
        "",
        "A transformers CPU test is failing on `main` and has failed across the "
        f"last {required_days} daily windows. Fix it with a minimal patch, or "
        "return an empty patch if it cannot be fixed safely.",
        "",
        "## Serge candidate failure group 1",
        "",
        f"- `{nodeid}` [cpu] (job `{candidate.get('test_job') or '?'}`, "
        f"seen {candidate.get('daily_windows_seen')}/{required_days} daily windows)",
        f"  - Job: `{candidate.get('test_job') or '?'}`",
        f"  - Test: `{nodeid}`",
        "",
        "The bullet above is the authoritative targeted test set for reproduce "
        "and verify. Reproduce that node id on the baseline first; only publish "
        "a PR after the patched tree passes that same node id.",
        "",
        f"- Test: `{nodeid}`",
        f"- Job: `{candidate.get('test_job') or '?'}`",
        f"- Daily windows with failure evidence: `{candidate.get('daily_windows_seen')}`",
    ]
    if last_failed:
        lines.append(f"- Last failed: `{last_failed}`")
    if trace_id:
        test_url = (
            f"{grafana_url.rstrip('/')}/d/pytest-test/test?"
            f"var-trace_id={urllib.parse.quote(str(trace_id))}&"
            f"var-test_nodeid={urllib.parse.quote(str(nodeid))}"
        )
        lines.append(f"- Trace: `{trace_id}` ({test_url})")
    exc_type = failure.get("exception_type") or candidate.get("exception_type") or "?"
    lines += ["", f"Exception type: `{exc_type}`"]
    if failure.get("exception_message"):
        lines += ["", "Exception message:", "```", failure["exception_message"], "```"]
    if failure.get("exception_stacktrace"):
        lines += [
            "",
            "Stacktrace (tail):",
            "```",
            failure["exception_stacktrace"],
            "```",
        ]
    return "\n".join(lines)


def build_cpu_main_payload(
    repo: str,
    base_ref: str,
    candidate: dict,
    failure: dict,
    fingerprint: str,
    context: str,
    *,
    existing_pr: int | None = None,
    tracking_issue: int | None = None,
    slack_channel: str | None = None,
    notify_task_finished: bool = False,
) -> dict:
    nodeid = candidate.get("test_nodeid") or failure.get("nodeid") or "?"
    exc_type = failure.get("exception_type") or candidate.get("exception_type") or "?"
    title = f"[serge] Fix main CPU test {nodeid} ({exc_type})"[:200]
    return build_task_payload(
        repo,
        base_ref,
        _INSTRUCTION,
        context,
        title,
        branch_prefix=task_branch_prefix(fingerprint),
        existing_pr=existing_pr,
        tracking_issue=tracking_issue,
        slack_channel=slack_channel,
        notify_task_finished=notify_task_finished,
    )


def existing_serge_pr(repo: str, fingerprint: str, token: str | None) -> int | None:
    return match_pr(
        list_open_pulls(repo, token),
        fingerprint_marker(fingerprint),
        task_branch_prefix(fingerprint),
    )


def _print_candidate(i: int, candidate: dict, fingerprint: str) -> None:
    print(f"[{i}] {candidate.get('test_job')} :: {candidate.get('test_nodeid')}")
    print(f"    days={candidate.get('daily_windows_seen')} fp={fingerprint}")
    if candidate.get("exception_type"):
        print(f"    error={candidate['exception_type']}")
    if candidate.get("trace_id"):
        print(f"    trace={candidate['trace_id']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--grafana-url", default=DEFAULT_GRAFANA_URL)
    p.add_argument("--prom-datasource-uid", default=DEFAULT_PROM_UID)
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--base-ref", default="main")
    p.add_argument("--pr", default="main", help='Prometheus pr label, default "main"')
    p.add_argument("--limit", type=int, default=10, help="recent failures to inspect")
    p.add_argument("--select", type=int, default=3, help="persistent tests to select")
    p.add_argument("--days", type=int, default=7, help="daily windows required")
    p.add_argument(
        "--run-key",
        default=None,
        help="tracking issue key, default is today's UTC date",
    )
    p.add_argument("--lookback", default="90d", help="dashboard lookback window")
    p.add_argument(
        "--exclude-job-regex",
        default=DEFAULT_CPU_JOB_RE,
        help="Prometheus regex of jobs to exclude; default excludes GPU jobs",
    )
    p.add_argument(
        "--trace-chars",
        type=int,
        default=int(os.environ.get("CPU_MAIN_TRACE_CHARS", "6000")),
    )
    p.add_argument("--serge-url", default=os.environ.get("SERGE_URL"))
    p.add_argument(
        "--serge-timeout",
        type=int,
        default=int(os.environ.get("SERGE_TIMEOUT", "240")),
    )
    p.add_argument(
        "--slack-channel",
        default=(os.environ.get("SERGE_SLACK_CHANNEL") or "").strip() or None,
    )
    p.add_argument("--notify-task-finished", action="store_true")
    p.add_argument(
        "--reconcile-timeout",
        type=int,
        default=int(os.environ.get("CPU_MAIN_RECONCILE_TIMEOUT", "300")),
        help="seconds to poll for Serge status / PR links after dispatch",
    )
    p.add_argument(
        "--poll-seconds",
        type=int,
        default=int(os.environ.get("CPU_MAIN_POLL_SECONDS", "20")),
        help="poll interval while reconciling",
    )
    p.add_argument(
        "--dispatch",
        action="store_true",
        help="actually POST tasks to Serge (default: dry-run)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="dispatch even if an open Serge PR already matches the fingerprint",
    )
    args = p.parse_args(argv)

    print("[1/5] Collecting recent main-branch CPU failures from Prometheus…")
    try:
        candidates = collect_persistent_failures(
            grafana_url=args.grafana_url,
            datasource_uid=args.prom_datasource_uid,
            pr=args.pr,
            limit=args.limit,
            select=args.select,
            lookback=args.lookback,
            days=args.days,
            exclude_job_regex=args.exclude_job_regex or None,
        )
    except PrometheusQueryError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(
        f"[2/5] Selected {len(candidates)} tests that failed across "
        f"{args.days} daily windows."
    )
    if not candidates:
        return 0

    gh_token = os.environ.get("GITHUB_TOKEN")
    serge_token = os.environ.get("SERGE_OIDC_TOKEN")
    run_key = args.run_key or run_key_for_today()
    rows: list[dict] = []
    pr_numbers: dict[str, int | None] = {}
    print("[3/5] Enriching selected failures…")
    for i, candidate in enumerate(candidates, 1):
        failure = fetch_failure_detail(candidate, args.grafana_url, args.trace_chars)
        fingerprint = failure_fingerprint(candidate, failure)
        existing_pr = (
            existing_serge_pr(args.repo, fingerprint, gh_token)
            if args.dispatch or gh_token
            else None
        )
        pr_numbers[fingerprint] = existing_pr
        _print_candidate(i, candidate, fingerprint)
        if existing_pr and not args.force:
            print(f"    locked=open Serge PR #{existing_pr}; skipping dispatch")
        rows.append(
            {
                "candidate": candidate,
                "failure": failure,
                "fingerprint": fingerprint,
                "existing_pr": existing_pr,
            }
        )

    issue_number: int | None = None
    if args.dispatch:
        initial_body = render_tracking_issue_body(
            rows, run_key, pr_numbers=pr_numbers
        )
        issue_number = ensure_tracking_issue(
            args.repo, run_key, initial_body, gh_token
        )
        if issue_number is not None:
            print(f"      tracking issue #{issue_number}")

    prepared: list[tuple[dict, dict, str, dict]] = []
    for row in rows:
        candidate = row["candidate"]
        failure = row["failure"]
        fingerprint = row["fingerprint"]
        context = render_context(candidate, failure, fingerprint, args.grafana_url)
        if issue_number is not None:
            context = render_context(
                candidate,
                failure,
                fingerprint,
                args.grafana_url,
                issue_number=issue_number,
            )
        payload = build_cpu_main_payload(
            args.repo,
            args.base_ref,
            candidate,
            failure,
            fingerprint,
            context,
            existing_pr=row["existing_pr"],
            tracking_issue=issue_number,
            slack_channel=args.slack_channel,
            notify_task_finished=args.notify_task_finished,
        )
        prepared.append((candidate, failure, fingerprint, payload))

    print("[4/5] Serge payloads:")
    print(json.dumps([p for *_, p in prepared], indent=2))

    if not args.dispatch:
        print("[5/5] Dry-run complete. Re-run with --dispatch to fire tasks.")
        return 0
    if not args.serge_url:
        print("error: --serge-url (or SERGE_URL) is required to --dispatch", file=sys.stderr)
        return 2
    if not serge_token:
        print("error: SERGE_OIDC_TOKEN is required to --dispatch", file=sys.stderr)
        return 2

    print("[5/5] Dispatching selected tasks to Serge…")
    failures = 0
    statuses: dict[str, str] = {}
    task_urls: dict[str, str] = {}
    job_ids: dict[str, str] = {}
    for i, (_candidate, _failure, fingerprint, payload) in enumerate(prepared, 1):
        existing_pr = payload.get("output", {}).get("pr_number")
        if existing_pr and not args.force:
            print(f"    [{i}] skipped: open PR #{existing_pr} already tracks {fingerprint}")
            statuses[fingerprint] = "existing_pr"
            continue
        try:
            resp = dispatch_to_serge(
                args.serge_url, serge_token, payload, timeout=args.serge_timeout
            )
        except SergeDispatchError as e:
            failures += 1
            print(f"    [{i}] failed: {e}", file=sys.stderr)
            continue
        task_url = resp.get("url")
        full_url = f"{args.serge_url.rstrip('/')}{task_url}" if task_url else ""
        if full_url:
            task_urls[fingerprint] = full_url
        if resp.get("id"):
            job_ids[fingerprint] = str(resp["id"])
        statuses[fingerprint] = "running"
        print(f"    [{i}] accepted id={resp.get('id') or '?'} {full_url}")
        if issue_number is not None:
            update_issue_body(
                args.repo,
                issue_number,
                render_tracking_issue_body(
                    rows,
                    run_key,
                    statuses=statuses,
                    task_urls=task_urls,
                    pr_numbers=pr_numbers,
                ),
                gh_token,
            )
    if issue_number is not None and args.reconcile_timeout > 0 and job_ids:
        deadline = time.monotonic() + args.reconcile_timeout
        marker_by_fp = {fp: fingerprint_marker(fp) for fp in job_ids}
        branch_by_fp = {fp: task_branch_prefix(fp) for fp in job_ids}
        while time.monotonic() < deadline:
            changed = False
            pulls = list_open_pulls(args.repo, gh_token)
            for fp, job_id in list(job_ids.items()):
                if not pr_numbers.get(fp):
                    pr = match_pr(pulls, marker_by_fp[fp], branch_by_fp[fp])
                    if pr:
                        pr_numbers[fp] = pr
                        statuses[fp] = "published"
                        changed = True
                if not pr_numbers.get(fp) and statuses.get(fp) not in {"no_fix", "error"}:
                    serge_token = mint_serge_oidc_token() or serge_token
                    detail = poll_serge_task(
                        args.serge_url, serge_token, args.repo, job_id
                    )
                    status = str(detail.get("status") or statuses.get(fp) or "")
                    if status and status != statuses.get(fp):
                        statuses[fp] = status
                        changed = True
            if changed:
                update_issue_body(
                    args.repo,
                    issue_number,
                    render_tracking_issue_body(
                        rows,
                        run_key,
                        statuses=statuses,
                        task_urls=task_urls,
                        pr_numbers=pr_numbers,
                    ),
                    gh_token,
                )
            if all(
                pr_numbers.get(fp) or statuses.get(fp) in {"no_fix", "error"}
                for fp in job_ids
            ):
                break
            time.sleep(args.poll_seconds)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
