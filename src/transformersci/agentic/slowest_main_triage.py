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

"""Slowest main-branch test selector → Serge investigation dispatcher."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
import urllib.error

from .cpu_main_failure_triage import (
    DEFAULT_GRAFANA_URL,
    DEFAULT_PROM_UID,
    DEFAULT_REPO,
    PrometheusQueryError,
    _quote_label_value,
    _value_as_float,
    prom_query,
)
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
    poll_serge_task,
)

_STATE_SOURCE = "slowest-main-triage"
DEFAULT_RECORDING_METRIC = "pytest_test:duration_avg_main:top20_7d"
DEFAULT_PANEL_LOOKBACK = "8d"
DEFAULT_NODEID_PREFIX = "tests/"

_INSTRUCTION = (
    "Investigate the single slow CPU test on the transformers main branch "
    "described in the report below. The report was selected from the Prometheus "
    "recording rule that backs the `Slowest Tests By Average Duration (main "
    "branch)` dashboard panel.\n\n"
    "A slow test is not automatically a bug. First decide whether the observed "
    "runtime is healthy for the coverage it provides. Look for unhealthy "
    "slowness such as loading far more weights than the assertion requires, "
    "quadratic or worse work on a critical execution path, repeated expensive "
    "setup inside parametrized cases, avoidable compilation/generation work, or "
    "test code that exercises a much larger model/configuration than needed.\n\n"
    "Do not treat expected framework overhead as a source bug. For example, CPU "
    "`torch.compile` can be inherently slow even for a tiny model. If your "
    "inspection finds only expected compile overhead, and no concrete excessive "
    "test size, repeated setup, avoidable generation work, or model-code "
    "complexity issue, stop and return a no-patch result. Do not propose a CPU "
    "skip merely because the test is slow; propose skipping only when there is "
    "clear evidence that CPU coverage is invalid or redundant with a stronger "
    "accelerator-only signal.\n\n"
    "Keep the investigation bounded: inspect the inherited test implementation, "
    "the target model tester/config, and at most a few comparable model-specific "
    "overrides. Once those checks explain the runtime as healthy or identify a "
    "specific fix, produce the final JSON. Do not loop over the same tradeoff. "
    "For inherited generic generation/compile tests, this normally means: read "
    "the inherited test, read the target model tester/config, inspect no more "
    "than three comparable model-specific overrides, then decide. If you reach "
    "the conclusion that the runtime is expected framework overhead, do not run "
    "more searches; immediately return final JSON with an empty patch and a body "
    "explaining the no-patch decision.\n\n"
    "Use reproduce-first / verify-before-PR discipline. First run the targeted "
    "node id on the baseline tree and record its duration. If the test does not "
    "pass on the baseline, bail without LLM work. If the slowness appears "
    "healthy, document that and produce no patch. If you find a real issue, "
    "make the smallest safe change in the test or model code, then verify the "
    "same targeted node id passes and compare its duration to the baseline. Do "
    "not open a PR unless the targeted test still passes and the patch "
    "plausibly improves unhealthy runtime without reducing meaningful coverage.\n\n"
    "If the local tool environment cannot execute shell commands, do not get "
    "stuck trying to reproduce. Treat the Prometheus duration in the report as "
    "the baseline timing evidence, continue with code inspection, and only "
    "return a patch if the inspection finds a concrete issue. Otherwise return "
    "no patch.\n\n"
    "Investigate the test and production code before editing. If a model has a "
    "`modular_<name>.py`, edit that source rather than generated modeling files. "
    "If no safe speedup is available, produce no patch."
)


def slowest_query(
    *,
    metric: str = DEFAULT_RECORDING_METRIC,
    limit: int = 10,
    lookback: str = DEFAULT_PANEL_LOOKBACK,
) -> str:
    return f"sort_desc(topk({limit}, last_over_time({metric}[{lookback}])))"


def persistence_query(
    candidate: dict, *, metric: str = DEFAULT_RECORDING_METRIC
) -> str:
    return (
        f"last_over_time({metric}"
        "{"
        f"test_job={_quote_label_value(candidate['test_job'])},"
        f"test_nodeid={_quote_label_value(candidate['test_nodeid'])}"
        "}[24h])"
    )


def parse_slowest(series: list[dict]) -> list[dict]:
    out: list[dict] = []
    for item in series:
        metric = item.get("metric") or {}
        test_job = metric.get("test_job")
        test_nodeid = metric.get("test_nodeid")
        if not test_job or not test_nodeid:
            continue
        out.append(
            {
                "test_job": str(test_job),
                "test_nodeid": str(test_nodeid),
                "test_module": str(metric.get("test_module") or ""),
                "test_class": str(metric.get("test_class") or ""),
                "test_function": str(metric.get("test_function") or ""),
                "avg_duration_seconds": _value_as_float(item.get("value")),
            }
        )
    out.sort(key=lambda c: c["avg_duration_seconds"], reverse=True)
    return out


def collect_persistent_slowest(
    *,
    grafana_url: str = DEFAULT_GRAFANA_URL,
    datasource_uid: str = DEFAULT_PROM_UID,
    metric: str = DEFAULT_RECORDING_METRIC,
    lookback: str = DEFAULT_PANEL_LOOKBACK,
    limit: int = 10,
    select: int = 3,
    days: int = 7,
    end: float | None = None,
    nodeid_prefix: str = DEFAULT_NODEID_PREFIX,
    test_nodeid: str | None = None,
) -> list[dict]:
    end = time.time() if end is None else end
    recent = parse_slowest(
        prom_query(
            slowest_query(metric=metric, limit=limit, lookback=lookback),
            grafana_url=grafana_url,
            datasource_uid=datasource_uid,
        )
    )
    if nodeid_prefix:
        recent = [
            candidate
            for candidate in recent
            if str(candidate.get("test_nodeid") or "").startswith(nodeid_prefix)
        ]
    if test_nodeid:
        recent = [
            candidate
            for candidate in recent
            if candidate.get("test_nodeid") == test_nodeid
        ]
    persistent: list[dict] = []
    for candidate in recent:
        seen_days = 0
        max_seen = candidate["avg_duration_seconds"]
        for day in range(days):
            rows = prom_query(
                persistence_query(candidate, metric=metric),
                eval_time=end - day * 86400,
                grafana_url=grafana_url,
                datasource_uid=datasource_uid,
            )
            if rows:
                seen_days += 1
                max_seen = max(max_seen, *(_value_as_float(r.get("value")) for r in rows))
        if seen_days >= days:
            candidate = dict(candidate)
            candidate["daily_windows_seen"] = seen_days
            candidate["required_daily_windows"] = days
            candidate["max_duration_seconds"] = max_seen
            persistent.append(candidate)
    return persistent[:select]


def failure_fingerprint(candidate: dict) -> str:
    basis = {
        "source": _STATE_SOURCE,
        "test_job": candidate.get("test_job") or "",
        "test_nodeid": candidate.get("test_nodeid") or "",
    }
    raw = json.dumps(basis, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def fingerprint_marker(fingerprint: str) -> str:
    return f"<!-- serge-slowest-main-fix:sha256:{fingerprint} -->"


def task_branch_prefix(fingerprint: str) -> str:
    return f"serge/optimize/slow-main-{fingerprint[:12]}"


def run_key_for_today(now: datetime.datetime | None = None) -> str:
    now = now or datetime.datetime.now(datetime.timezone.utc)
    return now.date().isoformat()


def tracking_issue_marker(run_key: str) -> str:
    return f"<!-- serge-triage-run:{_STATE_SOURCE}:{run_key} -->"


def tracking_issue_title(run_key: str) -> str:
    return f"[serge] Slowest main tests triage {run_key}"


def _md_cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


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
        f"Automated **slowest main test triage** for `{run_key}`.",
        "",
        "Serge should reproduce each targeted test on the baseline first, then "
        "investigate whether the slowness is healthy. It should open a PR only "
        "when it finds a real test or model-code issue and verifies the targeted "
        "test still passes with a plausible speedup.",
        "",
        "| Job | Test | Avg duration | Seen | Serge task | PR |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        fp = row["fingerprint"]
        candidate = row["candidate"]
        task_url = task_urls.get(fp)
        pr = pr_numbers.get(fp)
        status = statuses.get(fp)
        if pr:
            pr_cell = f"#{pr}"
        elif status:
            pr_cell = status
        else:
            pr_cell = f"`{task_branch_prefix(fp)}` (pending)"
        cells = [
            f"`{candidate.get('test_job') or '?'}`",
            f"`{candidate.get('test_nodeid') or '?'}`",
            f"{float(candidate.get('avg_duration_seconds') or 0):.2f}s",
            f"{candidate.get('daily_windows_seen')}/{candidate.get('required_daily_windows') or 7}",
            f"[task]({task_url})" if task_url else "-",
            pr_cell,
        ]
        lines.append("| " + " | ".join(_md_cell(c) for c in cells) + " |")
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
        print(f"      warning: could not create/update tracking issue: {e}")
        return None


def render_context(
    candidate: dict,
    fingerprint: str,
    grafana_url: str,
    *,
    issue_number: int | None = None,
) -> str:
    nodeid = candidate.get("test_nodeid") or "?"
    required_days = candidate.get("required_daily_windows") or 7
    lines = [
        fingerprint_marker(fingerprint),
        f"Serge task fingerprint: `{fingerprint}`.",
        "If you open or update a PR for this task, keep the HTML comment above "
        "in the PR body.",
    ]
    if issue_number is not None:
        lines.append(f"Also include `Relates to #{issue_number}` in the PR body.")
    lines += [
        "",
        "A transformers CPU test is among the slowest main-branch tests by "
        "average duration. Investigate whether that slowness is healthy or "
        "caused by avoidable test work or unhealthy model-code complexity.",
        "",
        "## Serge candidate failure group 1",
        "",
        f"- `{nodeid}` [cpu] (job `{candidate.get('test_job') or '?'}`, "
        f"avg {float(candidate.get('avg_duration_seconds') or 0):.2f}s, "
        f"seen {candidate.get('daily_windows_seen')}/{required_days} daily windows)",
        f"  - Job: `{candidate.get('test_job') or '?'}`",
        f"  - Test: `{nodeid}`",
        "",
        "The bullet above is the authoritative targeted test set for reproduce "
        "and verify. Reproduce that node id on the baseline first and record "
        "duration. Then inspect the test and relevant model code for unhealthy "
        "slowness, for example excessive weight loading or quadratic work on a "
        "critical path. If the slowness is healthy, produce no patch. Only "
        "publish a PR after the patched tree passes that same node id and the "
        "runtime is plausibly improved.",
        "",
        "Decision rule: if the only plausible explanation is expected framework "
        "cost, such as CPU `torch.compile` compilation overhead on an already "
        "tiny test model, return no patch. A skip is a coverage deletion and "
        "requires stronger evidence than runtime alone.",
        "",
        "Stop rule: after reading the inherited test, this model tester/config, "
        "and at most three comparable overrides, make the decision. If that "
        "decision is healthy framework overhead, do not perform additional "
        "searches; emit final JSON with an empty patch.",
        "",
        f"- Test: `{nodeid}`",
        f"- Job: `{candidate.get('test_job') or '?'}`",
        f"- Average duration: `{float(candidate.get('avg_duration_seconds') or 0):.3f}s`",
        f"- Max sampled duration: `{float(candidate.get('max_duration_seconds') or 0):.3f}s`",
        f"- Daily windows with slowest-panel evidence: `{candidate.get('daily_windows_seen')}`",
        f"- Dashboard: `{grafana_url.rstrip('/')}/d/pytest-test-health/test-health`",
    ]
    return "\n".join(lines)


def build_slowest_payload(
    repo: str,
    base_ref: str,
    candidate: dict,
    fingerprint: str,
    context: str,
    *,
    existing_pr: int | None = None,
    tracking_issue: int | None = None,
) -> dict:
    title = (
        f"[serge] Investigate slow main test {candidate.get('test_nodeid') or '?'} "
        f"({float(candidate.get('avg_duration_seconds') or 0):.1f}s)"
    )[:200]
    return build_task_payload(
        repo,
        base_ref,
        _INSTRUCTION,
        context,
        title,
        branch_prefix=task_branch_prefix(fingerprint),
        existing_pr=existing_pr,
        tracking_issue=tracking_issue,
    )


def existing_serge_pr(repo: str, fingerprint: str, token: str | None) -> int | None:
    return match_pr(
        list_open_pulls(repo, token),
        fingerprint_marker(fingerprint),
        task_branch_prefix(fingerprint),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grafana-url", default=DEFAULT_GRAFANA_URL)
    p.add_argument("--prom-datasource-uid", default=DEFAULT_PROM_UID)
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--base-ref", default="main")
    p.add_argument("--metric", default=DEFAULT_RECORDING_METRIC)
    p.add_argument("--lookback", default=DEFAULT_PANEL_LOOKBACK)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--select", type=int, default=3)
    p.add_argument("--days", type=int, default=7)
    p.add_argument(
        "--nodeid-prefix",
        default=DEFAULT_NODEID_PREFIX,
        help='only consider tests whose nodeid starts with this prefix; default "tests/"',
    )
    p.add_argument(
        "--test-nodeid",
        default=None,
        help="only consider this exact pytest node id",
    )
    p.add_argument("--run-key", default=None)
    p.add_argument("--serge-url", default=os.environ.get("SERGE_URL"))
    p.add_argument("--serge-timeout", type=int, default=240)
    p.add_argument("--dispatch", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    print("[1/5] Collecting slowest main-branch tests from Prometheus...")
    try:
        candidates = collect_persistent_slowest(
            grafana_url=args.grafana_url,
            datasource_uid=args.prom_datasource_uid,
            metric=args.metric,
            lookback=args.lookback,
            limit=args.limit,
            select=args.select,
            days=args.days,
            nodeid_prefix=args.nodeid_prefix,
            test_nodeid=args.test_nodeid,
        )
    except PrometheusQueryError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(
        f"[2/5] Selected {len(candidates)} tests present across "
        f"{args.days} daily windows."
    )
    if not candidates:
        return 0

    gh_token = os.environ.get("GITHUB_TOKEN")
    run_key = args.run_key or run_key_for_today()
    rows: list[dict] = []
    pr_numbers: dict[str, int | None] = {}
    print("[3/5] Preparing selected slow tests...")
    for i, candidate in enumerate(candidates, 1):
        fp = failure_fingerprint(candidate)
        existing_pr = (
            existing_serge_pr(args.repo, fp, gh_token)
            if args.dispatch or gh_token
            else None
        )
        pr_numbers[fp] = existing_pr
        print(
            f"[{i}] {candidate.get('test_job')} :: {candidate.get('test_nodeid')} "
            f"avg={float(candidate.get('avg_duration_seconds') or 0):.2f}s "
            f"days={candidate.get('daily_windows_seen')} fp={fp}"
        )
        rows.append({"candidate": candidate, "fingerprint": fp, "existing_pr": existing_pr})

    issue_number: int | None = None
    if args.dispatch:
        issue_number = ensure_tracking_issue(
            args.repo,
            run_key,
            render_tracking_issue_body(rows, run_key, pr_numbers=pr_numbers),
            gh_token,
        )

    prepared: list[tuple[str, dict]] = []
    for row in rows:
        candidate = row["candidate"]
        fp = row["fingerprint"]
        context = render_context(
            candidate, fp, args.grafana_url, issue_number=issue_number
        )
        payload = build_slowest_payload(
            args.repo,
            args.base_ref,
            candidate,
            fp,
            context,
            existing_pr=row["existing_pr"],
            tracking_issue=issue_number,
        )
        prepared.append((fp, payload))

    print("[4/5] Serge payloads:")
    print(json.dumps([payload for _, payload in prepared], indent=2))
    if not args.dispatch:
        print("[5/5] Dry-run complete. Re-run with --dispatch to fire tasks.")
        return 0
    if not args.serge_url:
        print("error: --serge-url (or SERGE_URL) is required to --dispatch", file=sys.stderr)
        return 2

    serge_token = os.environ.get("SERGE_OIDC_TOKEN")
    if not serge_token:
        print("error: SERGE_OIDC_TOKEN is required to --dispatch", file=sys.stderr)
        return 2
    statuses: dict[str, str] = {}
    task_urls: dict[str, str] = {}
    for fp, payload in prepared:
        try:
            task = dispatch_to_serge(
                args.serge_url,
                serge_token,
                payload,
                timeout=args.serge_timeout,
            )
        except SergeDispatchError as e:
            print(f"      error dispatching {fp}: {e}")
            statuses[fp] = "error"
            continue
        task_url = str(task.get("url") or task.get("task_url") or "")
        if task_url:
            task_urls[fp] = task_url
        task_id = str(task.get("id") or task.get("task_id") or "")
        if task_id:
            status = poll_serge_task(
                args.serge_url,
                serge_token,
                args.repo,
                task_id,
                timeout=args.serge_timeout,
            )
            statuses[fp] = str(status.get("status") or "dispatched")
    if issue_number is not None:
        ensure_tracking_issue(
            args.repo,
            run_key,
            render_tracking_issue_body(
                rows, run_key, statuses=statuses, task_urls=task_urls, pr_numbers=pr_numbers
            ),
            gh_token,
        )
    print("[5/5] Dispatch complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
