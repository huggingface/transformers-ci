#!/usr/bin/env python3
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
"""Compare the CI-observability dashboard's test counts against GitHub ground truth.

Given a PR number, this pulls the per-job test totals the trace-exporter published
to Prometheus (what the dashboards show) and lines them up against the *actual*
pytest outcomes parsed from the GitHub Actions shard logs for the same run. Use it
to confirm the exporter is capturing every shard and every test — e.g. after the
sharded-job enumeration fix and the Tempo ``max_bytes_per_trace`` change (see
``docs/session-notes-exporter-2026-06-23.md``).

What it checks per test job (e.g. ``tests_torch``):
  - shards: GitHub shard count   vs  distinct shard traces the exporter captured
  - tests:  GitHub executed/collected counts  vs  ``pytest_run_job_total_tests``

"Executed" = passed+failed+errors+xfailed+xpassed (everything that actually ran);
"collected" additionally includes skipped. The exporter emits one span per test
that ran, so the dashboard total should track GitHub's executed count (the script
prints both so you can see which it aligns with).

Usage:
  python dashboard/compare_pr_to_github.py 46754
  python dashboard/compare_pr_to_github.py 46754 --run-id 28036323378
  python dashboard/compare_pr_to_github.py 46754 --repo huggingface/transformers \
      --prom-base https://transformers-ci.lor-e.huggingface.cool

Requires: the `gh` CLI (authenticated) for the GitHub side; the Prometheus side
goes through the public Grafana datasource proxy (no auth needed).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.parse
import urllib.request

DEFAULT_PROM_BASE = "https://transformers-ci.lor-e.huggingface.cool"
DEFAULT_PROM_DS_UID = "prometheus"
DEFAULT_REPO = "huggingface/transformers"
DEFAULT_LOOKBACK = "14d"

# The pytest summary footer, e.g.
#   "= 7772 passed, 9200 skipped, 21 xfailed, 3 xpassed, 212 warnings in 227s ="
SUMMARY_LINE_RE = re.compile(
    r"={2,}\s+(.*?\b(?:passed|failed|error)\b.*?)\s+in\s+[\d.]+s"
)
OUTCOME_RE = re.compile(
    r"(\d+)\s+(passed|failed|errors?|xfailed|xpassed|skipped|deselected|warnings?)"
)
EXECUTED_OUTCOMES = {"passed", "failed", "error", "errors", "xfailed", "xpassed"}

# The per-process span tally the runner prints to stderr when
# TRANSFORMERSCI_OTEL_DEBUG=1 (see src/transformersci/otel/debug_exporter.py),
# e.g. "OTEL DEBUG SUMMARY produced=4814 submitted=4814 exported=4790 failed=24
# not_exported=0 export_calls=12 failed_calls=1 (...)". One line per process, so a
# shard running pytest-xdist prints one per worker — we sum every line found in a
# shard's log. ``failed`` (spans whose export() returned FAILURE/raised — lost on
# the wire) is the field that distinguishes transport loss from queue overflow;
# older logs lack ``submitted``/``failed`` and parse them as 0.
DEBUG_SUMMARY_RE = re.compile(r"OTEL DEBUG SUMMARY ([a-z_]+=-?\d+(?: [a-z_]+=-?\d+)*)")
DEBUG_FIELD_RE = re.compile(r"([a-z_]+)=(-?\d+)")


def prom_query(prom_base: str, ds_uid: str, query: str) -> list[dict]:
    """Run an instant PromQL query through the Grafana datasource proxy."""
    url = (
        f"{prom_base.rstrip('/')}/api/datasources/proxy/uid/{ds_uid}"
        f"/api/v1/query?{urllib.parse.urlencode({'query': query})}"
    )
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.load(resp)
    if payload.get("status") != "success":
        raise RuntimeError(f"Prometheus query failed: {payload}")
    return payload["data"]["result"]


def _gh_raw(path: str, *, paginate: bool = False) -> str:
    """Call the GitHub API via the `gh` CLI (uses the user's existing auth)."""
    cmd = ["gh", "api", path]
    if paginate:
        cmd.append("--paginate")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"gh api {path} failed: {proc.stderr.strip()}")
    return proc.stdout


def gh_api_json(path: str, *, paginate: bool = False) -> dict:
    return json.loads(_gh_raw(path, paginate=paginate))


def select_run_id(
    prom_base: str, ds_uid: str, pr: str, lookback: str, explicit: str | None
) -> tuple[str, dict[str, float]]:
    """Return (exporter_run_id, {test_job: total}) for the chosen run of this PR.

    Without --run-id, picks the most recent run for the PR by start time. The
    exporter run_id carries a ``:attempt`` suffix (e.g. ``28036323378:1``).
    """
    totals = prom_query(
        prom_base,
        ds_uid,
        f"max without (instance) (last_over_time("
        f'pytest_run_job_total_tests{{pr="{pr}"}}[{lookback}]))',
    )
    if not totals:
        raise SystemExit(
            f"No pytest_run_job_total_tests for PR {pr} in the last {lookback}. "
            f"Wrong PR, or the run aged out of the query window?"
        )

    by_run: dict[str, dict[str, float]] = {}
    for series in totals:
        run_id = series["metric"].get("run_id", "")
        job = series["metric"].get("test_job", "unknown")
        by_run.setdefault(run_id, {})[job] = float(series["value"][1])

    if explicit:
        # Accept either the bare GitHub id or the exporter's id:attempt form.
        matches = [r for r in by_run if r.split(":")[0] == explicit.split(":")[0]]
        if not matches:
            raise SystemExit(f"No exporter run matching run-id {explicit} for PR {pr}.")
        run_id = sorted(matches)[-1]
        return run_id, by_run[run_id]

    starts = prom_query(
        prom_base,
        ds_uid,
        f"max without (instance) (last_over_time("
        f'pytest_run_start_time_seconds{{pr="{pr}"}}[{lookback}]))',
    )
    start_by_run = {s["metric"].get("run_id", ""): float(s["value"][1]) for s in starts}
    run_id = max(by_run, key=lambda r: start_by_run.get(r, 0.0))
    return run_id, by_run[run_id]


def exporter_shard_counts(
    prom_base: str, ds_uid: str, run_id: str, lookback: str
) -> dict[str, int]:
    """Distinct shard traces the exporter captured per test job for this run."""
    result = prom_query(
        prom_base,
        ds_uid,
        f"count by (test_job) (count by (test_job, trace_id) ("
        f"max without (instance) (last_over_time("
        f'pytest_ci_runner_execution_info{{run_id="{run_id}"}}[{lookback}]))))',
    )
    return {
        s["metric"].get("test_job", "unknown"): int(float(s["value"][1]))
        for s in result
    }


def exporter_max_shard_tests(
    prom_base: str, ds_uid: str, run_id: str, lookback: str
) -> dict[str, int]:
    """Largest single-shard span count per job (to spot a ~2048 SDK-queue plateau)."""
    result = prom_query(
        prom_base,
        ds_uid,
        f"max by (test_job) (count by (test_job, trace_id) ("
        f"last_over_time("
        f'pytest_test_duration_seconds{{run_id="{run_id}"}}[{lookback}])))',
    )
    return {
        s["metric"].get("test_job", "unknown"): int(float(s["value"][1]))
        for s in result
    }


def parse_summary(log_text: str) -> dict[str, int]:
    """Parse the last pytest summary footer in a job log into outcome counts."""
    matches = SUMMARY_LINE_RE.findall(log_text)
    if not matches:
        return {}
    counts: dict[str, int] = {}
    for number, outcome in OUTCOME_RE.findall(matches[-1]):
        key = "error" if outcome.startswith("error") else outcome
        key = "warning" if key.startswith("warning") else key
        counts[key] = counts.get(key, 0) + int(number)
    return counts


def job_test_name(job_name: str, known_jobs: set[str]) -> str | None:
    """Map a GitHub job name to a known exporter test_job, or None.

    Job names look like ``pr-ci / tests_torch / tests_torch [shard 4/8]`` — the
    logical job is a ``/``-delimited segment (shard suffix stripped). We only
    accept segments that match a test_job the exporter actually reported, so
    setup/matrix jobs and unrelated jobs are ignored.
    """
    if "generate shard matrix" in job_name or "Setup" in job_name:
        return None
    for segment in job_name.split("/"):
        base = segment.split("[shard")[0].strip()
        if base in known_jobs:
            return base
    return None


def github_job_outcomes(
    repo: str, gh_run_id: str, known_jobs: set[str]
) -> dict[str, dict]:
    """Per test job: shard count + summed pytest outcomes, from GitHub logs.

    Only fetches logs for jobs matching a known exporter test_job (keeps the
    GitHub API calls bounded). A job counts as a shard once its log yields a
    pytest summary footer; logs without one are tallied as ``missing_logs``.
    """
    jobs = gh_api_json(
        f"repos/{repo}/actions/runs/{gh_run_id}/jobs?per_page=100", paginate=True
    )["jobs"]
    per_job: dict[str, dict] = {}
    for job in jobs:
        test_job = job_test_name(job["name"], known_jobs)
        if test_job is None:
            continue
        entry = per_job.setdefault(
            test_job,
            {
                "shards": 0,
                "outcomes": {},
                "missing_logs": 0,
                "debug": _empty_debug(),
            },
        )
        try:
            log_text = _gh_raw(f"repos/{repo}/actions/jobs/{job['id']}/logs")
        except RuntimeError:
            entry["missing_logs"] += 1
            continue
        # The span tally is independent of the pytest footer, so accumulate it
        # even for a shard whose footer we can't parse — it still tells us how
        # the export pipeline behaved on that shard.
        for key, value in parse_debug_summaries(log_text).items():
            entry["debug"][key] += value
        summary = parse_summary(log_text)
        if not summary:
            entry["missing_logs"] += 1
            continue
        entry["shards"] += 1
        for outcome, n in summary.items():
            entry["outcomes"][outcome] = entry["outcomes"].get(outcome, 0) + n
    return per_job


def _empty_debug() -> dict[str, int]:
    return {
        "produced": 0,
        "submitted": 0,
        "exported": 0,
        "failed": 0,
        "not_exported": 0,
        "summaries": 0,
    }


def parse_debug_summaries(log_text: str) -> dict[str, int]:
    """Sum the OTEL DEBUG SUMMARY tallies in one job log.

    Returns produced/submitted/exported/failed/not_exported summed across every
    summary line (one per process, so per pytest-xdist worker), plus
    ``summaries`` = how many lines were found (0 means the debug flag wasn't
    enabled for this shard). Parsed field-by-field so it tolerates older logs that
    only carry produced/exported/not_exported (the missing fields stay 0).
    """
    totals = _empty_debug()
    for body in DEBUG_SUMMARY_RE.findall(log_text):
        fields = {k: int(v) for k, v in DEBUG_FIELD_RE.findall(body)}
        for key in ("produced", "submitted", "exported", "failed", "not_exported"):
            totals[key] += fields.get(key, 0)
        totals["summaries"] += 1
    return totals


def executed(outcomes: dict[str, int]) -> int:
    return sum(n for o, n in outcomes.items() if o in EXECUTED_OUTCOMES)


def collected(outcomes: dict[str, int]) -> int:
    return executed(outcomes) + outcomes.get("skipped", 0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare dashboard test counts vs GitHub Actions for a PR."
    )
    parser.add_argument("pr", help="PR number, e.g. 46754")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--run-id",
        default=None,
        help="GitHub Actions run id to compare (default: most recent run for the PR)",
    )
    parser.add_argument("--prom-base", default=DEFAULT_PROM_BASE)
    parser.add_argument("--prom-ds-uid", default=DEFAULT_PROM_DS_UID)
    parser.add_argument(
        "--lookback",
        default=DEFAULT_LOOKBACK,
        help="Prometheus last_over_time window for the run's series (default 14d)",
    )
    args = parser.parse_args()

    run_id, exporter_totals = select_run_id(
        args.prom_base, args.prom_ds_uid, args.pr, args.lookback, args.run_id
    )
    gh_run_id = run_id.split(":")[0]
    shard_counts = exporter_shard_counts(
        args.prom_base, args.prom_ds_uid, run_id, args.lookback
    )
    max_shard = exporter_max_shard_tests(
        args.prom_base, args.prom_ds_uid, run_id, args.lookback
    )

    print(f"PR #{args.pr}  repo {args.repo}")
    print(f"exporter run_id {run_id}  (GitHub run {gh_run_id})")
    print(f"GitHub run: https://github.com/{args.repo}/actions/runs/{gh_run_id}")
    print("fetching GitHub job logs (this can take a moment)…\n")

    gh = github_job_outcomes(args.repo, gh_run_id, set(exporter_totals))

    jobs = sorted(set(exporter_totals) | set(gh))
    # ----------------------------------------------------------------------
    # HOW TO READ THIS TABLE (the denominator matters — read this first):
    #
    # The exporter records ONE test row per COLLECTED test — one canonical pytest
    # span per test item (the span whose name == the test nodeid; setup/call/
    # teardown phase sub-spans are NOT double-counted). pytest emits that span for
    # passed, failed, xfailed, xpassed AND skipped items, so the exporter's per-job
    # total tracks GitHub's COLLECTED count (executed + skipped), NOT executed.
    # Empirically confirmed: single-shard jobs match collected exactly (e.g.
    # tests_repo_utils 68==68, tests_custom_tokenizers 255==255, tests_training_ci
    # 75==75). So %coll is the headline; %exec is shown only for context.
    #
    # %coll ~100%  -> exporter agrees with GitHub; telemetry is complete.
    # %coll  < 95% on a run that finished >~10 min ago -> a REAL undercount. The
    #     known cause is large-trace read truncation: a single Tempo read of a big
    #     shard trace returns only a subset of its spans even though Tempo STORES
    #     them all (verified: a tests_torch shard had 13,914 test spans in Tempo
    #     via direct GET /api/traces/<id>, but the exporter extracted ~5,392). It
    #     correlates with trace SIZE (span count / bytes), so heavy jobs
    #     (tests_generate / tests_processors / tests_tokenization / tensor_parallel
    #     / tests_torch) undercount while small jobs match. This is downstream of
    #     ingestion (runner ships 100%, see the span-pipeline section) and is NOT
    #     the settle/refetch behaviour — re-reading does not help when each read is
    #     itself truncated. Root cause still open as of 2026-06-24.
    # %coll  < 100% on a run that finished in the LAST few minutes can ALSO be the
    #     roll-up lag (not loss): the roll-up settles ~run_settle (120s) after the
    #     last shard appears, but each shard's row count is only refreshed every
    #     PYTEST_TRACE_EXPORTER_TRACE_REFETCH_SECONDS (~300s), so it converges
    #     upward. Re-run once the run has been done >10 min to separate lag from
    #     the truncation undercount above.
    #
    # max/shard = largest single shard-trace test count captured; a value far
    # below collected/shards is the truncation fingerprint. %exec >100% just means
    # collected includes skipped that the exporter counts but GH "executed" omits.
    # ----------------------------------------------------------------------
    header = (
        f"{'test_job':24} {'shards GH/exp':>13}  "
        f"{'GH exec':>8} {'GH coll':>8}  "
        f"{'exporter':>8} {'%coll':>6} {'%exec':>6}  {'max/shard':>9}"
    )
    print(header)
    print("-" * len(header))
    worst = []
    for job in jobs:
        gh_entry = gh.get(
            job,
            {
                "shards": 0,
                "outcomes": {},
                "missing_logs": 0,
                "debug": _empty_debug(),
            },
        )
        gh_out = gh_entry["outcomes"]
        gh_coll = collected(gh_out)
        gh_exec = executed(gh_out)
        gh_shards = gh_entry["shards"]
        exp_total = int(exporter_totals.get(job, 0))
        exp_shards = shard_counts.get(job, 0)
        # Headline % is against COLLECTED (the count the exporter actually tracks —
        # one span per collected test, skipped included); %exec is informational.
        pct_coll = (100 * exp_total / gh_coll) if gh_coll else float("nan")
        pct_exec = (100 * exp_total / gh_exec) if gh_exec else float("nan")
        flag = ""
        if gh_entry["missing_logs"]:
            flag += f" !{gh_entry['missing_logs']} logs unparsed"
        if gh_shards and exp_shards != gh_shards:
            flag += " !shard mismatch"
        if gh_coll and pct_coll < 95:
            flag += " !undercount"
            worst.append((job, pct_coll))
        shards_col = f"{gh_shards}/{exp_shards}"
        pct_c = "  n/a" if not gh_coll else f"{pct_coll:5.0f}%"
        pct_e = "  n/a" if not gh_exec else f"{pct_exec:5.0f}%"
        print(
            f"{job:24} {shards_col:>13}  "
            f"{gh_exec:>8} {gh_coll:>8}  "
            f"{exp_total:>8} {pct_c:>6} {pct_e:>6}  {max_shard.get(job, 0):>9}{flag}"
        )

    print()
    print(
        "%coll = exporter / GitHub COLLECTED (passed+failed+errors+xfailed+xpassed"
        "+skipped) = the headline; the exporter records one span per collected"
        " test, skipped included.\n"
        "%exec = exporter / GitHub EXECUTED (collected minus skipped), shown for"
        " context; %exec >100% is just the skipped tests the exporter counts.\n"
        "'GH coll'=0 means logs weren't parsed (job still running, or a failed job"
        " printed no pytest summary footer)."
    )
    if worst:
        print("\nUndercounted jobs (exporter < 95% of GitHub COLLECTED):")
        for job, pct in sorted(worst, key=lambda x: x[1]):
            print(f"  - {job}: {pct:.0f}% of collected tests captured")
        print(
            "Hints: if the run finished <10 min ago, suspect the roll-up lag first"
            " (see the header note) and re-run later. Otherwise the usual cause is"
            " large-trace read truncation (the exporter under-reads big shard"
            " traces Tempo fully stores — compare a shard's exporter count to a"
            " direct GET /api/traces/<id> span count). A low max/shard relative to"
            " collected/shards is the fingerprint. (shard mismatch -> enumeration/"
            " search gap; runner under-produce/queue drops -> see span-pipeline"
            " section below.)"
        )
    else:
        print(
            "\nAll jobs within 5% of GitHub COLLECTED counts — exporter agrees with GitHub."
        )

    print_debug_section(jobs, gh)
    return 0


def print_debug_section(jobs: list[str], gh: dict[str, dict]) -> None:
    """Print the runner-side span tally (TRANSFORMERSCI_OTEL_DEBUG=1), if present.

    Splits the loss into its three possible causes:
      - produced < collected  -> spans never created (pytest plugin didn't
        instrument every test item) — the fix is upstream of the export pipeline;
      - queue_dropped large (produced - submitted) -> the BatchSpanProcessor
        queue overflowed, so raise OTEL_BSP_MAX_QUEUE_SIZE / _MAX_EXPORT_BATCH_SIZE;
      - failed large -> the exporter could not ship the batch (timeout/rejection
        at the collector/ingress) and dropped it after retries — the fix is on the
        ingest path (client timeout, collector capacity). This last cause was
        invisible before: the old tally counted a span as 'exported' the moment it
        was handed to export(), so transport failures never showed up.
    """
    with_debug = [
        job for job in jobs if gh.get(job, {}).get("debug", {}).get("summaries")
    ]
    if not with_debug:
        print(
            "\nNo OTEL DEBUG SUMMARY lines in the logs — run with "
            "TRANSFORMERSCI_OTEL_DEBUG=1 on the runner to capture the span tally."
        )
        return

    columns = (
        f"{'test_job':24} {'GH collect':>10} {'produced':>9} {'exported':>9} "
        f"{'failed':>8} {'lost%':>6} {'queued':>7} {'shards':>7}"
    )
    print(
        "\nRunner span pipeline (TRANSFORMERSCI_OTEL_DEBUG=1; summed across "
        "shards/workers):"
    )
    print(columns)
    print("-" * len(columns))
    for job in with_debug:
        gh_entry = gh[job]
        dbg = gh_entry["debug"]
        gh_coll = collected(gh_entry["outcomes"])
        produced = dbg["produced"]
        exported = dbg["exported"]
        failed = dbg["failed"]
        queue_dropped = dbg["not_exported"]
        # 'lost%' is the share of submitted spans the exporter failed to ship —
        # the transport-loss signal the old tally hid.
        submitted = dbg["submitted"] or (exported + failed)
        lost_pct = (100 * failed / submitted) if submitted else float("nan")
        lost_col = "   n/a" if not submitted else f"{lost_pct:5.0f}%"
        flag = ""
        if submitted and failed > 0.05 * submitted:
            flag = " !export failures (ingest timeout/capacity — raise OTLP timeout / scale collector)"
        elif gh_coll and produced < 0.95 * gh_coll:
            flag = " !under-produced (spans never created)"
        elif produced and queue_dropped > 0.05 * produced:
            flag = " !queue drops (raise OTEL_BSP_MAX_QUEUE_SIZE)"
        print(
            f"{job:24} {gh_coll:>10} {produced:>9} {exported:>9} "
            f"{failed:>8} {lost_col:>6} {queue_dropped:>7} {dbg['summaries']:>7}{flag}"
        )
    print(
        "\nproduced = spans handed to the BatchSpanProcessor; exported = spans the "
        "exporter CONFIRMED shipped (result=SUCCESS); failed = spans whose export "
        "returned FAILURE/raised (LOST after retries — transport/ingest loss); "
        "queued = produced-submitted = BSP queue overflow. 'lost%' = failed/submitted. "
        "'shards' = processes that reported (≈ shards × xdist workers)."
    )


if __name__ == "__main__":
    sys.exit(main())
