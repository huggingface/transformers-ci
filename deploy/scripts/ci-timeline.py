#!/usr/bin/env python3
"""How long a CI job takes to reach the dashboard, stage by stage.

For recent runs of one workflow, joins three clocks per job:

* GitHub Actions — run/job created, job started, the check step's start and
  end, job completed, and the last job of the run completed;
* Tempo — the earliest span the job emitted (``report-ci-start``);
* Prometheus — the first sample of the job's discovery series
  (``pytest_ci_runner_execution_info``), of ``pytest_run_job_active``, and of
  its results (``pytest_run_job_total_tests``).

and prints the distribution of each gap. Prometheus sample times are exact
(``timestamp()`` over a range query), not step-rounded. Nothing here measures
the browser: a rendered row is at least one Grafana refresh later.

Examples
--------
  # The repo-consistency and code-quality jobs of the last 40 completed PR runs:
  deploy/scripts/ci-timeline.py

  # Another job, keyed by its GitHub name and its telemetry test_job:
  deploy/scripts/ci-timeline.py --job "Check repository consistency=check_repository_consistency"

  # Keep the joined records for a before/after comparison:
  deploy/scripts/ci-timeline.py --runs 80 --json before.json

GitHub auth: ``GITHUB_TOKEN``, else ``gh auth token``. Grafana: as tempo.py.
Vanilla Python 3.10 + stdlib only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "tempo_cli", Path(__file__).with_name("tempo.py")
)
tempo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tempo)

GITHUB_API = "https://api.github.com"
DEFAULT_JOBS = (
    "Check repository consistency=check_repository_consistency",
    "Check code quality=check_code_quality",
)


def _github_token() -> str | None:
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    try:
        return subprocess.check_output(["gh", "auth", "token"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _github(path: str, token: str | None, attempts: int = 4) -> dict:
    request = urllib.request.Request(f"{GITHUB_API}{path}")
    request.add_header("Accept", "application/vnd.github+json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    # A 103-job listing page occasionally 502s; one bad page used to throw away a
    # ten-minute collection. Retry server errors and timeouts, not 4xx.
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code < 500 or attempt == attempts - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts - 1:
                raise
        time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def _ts(value: str | None) -> float | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _run_jobs(repo: str, run_id: int, attempt: int, token: str | None) -> list[dict]:
    jobs: list[dict] = []
    page = 1
    while True:
        body = _github(
            f"/repos/{repo}/actions/runs/{run_id}/attempts/{attempt}/jobs"
            f"?per_page=100&page={page}",
            token,
        )
        jobs.extend(body.get("jobs", []))
        if len(jobs) >= body.get("total_count", 0) or not body.get("jobs"):
            return jobs
        page += 1


def _prom_first_last(
    base: str, expr: str, start: float, end: float
) -> tuple[float | None, float | None]:
    body = tempo.proxy_get_json(
        base,
        tempo.PROM_UID,
        "/api/v1/query_range",
        {
            "query": f"timestamp({expr})",
            "start": int(start),
            "end": int(end),
            "step": 15,
        },
    )
    values = [float(v) for s in body["data"]["result"] for _, v in s["values"]]
    return (min(values), max(values)) if values else (None, None)


def _first_span(
    base: str, run: str, job_key: str, start: float, end: float
) -> float | None:
    query = (
        f'{{ resource.cicd.pipeline.run.id = "{run}" && '
        f'resource.cicd.pipeline.task.name = "{job_key}" }}'
    )
    body = tempo.proxy_get_json(
        base,
        tempo.TEMPO_UID,
        "/api/search",
        {"q": query, "start": int(start), "end": int(end), "limit": 50},
    )
    starts = [int(t["startTimeUnixNano"]) / 1e9 for t in body.get("traces", [])]
    return min(starts) if starts else None


def collect(args: argparse.Namespace) -> list[dict]:
    token = _github_token()
    jobs = dict(item.split("=", 1) for item in args.job or DEFAULT_JOBS)
    # Newest first, completed ones kept client-side. The API's own
    # status=completed filter goes through a search index and can answer with
    # runs weeks old (2026-09-25: the "latest" 30 were from Sep 4-6).
    runs: list[dict] = []
    for page in range(1, 11):
        query = urllib.parse.urlencode(
            {"per_page": 100, "event": args.event, "page": page}
        )
        batch = _github(
            f"/repos/{args.repo}/actions/workflows/{args.workflow}/runs?{query}", token
        )["workflow_runs"]
        runs.extend(r for r in batch if r.get("status") == "completed")
        if len(runs) >= args.runs or len(batch) < 100:
            break
    runs = runs[: args.runs]
    if runs:
        print(
            f"sample: {len(runs)} completed runs created "
            f"{runs[-1]['created_at']} .. {runs[0]['created_at']}",
            file=sys.stderr,
        )

    records: list[dict] = []
    for run in runs:
        run_key = f"{run['id']}:{run['run_attempt']}"
        all_jobs = _run_jobs(args.repo, run["id"], run["run_attempt"], token)
        run_done = max(
            (_ts(j["completed_at"]) for j in all_jobs if j["completed_at"]),
            default=None,
        )
        for job in all_jobs:
            name = job["name"].split(" / ")[-1]
            if name not in jobs or not job["started_at"]:
                continue
            key = jobs[name]
            step = next(
                (s for s in job["steps"] if s["name"] == name and s["started_at"]), None
            )
            window = (_ts(job["created_at"]) - 300, _ts(job["completed_at"]) + 3600)
            selector = f'{{run_id="{run_key}",test_job="{key}"}}'
            discovered = _prom_first_last(
                args.base, f"pytest_ci_runner_execution_info{selector}", *window
            )
            active = _prom_first_last(
                args.base, f"pytest_run_job_active{selector}", *window
            )
            results = _prom_first_last(
                args.base, f"max(pytest_run_job_total_tests{selector})", *window
            )
            records.append(
                {
                    "run": run_key,
                    "pr": (run.get("pull_requests") or [{}])[0].get("number"),
                    "job": key,
                    "conclusion": job["conclusion"],
                    "run_created": _ts(run["created_at"]),
                    "job_created": _ts(job["created_at"]),
                    "job_started": _ts(job["started_at"]),
                    "job_completed": _ts(job["completed_at"]),
                    "step_started": _ts(step["started_at"]) if step else None,
                    "step_completed": _ts(step["completed_at"]) if step else None,
                    "run_last_job_completed": run_done,
                    "first_span": _first_span(args.base, run_key, key, *window),
                    "discovered": discovered[0],
                    "active_first": active[0],
                    "active_last": active[1],
                    "results": results[0],
                }
            )
            print(f"  {run_key} {key}", file=sys.stderr)
    return records


def _gap(record: dict, later: str, earlier: str) -> float | None:
    if record.get(later) is None or record.get(earlier) is None:
        return None
    return record[later] - record[earlier]


STAGES = (
    ("run created -> job created", "job_created", "run_created"),
    ("job created -> job started (runner queue)", "job_started", "job_created"),
    ("job started -> first span", "first_span", "job_started"),
    ("check step started -> first span", "first_span", "step_started"),
    ("first span -> discovered in Prometheus", "discovered", "first_span"),
    ("job created -> discovered (end to end)", "discovered", "job_created"),
    ("check step ended -> results in Prometheus", "results", "step_completed"),
    ("run's last job completed -> results", "results", "run_last_job_completed"),
    ("job completed -> last 'active' sample", "active_last", "job_completed"),
)


def _describe(values: list[float]) -> str:
    values = sorted(values)
    if not values:
        return "n/a"
    p90 = values[min(len(values) - 1, int(0.9 * len(values)))]
    return (
        f"n={len(values):3d}  min={values[0]:6.0f}  p50={statistics.median(values):6.0f}"
        f"  p90={p90:6.0f}  max={values[-1]:6.0f}"
    )


def report(records: list[dict]) -> None:
    for job in sorted({r["job"] for r in records}):
        rows = [r for r in records if r["job"] == job]
        print(f"== {job}  ({len(rows)} jobs, seconds)")
        for label, later, earlier in STAGES:
            gaps = [g for r in rows if (g := _gap(r, later, earlier)) is not None]
            print(f"  {label:45s} {_describe(gaps)}")
        seen = [r for r in rows if r["discovered"] is not None]
        late = [r for r in seen if r["discovered"] > r["job_completed"]]
        print(f"  first seen only after it had completed: {len(late)}/{len(seen)}")
        for r in rows:
            if r["first_span"] is None:
                print(f"  no telemetry: run {r['run']} ({r['conclusion']})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base", default=tempo.DEFAULT_BASE, help="Grafana base URL")
    parser.add_argument("--repo", default="huggingface/transformers")
    parser.add_argument("--workflow", default="pr-ci-caller.yml")
    parser.add_argument("--event", default="pull_request")
    parser.add_argument("--runs", type=int, default=40, help="completed runs to join")
    parser.add_argument(
        "--job",
        action="append",
        metavar="GITHUB_NAME=TEST_JOB",
        help="job to follow (repeatable; default: repo-consistency and code quality)",
    )
    parser.add_argument("--json", metavar="PATH", help="also write the joined records")
    parser.add_argument(
        "--from-json",
        metavar="PATH",
        help="report on saved records instead of fetching",
    )
    args = parser.parse_args(argv)

    if args.from_json:
        records = json.loads(Path(args.from_json).read_text())
    else:
        records = collect(args)
    if args.json:
        Path(args.json).write_text(json.dumps(records, indent=1))
    report(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
