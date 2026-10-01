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
"""Report the daily-CI model jobs that were killed (exit 137) as failing tests.

When a daily-CI model job's pod goes over its memory limit, the whole job
container is killed: pytest never exports the running test's span, and no
later step of that job can run to report it, since the container is gone. The
dashboards then show the job with no failure.

This runs as a separate job after the model jobs. It reads the run's failed
jobs from the GitHub API, keeps those whose log shows ``exit code 137``, takes
the killed test from the ``pytest -v`` output in the log, and emits it through
``report-ci-failure --kind oom_killed`` with the killed job's own labels
(hardware, ci_event, job id), so the span lands next to that job's other spans.

The OTEL wrapper is passed after ``--`` (without ``--suite``, which is set per
job here):

    report-killed-jobs --repo huggingface/transformers --run-id "$GITHUB_RUN_ID" \\
      --attempt "$GITHUB_RUN_ATTEMPT" --ci-event "$CI_EVENT" -- \\
      configure-ci-otel --service-name pytest-observability --protocol http \\
        --otlp-endpoint "$OTEL_EXPORTER_OTLP_ENDPOINT" --token "$OTEL_TOKEN"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from transformersci.agentic.github_api import gh_headers

from .report_failure import parse_running_nodeid

_API = "https://api.github.com"

# Model-job names as GitHub shows them for the daily reusable workflows, e.g.
#   Model CI / run_models_gpu (aws-g5-4xlarge-cache, 0) / run_models_gpu (models/exaone4)
# The machine type is the caller's matrix value, not the runner the job got.
_MODEL_JOB_NAME = re.compile(
    r"\((?P<machine>aws-[^,()]+), \d+\) / (?P<suite>[a-z_]+) \((?P<folder>[^()]+)\)$"
)

# The k8s container hook's error once the step's process is SIGKILLed.
_KILLED_MARKER = re.compile(r"##\[error\].*exit code 137\b")

# `2026-10-01T17:28:39.8988650Z ` at the start of every job-log line.
_LOG_TIMESTAMP = re.compile(r"^\d{4}-\d\d-\d\dT[\d:.]+Z ", re.MULTILINE)
# A runner `::debug::` payload glued to the end of the line pytest was printing.
_LOG_DEBUG_SUFFIX = re.compile(r"::debug::.*$", re.MULTILINE)

# Same mapping as the model job's "Set `machine_type`" step.
_HARDWARE = {
    "aws-g5-4xlarge-cache": "single-gpu",
    "aws-g5-12xlarge-cache": "multi-gpu",
}


@dataclass(frozen=True)
class ModelJob:
    job_id: int
    suite: str
    folder: str
    hardware: str


def parse_model_job(job: Mapping[str, object]) -> ModelJob | None:
    """The daily model job a GitHub job entry is, or ``None`` for any other job."""
    match = _MODEL_JOB_NAME.search(str(job.get("name", "")))
    if match is None:
        return None
    machine = match.group("machine")
    return ModelJob(
        job_id=int(job["id"]),  # type: ignore[arg-type]
        suite=match.group("suite"),
        folder=match.group("folder"),
        hardware=_HARDWARE.get(machine, machine),
    )


def pytest_output_from_job_log(log: str) -> str | None:
    """The pytest output of a killed job's log, or ``None`` if it was not killed.

    The log is cut at the kill, and each line loses the runner's timestamp, so
    the result reads like the job's own ``test_outputs.txt``.
    """
    killed = _KILLED_MARKER.search(log)
    if killed is None:
        return None
    text = _LOG_TIMESTAMP.sub("", log[: killed.start()])
    return _LOG_DEBUG_SUFFIX.sub("", text)


def ci_event_slug(ci_event: str) -> str:
    """Same slug as the model job's "Run all tests on GPU" step ("Daily CI" -> "daily")."""
    slug = re.sub(r" ci$", "", ci_event.lower())
    slug = re.sub(r"[^a-z0-9._]+", "-", slug).strip("-")
    return slug or "daily"


class _DropAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """The job-log endpoint redirects to blob storage, which rejects the GitHub token."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            new.remove_header("Authorization")
        return new


_opener = urllib.request.build_opener(_DropAuthOnRedirect)


def _get(url: str, token: str) -> bytes:
    request = urllib.request.Request(url, headers=gh_headers(token))
    with _opener.open(request, timeout=60) as response:
        return response.read()


def list_failed_jobs(repo: str, run_id: str, attempt: str, token: str) -> list[dict]:
    jobs: list[dict] = []
    page = 1
    while True:
        url = (
            f"{_API}/repos/{repo}/actions/runs/{run_id}/attempts/{attempt}/jobs"
            f"?per_page=100&page={page}"
        )
        batch = json.loads(_get(url, token))["jobs"]
        jobs.extend(job for job in batch if job.get("conclusion") == "failure")
        if len(batch) < 100:
            return jobs
        page += 1


def fetch_job_log(repo: str, job_id: int, token: str) -> str:
    url = f"{_API}/repos/{repo}/actions/jobs/{job_id}/logs"
    return _get(url, token).decode("utf-8", errors="replace")


def report_command(wrapper: Sequence[str], job: ModelJob, crash_log: str) -> list[str]:
    return [
        *wrapper,
        "--suite",
        job.suite,
        "--",
        "report-ci-failure",
        "--kind",
        "oom_killed",
        "--crash-log",
        crash_log,
        "--message",
        f"{job.folder}: pytest was killed (exit 137, likely out of host memory)",
    ]


def report_env(env: Mapping[str, str], job: ModelJob, ci_event: str) -> dict[str, str]:
    """The env the killed job's own spans were emitted with, for its labels."""
    attributes = (
        f"transformers.test.ci_event={ci_event_slug(ci_event)},"
        f"transformers.test.hardware={job.hardware},"
        f"cicd.pipeline.task.run.id={job.job_id}"
    )
    return {**env, "OTEL_RESOURCE_ATTRIBUTES": attributes}


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    wrapper: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, wrapper = argv[:split], argv[split + 1 :]

    parser = argparse.ArgumentParser(
        description=(
            "Emit a failing test span for each daily-CI model job of a run that "
            "was killed (exit 137). Pass the configure-ci-otel wrapper after `--`."
        )
    )
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt", default="1")
    parser.add_argument("--ci-event", default="Daily CI")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be reported instead of emitting it.",
    )
    args = parser.parse_args(argv)
    if not wrapper and not args.dry_run:
        parser.error("pass the configure-ci-otel wrapper after `--`")

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        parser.error("GITHUB_TOKEN is not set")

    try:
        failed = list_failed_jobs(args.repo, args.run_id, args.attempt, token)
    except (urllib.error.URLError, KeyError, ValueError) as error:
        print(f"report-killed-jobs: cannot list jobs: {error}", file=sys.stderr)
        return 1

    status = 0
    for entry in failed:
        job = parse_model_job(entry)
        if job is None:
            continue
        try:
            output = pytest_output_from_job_log(
                fetch_job_log(args.repo, job.job_id, token)
            )
        except urllib.error.URLError as error:
            print(
                f"report-killed-jobs: cannot read the log of job {job.job_id}: {error}",
                file=sys.stderr,
            )
            status = 1
            continue
        if output is None:
            continue
        print(
            f"report-killed-jobs: job {job.job_id} ({job.suite}, {job.folder}, "
            f"{job.hardware}) was killed while running "
            f"{parse_running_nodeid(output) or 'no identifiable test'}",
            flush=True,
        )
        if args.dry_run:
            continue
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", encoding="utf-8", delete=False
        ) as handle:
            handle.write(output)
        try:
            result = subprocess.run(
                report_command(wrapper, job, handle.name),
                env=report_env(os.environ, job, args.ci_event),
                check=False,
            )
        finally:
            os.unlink(handle.name)
        if result.returncode != 0:
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
