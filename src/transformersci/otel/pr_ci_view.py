"""Combine the newest PR CI and run-slow runs for the PR dashboard."""

from __future__ import annotations

import re

RUN_LINK = re.compile(
    r"https://github\.com/huggingface/transformers/actions/runs/(\d+)"
)
ARTIFACT_LINK = re.compile(
    r"https://huggingface\.co/datasets/[^\s)]+/new_failures_with_bad_commit\.json"
)
TEST_LINE = re.compile(r"^\s+(tests/\S+::\S+)\s+\(", re.MULTILINE)
COMMIT = re.compile(
    r"^\|\s*(RUN|PR|main)\s*\|\s*\[[^]]+\]"
    r"\(https://github\.com/[^)]+/commit/([0-9a-f]{40})\)",
    re.MULTILINE,
)


def section(body: str, platform: str) -> str:
    """The run-slow bot updates two marked sections in one GitHub comment."""
    marker = rf"<!-- {platform}-section-start -->(.*?)<!-- {platform}-section-end -->"
    match = re.search(marker, body, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    if body.startswith(f"## CI Results ({platform.title()})"):
        return body
    return ""


def matching_comment(comments: list[dict], run_id: str) -> dict | None:
    number = run_id.split(":", 1)[0]
    for comment in sorted(
        comments, key=lambda item: item.get("updated_at", ""), reverse=True
    ):
        body = str(comment.get("body") or "")
        if number in RUN_LINK.findall(section(body, "nvidia")):
            return comment
    return None


def latest_platform_comment(comments: list[dict], platform: str) -> dict | None:
    for comment in sorted(
        comments, key=lambda item: item.get("updated_at", ""), reverse=True
    ):
        if RUN_LINK.search(section(str(comment.get("body") or ""), platform)):
            return comment
    return None


def comparison(body: str, platform: str) -> dict:
    content = section(body, platform)
    runs = RUN_LINK.findall(content)
    artifact = ARTIFACT_LINK.search(content)
    result = {
        "run_id": runs[0] if runs else "",
        "commits": {kind.lower(): sha for kind, sha in COMMIT.findall(content)},
        "report_url": artifact.group(0) if artifact else "",
        "specific": [],
        "state": "pending",
    }
    if not content:
        return result
    if "No failing test specific to this PR" in content:
        result["state"] = "complete"
        return result
    if "failed to report results" in content or "could not be completed" in content:
        result["state"] = "unavailable"
        return result
    if "new failed tests from this PR" not in content:
        return result
    result["specific"] = [
        {"nodeid": nodeid, "hardware": "", "status": "", "job_url": ""}
        for nodeid in dict.fromkeys(TEST_LINE.findall(content))
    ]
    result["state"] = "complete" if result["specific"] else "unavailable"
    return result


def newest_run(lane: dict) -> dict | None:
    candidates = [
        run for run in [lane.get("source_run"), *(lane.get("active_runs") or [])] if run
    ]
    return max(candidates, key=lambda run: run.get("created_at") or "", default=None)


def _job_identity(name: str, lane: str) -> tuple[str, str] | None:
    if lane == "cpu":
        match = re.search(
            r"^pr-ci / ((?:tests_|pipelines_|examples_)[a-z0-9_]+|Check (?:code quality|repository consistency))",
            name,
        )
        if not match or "Setup —" in name or "Collect" in name:
            return None
        job = match.group(1).lower().replace(" ", "_")
        return job, "cpu"
    match = re.search(r"/ (run_(?:models|quantization_torch)_gpu) \(", name)
    if not match:
        return None
    if lane == "amd":
        hardware = "amd-2gpu" if "2gpu" in name else "amd-1gpu"
    else:
        hardware = "multi-gpu" if "12xlarge" in name else "single-gpu"
    return match.group(1), hardware


def summarize_jobs(
    raw_jobs: list[dict], rows: list[dict], lane: str, run_conclusion: str = ""
) -> list[dict]:
    jobs: dict[tuple[str, str], dict] = {}
    github_jobs: set[tuple[str, str]] = set()
    for raw in raw_jobs:
        identity = _job_identity(str(raw.get("name") or ""), lane)
        if not identity:
            continue
        github_jobs.add(identity)
        job, hardware = identity
        item = jobs.setdefault(
            identity,
            {
                "name": job,
                "hardware": hardware,
                "status": "skipped",
                "pass": None,
                "fail": None,
                "started_at": "",
                "url": "",
            },
        )
        status = str(raw.get("conclusion") or raw.get("status") or "")
        priority = {
            "in_progress": 6,
            "queued": 6,
            "waiting": 6,
            "requested": 6,
            "pending": 6,
            "failure": 5,
            "timed_out": 5,
            "startup_failure": 5,
            "success": 3,
            "cancelled": 2,
            "skipped": 1,
        }
        if priority.get(status, 0) > priority.get(item["status"], 0):
            item["status"] = status
        started = str(raw.get("started_at") or "")
        if started and (not item["started_at"] or started < item["started_at"]):
            item["started_at"] = started
        if not item["url"] and raw.get("html_url"):
            item["url"] = str(raw["html_url"])
    for row in rows:
        job, hardware = str(row.get("test_job") or ""), str(row.get("hardware") or "")
        if not job or not hardware:
            continue
        if lane == "amd":
            continue
        identity = (job, hardware)
        item = jobs.setdefault(
            identity,
            {
                "name": job,
                "hardware": hardware,
                "status": "recorded",
                "pass": None,
                "fail": None,
                "started_at": "",
                "url": "",
            },
        )
        status = row.get("status_code")
        if status in {"OK", "ERROR"}:
            field = "fail" if status == "ERROR" else "pass"
            item[field] = (item[field] or 0) + 1
    for identity, item in jobs.items():
        if identity not in github_jobs:
            if item["fail"]:
                item["status"] = "failure"
            elif run_conclusion == "success" and item["pass"]:
                item["status"] = "success"
    return sorted(jobs.values(), key=lambda item: (item["hardware"], item["name"]))


def failures(rows: list[dict], comparison_data: dict | None = None) -> list[dict]:
    specific = {
        (item["nodeid"], item["hardware"])
        for item in (comparison_data or {}).get("specific", [])
    }
    name_only = {name for name, hardware in specific if not hardware}
    complete = (comparison_data or {}).get("state") == "complete"
    seen = set()
    result = []
    for row in rows:
        if row.get("status_code") != "ERROR":
            continue
        nodeid = str(row.get("test_nodeid") or "")
        hardware = str(row.get("hardware") or "")
        job = str(row.get("test_job") or "")
        key = (nodeid, hardware, job)
        if not nodeid or key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "nodeid": nodeid,
                "hardware": hardware,
                "job": job,
                "trace_id": str(row.get("trace_id") or ""),
                "duration_seconds": float(row.get("duration_seconds") or 0),
                "exception_type": str(row.get("exception_type") or ""),
                "exception_message": str(row.get("exception_message") or ""),
                "specific": ((nodeid, hardware) in specific or nodeid in name_only)
                if complete
                else None,
            }
        )
    return result[:300]
