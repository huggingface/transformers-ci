"""MIC failure rows from the existing failure index and persisted run details."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote, urlencode

PAGE_HTML = Path(__file__).with_suffix(".html").read_text()
SCRIPT = Path(__file__).with_suffix(".js").read_text()
FIELDS = (
    "contract",
    "target",
    "variant",
    "framework",
    "platform",
    "selected_by",
    "baseline_repo",
    "baseline_revision",
    "baseline_path",
    "status",
    "known_failure",
)
ROW_FIELDS = tuple(f"mic.{field}" for field in FIELDS) + (
    "mic.failure",
    "mic.repository",
    "mic.commit_sha",
)
LIMIT = 200


def parse_days(raw):
    return int(raw) if str(raw) in {"1", "7", "14", "30"} else 7


def _samples(items):
    for item in items:
        try:
            yield item["metric"], float(item["value"][1])
        except (KeyError, TypeError, ValueError, IndexError):
            continue


def build_rows(query, load_rows, *, days=7):
    days = parse_days(days)
    selector = 'test_job="fast-integration",test_nodeid=~"tests/contracts::.*"'
    failures = list(
        _samples(
            query(
                "max by (pr, run_id, trace_id, test_nodeid, exception_type) "
                f"(last_over_time(pytest_test_last_failure_info{{{selector}}}[{days}d]))"
            )
        )
    )
    starts = {
        labels.get("run_id", ""): stamp
        for labels, stamp in _samples(
            query(
                "max by (run_id) "
                f"(last_over_time(pytest_run_start_time_seconds[{days}d]))"
            )
        )
    }
    failures.sort(
        key=lambda item: (
            starts.get(item[0].get("run_id", ""), 0),
            item[0].get("run_id", ""),
        ),
        reverse=True,
    )
    loaded = {}
    rows = []
    seen = set()
    for labels, _ in failures:
        nodeid = labels.get("test_nodeid", "")
        run_id, trace_id = labels.get("run_id", ""), labels.get("trace_id", "")
        if not nodeid.startswith("tests/contracts::") or not re.fullmatch(
            r"\d+(?::\d+)?", run_id
        ):
            continue
        key = (run_id, trace_id, nodeid)
        if key in seen:
            continue
        seen.add(key)
        if len(rows) >= LIMIT:
            continue
        if run_id not in loaded:
            loaded[run_id] = {
                (r.get("trace_id"), r.get("test_nodeid")): r
                for r in (load_rows(run_id) or [])
                if r.get("test_job") == "fast-integration"
            }
        detail = loaded[run_id].get((trace_id, nodeid), {})
        # Expected/unsupported cases are never displayed as active failures.
        if detail.get("mic.status") in {"xfail", "unsupported", "pass"}:
            continue
        parts = nodeid.split("::")
        target = parts[2] if len(parts) > 2 else "runner"
        target, _, variant = target.partition("[")
        row = {field: str(detail.get(f"mic.{field}", "")) for field in FIELDS}
        repository = str(detail.get("mic.repository") or "huggingface/transformers")
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository):
            repository = "huggingface/transformers"
        row.update(
            nodeid=nodeid,
            run_id=run_id,
            trace_id=trace_id,
            pr=labels.get("pr", ""),
            started_at=starts.get(run_id, 0),
            target=row["target"] or target,
            variant=row["variant"] or variant.rstrip("]"),
            contract=row["contract"] or (parts[1] if len(parts) > 1 else "runner"),
            status=row["status"] or "fail",
            failure=str(
                detail.get("mic.failure")
                or detail.get("exception_message")
                or labels.get("exception_type", "")
            ),
            commit_sha=str(detail.get("mic.commit_sha", "")),
            run_url=f"https://github.com/{repository}/actions/runs/{run_id.split(':')[0]}",
            trace_url="/failure?"
            + urlencode({"trace_id": trace_id, "test_nodeid": nodeid}),
            baseline_url="",
        )
        repo, rev, path = (
            row["baseline_repo"],
            row["baseline_revision"],
            row["baseline_path"],
        )
        if (
            re.fullmatch(r"[\w.-]+/[\w.-]+", repo)
            and re.fullmatch(r"[a-fA-F0-9]{40}", rev)
            and path
            and ".." not in path.split("/")
        ):
            row["baseline_url"] = (
                f"https://huggingface.co/{repo}/blob/{rev}/{quote(path, safe='/')}"
            )
        rows.append(row)
    return {"rows": rows, "days": days, "total": len(seen), "limit": LIMIT}
