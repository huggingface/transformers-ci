"""MIC failure rows from the existing failure index and persisted run details."""

from __future__ import annotations

import re
import json
from functools import lru_cache
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote, urlencode

PAGE_HTML = Path(__file__).with_suffix(".html").read_text()
SCRIPT = Path(__file__).with_suffix(".js").read_text()
FIELDS = (
    "contract",
    "revision",
    "fixture_repo",
    "source_model",
    "weights",
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


@lru_cache(maxsize=1)
def fixture_catalog():
    fixtures = {}
    for name in ("fleet-lock.json", "prototype-fleet.lock"):
        path = Path(__file__).parents[1] / "contracts/data" / name
        for contract in json.loads(path.read_text())["contracts"].values():
            for repo, pin in contract.get("fixtures", {}).items():
                item = (repo, pin["revision"])
                fixtures[pin.get("target", repo.split("/")[-1])] = item
                fixtures[repo.split("/")[-1]] = item
    return fixtures


def failure_kind(status, exception, target, text):
    if target == "runner":
        return "Setup error"
    if status == "error" or exception in {"MICExecutionError", "MICRunnerError"}:
        return "Runtime traceback"
    if "no reviewed baseline" in text:
        return "Missing baseline"
    if text.startswith("listed in known_failures but passes"):
        return "Unexpected pass"
    return "Baseline assertion"


def parse_scope(raw):
    return raw if raw in {"all", "main", "prs"} else "all"


def _parallel(items, loader):
    def one(item):
        try:
            return item, loader(item)
        except Exception:
            return item, None

    if not items or loader is None:
        return {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        return dict(pool.map(one, items))


def build_rows(
    query,
    load_rows,
    *,
    days=7,
    scope="all",
    variant="",
    platform="",
    failure_type="",
    trace_rows=None,
    build_metadata=None,
    now=None,
):
    days = parse_days(days)
    scope = parse_scope(scope)
    selector = 'test_job="fast-integration",test_nodeid=~"tests/contracts::.*"'
    if scope == "main":
        selector += ',pr="main"'
    elif scope == "prs":
        selector += ',pr=~"[0-9]+|gh-readonly-queue/.*"'
    failures = list(
        _samples(
            query(
                "max by (pr, run_id, trace_id, test_nodeid, exception_type) "
                f"(last_over_time(pytest_test_last_failure_info{{{selector}}}[{days}d]))"
            )
        )
    )
    failures = [
        (labels, value)
        for labels, value in failures
        if labels.get("test_nodeid", "").startswith("tests/contracts::")
        and re.fullmatch(r"\d+(?::\d+)?", labels.get("run_id", ""))
    ]
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
    # Load compact MIC records once per run, then recover missing older details
    # from small MIC traces. Bound concurrent Tempo requests and history backfill.
    run_ids = {
        item[0].get("run_id", "")
        for item in failures
        if re.fullmatch(r"\d+(?::\d+)?", item[0].get("run_id", ""))
    }
    loaded = {
        run: {
            (r.get("trace_id"), r.get("test_nodeid")): r
            for r in (records or [])
            if r.get("test_job") == "fast-integration"
        }
        for run, records in _parallel(run_ids, load_rows).items()
    }
    missing = []
    for labels, _ in failures:
        run, trace, node = (
            labels.get(k, "") for k in ("run_id", "trace_id", "test_nodeid")
        )
        record = loaded.get(run, {}).get((trace, node), {})
        if (
            not record.get("mic.failure")
            and re.fullmatch(r"[0-9a-fA-F]{32}", trace)
            and trace not in missing
        ):
            missing.append(trace)
    for trace, records in _parallel(missing[:256], trace_rows).items():
        for record in records or []:
            if record.get("test_job") == "fast-integration":
                loaded.setdefault(record.get("run_id", ""), {})[
                    (trace, record.get("test_nodeid"))
                ] = record
    titles = {}
    numbers = sorted(
        {
            match.group(1)
            for labels, _ in failures
            if (
                match := re.fullmatch(
                    r"(?:gh-readonly-queue/[^/]+/pr-)?([0-9]+)(?:-.*)?",
                    labels.get("pr", ""),
                )
            )
        }
    )
    matcher = "|".join(numbers) or "0"
    title_query = (
        "topk by (pr) (1, max by (pr, title) (max_over_time("
        f'(timestamp(pytest_pr_info{{pr=~"{matcher}",title!=""}}))[{days}d:]))) '
        "or on (pr) topk by (pr) (1, max by (pr, title) (max_over_time("
        f'(timestamp(ci_github_run_title_info{{pr=~"{matcher}",title!=""}}))[{days}d:])))'
    )
    try:
        titles = {
            labels["pr"]: labels.get("title", "")
            for labels, _ in _samples(query(title_query))
            if "pr" in labels
        }
    except Exception:
        pass  # PR metadata may lag the failure index; case/trace links still work.
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
        detail = loaded.get(run_id, {}).get((trace_id, nodeid), {})
        # Expected/unsupported cases are never displayed as active failures.
        if detail.get("mic.status") in {"xfail", "unsupported", "pass"}:
            continue
        parts = nodeid.split("::")
        target = parts[2] if len(parts) > 2 else "runner"
        target, _, case_variant = target.partition("[")
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
            variant=row["variant"] or case_variant.rstrip("]"),
            contract=row["contract"] or (parts[1] if len(parts) > 1 else "runner"),
            status=row["status"] or "fail",
            failure=str(
                detail.get("mic.failure")
                or detail.get("exception_message")
                or labels.get("exception_type", "")
            ),
            commit_sha=str(detail.get("mic.commit_sha", "")),
            repository=repository,
            run_url="/d/pytest-observability-by-run/pytest-observability-run?"
            + urlencode(
                {
                    "var-run_id": run_id,
                    "var-pr": labels.get("pr", ""),
                    "var-gh_run_id": run_id.split(":")[0],
                }
            ),
            trace_url="/d/pytest-test/test?"
            + urlencode(
                {
                    "var-trace_id": trace_id,
                    "var-test_nodeid": nodeid,
                    "var-run_id": run_id,
                    "var-pr": labels.get("pr", ""),
                    "var-gh_run_id": run_id.split(":")[0],
                    "from": f"now-{days}d",
                    "to": "now",
                }
            ),
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
        number = re.fullmatch(
            r"(?:gh-readonly-queue/[^/]+/pr-)?([0-9]+)(?:-.*)?", row["pr"]
        )
        row["pr_number"] = number.group(1) if number else ""
        row["pr_title"] = titles.get(row["pr_number"], titles.get(row["pr"], ""))
        row["pr_url"] = (
            "/d/pytest-observability-by-pr/pytest-observability-branch?"
            + urlencode({"var-pr": row["pr_number"] or row["pr"]})
        )
        row["fixture_repo"] = (
            row["fixture_repo"]
            or row["baseline_repo"]
            or (row["contract"] if "/" in row["contract"] else "")
        )
        row["fixture_revision"] = row["revision"] or row["baseline_revision"]
        entry = fixture_catalog().get(row["target"])
        if entry:
            # Renamed legacy fixture repositories retain their immutable history.
            # A catalog-only identity may name a model, but cannot establish the
            # weights used by a historical run without its recorded revision.
            if not row["fixture_repo"] or row["fixture_repo"].endswith("-integration"):
                row["fixture_repo"] = entry[0]
            row["catalog_revision"] = entry[1]
        else:
            row["catalog_revision"] = ""
        row["failure_kind"] = failure_kind(
            row["status"],
            labels.get("exception_type", ""),
            row["target"],
            row["failure"],
        )
        rows.append(row)
    builds = _parallel(
        {
            (r["fixture_repo"], r["fixture_revision"] or r["catalog_revision"])
            for r in rows
            if r["fixture_repo"] and (r["fixture_revision"] or r["catalog_revision"])
        },
        build_metadata,
    )
    for row in rows:
        provenance = (
            builds.get(
                (
                    row["fixture_repo"],
                    row["fixture_revision"] or row["catalog_revision"],
                )
            )
            or {}
        )
        row["source_model"] = row["source_model"] or provenance.get("source_model", "")
        row["weights"] = row["weights"] or (
            provenance.get("weights", "unknown")
            if row["fixture_revision"]
            else "unknown"
        )
        for key in ("source_model", "fixture_repo"):
            value = row[key]
            row[key + "_url"] = (
                f"https://huggingface.co/{value}"
                if re.fullmatch(r"[\w.-]+/[\w.-]+", value)
                else ""
            )
        if row["failure"] in {
            "MICAssertionError",
            "MICExecutionError",
            "MICRunnerError",
            "",
        }:
            row["failure"] = "Failure details unavailable; open the trace."
    options = {
        "failure_type": sorted({r["failure_kind"] for r in rows}),
        "variant": sorted({r["variant"] for r in rows if r["variant"]}),
        "platform": sorted({r["platform"] for r in rows if r["platform"]}),
    }
    filtered = [
        r
        for r in rows
        if (not variant or r["variant"] == variant)
        and (not platform or r["platform"] == platform)
        and (not failure_type or r["failure_kind"] == failure_type)
    ]
    now = time.time() if now is None else now
    first_day = int(now // 86400) - days + 1
    grouped = {}
    for row in filtered:
        key = (
            row["fixture_repo"] or row["nodeid"],
            row["variant"],
            row["platform"],
            row["weights"],
        )
        if key not in grouped:
            grouped[key] = {**row, "days": [0] * days, "_runs": set()}
        group = grouped[key]
        if row["run_id"] in group["_runs"]:
            continue
        group["_runs"].add(row["run_id"])
        day = int(row["started_at"] // 86400) - first_day
        if row["started_at"] and 0 <= day < days:
            group["days"][day] += 1
    cases = list(grouped.values())
    for case in cases:
        case["fails"] = len(case.pop("_runs"))
    return {
        "rows": cases[:LIMIT],
        "days": days,
        "scope": scope,
        "options": options,
        "total": len(filtered),
        "cases": len(cases),
        "first_day": first_day * 86400,
        "limit": LIMIT,
    }
