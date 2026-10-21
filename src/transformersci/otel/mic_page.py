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
    "contract_revision",
    "revision",
    "fixture_repo",
    "source_model",
    "weights",
    "target",
    "variant",
    "framework",
    "fleet_repository",
    "fleet_revision",
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
    path = Path(__file__).parents[1] / "contracts/data/fixture-provenance.json"
    for pin, metadata in json.loads(path.read_text())["fixtures"].items():
        repo, revision = pin.rsplit("@", 1)
        if target := metadata.get("target"):
            fixtures.setdefault(target, (repo, revision))
            fixtures.setdefault(repo.split("/")[-1], (repo, revision))
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


def failure_summary(query, *, days, scope):
    """Count recorded MIC check outcomes across all variants and platforms."""
    selector = 'test_job="fast-integration"'
    if scope == "main":
        selector += ',pr="main"'
    elif scope == "prs":
        selector += ',pr=~"[0-9]+|gh-readonly-queue/.*"'
    counts = {}
    for field in ("total", "failed"):
        samples = query(
            "max by (run_id, hardware) "
            f"(last_over_time(pytest_run_job_{field}_tests{{{selector}}}[{days}d]))"
        )
        counts[field] = sum(
            int(value)
            for labels, value in _samples(samples)
            if re.fullmatch(r"\d+(?::\d+)?", labels.get("run_id", "")) and value >= 0
        )
    counts["rate"] = (
        100 * counts["failed"] / counts["total"] if counts["total"] else None
    )
    return counts


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


def _choices(value):
    return {value} if isinstance(value, str) and value else set(value or [])


def _case_key(row):
    return (
        row["fixture_repo"] or row["nodeid"],
        row["variant"],
        row["platform"],
        row["weights"],
        row.get("pr", ""),
    )


def add_history(cases, query, load_rows, loaded, starts, *, days, scope, first_day):
    """Use observed case records, never job totals, to establish passes and recovery."""
    if not cases:
        return
    selector = 'test_job="fast-integration"'
    if scope == "main":
        selector += ',pr="main"'
    elif scope == "prs":
        selector += ',pr=~"[0-9]+|gh-readonly-queue/.*"'
    totals = {
        labels["run_id"]: (labels.get("pr", ""), int(value))
        for labels, value in _samples(
            query(
                "max by (run_id, pr) "
                f"(last_over_time(pytest_run_job_total_tests{{{selector}}}[{days}d]))"
            )
        )
        if re.fullmatch(r"\d+(?::\d+)?", labels.get("run_id", ""))
    }
    runs = {run: pr for run, (pr, _) in totals.items()}
    records = dict(loaded)
    for run, rows in _parallel(set(runs) - set(loaded), load_rows).items():
        records[run] = {
            (r.get("trace_id"), r.get("test_nodeid")): r
            for r in rows or []
            if r.get("test_job") == "fast-integration"
        }
    by_key = {_case_key(case): case for case in cases}
    for case in cases:
        case["daily"] = [{"pass": 0, "fail": n, "skipped": 0} for n in case["days"]]
        case["latest"] = {
            "status": case["status"],
            "run_id": case["run_id"],
            "started_at": case["started_at"],
            "run_url": case["run_url"],
        }
        case["_observed"] = set(case["_runs"])

    def observe(case, run, pr, status):
        stamp = starts.get(run, 0)
        day = int(stamp // 86400) - first_day
        if run not in case["_observed"]:
            case["_observed"].add(run)
            if stamp and 0 <= day < days:
                bucket = (
                    "pass"
                    if status == "pass"
                    else "fail"
                    if status in {"fail", "error"}
                    else "skipped"
                )
                case["daily"][day][bucket] += 1
        if stamp > case["latest"]["started_at"]:
            case["latest"] = {
                "status": status,
                "run_id": run,
                "started_at": stamp,
                "run_url": "/d/pytest-observability-by-run/pytest-observability-run?"
                + urlencode(
                    {
                        "var-run_id": run,
                        "var-pr": pr,
                        "var-gh_run_id": run.split(":")[0],
                    }
                ),
            }

    for run, rows in records.items():
        for record in rows.values():
            status = record.get("mic.status")
            if status not in {"pass", "fail", "error", "xfail", "unsupported"}:
                continue
            node = str(record.get("test_nodeid", ""))
            if not node.startswith("tests/contracts::"):
                continue
            parts = node.split("::")
            target, _, variant = parts[-1].partition("[")
            repo = (
                record.get("mic.fixture_repo")
                or record.get("mic.baseline_repo")
                or record.get("mic.contract", "")
            )
            if not repo or repo.endswith("-integration"):
                repo = fixture_catalog().get(target, (repo, ""))[0]
            pr = runs.get(run)
            if pr is None:
                # Failure-only indexes already supply the branch identity.
                pr = next((c["pr"] for c in cases if run in c["_runs"]), "")
            key = (
                repo or node,
                record.get("mic.variant") or variant.rstrip("]"),
                record.get("mic.platform", ""),
                record.get("mic.weights") or "unknown",
                pr,
            )
            case = by_key.get(key)
            if case is None:
                continue
            observe(case, run, pr, status)
        # Setup errors have no success span. Clear them only when all recorded
        # checks of a later job are present and non-failing, never from totals alone.
        pr, total = totals.get(run, ("", 0))
        checks = {r.get("test_nodeid"): r.get("mic.status") for r in rows.values()}
        if (
            total > 0
            and len(checks) == total
            and all(
                status in {"pass", "xfail", "unsupported"} for status in checks.values()
            )
        ):
            for case in cases:
                if case["target"] == "runner" and case["pr"] == pr:
                    observe(case, run, pr, "pass")
    for case in cases:
        case.pop("_observed")
        case["sticky"] = case["latest"]["status"] in {"fail", "error"}


def build_rows(
    query,
    load_rows,
    *,
    days=7,
    scope="all",
    variant="",
    platform="",
    failure_type="",
    sticky_only=False,
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
    variants, platforms, failure_types = map(
        _choices, (variant, platform, failure_type)
    )
    filtered = [
        r
        for r in rows
        if (not variants or r["variant"] in variants)
        and (not platforms or r["platform"] in platforms)
        and (not failure_types or r["failure_kind"] in failure_types)
    ]
    now = time.time() if now is None else now
    first_day = int(now // 86400) - days + 1
    grouped = {}
    for row in filtered:
        key = _case_key(row)
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
    add_history(
        cases,
        query,
        load_rows,
        loaded,
        starts,
        days=days,
        scope=scope,
        first_day=first_day,
    )
    for case in cases:
        case["fails"] = len(case.pop("_runs"))
        repo = case["contract"]
        revision = case.get("contract_revision") or "main"
        case["contract_url"] = (
            f"https://huggingface.co/{repo}/blob/{revision}/integrations/transformers/usage.md"
            if re.fullmatch(r"[\w.-]+/[\w.-]+", repo)
            and (revision == "main" or re.fullmatch(r"[a-fA-F0-9]{40}", revision))
            else ""
        )
    if sticky_only:
        cases = [case for case in cases if case["sticky"]]
    return {
        "summary": failure_summary(query, days=days, scope=scope),
        "rows": cases[:LIMIT],
        "days": days,
        "scope": scope,
        "options": options,
        "total": len(filtered),
        "cases": len(cases),
        "first_day": first_day * 86400,
        "limit": LIMIT,
    }
