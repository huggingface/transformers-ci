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
# Above this many runs the start-time query reads all runs instead of naming
# them, keeping the request URL bounded (~15 bytes per run id).
MAX_RUN_MATCHER = 2000


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


def parse_group(raw):
    return raw if raw in {"model", "pr"} else "model"


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


def run_totals(query, *, days, scope):
    """``run_id -> (pr, total checks)`` for every MIC job in the window."""
    selector = 'test_job="fast-integration"'
    if scope == "main":
        selector += ',pr="main"'
    elif scope == "prs":
        selector += ',pr=~"[0-9]+|gh-readonly-queue/.*"'
    return {
        labels["run_id"]: (labels.get("pr", ""), int(value))
        for labels, value in _samples(
            query(
                "max by (run_id, pr) "
                f"(last_over_time(pytest_run_job_total_tests{{{selector}}}[{days}d]))"
            )
        )
        if re.fullmatch(r"\d+(?::\d+)?", labels.get("run_id", ""))
    }


def run_starts(query, run_ids, *, days):
    """Start times of the given runs only: the unfiltered query reads every
    CI run of the window (4,188 series over 30 days, 1.2-20s on 2026-10-22)."""
    if not run_ids:
        return {}
    matcher = (
        f'{{run_id=~"{"|".join(sorted(run_ids))}"}}'
        if len(run_ids) <= MAX_RUN_MATCHER
        else ""
    )
    return {
        labels.get("run_id", ""): stamp
        for labels, stamp in _samples(
            query(
                "max by (run_id) "
                f"(last_over_time(pytest_run_start_time_seconds{matcher}[{days}d]))"
            )
        )
    }


def add_history(
    cases,
    query,
    load_rows,
    loaded,
    starts,
    *,
    days,
    scope,
    first_day,
    totals=None,
    build_metadata=None,
):
    """Use observed case records, never job totals, to establish passes and recovery.

    Returns the latest status of every case seen, the failing ones and every
    recorded check, keyed like ``_case_key``."""
    if not cases:
        return {}
    if totals is None:
        totals = run_totals(query, days=days, scope=scope)
    runs = {run: pr for run, (pr, _) in totals.items()}
    records = dict(loaded)
    for run, rows in _parallel(set(runs) - set(loaded), load_rows).items():
        records[run] = {
            (r.get("trace_id"), r.get("test_nodeid")): r
            for r in rows or []
            if r.get("test_job") == "fast-integration"
        }
    by_key = {_case_key(case): case for case in cases}
    latest = {key: (case["started_at"], case["status"]) for key, case in by_key.items()}

    provenance = {}

    def weights(repo, record):
        # As build_rows fills a row's weights: older records carry none, and
        # keyed "unknown" they never matched their case, so an old failure
        # read as a separate case that no later pass could clear.
        revision = record.get("mic.revision") or record.get("mic.baseline_revision")
        if not (repo and revision and build_metadata):
            return "unknown"
        if (repo, revision) not in provenance:
            try:
                provenance[repo, revision] = build_metadata((repo, revision)) or {}
            except Exception:
                provenance[repo, revision] = {}
        return provenance[repo, revision].get("weights", "unknown")

    def note(key, run, status):
        stamp = starts.get(run, 0)
        if key not in latest or stamp > latest[key][0]:
            latest[key] = (stamp, status)

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
                record.get("mic.weights") or weights(repo, record),
                pr,
            )
            note(key, run, status)
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
                    note(_case_key(case), run, "pass")
    for case in cases:
        case.pop("_observed")
        case["sticky"] = case["latest"]["status"] in {"fail", "error"}
    return {key: status for key, (_stamp, status) in latest.items()}


def _group_cases(rows, *, days, first_day):
    grouped = {}
    for row in rows:
        key = _case_key(row)
        if key not in grouped:
            grouped[key] = {**row, "days": [0] * days, "_runs": set()}
        case = grouped[key]
        if row["run_id"] in case["_runs"]:
            continue
        case["_runs"].add(row["run_id"])
        day = int(row["started_at"] // 86400) - first_day
        if row["started_at"] and 0 <= day < days:
            case["days"][day] += 1
    return list(grouped.values())


def failing_now(latest):
    """Cases whose latest recorded check fails, out of every case with a check
    in the window: what Sticky shows, where the global rate counts every failed
    outcome, including the ones a later run cleared."""
    checked = [status for status in latest.values() if status != "unsupported"]
    failing = sum(status in {"fail", "error"} for status in checked)
    return {
        "failing": failing,
        "cases": len(checked),
        "rate": 100 * failing / len(checked) if checked else None,
    }


def failure_index(query, *, days, scope):
    """``(labels, value)`` of every MIC check failure in the window."""
    selector = 'test_job="fast-integration",test_nodeid=~"tests/contracts::.*"'
    if scope == "main":
        selector += ',pr="main"'
    elif scope == "prs":
        selector += ',pr=~"[0-9]+|gh-readonly-queue/.*"'
    return [
        (labels, value)
        for labels, value in _samples(
            query(
                "max by (pr, run_id, trace_id, test_nodeid, exception_type) "
                f"(last_over_time(pytest_test_last_failure_info{{{selector}}}[{days}d]))"
            )
        )
        if labels.get("test_nodeid", "").startswith("tests/contracts::")
        and re.fullmatch(r"\d+(?::\d+)?", labels.get("run_id", ""))
    ]


def pr_titles(query, failures, *, days):
    """Newest title of each PR the failures name; empty when unavailable."""
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
    if not numbers:
        return {}
    try:
        titles = _newest_titles(query, "pytest_pr_info", numbers, days=days)
        rest = [number for number in numbers if number not in titles]
        if rest:
            titles.update(
                _newest_titles(query, "ci_github_run_title_info", rest, days=days)
            )
        return titles
    except Exception:
        return {}  # PR metadata may lag the failure index; case/trace links still work.


# A renamed PR has one title series per name. The newest name is the one seen in
# the narrowest window; a subquery over the whole window (per-minute timestamps
# of every series, 6.5s for 30 days) is only needed to break ties inside it.
TITLE_WINDOWS = ((3600, "1h"), (6 * 3600, "6h"), (86400, "1d"), (7 * 86400, "7d"))


def _newest_titles(query, metric, numbers, *, days):
    windows = [label for seconds, label in TITLE_WINDOWS if seconds < days * 86400]
    windows.append(f"{days}d")
    selector = f'{metric}{{pr=~"{"|".join(numbers)}",title!=""}}'
    seen = {}
    for window, result in zip(
        windows,
        _map(
            lambda window: query(
                f"max by (pr, title) (last_over_time({selector}[{window}]))"
            ),
            windows,
        ),
    ):
        for labels, _ in _samples(result):
            if "pr" in labels:
                seen.setdefault(window, {}).setdefault(labels["pr"], set()).add(
                    labels.get("title", "")
                )
    titles, ties = {}, {}
    for number in numbers:
        for window in windows:
            names = seen.get(window, {}).get(number)
            if names:
                if len(names) == 1:
                    titles[number] = next(iter(names))
                else:
                    ties.setdefault(window, []).append(number)
                break
    for window, tied in ties.items():
        for labels, _ in _samples(
            query(
                "topk by (pr) (1, max by (pr, title) (max_over_time((timestamp("
                f'{metric}{{pr=~"{"|".join(tied)}",title!=""}}))[{window}:])))'
            )
        ):
            if "pr" in labels:
                titles[labels["pr"]] = labels.get("title", "")
    return titles


def _map(function, items):
    """``function`` over ``items`` concurrently, in order; raises the first error."""
    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(function, items))


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
    group="model",
    trace_rows=None,
    build_metadata=None,
    now=None,
):
    days = parse_days(days)
    scope = parse_scope(scope)
    group = parse_group(group)
    with ThreadPoolExecutor(max_workers=3) as pool:
        # Queries that do not depend on the failures run beside them.
        summary = pool.submit(failure_summary, query, days=days, scope=scope)
        totals = pool.submit(run_totals, query, days=days, scope=scope)
        failures = failure_index(query, days=days, scope=scope)
        titles = pool.submit(pr_titles, query, failures, days=days)
        totals = totals.result()
        run_ids = {item[0].get("run_id", "") for item in failures}
        starts = run_starts(query, run_ids | set(totals), days=days)
        failures.sort(
            key=lambda item: (
                starts.get(item[0].get("run_id", ""), 0),
                item[0].get("run_id", ""),
            ),
            reverse=True,
        )
        # Load compact MIC records once per run (the failing runs and the ones
        # the history reads), then recover missing older details from small MIC
        # traces. Bound concurrent Tempo requests and history backfill.
        loaded = {
            run: {
                (r.get("trace_id"), r.get("test_nodeid")): r
                for r in (records or [])
                if r.get("test_job") == "fast-integration"
            }
            for run, records in _parallel(run_ids | set(totals), load_rows).items()
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
        titles = titles.result()
        summary = summary.result()
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
    history = dict(
        query=query,
        load_rows=load_rows,
        loaded=loaded,
        starts=starts,
        days=days,
        scope=scope,
        first_day=first_day,
        totals=totals,
        build_metadata=build_metadata,
    )
    # Like the global rate, independent of the table's filters.
    current = failing_now(
        add_history(_group_cases(rows, days=days, first_day=first_day), **history)
    )
    cases = _group_cases(filtered, days=days, first_day=first_day)
    add_history(cases, **history)
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

    # Limit whole displayed groups, never split their error list or summary.
    def model_key(case):
        return case["source_model"] or case["fixture_repo"] or case["contract"]

    models = list(dict.fromkeys(model_key(case) for case in cases))

    def group_key(case):
        return (
            (case["repository"], case["pr_number"] or case["pr"])
            if group == "pr"
            else model_key(case)
        )

    groups = list(dict.fromkeys(group_key(case) for case in cases))
    shown_groups = set(groups[:LIMIT])
    return {
        "summary": {**summary, "now": current},
        "rows": [case for case in cases if group_key(case) in shown_groups],
        "days": days,
        "scope": scope,
        "options": options,
        "total": len(filtered),
        "cases": len(cases),
        "models": len(models),
        "group": group,
        "groups": len(groups),
        "first_day": first_day * 86400,
        "limit": LIMIT,
    }
