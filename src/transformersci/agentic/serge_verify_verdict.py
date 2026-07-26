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

"""Verdict computer for the serge GPU verify loop.

The verify workflow (``.github/workflows/serge-verify-slow.yml``) runs the
targeted failing tests on two trees — the pre-patch tree (``baseline``) and
serge's candidate (``patched``) — and repeats each tree's run several times
(fresh process each) to rule out flakiness, writing one pytest JUnit XML per
run. This tool reads all of them (one ``--baseline`` / ``--patched`` XML per
run) and emits the machine-readable verdict serge polls for: the patch turned
the targeted tests red → green on EVERY run, without breaking neighbours.

Why JUnit XML: it is built into pytest core (no extra dependency in the CI
container), gives one structured ``<testcase>`` per test with an explicit
``<failure>``/``<error>``/``<skipped>`` child, and carries the failure text we
feed back to the LLM. We deliberately do NOT reuse transformers' ``--make-
reports`` output here — that pipeline targets the daily-CI HF dataset, whereas
we just need per-nodeid pass/fail for a handful of tests we already know.

Verdicts (``mode=verify``, the default — baseline + patched):
  fixed            every targeted test was red at baseline and green when patched
  not_fixed        at least one targeted test is still red when patched
  already_passing  a targeted test was NOT red at baseline (self-healed / flaky)
                   — the baseline-red guard; serge must NOT open a PR
  broke_others     patch is green on target but introduced NEW collateral failures
                   (only computed when a collateral baseline is supplied)
  error            a targeted test could not be verified (missing / skipped /
                   collection error)

Verdicts (``mode=reproduce`` — baseline only, run BEFORE serge investigates):
  reproduced       every targeted test is red at ``base_sha`` — a real, confirmed
                   failure; serge captures the traceback and proceeds to investigate
  not_reproduced   a targeted test is green at ``base_sha`` (stale / flaky / env) —
                   serge must NOT investigate, and bails to the next group
  error            a targeted test could not be verified (missing / skipped /
                   collection error)

See ``docs/plans/serge-gpu-verify-loop.md`` and ``docs/plans/serge-reproduce-first.md``
in transformers-ci-playbooks.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET


def nodeid_key(nodeid: str) -> tuple[str, str]:
    """``(class, method)`` key for matching a pytest node-id to a JUnit
    ``<testcase>``. ``tests/…/test_x.py::SomeTest::test_foo[p]`` → ``("SomeTest",
    "test_foo[p]")``. Function-level tests (no class) yield ``("", method)``."""
    parts = nodeid.split("::")
    method = parts[-1].strip()
    cls = parts[-2].strip() if len(parts) >= 3 else ""
    return (cls, method)


# Literal prefixes whose (paren-balanced) bodies we cap in a failure traceback.
# These are the big numeric dumps `pytest --showlocals` (+ the serge_showlocals
# plugin) emits — model outputs, intermediate embeddings, logits — that bury the
# tiny asserted slice under tens of KB. torch tensors and numpy arrays cover the
# integration-test failures serge fixes.
_BIG_LITERAL_PREFIXES = ("tensor(", "array(")


def distill_failure(text: str, tensor_cap: int = 320) -> str:
    """Deterministically shrink a pytest ``--showlocals`` failure longrepr.

    Caps every ``tensor(...)`` / ``array(...)`` literal to ``tensor_cap`` chars of
    its body and leaves everything else — the assertion header, the mismatch
    stats, the source, and every variable name — untouched. The asserted slice
    (a handful of elements) prints in full; the KB-scale intermediate tensors get
    elided, whether they are frame locals or fields nested inside a big
    ``ModelOutput(...)`` repr. Nesting-agnostic and stable, so the actual-vs-
    expected the LLM needs survives instead of being pushed out by a blunt tail/
    head truncation (see the owlvit case: a 100 KB traceback where the asserted
    ``pred_boxes`` sat behind ~90 KB of intermediate embeddings)."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        # earliest next big-literal prefix from i
        j, prefix = -1, ""
        for pre in _BIG_LITERAL_PREFIXES:
            p = text.find(pre, i)
            if p != -1 and (j == -1 or p < j):
                j, prefix = p, pre
        if j == -1:
            out.append(text[i:])
            break
        out.append(text[i:j])
        # walk to the matching close paren of the literal
        k = j + len(prefix) - 1  # index of the opening '('
        depth = 0
        while k < n:
            ch = text[k]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    k += 1
                    break
            k += 1
        body = text[j:k]
        if len(body) <= tensor_cap:
            out.append(body)
        else:
            out.append(f"{body[:tensor_cap]}…<+{len(body) - tensor_cap} chars elided>)")
        i = k
    return "".join(out)


def _testcase_outcome(tc: ET.Element) -> tuple[str, str]:
    """``(outcome, detail_text)`` for one JUnit ``<testcase>``."""
    failure = tc.find("failure")
    if failure is not None:
        return ("failed", _detail(failure))
    error = tc.find("error")
    if error is not None:
        return ("error", _detail(error))
    if tc.find("skipped") is not None:
        return ("skipped", "")
    return ("green", "")


def _detail(el: ET.Element) -> str:
    """The failure/error text: prefer the element body (full traceback), fall
    back to the ``message`` attribute. The traceback is distilled
    (:func:`distill_failure`) so serge is seeded the assert diff + asserted
    tensors, not tens of KB of intermediate-tensor ``--showlocals`` noise."""
    body = (el.text or "").strip()
    msg = (el.get("message") or "").strip()
    combined = f"{msg}\n{body}" if (body and msg and msg not in body) else (body or msg)
    return distill_failure(combined)


def parse_junit(path: str | None) -> dict[tuple[str, str], dict]:
    """Map ``(class, method) -> {"outcome", "detail"}`` for a JUnit XML file.

    Returns ``{}`` for a missing/empty/None path so callers can treat "no report"
    as "nothing verified" rather than crashing."""
    if not path:
        return {}
    try:
        tree = ET.parse(path)
    except (OSError, ET.ParseError):
        return {}
    out: dict[tuple[str, str], dict] = {}
    for tc in tree.iter("testcase"):
        name = (tc.get("name") or "").strip()
        if not name:
            continue
        cls = (tc.get("classname") or "").rsplit(".", 1)[-1].strip()
        outcome, detail = _testcase_outcome(tc)
        out[(cls, name)] = {"outcome": outcome, "detail": detail}
    return out


def _as_reports(
    reports: dict[tuple[str, str], dict] | list[dict[tuple[str, str], dict]] | None,
) -> list[dict[tuple[str, str], dict]]:
    """Normalise a phase's parsed reports to a list. Each phase (baseline /
    patched) is run several times — once per JUnit XML — so callers pass a list
    of reports. A bare dict (a single run) or ``None`` is accepted for
    convenience and coerced."""
    if reports is None:
        return []
    if isinstance(reports, dict):
        return [reports]
    return list(reports)


def _one(report: dict[tuple[str, str], dict], nodeid: str) -> dict:
    """Outcome for one node-id in ONE run's report. Exact ``(class, method)``
    match, then a method-only fallback for classname/module skew — taken only
    when unambiguous (that method lives in exactly one class). ``missing`` when
    the test is absent from this run."""
    cls, method = nodeid_key(nodeid)
    if (cls, method) in report:
        return report[(cls, method)]
    grouped: dict[str, dict] = {c: v for (c, n), v in report.items() if n == method}
    if len(grouped) == 1:
        return next(iter(grouped.values()))
    return {"outcome": "missing", "detail": ""}


def _collect(
    reports: dict[tuple[str, str], dict] | list[dict[tuple[str, str], dict]] | None,
    nodeid: str,
) -> list[dict]:
    """Every run's outcome for one node-id, one entry per repeated run.

    The targeted tests are run several times (fresh process each) so a flaky
    pass/fail can't drive the verdict; this gathers all those runs so the caller
    can require stability across every one. ``[{"outcome": "missing"}]`` when the
    node-id is absent from every run."""
    runs = [_one(report, nodeid) for report in _as_reports(reports)]
    return runs or [{"outcome": "missing", "detail": ""}]


# Display precedence for collapsing the repeated runs into the single outcome
# shown per node-id in the verdict's ``targeted`` list. The authoritative
# pass/fail decision is the verdict itself; this is diagnostic, and surfaces the
# "worst" outcome across runs. Order chosen so a single run reproduces that run's
# outcome exactly.
_DISPLAY_ORDER = ("failed", "error", "missing", "skipped", "green")


def _summarize(runs: list[dict]) -> str:
    present = {r["outcome"] for r in runs}
    for outcome in _DISPLAY_ORDER:
        if outcome in present:
            return outcome
    return "green"


def _first_detail(runs: list[dict]) -> str:
    """First non-empty failure/error traceback among the repeated runs."""
    for r in runs:
        if r["outcome"] in ("failed", "error") and r["detail"]:
            return r["detail"]
    return ""


def build_verdict(
    nodeids: list[str],
    baseline: dict[tuple[str, str], dict] | list[dict[tuple[str, str], dict]],
    patched: dict[tuple[str, str], dict] | list[dict[tuple[str, str], dict]],
    collateral: dict[tuple[str, str], dict] | None = None,
    collateral_baseline: dict[tuple[str, str], dict] | None = None,
    *,
    base_sha: str = "",
    commit_sha: str = "",
    machine_type: str = "",
) -> dict:
    """Compute the verify verdict from parsed JUnit reports. Pure function —
    all I/O happens in ``main``/``parse_junit`` — so it is trivially testable.

    ``baseline`` and ``patched`` are each a list of parsed reports — one per
    repeated run of the targeted tests (a bare dict is accepted as a single
    run). The guards are evaluated across ALL runs, and the conservative outcome
    wins: a single green baseline run makes the failure untrustworthy
    (``already_passing``); a single red patched run makes the fix unstable
    (``not_fixed``). ``fixed`` therefore requires every run to flip red→green."""
    targeted: list[dict] = []
    tracebacks: dict[str, str] = {}
    any_baseline_passing = False
    any_still_red = False
    any_unverifiable = False

    for nodeid in nodeids:
        b_runs = _collect(baseline, nodeid)
        p_runs = _collect(patched, nodeid)
        b_outcomes = {r["outcome"] for r in b_runs}
        p_outcomes = {r["outcome"] for r in p_runs}
        targeted.append(
            {
                "nodeid": nodeid,
                "baseline": _summarize(b_runs),
                "patched": _summarize(p_runs),
            }
        )
        # Baseline-red guard: the test must have been failing before the patch —
        # on EVERY run. Any green baseline run = flaky/self-healed.
        if "green" in b_outcomes:
            any_baseline_passing = True
        elif b_outcomes & {"missing", "skipped"}:
            any_unverifiable = True
        # Green-patched gate: any red run means the fix is not stable.
        if p_outcomes & {"failed", "error"}:
            any_still_red = True
            detail = _first_detail(p_runs)
            if detail:
                tracebacks[nodeid] = detail
        elif p_outcomes & {"missing", "skipped"}:
            any_unverifiable = True

    # New failures the patch introduced outside the targeted set. Only sound
    # when we also ran the suite on the baseline tree; otherwise advisory (we
    # can't tell a pre-existing failure from a regression), so we leave it empty.
    collateral_new_failures: list[str] = []
    if collateral and collateral_baseline is not None:
        targeted_keys = {nodeid_key(n) for n in nodeids}
        for key, res in collateral.items():
            if key in targeted_keys or res["outcome"] not in ("failed", "error"):
                continue
            was = collateral_baseline.get(key, {}).get("outcome", "missing")
            if was == "green":
                collateral_new_failures.append("::".join(k for k in key if k))

    if any_baseline_passing:
        verdict = "already_passing"
    elif any_unverifiable:
        verdict = "error"
    elif any_still_red:
        verdict = "not_fixed"
    elif collateral_new_failures:
        verdict = "broke_others"
    else:
        verdict = "fixed"

    return {
        "mode": "verify",
        "base_sha": base_sha,
        "commit_sha": commit_sha,
        "machine_type": machine_type,
        # How many times each targeted test was run per tree (baseline/patched).
        # serge surfaces this in the PR body ("run N times to rule out flakiness").
        "runs": len(_as_reports(baseline)),
        "targeted": targeted,
        "collateral_new_failures": sorted(collateral_new_failures),
        "verdict": verdict,
        "tracebacks": tracebacks,
    }


def build_reproduce_verdict(
    nodeids: list[str],
    baseline: dict[tuple[str, str], dict] | list[dict[tuple[str, str], dict]],
    *,
    base_sha: str = "",
    machine_type: str = "",
) -> dict:
    """Compute the reproduce verdict from the baseline-only JUnit reports.

    Runs BEFORE serge investigates: confirms the group's failure is real at
    ``base_sha`` before spending any LLM budget. Verdict is ``reproduced`` only
    when every targeted test is red at baseline — the same conservatism as the
    ``verify`` baseline-red guard, so a ``reproduced`` group is exactly one the
    later ``verify`` step can adjudicate. Pure function — I/O lives in ``main``.

    ``baseline`` is a list of parsed reports (one per repeated run; a bare dict
    is accepted as a single run). A single green baseline run makes the failure
    flaky → ``not_reproduced``; ``reproduced`` requires every run to be red."""
    targeted: list[dict] = []
    tracebacks: dict[str, str] = {}
    any_passing = False
    any_unverifiable = False
    any_red = False

    for nodeid in nodeids:
        b_runs = _collect(baseline, nodeid)
        b_outcomes = {r["outcome"] for r in b_runs}
        targeted.append({"nodeid": nodeid, "baseline": _summarize(b_runs)})
        if "green" in b_outcomes:
            any_passing = True
        elif b_outcomes & {"missing", "skipped"}:
            any_unverifiable = True
        elif b_outcomes & {"failed", "error"}:
            any_red = True
            detail = _first_detail(b_runs)
            if detail:
                tracebacks[nodeid] = detail

    if any_passing:
        verdict = "not_reproduced"
    elif any_unverifiable:
        verdict = "error"
    elif any_red:
        verdict = "reproduced"
    else:
        # No node-ids, or none classifiable — nothing to reproduce.
        verdict = "not_reproduced"

    return {
        "mode": "reproduce",
        "base_sha": base_sha,
        "machine_type": machine_type,
        "runs": len(_as_reports(baseline)),
        "targeted": targeted,
        "verdict": verdict,
        "tracebacks": tracebacks,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compute the serge verify-loop verdict.")
    ap.add_argument(
        "--mode",
        choices=("verify", "reproduce"),
        default="verify",
        help="verify: baseline+patched red→green gate (default). "
        "reproduce: baseline only, confirm the failure is real before investigating.",
    )
    ap.add_argument("--nodeids", required=True, help="space-separated pytest node-ids")
    ap.add_argument(
        "--baseline",
        required=True,
        nargs="+",
        help="JUnit XML(s) from the pre-patch runs — one per repeated run. The "
        "targeted tests are run several times; pass every baseline_*.xml so a "
        "flaky pass can't slip past the baseline-red guard.",
    )
    ap.add_argument(
        "--patched",
        nargs="*",
        default=[],
        help="JUnit XML(s) from the patched runs — one per repeated run "
        "(required for mode=verify). The fix counts as fixed only if every run "
        "is green.",
    )
    ap.add_argument(
        "--collateral", default=None, help="JUnit XML: patched full-model run"
    )
    ap.add_argument(
        "--collateral-baseline",
        default=None,
        help="JUnit XML: baseline full-model run (enables broke_others)",
    )
    ap.add_argument("--base-sha", default="")
    ap.add_argument("--commit-sha", default="")
    ap.add_argument("--machine-type", default="")
    ap.add_argument(
        "--out", default="-", help="output path for the verdict JSON ('-' = stdout)"
    )
    args = ap.parse_args(argv)

    nodeids = args.nodeids.split()
    if not nodeids:
        print("no node-ids supplied", file=sys.stderr)
        return 2

    baseline_runs = [parse_junit(p) for p in args.baseline]
    patched_runs = [parse_junit(p) for p in args.patched]

    if args.mode == "reproduce":
        verdict = build_reproduce_verdict(
            nodeids,
            baseline_runs,
            base_sha=args.base_sha,
            machine_type=args.machine_type,
        )
        ok = "reproduced"
    else:
        verdict = build_verdict(
            nodeids,
            baseline_runs,
            patched_runs,
            collateral=parse_junit(args.collateral),
            collateral_baseline=parse_junit(args.collateral_baseline),
            base_sha=args.base_sha,
            commit_sha=args.commit_sha,
            machine_type=args.machine_type,
        )
        ok = "fixed"
    text = json.dumps(verdict, indent=2)
    if args.out == "-":
        print(text)
    else:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    # Non-zero exit for the non-success verdicts so the workflow step reflects it,
    # but serge decides off the JSON `verdict`, not the exit code.
    return 0 if verdict["verdict"] == ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
