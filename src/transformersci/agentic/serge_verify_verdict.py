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
targeted failing tests twice — once on the pre-patch tree (``baseline``) and
once on serge's candidate (``patched``) — writing a pytest JUnit XML per phase.
This tool reads those XMLs and emits the machine-readable verdict serge polls
for, deciding whether the patch actually turned the targeted tests red → green
without breaking neighbours.

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
    back to the ``message`` attribute."""
    body = (el.text or "").strip()
    msg = (el.get("message") or "").strip()
    if body and msg and msg not in body:
        return f"{msg}\n{body}"
    return body or msg


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


def _lookup(report: dict[tuple[str, str], dict], nodeid: str) -> dict:
    """Outcome for one node-id in a parsed report. Exact ``(class, method)``
    match, then a method-only fallback (handles module/classname skew).
    ``missing`` when the test is absent from the report."""
    key = nodeid_key(nodeid)
    if key in report:
        return report[key]
    _, method = key
    by_method = [val for (_c, m), val in report.items() if m == method]
    if len(by_method) == 1:
        return by_method[0]
    return {"outcome": "missing", "detail": ""}


def build_verdict(
    nodeids: list[str],
    baseline: dict[tuple[str, str], dict],
    patched: dict[tuple[str, str], dict],
    collateral: dict[tuple[str, str], dict] | None = None,
    collateral_baseline: dict[tuple[str, str], dict] | None = None,
    *,
    base_sha: str = "",
    commit_sha: str = "",
    machine_type: str = "",
) -> dict:
    """Compute the verify verdict from parsed JUnit reports. Pure function —
    all I/O happens in ``main``/``parse_junit`` — so it is trivially testable."""
    targeted: list[dict] = []
    tracebacks: dict[str, str] = {}
    any_baseline_passing = False
    any_still_red = False
    any_unverifiable = False

    for nodeid in nodeids:
        b = _lookup(baseline, nodeid)
        p = _lookup(patched, nodeid)
        targeted.append(
            {"nodeid": nodeid, "baseline": b["outcome"], "patched": p["outcome"]}
        )
        # Baseline-red guard: the test must have been failing before the patch.
        if b["outcome"] == "green":
            any_baseline_passing = True
        elif b["outcome"] in ("missing", "skipped"):
            any_unverifiable = True
        if p["outcome"] == "failed" or p["outcome"] == "error":
            any_still_red = True
            if p["detail"]:
                tracebacks[nodeid] = p["detail"]
        elif p["outcome"] in ("missing", "skipped"):
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
        "targeted": targeted,
        "collateral_new_failures": sorted(collateral_new_failures),
        "verdict": verdict,
        "tracebacks": tracebacks,
    }


def build_reproduce_verdict(
    nodeids: list[str],
    baseline: dict[tuple[str, str], dict],
    *,
    base_sha: str = "",
    machine_type: str = "",
) -> dict:
    """Compute the reproduce verdict from the baseline-only JUnit report.

    Runs BEFORE serge investigates: confirms the group's failure is real at
    ``base_sha`` before spending any LLM budget. Verdict is ``reproduced`` only
    when every targeted test is red at baseline — the same conservatism as the
    ``verify`` baseline-red guard, so a ``reproduced`` group is exactly one the
    later ``verify`` step can adjudicate. Pure function — I/O lives in ``main``."""
    targeted: list[dict] = []
    tracebacks: dict[str, str] = {}
    any_passing = False
    any_unverifiable = False
    any_red = False

    for nodeid in nodeids:
        b = _lookup(baseline, nodeid)
        targeted.append({"nodeid": nodeid, "baseline": b["outcome"]})
        if b["outcome"] == "green":
            any_passing = True
        elif b["outcome"] in ("missing", "skipped"):
            any_unverifiable = True
        elif b["outcome"] in ("failed", "error"):
            any_red = True
            if b["detail"]:
                tracebacks[nodeid] = b["detail"]

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
        "--baseline", required=True, help="JUnit XML from the pre-patch run"
    )
    ap.add_argument(
        "--patched",
        default=None,
        help="JUnit XML from the patched run (required for mode=verify)",
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

    if args.mode == "reproduce":
        verdict = build_reproduce_verdict(
            nodeids,
            parse_junit(args.baseline),
            base_sha=args.base_sha,
            machine_type=args.machine_type,
        )
        ok = "reproduced"
    else:
        verdict = build_verdict(
            nodeids,
            parse_junit(args.baseline),
            parse_junit(args.patched),
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
