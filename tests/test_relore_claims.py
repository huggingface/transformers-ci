"""The dispatch-side "a human already has this" check.

Two things carry the weight here and the rest is plumbing:

* **What counts as a claim.** Only an OPEN pull request by a human. Merged is
  new information and must still be dispatched; closed-unmerged is a human
  having declined, which is the other defer reason's job; our own bots are not
  humans. Getting any of these wrong produces a *false defer* — a real failure
  nobody works on, and nothing downstream says so.
* **Failing soft.** Daemon down, version skew, timeout, garbage output: all mean
  dispatch as today. A retrieval outage may cost a wasted dispatch; it may never
  shrink the night's work.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from transformersci.agentic import relore_claims as rc


def _target(*tests: str) -> dict:
    return {
        "kind": "model_failures",
        "label": "2 integration tests for model `emu3`",
        "failures": [{"test": t, "gpu": "a10"} for t in tests],
    }


def _claim(**over) -> dict:
    base = {
        "type": "pr",
        "number": 48652,
        "title": "Fix RoPE ignoring partial_rotary_factor",
        "url": "https://github.com/huggingface/transformers/pull/48652",
        "author": "blipbyte",
        "state": "open",
        "draft": False,
        "merged": False,
    }
    base.update(over)
    return base


def _wire(monkeypatch, *, hits=None, claims=None, record=None):
    """Stand in for the two relore calls; record the argv of each."""

    def run(argv, **kwargs):
        if record is not None:
            record.setdefault("calls", []).append(argv)
        verb = argv[2] if len(argv) > 2 else ""
        if verb == "search":
            payload = {"hits": hits if hits is not None else []}
        elif verb == "inflight":
            payload = {"claims": claims if claims is not None else []}
        else:
            payload = {}
        return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")

    monkeypatch.setattr(rc.subprocess, "run", run)


# -- what counts as a claim ------------------------------------------------


def test_an_open_human_pr_is_a_claim(monkeypatch):
    _wire(monkeypatch, hits=[{"number": 48630}], claims=[_claim()])
    reason = rc.superseded_by_human_reason(_target("tests/a.py::T::t"), repo="o/r")
    assert "#48652" in reason and "@blipbyte" in reason
    assert "found via #48630" in reason


def test_a_merged_pr_is_not_a_claim(monkeypatch):
    # A merged fix that did not stop the failure is genuinely new information.
    _wire(
        monkeypatch,
        hits=[{"number": 1}],
        claims=[_claim(state="closed", merged=True)],
    )
    assert rc.superseded_by_human_reason(_target("tests/a.py::T::t"), repo="o/r") == ""


def test_a_closed_unmerged_pr_is_not_a_claim(monkeypatch):
    # That is a human having declined — rejected_attempts_reason's job, and it
    # is keyed on our fingerprint rather than on "mentions the same test".
    _wire(monkeypatch, hits=[{"number": 1}], claims=[_claim(state="closed")])
    assert rc.superseded_by_human_reason(_target("tests/a.py::T::t"), repo="o/r") == ""


@pytest.mark.parametrize(
    "author", ["sergereview[bot]", "github-actions[bot]", "HuggingFaceDocBuilderDev"]
)
def test_our_own_bots_are_not_humans(monkeypatch, author):
    _wire(monkeypatch, hits=[{"number": 1}], claims=[_claim(author=author)])
    assert rc.superseded_by_human_reason(_target("tests/a.py::T::t"), repo="o/r") == ""


def test_extra_bot_accounts_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("ITF_RELORE_BOT_ACCOUNTS", "somenewbot , other")
    _wire(monkeypatch, hits=[{"number": 1}], claims=[_claim(author="SomeNewBot")])
    assert rc.superseded_by_human_reason(_target("tests/a.py::T::t"), repo="o/r") == ""


def test_a_draft_counts_but_is_labelled(monkeypatch):
    # Someone mid-fix is still someone to not duplicate; the label lets the
    # human reading the tracking issue judge it.
    _wire(monkeypatch, hits=[{"number": 1}], claims=[_claim(draft=True)])
    reason = rc.superseded_by_human_reason(_target("tests/a.py::T::t"), repo="o/r")
    assert "(draft)" in reason


def test_an_issue_is_not_a_claim(monkeypatch):
    _wire(monkeypatch, hits=[{"number": 1}], claims=[_claim(type="issue")])
    assert rc.superseded_by_human_reason(_target("tests/a.py::T::t"), repo="o/r") == ""


# -- the two hops ----------------------------------------------------------


def test_tests_become_repeated_test_filters(monkeypatch):
    record: dict = {}
    _wire(monkeypatch, hits=[], record=record)
    rc.superseded_by_human_reason(
        _target("tests/a.py::T::t1", "tests/b.py::T::t2"),
        repo="huggingface/transformers",
    )
    search = record["calls"][0]
    assert search[:3] == ["relore", "--json", "search"]
    assert search.count("--test") == 2
    assert "huggingface/transformers" in search


def test_only_the_top_candidates_are_asked_about(monkeypatch):
    record: dict = {}
    _wire(
        monkeypatch,
        hits=[{"number": n} for n in (10, 11, 12, 13, 14)],
        claims=[],
        record=record,
    )
    rc.find_open_claims(["tests/a.py::T::t"], repo="o/r", max_candidates=2)
    inflights = [c for c in record["calls"] if c[2] == "inflight"]
    assert len(inflights) == 2


def test_duplicate_hits_on_one_thread_are_asked_once(monkeypatch):
    # Several hits routinely come from different comments on the same PR.
    record: dict = {}
    _wire(
        monkeypatch, hits=[{"number": 7}, {"number": 7}, {"number": 7}], record=record
    )
    rc.find_open_claims(["tests/a.py::T::t"], repo="o/r")
    assert len([c for c in record["calls"] if c[2] == "inflight"]) == 1


def test_group_tests_dedupes_and_caps(monkeypatch):
    t = _target("a", "a", "b", "c", "d", "e")
    assert rc.group_tests(t, limit=3) == ["a", "b", "c"]


def test_a_group_with_no_tests_asks_nothing(monkeypatch):
    record: dict = {}
    _wire(monkeypatch, hits=[{"number": 1}], record=record)
    assert rc.superseded_by_human_reason({"failures": []}, repo="o/r") == ""
    assert not record.get("calls")


# -- failing soft ----------------------------------------------------------


def test_a_missing_client_dispatches(monkeypatch):
    def run(argv, **kwargs):
        raise FileNotFoundError("relore")

    monkeypatch.setattr(rc.subprocess, "run", run)
    assert rc.superseded_by_human_reason(_target("tests/a.py::T::t"), repo="o/r") == ""


def test_a_timeout_dispatches(monkeypatch):
    def run(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 20)

    monkeypatch.setattr(rc.subprocess, "run", run)
    assert rc.superseded_by_human_reason(_target("tests/a.py::T::t"), repo="o/r") == ""


def test_a_nonzero_exit_dispatches(monkeypatch):
    # This is what a version skew looks like: relore exits 1 with the 426.
    monkeypatch.setattr(
        rc.subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 1, "", "426"),
    )
    assert rc.superseded_by_human_reason(_target("tests/a.py::T::t"), repo="o/r") == ""


def test_unparseable_output_dispatches(monkeypatch):
    monkeypatch.setattr(
        rc.subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 0, "not json", ""),
    )
    assert rc.superseded_by_human_reason(_target("tests/a.py::T::t"), repo="o/r") == ""


def test_an_empty_search_asks_no_inflight(monkeypatch):
    record: dict = {}
    _wire(monkeypatch, hits=[], record=record)
    assert rc.superseded_by_human_reason(_target("tests/a.py::T::t"), repo="o/r") == ""
    assert not [c for c in record["calls"] if c[2] == "inflight"]


# -- the flag, and what each state does to dispatch ------------------------


from transformersci.agentic import integration_failure_triage as itf  # noqa: E402


def _dispatchable() -> dict:
    # A group no cheap local reason defers: not OOM, not a dependency exception.
    return {
        "kind": "model_failures",
        "label": "1 integration test for model `emu3` failing with `output_mismatch`",
        "failure_mode": "output_mismatch",
        "terminal_exc": "AssertionError",
        "model": "emu3",
        "failures": [
            {
                "test": "tests/a.py::T::t",
                "gpu": "a10",
                "model": "emu3",
                "days_seen": 3,
                "failure_mode": "output_mismatch",
            }
        ],
    }


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("", itf.SUPERSEDED_OFF),
        ("off", itf.SUPERSEDED_OFF),
        ("report", itf.SUPERSEDED_REPORT),
        ("defer", itf.SUPERSEDED_DEFER),
        ("nonsense", itf.SUPERSEDED_OFF),
        # A boolean spelling is what an operator reaches for first. Resolve it to
        # the cautious end rather than silently doing nothing.
        ("1", itf.SUPERSEDED_REPORT),
        ("true", itf.SUPERSEDED_REPORT),
    ],
)
def test_the_mode_is_three_state_and_off_by_default(monkeypatch, raw, expected):
    monkeypatch.setenv("ITF_RELORE_SUPERSEDED", raw)
    assert itf.superseded_mode() == expected


def test_off_never_calls_relore(monkeypatch):
    monkeypatch.delenv("ITF_RELORE_SUPERSEDED", raising=False)
    called = {"n": 0}
    monkeypatch.setattr(
        itf,
        "superseded_by_human_reason",
        lambda *a, **kw: called.__setitem__("n", called["n"] + 1) or "claimed",
    )
    dispatch, deferred = itf.partition_targets([_dispatchable()], repo="o/r")
    assert called["n"] == 0
    assert len(dispatch) == 1 and not deferred
    assert "superseded_note" not in dispatch[0]


def test_report_dispatches_but_records_what_it_would_have_done(monkeypatch):
    monkeypatch.setenv("ITF_RELORE_SUPERSEDED", "report")
    monkeypatch.setattr(
        itf, "superseded_by_human_reason", lambda *a, **kw: "claimed by #9"
    )
    dispatch, deferred = itf.partition_targets([_dispatchable()], repo="o/r")
    assert not deferred
    assert dispatch[0]["superseded_note"] == "claimed by #9"


def test_defer_moves_the_group_out_of_dispatch(monkeypatch):
    monkeypatch.setenv("ITF_RELORE_SUPERSEDED", "defer")
    monkeypatch.setattr(
        itf, "superseded_by_human_reason", lambda *a, **kw: "claimed by #9"
    )
    dispatch, deferred = itf.partition_targets([_dispatchable()], repo="o/r")
    assert not dispatch
    assert deferred[0]["defer_reason"] == "claimed by #9"


def test_a_cheaper_local_reason_wins_and_skips_the_network_call(monkeypatch):
    # Asked last, and only for a group that would otherwise be dispatched: it is
    # the only reason that costs a call, and it keeps the measurement clean —
    # a group counted here is one no other reason caught.
    monkeypatch.setenv("ITF_RELORE_SUPERSEDED", "defer")
    called = {"n": 0}
    monkeypatch.setattr(
        itf,
        "superseded_by_human_reason",
        lambda *a, **kw: called.__setitem__("n", called["n"] + 1) or "claimed",
    )
    oom = {**_dispatchable(), "failure_mode": "OOM", "terminal_exc": "OutOfMemoryError"}
    dispatch, deferred = itf.partition_targets([oom], repo="o/r")
    assert called["n"] == 0
    assert deferred and "device memory" in deferred[0]["defer_reason"]


def test_report_only_notes_reach_the_tracking_issue():
    target = {**_dispatchable(), "superseded_note": "claimed by #9 by @someone"}
    body = itf.render_tracking_issue_body([target], ["2026-09-16"], "run-key")
    assert "report-only" in body
    assert "claimed by #9 by @someone" in body
    # No `| PR |` header in this section: _carry_forward_rows must not adopt a
    # report-only line as a dispatched row on a later same-day run.
    section = body[body.index("report-only") :]
    assert "| PR |" not in section


def test_the_tracking_issue_is_unchanged_without_notes():
    body = itf.render_tracking_issue_body([_dispatchable()], ["2026-09-16"], "run-key")
    assert "report-only" not in body
