"""record-mic-reconcile: settle the overlays of PRs that are no longer open."""

import json
import subprocess

from transformersci.contracts import record_mic_reconcile as rec


def run_with(monkeypatch, tmp_path, states, outputs, execute=True, merged_at=None):
    monkeypatch.setattr(rec, "overlays", lambda: sorted(states))
    monkeypatch.setattr(
        rec,
        "pr_state",
        lambda pr: (
            states[pr],
            f"sha{pr}" if states[pr] == "merged" else None,
            (merged_at or {}).get(pr, f"2026-11-{pr:02d}T00:00:00Z")
            if states[pr] == "merged"
            else None,
        ),
    )
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        pr = int(args[args.index("--pr") + 1])
        code, out = outputs[pr]
        return subprocess.CompletedProcess(
            args, code, stdout=out, stderr="boom" if code else ""
        )

    monkeypatch.setattr(rec.subprocess, "run", fake_run)
    summary = tmp_path / "summary.md"
    argv = ["--summary", str(summary)] + (["--execute"] if execute else [])
    return rec.main(argv), calls, summary.read_text()


def test_merged_closed_and_open(monkeypatch, tmp_path):
    merged = json.dumps(
        {
            "pr": 1,
            "control": {"applied": ["gpt-j:repo"], "stale": []},
            "prototypes": {"applied": [], "stale": []},
        }
    )
    code, calls, text = run_with(
        monkeypatch,
        tmp_path,
        {1: "merged", 2: "closed", 3: "open"},
        {1: (0, merged), 2: (0, "{}")},
    )
    assert code == 0
    assert [c[3:5] for c in calls] == [["merge", "--pr"], ["discard", "--pr"]]
    assert calls[0][-3:] == [
        "--execute",
        "--since",
        "sha1",
    ]  # history: the PR's commit on main
    assert calls[1][-1] == "--execute" and "--since" not in calls[1]
    assert "#1: merged, merged into the fleet" in text
    assert "#2: closed, dropped (PR closed)" in text
    assert "#3: open, kept" in text


def test_stale_entries_are_reported(monkeypatch, tmp_path, capsys):
    out = json.dumps(
        {
            "control": {"applied": [], "stale": ["gpt-j:repo"]},
            "prototypes": {"applied": [], "stale": []},
        }
    )
    code, _, text = run_with(monkeypatch, tmp_path, {1: "merged"}, {1: (1, out)})
    assert code == 0  # stale is a report, not a failure of the reconciler
    assert "stale, not applied: gpt-j:repo" in text
    assert "::warning::#1" in capsys.readouterr().out


def test_failures_fail_the_run(monkeypatch, tmp_path):
    code, _, text = run_with(monkeypatch, tmp_path, {1: "closed"}, {1: (1, "")})
    assert code == 1 and "FAILED" in text


def test_dry_run_does_not_execute(monkeypatch, tmp_path):
    code, calls, text = run_with(
        monkeypatch, tmp_path, {1: "closed"}, {1: (0, "{}")}, execute=False
    )
    assert code == 0 and "--execute" not in calls[0] and "would be dropped" in text


def test_merged_prs_settle_in_main_order_not_by_number(monkeypatch, tmp_path):
    # #5 merged before #4: history rows must follow main.
    ok = json.dumps({"control": {"applied": [], "stale": []}})
    code, calls, _ = run_with(
        monkeypatch,
        tmp_path,
        {4: "merged", 5: "merged", 6: "closed"},
        {4: (0, ok), 5: (0, ok), 6: (0, "{}")},
        merged_at={4: "2026-11-24T10:00:00Z", 5: "2026-11-24T09:00:00Z"},
    )
    assert code == 0
    assert [c[c.index("--pr") + 1] for c in calls] == ["5", "4", "6"]
