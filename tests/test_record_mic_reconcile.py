"""record-mic-reconcile: settle the overlays of PRs that are no longer open."""

import json
import subprocess

from transformersci.contracts import record_mic_reconcile as rec


def run_with(monkeypatch, tmp_path, states, outputs, execute=True):
    monkeypatch.setattr(rec, "overlays", lambda: sorted(states))
    monkeypatch.setattr(rec, "pr_state", lambda pr: states[pr])
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
    assert all(c[-1] == "--execute" for c in calls)
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
