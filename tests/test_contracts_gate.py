"""gate-model-contracts: which red contracts block a PR."""

import json
import subprocess
import sys
import textwrap

from transformersci.contracts import gate


def rec(target, status="pass", failures=(), fleet="prototypes", variant="bfloat16"):
    return {
        "contract": f"huggingface-mic/{target}",
        "contract_name": target,
        "target": target,
        "fleet": fleet,
        "variant": variant,
        "fleet_revision": "f" * 40,
        "status": status,
        "failures": list(failures),
    }


def verdicts(decisions):
    return {d["target"]: (d["verdict"], d["reason"].split(":")[0]) for d in decisions}


def test_decide_blocks_only_cases_green_on_the_base():
    records = [
        rec("broke", "fail", ["logits: max delta 0.1 over tolerance"]),
        rec("main-red", "error", ["Traceback\nAttributeError: x"]),
        rec("xpass", "fail", ["listed in known_failures but passes; remove it: #1"]),
        rec("setup", "error", ["no reviewed baseline for cpu-x86_64-bf16"]),
        rec("green"),
    ]
    base = [rec("broke"), rec("main-red", "error", ["Traceback\nAttributeError: x"])]
    cases = gate.to_replay(records)
    assert [r["target"] for r in cases] == ["broke", "main-red"]
    got = verdicts(gate.decide(records, base, [], cases))
    assert got == {
        "broke": ("blocked", "green on the base (pass), red with this change"),
        "main-red": ("warn", "red on the base too"),
        "xpass": ("warn", "known failure now passes"),
        "setup": ("warn", "setup error"),
    }


def test_intended_label_and_unmeasured_cases_warn():
    records = [rec("broke", "fail", ["x"]), rec("lost", "fail", ["y"])]
    cases = gate.to_replay(records)
    base = [rec("broke")]  # the replay did not report "lost"
    got = verdicts(gate.decide(records, base, [gate.LABEL], cases))
    assert got["broke"][0] == "warn" and gate.LABEL in got["broke"][1]
    assert got["lost"] == ("warn", "the base replay did not report this case")


def test_replay_limit(monkeypatch):
    monkeypatch.setattr(gate, "REPLAY_LIMIT", 2)
    records = [rec(f"t{i}", "fail", ["x"]) for i in range(3)]
    cases = gate.to_replay(records)
    got = verdicts(gate.decide(records, [rec("t0"), rec("t1")], [], cases))
    assert got["t2"][0] == "warn" and got["t0"][0] == got["t1"][0] == "blocked"


def test_same_target_in_two_fleets_is_two_cases():
    records = [
        rec("x", "fail", ["d"], fleet="control"),
        rec("x", "pass", fleet="prototypes"),
    ]
    base = [rec("x", fleet="control"), rec("x", "fail", ["d"], fleet="prototypes")]
    decisions = gate.decide(records, base, [], gate.to_replay(records))
    assert [(d["fleet"], d["verdict"]) for d in decisions] == [("control", "blocked")]


def test_replay_env_drops_tracing(monkeypatch, tmp_path):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector")
    monkeypatch.setenv("PYTHONPATH", "/pr/src")
    env = gate.replay_env(tmp_path / "src")
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in env
    assert env["PYTHONPATH"].split(":")[0] == str(tmp_path / "src")


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_main_replays_red_cases_on_the_base_tree(tmp_path):
    """End to end: a two-commit repo whose HEAD breaks `broke`; the fake runner reports what the
    tree on PYTHONPATH says, so the replay must really run on HEAD^1."""
    repo = tmp_path / "transformers"
    (repo / "src/transformers").mkdir(parents=True)
    init = repo / "src/transformers/__init__.py"
    init.write_text("BROKEN = False\n")
    git(repo, "init", "-q")
    git(
        repo,
        "-c",
        "user.email=t@t",
        "-c",
        "user.name=t",
        "commit",
        "-qam",
        "base",
        "--allow-empty",
    )
    git(repo, "add", ".")
    git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base")
    init.write_text("BROKEN = True\n")
    git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam", "pr")

    runner = tmp_path / "runner.py"
    runner.write_text(
        textwrap.dedent(
            """
            import argparse, json, transformers
            p = argparse.ArgumentParser()
            p.add_argument("--fleet"); p.add_argument("--variant"); p.add_argument("--output")
            p.add_argument("--fleet-revision"); p.add_argument("--target", action="append")
            p.add_argument("--code-revision")
            p.add_argument("--fixtures-only", action="store_true")
            a = p.parse_args()
            with open(a.output, "a") as f:
                for t in a.target:
                    status = "fail" if t == "always" or (t == "broke" and transformers.BROKEN) else "pass"
                    f.write(json.dumps({"contract": "huggingface-mic/" + t, "target": t, "fleet": a.fleet,
                                        "variant": a.variant, "status": status, "failures": []}) + "\\n")
            """
        )
    )
    results = tmp_path / "results.jsonl"
    results.write_text(
        "".join(
            json.dumps(r) + "\n"
            for r in [
                rec("broke", "fail", ["d"]),
                rec("always", "fail", ["d"]),
                rec("ok"),
            ]
        )
    )
    out, summary = tmp_path / "gate.json", tmp_path / "summary.md"
    code = gate.main(
        [
            str(results),
            "--repo",
            str(repo),
            "--runner",
            f"{sys.executable} {runner}",
            "--output",
            str(out),
            "--summary",
            str(summary),
            "--enforce",
            "--log",
            str(tmp_path / "log"),
        ]
    )
    decisions = {d["target"]: d for d in json.loads(out.read_text())["decisions"]}
    assert code == 1
    assert (
        decisions["broke"]["verdict"] == "blocked"
        and decisions["broke"]["base_status"] == "pass"
    )
    assert (
        decisions["always"]["verdict"] == "warn"
        and decisions["always"]["base_status"] == "fail"
    )
    assert "this PR is blocked" in summary.read_text()
    # the base worktree is cleaned up
    listed = subprocess.run(
        ["git", "-C", str(repo), "worktree", "list"], capture_output=True, text=True
    ).stdout
    assert len(listed.strip().splitlines()) == 1

    # not enforced: same decisions, exit 0
    assert (
        gate.main(
            [
                str(results),
                "--repo",
                str(repo),
                "--runner",
                f"{sys.executable} {runner}",
                "--log",
                str(tmp_path / "log"),
            ]
        )
        == 0
    )


def test_gate_json_reports_the_replay_cost(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "replay", lambda cases, *a, **k: [rec("x")])
    results = tmp_path / "results.jsonl"
    results.write_text(json.dumps(rec("x", "fail", ["d"])) + "\n")
    out = tmp_path / "gate.json"
    assert gate.main([str(results), "--output", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["replayed"] == 1 and isinstance(report["replay_seconds"], float)
    assert report["blocked"] == 1  # green on the base: would block, not enforced


def test_replay_applies_the_prs_overlay_on_the_base(tmp_path, monkeypatch):
    """With --fleet-pr the base is checked against the PR's re-recorded values too."""
    commands = []

    def fake_run(command, env=None, **kwargs):
        commands.append(command)
        if "rev-parse" in command:
            return subprocess.CompletedProcess(
                command, 0, stdout="b" * 40 + "\n", stderr=""
            )
        # `import transformers` resolves under the base tree the replay put first on PYTHONPATH
        where = (
            env["PYTHONPATH"].split(":")[0] + "/transformers/__init__.py" if env else ""
        )
        return subprocess.CompletedProcess(command, 0, stdout=where, stderr="")

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    gate.replay(
        [rec("x", "fail", ["d"])],
        "HEAD^1",
        ".",
        ["runner"],
        False,
        sys.stderr,
        fleet_pr=42,
    )
    runner = next(c for c in commands if c[0] == "runner")
    assert runner[runner.index("--fleet-pr") + 1] == "42"
    assert (
        runner[runner.index("--code-revision") + 1] == "b" * 40
    )  # the base meets its own values


def test_run_model_contracts_forwards_fleet_pr(monkeypatch):
    from types import ModuleType

    from transformersci.contracts import run

    fake = ModuleType("transformers_mic.fleet")
    fake.REPOSITORY = "huggingface-mic/transformers-fleet"
    fake.resolve = lambda revision: revision
    fake.read_pin = lambda path: "main"
    package = ModuleType("transformers_mic")
    package.fleet = fake
    monkeypatch.setitem(sys.modules, "transformers_mic", package)
    monkeypatch.setitem(sys.modules, "transformers_mic.fleet", fake)

    args = run.parser().parse_args(
        ["--fleet", "prototypes", "--fleet-revision", "f" * 40, "--fleet-pr", "7"]
    )
    options, metadata = run.resolve_fleet(args)
    assert options[-2:] == ["--fleet-pr", "7"] and metadata["fleet_pr"] == 7
    args = run.parser().parse_args(
        [
            "--fleet",
            "control",
            "--fleet-revision",
            "f" * 40,
            "--code-revision",
            "c" * 40,
        ]
    )
    options, metadata = run.resolve_fleet(args)
    assert (
        options[-2:] == ["--code-revision", "c" * 40]
        and metadata["code_revision"] == "c" * 40
    )
    args = run.parser().parse_args(
        ["--fleet", "prototypes", "--fleet-revision", "f" * 40]
    )
    assert "--fleet-pr" not in run.resolve_fleet(args)[0]
