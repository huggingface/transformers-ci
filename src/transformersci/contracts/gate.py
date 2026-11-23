"""Decide whether a PR's red contracts block it.

    gate-model-contracts results.jsonl --base HEAD^1 --labels '["mic-intended-change"]' \
        --summary "$GITHUB_STEP_SUMMARY" --output gate.json [--enforce]

A red case (fail or error) blocks only when the PR caused it. Everything else warns:

- a known failure that now passes ("listed in known_failures but passes"): fleet hygiene, the
  lock entry should go, not the PR;
- a setup error (missing baseline, wrong hardware, fixture drift): the fleet, not the code;
- red on the PR's base too: main is already red (an earlier merge, an image or fleet change), or
  this is a re-run of an old merge commit against newer baselines. The base replay runs the same
  cases, fleet revision and environment on HEAD^1, the commit the merge was built on, so both
  cases compare like with like;
- the `mic-intended-change` label: a maintainer reviewed the change as intended but could not
  re-record it (a fallback for when record-mic is unavailable).

The usual path for an intended change is record-mic: someone with write access comments
`record-mic: <model>` on the PR (or runs `make record-mic`), which re-records the expected
values on the PR's code for this PR only (a fleet overlay, applied with --fleet-pr). The case
then passes; the base replay applies the same overlay, so the base is checked against the PR's
values and its red warns.

A case the replay could not measure (the replay failed, or more red cases than REPLAY_LIMIT)
warns: infrastructure must not block a PR. Without --enforce the exit status is always 0 and
the summary says what would have been blocked.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

LABEL = "mic-intended-change"
KNOWN_PASS = "listed in known_failures but passes"
# The runner's setup failures: the fleet or its baselines, not the code (mirrors
# transformers-mic experiments/impact.py SETUP_ERRORS).
SETUP_ERRORS = (
    "no reviewed baseline",
    "no golden baseline",
    "fixture files differ",
    "wrong hardware",
    "not measured",
    "is required for a normal functioning of this module",
)
REPLAY_LIMIT = (
    40  # red cases replayed on the base at most; the rest warn as not measured
)
RED = ("fail", "error")
RERECORD = "https://github.com/huggingface/transformers-mic/blob/main/docs/improving-contracts.md"


def key(record):
    return (
        record.get("fleet") or "",
        record.get("variant") or "default",
        record.get("contract") or "",
        record["target"],
    )


def read(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def known_pass(record):
    failures = record.get("failures") or []
    return bool(failures) and all(f.startswith(KNOWN_PASS) for f in failures)


def setup_error(record):
    text = "\n".join(record.get("failures") or []).lower()
    return any(s in text for s in SETUP_ERRORS)


def first_line(record):
    failures = record.get("failures") or []
    if not failures:
        return ""
    lines = [line.strip() for line in failures[0].strip().splitlines() if line.strip()]
    return (lines[-1] if len(lines) > 1 else lines[0])[:200]


def to_replay(records):
    """The red cases whose base status decides them, at most REPLAY_LIMIT."""
    reds = [
        r
        for r in records
        if r["status"] in RED and not known_pass(r) and not setup_error(r)
    ]
    return sorted(reds, key=key)[:REPLAY_LIMIT]


def decide(records, base, labels, replayed):
    """One decision per red case: {"case", "verdict": block|warn, "reason", ...}."""
    base = {key(r): r for r in base}
    replayed = {key(r) for r in replayed}
    intended = LABEL in labels
    decisions = []
    for record in sorted((r for r in records if r["status"] in RED), key=key):
        k = key(record)
        if known_pass(record):
            verdict, reason = (
                "warn",
                "known failure now passes: remove its known_failures entry",
            )
        elif setup_error(record):
            verdict, reason = (
                "warn",
                "setup error: the fleet or its baselines, not this change",
            )
        elif k not in replayed:
            verdict, reason = (
                "warn",
                f"not replayed on the base (more than {REPLAY_LIMIT} red cases)",
            )
        elif k not in base:
            verdict, reason = "warn", "the base replay did not report this case"
        elif base[k]["status"] in RED:
            verdict, reason = (
                "warn",
                "red on the base too: main is already red, or the run tests an old merge commit",
            )
        elif intended:
            verdict, reason = (
                "warn",
                f"`{LABEL}`: reviewed as intended; re-record the expected values",
            )
        else:
            verdict, reason = (
                "block",
                f"green on the base ({base[k]['status']}), red with this change",
            )
        decisions.append(
            {
                "fleet": k[0],
                "variant": k[1],
                "contract": record.get("contract_name") or k[2],
                "target": k[3],
                "status": record["status"],
                "base_status": base[k]["status"] if k in base else None,
                "verdict": verdict,
                "reason": reason,
                "failure": first_line(record),
            }
        )
    return decisions


def replay_env(base_src):
    """The runner's environment on the base tree, with tracing off: the base is not this PR."""
    env = {
        k: v for k, v in os.environ.items() if not k.startswith("OTEL_EXPORTER_OTLP")
    }
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(base_src), env.get("PYTHONPATH", "")) if p
    )
    return env


def replay(cases, base, repo, runner, run_declared, log, fleet_pr=None):
    """Run `cases` on the `base` revision of `repo`; the records it reports (empty on failure)."""
    if not cases:
        return []
    tree = Path(tempfile.mkdtemp(prefix="mic-base-"))
    out = tree.parent / f"{tree.name}.jsonl"
    try:
        subprocess.run(
            ["git", "-C", repo, "worktree", "add", "--detach", str(tree), base],
            check=True,
            stdout=log,
            stderr=log,
        )
        env = replay_env(tree / "src")
        where = subprocess.run(
            [sys.executable, "-c", "import transformers; print(transformers.__file__)"],
            env=env,
            capture_output=True,
            text=True,
        )
        if not where.stdout.strip().startswith(
            str(tree.resolve())
        ) and not where.stdout.strip().startswith(str(tree)):
            print(
                f"base replay: transformers resolves to {where.stdout.strip() or where.stderr.strip()!r}, not {tree}",
                file=log,
            )
            return []
        groups = defaultdict(lambda: {"targets": set(), "revision": None})
        for r in cases:
            group = groups[(r.get("fleet") or "control", r.get("variant") or "default")]
            group["targets"].add(r["target"])
            group["revision"] = group["revision"] or r.get("fleet_revision")
        for (fleet, variant), group in sorted(groups.items()):
            command = [
                *runner,
                "--fixtures-only",
                "--fleet",
                fleet,
                "--variant",
                variant,
                "--output",
                str(out),
            ]
            if group["revision"]:
                command += ["--fleet-revision", group["revision"]]
            if run_declared:
                command.append("--run-declared")
            if fleet_pr is not None:
                command += ["--fleet-pr", str(fleet_pr)]
            for target in sorted(group["targets"]):
                command += ["--target", target]
            print("base replay:", " ".join(command), file=log, flush=True)
            subprocess.run(command, env=env, stdout=subprocess.DEVNULL, stderr=log)
        return read(out)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"base replay failed: {error}", file=log)
        return []
    finally:
        subprocess.run(
            ["git", "-C", repo, "worktree", "remove", "--force", str(tree)],
            stdout=log,
            stderr=log,
        )
        shutil.rmtree(tree, ignore_errors=True)
        out.unlink(missing_ok=True)


def summary(decisions, enforce, base, labels):
    blocks = [d for d in decisions if d["verdict"] == "block"]
    warns = [d for d in decisions if d["verdict"] == "warn"]
    lines = ["### Contracts gate", ""]
    if not decisions:
        return "\n".join(lines + ["✅ No red contract.", ""])
    if blocks:
        lines.append(
            f"❌ **{len(blocks)} red case(s) caused by this change** — "
            + (
                "this PR is blocked."
                if enforce
                else "would block this PR (the gate is not enforced yet)."
            )
        )
    else:
        lines.append("⚠️ Red contracts, none caused by this change: not blocking.")
    lines += [
        "",
        f"Base: `{base}`. Labels: {', '.join(f'`{label}`' for label in labels) or 'none'}.",
        "",
    ]
    lines += [
        "| verdict | contract | target | variant | PR | base | why |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in blocks + warns:
        icon = "❌ block" if d["verdict"] == "block" else "⚠️ warn"
        lines.append(
            f"| {icon} | {d['contract']} | {d['target']} | {d['variant']} | {d['status']}: "
            f"{d['failure'].replace('|', chr(92) + '|')} | {d['base_status'] or '—'} | {d['reason']} |"
        )
    if blocks:
        lines += [
            "",
            "A blocked case is green on the base and red here. If it is a bug, fix it in this PR. If the change "
            "is intended, someone with write access comments `record-mic: <model>` on this PR: the expected "
            "values are re-recorded on this PR's code, posted here for review, and apply to this PR only until "
            f"it merges ([details]({RERECORD})).",
        ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--base",
        default="HEAD^1",
        help="revision to replay red cases on (default: HEAD^1)",
    )
    parser.add_argument("--repo", default=".", help="the Transformers checkout")
    parser.add_argument(
        "--labels", default="[]", help="the PR's labels, as a JSON list"
    )
    parser.add_argument(
        "--runner",
        default="run-model-contracts",
        help="the runner command (split on spaces)",
    )
    parser.add_argument(
        "--run-declared", action="store_true", help="the run included declared variants"
    )
    parser.add_argument(
        "--fleet-pr",
        type=int,
        help="the PR whose record-mic overlay the run applied; the replay applies it too",
    )
    parser.add_argument(
        "--summary", type=Path, help="append the markdown summary to this file"
    )
    parser.add_argument("--output", type=Path, help="write the decisions as JSON")
    parser.add_argument(
        "--log", type=Path, help="the replay's stderr (default: stderr)"
    )
    parser.add_argument(
        "--enforce", action="store_true", help="exit 1 when a case blocks"
    )
    args = parser.parse_args(argv)

    labels = json.loads(args.labels or "[]")
    records = read(args.results)
    cases = to_replay(records)
    log = args.log.open("a") if args.log else sys.stderr
    started = time.monotonic()
    try:
        base = replay(
            cases,
            args.base,
            args.repo,
            args.runner.split(),
            args.run_declared,
            log,
            args.fleet_pr,
        )
    finally:
        if args.log:
            log.close()
    replay_seconds = round(time.monotonic() - started, 1)
    decisions = decide(records, base, labels, cases)
    text = summary(decisions, args.enforce, args.base, labels)
    if cases:
        text += f"\nBase replay: {len(cases)} red case(s) in {replay_seconds} s.\n"
    print(text)
    if args.summary:
        with args.summary.open("a") as f:
            f.write(text)
    blocks = [d for d in decisions if d["verdict"] == "block"]
    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "enforced": args.enforce,
                    "blocks": len(blocks),
                    "replayed": len(cases),
                    "replay_seconds": replay_seconds,
                    "decisions": decisions,
                },
                indent=2,
            )
            + "\n"
        )
    for d in decisions:
        level = "error" if d["verdict"] == "block" and args.enforce else "warning"
        print(
            f"::{level}::contract {d['contract']}/{d['target']} [{d['variant']}] {d['verdict']}: {d['reason']}"
        )
    return 1 if blocks and args.enforce else 0


if __name__ == "__main__":
    sys.exit(main())
