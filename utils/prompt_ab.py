#!/usr/bin/env python
"""Render the agent prompt for one failure group, before and after a prompt change.

Why this exists
---------------
Prompt edits to `integration_failure_triage.py` are shipped by merging — the
nightly pip-installs `transformers-ci@main` — and the only feedback loop was
waiting for 22:00 UTC and reading what serge did. That is a ~24h cycle for a
prose change, on a run that costs GPU minutes and millions of tokens.

Twice now a prompt change looked right and was not:

* transformers#48426 — serge set `force_cpu=True` in `conversion_mapping.py`, a
  production loading path, for a CI OOM. The first fix added the rule to
  `_OOM_LOAD_GUIDANCE`; rendering the real group showed that group receives an
  **empty addendum**, so none of it would have arrived.
* The same shape had already happened with the `Expectations` device-key
  convention, shipped without the drift threshold that makes it safe.

Both were caught by rendering the prompt for the actual group instead of
trusting that the block would be reached. That is all this script does, and it
is the cheap half of the loop: it proves what the agent *receives*. Proving what
the agent then *does* needs an LLM, which is the `--agent-brief` step below.

Usage
-----
    # what does the live prompt say for one group?
    python utils/prompt_ab.py --group phimoe-conversion

    # A/B against the prompt as it was at some ref
    python utils/prompt_ab.py --group phimoe-conversion --before 75d35e4

    # also prepare two identical transformers worktrees + a brief per side,
    # so an agent can be run against each and the patches compared
    python utils/prompt_ab.py --group phimoe-conversion --before 75d35e4 \
        --transformers ~/Dev/transformers --base ccba41e1c3 --agent-brief /tmp/ab

Add a group to `GROUPS` when you fix a new failure of a shape worth regression
testing. A group is just the target dict the triage builds, so anything
`instruction_addendum` dispatches on can be reproduced here.

The other half of the loop
--------------------------
`--payload-out` writes a `POST /tasks`-shaped payload per side, which is the
input to **`serge/playbooks/run-local-task-prompt.py`** in the playbooks repo.
That script replays serge's real agent loop — the actual model, the actual
tools — against a local checkout and writes the proposed patch to an artifact
dir, without creating a task or opening a PR. Use it when you need to know what
the agent *does*, not just what it is *told*:

    serge/checkout/.venv/bin/python serge/playbooks/run-local-task-prompt.py \
        --payload-json /tmp/ab/payload-after.json

Running any other model against the brief answers a weaker question — whether a
competent agent would be steered right — which is worth something, but is not
the same as serge's own model on serge's own loop.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

# ── Known groups, keyed by the bug they came from ────────────────────────────
# Each value is the `target` dict as the triage renders it. Keep the real trace
# text: `instruction_addendum` routes on it (see `_CONVERSION_PAT`).
GROUPS: dict[str, dict] = {
    # transformers#48426. A bad-commit cluster whose CI message is a wrapper that
    # never says OOM, so `classify` files it `other` and — before #108 — the
    # cluster branch returned "" and the agent got the trunk alone.
    "phimoe-conversion": {
        "kind": "cluster",
        "label": "1 integration tests regressed by commit bd9509355c8a (PR #47493)",
        "model": "phimoe",
        "failure_mode": "other",
        "failures": [
            {
                "test": (
                    "tests/models/phimoe/test_modeling_phimoe.py"
                    "::PhimoeIntegrationTest::test_phimoe_instruct_generation"
                ),
                "gpu": "multi-gpu",
                "failure_mode": "other",
                "latest_trace": (
                    "RuntimeError: We encountered some issues during automatic "
                    "conversion of the weights. For details look at the "
                    "`CONVERSION` entries of the above report!"
                ),
            }
        ],
    },
    # transformers#48437. An expectation rewrite to a degenerate value; the group
    # arrives as `other` with an AssertionError, which must still reach the
    # mismatch guidance rather than "this is a library bug".
    "big-bird-expectation": {
        "kind": "model_failures",
        "label": "2 tests for model `big_bird` failing with `other`",
        "model": "big_bird",
        "failure_mode": "other",
        "terminal_exc": "AssertionError",
        "failures": [
            {
                "test": (
                    "tests/models/big_bird/test_modeling_big_bird.py"
                    "::BigBirdModelIntegrationTest::test_fill_mask"
                ),
                "gpu": "single-gpu",
                "failure_mode": "other",
                "latest_trace": "AssertionError: 'happiness' != '<unk>'",
            }
        ],
    },
    # The flex_olmo case: a drift within an EXISTING ("cuda", 8) key, which the
    # convention in #105 does not cover (it tells you to ADD a key).
    "flex-olmo-drift": {
        "kind": "model_failures",
        "label": "2 tests for model `flex_olmo` failing with `output_mismatch`",
        "model": "flex_olmo",
        "failure_mode": "output_mismatch",
        "failures": [
            {
                "test": (
                    "tests/models/flex_olmo/test_modeling_flex_olmo.py"
                    "::FlexOlmoIntegrationTest::test_model_7b_logits"
                ),
                "gpu": "single-gpu",
                "failure_mode": "output_mismatch",
                "latest_trace": (
                    "AssertionError: Tensor-likes are not close! Mismatched "
                    "elements: 1 / 4. Greatest absolute difference: 0.0403"
                ),
            }
        ],
    },
}


def _render(target: dict) -> tuple[str, str]:
    """(trunk+addendum, addendum) for one target, using the installed module."""
    from transformersci.agentic import integration_failure_triage as itf

    return itf.build_instruction(target), itf.instruction_addendum(target)


def _render_at_ref(target: dict, ref: str) -> tuple[str, str]:
    """The same, from the module as it was at ``ref``.

    Runs in a subprocess against a checkout of that one file, because the
    prompt lives in module-level string constants — importing two versions in
    one interpreter would just give you whichever loaded first.
    """
    here = Path(__file__).resolve().parents[1]
    src = "src/transformersci/agentic/integration_failure_triage.py"
    old = subprocess.run(
        ["git", "-C", str(here), "show", f"{ref}:{src}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    with __import__("tempfile").TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "transformersci" / "agentic"
        pkg.mkdir(parents=True)
        # Mirror the package so the module's relative imports still resolve, then
        # overwrite just the one file with its old revision.
        for part in ("transformersci", "transformersci/agentic"):
            for f in (here / "src" / part).glob("*.py"):
                (Path(tmp) / part / f.name).write_text(f.read_text())
        (pkg / "integration_failure_triage.py").write_text(old)
        code = (
            "import json,sys;"
            "from transformersci.agentic import integration_failure_triage as itf;"
            "t=json.load(sys.stdin);"
            "print(json.dumps([itf.build_instruction(t), itf.instruction_addendum(t)]))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            input=json.dumps(target),
            capture_output=True,
            text=True,
            env={"PYTHONPATH": tmp, "PATH": "/usr/bin:/bin"},
        )
        if out.returncode != 0:
            raise SystemExit(f"could not render at {ref}:\n{out.stderr[-800:]}")
        return tuple(json.loads(out.stdout))


def _worktree(transformers: Path, base: str, dest: Path) -> None:
    if dest.exists():
        return
    subprocess.run(
        ["git", "-C", str(transformers), "worktree", "add", str(dest), base],
        check=True,
        capture_output=True,
    )


BRIEF = """\
You are the serge task agent. Work ONLY in {worktree}.

Below is the instruction you were given and the failure report. Follow the
instruction exactly. Investigate, then either produce a minimal patch as a
unified diff, or produce no patch and explain why.

Do NOT run the test suite (there is no GPU here). Do not commit or push. Write
your answer to {out}: the unified diff, then a short rationale.

{prompt}
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("Usage")[0])
    ap.add_argument("--group", required=True, choices=sorted(GROUPS))
    ap.add_argument("--before", help="git ref to render the prompt as it was then")
    ap.add_argument("--transformers", type=Path, help="a transformers clone")
    ap.add_argument("--base", help="commit to check the worktrees out at")
    ap.add_argument("--agent-brief", type=Path, help="write worktrees + briefs here")
    ap.add_argument("--context", type=Path, help="failure report to append verbatim")
    ap.add_argument(
        "--payload-out",
        type=Path,
        help="write a POST /tasks-shaped payload per side, for "
        "serge/playbooks/run-local-task-prompt.py",
    )
    args = ap.parse_args()

    target = GROUPS[args.group]
    after, after_add = _render(target)
    print(f"=== {args.group}: prompt NOW ===")
    print(f"  instruction {len(after):,} chars · addendum {len(after_add):,} chars")
    if not after_add:
        print("  !! EMPTY ADDENDUM — this group gets the trunk alone.")

    sides = {"after": after}
    if args.before:
        before, before_add = _render_at_ref(target, args.before)
        sides["before"] = before
        print(f"=== {args.group}: prompt at {args.before} ===")
        print(f"  instruction {len(before):,} chars · addendum {len(before_add):,} chars")
        if not before_add:
            print("  (empty addendum — trunk alone)")
        only_now = sorted(
            {ln.strip() for ln in after.splitlines() if ln.strip()}
            - {ln.strip() for ln in before.splitlines() if ln.strip()}
        )
        print(f"\n  lines the change ADDS for this group: {len(only_now)}")
        for ln in only_now[:12]:
            print("    + " + textwrap.shorten(ln, 100))

    if args.payload_out:
        ctx = args.context.read_text() if args.context else ""
        args.payload_out.mkdir(parents=True, exist_ok=True)
        for name, prompt in sides.items():
            payload = {
                "repo": "huggingface/transformers",
                "base_ref": args.base or "main",
                "instruction": prompt,
                "context": ctx,
                "output": {
                    "mode": "new_pr",
                    "branch_prefix": f"serge/promptab-{args.group}-{name}",
                    "title": f"[prompt-ab {name}] {target['label']}",
                },
            }
            out = args.payload_out / f"payload-{name}.json"
            out.write_text(json.dumps(payload, indent=2))
            print(f"  payload ({name}): {out}")
        print(
            "\nReplay with serge's own model and loop:\n"
            "  serge/checkout/.venv/bin/python serge/playbooks/run-local-task-prompt.py"
            f" --payload-json {args.payload_out}/payload-after.json"
        )

    if args.agent_brief:
        if not (args.transformers and args.base):
            raise SystemExit("--agent-brief needs --transformers and --base")
        ctx = args.context.read_text() if args.context else ""
        args.agent_brief.mkdir(parents=True, exist_ok=True)
        for name, prompt in sides.items():
            wt = args.agent_brief / f"wt-{name}"
            _worktree(args.transformers.expanduser(), args.base, wt)
            out = args.agent_brief / f"answer-{name}.md"
            (args.agent_brief / f"brief-{name}.md").write_text(
                BRIEF.format(worktree=wt, out=out, prompt=prompt + "\n" + ctx)
            )
            print(f"\n{name}: worktree {wt}\n  brief {args.agent_brief}/brief-{name}.md")
        print(
            "\nRun one FRESH agent per brief — a context that has already seen the "
            "answer cannot test the prompt — then diff the two answers."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
