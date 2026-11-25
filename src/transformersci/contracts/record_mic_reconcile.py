"""Settle the record-mic overlays of Transformers PRs that are no longer open (run hourly).

    record-mic-reconcile            # what it would do
    record-mic-reconcile --execute  # merge or drop

`record-mic` writes a PR's re-recorded expected values to `overlays/transformers-pr-<N>.json` on
the fleet dataset, applied to that PR's runs only. This replaces a PR-close hook in the caller
repository: for each overlay it asks GitHub for the PR's state, then

- merged: `pr_record merge` writes the overlay into the fleet tables (the values go live with the
  code) and deletes it; entries whose contract moved on the fleet's main since the record are
  reported as stale and not applied;
- closed without merging: `pr_record discard` deletes it and the mic-repo branches;
- open: nothing.

Merged PRs are settled in the order they merged on main (`merged_at`), not by number: the fleet
history says "from this commit on main, these values", so its rows follow main.

Between a PR's merge and the next pass, a contracts run whose code contains the merge already meets
its values: `tmic check --code-revision` applies the overlays of merged PRs not settled yet. This
pass is bookkeeping (pins, history, cleanup), not what makes the values live.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request

REPOSITORY = "huggingface/transformers"
OVERLAY = re.compile(r"overlays/transformers-pr-(\d+)\.json")


def overlays():
    """PR numbers that have an overlay on the fleet dataset's main."""
    from huggingface_hub import HfApi
    from transformers_mic import fleet

    files = HfApi().list_repo_files(fleet.REPOSITORY, repo_type="dataset")
    return sorted(int(m.group(1)) for f in files if (m := OVERLAY.fullmatch(f)))


def pr_state(pr, repository=REPOSITORY):
    """(open | merged | closed, the PR's commit on the base branch once merged, merged_at or None)."""
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repository}/pulls/{pr}",
        headers={"Accept": "application/vnd.github+json"},
    )
    if token := os.environ.get("GITHUB_TOKEN"):
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.load(response)
    if data.get("merged_at"):
        return "merged", data.get("merge_commit_sha"), data["merged_at"]
    return ("open" if data["state"] == "open" else "closed"), None, None


def settle(pr, state, execute, since=None):
    """Run pr_record for a PR that is no longer open; (exit code, its JSON output or None)."""
    command = "merge" if state == "merged" else "discard"
    args = [
        sys.executable,
        "-m",
        "transformers_mic.pr_record",
        command,
        "--pr",
        str(pr),
    ]
    if execute:
        args.append("--execute")
    if state == "merged" and since:
        # Runs of code without this commit keep meeting the values before it (fleet history).
        args += ["--since", since]
    done = subprocess.run(args, capture_output=True, text=True)
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    return done.returncode, json.loads(lines[-1]) if lines else None, done.stderr


def stale(result):
    return [
        entry
        for name in ("control", "prototypes")
        for entry in (result or {}).get(name, {}).get("stale", [])
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--execute", action="store_true", help="merge or drop (default: report)"
    )
    parser.add_argument(
        "--summary", help="append a markdown report to this file (the job summary)"
    )
    args = parser.parse_args(argv)

    lines, failed = ["### record-mic overlays", ""], False
    states = {pr: pr_state(pr) for pr in overlays()}
    if not states:
        lines.append("No overlay.")
    # Merged PRs in the order they merged on main (ISO timestamps sort as text), then the rest.
    order = sorted(
        states, key=lambda pr: (states[pr][2] is None, states[pr][2] or "", pr)
    )
    for pr in order:
        state, since, _ = states[pr]
        if state == "open":
            lines.append(f"- #{pr}: open, kept")
            continue
        code, result, stderr = settle(pr, state, args.execute, since)
        verb = "merged into the fleet" if state == "merged" else "dropped (PR closed)"
        if not args.execute:
            verb = f"would be {verb}"
        if result is None and code:
            failed = True
            lines.append(
                f"- #{pr}: {state}, FAILED: {stderr.strip().splitlines()[-1:] or ['no output']}"
            )
            continue
        note = (
            f"; stale, not applied: {', '.join(stale(result))}" if stale(result) else ""
        )
        lines.append(f"- #{pr}: {state}, {verb}{note}")
        if note:
            print(
                f"::warning::#{pr}: re-recorded values not applied (contract moved on the fleet): {', '.join(stale(result))}"
            )
    text = "\n".join(lines) + "\n"
    print(text)
    if args.summary:
        with open(args.summary, "a") as f:
            f.write(text)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
