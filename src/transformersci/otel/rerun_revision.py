"""Validate the immutable source revision before dispatching a targeted rerun.

Also executable directly by the reusable GitHub Actions workflows using only
the standard library.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from urllib.request import Request, urlopen


def validate(
    api: Callable[[str], dict],
    *,
    pr: str,
    lane: str,
    head_sha: str,
    tested_sha: str,
    source_run: str,
) -> None:
    """Require the original run and tested merge to belong to the current PR head."""
    if not re.fullmatch(r"[1-9][0-9]{0,9}", pr) or lane not in {"cpu", "gpu"}:
        raise ValueError("Invalid PR number or lane")
    if not all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in (head_sha, tested_sha)):
        raise ValueError("Invalid head or tested SHA")
    match = re.fullmatch(r"([1-9][0-9]*):([1-9][0-9]*)", source_run)
    if not match:
        raise ValueError("Invalid source run and attempt")
    run_id, attempt = match.groups()
    pull = api(f"pulls/{pr}")
    if pull.get("state") != "open" or pull.get("head", {}).get("sha") != head_sha:
        raise ValueError("PR is closed or its head has changed")
    run = api(f"actions/runs/{run_id}/attempts/{attempt}")
    workflow = "pr-ci-caller.yml" if lane == "cpu" else "self-comment-ci.yml"
    event = "pull_request" if lane == "cpu" else "issue_comment"
    if (
        str(run.get("id")) != run_id
        or str(run.get("run_attempt")) != attempt
        or run.get("status") != "completed"
        or str(run.get("path", "")).split("@", 1)[0] != f".github/workflows/{workflow}"
        or run.get("event") != event
    ):
        raise ValueError("Source run does not match the completed source workflow")
    associations = run.get("pull_requests") or []
    if associations and not any(str(item.get("number")) == pr for item in associations):
        raise ValueError("Source run belongs to another PR")
    if lane == "cpu" and run.get("head_sha") != head_sha:
        raise ValueError("Source run belongs to another PR head")
    commit = api(f"commits/{tested_sha}")
    parents = commit.get("parents") or []
    if (
        commit.get("sha") != tested_sha
        or len(parents) != 2
        or parents[1].get("sha") != head_sha
    ):
        raise ValueError("Tested commit is not a merge of the requested PR head")


def main() -> None:
    repository = os.environ["REPOSITORY"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Invalid repository")

    def api(path: str) -> dict:
        request = Request(
            f"https://api.github.com/repos/{repository}/{path}",
            headers={
                "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urlopen(request, timeout=30) as response:
            return json.load(response)

    validate(
        api,
        pr=os.environ["PR_NUMBER"],
        lane=os.environ["LANE"],
        head_sha=os.environ["HEAD_SHA"],
        tested_sha=os.environ["TESTED_SHA"],
        source_run=os.environ["SOURCE_RUN"],
    )


if __name__ == "__main__":
    main()
