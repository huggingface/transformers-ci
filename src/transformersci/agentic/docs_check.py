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

"""Checks whether a PR needs docs and shows the answer as a `docs-check` status on
the PR. Other repos run this workflow from their own workflow file. For
transformers, that's `.github/workflows/docs-check.yml`.

The code that does the checking is the `docs-check` command in this package, so
this workflow installs the package. The rules Serge follows come from the repo
that runs this workflow, because each repo decides what needs docs.

It runs in one of two modes:
- `approval`: runs after a maintainer approves a PR. It asks Serge
  whether the PR needs docs, saves the answer, leaves a comment
  if docs are needed, and sets the status. Serge reads the PR's files through
  the GitHub API, so the PR's code is never checked out, let alone run.
- `push`: runs when a PR is opened or updated. It doesn't ask Serge
  and doesn't download the PR's code. It reuses the last saved answer to set the
  status on the new commit.

The status doesn't block merging unless the calling repo makes it a required check.

The rules Serge follows live in the repo being checked. For transformers,
that's `.ai/docs-check-rules.md`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from reviewbot.github_client import GitHubClient
    from reviewbot.llm_client import ChatCompletionClient, ToolCall


log = logging.getLogger(__name__)

STATUS_CONTEXT = "docs-check"
MARKER = "<!-- docs-check -->"
MAINTAINER_ASSOCIATIONS = {"MEMBER", "OWNER", "COLLABORATOR"}
DOCS_TOOLS = {"read_file", "list_dir", "grep"}
PR_FILE_TOOL = "read_pr_file"
MAX_TOOL_ROUNDS = 6
MAX_DIFF_CHARS = 120_000
SWEEP_GRACE_MINUTES = 30
MAX_NO_ANSWERS = 3
NO_ANSWER_STATUS = "Serge couldn't give an answer. The sweep tries again later."
MAX_BODY_CHARS = 4_000
MAX_REASON_CHARS = 1_000
MAX_STATUS_CHARS = 140


def approved_shas(reviews: list[dict]) -> set[str]:
    """List the commits on the PR that a maintainer approved."""
    return {
        r["commit_id"]
        for r in reviews
        if r.get("state") == "APPROVED"
        and r.get("author_association") in MAINTAINER_ASSOCIATIONS
    }


def edits_docs(files: list[dict], docs_path: str) -> bool:
    """Check whether the PR changes any file in the docs folder."""
    return any(f["filename"].startswith(docs_path) for f in files)


def skip_reason(files: list[dict], docs_path: str, library_path: str) -> str | None:
    """Quickly rule out PRs that clearly don't need docs, so Serge isn't asked.

    A PR is skipped if it already edits the docs, doesn't touch the library code
    in `library_path`, or only changes private files (names starting with one
    underscore).

    Returns a short reason when the PR is skipped, or None when Serge should decide.
    """
    if edits_docs(files, docs_path):
        return "this PR updates the docs"
    src = [f["filename"] for f in files if f["filename"].startswith(library_path)]
    if not src:
        return f"no changes under {library_path}"
    if all(
        os.path.basename(p).startswith("_") and not os.path.basename(p).startswith("__")
        for p in src
    ):
        return "only changes private files"
    return None


def decide_status(
    files: list[dict],
    verdict: dict | None,
    docs_path: str,
    library_path: str,
) -> tuple[str, str]:
    """Choose the `docs-check` status to show on the PR: green, red, or pending.

    Returns `(state, description)`. `state` is "success" (green), "failure"
    (red), or "pending". `description` is the short text shown next to the
    status on the PR.

    `verdict` is Serge's last saved answer for this PR. It's None if Serge
    hasn't been asked yet, or couldn't give an answer.

    Checks the PR's own files first (with `skip_reason`), because they settle the
    answer whatever Serge said: docs edits turn it green, even after Serge said
    docs are needed.
    """
    if reason := skip_reason(files, docs_path, library_path):
        return "success", f"No docs needed: {reason}."
    if verdict is None:
        return "pending", "Not checked yet. Runs after a maintainer approves."
    if verdict["needs_docs"]:
        return "failure", "Serge thinks this PR needs docs. It doesn't block merging."
    return "success", "Serge found no docs needed."


def format_diff(files: list[dict], budget: int = MAX_DIFF_CHARS) -> str:
    """Put the changes from every file into one block of text for Serge.

    The text is capped at `budget` characters so it doesn't overwhelm Serge.
    When a file's changes don't fit, only its name is listed. When even the
    name doesn't fit, the text ends with a note saying how many files were left
    out. Serge can still open any file itself.
    """
    parts = []
    used = 0
    for i, f in enumerate(files):
        header = (
            f"### {f['filename']} ({f['status']}, +{f['additions']} -{f['deletions']})"
        )
        patch = f.get("patch") or "(no patch: binary or too large)"
        if used + len(header) + len(patch) > budget:
            patch = "(patch omitted: diff budget exhausted, use read_file if needed)"
        if used + len(header) + len(patch) > budget:
            parts.append(f"(... {len(files) - i} more changed files not shown)")
            break
        used += len(header) + len(patch)
        parts.append(f"{header}\n{patch}")
    return "\n\n".join(parts)


def build_messages(rules: str, pr: dict, files: list[dict]) -> list[dict]:
    """Write what Serge is sent: the rules first, then the PR's title, description, and changes."""
    body = (pr.get("body") or "")[:MAX_BODY_CHARS]
    user = (
        f"PR #{pr['number']}: {pr['title']}\n\n"
        f"Description:\n{body or '(empty)'}\n\n"
        f"Changed files and diff:\n\n{format_diff(files)}"
    )
    return [{"role": "system", "content": rules}, {"role": "user", "content": user}]


def tool_call_dict(tc: ToolCall) -> dict:
    """Record Serge's request to read a file in the format the model service expects.

    Copied from Serge's private `_assistant_tool_call_dict`, because Serge could
    rename that function at any time and break this script.
    """
    call = {
        "id": tc.id,
        "type": "function",
        "function": {"name": tc.name, "arguments": tc.arguments},
    }
    if tc.thought_signature:
        call["extra_content"] = {"google": {"thought_signature": tc.thought_signature}}
    return call


def tool_specs(docs_path: str) -> list[dict]:
    """List the tools Serge can use to look around before answering.

    `read_file`, `list_dir`, and `grep` are Serge's own tools, run on a checkout
    of the docs folder from the base branch. Their descriptions are rewritten
    because Serge's say they read the PR head. `read_pr_file` reads a file from
    the PR itself, through the GitHub API.
    """
    from reviewbot.tools import MAX_READ_LINES, TOOL_SPECS

    where = (
        f"the docs folder (`{docs_path}`) as it is on the base branch, before this PR"
    )
    descriptions = {
        "read_file": f"Read a slice of a file in {where}. Lines are 1-indexed.",
        "list_dir": f"List the files and subfolders of a folder in {where}.",
        "grep": (
            f"Search {where} for a regex (git grep -E). Use this to find pages "
            "that mention the code the PR changes. Returns `path:line:text`."
        ),
    }
    specs = []
    for spec in TOOL_SPECS:
        name = spec["function"]["name"]
        if name in DOCS_TOOLS:
            specs.append(
                {
                    **spec,
                    "function": {**spec["function"], "description": descriptions[name]},
                }
            )
    specs.append(
        {
            "type": "function",
            "function": {
                "name": PR_FILE_TOOL,
                "description": (
                    "Read a slice of a file as it is in this PR. Use this when the "
                    "diff doesn't show enough, for example the rest of a changed "
                    "class. Lines are 1-indexed."
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path relative to the repository root.",
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "First line to return. Defaults to 1.",
                        },
                        "end_line": {
                            "type": "integer",
                            "description": f"Last line to return. At most {MAX_READ_LINES} lines per call.",
                        },
                    },
                    "required": ["path"],
                },
            },
        }
    )
    return specs


def ask_llm(
    llm: ChatCompletionClient,
    messages: list[dict],
    tools: list[dict] | None,
    run_tool: Callable[[str, dict], str],
) -> str:
    """Ask Serge whether the PR needs docs, and return its answer.

    Before answering, Serge can look around with `tools`: the docs on the base
    branch, and the PR's own files. `run_tool` runs each tool Serge asks for.
    After `MAX_TOOL_ROUNDS` rounds of looking, it's told to stop and answer.
    """
    for _ in range(MAX_TOOL_ROUNDS):
        chat = llm.complete(messages, tools=tools, temperature=0.0, max_tokens=4096)
        if not chat.tool_calls:
            return chat.content
        messages.append(
            {
                "role": "assistant",
                "content": chat.content or "",
                "tool_calls": [tool_call_dict(tc) for tc in chat.tool_calls],
            }
        )
        for tc in chat.tool_calls:
            try:
                args = json.loads(tc.arguments or "{}")
            except ValueError:
                args = {}
            result = run_tool(tc.name, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": result,
                }
            )
    messages.append(
        {"role": "user", "content": "Stop browsing. Reply with the JSON object now."}
    )
    return llm.complete(
        messages,
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=1024,
    ).content


SAFE_PATH_RE = re.compile(r"[\w.-]+(?:/[\w.-]+)*")


def allowed_pr_path(path: str, files: list[dict], library_path: str) -> bool:
    """Decide which of the PR's files Serge may read with `read_pr_file`.

    Serge may read any file the PR changes, wherever it is, and any file in the
    library code, changed or not, so it can check whether something is public
    (for example in an unchanged `__init__.py`).

    `path` is already known to be a plain relative path (no `..`, no leading `/`).
    `files` are the PR's changed files, as GitHub lists them.
    """
    return path.startswith(library_path) or path in {f["filename"] for f in files}


def read_pr_file(
    gh: GitHubClient,
    owner: str,
    repo: str,
    sha: str,
    args: dict,
    allowed: Callable[[str], bool],
) -> str:
    """Read a slice of a file at the PR's head commit, through the GitHub API.

    This is how Serge sees the PR's code without it being checked out. It reads
    the same blobs the GitHub UI shows, at the commit the maintainer approved.
    Always returns text for Serge, even on failure, so Serge can try something else.

    Uses Serge's own limits for one read, so it returns as much as Serge's
    `read_file` does on the docs.
    """
    from reviewbot.tools import MAX_READ_LINES, MAX_TOOL_OUTPUT_CHARS

    path = args.get("path")
    if (
        not isinstance(path, str)
        or not SAFE_PATH_RE.fullmatch(path)
        or ".." in path.split("/")
    ):
        return "error: path must be a plain path relative to the repository root"
    if not allowed(path):
        return f"error: {path} can't be read with {PR_FILE_TOOL}"
    # The raw media type returns the file itself, up to 100 MB. The default JSON
    # response leaves `content` empty for files over 1 MB, which would read as
    # an empty file.
    try:
        r = gh.session.get(
            f"https://api.github.com/repos/{owner}/{repo}/contents/{path}",
            params={"ref": sha},
            headers={"Accept": "application/vnd.github.raw"},
            timeout=60,
        )
        if r.status_code == 404:
            return f"error: {path} doesn't exist in this PR"
        r.raise_for_status()
    except Exception:
        log.exception("Couldn't read %s at %s.", path, sha)
        return f"error: couldn't read {path}"
    if "json" in r.headers.get("Content-Type", "") and r.text.lstrip().startswith("["):
        return f"error: {path} is a folder, not a file"
    text = r.content.decode("utf-8", errors="replace")
    try:
        start = max(1, int(args.get("start_line") or 1))
        end = int(args.get("end_line") or start + MAX_READ_LINES - 1)
    except (TypeError, ValueError):
        return "error: start_line and end_line must be integers"
    end = min(end, start + MAX_READ_LINES - 1)
    lines = text.splitlines()[start - 1 : end]
    if not lines:
        return f"{path}: no lines in range {start}-{end}"
    body = "\n".join(f"{i:>6}\t{line}" for i, line in enumerate(lines, start=start))
    out = f"{path} (lines {start}-{start + len(lines) - 1}):\n{body}"
    if len(out) <= MAX_TOOL_OUTPUT_CHARS:
        return out
    # Same note as Serge's own tools, so Serge knows it didn't see every line.
    return (
        out[:MAX_TOOL_OUTPUT_CHARS]
        + f"\n[... truncated; output exceeded {MAX_TOOL_OUTPUT_CHARS} chars; "
        "re-call with a narrower line range ...]"
    )


def parse_verdict(content: str, docs_path: str) -> dict | None:
    """Read Serge's answer. Returns None if the answer can't be understood.

    Serge is asked to reply in JSON, but some models add extra text around it,
    so this also tries just the part between the first `{` and the last `}`.

    Only suggested pages that are plain file paths inside `docs_path` are kept,
    and no more than 3 of them.
    """
    raw = content or ""
    try:
        obj = json.loads(raw)
    except ValueError:
        start, end = raw.find("{"), raw.rfind("}")
        try:
            obj = json.loads(raw[start : end + 1]) if 0 <= start < end else None
        except ValueError:
            obj = None
    if not isinstance(obj, dict) or not isinstance(obj.get("needs_docs"), bool):
        return None
    page_re = re.compile(re.escape(docs_path) + r"[\w./-]+")
    pages = obj.get("suggested_pages") or []
    return {
        "needs_docs": obj["needs_docs"],
        "reason": str(obj.get("reason", "")).strip(),
        "suggested_pages": [
            p for p in pages if isinstance(p, str) and page_re.fullmatch(p)
        ][:3],
    }


def neutralize(text: str) -> str:
    """Clean up Serge's text so it's safe to post on the PR.

    Anyone can write a PR, and Serge might repeat what they wrote. So this
    removes links, stops @mentions from notifying anyone, turns HTML into plain
    text, and shortens the text to `MAX_REASON_CHARS` characters.
    """
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(?:https?|ftp)://\S+|www\.\S+", "(link removed)", text)
    text = text.replace("@", "@\u200b")
    return text[:MAX_REASON_CHARS]


def format_comment(verdict: dict, docs_path: str) -> str:
    """Write the comment to post on the PR from Serge's answer.

    The comment starts with a hidden tag so the bot can find its own comment
    later and doesn't comment twice.
    """
    lines = [MARKER, "**This PR might need docs.**", "", neutralize(verdict["reason"])]
    if verdict["suggested_pages"]:
        lines += [
            "",
            "Pages that might need an update:",
            *[f"- `{p}`" for p in verdict["suggested_pages"]],
        ]
    lines += [
        "",
        (
            f"<sub>Posted automatically after approval. Docs live in `{docs_path}`, "
            f"and updating them turns the `{STATUS_CONTEXT}` check green. "
            "The check doesn't block merging.</sub>"
        ),
    ]
    return "\n".join(lines)


def load_verdict(path: str | None) -> dict | None:
    """Read the last saved answer from Serge for this PR, or None if there isn't one."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            verdict = json.load(f)
    except (OSError, ValueError):
        return None
    return verdict if isinstance(verdict, dict) and "needs_docs" in verdict else None


def save_verdict(path: str, verdict: dict) -> None:
    """Save Serge's answer so later runs can reuse it without asking again."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(verdict, f)


def already_commented(gh: GitHubClient, owner: str, repo: str, number: int) -> bool:
    """Check whether the bot has already commented on this PR, by looking for its hidden tag."""
    page = 1
    while True:
        r = gh.session.get(
            f"https://api.github.com/repos/{owner}/{repo}/issues/{number}/comments",
            params={"per_page": 100, "page": page},
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json()
        if any(MARKER in (c.get("body") or "") for c in batch):
            return True
        if len(batch) < 100:
            return False
        page += 1


def set_status(
    gh: GitHubClient, owner: str, repo: str, sha: str, state: str, description: str
) -> None:
    """Set the `docs-check` status on a commit. Clicking it opens this workflow run."""
    payload = {
        "state": state,
        "context": STATUS_CONTEXT,
        "description": description[:MAX_STATUS_CHARS],
    }
    if os.environ.get("GITHUB_RUN_ID"):
        server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
        payload["target_url"] = (
            f"{server}/{os.environ['GITHUB_REPOSITORY']}/actions/runs/"
            f"{os.environ['GITHUB_RUN_ID']}"
        )
    r = gh.session.post(
        f"https://api.github.com/repos/{owner}/{repo}/statuses/{sha}",
        json=payload,
        timeout=30,
    )
    r.raise_for_status()


def missing_setup(args) -> str | None:
    """Check that everything Serge needs is set up. Returns what's missing, or None.

    Without a model, Serge would quietly use whichever model the service lists
    first, so a missing model is treated as a setup error, not a default.
    """
    if not os.environ.get("LLM_MODEL"):
        return "no model is set (LLM_MODEL)"
    if not os.environ.get("LLM_API_KEY"):
        return "no key for the model service is set (LLM_API_KEY)"
    if not args.rules or not os.path.isfile(args.rules):
        return f"the rules file wasn't found ({args.rules})"
    return None


def ask_for_verdict(
    args,
    gh: GitHubClient,
    owner: str,
    repo: str,
    pr: dict,
    files: list[dict],
    library_path: str,
) -> dict | None:
    """Ask Serge about the PR. Returns None if the call fails or the answer can't be read."""
    from reviewbot import tools as serge_tools
    from reviewbot.llm_client import ChatCompletionClient

    with open(args.rules, encoding="utf-8") as f:
        rules = f.read()
    llm = ChatCompletionClient(
        api_base=os.environ.get("LLM_API_BASE") or "https://router.huggingface.co/v1",
        api_key=os.environ["LLM_API_KEY"],
        model=os.environ.get("LLM_MODEL") or None,
        bill_to=os.environ.get("LLM_BILL_TO") or None,
    )
    docs_root = (
        os.path.join(args.base_checkout, args.docs_path) if args.base_checkout else None
    )
    if not docs_root or not os.path.isdir(docs_root):
        docs_root = None
    tool_env = serge_tools.ToolEnv(repo_root=args.base_checkout) if docs_root else None

    def run_tool(name: str, tool_args: dict) -> str:
        if name == PR_FILE_TOOL:
            return read_pr_file(
                gh,
                owner,
                repo,
                args.head_sha,
                tool_args,
                lambda p: allowed_pr_path(p, files, library_path),
            )
        if name in DOCS_TOOLS and tool_env:
            if name == "grep" and not tool_args.get("path"):
                tool_args = {**tool_args, "path": args.docs_path}
            return serge_tools.run_tool(tool_env, name, tool_args)
        return f"error: unknown tool {name!r}"

    tools = [
        spec
        for spec in tool_specs(args.docs_path)
        if tool_env or spec["function"]["name"] == PR_FILE_TOOL
    ]
    try:
        reply = ask_llm(llm, build_messages(rules, pr, files), tools, run_tool)
    except Exception:
        log.exception("LLM call failed.")
        return None
    return parse_verdict(reply, args.docs_path)


def commit_files(
    gh: GitHubClient, owner: str, repo: str, base_sha: str, sha: str
) -> list[dict]:
    """List what a PR changed as of `sha`, in the same shape as `get_pr_files`.

    This is the diff GitHub would have shown at that commit: from where the PR
    branched off `base_sha` up to `sha`. The compare API lists at most 300 files.
    """
    r = gh.session.get(
        f"https://api.github.com/repos/{owner}/{repo}/compare/{base_sha}...{sha}",
        timeout=60,
    )
    r.raise_for_status()
    return r.json().get("files", [])


SWEEP_QUERY = """
query($owner: String!, $repo: String!, $base: String!, $cursor: String, $context: String!) {
  repository(owner: $owner, name: $repo) {
    pullRequests(baseRefName: $base, states: [OPEN, MERGED], first: 50, after: $cursor,
                 orderBy: {field: UPDATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        updatedAt
        headRefOid
        reviews(states: APPROVED, last: 50) {
          nodes { authorAssociation submittedAt commit { oid } }
        }
        commits(last: 1) {
          nodes { commit { status { context(name: $context) { state } } } }
        }
      }
    }
  }
}
"""


def approved_prs(
    gh: GitHubClient, owner: str, repo: str, base: str, since: str
) -> list[tuple[dict, list[dict], str | None]]:
    """List the open or merged PRs into `base` updated since `since` that have an approval.

    Returns `(pr, reviews, state)` for each, newest first. `pr` and `reviews`
    have the same keys as the REST API's, as far as `sha_to_check` reads them.
    `state` is the `docs-check` status on the PR's head commit: "success",
    "failure", "pending", "error", or None if there's none.

    One GraphQL query covers 50 PRs with their approvals and status, so a busy
    repo doesn't use up its API limit. `since` is an ISO 8601 time, which sorts
    the same way as GitHub's `updatedAt`.
    """
    found = []
    cursor = None
    while True:
        r = gh.session.post(
            "https://api.github.com/graphql",
            json={
                "query": SWEEP_QUERY,
                "variables": {
                    "owner": owner,
                    "repo": repo,
                    "base": base,
                    "cursor": cursor,
                    "context": STATUS_CONTEXT,
                },
            },
            timeout=60,
        )
        r.raise_for_status()
        body = r.json()
        if body.get("errors"):
            raise RuntimeError(f"GraphQL query failed: {body['errors']}")
        page = body["data"]["repository"]["pullRequests"]
        for node in page["nodes"]:
            if node["updatedAt"] < since:
                return found
            reviews = [
                {
                    "state": "APPROVED",
                    "commit_id": (rv.get("commit") or {}).get("oid"),
                    "author_association": rv["authorAssociation"],
                    "submitted_at": rv["submittedAt"],
                }
                for rv in node["reviews"]["nodes"]
            ]
            if not reviews:
                continue
            commits = node["commits"]["nodes"]
            status = (commits[0]["commit"].get("status") or {}) if commits else {}
            context = status.get("context") or {}
            state = context["state"].lower() if context.get("state") else None
            pr = {"number": node["number"], "head": {"sha": node["headRefOid"]}}
            found.append((pr, reviews, state))
        if not page["pageInfo"]["hasNextPage"]:
            return found
        cursor = page["pageInfo"]["endCursor"]


def no_answer_count(gh: GitHubClient, owner: str, repo: str, sha: str) -> int:
    """Count how many times Serge was asked about a commit and couldn't answer.

    Reads the commit's status history, newest first, which keeps every
    `docs-check` status that was set, not only the latest.
    """
    r = gh.session.get(
        f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}/statuses",
        params={"per_page": 100},
        timeout=30,
    )
    r.raise_for_status()
    return sum(
        1
        for s in r.json()
        if s["context"] == STATUS_CONTEXT and s.get("description") == NO_ANSWER_STATUS
    )


def sha_to_check(pr: dict, reviews: list[dict], state: str | None) -> str | None:
    """Decide whether the sweep should run the approval check on this PR, and on which commit.

    `reviews` are the PR's reviews, as GitHub lists them, oldest first.
    `state` is the `docs-check` status on the PR's head commit (see `approved_prs`).

    Returns the commit to check, or None to leave the PR alone. Approval mode
    stops unless a maintainer approved that exact commit, so the commit must be
    one of `approved_shas(reviews)`.

    Green and red are final answers, so those PRs are left alone. Anything else
    on an approved PR means the approval run never answered: no status (it never
    ran), "error" (it failed), or "pending" (the push job's placeholder, which
    only the approval run replaces). The head commit is checked when it was
    approved, otherwise the most recently approved commit.
    """
    if state in ("success", "failure"):
        return None
    approved = approved_shas(reviews)
    if pr["head"]["sha"] in approved:
        return pr["head"]["sha"]
    for review in reversed(reviews):
        if review.get("commit_id") in approved:
            return review["commit_id"]
    return None


def sweep(gh: GitHubClient, owner: str, repo: str, args) -> int:
    """Find approved PRs that never got an answer, so the workflow can check them.

    The approval run misses some PRs: it can't run for a fork PR until someone
    approves the workflow run, and it can fail before Serge is set up. This
    looks at PRs updated in the last `--since-hours` hours and lists the ones
    `sha_to_check` picks, as a JSON list of `{"pr": ..., "sha": ...}` in the
    `prs` output. The workflow then runs approval mode on each.

    Two kinds of PR are left out even when `sha_to_check` picks them:
    - approved in the last `SWEEP_GRACE_MINUTES`, because their approval run
      may still be going, and both would ask Serge about the same commit
    - PRs Serge already failed to answer `MAX_NO_ANSWERS` times, so a PR it
      can't handle isn't retried every run
    """
    now = datetime.now(timezone.utc)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    since = (now - timedelta(hours=args.since_hours)).strftime(fmt)
    grace = (now - timedelta(minutes=SWEEP_GRACE_MINUTES)).strftime(fmt)
    todo = []
    for pr, reviews, state in approved_prs(gh, owner, repo, args.base_branch, since):
        sha = sha_to_check(pr, reviews, state)
        if not sha:
            continue
        if any(
            r["submitted_at"] > grace
            for r in reviews
            if r["author_association"] in MAINTAINER_ASSOCIATIONS
        ):
            log.info(
                "PR #%d was just approved; leaving it to the approval run.",
                pr["number"],
            )
            continue
        if no_answer_count(gh, owner, repo, pr["head"]["sha"]) >= MAX_NO_ANSWERS:
            log.info(
                "PR #%d: Serge couldn't answer %d times; giving up.",
                pr["number"],
                MAX_NO_ANSWERS,
            )
            continue
        todo.append({"pr": str(pr["number"]), "sha": sha})
        if len(todo) >= args.max_prs:
            break
    log.info("Sweep found %d PRs to check: %s", len(todo), todo)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"prs={json.dumps(todo)}\n")
    return 0


def main() -> int:
    """Run the docs check on one PR.

    Returns 0 whether or not it comments, so a quiet run isn't marked as failed.
    The run only fails if GitHub can't be reached, an option is wrong, or Serge
    isn't set up (no model, no key, or no rules file).
    """
    from reviewbot.github_client import GitHubClient

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mode",
        choices=["approval", "push", "sweep"],
        required=True,
        help=(
            "approval: ask Serge. push: reuse the last saved answer. "
            "sweep: list approved PRs that never got an answer."
        ),
    )
    parser.add_argument(
        "--pr", type=int, help="The PR number. Required in approval and push mode."
    )
    parser.add_argument(
        "--since-hours",
        type=int,
        default=24,
        help="Sweep mode: look at PRs updated in this many hours.",
    )
    parser.add_argument(
        "--max-prs",
        type=int,
        default=20,
        help="Sweep mode: list at most this many PRs, so one run can't ask Serge too often.",
    )
    parser.add_argument(
        "--head-sha",
        help="Approval mode: the commit that was approved. Stops if no maintainer approved it.",
    )
    parser.add_argument(
        "--rules", help="Approval mode: the file with the rules Serge follows."
    )
    parser.add_argument(
        "--base-checkout",
        help=(
            "Approval mode: a checkout of the base branch with the docs folder, "
            "for Serge to search. The PR's own files are read through the GitHub API."
        ),
    )
    parser.add_argument(
        "--verdict-file",
        help="The file that holds Serge's last answer. Approval mode writes it; push mode reads it.",
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="The repo, as owner/name.",
    )
    parser.add_argument(
        "--base-branch", default="main", help="Only PRs into this branch are checked."
    )
    parser.add_argument(
        "--docs-path",
        default="docs/source/en/",
        help="The docs folder. A PR that edits it turns the status green.",
    )
    parser.add_argument(
        "--library-path", help="The library code folder. Defaults to src/<repo name>/."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the comment and status instead of posting them. Also skips the approval check.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    owner, repo = args.repo.split("/")
    library_path = args.library_path or f"src/{repo.replace('-', '_')}/"
    gh = GitHubClient(os.environ["GITHUB_TOKEN"])

    if args.mode == "sweep":
        return sweep(gh, owner, repo, args)
    if args.pr is None:
        parser.error("--pr is required in approval and push mode")
    pr = gh.get_pr(owner, repo, args.pr)

    # Approval mode also checks merged PRs: maintainers often merge seconds after
    # approving, before this run starts, and the docs gap is still worth flagging.
    checkable = pr["state"] == "open" or (
        args.mode == "approval" and pr.get("merged_at")
    )
    if not checkable or pr["base"]["ref"] != args.base_branch:
        log.info("PR #%d is not open against %s; skipping.", args.pr, args.base_branch)
        return 0

    files = gh.get_pr_files(owner, repo, args.pr)
    verdict = load_verdict(args.verdict_file)

    asked = False
    if args.mode == "approval":
        # A dry run can leave out --head-sha. Check the PR's current head then,
        # so Serge doesn't read files from the default branch.
        args.head_sha = args.head_sha or pr["head"]["sha"]
        reviews = gh.get_pr_reviews(owner, repo, args.pr)
        if not args.dry_run and args.head_sha not in approved_shas(reviews):
            log.info("PR #%d has no maintainer approval on %s.", args.pr, args.head_sha)
            return 0
        # The approved commit can be older than the head, for example when the
        # sweep checks the last approved commit. Judge that commit's own changes,
        # so the diff Serge gets matches the files it reads.
        approved_files = (
            files
            if args.head_sha == pr["head"]["sha"]
            else commit_files(gh, owner, repo, pr["base"]["sha"], args.head_sha)
        )
        if reason := skip_reason(approved_files, args.docs_path, library_path):
            log.info("PR #%d doesn't need Serge: %s", args.pr, reason)
        else:
            if problem := missing_setup(args):
                print(f"::error::docs-check isn't set up: {problem}")
                # Without this the PR keeps the push job's "Not checked yet"
                # status, which hides the failure. The sweep retries it later,
                # cheaply: this stops before Serge is asked.
                if not args.dry_run:
                    set_status(
                        gh,
                        owner,
                        repo,
                        pr["head"]["sha"],
                        "error",
                        f"docs-check isn't set up: {problem}",
                    )
                return 1
            asked = True
            verdict = ask_for_verdict(
                args, gh, owner, repo, pr, approved_files, library_path
            )
            log.info("PR #%d verdict: %s", args.pr, verdict)
            if verdict and args.verdict_file:
                save_verdict(args.verdict_file, verdict)
                if os.environ.get("GITHUB_OUTPUT"):
                    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
                        f.write("verdict=done\n")
            if verdict and verdict["needs_docs"] and verdict["reason"]:
                body = format_comment(verdict, args.docs_path)
                if args.dry_run:
                    print(body)
                elif not already_commented(gh, owner, repo, args.pr):
                    gh.post_issue_comment(owner, repo, args.pr, body)

        latest = gh.get_pr(owner, repo, args.pr)
        if latest["head"]["sha"] != pr["head"]["sha"]:
            pr, files = latest, gh.get_pr_files(owner, repo, args.pr)

    state, description = decide_status(files, verdict, args.docs_path, library_path)
    if state == "pending" and asked:
        # Serge was asked but its call failed or its answer couldn't be read.
        # The sweep counts these so it stops retrying after a few tries.
        state, description = "error", NO_ANSWER_STATUS
    log.info("PR #%d status: %s (%s)", args.pr, state, description)
    if args.dry_run:
        print(f"{STATUS_CONTEXT}: {state} ({description})")
    else:
        set_status(gh, owner, repo, pr["head"]["sha"], state, description)
    return 0


if __name__ == "__main__":
    sys.exit(main())
