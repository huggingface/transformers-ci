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
  if docs are needed, and sets the status. Serge can read the PR's code, but
  none of it is run.
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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reviewbot.github_client import GitHubClient
    from reviewbot.llm_client import ChatCompletionClient, ToolCall
    from reviewbot.tools import ToolEnv


log = logging.getLogger(__name__)

STATUS_CONTEXT = "docs-check"
MARKER = "<!-- docs-check -->"
MAINTAINER_ASSOCIATIONS = {"MEMBER", "OWNER", "COLLABORATOR"}
BROWSE_TOOLS = {"read_file", "list_dir", "grep"}
MAX_TOOL_ROUNDS = 6
MAX_DIFF_CHARS = 120_000
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


def ask_llm(
    llm: ChatCompletionClient, messages: list[dict], tool_env: ToolEnv | None
) -> str:
    """Ask Serge whether the PR needs docs, and return its answer.

    Before answering, Serge can look around the PR's code: read a file, list a
    folder, or search for text. After `MAX_TOOL_ROUNDS` rounds of looking, it's
    told to stop and answer.
    """
    from reviewbot.tools import TOOL_SPECS, run_tool

    tools = (
        [spec for spec in TOOL_SPECS if spec["function"]["name"] in BROWSE_TOOLS]
        if tool_env
        else None
    )
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
            result = (
                run_tool(tool_env, tc.name, args)
                if tc.name in BROWSE_TOOLS
                else f"error: unknown tool {tc.name!r}"
            )
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


def ask_for_verdict(args, pr: dict, files: list[dict]) -> dict | None:
    """Ask Serge about the PR. Returns None if the call fails or the answer can't be read."""
    from reviewbot.llm_client import ChatCompletionClient
    from reviewbot.tools import ToolEnv

    with open(args.rules, encoding="utf-8") as f:
        rules = f.read()
    llm = ChatCompletionClient(
        api_base=os.environ.get("LLM_API_BASE") or "https://router.huggingface.co/v1",
        api_key=os.environ["LLM_API_KEY"],
        model=os.environ.get("LLM_MODEL") or None,
        bill_to=os.environ.get("LLM_BILL_TO") or None,
    )
    tool_env = ToolEnv(repo_root=args.checkout) if args.checkout else None
    try:
        reply = ask_llm(llm, build_messages(rules, pr, files), tool_env)
    except Exception:
        log.exception("LLM call failed.")
        return None
    return parse_verdict(reply, args.docs_path)


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
        choices=["approval", "push"],
        required=True,
        help="approval: ask Serge. push: reuse the last saved answer.",
    )
    parser.add_argument("--pr", type=int, required=True, help="The PR number.")
    parser.add_argument(
        "--head-sha",
        help="Approval mode: the commit that was approved. Stops if no maintainer approved it.",
    )
    parser.add_argument(
        "--rules", help="Approval mode: the file with the rules Serge follows."
    )
    parser.add_argument(
        "--checkout",
        help="Approval mode: the folder with the PR's code, for Serge to read.",
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
    pr = gh.get_pr(owner, repo, args.pr)

    if pr["state"] != "open" or pr["base"]["ref"] != args.base_branch:
        log.info("PR #%d is not open against %s; skipping.", args.pr, args.base_branch)
        return 0

    files = gh.get_pr_files(owner, repo, args.pr)
    verdict = load_verdict(args.verdict_file)

    if args.mode == "approval":
        reviews = gh.get_pr_reviews(owner, repo, args.pr)
        if not args.dry_run and args.head_sha not in approved_shas(reviews):
            log.info("PR #%d has no maintainer approval on %s.", args.pr, args.head_sha)
            return 0
        if reason := skip_reason(files, args.docs_path, library_path):
            log.info("PR #%d doesn't need Serge: %s", args.pr, reason)
        else:
            if problem := missing_setup(args):
                print(f"::error::docs-check isn't set up: {problem}")
                return 1
            verdict = ask_for_verdict(args, pr, files)
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
    log.info("PR #%d status: %s (%s)", args.pr, state, description)
    if args.dry_run:
        print(f"{STATUS_CONTEXT}: {state} ({description})")
    else:
        set_status(gh, owner, repo, pr["head"]["sha"], state, description)
    return 0


if __name__ == "__main__":
    sys.exit(main())
