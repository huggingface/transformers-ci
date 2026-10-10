"""Read-only PR diff endpoint and its iframe UI."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import threading
import time
from urllib.request import Request, urlopen
from urllib.error import URLError

from transformersci.agentic.github_api import gh_headers

PAGE_HTML = Path(__file__).with_suffix(".html").read_text()
SCRIPT = Path(__file__).with_suffix(".js").read_text()
PR_VIEW_SCRIPT = Path(__file__).with_name("pr_view.js").read_text()
MAX_BYTES = 4 * 1024 * 1024
_cache: OrderedDict[tuple[str, str], tuple[float, str]] = OrderedDict()
_lock = threading.Lock()
_comments_cache: OrderedDict[tuple[str, str], tuple[float, dict]] = OrderedDict()
BOT_ACCOUNTS = {"huggingfacedocbuilderdev"}


def is_bot_user(user: dict) -> bool:
    login = user.get("login", "").casefold()
    return user.get("type") == "Bot" or login.endswith("[bot]") or login in BOT_ACCOUNTS


THREAD_QUERY = """query($owner:String!, $name:String!, $number:Int!, $endCursor:String) {
 repository(owner:$owner, name:$name) { pullRequest(number:$number) {
  reviewThreads(first:100, after:$endCursor) {
   pageInfo { hasNextPage endCursor }
   nodes { id isResolved isOutdated path line startLine diffSide startDiffSide
    comments(first:1) { nodes { databaseId } } }
  }
 } }
}"""


def fetch_threads(repository: str, pr: str, token: str | None) -> list[dict] | None:
    """Resolution is a GraphQL thread property, not a REST comment property."""
    if not token:
        return None
    owner, name = repository.split("/")
    threads = []
    cursor = None
    try:
        for _ in range(10):
            request = Request(
                "https://api.github.com/graphql",
                data=json.dumps(
                    {
                        "query": THREAD_QUERY,
                        "variables": {
                            "owner": owner,
                            "name": name,
                            "number": int(pr),
                            "endCursor": cursor,
                        },
                    }
                ).encode(),
                headers={**gh_headers(token), "Content-Type": "application/json"},
            )
            with urlopen(request, timeout=20) as response:
                payload = response.read(MAX_BYTES + 1)
            if len(payload) > MAX_BYTES:
                return None
            result = json.loads(payload)
            if result.get("errors"):
                return None
            connection = result["data"]["repository"]["pullRequest"]["reviewThreads"]
            threads.extend(connection["nodes"])
            if not connection["pageInfo"]["hasNextPage"]:
                return threads
            cursor = connection["pageInfo"]["endCursor"]
    except (OSError, URLError, ValueError, KeyError, TypeError):
        return None
    return None


def enrich_threads(comments: list[dict], threads: list[dict] | None) -> None:
    roots = {}
    for thread in threads or []:
        for root in thread["comments"]["nodes"]:
            roots[root["databaseId"]] = thread
    by_id = {
        comment["id"]: comment for comment in comments if comment["kind"] == "inline"
    }
    for comment in by_id.values():
        root = comment
        seen = set()
        while root.get("reply_to") in by_id and root["id"] not in seen:
            seen.add(root["id"])
            root = by_id[root["reply_to"]]
        thread = roots.get(root["id"])
        comment["resolved"] = thread["isResolved"] if thread else None
        if thread:
            comment["outdated"] = thread["isOutdated"]
            comment["thread_id"] = thread["id"]
        # Replies share their parent comment's line range and diff context.
        for field in (
            "path",
            "line",
            "start_line",
            "original_line",
            "original_start_line",
            "side",
            "diff_hunk",
        ):
            if root.get(field) is not None:
                comment[field] = root[field]


def _validate(repository: str, pr: str) -> None:
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repository
    ):
        raise ValueError("Invalid repository")
    if not re.fullmatch(r"[1-9][0-9]{0,9}", pr):
        raise ValueError("Select a PR to view its diff.")


def fetch_diff(repository: str, pr: str, token: str | None = None) -> str:
    """Fetch the complete unified diff, bounded and cached for one minute."""
    _validate(repository, pr)
    key = (repository, pr)
    with _lock:
        cached = _cache.get(key)
        if cached and time.monotonic() - cached[0] < 60:
            _cache.move_to_end(key)
            return cached[1]
    headers = gh_headers(token)
    headers["Accept"] = "application/vnd.github.diff"
    headers["User-Agent"] = "transformers-ci-dashboard"
    request = Request(
        f"https://api.github.com/repos/{repository}/pulls/{pr}", headers=headers
    )
    with urlopen(request, timeout=30) as response:
        payload = response.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise ValueError("This diff is too large to display. Open it on GitHub.")
    diff = payload.decode("utf-8", errors="replace")
    with _lock:
        _cache[key] = (time.monotonic(), diff)
        _cache.move_to_end(key)
        while len(_cache) > 16:
            _cache.popitem(last=False)
    return diff


def fetch_comments(repository: str, pr: str, token: str | None = None) -> dict:
    """Combine discussion, review summaries and inline comments chronologically."""
    _validate(repository, pr)
    key = (repository, pr)
    with _lock:
        cached = _comments_cache.get(key)
        if cached and time.monotonic() - cached[0] < 60:
            _comments_cache.move_to_end(key)
            return cached[1]

    def fetch(kind: str, endpoint: str) -> tuple[list[dict], bool]:
        comments = []
        for page in range(1, 6):
            request = Request(
                f"https://api.github.com/repos/{repository}/{endpoint}?per_page=100&page={page}",
                headers={
                    **gh_headers(token),
                    "User-Agent": "transformers-ci-dashboard",
                },
            )
            with urlopen(request, timeout=20) as response:
                payload = response.read(MAX_BYTES + 1)
            if len(payload) > MAX_BYTES:
                raise ValueError(
                    "Comments are too large to display. Open them on GitHub."
                )
            batch = json.loads(payload)
            for item in batch:
                if not (item.get("body") or "").strip():
                    continue
                comments.append(
                    {
                        "id": item["id"],
                        "kind": kind,
                        "author": (item.get("user") or {}).get("login", "Unknown"),
                        "author_association": item.get("author_association", "NONE"),
                        "is_maintainer": item.get("author_association")
                        in {"OWNER", "MEMBER", "COLLABORATOR"},
                        "is_bot": is_bot_user(item.get("user") or {}),
                        "body": item["body"],
                        "url": item.get("html_url", ""),
                        "created_at": item.get("created_at")
                        or item.get("submitted_at")
                        or "",
                        "path": item.get("path"),
                        "line": item.get("line") or item.get("original_line"),
                        "start_line": item.get("start_line")
                        or item.get("original_start_line"),
                        "original_line": item.get("original_line"),
                        "original_start_line": item.get("original_start_line"),
                        "side": item.get("side"),
                        "diff_hunk": item.get("diff_hunk"),
                        "outdated": kind == "inline"
                        and item.get("line") is None
                        and item.get("subject_type") != "file",
                        "reply_to": item.get("in_reply_to_id"),
                        "state": item.get("state"),
                    }
                )
            if len(batch) < 100:
                return comments, False
        return comments, True

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(fetch, kind, endpoint)
            for kind, endpoint in (
                ("discussion", f"issues/{pr}/comments"),
                ("review", f"pulls/{pr}/reviews"),
                ("inline", f"pulls/{pr}/comments"),
            )
        ]
        batches = [future.result() for future in futures]
    threads = fetch_threads(repository, pr, token)
    result = {
        "comments": sorted(
            [comment for batch, _ in batches for comment in batch],
            key=lambda c: (c["created_at"], c["id"]),
        ),
        "truncated": any(truncated for _, truncated in batches),
        "resolution_available": threads is not None,
    }
    enrich_threads(result["comments"], threads)
    with _lock:
        _comments_cache[key] = (time.monotonic(), result)
        _comments_cache.move_to_end(key)
        while len(_comments_cache) > 16:
            _comments_cache.popitem(last=False)
    return result
