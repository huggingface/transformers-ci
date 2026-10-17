"""The probable fix for a failing test: one rule for the Test page and Project health.

An index of the recent PRs that are open or merged (a paged GraphQL search, ~17
pages of 1 point each, rebuilt in the background) maps each test file to the
PRs that change it. A PR is a candidate when it changes the test file and names
the file's topic (usually the model) in its title. Each candidate is judged by
when the test last failed on main: a PR merged before that failure did not fix
it. None of this proves a fix, so the UI says "probable".
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from datetime import datetime

from transformersci.otel import related_issues

REPOSITORY = related_issues.REPOSITORY
_TOPIC_SKIP = {"test", "tests", "modeling", "utils", "integration", "model", "models"}
# A PR that changes the test file and names the model is often a feature PR;
# a fix says so. Checked on prod rows on 2026-10-17: without it the best match
# for gemma3n's failing test was "dummy: trigger CI on gemma3n ...".
_FIX_WORDS = re.compile(
    r"\b(fix\w*|revert\w*|skip\w*|flak\w*|expect\w*|regression\w*|broke\w*|"
    r"restor\w*|repair\w*|resolv\w*|correct\w*|crash\w*|fail\w*|nan|oom)\b",
    re.IGNORECASE,
)


def test_file(nodeid: str) -> str | None:
    path = nodeid.split("::", 1)[0]
    if not re.fullmatch(r"tests/[\w./-]+\.py", path) or ".." in path.split("/"):
        return None
    return path


def title_topic(path: str) -> str | None:
    words = re.findall(r"[a-z0-9]+", path.rsplit("/", 1)[-1][:-3].lower())
    words = [word for word in words if word not in _TOPIC_SKIP and len(word) >= 3]
    if not words:
        words = [word for word in path.split("/")[-2].split("_") if len(word) >= 3]
    if not words:
        return None
    word = words[-1]
    return word[:-1] if word.endswith("s") and not word.endswith("ss") else word


INDEX_DAYS = 45
INDEX_REFRESH_SECONDS = 15 * 60
INDEX_SEARCHES = (
    "is:pr is:open created:>={since}",
    "is:pr is:merged merged:>={since}",
)
_INDEX_MAX_PAGES = 25  # 1250 PRs per search; 45 days is ~300 open + ~500 merged
_index: dict = {"built_at": 0.0, "by_file": {}}
_index_lock = threading.Lock()


def index_query(search: str, after: str | None) -> str:
    text = json.dumps(f"repo:{REPOSITORY} {search}")
    cursor = f", after: {json.dumps(after)}" if after else ""
    return (
        f"{{ search(query: {text}, type: ISSUE, first: 50{cursor}) "
        "{ pageInfo { hasNextPage endCursor } nodes { ... on PullRequest "
        "{ number title isDraft mergedAt files(first: 100) { nodes { path } } } } } }"
    )


def build_index(graphql: Callable[[str], dict], now: float) -> dict[str, list[dict]]:
    """Test file -> the recent open or merged PRs that change it."""
    since = time.strftime("%Y-%m-%d", time.gmtime(now - INDEX_DAYS * 86400))
    by_file: dict[str, list[dict]] = {}
    for search in INDEX_SEARCHES:
        after = None
        for _page in range(_INDEX_MAX_PAGES):
            reply = graphql(index_query(search.format(since=since), after))["search"]
            for node in reply.get("nodes") or []:
                number = node.get("number") if isinstance(node, dict) else None
                # A draft is not up for review; "dummy: trigger CI" PRs live there.
                if type(number) is not int or node.get("isDraft"):
                    continue
                merged_at = node.get("mergedAt")
                pr = {
                    "number": number,
                    "title": str(node.get("title") or "")[:300],
                    "url": f"https://github.com/{REPOSITORY}/pull/{number}",
                    "state": "merged" if merged_at else "open",
                    "merged_at": datetime.fromisoformat(
                        merged_at.replace("Z", "+00:00")
                    ).timestamp()
                    if merged_at
                    else None,
                }
                for file in (node.get("files") or {}).get("nodes") or []:
                    path = (file or {}).get("path") or ""
                    if test_file(path):
                        by_file.setdefault(path, []).append(pr)
            info = reply.get("pageInfo") or {}
            if not info.get("hasNextPage"):
                break
            after = info.get("endCursor")
    return by_file


def refresh_index(graphql: Callable[[str], dict], now: float | None = None) -> int:
    """Rebuild the index; on failure the previous one stays. Returns its size."""
    now = time.time() if now is None else now
    by_file = build_index(graphql, now)
    with _index_lock:
        _index["built_at"], _index["by_file"] = now, by_file
    return len(by_file)


def _names_this_test(title: str, nodeid: str) -> bool:
    """False when the title names tests (``test_x``, ``FooTest``) but not this one:
    "Fix Gemma3nVision2TextModelTest::test_model_parallelism" fixes another test
    in the same file."""
    parts = re.split(r"::|\[", nodeid)
    # A named function is the most specific: it must be this one.
    names = re.findall(r"\btest_\w+", title) or re.findall(r"\b\w+Test\b", title)
    return not names or any(name in parts for name in names)


def for_test(
    nodeid: str, last_failed: float | None, by_file: dict | None = None
) -> dict | None:
    """The PR to show for a test, best first: an open one, then one merged after
    the test last failed on main, then one merged before (it did not fix it).
    ``last_failed`` None means no failure on main is known: a merge counts as after."""
    path = test_file(nodeid)
    if by_file is None:
        with _index_lock:
            by_file = _index["by_file"]
    topic = title_topic(path) if path else None
    if not topic:
        return None
    matches = {
        pr["number"]: pr
        for pr in by_file.get(path, [])
        if topic in pr["title"].lower()
        and _FIX_WORDS.search(pr["title"])
        and _names_this_test(pr["title"], nodeid)
    }
    if not matches:
        return None

    def verdict(pr: dict) -> str:
        if pr["state"] == "open":
            return "open"
        return (
            "merged_after" if pr["merged_at"] > (last_failed or 0) else "merged_before"
        )

    rank = {"open": 0, "merged_after": 1, "merged_before": 2}
    best = min(
        matches.values(),
        key=lambda pr: (rank[verdict(pr)], -(pr["merged_at"] or 0), -pr["number"]),
    )
    return {
        "number": best["number"],
        "title": best["title"],
        "url": best["url"],
        "verdict": verdict(best),
        "merged_at": best["merged_at"],
        "others": len(matches) - 1,
    }


def index_built_at() -> float:
    with _index_lock:
        return _index["built_at"]
