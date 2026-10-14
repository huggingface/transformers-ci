"""Find a recent PR plausibly fixing the selected test.

Relore's inflight edge finds PRs that explicitly close a related issue. Recent
PRs without such an edge are found by title, then checked against changed files.
Neither signal proves the failing run is fixed, so the UI says "probable".
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from transformersci.otel import related_issues

REPOSITORY = related_issues.REPOSITORY
_cache: OrderedDict[tuple[str, str], tuple[float, dict]] = OrderedDict()
_inflight: dict[tuple[str, str], threading.Event] = {}
_lock = threading.Lock()
_slots = threading.BoundedSemaphore(2)
_TOPIC_SKIP = {"test", "tests", "modeling", "utils", "integration", "model", "models"}


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


def inflight(issue: int) -> list[dict]:
    base = os.getenv("PYTEST_TRACE_EXPORTER_RELORE_URL", "").strip()
    if not base:
        return []
    query = urlencode({"repo": REPOSITORY})
    request = Request(
        f"{base.rstrip('/')}/api/v1/inflight/{issue}?{query}",
        headers={"x-relore-client": related_issues.WIRE_VERSION},
    )
    with urlopen(request, timeout=8) as response:
        if response.headers.get("x-relore-version") != related_issues.WIRE_VERSION:
            raise ValueError("Relore wire version mismatch")
        raw = response.read(128 * 1024 + 1)
    if len(raw) > 128 * 1024:
        raise ValueError("Relore response too large")
    data = json.loads(raw)
    if (
        not isinstance(data, dict)
        or data.get("repo") != REPOSITORY
        or data.get("number") != issue
    ):
        raise ValueError("Invalid Relore inflight response")
    return data.get("claims") if isinstance(data.get("claims"), list) else []


def _candidate(item: dict, *, issue: int | None = None) -> dict | None:
    number = item.get("number")
    if type(number) is not int or number <= 0:
        return None
    if (
        item.get("state") != "open"
        and not item.get("merged")
        and not (
            isinstance(item.get("pull_request"), dict)
            and item["pull_request"].get("merged_at")
        )
    ):
        return None
    if issue is not None:
        if item.get("repo") != REPOSITORY or item.get("type") != "pr":
            return None
    elif not isinstance(item.get("pull_request"), dict):
        return None
    return {
        "number": number,
        "title": str(item.get("title") or "")[:300],
        "url": f"https://github.com/{REPOSITORY}/pull/{number}",
        "state": "merged"
        if item.get("merged") or (item.get("pull_request") or {}).get("merged_at")
        else "open",
        "issue": issue,
    }


def _search_query(path: str) -> str | None:
    topic = title_topic(path)
    if not topic:
        return None
    since = (datetime.now(timezone.utc) - timedelta(days=45)).date().isoformat()
    return f"repo:{REPOSITORY} is:pr in:title {topic} created:>={since}"


def find(
    trace_id: str,
    nodeid: str,
    load_details: Callable[[], list[dict]],
    search_prs: Callable[..., list],
    github_get: Callable[[str], object],
) -> dict:
    """Rank only recent, live PRs that change this exact test file."""
    path = test_file(nodeid)
    if not path:
        return {"status": "none"}
    key = (trace_id, nodeid)
    with _lock:
        cached = _cache.get(key)
        if cached and cached[0] > time.monotonic():
            return cached[1]
        pending = _inflight.get(key)
        if pending is None:
            done = _inflight[key] = threading.Event()
    if pending is not None:
        pending.wait(45)
        with _lock:
            cached = _cache.get(key)
        return cached[1] if cached else {"status": "busy"}
    if not _slots.acquire(timeout=5):
        result, ttl = {"status": "busy"}, 5
    else:
        try:
            candidates: dict[int, dict] = {}
            related = related_issues.lookup(trace_id, nodeid, load_details)
            for hit in related.get("hits", [])[:3]:
                if hit.get("type") != "issue":
                    continue
                try:
                    claims = inflight(hit["number"])
                except (OSError, ValueError):
                    continue  # direct PR discovery can still answer
                for claim in claims[:10]:
                    if isinstance(claim, dict) and (
                        pr := _candidate(claim, issue=hit["number"])
                    ):
                        candidates[pr["number"]] = pr
                if len(candidates) >= 5:
                    break
            query = _search_query(path)
            if query:
                try:
                    recent = search_prs(query, limit=20)
                except (OSError, ValueError):
                    if not candidates:
                        raise
                    recent = []
                for item in recent:
                    if isinstance(item, dict) and (pr := _candidate(item)):
                        candidates.setdefault(pr["number"], pr)
            api = os.getenv("PYTEST_GITHUB_API_URL", "https://api.github.com").rstrip(
                "/"
            )
            best = None
            topic = title_topic(path) or ""
            # Inflight matches lead; the title search only supplies a fallback.
            ordered = sorted(
                candidates.values(), key=lambda pr: (pr["issue"] is None, -pr["number"])
            )
            for pr in ordered[:5]:
                try:
                    files = github_get(
                        f"{api}/repos/{REPOSITORY}/pulls/{pr['number']}/files?per_page=100"
                    )
                except (OSError, ValueError):
                    continue
                if not isinstance(files, list) or not any(
                    isinstance(file, dict) and file.get("filename") == path
                    for file in files
                ):
                    continue
                if pr["issue"] is None and topic not in pr["title"].lower():
                    continue
                best = pr
                break
            result = {"status": "ok", "pr": best} if best else {"status": "none"}
            ttl = 600 if best else 120
        except (OSError, ValueError, KeyError, TypeError):
            result, ttl = {"status": "unavailable"}, 30
        finally:
            _slots.release()
    with _lock:
        _cache[key] = (time.monotonic() + ttl, result)
        while len(_cache) > 256:
            _cache.popitem(last=False)
        _inflight.pop(key, None)
        done.set()
    return result
