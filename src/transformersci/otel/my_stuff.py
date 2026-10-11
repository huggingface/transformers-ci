"""Public PR candidates involving the authenticated dashboard user."""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

CACHE_SECONDS = 300
MAX_USERS = 64
ROLES = ("author", "commenter", "reviewed-by", "review-requested")
_cache: OrderedDict[tuple, tuple[float, tuple[str, ...]]] = OrderedDict()
_inflight: dict[tuple, Future] = {}
_lock = threading.Lock()


def candidates(
    login: str, repositories: tuple[str, ...], search: Callable[[str], list]
) -> list[str]:
    """Union recent authored, commented, reviewed, and requested-review PRs.

    Grafana chooses the latest ten with CI activity from this bounded set.
    The caller supplies a session-verified login, never a query-string identity.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}", login):
        raise ValueError("The signed-in account needs a GitHub username")
    if not repositories or any(
        not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo)
        for repo in repositories
    ):
        raise ValueError("No valid public repositories configured")
    key = (login.lower(), tuple(sorted(repositories)))
    with _lock:
        cached = _cache.get(key)
        if cached and time.monotonic() - cached[0] < CACHE_SECONDS:
            _cache.move_to_end(key)
            return list(cached[1])
        future = _inflight.get(key)
        owner = future is None
        if owner:
            future = Future()
            _inflight[key] = future
    if not owner:
        return list(future.result(timeout=30))
    try:
        scope = "is:pr " + " ".join(f"repo:{repo}" for repo in repositories)
        with ThreadPoolExecutor(max_workers=len(ROLES)) as pool:
            groups = list(
                pool.map(search, [f"{scope} {role}:{login}" for role in ROLES])
            )
        numbers = sorted(
            {
                str(item["number"])
                for group in groups
                for item in group
                if isinstance(item, dict)
                and isinstance(item.get("number"), int)
                and item["number"] > 0
                and "pull_request" in item
            },
            key=int,
            reverse=True,
        )
        with _lock:
            _cache[key] = (time.monotonic(), tuple(numbers))
            _cache.move_to_end(key)
            while len(_cache) > MAX_USERS:
                _cache.popitem(last=False)
        future.set_result(tuple(numbers))
        return numbers
    except Exception as error:
        future.set_exception(error)
        raise
    finally:
        with _lock:
            _inflight.pop(key, None)
