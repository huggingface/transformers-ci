from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from transformersci.otel import my_stuff, trace_exporter


@pytest.fixture(autouse=True)
def empty_cache(monkeypatch):
    monkeypatch.setattr(my_stuff, "_cache", my_stuff.OrderedDict())
    monkeypatch.setattr(my_stuff, "_inflight", {})


def pr(number):
    return {"number": number, "pull_request": {}}


def test_all_interactions_are_unioned_and_scoped():
    calls = []

    def search(query):
        calls.append(query)
        role = query.split()[-1].split(":")[0]
        return {
            "author": [pr(7), pr(2)],
            "commenter": [pr(8), pr(7)],
            "reviewed-by": [pr(9)],
            "review-requested": [pr(10), {"number": 99}],
        }[role]

    assert my_stuff.candidates("octocat", ("org/repo",), search) == [
        "10",
        "9",
        "8",
        "7",
        "2",
    ]
    assert set(calls) == {
        f"is:pr repo:org/repo {role}:octocat" for role in my_stuff.ROLES
    }


def test_cache_is_per_user_and_repository_and_expires(monkeypatch):
    clock = [0]
    monkeypatch.setattr(my_stuff.time, "monotonic", lambda: clock[0])
    calls = []

    def search(query):
        calls.append(query)
        return [pr(1)]

    result = my_stuff.candidates("alice", ("org/repo",), search)
    result.append("999")
    assert my_stuff.candidates("alice", ("org/repo",), search) == ["1"]
    assert len(calls) == 4
    my_stuff.candidates("bob", ("org/repo",), search)
    my_stuff.candidates("alice", ("other/repo",), search)
    assert len(calls) == 12
    clock[0] = my_stuff.CACHE_SECONDS + 1
    my_stuff.candidates("alice", ("org/repo",), search)
    assert len(calls) == 16


def test_failed_search_is_not_cached():
    def fail(query):
        raise OSError("upstream unavailable")

    with pytest.raises(OSError):
        my_stuff.candidates("alice", ("org/repo",), fail)
    assert my_stuff.candidates("alice", ("org/repo",), lambda q: [pr(5)]) == ["5"]


def test_concurrent_requests_share_the_lookup():
    started, release = threading.Event(), threading.Event()
    calls = []

    def search(query):
        calls.append(query)
        started.set()
        assert release.wait(5)
        return [pr(3)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        first = pool.submit(my_stuff.candidates, "alice", ("org/repo",), search)
        assert started.wait(5)
        others = [
            pool.submit(my_stuff.candidates, "alice", ("org/repo",), search)
            for _ in range(7)
        ]
        release.set()
        assert first.result() == ["3"]
        assert all(f.result() == ["3"] for f in others)
    assert len(calls) == 4


@pytest.mark.parametrize("login", ["", "bob is:issue", "alice@example.com"])
def test_invalid_identity_never_reaches_search(login):
    with pytest.raises(ValueError):
        my_stuff.candidates(login, ("org/repo",), lambda q: pytest.fail(q))


def test_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(my_stuff, "MAX_USERS", 2)
    for login in ("alice", "bob", "carol"):
        my_stuff.candidates(login, ("org/repo",), lambda q: [pr(1)])
    assert len(my_stuff._cache) == 2
    assert ("alice", ("org/repo",)) not in my_stuff._cache


def test_endpoint_requires_session_and_ignores_supplied_identity(monkeypatch):
    monkeypatch.setattr(
        trace_exporter.wdyt, "config", lambda: {"grafana": "http://grafana"}
    )
    monkeypatch.setattr(
        trace_exporter.wdyt,
        "grafana_login",
        lambda url, cookies: "alice" if cookies == "grafana_session=valid" else None,
    )
    monkeypatch.setattr(trace_exporter, "public_repositories", lambda: ("org/repo",))
    calls = []

    def search(query, **options):
        calls.append((query, options))
        return [pr(12)]

    monkeypatch.setattr(trace_exporter, "search_github_issues", search)
    server = ThreadingHTTPServer(("127.0.0.1", 0), trace_exporter.MetricsHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/pr-search/mine?login=mallory"
        with pytest.raises(HTTPError) as error:
            urlopen(url, timeout=5)
        assert error.value.code == 401
        assert not calls
        request = Request(url, headers={"Cookie": "grafana_session=valid"})
        with urlopen(request, timeout=5) as response:
            assert response.headers["Cache-Control"] == "no-store"
            assert json.load(response) == {"login": "alice", "prs": ["12"]}
        assert all(query.endswith(":alice") for query, _ in calls)
        assert all(options == {"sort": "updated", "limit": 100} for _, options in calls)
    finally:
        server.shutdown()
        server.server_close()
