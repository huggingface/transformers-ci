from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from transformersci.otel import claims, new_issue, related_issues, trace_exporter, wdyt

NODEID = "tests/models/bert/test_modeling_bert.py::BertModelTest::test_fa2"
TRACE = "a" * 32


def test_first_drafter_holds_the_test_and_others_are_told_who() -> None:
    store = claims.Store(":memory:")
    assert store.reserve(NODEID, "alice", now=100) == {"status": "ok"}
    assert store.reserve(NODEID, "bob", now=110) == {
        "status": "drafting",
        "login": "alice",
        "since": 100,
    }
    # The same person can come back (a second tab, or publishing later).
    assert store.reserve(NODEID, "alice", now=120) == {"status": "ok"}
    assert store.snapshot(now=130)[NODEID]["created"] == 100


def test_an_abandoned_draft_lapses() -> None:
    store = claims.Store(":memory:")
    store.reserve(NODEID, "alice", now=100)
    later = 100 + claims.RESERVE_SECONDS + 1
    assert store.snapshot(now=later) == {}
    assert store.reserve(NODEID, "bob", now=later) == {"status": "ok"}


def test_a_published_claim_blocks_everyone_and_never_lapses() -> None:
    store = claims.Store(":memory:")
    store.reserve(NODEID, "alice", now=100)
    url = "https://github.com/huggingface/transformers/issues/7"
    store.confirm(NODEID, "alice", 7, url, now=200)
    for login in ("bob", "alice"):
        assert store.reserve(NODEID, login, now=10**9) == {
            "status": "claimed",
            "login": "alice",
            "number": 7,
            "url": url,
        }
    store.release(NODEID, "alice")
    assert store.snapshot(now=10**9)[NODEID]["state"] == "claimed"


def test_release_only_drops_your_own_draft() -> None:
    store = claims.Store(":memory:")
    store.reserve(NODEID, "alice", now=100)
    store.release(NODEID, "bob")
    assert store.snapshot(now=101)[NODEID]["login"] == "alice"
    store.release(NODEID, "alice")
    assert store.snapshot(now=101) == {}


def test_concurrent_new_issue_clicks_reserve_exactly_once(tmp_path) -> None:
    store = claims.Store(str(tmp_path / "claims.sqlite3"))
    barrier = threading.Barrier(16)
    results: list[dict] = []

    def click(login: str) -> None:
        barrier.wait()
        results.append(store.reserve(NODEID, login))

    threads = [threading.Thread(target=click, args=(f"user{i}",)) for i in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(result["status"] == "ok" for result in results) == 1
    holder = store.snapshot()[NODEID]["login"]
    assert all(r["login"] == holder for r in results if r["status"] != "ok")


def test_a_closed_issue_frees_the_test_and_a_reopened_one_holds_it_again() -> None:
    store = claims.Store(":memory:")
    url = "https://github.com/huggingface/transformers/issues/7"
    store.reserve(NODEID, "alice", now=100)
    store.confirm(NODEID, "alice", 7, url, now=200)
    # Read at confirm time, so not due until CHECK_SECONDS later.
    assert store.due(now=201) == []
    later = 200 + claims.CHECK_SECONDS + 1
    assert store.due(now=later) == [(NODEID, 7)]

    closed = {"state": "closed", "closed_at": "2026-10-17T11:12:17Z"}
    assert store.refresh(lambda number: closed, now=later) == 1
    claim = store.snapshot(now=later)[NODEID]
    assert (claim["issue_state"], claim["closed_at"]) == ("closed", 1792235537.0)
    assert store.reserve(NODEID, "bob", now=later + 1) == {"status": "ok"}

    store.confirm(NODEID, "bob", 8, url.replace("7", "8"), now=later + 2)
    reopened = later + 2 + claims.CHECK_SECONDS + 1
    store.refresh(lambda number: {"state": "open"}, now=reopened)
    assert store.reserve(NODEID, "carol", now=reopened)["status"] == "claimed"


def test_refresh_skips_failed_reads_and_leaves_newer_claims_alone() -> None:
    store = claims.Store(":memory:")
    url = "https://github.com/huggingface/transformers/issues/7"
    store.confirm(NODEID, "alice", 7, url, now=0)

    def broken(number):
        raise OSError("GitHub down")

    assert store.refresh(broken, now=10**6) == 0
    assert store.snapshot(now=10**6)[NODEID]["issue_state"] == "open"
    # An answer about issue 7 never closes a newer claim (issue 8) on the test.
    store.confirm(NODEID, "bob", 8, url.replace("7", "8"), now=1)
    store.set_issue_state(NODEID, 7, "closed", 5.0, now=2)
    assert store.snapshot(now=3)[NODEID]["issue_state"] == "open"


def test_an_existing_database_gains_the_issue_columns(tmp_path) -> None:
    import sqlite3

    path = str(tmp_path / "claims.sqlite3")
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE claims (nodeid TEXT PRIMARY KEY, login TEXT NOT NULL,"
        " state TEXT NOT NULL, created REAL NOT NULL, expires REAL,"
        " number INTEGER, url TEXT)"
    )
    db.execute(
        "INSERT INTO claims VALUES (?, 'tarekziade', 'claimed', 1.0, NULL, 49691, ?)",
        (NODEID, "https://github.com/huggingface/transformers/issues/49691"),
    )
    db.commit()
    db.close()
    store = claims.Store(path)
    assert store.due(now=10.0) == [(NODEID, 49691)]
    store.refresh(lambda number: {"state": "closed", "closed_at": None}, now=10.0)
    assert store.reserve(NODEID, "bob", now=11.0) == {"status": "ok"}


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(claims, "_store", claims.Store(":memory:"))
    monkeypatch.setattr(
        wdyt, "grafana_login", lambda config, cookie: cookie.removeprefix("user=")
    )
    monkeypatch.setattr(trace_exporter, "get_trace", lambda trace_id: {"spans": []})
    monkeypatch.setattr(
        trace_exporter, "trace_repository", lambda trace: related_issues.REPOSITORY
    )
    monkeypatch.setattr(
        trace_exporter,
        "extract_failure_details",
        lambda trace, nodeid: [{"exception_type": "AssertionError"}],
    )
    monkeypatch.setattr(trace_exporter, "issue_environment", lambda trace, nodeid: {})
    monkeypatch.setattr(trace_exporter, "_prometheus_vector", lambda query: [])
    monkeypatch.setattr(trace_exporter, "_project_health_cache", {})
    monkeypatch.setattr(
        new_issue,
        "prepare",
        lambda login, *args, **kwargs: new_issue._issue_draft(login, "[CI] t", "body"),
    )
    created: list[tuple[str, str]] = []

    def create_issue(title: str, body: str) -> dict:
        created.append((title, body))
        number = len(created)
        return {
            "number": number,
            "html_url": f"https://github.com/huggingface/transformers/issues/{number}",
        }

    monkeypatch.setattr(trace_exporter, "_github_create_issue", create_issue)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), trace_exporter.MetricsHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}", created
    httpd.shutdown()


def post(base: str, path: str, login: str, body: dict) -> tuple[int, dict]:
    request = Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "X-TCI-Action": "1",
            "Cookie": f"user={login}",
        },
        method="POST",
    )
    try:
        with urlopen(request) as response:
            text = response.read().decode()
            status = response.status
    except HTTPError as error:
        return error.code, json.loads(error.read())
    lines = [json.loads(line) for line in text.splitlines() if line.strip()]
    return status, lines[-1]


def test_draft_publish_and_release_keep_one_issue_per_test(server) -> None:
    base, created = server
    target = {"trace_id": TRACE, "test_nodeid": NODEID}

    status, draft = post(base, "/serge-actions/new-issue/draft", "alice", target)
    assert (status, draft["status"]) == (200, "ok")

    status, other = post(base, "/serge-actions/new-issue/draft", "bob", target)
    assert status == 409
    assert (other["status"], other["login"]) == ("drafting", "alice")

    # Rows carry the reservation, so the table shows who is drafting.
    rows = json.loads(urlopen(f"{base}/project-health/rows?days=7").read())
    assert rows["claims"][NODEID]["login"] == "alice"
    assert rows["claims"][NODEID]["state"] == "drafting"

    status, issue = post(
        base,
        "/serge-actions/new-issue/publish",
        "alice",
        {"draft_id": draft["draft_id"], "title": "[CI] t", "body": "edited"},
    )
    assert (status, issue["status"], issue["number"]) == (200, "ok", 1)
    assert created[0][1] == "edited\n\n" + trace_exporter.claim_line("alice")

    status, again = post(base, "/serge-actions/new-issue/draft", "bob", target)
    assert status == 409
    assert (again["status"], again["login"], again["number"]) == ("claimed", "alice", 1)


def test_closing_a_draft_frees_the_test(server) -> None:
    base, created = server
    target = {"trace_id": TRACE, "test_nodeid": NODEID}
    _, draft = post(base, "/serge-actions/new-issue/draft", "alice", target)
    status, _ = post(
        base,
        "/serge-actions/new-issue/release",
        "alice",
        {"draft_id": draft["draft_id"]},
    )
    assert status == 200
    status, mine = post(base, "/serge-actions/new-issue/draft", "bob", target)
    assert (status, mine["status"]) == (200, "ok")
    assert created == []


def test_a_lapsed_draft_taken_by_someone_else_cannot_be_published(server) -> None:
    base, created = server
    target = {"trace_id": TRACE, "test_nodeid": NODEID}
    _, draft = post(base, "/serge-actions/new-issue/draft", "alice", target)
    store = claims.store()
    store.release(NODEID, "alice")  # as if it lapsed
    assert store.reserve(NODEID, "bob")["status"] == "ok"
    status, refused = post(
        base,
        "/serge-actions/new-issue/publish",
        "alice",
        {"draft_id": draft["draft_id"], "title": "[CI] t", "body": "b"},
    )
    assert (status, refused["status"], refused["login"]) == (409, "drafting", "bob")
    assert created == []


def test_slow_new_issue_drafts_from_timings_and_reserves_the_test(server) -> None:
    base, created = server
    slow = {
        "kind": "slow",
        "test_nodeid": NODEID,
        "job": "run_models_gpu",
        "runner": "GPU",
        "first_day": "2026-10-15",
        "durations": [300, None, 4000],
    }
    status, draft = post(base, "/serge-actions/new-issue/draft", "alice", slow)
    assert (status, draft["status"]) == (200, "ok")
    assert draft["title"] == "[CI] test_fa2 is slow on main (35.8 min on GPU)"
    assert "| Oct 17 | 66.7 min |" in draft["body"]
    status, other = post(base, "/serge-actions/new-issue/draft", "bob", slow)
    assert (status, other["status"], other["login"]) == (409, "drafting", "alice")
    status, issue = post(
        base,
        "/serge-actions/new-issue/publish",
        "alice",
        {"draft_id": draft["draft_id"], "title": draft["title"], "body": draft["body"]},
    )
    assert issue["status"] == "ok"
    assert claims.store().snapshot()[NODEID]["number"] == 1


def test_slow_new_issue_rejects_bad_timings(server) -> None:
    base, _ = server
    bad = {
        "kind": "slow",
        "test_nodeid": NODEID,
        "job": "run_models_gpu",
        "runner": "GPU",
        "first_day": "2026-10-15",
    }
    for durations in ([], [None], ["300"], [True], [1e9]):
        status, reply = post(
            base,
            "/serge-actions/new-issue/draft",
            "alice",
            {**bad, "durations": durations},
        )
        assert (status, reply["status"]) == (400, "invalid")
    assert claims.store().snapshot() == {}


SLOW = {
    "kind": "slow",
    "test_nodeid": NODEID,
    "job": "run_models_gpu",
    "runner": "GPU",
    "first_day": "2026-10-15",
    "durations": [300, None, 4000],
}


def test_slow_wdyt_asks_with_the_test_source_and_the_slow_prompt(
    server, monkeypatch
) -> None:
    base, _ = server
    monkeypatch.setattr(
        wdyt,
        "config",
        lambda: {"api_base": "x", "model": "m", "api_key": "k", "grafana": ""},
    )
    monkeypatch.setattr(wdyt, "_cache", type(wdyt._cache)())
    monkeypatch.setattr(
        new_issue,
        "failing_test_source",
        lambda nodeid, details, commit, get: "def test_fa2(self): ...",
    )
    seen = {}

    def ask(cfg, prompt, emit, sid="", system=""):
        seen.update(prompt=prompt, system=system)
        return "The test looks fine; the slowness is probably in the model code."

    monkeypatch.setattr(wdyt, "ask", ask)
    status, answer = post(base, "/serge-actions/wdyt?kind=slow", "alice", SLOW)
    assert (status, answer["status"]) == (200, "ok")
    assert seen["system"] == wdyt.SLOW_SYSTEM_PROMPT
    assert "<test_source>\ndef test_fa2(self): ...\n</test_source>" in seen["prompt"]
    assert "300s, -, 4000s" in seen["prompt"]
    # A failure WDYT? without a trace is still refused.
    status, refused = post(base, "/serge-actions/wdyt", "alice", SLOW)
    assert (status, refused["status"]) == (400, "invalid")


def test_slow_who_can_help_looks_at_the_test_without_a_trace(
    server, monkeypatch
) -> None:
    base, _ = server
    asked = {}

    def suggest(login, trace_id, nodeid, details, github_get=None, emit=None):
        asked.update(login=login, trace_id=trace_id, nodeid=nodeid, details=details)
        return {"status": "ok", "people": [], "location": "tests/x.py:1"}

    monkeypatch.setattr(trace_exporter.who_can_help, "suggest", suggest)
    status, result = post(base, "/serge-actions/who-can-help?kind=slow", "bob", SLOW)
    assert (status, result["status"]) == (200, "ok")
    assert asked == {
        "login": "bob",
        "trace_id": "slow",
        "nodeid": NODEID,
        "details": [],
    }
    status, refused = post(
        base, "/serge-actions/who-can-help?kind=slow", "bob", {**SLOW, "runner": "TPU"}
    )
    assert (status, refused["status"]) == (400, "invalid")
