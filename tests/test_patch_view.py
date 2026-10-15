"""PR diff retrieval must preserve bytes, bound size, and reject invalid paths."""

from io import BytesIO
from http.server import ThreadingHTTPServer
import json
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from transformersci.otel import patch_view


@pytest.fixture(autouse=True)
def clear_cache():
    patch_view._cache.clear()
    patch_view._comments_cache.clear()


@pytest.mark.parametrize(
    "user, expected",
    [
        ({"login": "HuggingFaceDocBuilderDev", "type": "User"}, True),
        ({"login": "huggingfacedocbuilderdev"}, True),
        ({"login": "github-actions[bot]"}, True),
        ({"login": "automation", "type": "Bot"}, True),
        ({"login": "human", "type": "User"}, False),
    ],
)
def test_bot_classification(user, expected):
    assert patch_view.is_bot_user(user) is expected


def test_diff_accept_header_and_cache(monkeypatch):
    calls = []
    diff = b"diff --git a/a.py b/a.py\n@@ -1 +1 @@\n-old\n+new\n"

    def fetch(request, timeout):
        calls.append(request)
        assert timeout == 30
        return BytesIO(diff)

    monkeypatch.setattr(patch_view, "urlopen", fetch)
    assert patch_view.fetch_diff("huggingface/transformers", "123") == diff.decode()
    assert patch_view.fetch_diff("huggingface/transformers", "123") == diff.decode()
    assert len(calls) == 1
    assert calls[0].get_header("Accept") == "application/vnd.github.diff"


@pytest.mark.parametrize(
    "repository, pr",
    [("../foo", "1"), ("owner/repo", "0"), ("owner/repo", "1/../../issues")],
)
def test_invalid_input_does_not_call_github(repository, pr, monkeypatch):
    def fetch(*args, **kwargs):
        pytest.fail("Invalid input must not make an upstream request")

    monkeypatch.setattr(patch_view, "urlopen", fetch)
    with pytest.raises(ValueError):
        patch_view.fetch_diff(repository, pr)


def test_large_diff_is_rejected_not_silently_truncated(monkeypatch):
    monkeypatch.setattr(patch_view, "MAX_BYTES", 8)
    monkeypatch.setattr(patch_view, "urlopen", lambda *a, **kw: BytesIO(b"123456789"))
    with pytest.raises(ValueError, match="too large"):
        patch_view.fetch_diff("owner/repo", "1")
    assert not patch_view._cache


def test_too_large_diff_is_rebuilt_from_file_patches(monkeypatch):
    files = [
        {
            "filename": "a.py",
            "status": "modified",
            "changes": 2,
            "patch": "@@ -1 +1 @@\n-old\n+new",
        },
        {
            "filename": "new.py",
            "previous_filename": "old.py",
            "status": "renamed",
            "changes": 0,
        },
        {
            "filename": "big.json",
            "status": "added",
            "changes": 7000,
            "additions": 7000,
            "deletions": 0,
        },
        {
            "filename": "gone.py",
            "status": "removed",
            "changes": 1,
            "patch": "@@ -1 +0,0 @@\n-x",
        },
    ]
    urls = []

    def fetch(request, timeout):
        urls.append(request.full_url)
        if request.get_header("Accept") == "application/vnd.github.diff":
            raise HTTPError(request.full_url, 406, "too_large", {}, None)
        return BytesIO(json.dumps(files).encode())

    monkeypatch.setattr(patch_view, "urlopen", fetch)
    diff = patch_view.fetch_diff("owner/repo", "7")
    assert urls[1].endswith("/pulls/7/files?per_page=100&page=1")
    assert len(urls) == 2
    assert diff == (
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n"
        "diff --git a/old.py b/new.py\n--- a/old.py\n+++ b/new.py\n"
        "diff --git a/big.json b/big.json\n--- /dev/null\n+++ b/big.json\n"
        "GitHub does not show this file's diff (7000 additions, 0 deletions). "
        "Open it on GitHub.\n"
        "diff --git a/gone.py b/gone.py\n--- a/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n"
    )


def test_file_listing_beyond_github_cap_is_rejected(monkeypatch):
    monkeypatch.setattr(patch_view, "MAX_FILE_PAGES", 2)
    page = [
        {
            "filename": f"f{i}.py",
            "status": "added",
            "changes": 1,
            "patch": "@@ -0,0 +1 @@\n+x",
        }
        for i in range(100)
    ]

    def fetch(request, timeout):
        if request.get_header("Accept") == "application/vnd.github.diff":
            raise HTTPError(request.full_url, 406, "too_large", {}, None)
        return BytesIO(json.dumps(page).encode())

    monkeypatch.setattr(patch_view, "urlopen", fetch)
    with pytest.raises(ValueError, match="too many files"):
        patch_view.fetch_diff("owner/repo", "1")
    assert not patch_view._cache


def test_other_diff_errors_are_not_masked(monkeypatch):
    def fetch(request, timeout):
        raise HTTPError(request.full_url, 404, "missing", {}, None)

    monkeypatch.setattr(patch_view, "urlopen", fetch)
    with pytest.raises(HTTPError):
        patch_view.fetch_diff("owner/repo", "1")


def test_comments_merge_paginate_and_cache(monkeypatch):
    calls = []
    discussion = {
        "id": 1,
        "author_association": "MEMBER",
        "body": "Discussion",
        "created_at": "2026-01-01T00:00:00Z",
        "user": {"login": "alice"},
    }
    inline = {
        "id": 2,
        "body": "Fix this line",
        "created_at": "2026-01-03T00:00:00Z",
        "path": "a.py",
        "line": None,
        "original_line": 7,
        "in_reply_to_id": 3,
    }
    review = {
        "id": 3,
        "user": {"login": "automation", "type": "Bot"},
        "body": "Review summary",
        "submitted_at": "2026-01-02T00:00:00Z",
        "state": "CHANGES_REQUESTED",
    }

    def fetch(request, timeout):
        calls.append(request.full_url)
        assert request.get_header("Accept") == "application/vnd.github+json"
        if "/issues/" in request.full_url:
            batch = [discussion] * 100 if request.full_url.endswith("&page=1") else []
        elif "/reviews?" in request.full_url:
            batch = [review, {"id": 4, "body": ""}]
        else:
            batch = [inline]
        return BytesIO(json.dumps(batch).encode())

    monkeypatch.setattr(patch_view, "urlopen", fetch)
    result = patch_view.fetch_comments("owner/repo", "123")
    assert len(calls) == 4

    assert len(result["comments"]) == 102
    assert result["comments"][0]["is_maintainer"]
    assert result["comments"][0]["author_association"] == "MEMBER"
    assert not result["comments"][-1]["is_maintainer"]
    assert not result["truncated"]
    assert result["comments"][-2]["kind"] == "review"
    assert result["comments"][-2]["is_bot"]
    assert not result["comments"][-1]["is_bot"]
    assert result["comments"][-1]["kind"] == "inline"
    assert result["comments"][-1]["outdated"]
    assert result["comments"][-1]["line"] == 7
    assert result["comments"][-1]["reply_to"] == 3
    assert patch_view.fetch_comments("owner/repo", "123") == result
    assert len(calls) == 4


def test_thread_resolution_and_location_apply_to_replies():
    root = {
        "id": 1,
        "kind": "inline",
        "path": "a.py",
        "line": 12,
        "start_line": 10,
        "side": "LEFT",
        "diff_hunk": "@@ -10,3 +10,3 @@\n old\n-old\n old",
    }
    reply = {"id": 2, "kind": "inline", "reply_to": 1}
    unknown = {"id": 3, "kind": "inline"}
    threads = [
        {
            "id": "thread1",
            "isResolved": True,
            "isOutdated": True,
            "comments": {"nodes": [{"databaseId": 1}]},
        }
    ]
    patch_view.enrich_threads([root, reply, unknown], threads)
    assert root["resolved"] is True
    assert reply["resolved"] is True
    assert reply["outdated"] is True
    assert reply["line"] == 12 and reply["start_line"] == 10
    assert reply["side"] == "LEFT"
    assert reply["diff_hunk"] == root["diff_hunk"]
    assert unknown["resolved"] is None


def test_graphql_thread_pagination(monkeypatch):
    cursors = []

    def fetch(request, timeout):
        variables = json.loads(request.data)["variables"]
        cursors.append(variables["endCursor"])
        page = len(cursors)
        result = {
            "data": {
                "repository": {
                    "pullRequest": {
                        "reviewThreads": {
                            "nodes": [{"id": str(page)}],
                            "pageInfo": {"hasNextPage": page == 1, "endCursor": "next"},
                        }
                    }
                }
            }
        }
        return BytesIO(json.dumps(result).encode())

    monkeypatch.setattr(patch_view, "urlopen", fetch)
    assert patch_view.fetch_threads("owner/repo", "1", "token") == [
        {"id": "1"},
        {"id": "2"},
    ]
    assert cursors == [None, "next"]


def test_http_routes_serve_assets_and_enforce_public_repository(monkeypatch):
    from transformersci.otel import trace_exporter

    calls = []
    monkeypatch.setattr(trace_exporter, "public_repositories", lambda: ("owner/repo",))
    monkeypatch.setattr(trace_exporter, "github_api_token", lambda: "test-token")

    def fetch_diff(repository, pr, token):
        calls.append((repository, pr, token))
        patch_view._validate(repository, pr)
        return "diff --git a/a.py b/a.py\n-old\n+new\n"

    monkeypatch.setattr(patch_view, "fetch_diff", fetch_diff)
    monkeypatch.setattr(patch_view, "fetch_comments", lambda *args: {"comments": []})
    server = ThreadingHTTPServer(("127.0.0.1", 0), trace_exporter.MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        for path, content_type in (
            ("/patch-view", "text/html"),
            ("/patch-view/script.js", "text/javascript"),
            ("/patch-view/pr-view.js", "text/javascript"),
            ("/patch-view/data?repository=owner/repo&pr=12", "text/plain"),
            ("/patch-view/comments?repository=owner/repo&pr=12", "application/json"),
        ):
            with urlopen(base + path, timeout=5) as response:
                assert response.status == 200
                assert response.headers["Content-Type"].startswith(content_type)
                assert response.headers["Cache-Control"] == "no-store"
                assert response.read()
        assert calls == [("owner/repo", "12", "test-token")]
        for route in ("data", "comments"):
            with pytest.raises(HTTPError) as exc:
                urlopen(
                    base + f"/patch-view/{route}?repository=private/repo&pr=12",
                    timeout=5,
                )
            assert exc.value.code == 400
        assert calls == [("owner/repo", "12", "test-token")]
        with pytest.raises(HTTPError) as exc:
            urlopen(base + "/patch-view/data?repository=owner/repo&pr=0", timeout=5)
        assert exc.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
