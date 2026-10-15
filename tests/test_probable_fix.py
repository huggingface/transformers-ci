"""A probable fix needs a live PR that changes the selected test file."""

from __future__ import annotations

import io
import json

from transformersci.otel import probable_fix as pf
from transformersci.otel import related_issues as ri

NODE = "tests/generation/test_streamers.py::StreamerTest"
TRACE = "f8a67fa26dd021980a0457850169694a"


def setup_function():
    pf._cache.clear()
    pf._inflight.clear()


def pr(number: int, title: str, state: str = "open") -> dict:
    return {
        "number": number,
        "title": title,
        "state": state,
        "pull_request": {"merged_at": None},
    }


def test_recent_pr_without_closing_issue_can_be_found(monkeypatch):
    monkeypatch.setattr(ri, "lookup", lambda *args: {"status": "ok", "hits": []})
    queries, files = [], []

    def search(query, *, limit):
        queries.append((query, limit))
        return [pr(49583, "Fix flaky streamer timeout tests")]

    def get(url):
        files.append(url)
        return [
            {"filename": "tests/generation/test_streamers.py"},
            {"filename": "docs/source/en/fsdp.md"},
        ]

    result = pf.find(TRACE, NODE, list, search, get)
    assert result == {
        "status": "ok",
        "pr": {
            "number": 49583,
            "title": "Fix flaky streamer timeout tests",
            "url": "https://github.com/huggingface/transformers/pull/49583",
            "state": "open",
            "issue": None,
        },
    }
    assert "in:title streamer" in queries[0][0]
    assert files[0].endswith("/pulls/49583/files?per_page=100")
    assert pf.find(TRACE, NODE, list, search, get) == result
    assert len(queries) == 1


def test_inflight_claim_is_checked_against_changed_file(monkeypatch):
    monkeypatch.setattr(
        ri,
        "lookup",
        lambda *args: {"status": "ok", "hits": [{"number": 48, "type": "issue"}]},
    )
    calls = []
    monkeypatch.setattr(
        pf,
        "inflight",
        lambda number: (
            calls.append(number)
            or [
                {
                    "repo": ri.REPOSITORY,
                    "type": "pr",
                    "number": 50,
                    "title": "Fix the queue race",
                    "state": "open",
                    "merged": False,
                }
            ]
        ),
    )
    result = pf.find(
        TRACE,
        NODE,
        list,
        lambda *args, **kwargs: [],
        lambda url: [{"filename": "tests/generation/test_streamers.py"}],
    )
    assert calls == [48]
    assert result["pr"]["issue"] == 48
    assert result["pr"]["number"] == 50


def test_inflight_survives_github_search_failure(monkeypatch):
    monkeypatch.setattr(
        ri,
        "lookup",
        lambda *args: {"status": "ok", "hits": [{"number": 48, "type": "issue"}]},
    )
    monkeypatch.setattr(
        pf,
        "inflight",
        lambda number: [
            {
                "repo": ri.REPOSITORY,
                "type": "pr",
                "number": 50,
                "title": "Fix the queue race",
                "state": "open",
                "merged": False,
            }
        ],
    )

    def failed_search(*args, **kwargs):
        raise OSError("search down")

    result = pf.find(
        TRACE,
        NODE,
        list,
        failed_search,
        lambda url: [{"filename": "tests/generation/test_streamers.py"}],
    )
    assert result["pr"]["number"] == 50


def test_unrelated_file_and_closed_pr_are_not_probable(monkeypatch):
    monkeypatch.setattr(ri, "lookup", lambda *args: {"status": "ok", "hits": []})
    result = pf.find(
        TRACE,
        NODE,
        list,
        lambda *args, **kwargs: [
            pr(1, "Fix streamer tests", "closed"),
            pr(2, "Fix streamer tests"),
        ],
        lambda url: [{"filename": "tests/generation/test_beam_search.py"}],
    )
    assert result == {"status": "none"}


def test_inflight_uses_repo_and_version(monkeypatch):
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_RELORE_URL", "https://relore.test")

    def open_request(request, timeout):
        assert timeout == 8
        assert request.get_header("X-relore-client") == ri.WIRE_VERSION
        assert request.full_url.endswith(
            "/api/v1/inflight/48?repo=huggingface%2Ftransformers"
        )
        response = io.BytesIO(
            json.dumps(
                {"repo": ri.REPOSITORY, "number": 48, "claims": [{"number": 50}]}
            ).encode()
        )
        response.headers = {"x-relore-version": ri.WIRE_VERSION}
        return response

    monkeypatch.setattr(pf, "urlopen", open_request)
    assert pf.inflight(48) == [{"number": 50}]
