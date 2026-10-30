import io
import json
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError

from transformersci.otel import slack_pr, slack_commands


def test_pr_pages_all_checks_and_preserves_head_and_metadata(monkeypatch):
    seen = []
    monkeypatch.setattr(
        slack_pr, "brief", lambda pr: "Relore: reviewer history and competing work"
    )

    def github(path):
        seen.append(path)
        if path == "pulls/50071":
            return {
                "title": "Fix <@everyone>",
                "state": "open",
                "user": {"login": "author"},
                "head": {"sha": "a" * 40, "ref": "fix"},
                "base": {"ref": "main"},
                "created_at": (
                    datetime.now(timezone.utc) - timedelta(days=3)
                ).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "changed_files": 2,
                "additions": 10,
                "deletions": 3,
            }
        if "check-runs" in path:
            return {
                "total_count": 101,
                "check_runs": [
                    {
                        "name": "GPU failure",
                        "status": "completed",
                        "conclusion": "failure",
                    }
                ]
                if "page=2" in path
                else [{"status": "completed", "conclusion": "success"}] * 100,
            }
        return {
            "total_count": 1,
            "statuses": [{"context": "external", "state": "pending"}],
        }

    result = slack_commands.pr_status("50071", github, "https://dashboard")
    assert (
        "https://dashboard/d/pytest-observability-by-pr/branch?var-pr=50071&from=now-90d&to=now&timezone=browser&refresh=10s"
        in result
    )
    assert "Fix &lt;@everyone&gt;" in result
    assert "@author" in result and "opened 3d" in result
    assert "2 files, +10 / −3" in result
    assert "100 passed, 1 failed, 1 pending" in result
    assert "GPU failure" in result and "Relore:" in result
    assert len(seen) == 4 and all("a" * 40 in p for p in seen[1:])


def test_checks_cap_is_explicit_and_status_contexts_deduplicated(monkeypatch):
    monkeypatch.setattr(slack_pr, "MAX_PAGES", 2)

    def github(path):
        if "check-runs" in path:
            return {
                "total_count": 300,
                "check_runs": [{"status": "completed", "conclusion": "success"}] * 100,
            }
        return {
            "total_count": 2,
            "statuses": [
                {"context": "ci", "state": "success"},
                {"context": "ci", "state": "failure"},
            ],
        }

    result = slack_pr.checks("a" * 40, github)
    assert "201 passed, 0 failed" in result
    assert "partial" in result


class Response(io.BytesIO):
    def __init__(self, data, version="0.3.19"):
        super().__init__(json.dumps(data).encode())
        self.headers = {"x-relore-version": version}


def test_full_relore_brief_and_supported_version_retry(monkeypatch):
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_RELORE_URL", "http://relore")
    calls = []

    def open_(request, timeout):
        version = request.get_header("X-relore-client")
        calls.append(version)
        assert "compact=false" in request.full_url
        if version != "0.3.19":
            raise HTTPError(
                request.full_url, 426, "version", {"x-relore-version": "0.3.19"}, None
            )
        return Response(
            {
                "thread": {
                    "repo": "huggingface/transformers",
                    "number": 42,
                    "type": "pr",
                    "indexed_at": "today",
                },
                "people": {},
                "same_error": {"error": "<@everyone>", "threads": []},
            }
        )

    monkeypatch.setattr(slack_pr, "urlopen", open_)
    result = slack_pr.brief("42")
    assert "History on these files" in result and "Same error" in result
    assert "&lt;@everyone&gt;" in result and "indexed today" in result
    assert calls == ["0.3.18", "0.3.19"]


def test_relore_missing_or_incompatible_is_not_empty_evidence(monkeypatch):
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_RELORE_URL", "http://relore")
    for code, version, expected in [
        (404, "0.3.18", "not indexed"),
        (426, "99.0.0", "temporarily unavailable"),
    ]:
        calls = []

        def open_(*args, **kwargs):
            calls.append(1)
            raise HTTPError(
                "http://relore", code, "error", {"x-relore-version": version}, None
            )

        monkeypatch.setattr(slack_pr, "urlopen", open_)
        assert expected in slack_pr.brief("42")
        assert len(calls) == 1


def test_relore_rejects_wrong_pr_and_marks_truncation(monkeypatch):
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_RELORE_URL", "http://relore")
    data = {
        "thread": {"repo": "huggingface/transformers", "number": 43, "type": "pr"},
        "people": {},
        "same_error": {"error": "x" * 30000, "threads": []},
    }
    monkeypatch.setattr(slack_pr, "urlopen", lambda *a, **k: Response(data, "0.3.18"))
    assert "unavailable" in slack_pr.brief("42")
    data["thread"]["number"] = 42
    assert "truncated for Slack" in slack_pr.brief("42")


def test_human_brief_preserves_evidence_links_and_coverage():
    from transformersci.otel.slack_brief import render

    data = {
        "thread": {
            "review_decision": "changes_requested",
            "review_decision_by": ["reviewer"],
            "requested_reviewers": ["next"],
            "indexed_at": "2026-10-30",
            "files_total": 624,
            "files_collected": 100,
        },
        "opener": {
            "author": "author",
            "trust": "authoritative",
            "prs": 203,
            "merged": 157,
            "issues": 19,
        },
        "people": {
            "total": 2,
            "authoritative": 1,
            "listed": [{"author": "reviewer", "comments": 2, "trust": "authoritative"}],
            "machine_comments": 3,
        },
        "closes": [{"number": 1}],
        "closes_total": 3,
        "competing": {"links_indexed": 0},
        "file_history": {
            "paths_used": ["a.py"],
            "paths_total": 100,
            "basis": "changed",
            "untested": 42,
            "threads": [
                {
                    "number": 2,
                    "title": "Prior fix <@everyone>\n*Review:* approved",
                    "url": "https://github.com/huggingface/transformers/pull/2",
                    "state": "closed",
                    "merged": True,
                    "evidence": {
                        "author": "reviewer",
                        "source_type": "review_comment",
                        "snippet": "Do not remove this",
                        "url": "https://github.com/huggingface/transformers/pull/2#discussion_r1",
                    },
                }
            ],
        },
        "same_error": {"error": "ValueError", "threads": []},
        "similar": {"threads": []},
    }
    result = render(data)
    assert (
        "Changes requested" in result and "203 other indexed PRs, 157 merged" in result
    )
    assert "github.com/reviewer|@reviewer" in result and "write access" in result
    assert (
        "1 more participants not shown" in result
        and "3 automated comments omitted" in result
    )
    assert (
        "#1 — not indexed" in result and "2 more closing links not looked up" in result
    )
    assert "Unknown: this repository has no indexed relationship data" in result
    assert (
        "1 of 100 indexed changed paths" in result
        and "42 PRs have no collected file list" in result
    )
    assert "100 of this PR's 624 changed files" in result
    assert (
        "#discussion_r1|review comment>" in result
        and "> “Do not remove this”" in result
    )
    assert "&lt;@everyone&gt; ∗Review:∗ approved" in result
    assert "```" not in result and "RELORE-UNTRUSTED" not in result
    assert "may lag GitHub" in result and "not exhaustive history" in result
