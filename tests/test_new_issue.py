"""Issue drafts use the repository template and need a signed-in publisher."""

from __future__ import annotations

import base64
import json

from transformersci.otel import new_issue
from transformersci.otel import trace_exporter as te

NODE = "tests/generation/test_streamers.py::StreamerTest::test_stream"
TRACE = "a" * 32
TEMPLATE = """body:
  - type: textarea
    id: system-info
    attributes:
      label: System Info
    validations:
      required: true
  - type: checkboxes
    id: information
    attributes:
      label: Information
      options:
        - label: Official examples
  - type: textarea
    id: reproduction
    attributes:
      label: Reproduction
    validations:
      required: true
  - type: textarea
    id: expected-behavior
    attributes:
      label: Expected behavior
    validations:
      required: true
"""


def setup_function():
    new_issue._drafts.clear()
    new_issue._draft_cache.clear()
    new_issue._asked.clear()


def test_template_and_draft_keep_required_sections_and_trace(monkeypatch):
    urls = []

    def get(url):
        urls.append(url)
        return {
            "encoding": "base64",
            "content": base64.b64encode(TEMPLATE.encode()).decode(),
        }

    fields = new_issue.bug_template(get)
    assert [field["id"] for field in fields] == [
        "system-info",
        "information",
        "reproduction",
        "expected-behavior",
    ]
    assert urls == [
        "https://api.github.com/repos/huggingface/transformers/contents/.github/ISSUE_TEMPLATE/bug-report.yml"
    ]
    monkeypatch.setattr(
        new_issue.who_can_help,
        "source_line",
        lambda *args: ("tests/generation/test_streamers.py", 10),
    )
    monkeypatch.setattr(new_issue.who_can_help, "why", lambda *args: {"number": 42})
    monkeypatch.setattr(new_issue.related_issues, "lookup", lambda *args: {"hits": []})
    helper = {
        "login": "maintainer",
        "why": "Reviewed the affected line",
        "evidence": "PR #7",
        "url": "https://github.com/huggingface/transformers/pull/7",
    }
    monkeypatch.setattr(
        new_issue.who_can_help,
        "suggest",
        lambda *args: {"status": "ok", "people": [helper]},
    )
    captured = []
    monkeypatch.setattr(
        new_issue,
        "model_draft",
        lambda *args: captured.append(args[6])
        or {
            "title": "Streamer fails in CI",
            "sections": {
                "system-info": "",
                "reproduction": "Observed in CI",
                "expected-behavior": "Should pass",
            },
        },
    )
    events = []
    draft = new_issue.prepare(
        "alice",
        TRACE,
        NODE,
        [{"exception_message": "bad stream"}],
        get,
        emit=events.append,
    )
    assert draft["status"] == "ok"
    assert captured == [[helper]]
    assert [event["text"] for event in events if event["event"] == "step"] == [
        "Reading the Transformers bug report template",
        "Locating the failing test in the repository",
        "Reading Relore history for the failing line",
        "Searching related issues and PRs",
        "Checking recent changes to the test file",
        "Finding people who can help (slow)",
        "Asking the model to draft the report (slow)",
        "Preparing the editable issue form",
    ]
    assert draft["title"] == "Streamer fails in CI"
    assert "### System Info" in draft["body"]
    assert "### Reproduction" in draft["body"]
    assert "### Expected behavior" in draft["body"]
    assert "- [ ] Official examples" in draft["body"]
    assert "bad stream" in draft["body"] and "var-trace_id=" + TRACE in draft["body"]


def test_cached_draft_reuses_model_output_with_distinct_publish_ids(monkeypatch):
    calls = []
    monkeypatch.setattr(
        new_issue,
        "bug_template",
        lambda *args: [
            {
                "id": "reproduction",
                "type": "textarea",
                "label": "Reproduction",
                "required": True,
            }
        ],
    )
    monkeypatch.setattr(new_issue.who_can_help, "source_line", lambda *args: None)
    monkeypatch.setattr(
        new_issue.who_can_help, "frequent_test_modifier", lambda *args: []
    )
    monkeypatch.setattr(new_issue.related_issues, "lookup", lambda *args: {"hits": []})
    monkeypatch.setattr(
        new_issue.who_can_help, "suggest", lambda *args: {"status": "no_people"}
    )

    def draft(*args):
        calls.append(1)
        return {"title": "Failure", "sections": {"reproduction": "Steps"}}

    monkeypatch.setattr(new_issue, "model_draft", draft)
    first = new_issue.prepare("alice", TRACE, NODE, [{}], lambda url: {})
    events = []
    second = new_issue.prepare("bob", TRACE, NODE, [{}], lambda url: {}, events.append)
    assert first["status"] == second["status"] == "ok"
    assert first["draft_id"] != second["draft_id"]
    assert first["body"] == second["body"]
    assert calls == [1]
    assert events == [
        {"event": "step", "text": "Using an earlier draft for this failure"}
    ]
    assert new_issue._drafts[first["draft_id"]][1] == "alice"
    assert new_issue._drafts[second["draft_id"]][1] == "bob"


def test_draft_uses_recent_test_modifier_when_relore_has_no_people(monkeypatch):
    monkeypatch.setattr(
        new_issue,
        "bug_template",
        lambda *args: [
            {
                "id": "reproduction",
                "type": "textarea",
                "label": "Reproduction",
                "required": True,
            }
        ],
    )
    monkeypatch.setattr(new_issue.who_can_help, "source_line", lambda *args: None)
    monkeypatch.setattr(new_issue.related_issues, "lookup", lambda *args: {"hits": []})
    monkeypatch.setattr(
        new_issue.who_can_help, "suggest", lambda *args: {"status": "no_people"}
    )
    monkeypatch.setattr(
        new_issue.who_can_help,
        "frequent_test_modifier",
        lambda *args: [
            {
                "login": "alice",
                "evidence": "recent edits",
                "url": "https://github.com/x",
            }
        ],
    )
    seen = []

    def draft(*args):
        seen.extend(args[5])
        return {"title": "Failure", "sections": {"reproduction": "Steps"}}

    monkeypatch.setattr(new_issue, "model_draft", draft)
    result = new_issue.prepare("alice", TRACE, NODE, [{}], lambda url: {})
    assert result["status"] == "ok"
    assert seen[0]["login"] == "alice"


def test_publish_uses_edited_text_once_and_checks_login():
    draft_id = "f" * 32
    new_issue._drafts[draft_id] = (float("inf"), "alice")
    calls = []

    def create(title, body):
        calls.append((title, body))
        return {
            "number": 123,
            "html_url": "https://github.com/huggingface/transformers/issues/123",
        }

    assert new_issue.publish("bob", draft_id, "Edited", "Edited body", create) == {
        "status": "invalid"
    }
    assert new_issue.publish("alice", draft_id, "Edited", "Edited body", create) == {
        "status": "ok",
        "number": 123,
        "url": "https://github.com/huggingface/transformers/issues/123",
    }
    assert calls == [("Edited", "Edited body")]
    assert new_issue.publish("alice", draft_id, "Again", "Again", create) == {
        "status": "invalid"
    }


def test_failed_publish_keeps_draft_for_retry():
    draft_id = "e" * 32
    new_issue._drafts[draft_id] = (float("inf"), "alice")

    def fail(title, body):
        raise OSError("GitHub unavailable")

    assert new_issue.publish("alice", draft_id, "Title", "Body", fail) == {
        "status": "unavailable"
    }
    assert draft_id in new_issue._drafts


def test_publish_uses_write_token_not_read_only_app(monkeypatch):
    monkeypatch.setenv("PYTEST_GITHUB_TOKEN", "write-token")
    monkeypatch.setattr(
        te,
        "github_api_token",
        lambda: (_ for _ in ()).throw(AssertionError("read-only App used")),
    )
    sent = []

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, size):
            return b'{"number": 7}'

    def open_request(request, timeout):
        sent.append(request)
        return Reply()

    monkeypatch.setattr(te, "urlopen", open_request)
    assert te._github_create_issue("Edited", "Edited body") == {"number": 7}
    request = sent[0]
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer write-token"
    assert json.loads(request.data) == {
        "title": "Edited",
        "body": "Edited body",
        "labels": ["bug"],
    }
