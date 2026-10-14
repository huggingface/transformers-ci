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
        "failure_line",
        lambda *args: ("tests/generation/test_streamers.py", 10),
    )
    monkeypatch.setattr(new_issue.who_can_help, "why", lambda *args: {"number": 42})
    monkeypatch.setattr(new_issue.related_issues, "lookup", lambda *args: {"hits": []})
    monkeypatch.setattr(
        new_issue,
        "model_draft",
        lambda *args: {
            "title": "Streamer fails in CI",
            "sections": {
                "system-info": "",
                "reproduction": "Observed in CI",
                "expected-behavior": "Should pass",
            },
        },
    )
    draft = new_issue.prepare(
        "alice", TRACE, NODE, [{"exception_message": "bad stream"}], get
    )
    assert draft["status"] == "ok"
    assert draft["title"] == "Streamer fails in CI"
    assert "### System Info" in draft["body"]
    assert "### Reproduction" in draft["body"]
    assert "### Expected behavior" in draft["body"]
    assert "- [ ] Official examples" in draft["body"]
    assert "bad stream" in draft["body"] and "var-trace_id=" + TRACE in draft["body"]


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
