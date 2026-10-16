"""People suggestions only use the selected trace and linked Relore evidence."""

from __future__ import annotations

import json
import base64
from datetime import datetime, timedelta, timezone

from transformersci.otel import who_can_help as helpers

NODE = "tests/models/foo/test_modeling_foo.py::FooTest::test_x"
TRACE = "a" * 32
DETAILS = [
    {
        "exception_message": "bad value",
        "exception_stacktrace": (
            'File "tests/models/foo/test_modeling_foo.py", line 42, in test_x\n'
            'File "src/transformers/models/foo/modeling_foo.py", line 91, in forward\n'
        ),
    }
]


def setup_function():
    helpers._cache.clear()
    helpers._modifier_cache.clear()
    helpers._inflight.clear()
    helpers._asked.clear()


def test_line_is_selected_from_selected_failure_only():
    assert helpers.failure_line(NODE, DETAILS) == (
        "src/transformers/models/foo/modeling_foo.py",
        91,
    )
    assert helpers.failure_line(
        NODE,
        [
            {
                "exception_stacktrace": (
                    'File "/workspace/transformers/src/transformers/modeling_utils.py", '
                    "line 217, in forward"
                )
            }
        ],
    ) == ("src/transformers/modeling_utils.py", 217)
    assert helpers.failure_line(NODE, [{"exception_stacktrace": "no frames"}]) is None
    assert helpers.failure_line("../../etc/passwd::test_x", DETAILS) is None


def test_truncated_trace_falls_back_to_model_test_class():
    node = (
        "tests/models/deepseek_ocr2/test_modeling_deepseek_ocr2.py::"
        "DeepseekOcr2ModelTest::test_tp_plan_matches_params"
    )
    source = "import unittest\n\nclass DeepseekOcr2ModelTest(VLMModelTest, unittest.TestCase):\n    pass\n"
    urls = []

    def get(url):
        urls.append(url)
        return {
            "encoding": "base64",
            "content": base64.b64encode(source.encode()).decode(),
        }

    assert helpers.source_line(
        node,
        [{"exception_stacktrace": "def test_tp_plan_matches_params(self):\n"}],
        get,
    ) == ("tests/models/deepseek_ocr2/test_modeling_deepseek_ocr2.py", 3)
    assert urls[0].endswith(
        "/contents/tests/models/deepseek_ocr2/test_modeling_deepseek_ocr2.py"
    )


def test_source_line_prefers_local_method_over_class():
    node = "tests/models/foo/test_modeling_foo.py::FooTest::test_x"
    source = "class FooTest:\n    def test_x(self):\n        pass\n"

    def get(url):
        return {
            "encoding": "base64",
            "content": base64.b64encode(source.encode()).decode(),
        }

    assert helpers.source_line(node, [{"exception_stacktrace": "truncated"}], get) == (
        "tests/models/foo/test_modeling_foo.py",
        2,
    )


def test_candidates_require_linked_human_evidence():
    history = {
        "number": 12,
        "thread": {"author": "author"},
        "anchored": [{"author": "reviewer"}, {"author": "bad/name"}],
        "reviews": [{"author": "reviewer"}],
    }
    hits = [
        {
            "author": "maintainer",
            "trust": "authoritative",
            "number": 13,
            "type": "issue",
            "url": "https://github.com/huggingface/transformers/issues/13",
        },
        {
            "author": "unverified",
            "trust": "reported",
            "number": 14,
            "type": "issue",
            "url": "https://github.com/huggingface/transformers/issues/14",
        },
    ]
    people = helpers.candidates(history, hits)
    assert [p["login"] for p in people] == ["reviewer", "author", "maintainer"]
    assert people[0]["url"].endswith("/pull/12")


def test_model_cannot_add_or_repeat_people(monkeypatch):
    from transformersci.otel import llm

    class Opener:
        def open(self, request, timeout):
            assert timeout == 45
            assert json.loads(request.data)["max_tokens"] == 2000

            class Reply:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

                def read(self, size):
                    return json.dumps(
                        {
                            "choices": [
                                {
                                    "message": {
                                        "content": json.dumps(
                                            {
                                                "people": [
                                                    {
                                                        "login": "reviewer",
                                                        "why": "Can inspect the review",
                                                    },
                                                    {
                                                        "login": "invented",
                                                        "why": "No evidence",
                                                    },
                                                    {
                                                        "login": "reviewer",
                                                        "why": "Again",
                                                    },
                                                ]
                                            }
                                        )
                                    }
                                }
                            ]
                        }
                    ).encode()

            return Reply()

    monkeypatch.setattr(llm, "build_opener", lambda *args: Opener())
    monkeypatch.setattr(
        llm,
        "config",
        lambda: {
            "api_base": "https://llm.test/v1",
            "model": "small",
            "api_key": "secret",
            "proxy": "",
            "bill_to": "huggingface",
        },
    )
    people = [
        {
            "login": "reviewer",
            "evidence": "reviewed line",
            "url": "https://github.com/x",
        }
    ]
    result = helpers.explain(NODE, ("tests/test.py", 12), {}, people, DETAILS)
    assert result == [{"person": people[0], "why": "Can inspect the review"}]


def test_suggest_caches_result_and_limits_names(monkeypatch):
    calls = []
    monkeypatch.setattr(
        helpers,
        "why",
        lambda *args: (
            calls.append(args)
            or {
                "repo": "huggingface/transformers",
                "number": 12,
                "thread": {"author": "author"},
            }
        ),
    )
    monkeypatch.setattr(helpers.related_issues, "lookup", lambda *args: {"hits": []})
    monkeypatch.setattr(
        helpers,
        "explain",
        lambda *args, **kwargs: [
            {"person": args[3][0], "why": "Can explain the change"}
        ],
    )
    events = []
    first = helpers.suggest("alice", TRACE, NODE, DETAILS, emit=events.append)
    assert first["people"][0]["login"] == "author"
    assert [event["text"] for event in events if event["event"] == "step"] == [
        "Locating the failing test in the repository",
        "Reading Relore history for the failing line",
        "Searching related issues and PRs",
        "Finding reviewers and thread participants",
        "Asking the model where each person can help (slow)",
    ]
    assert helpers.suggest("bob", TRACE, NODE, DETAILS) == first
    assert calls == [("src/transformers/models/foo/modeling_foo.py", 91)]


def test_suggest_uses_evidence_when_model_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        helpers,
        "why",
        lambda *args: {
            "number": 12,
            "thread": {"author": "author"},
            "reviews": [{"author": "reviewer"}],
        },
    )
    monkeypatch.setattr(helpers.related_issues, "lookup", lambda *args: {"hits": []})

    calls = []

    def timeout(*args, **kwargs):
        calls.append(kwargs["timeout"])
        raise OSError("timeout")

    monkeypatch.setattr(helpers, "explain", timeout)
    monkeypatch.setattr(helpers.time, "sleep", lambda seconds: None)
    events = []

    result = helpers.suggest("caller", TRACE, NODE, DETAILS, emit=events.append)

    assert result["status"] == "ok"
    assert [person["login"] for person in result["people"]] == [
        "reviewer",
        "author",
    ]
    assert result["people"][0]["why"] == (
        "May be able to help because they reviewed the related PR in PR #12."
    )
    assert result["location"] == "src/transformers/models/foo/modeling_foo.py:91"
    assert calls == [20, 20]
    assert {event.get("text") for event in events} >= {
        "Retrying helper suggestions",
        "Using repository evidence for suggestions",
    }


def test_suggest_retries_model_once_then_uses_its_answer(monkeypatch):
    monkeypatch.setattr(
        helpers,
        "why",
        lambda *args: {"number": 12, "thread": {"author": "author"}},
    )
    monkeypatch.setattr(helpers.related_issues, "lookup", lambda *args: {"hits": []})
    monkeypatch.setattr(helpers.time, "sleep", lambda seconds: None)
    calls = []

    def explain(*args, **kwargs):
        calls.append(kwargs["timeout"])
        if len(calls) == 1:
            raise TimeoutError("temporary timeout")
        return [{"person": args[3][0], "why": "Can inspect the change"}]

    monkeypatch.setattr(helpers, "explain", explain)

    result = helpers.suggest("caller", TRACE, NODE, DETAILS)

    assert calls == [20, 20]
    assert result["status"] == "ok"
    assert result["people"][0]["why"] == "Can inspect the change"


def test_missing_relore_people_use_most_frequent_recent_test_modifier(monkeypatch):
    monkeypatch.setattr(helpers, "why", lambda *args: {})
    monkeypatch.setattr(helpers.related_issues, "lookup", lambda *args: {"hits": []})
    monkeypatch.setattr(
        helpers,
        "explain",
        lambda *args, **kwargs: [
            {"person": args[3][0], "why": "Can inspect recent test changes"}
        ],
    )
    commits = [
        {"author": {"login": login}, "sha": f"{index:040x}"}
        for index, login in enumerate(
            [
                "alice",
                "bob",
                "bob",
                "alice",
                "bob",
                "alice",
                "bob",
                "alice",
                "bob",
                "alice",
            ]
        )
    ]
    urls = []

    def get(url):
        urls.append(url)
        if "/commits?" in url:
            return commits
        return {}

    events = []
    result = helpers.suggest("caller", TRACE, NODE, DETAILS, get, events.append)
    assert result["status"] == "ok"
    assert result["people"][0]["login"] == "alice"
    assert "5 of its last 10 commits" in result["people"][0]["evidence"]
    assert "path=tests%2Fmodels%2Ffoo%2Ftest_modeling_foo.py&per_page=10" in urls[0]
    assert "Checking recent changes to the test file" in [
        event["text"] for event in events
    ]
    assert helpers.frequent_test_modifier(NODE, get)[0]["login"] == "alice"
    assert len([url for url in urls if "/commits?" in url]) == 1


def test_missing_trace_line_still_uses_test_commit_fallback(monkeypatch):
    monkeypatch.setattr(helpers.related_issues, "lookup", lambda *args: {"hits": []})
    monkeypatch.setattr(
        helpers,
        "explain",
        lambda *args, **kwargs: [{"person": args[3][0], "why": "Can inspect the test"}],
    )

    def get(url):
        if "/commits?" in url:
            return [{"author": {"login": "alice"}, "sha": "f" * 40}]
        return {}

    result = helpers.suggest(
        "caller", TRACE, NODE, [{"exception_stacktrace": "truncated"}], get
    )
    assert result["status"] == "ok"
    assert result["location"] == "tests/models/foo/test_modeling_foo.py"
    assert result["people"][0]["login"] == "alice"


def test_recent_pr_links_use_dashboard_only_for_confirmed_recent_prs():
    now = datetime(2026, 10, 14, tzinfo=timezone.utc)
    people = [
        {
            "why": "Review PR #12, issue #13, and PR #14",
            "evidence": "reviewed PR #15",
            "url": "https://github.com/huggingface/transformers/pull/16",
        }
    ]
    dates = {
        "12": now - timedelta(days=89),
        "13": now - timedelta(days=1),
        "14": now - timedelta(days=90),
        "15": now + timedelta(days=1),
    }

    def get(url):
        number = url.rsplit("/", 1)[-1]
        if number == "16":
            raise OSError("GitHub unavailable")
        payload = {"created_at": dates[number].isoformat()}
        if number != "13":
            payload["pull_request"] = {"url": "https://api.github.com/pull"}
        return payload

    assert helpers.recent_pr_links(people, get, now=now) == {
        "12": "/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=12"
    }


def test_suggest_with_evidence_does_not_repeat_the_lookups(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("lookup repeated")

    monkeypatch.setattr(helpers, "source_line", unexpected)
    monkeypatch.setattr(helpers, "why", unexpected)
    monkeypatch.setattr(helpers.related_issues, "lookup", unexpected)
    monkeypatch.setattr(helpers, "frequent_test_modifier", unexpected)
    person = {
        "login": "alice",
        "evidence": "reviewed it",
        "url": "https://github.com/x",
    }
    monkeypatch.setattr(
        helpers,
        "explain",
        lambda *args, **kwargs: [{"person": person, "why": "Reviewed the line"}],
    )
    result = helpers.suggest(
        "bob",
        "c" * 32,
        "tests/a/test_b.py::T::test_x",
        [],
        None,
        evidence=(("tests/a/test_b.py", 3), {"number": 1}, [], [person]),
    )
    assert result["status"] == "ok"
    assert result["people"][0]["login"] == "alice"
    assert result["location"] == "tests/a/test_b.py:3"
