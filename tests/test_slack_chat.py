import pytest
from transformersci.otel import slack_chat


@pytest.fixture
def chat(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_RUN_STORE", str(tmp_path))
    monkeypatch.setattr(slack_chat.llm, "available", lambda cfg: True)
    calls = []

    def answer(cfg, messages, **options):
        calls.append(messages)
        return "Answer <@everyone>"

    monkeypatch.setattr(slack_chat.llm, "chat", answer)
    return calls


def test_chat_history_isolated_and_bounded(chat):
    assert "&lt;@everyone&gt;" in slack_chat.answer(
        "private:T:U1:D", "U1", "private question", "https://dashboard"
    )
    slack_chat.answer(
        "channel:T:C:1", "U1", "public question", "https://dashboard", shared=True
    )
    assert all("private question" not in m["content"] for m in chat[-1])
    slack_chat.answer(
        "channel:T:C:1", "U2", "follow up", "https://dashboard", shared=True
    )
    assert any("public question" in m["content"] for m in chat[-1])
    slack_chat.answer(
        "channel:T:C:2", "U2", "new thread", "https://dashboard", shared=True
    )
    assert all("public question" not in m["content"] for m in chat[-1])
    for n in range(10):
        slack_chat.answer("private:T:U1:D", "U1", f"turn {n}", "https://dashboard")
    assert len(chat[-1]) == 2 + 2 * slack_chat.HISTORY_TURNS
    assert all("private question" not in m["content"] for m in chat[-1])
    assert "no tools" in chat[-1][0]["content"]


def test_chat_limits_failure_and_retention(chat, monkeypatch):
    assert "too long" in slack_chat.answer(
        "scope", "U1", "x" * 8001, "https://dashboard"
    )
    assert not chat
    monkeypatch.setattr(slack_chat.time, "time", lambda: 1000)
    slack_chat.answer("scope", "U1", "expired conversation", "https://dashboard")
    monkeypatch.setattr(
        slack_chat.time, "time", lambda: 1000 + slack_chat.RETENTION + 1
    )
    slack_chat.answer("scope", "U1", "new conversation", "https://dashboard")
    assert all("expired conversation" not in m["content"] for m in chat[-1])
    monkeypatch.setattr(
        slack_chat.llm,
        "chat",
        lambda *a, **kw: (_ for _ in ()).throw(ValueError("secret")),
    )
    result = slack_chat.answer("scope", "U1", "question", "https://dashboard")
    assert "couldn’t answer" in result and "secret" not in result


def test_intent_routes_once_without_saving_private_result(chat, monkeypatch):
    requested = []
    monkeypatch.setattr(
        slack_chat.llm,
        "chat_json",
        lambda *a, **k: {"reply": "", "command": {"name": "pr", "argument": "42"}},
    )

    def dispatch(name, argument):
        requested.append((name, argument))
        return "private result"

    assert (
        slack_chat.answer(
            "scope", "U1", "How is PR 42 doing?", "https://dashboard", dispatch=dispatch
        )
        == "private result"
    )
    assert requested == [("pr", "42")]
    with slack_chat.slack_commands.connect() as db:
        saved = db.execute("SELECT answer FROM slack_chat_turns").fetchone()[0]
    assert "private result" not in saved and "PR 42" in saved
    monkeypatch.setattr(
        slack_chat.llm,
        "chat_json",
        lambda *a, **k: {"reply": "", "command": {"name": "rerun", "argument": "42"}},
    )
    slack_chat.answer(
        "scope",
        "U1",
        "Please rerun its failed CI",
        "https://dashboard",
        dispatch=dispatch,
    )
    assert requested[-1] == ("rerun", "42")


def test_intent_requires_grounded_unambiguous_target(chat, monkeypatch):
    def fail(*args):
        pytest.fail("Invalid command must never execute")

    for prompt, command in [
        ("Please check PR 42", {"name": "pr", "argument": "99"}),
        (
            "Check https://github.com/another/project/pull/42",
            {"name": "pr", "argument": "42"},
        ),
        ("Rerun PR 42 and PR 99", {"name": "rerun", "argument": "42"}),
        ("Create the PR", {"name": "approve", "argument": "42"}),
        ("Fix this", {"name": "traceback", "argument": "ValueError: invented"}),
        ("Fix this", {"name": "traceback", "argument": "this"}),
    ]:
        monkeypatch.setattr(
            slack_chat.llm,
            "chat_json",
            lambda *a, **k: {"reply": "", "command": command},
        )
        slack_chat.answer(prompt, "U1", prompt, "https://dashboard", dispatch=fail)


def test_intent_quoted_evidence_and_clarification(chat, monkeypatch):
    requested = []
    monkeypatch.setattr(
        slack_chat.llm,
        "chat_json",
        lambda *a, **k: {"reply": "Which PR?", "command": None},
    )
    assert (
        slack_chat.answer(
            "scope",
            "U1",
            "Please rerun failed CI",
            "https://dashboard",
            dispatch=lambda *a: None,
        )
        == "Which PR?"
    )
    monkeypatch.setattr(
        slack_chat.llm,
        "chat_json",
        lambda *a, **k: {"reply": "", "command": {"name": "rerun", "argument": "42"}},
    )
    slack_chat.answer(
        "scope",
        "U1",
        "42",
        "https://dashboard",
        dispatch=lambda *a: requested.append(a) or "ok",
    )
    assert requested == [("rerun", "42")]
    monkeypatch.setattr(
        slack_chat.llm,
        "chat_json",
        lambda *a, **k: {
            "reply": "",
            "command": {"name": "traceback", "argument": "ValueError: bad shape"},
        },
    )
    slack_chat.answer(
        "failure",
        "U1",
        "Can you fix this? ValueError: bad shape",
        "https://dashboard",
        dispatch=lambda *a: requested.append(a) or "ok",
    )
    assert requested[-1] == ("traceback", "ValueError: bad shape")
