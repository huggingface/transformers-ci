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
