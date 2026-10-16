"""Every dashboard LLM feature goes through llm.chat, which names itself:
Cerebras's Cloudflare 403s urllib's default User-Agent."""

from __future__ import annotations

import pytest

from transformersci.otel import llm, new_issue, wdyt, who_can_help

CFG = {
    "api_base": "https://router.example/v1",
    "model": "m",
    "api_key": "k",
    "proxy": "",
    "grafana": "",
    "bill_to": "",
}


def capture(monkeypatch):
    sent = []

    class Opener:
        def open(self, request, timeout):
            sent.append(request)
            raise OSError("stop after capture")

    monkeypatch.setattr(llm, "build_opener", lambda *args: Opener())
    return sent


@pytest.mark.parametrize(
    "call",
    [
        lambda: wdyt.ask(CFG, "q", lambda event: None),
        lambda: who_can_help.explain(
            "tests/a.py::t",
            "tests/a.py",
            {},
            [{"login": "a", "evidence": "e", "url": "u"}],
            [],
        ),
        lambda: new_issue.model_draft("tests/a.py::t", [], {}, [], [], []),
    ],
    ids=["wdyt", "who_can_help", "new_issue"],
)
def test_llm_requests_send_a_user_agent(monkeypatch, call):
    monkeypatch.setattr(llm, "config", lambda: dict(CFG))
    sent = capture(monkeypatch)
    with pytest.raises(OSError):
        call()
    assert sent[0].get_header("User-agent") == llm.USER_AGENT
