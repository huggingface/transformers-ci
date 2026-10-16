"""Every dashboard LLM call names itself: Cerebras's Cloudflare 403s urllib's default."""

from __future__ import annotations

import pytest

from transformersci.otel import new_issue, wdyt, who_can_help

CFG = {
    "api_base": "https://router.example/v1",
    "model": "m",
    "api_key": "k",
    "proxy": "",
    "grafana": "",
    "bill_to": "",
}


def capture(monkeypatch, module):
    sent = []

    class Opener:
        def open(self, request, timeout):
            sent.append(request)
            raise OSError("stop after capture")

    monkeypatch.setattr(module, "build_opener", lambda *args: Opener())
    return sent


@pytest.mark.parametrize(
    "module, call",
    [
        (wdyt, lambda: wdyt.ask(CFG, "q", lambda event: None)),
        (
            who_can_help,
            lambda: who_can_help.explain(
                "tests/a.py::t",
                "tests/a.py",
                {},
                [{"login": "a", "evidence": "e", "url": "u"}],
                [],
            ),
        ),
        (new_issue, lambda: new_issue.model_draft("tests/a.py::t", [], {}, [], [], [])),
    ],
)
def test_llm_requests_send_a_user_agent(monkeypatch, module, call):
    monkeypatch.setattr(wdyt, "config", lambda: dict(CFG))
    sent = capture(monkeypatch, module)
    with pytest.raises(OSError):
        call()
    assert sent[0].get_header("User-agent") == wdyt.LLM_USER_AGENT
