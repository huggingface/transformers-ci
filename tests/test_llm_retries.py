"""llm.chat retries what is transient (no answer, 429/5xx, a dropped
connection, an answer that is not JSON) and nothing else, and caps reasoning:
on 2026-10-17 Qwen3.8-27B:cerebras spent a New issue draft's whole 8000-token
budget reasoning on 4 of 4 tries."""

from __future__ import annotations

import io
import json
from email.message import Message
from urllib.error import HTTPError, URLError

import pytest

from transformersci.otel import llm

CFG = {
    "api_base": "https://router.example/v1",
    "model": "m",
    "api_key": "k",
    "proxy": "",
    "bill_to": "",
    "reasoning_effort": "low",
}


class Response:
    def __init__(self, body: bytes, content_type: str = "application/json"):
        self.body, self.headers = body, {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, limit=-1):
        return self.body

    def __iter__(self):
        return iter(self.body.splitlines(keepends=True))


def answer(content: str, finish_reason: str = "stop") -> Response:
    reply = {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}]
    }
    return Response(json.dumps(reply).encode())


def stream(*deltas: dict, cut: bool = False) -> Response:
    events = "".join(
        "data: " + json.dumps({"choices": [{"delta": d}]}) + "\n\n" for d in deltas
    )
    if not cut:
        events += "data: [DONE]\n\n"
    return Response(events.encode(), "text/event-stream")


def http_error(code: int, body: str = "", retry_after: str = "") -> HTTPError:
    headers = Message()
    if retry_after:
        headers["Retry-After"] = retry_after
    return HTTPError("u", code, "x", headers, io.BytesIO(body.encode()))


class Opener:
    """Replays one outcome per request: a Response, or an exception to raise."""

    def __init__(self, *outcomes):
        self.outcomes, self.bodies = list(outcomes), []

    def __call__(self, handler):
        return self

    def open(self, request, timeout):
        self.bodies.append(json.loads(request.data))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, Response) and isinstance(outcome.body, bytes):
            body = outcome.body
            if b"data:" in body and b"[DONE]" not in body:
                # A stream cut mid-answer: the deltas, then a dropped connection.
                class Cut(Response):
                    def __iter__(inner):
                        yield from body.splitlines(keepends=True)
                        raise ConnectionResetError("peer closed")

                return Cut(body, "text/event-stream")
        return outcome


@pytest.fixture
def waits(monkeypatch):
    slept = []
    monkeypatch.setattr(llm, "_sleep", slept.append)
    return slept


def ask(opener, monkeypatch, **options):
    monkeypatch.setattr(llm, "build_opener", opener)
    return llm.chat(
        CFG,
        [{"role": "user", "content": "q"}],
        max_tokens=100,
        temperature=0,
        timeout=60,
        **options,
    )


def test_no_answer_is_retried_and_reasoning_is_capped(monkeypatch, waits, capsys):
    opener = Opener(answer("", "length"), answer("fine"))
    assert ask(opener, monkeypatch) == "fine"
    assert [body["reasoning_effort"] for body in opener.bodies] == ["low", "low"]
    assert waits == [1.0]
    log = capsys.readouterr().err
    assert "attempt 1/3 failed, retrying: EmptyAnswer" in log
    assert "finish_reason=length" in log


def test_rate_limit_waits_for_retry_after(monkeypatch, waits):
    opener = Opener(http_error(429, "slow down", retry_after="7"), answer("fine"))
    assert ask(opener, monkeypatch) == "fine"
    assert waits == [7.0]


def test_connection_errors_and_5xx_are_retried(monkeypatch, waits):
    opener = Opener(URLError("reset"), http_error(503), answer("fine"))
    assert ask(opener, monkeypatch) == "fine"
    assert waits == [1.0, 2.0]


def test_gives_up_after_three_attempts_with_the_original_error(
    monkeypatch, waits, capsys
):
    opener = Opener(TimeoutError("t1"), TimeoutError("t2"), TimeoutError("t3"))
    with pytest.raises(TimeoutError, match="t3"):
        ask(opener, monkeypatch)
    assert len(opener.bodies) == 3
    assert "attempt 3/3 failed, giving up: TimeoutError: t3" in capsys.readouterr().err


def test_a_bad_request_is_not_retried(monkeypatch, waits):
    opener = Opener(http_error(400, "bad messages"))
    with pytest.raises(HTTPError):
        ask(opener, monkeypatch)
    assert len(opener.bodies) == 1 and waits == []


def test_a_provider_rejecting_reasoning_effort_is_asked_again_without_it(
    monkeypatch, waits
):
    rejected = http_error(
        400, '{"message":"reasoning_effort: property is unsupported"}'
    )
    opener = Opener(rejected, answer("fine"))
    assert ask(opener, monkeypatch) == "fine"
    assert "reasoning_effort" in opener.bodies[0]
    assert "reasoning_effort" not in opener.bodies[1]


def test_empty_reasoning_effort_is_not_sent(monkeypatch):
    opener = Opener(answer("fine"))
    monkeypatch.setattr(llm, "build_opener", opener)
    llm.chat(
        {**CFG, "reasoning_effort": ""}, [], max_tokens=1, temperature=0, timeout=5
    )
    assert "reasoning_effort" not in opener.bodies[0]


def test_json_answers_that_do_not_parse_are_retried(monkeypatch, waits):
    opener = Opener(answer("{not json"), answer('```json\n{"people": []}\n```'))
    monkeypatch.setattr(llm, "build_opener", opener)
    result = llm.chat_json(CFG, [], max_tokens=1, temperature=0, timeout=60)
    assert result == {"people": []}


def test_a_stream_that_only_reasoned_is_retried_with_a_progress_step(
    monkeypatch, waits
):
    opener = Opener(stream({"reasoning_content": "hmm"}), stream({"content": "answer"}))
    events = []
    assert ask(opener, monkeypatch, emit=events.append) == "answer"
    assert {
        "event": "step",
        "text": "The model did not answer, retrying (2/3)",
    } in events


def test_a_stream_is_not_retried_once_text_was_shown(monkeypatch, waits):
    opener = Opener(
        stream({"content": "half an ans"}, cut=True), stream({"content": "x"})
    )
    events = []
    with pytest.raises(ConnectionResetError):
        ask(opener, monkeypatch, emit=events.append)
    assert len(opener.bodies) == 1


def test_retries_stay_within_the_call_timeout(monkeypatch, waits):
    clock = iter([0.0, 0.0, 59.0, 59.0])
    monkeypatch.setattr(llm.time, "monotonic", lambda: next(clock))
    opener = Opener(TimeoutError("slow"), answer("never asked"))
    with pytest.raises(TimeoutError):
        ask(opener, monkeypatch)
    assert len(opener.bodies) == 1 and waits == []
