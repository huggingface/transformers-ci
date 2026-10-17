"""The one LLM client behind every dashboard feature (WDYT?, Who can help, New issue).

An OpenAI-compatible ``/chat/completions`` endpoint (the HF router in prod),
configured from ``PYTEST_TRACE_EXPORTER_LLM_*``. Standard library only: the
exporter's slim image has no HTTP or SDK client installed.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections.abc import Callable
from http.client import HTTPException
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

# Some inference providers sit behind Cloudflare rules that refuse urllib's
# default "Python-urllib/x.y" User-Agent with a 403 HTML page (Cerebras does,
# through the HF router), so every request names itself.
USER_AGENT = "transformersci-trace-exporter"
# A non-streamed reply carries the model's reasoning too: thousands of tokens
# on a reasoning model, so this is far above any answer we ask for.
MAX_RESPONSE_BYTES = 1024 * 1024
# A streamed answer stops being read past this many characters.
MAX_STREAM_CHARS = 60000
# Each call is tried this many times within its own ``timeout``. Measured on
# 2026-10-17: with no reasoning cap, 4 of 4 New issue drafts on
# Qwen3.8-27B:cerebras spent all 8000 tokens reasoning and answered nothing.
ATTEMPTS = 3
RETRY_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}
MAX_RETRY_WAIT_SECONDS = 10
_sleep = time.sleep


class EmptyAnswer(ValueError):
    """The model returned no answer, usually because reasoning used the whole
    token budget. Worth another try: the next sample often answers."""


def config() -> dict[str, str]:
    return {
        "api_base": os.getenv("PYTEST_TRACE_EXPORTER_LLM_API_BASE", "").strip(),
        "model": os.getenv("PYTEST_TRACE_EXPORTER_LLM_MODEL", "").strip(),
        "api_key": os.getenv("PYTEST_TRACE_EXPORTER_LLM_API_KEY", "").strip(),
        "proxy": os.getenv("PYTEST_TRACE_EXPORTER_LLM_PROXY", "").strip(),
        "grafana": os.getenv("PYTEST_TRACE_EXPORTER_GRAFANA_URL", "").strip(),
        # HF org to bill (X-HF-Bill-To), as serge does with LLM_BILL_TO.
        "bill_to": os.getenv("PYTEST_TRACE_EXPORTER_LLM_BILL_TO", "").strip(),
        # Sent as reasoning_effort; "" leaves it out for providers that
        # reject it. "low" keeps Qwen3 on Cerebras from reasoning until the
        # budget runs out (9 of 9 answered, against 0 of 4 without it).
        "reasoning_effort": os.getenv(
            "PYTEST_TRACE_EXPORTER_LLM_REASONING_EFFORT", "low"
        ).strip(),
    }


def available(cfg: dict[str, str]) -> bool:
    return all(cfg.get(key) for key in ("api_base", "model", "api_key"))


def visible(text: str) -> str:
    """The part of an answer that is safe to show: no reasoning, and no
    trailing fragment that might still turn into a ``<think>`` tag."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start = text.find("<think>")
    if start >= 0:
        text = text[:start]
    for k in range(min(len("<think>") - 1, len(text)), 0, -1):
        if "<think>".startswith(text[-k:]):
            text = text[:-k]
            break
    return text.lstrip()


def chat(
    cfg: dict[str, str],
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float,
    timeout: float,
    session_id: str = "",
    emit: Callable[[dict], None] | None = None,
    parse: Callable[[str], object] | None = None,
) -> str:
    """One chat completion; returns the answer with any reasoning removed.

    ``max_tokens`` includes reasoning tokens on providers that count them
    (Cerebras does), so leave room beyond the answer itself. With ``emit`` the
    reply is streamed and the caller receives ``thinking``, ``tick`` (at least
    every 10 s during a long reasoning phase, so a proxy's idle timeout does
    not cut it) and ``delta`` events. Raises ``ValueError`` when the model
    returned no answer (e.g. it spent the whole budget reasoning).

    Transient failures (429 and 5xx, timeouts, a dropped connection, no
    answer, an answer ``parse`` rejects) are retried up to ``ATTEMPTS`` times
    within ``timeout`` overall, and each one is logged. A stream is never
    retried once part of its answer was shown. With ``parse`` the parsed
    value is returned instead of the text.
    """
    if not available(cfg):
        raise ValueError("LLM unavailable")
    deadline = time.monotonic() + timeout
    cfg = dict(cfg)
    for attempt in range(1, ATTEMPTS + 1):
        shown = []

        def relay(event: dict) -> None:
            if event.get("event") == "delta":
                shown.append(True)
            emit(event)

        wait = 2.0 ** (attempt - 1)
        try:
            text = _complete(
                cfg,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=max(deadline - time.monotonic(), 1),
                session_id=session_id,
                emit=relay if emit else None,
            )
            return parse(text) if parse else text
        except HTTPError as error:
            detail = _error_body(error)
            if (
                error.code == 400
                and cfg.get("reasoning_effort")
                and "reasoning_effort" in detail
            ):
                cfg["reasoning_effort"] = ""  # this provider rejects it
                retry = True
            else:
                retry = error.code in RETRY_STATUSES
            wait = max(wait, _retry_after(error))
            last, failure = error, f"HTTP {error.code}: {detail[:200]}"
        except (
            EmptyAnswer,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            OSError,
            HTTPException,
        ) as error:
            retry, last = True, error
            failure = f"{type(error).__name__}: {error}"
        except (ValueError, TypeError) as error:
            _log(cfg, attempt, f"{type(error).__name__}: {error}", final=True)
            raise
        remaining = deadline - time.monotonic()
        final = not retry or bool(shown) or attempt == ATTEMPTS or remaining < wait + 2
        _log(cfg, attempt, failure, final=final)
        if final:
            raise last
        if emit:
            emit(
                {
                    "event": "step",
                    "text": f"The model did not answer, retrying ({attempt + 1}/{ATTEMPTS})",
                }
            )
        _sleep(wait)
    raise AssertionError("unreachable")


def _complete(
    cfg: dict[str, str],
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float,
    timeout: float,
    session_id: str,
    emit: Callable[[dict], None] | None,
) -> str:
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }
    if cfg.get("bill_to"):
        headers["X-HF-Bill-To"] = cfg["bill_to"]
    if session_id:
        # Groups one question's router calls for a billing lookup.
        headers["X-HF-Session-id"] = session_id
    body = {
        "model": cfg["model"],
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        **(
            {"reasoning_effort": cfg["reasoning_effort"]}
            if cfg.get("reasoning_effort")
            else {}
        ),
        **({"stream": True} if emit else {}),
    }
    request = Request(
        cfg["api_base"].rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers=headers,
        method="POST",
    )
    proxies = {"http": cfg["proxy"], "https": cfg["proxy"]} if cfg.get("proxy") else {}
    with build_opener(ProxyHandler(proxies)).open(request, timeout=timeout) as response:
        if emit and "text/event-stream" in response.headers.get("Content-Type", ""):
            raw = _read_stream(response, emit)
        else:
            payload = response.read(MAX_RESPONSE_BYTES + 1)
            if len(payload) > MAX_RESPONSE_BYTES:
                raise ValueError("LLM response too large")
            choice = json.loads(payload)["choices"][0]
            raw = choice["message"].get("content") or ""
            if not raw.strip():
                raise EmptyAnswer(
                    f"LLM returned no answer (finish_reason={choice.get('finish_reason')})"
                )
    text = visible(raw + "\n").strip()
    if not text:
        raise EmptyAnswer("LLM returned no answer")
    return text


def _error_body(error: HTTPError) -> str:
    try:
        return error.read(2000).decode("utf-8", "replace")
    except (OSError, HTTPException, AttributeError):
        return ""


def _retry_after(error: HTTPError) -> float:
    try:
        value = float((error.headers or {}).get("Retry-After") or 0)
    except (TypeError, ValueError):
        return 0
    return min(max(value, 0), MAX_RETRY_WAIT_SECONDS)


def _log(cfg: dict[str, str], attempt: int, failure: str, *, final: bool) -> None:
    """One line per failed attempt: never the prompt, which carries test output."""
    outcome = "giving up" if final else "retrying"
    print(
        f"[llm] {cfg.get('model')} attempt {attempt}/{ATTEMPTS} failed, "
        f"{outcome}: {failure}",
        file=sys.stderr,
        flush=True,
    )


def log_failure(feature: str, nodeid: str, error: BaseException) -> None:
    """Why a feature answered "unavailable"; the dialog itself only says
    "try again", so this line is the only record of the cause."""
    print(
        f"[{feature}] {nodeid}: {type(error).__name__}: {str(error)[:300]}",
        file=sys.stderr,
        flush=True,
    )


def chat_json(cfg: dict[str, str], messages: list[dict], **options) -> object:
    """``chat`` for a JSON answer, tolerating a ```json fence around it. An
    answer that is not JSON is retried like an empty one."""
    return chat(cfg, messages, parse=_json_answer, **options)


def _json_answer(text: str) -> object:
    return json.loads(text.removeprefix("```json").removesuffix("```").strip())


def _read_stream(response, emit: Callable[[dict], None]) -> str:
    raw, shown, thinking = "", 0, False
    last = time.monotonic()
    for line in response:
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        choices = json.loads(data).get("choices") or [{}]
        delta = choices[0].get("delta") or {}
        if delta.get("reasoning_content") or delta.get("reasoning"):
            if not thinking:
                thinking = True
                emit({"event": "thinking"})
            elif time.monotonic() - last > 10:
                emit({"event": "tick"})
                last = time.monotonic()
        raw += delta.get("content") or ""
        if "<think>" in raw and not thinking:
            thinking = True
            emit({"event": "thinking"})
        shown_text = visible(raw)
        if len(shown_text) > shown:
            emit({"event": "delta", "text": shown_text[shown:]})
            shown, last = len(shown_text), time.monotonic()
        if len(raw) > MAX_STREAM_CHARS:
            break
    return raw
