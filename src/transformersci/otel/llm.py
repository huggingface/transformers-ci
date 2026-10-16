"""The one LLM client behind every dashboard feature (WDYT?, Who can help, New issue).

An OpenAI-compatible ``/chat/completions`` endpoint (the HF router in prod),
configured from ``PYTEST_TRACE_EXPORTER_LLM_*``. Standard library only: the
exporter's slim image has no HTTP or SDK client installed.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
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


def config() -> dict[str, str]:
    return {
        "api_base": os.getenv("PYTEST_TRACE_EXPORTER_LLM_API_BASE", "").strip(),
        "model": os.getenv("PYTEST_TRACE_EXPORTER_LLM_MODEL", "").strip(),
        "api_key": os.getenv("PYTEST_TRACE_EXPORTER_LLM_API_KEY", "").strip(),
        "proxy": os.getenv("PYTEST_TRACE_EXPORTER_LLM_PROXY", "").strip(),
        "grafana": os.getenv("PYTEST_TRACE_EXPORTER_GRAFANA_URL", "").strip(),
        # HF org to bill (X-HF-Bill-To), as serge does with LLM_BILL_TO.
        "bill_to": os.getenv("PYTEST_TRACE_EXPORTER_LLM_BILL_TO", "").strip(),
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
) -> str:
    """One chat completion; returns the answer with any reasoning removed.

    ``max_tokens`` includes reasoning tokens on providers that count them
    (Cerebras does), so leave room beyond the answer itself. With ``emit`` the
    reply is streamed and the caller receives ``thinking``, ``tick`` (at least
    every 10 s during a long reasoning phase, so a proxy's idle timeout does
    not cut it) and ``delta`` events. Raises ``ValueError`` when the model
    returned no answer (e.g. it spent the whole budget reasoning).
    """
    if not available(cfg):
        raise ValueError("LLM unavailable")
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
                raise ValueError(
                    f"LLM returned no answer (finish_reason={choice.get('finish_reason')})"
                )
    text = visible(raw + "\n").strip()
    if not text:
        raise ValueError("LLM returned no answer")
    return text


def chat_json(cfg: dict[str, str], messages: list[dict], **options) -> object:
    """``chat`` for a JSON answer, tolerating a ```json fence around it."""
    text = chat(cfg, messages, **options)
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
