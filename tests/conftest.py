from __future__ import annotations

import pytest

from transformersci.otel import claims, trace_exporter


@pytest.fixture(autouse=True)
def _fresh_claims(monkeypatch):
    """Claims are process-global; each test starts with none, and no
    background issue refresh reaches GitHub."""
    monkeypatch.setattr(claims, "_store", claims.Store(":memory:"))
    monkeypatch.setattr(trace_exporter, "_claim_refresher_started", True)
    monkeypatch.setattr(trace_exporter, "_fix_index_started", True)
    # Slow rows come from a range query; tests that need them patch it again.
    monkeypatch.setattr(trace_exporter, "_prometheus_matrix", lambda *args: [])


@pytest.fixture(autouse=True)
def _no_llm_retry_wait(monkeypatch):
    """LLM retries back off between attempts; tests do not wait for it."""
    from transformersci.otel import llm

    monkeypatch.setattr(llm, "_sleep", lambda seconds: None)
