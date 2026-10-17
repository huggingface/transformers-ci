from __future__ import annotations

import pytest

from transformersci.otel import claims, trace_exporter


@pytest.fixture(autouse=True)
def _fresh_claims(monkeypatch):
    """Claims are process-global; each test starts with none, and no
    background issue refresh reaches GitHub."""
    monkeypatch.setattr(claims, "_store", claims.Store(":memory:"))
    monkeypatch.setattr(trace_exporter, "_claim_refresher_started", True)
