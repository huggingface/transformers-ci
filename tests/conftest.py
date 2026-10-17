from __future__ import annotations

import pytest

from transformersci.otel import claims


@pytest.fixture(autouse=True)
def _fresh_claims(monkeypatch):
    """Claims are process-global; each test starts with none."""
    monkeypatch.setattr(claims, "_store", claims.Store(":memory:"))
