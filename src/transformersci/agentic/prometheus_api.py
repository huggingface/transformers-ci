# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Read-only instant queries against the transformers-ci Prometheus.

Everything goes through the **public Grafana datasource proxy** — the same
read-only path ``deploy/scripts/tempo.py`` uses — so this needs no cluster
access, no kubectl and no token.

Best-effort by contract: :func:`instant_query` returns ``[]`` for every failure
mode (unreachable, slow, non-200, malformed body, Prometheus reporting an
error). Callers run unattended from the nightly cron, so "no data" must degrade
to "keep the previous behaviour", never to a crashed run.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from collections.abc import Callable

DEFAULT_BASE_URL = "https://transformers-ci.lor-e.huggingface.cool"
QUERY_PATH = "/api/datasources/proxy/uid/prometheus/api/v1/query"
DEFAULT_TIMEOUT = 30.0

# A fetch takes the fully-built URL and returns the raw body. Injected by tests
# so the suite never reaches the network (same rule collect_prior_feedback uses).
Fetch = Callable[[str], "bytes | str"]


def query_url(expr: str, *, base_url: str | None = None) -> str:
    """The proxy URL for an instant query of ``expr``."""
    base = (
        base_url or os.environ.get("ITF_PROMETHEUS_URL") or DEFAULT_BASE_URL
    ).rstrip("/")
    return f"{base}{QUERY_PATH}?" + urllib.parse.urlencode({"query": expr})


def _http_get(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "transformers-ci-triage"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def instant_query(
    expr: str,
    *,
    base_url: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    fetch: Fetch | None = None,
) -> list[dict]:
    """Result series for an instant query, or ``[]`` on ANY failure.

    Each element is ``{"metric": {label: value, ...}, "value": [ts, "num"]}``.
    """
    url = query_url(expr, base_url=base_url)
    get: Fetch = fetch or (lambda u: _http_get(u, timeout))
    try:
        payload = json.loads(get(url))
    except Exception:
        return []
    if not isinstance(payload, dict) or payload.get("status") != "success":
        return []
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    result = data.get("result")
    return result if isinstance(result, list) else []


def scalars_by_label(series: list[dict], label: str) -> dict[str, float]:
    """``{series[label]: float(value)}`` for series that have both.

    Skips anything unparseable rather than raising — a single odd sample must
    not lose the rest of the response.
    """
    out: dict[str, float] = {}
    for s in series:
        if not isinstance(s, dict):
            continue
        key = (s.get("metric") or {}).get(label)
        value = s.get("value")
        if not key or not isinstance(value, list) or len(value) < 2:
            continue
        try:
            out[str(key)] = float(value[1])
        except (TypeError, ValueError):
            continue
    return out
