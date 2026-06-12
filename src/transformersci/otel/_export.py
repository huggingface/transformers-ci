# Copyright 2026 The HuggingFace Inc. team.
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
"""OTLP span-exporter construction shared by every producer-side code path.

Both the pytest staging mirror (:mod:`resource_plugin`) and the non-pytest
in-process tracer (:mod:`instrument`) need to turn an OTLP endpoint + transport
+ headers into a span exporter, with the same quirks handled the same way
(HTTP signal-path suffix, plaintext-vs-TLS gRPC, the gRPC lowercase-header
fix). Keeping that in one place means a single fix applies everywhere.

- :func:`build_exporter` — the low-level builder: (endpoint, protocol, headers)
  → an ``OTLPSpanExporter``.
- :func:`build_primary_exporter` / :func:`build_staging_exporter` — resolve the
  primary / staging endpoint, protocol, and headers out of the standard
  ``OTEL_*`` / ``TRANSFORMERS_TEST_OTEL_STAGING_*`` env and call
  :func:`build_exporter`. Return ``None`` when no endpoint is configured.
"""

from __future__ import annotations

from collections.abc import Mapping

# Staging/primary export is best-effort over a possibly-flaky link: bound each
# export attempt so a dead/black-holed box can't make the end-of-session flush
# hang the CI job for long. A healthy same-network backend exports well under a
# second, so a few seconds is plenty.
STAGING_EXPORT_TIMEOUT_SECONDS = 5


def parse_otlp_headers(raw: str | None) -> dict[str, str] | None:
    """Parse an ``OTEL_EXPORTER_OTLP_HEADERS``-style string into a dict.

    The value is a comma-separated list of ``key=value`` pairs (e.g.
    ``Authorization=Bearer abc``), matching the W3C Baggage format the SDK uses.
    Returns ``None`` when there is nothing usable, so the exporter falls back to
    its own defaults.
    """
    if not raw:
        return None
    headers: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        headers[key.strip()] = value.strip()
    return headers or None


def build_exporter(endpoint: str, protocol: str, headers):
    """Build an OTLP span exporter for ``endpoint`` over ``protocol``.

    HTTP/protobuf uses the http exporter (which wants the full ``/v1/traces``
    signal path), everything else uses gRPC. A plaintext (``http://`` or
    scheme-less) gRPC endpoint is exported insecurely; ``https://`` keeps TLS.
    """
    if protocol in ("http/protobuf", "http", "https"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        url = endpoint.rstrip("/")
        if not url.endswith("/v1/traces"):
            url = f"{url}/v1/traces"
        return OTLPSpanExporter(
            endpoint=url, headers=headers, timeout=STAGING_EXPORT_TIMEOUT_SECONDS
        )

    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )

    insecure = not endpoint.lower().startswith("https://")
    # gRPC metadata keys must be lowercase (HTTP/2 rule). The OTLP gRPC exporter
    # passes headers through verbatim, so a capitalized key like "Authorization"
    # — valid over HTTP, and exactly what OTEL_EXPORTER_OTLP_HEADERS carries — is
    # rejected by gRPC core with "Illegal header key", failing the whole export
    # before any span ships.
    grpc_headers = {k.lower(): v for k, v in headers.items()} if headers else headers
    return OTLPSpanExporter(
        endpoint=endpoint,
        headers=grpc_headers,
        insecure=insecure,
        timeout=STAGING_EXPORT_TIMEOUT_SECONDS,
    )


def _resolve_protocol(raw: str | None) -> str:
    protocol = (raw or "grpc").lower()
    if protocol in ("http", "https"):
        return "http/protobuf"
    return protocol


def build_primary_exporter(env: Mapping[str, str]):
    """Build the primary span exporter from the standard ``OTEL_*`` env.

    Returns ``None`` when no OTLP endpoint is configured (the caller then emits
    no spans). Mirrors how pytest-opentelemetry resolves the primary export on
    the pytest path, so non-pytest tooling lands spans in the same backend.
    """
    endpoint = env.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or env.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    if not endpoint:
        return None
    protocol = _resolve_protocol(env.get("OTEL_EXPORTER_OTLP_PROTOCOL"))
    headers = parse_otlp_headers(
        env.get("OTEL_EXPORTER_OTLP_TRACES_HEADERS")
        or env.get("OTEL_EXPORTER_OTLP_HEADERS")
    )
    return build_exporter(endpoint, protocol, headers)


def build_staging_exporter(env: Mapping[str, str]):
    """Build the staging-mirror span exporter, or ``None`` when not configured.

    Reads ``TRANSFORMERS_TEST_OTEL_STAGING_*``, falling back to the primary
    protocol/headers so staging can authenticate and transport independently of
    prod while still inheriting sane defaults.
    """
    endpoint = env.get("TRANSFORMERS_TEST_OTEL_STAGING_ENDPOINT")
    if not endpoint:
        return None
    protocol = _resolve_protocol(
        env.get("TRANSFORMERS_TEST_OTEL_STAGING_PROTOCOL")
        or env.get("OTEL_EXPORTER_OTLP_PROTOCOL")
    )
    headers = parse_otlp_headers(
        env.get("TRANSFORMERS_TEST_OTEL_STAGING_HEADERS")
        or env.get("OTEL_EXPORTER_OTLP_HEADERS")
    )
    return build_exporter(endpoint, protocol, headers)
