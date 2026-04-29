from __future__ import annotations

import os
import sys
import time
from typing import TYPE_CHECKING, Sequence


if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class DebugWrappingSpanExporter:
    """Logs each export call before delegating to the inner OTLP exporter.

    Used to confirm whether the worker is actually shipping spans to the
    configured OTLP endpoint — works for both gRPC and HTTP transports
    because we wrap at the SpanExporter level, not the wire level.
    """

    def __init__(self, inner: "SpanExporter") -> None:
        self._inner = inner

    def export(self, spans: "Sequence[ReadableSpan]") -> "SpanExportResult":
        endpoint = (
            os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
            or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            or "<unset>"
        )
        protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
        start = time.monotonic()
        try:
            result = self._inner.export(spans)
        except BaseException as error:
            duration_ms = (time.monotonic() - start) * 1000
            print(
                f"OTEL DEBUG EXPORT spans={len(spans)} result=EXCEPTION "
                f"duration_ms={duration_ms:.1f} protocol={protocol} "
                f"endpoint={endpoint} error={error!r}",
                file=sys.stderr,
                flush=True,
            )
            raise
        duration_ms = (time.monotonic() - start) * 1000
        result_name = getattr(result, "name", str(result))
        print(
            f"OTEL DEBUG EXPORT spans={len(spans)} result={result_name} "
            f"duration_ms={duration_ms:.1f} protocol={protocol} endpoint={endpoint}",
            file=sys.stderr,
            flush=True,
        )
        return result

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        force_flush = getattr(self._inner, "force_flush", None)
        if force_flush is None:
            return True
        return force_flush(timeout_millis)
