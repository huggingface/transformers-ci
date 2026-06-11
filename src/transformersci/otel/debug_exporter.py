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
"""Opt-in diagnostic that logs every OTLP span export to stderr.

When traces silently fail to reach the backend (wrong endpoint, auth rejected,
transport mismatch), nothing surfaces — the export just drops. This module
monkeypatches the active span exporter's ``export()`` so each attempt prints a
one-line record: span count, result/exception, duration, protocol, and the
resolved endpoint. A module-level flag keeps the patch idempotent so repeated
installs don't stack wrappers. Intended to be enabled only while debugging
export connectivity, never in steady-state CI.
"""

from __future__ import annotations

import os
import sys
import time


_PATCH_FLAG = "_transformersci_debug_patched"


def _make_logging_export(original):
    def export(self, spans):
        endpoint = (
            os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
            or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            or "<unset>"
        )
        protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
        start = time.monotonic()
        try:
            result = original(self, spans)
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

    return export


def install_debug_logging() -> int:
    """Monkey-patch OTLP exporter classes' export() to log each call.

    Patches at the class level so it works regardless of how the SDK
    pipes spans through processors (incl. SDKs where BatchSpanProcessor
    exposes span_exporter as a read-only property).

    Returns the number of exporter classes patched.
    """
    classes = []
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as GrpcSpanExporter,
        )

        classes.append(GrpcSpanExporter)
    except ImportError:
        pass
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as HttpSpanExporter,
        )

        classes.append(HttpSpanExporter)
    except ImportError:
        pass

    patched = 0
    for cls in classes:
        if getattr(cls, _PATCH_FLAG, False):
            continue
        cls.export = _make_logging_export(cls.export)
        setattr(cls, _PATCH_FLAG, True)
        patched += 1
    return patched
