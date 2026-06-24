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
"""Opt-in diagnostic that logs OTLP span exports and a per-process span tally.

When traces silently fail to reach the backend (wrong endpoint, auth rejected,
transport mismatch) nothing surfaces — the export just drops. When the test
session out-produces the export pipeline, the ``BatchSpanProcessor`` silently
drops spans off the *front* (its queue fills), so a shard that ran 13k tests
lands only a few thousand spans in Tempo and the dashboards under-count.

This module makes both visible, gated behind ``TRANSFORMERSCI_OTEL_DEBUG=1``:

- Each ``export()`` attempt prints a one-line record: span count, result, the
  duration, protocol, and resolved endpoint (catches transport/auth failures).
- Every span handed to a ``BatchSpanProcessor`` (its ``on_end``) is tallied, and
  every span passed to an exporter is tallied. At process exit a single SUMMARY
  line prints ``produced`` vs ``exported`` — their difference is the queue drop.
  Each pytest shard is its own process, so the summary is per-shard, which lines
  up directly with the per-shard GitHub ``collected`` counts.

All patches are class-level and idempotent (a module-level flag keeps repeated
installs from stacking wrappers). Intended only while debugging, never in
steady-state CI.
"""

from __future__ import annotations

import atexit
import os
import sys
import threading
import time


_PATCH_FLAG = "_transformersci_debug_patched"


class _SpanTally:
    """Thread-safe span accounting, split by destination (prod vs staging).

    The client fans every span out to TWO pipelines — the prod backend and the
    staging mirror — so a tally that summed both made every prod comparison wrong
    (produced doubled; the staging mirror's own timeouts inflated the failure
    count). We therefore keep separate counters per destination and report PROD as
    the headline; staging is reported under ``stage_*`` and is excluded from any
    prod-vs-Tempo comparison.

    Per destination:

    - ``produced`` — ``BatchSpanProcessor.on_end`` calls for that pipeline (spans
      the SDK enqueued for it).
    - ``submitted`` — spans passed to that exporter's ``export()`` (any outcome).
    - ``exported`` — spans whose ``export()`` returned ``SUCCESS``.
    - ``failed`` — spans whose ``export()`` returned ``FAILURE``/raised — **lost**
      once retries are exhausted.

    ``produced - submitted`` is BSP queue overflow; ``failed`` is transport loss.
    The outcome is recorded *after* the call so a timeout/rejection counts as
    ``failed`` (the earlier blind tally recorded ``exported`` before the call and
    hid that). ``on_end`` runs on the test threads, ``export`` on the batch worker
    thread, hence the lock.
    """

    _DESTS = ("prod", "stage")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_dest = {dest: self._zero() for dest in self._DESTS}

    @staticmethod
    def _zero() -> dict:
        return {
            "produced": 0,
            "submitted": 0,
            "exported": 0,
            "failed": 0,
            "export_calls": 0,
            "failed_calls": 0,
        }

    def _bucket(self, dest: str) -> dict:
        return self._by_dest["stage" if dest == "stage" else "prod"]

    def record_produced(self, n: int = 1, *, dest: str = "prod") -> None:
        with self._lock:
            self._bucket(dest)["produced"] += n

    def record_export(self, n: int, *, ok: bool, dest: str = "prod") -> None:
        """Record the outcome of one ``export()`` of ``n`` spans to ``dest``."""
        with self._lock:
            bucket = self._bucket(dest)
            bucket["submitted"] += n
            bucket["export_calls"] += 1
            if ok:
                bucket["exported"] += n
            else:
                bucket["failed"] += n
                bucket["failed_calls"] += 1

    # Back-compat read accessors (prod): existing call sites / tests read these.
    @property
    def produced(self) -> int:
        return self._by_dest["prod"]["produced"]

    @property
    def submitted(self) -> int:
        return self._by_dest["prod"]["submitted"]

    @property
    def exported(self) -> int:
        return self._by_dest["prod"]["exported"]

    @property
    def failed(self) -> int:
        return self._by_dest["prod"]["failed"]

    @property
    def export_calls(self) -> int:
        return self._by_dest["prod"]["export_calls"]

    @property
    def failed_calls(self) -> int:
        return self._by_dest["prod"]["failed_calls"]

    def summary_line(self) -> str:
        with self._lock:
            p = self._by_dest["prod"]
            s = self._by_dest["stage"]
            prod_queue_dropped = p["produced"] - p["submitted"]
            return (
                f"OTEL DEBUG SUMMARY produced={p['produced']} "
                f"submitted={p['submitted']} exported={p['exported']} "
                f"failed={p['failed']} not_exported={prod_queue_dropped} "
                f"export_calls={p['export_calls']} "
                f"failed_calls={p['failed_calls']} "
                f"stage_produced={s['produced']} stage_exported={s['exported']} "
                f"stage_failed={s['failed']} "
                f"(headline fields are PROD ONLY — staging mirror excluded; "
                f"produced=spans handed to BatchSpanProcessor; "
                f"submitted=spans passed to exporter; "
                f"exported=spans the exporter confirmed shipped (result=SUCCESS); "
                f"failed=spans whose export returned FAILURE/raised — LOST after "
                f"retries; not_exported=produced-submitted=BSP queue overflow; "
                f"stage_*=staging mirror, NOT part of prod comparisons)"
            )


_TALLY = _SpanTally()
_SUMMARY_REGISTERED = False
# Print the BatchSpanProcessor config the first time we see a span flushed, so the
# log records the effective queue/batch/timeout knobs (incl. anything
# configure-ci-otel set via OTEL_BSP_* / OTEL_EXPORTER_OTLP_TIMEOUT) exactly once.
_BSP_CONFIG_LOGGED = False
_BSP_CONFIG_LOCK = threading.Lock()


def _exporter_timeout_s(exporter) -> object:
    """Best-effort read of an OTLP exporter's per-export timeout (seconds).

    Both the HTTP and gRPC ``OTLPSpanExporter`` store the resolved timeout —
    which is where ``OTEL_EXPORTER_OTLP_TIMEOUT`` / the constructor ``timeout``
    actually lands — on ``_timeout``. Surfacing it next to ``duration_ms`` makes a
    timeout failure self-evident (``duration_ms≈timeout_s*1000``) and confirms the
    configured value really took effect. Returns "?" if the attribute is absent.
    """
    return getattr(exporter, "_timeout", "?")


# Env carrying the staging-mirror endpoint (set by configure-ci-otel). An exporter
# whose endpoint matches this is the staging pipeline; everything else is prod.
_STAGING_ENDPOINT_ENV = "TRANSFORMERS_TEST_OTEL_STAGING_ENDPOINT"


def _endpoint_host(url: str) -> str:
    """Return the ``host:port`` of an OTLP endpoint, ignoring scheme/path.

    The HTTP exporter appends ``/v1/traces`` and the gRPC endpoint may be
    scheme-less, so compare on netloc only.
    """
    from urllib.parse import urlparse

    if not url:
        return ""
    parsed = urlparse(url if "//" in url else f"//{url}")
    return parsed.netloc or parsed.path.split("/")[0]


def _exporter_endpoint(exporter) -> str:
    return str(getattr(exporter, "_endpoint", "") or "")


def _dest_of_exporter(exporter) -> str:
    """Classify an exporter as ``"stage"`` (the staging mirror) or ``"prod"``.

    The client fans every span out to both, so the tally must attribute each
    ``export()`` to the right backend — otherwise prod numbers are polluted by the
    staging mirror's separate (and flakier) delivery.
    """
    staging = os.getenv(_STAGING_ENDPOINT_ENV, "")
    if not staging:
        return "prod"
    endpoint_host = _endpoint_host(_exporter_endpoint(exporter))
    return (
        "stage"
        if endpoint_host and endpoint_host == _endpoint_host(staging)
        else "prod"
    )


def _processor_exporter(processor):
    """Find the SpanExporter a BatchSpanProcessor delegates to, across SDK layouts."""
    inner = getattr(processor, "_batch_processor", None)
    for obj in (processor, inner):
        if obj is None:
            continue
        for attr in ("span_exporter", "_exporter", "_span_exporter"):
            exporter = getattr(obj, attr, None)
            if exporter is not None:
                return exporter
    return None


def _bsp_attr(processor, name: str) -> object:
    """Read a BatchSpanProcessor config knob across SDK layouts.

    Different opentelemetry-python versions expose these as ``max_queue_size`` on
    the processor, or as ``_max_queue_size`` on an inner ``_batch_processor``
    delegate. Try each before giving up so the config line isn't all ``?``.
    """
    inner = getattr(processor, "_batch_processor", None)
    for obj in (processor, inner):
        if obj is None:
            continue
        for attr in (name, f"_{name}"):
            value = getattr(obj, attr, None)
            if value is not None:
                return value
    return "?"


def _log_bsp_config_once(processor) -> None:
    """Log the BatchSpanProcessor's queue/batch/timeout settings exactly once."""
    global _BSP_CONFIG_LOGGED
    if _BSP_CONFIG_LOGGED:
        return
    with _BSP_CONFIG_LOCK:
        if _BSP_CONFIG_LOGGED:
            return
        _BSP_CONFIG_LOGGED = True
    print(
        "OTEL DEBUG BSP CONFIG "
        f"max_queue_size={_bsp_attr(processor, 'max_queue_size')} "
        f"max_export_batch_size={_bsp_attr(processor, 'max_export_batch_size')} "
        f"schedule_delay_millis={_bsp_attr(processor, 'schedule_delay_millis')} "
        f"export_timeout_millis={_bsp_attr(processor, 'export_timeout_millis')} "
        f"otlp_timeout_env={os.getenv('OTEL_EXPORTER_OTLP_TIMEOUT', '<unset>')} "
        f"(export_timeout_millis=BSP per-flush deadline; otlp_timeout_env=per-export "
        f"OTLP client timeout — see timeout_s on each EXPORT line for the resolved value)",
        file=sys.stderr,
        flush=True,
    )


def _make_logging_export(original):
    def export(self, spans):
        n = len(spans)
        endpoint = (
            os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
            or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            or "<unset>"
        )
        protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
        timeout_s = _exporter_timeout_s(self)
        dest = _dest_of_exporter(self)
        start = time.monotonic()
        try:
            result = original(self, spans)
        except BaseException as error:
            # An exception escaping export() means the batch was not shipped —
            # tally it as failed (lost), not exported.
            _TALLY.record_export(n, ok=False, dest=dest)
            duration_ms = (time.monotonic() - start) * 1000
            print(
                f"OTEL DEBUG EXPORT spans={n} result=EXCEPTION dest={dest} "
                f"duration_ms={duration_ms:.1f} timeout_s={timeout_s} "
                f"protocol={protocol} endpoint={endpoint} error={error!r}",
                file=sys.stderr,
                flush=True,
            )
            raise
        duration_ms = (time.monotonic() - start) * 1000
        result_name = getattr(result, "name", str(result))
        # SpanExportResult.SUCCESS.name == "SUCCESS"; anything else (FAILURE) means
        # the SDK exhausted its retries and dropped the batch — count it as lost.
        _TALLY.record_export(n, ok=result_name == "SUCCESS", dest=dest)
        print(
            f"OTEL DEBUG EXPORT spans={n} result={result_name} dest={dest} "
            f"duration_ms={duration_ms:.1f} timeout_s={timeout_s} "
            f"protocol={protocol} endpoint={endpoint}",
            file=sys.stderr,
            flush=True,
        )
        return result

    return export


def _make_counting_on_end(original):
    def on_end(self, span):
        _log_bsp_config_once(self)
        # Attribute the produced span to the destination of THIS processor's
        # exporter, so prod and staging produced-counts stay separate.
        dest = _dest_of_exporter(_processor_exporter(self))
        _TALLY.record_produced(1, dest=dest)
        return original(self, span)

    return on_end


def _patch_exporters() -> int:
    """Patch OTLP exporter classes' ``export()`` to log + tally each call."""
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


def _patch_batch_processor() -> int:
    """Patch ``BatchSpanProcessor.on_end`` to tally every span produced."""
    try:
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        return 0
    if getattr(BatchSpanProcessor, _PATCH_FLAG, False):
        return 0
    BatchSpanProcessor.on_end = _make_counting_on_end(BatchSpanProcessor.on_end)
    setattr(BatchSpanProcessor, _PATCH_FLAG, True)
    return 1


def _register_summary() -> None:
    """Print the produced-vs-exported tally once, at process exit."""
    global _SUMMARY_REGISTERED
    if _SUMMARY_REGISTERED:
        return
    atexit.register(lambda: print(_TALLY.summary_line(), file=sys.stderr, flush=True))
    _SUMMARY_REGISTERED = True


def install_debug_logging() -> int:
    """Monkey-patch the OTLP export path to log each call and tally spans.

    Patches at the class level so it works regardless of how the SDK pipes spans
    through processors (incl. SDKs where ``BatchSpanProcessor`` exposes
    ``span_exporter`` as a read-only property), counts spans entering every
    ``BatchSpanProcessor``, and registers a per-process exit summary.

    Returns the number of exporter classes patched (the batch-processor patch
    and the exit summary are best-effort side effects).
    """
    patched = _patch_exporters()
    _patch_batch_processor()
    _register_summary()
    return patched
