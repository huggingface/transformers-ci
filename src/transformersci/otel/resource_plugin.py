from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest


try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def detect_provider() -> str:
    if os.getenv("GITHUB_ACTIONS"):
        return "github_actions"
    if os.getenv("CIRCLECI") or os.getenv("CIRCLE_WORKFLOW_ID"):
        return "circleci"
    return "local"


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("transformersci-otel")
    group.addoption(
        "--resource-metrics-file",
        action="store",
        default=None,
        help="Write per-test CPU, RSS, and optional CUDA metrics to the given JSONL file.",
    )


def split_pytest_nodeid(nodeid: str) -> dict[str, str]:
    parts = nodeid.split("::")
    module_name = Path(parts[0]).name if parts else ""
    if len(parts) >= 3:
        class_name = parts[-2]
        function_name = parts[-1]
    elif len(parts) == 2:
        class_name = ""
        function_name = parts[-1]
    else:
        class_name = ""
        function_name = ""

    return {
        "test_class": class_name,
        "test_function": function_name,
        "test_module": module_name,
    }


class ResourceSampler:
    def __init__(self) -> None:
        if psutil is None:  # pragma: no cover
            raise RuntimeError("psutil is required for pytest resource monitoring")

        self.process = psutil.Process(os.getpid())
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.start_cpu_time = self._cpu_time_seconds()
        self.start_rss_bytes = self._rss_bytes()
        self.peak_rss_bytes = self.start_rss_bytes
        self.cuda_available = bool(torch is not None and torch.cuda.is_available())
        self.start_cuda_allocated_bytes = self._cuda_allocated_bytes()
        self.peak_cuda_allocated_bytes = self.start_cuda_allocated_bytes

    def _cpu_time_seconds(self) -> float:
        cpu_times = self.process.cpu_times()
        return float(cpu_times.user + cpu_times.system)

    def _rss_bytes(self) -> int:
        return int(self.process.memory_info().rss)

    def _cuda_allocated_bytes(self) -> int:
        if not self.cuda_available:
            return 0
        try:
            return int(torch.cuda.memory_allocated())  # type: ignore[union-attr]
        except Exception:  # pragma: no cover
            return 0

    def _sample_loop(self) -> None:
        while not self.stop_event.wait(0.05):
            self.peak_rss_bytes = max(self.peak_rss_bytes, self._rss_bytes())
            if self.cuda_available:
                self.peak_cuda_allocated_bytes = max(
                    self.peak_cuda_allocated_bytes, self._cuda_allocated_bytes()
                )

    def start(self) -> None:
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()

    def stop(self) -> dict[str, float | int]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1)

        end_rss_bytes = self._rss_bytes()
        end_cpu_time = self._cpu_time_seconds()
        end_cuda_allocated_bytes = self._cuda_allocated_bytes()
        self.peak_rss_bytes = max(self.peak_rss_bytes, end_rss_bytes)
        self.peak_cuda_allocated_bytes = max(
            self.peak_cuda_allocated_bytes, end_cuda_allocated_bytes
        )

        return {
            "cpu_time_seconds": max(0.0, end_cpu_time - self.start_cpu_time),
            "rss_delta_bytes": end_rss_bytes - self.start_rss_bytes,
            "rss_end_bytes": end_rss_bytes,
            "rss_peak_bytes": self.peak_rss_bytes,
            "cuda_end_allocated_bytes": end_cuda_allocated_bytes,
            "cuda_peak_allocated_bytes": self.peak_cuda_allocated_bytes,
        }


def metrics_file_path(config: pytest.Config | None = None) -> Path | None:
    raw_path = None
    if config is not None:
        raw_path = config.getoption("resource_metrics_file", default=None)
    raw_path = (
        raw_path
        or os.getenv("PYTEST_RESOURCE_METRICS_FILE")
        or os.getenv("TRANSFORMERS_TEST_RESOURCE_METRICS_FILE")
    )
    if not raw_path:
        return None
    return Path(raw_path)


def write_resource_record(item: pytest.Item, metrics: dict[str, float | int]) -> None:
    path = metrics_file_path(item.config)
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    node_parts = split_pytest_nodeid(item.nodeid)
    test_job = os.getenv("TRANSFORMERS_TEST_OTEL_JOB") or os.getenv(
        "TRANSFORMERS_TEST_OTEL_SUITE", "local_pytest"
    )
    record: dict[str, Any] = {
        "pr": os.getenv("TRANSFORMERS_TEST_OTEL_PR", "none"),
        "provider": detect_provider(),
        "run_id": os.getenv("TRANSFORMERS_TEST_OTEL_RUN_ID", "unknown"),
        "service_name": os.getenv("OTEL_SERVICE_NAME", "transformers-tests"),
        "test_job": test_job,
        "test_nodeid": item.nodeid,
        "timestamp": time.time(),
        **node_parts,
        **metrics,
    }
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")


def _parse_otlp_headers(raw: str | None) -> dict[str, str] | None:
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


# Staging is a best-effort mirror: bound each export attempt so a dead/black-holed
# staging box can't make the end-of-session flush hang the CI job for long. A
# healthy same-network mirror exports well under a second, so a few seconds is
# plenty; when staging is down this caps the extra teardown delay per shard.
STAGING_EXPORT_TIMEOUT_SECONDS = 5


def _build_staging_exporter(endpoint: str, protocol: str, headers):
    """Build an OTLP span exporter for the staging mirror.

    Mirrors the primary transport so spans land in the same shape on staging:
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
    return OTLPSpanExporter(
        endpoint=endpoint,
        headers=headers,
        insecure=insecure,
        timeout=STAGING_EXPORT_TIMEOUT_SECONDS,
    )


def _install_staging_span_processor() -> None:
    """Mirror every span to a staging backend via a second span processor.

    The primary export pipeline is configured by pytest-opentelemetry from
    ``OTEL_EXPORTER_OTLP_*`` env vars (a single exporter). When
    ``TRANSFORMERS_TEST_OTEL_STAGING_ENDPOINT`` is set (by
    ``configure-ci-otel --staging-endpoint``), we attach a SECOND
    ``BatchSpanProcessor`` to the live SDK tracer provider so the same spans are
    also exported to staging. Staging auth comes from its own headers env
    (``TRANSFORMERS_TEST_OTEL_STAGING_HEADERS``), falling back to the primary's.
    """
    endpoint = os.getenv("TRANSFORMERS_TEST_OTEL_STAGING_ENDPOINT")
    if not endpoint:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as error:  # pragma: no cover
        print(
            f"OTEL STAGING SDK unavailable, not mirroring to {endpoint}: {error!r}",
            file=sys.stderr,
            flush=True,
        )
        return

    provider = trace.get_tracer_provider()
    add_span_processor = getattr(provider, "add_span_processor", None)
    if add_span_processor is None:
        # No real SDK tracer provider is active (primary trace export is off),
        # so there is nothing to mirror.
        print(
            f"OTEL STAGING no active SDK tracer provider; not mirroring to {endpoint}",
            file=sys.stderr,
            flush=True,
        )
        return

    protocol = (
        os.getenv("TRANSFORMERS_TEST_OTEL_STAGING_PROTOCOL")
        or os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL")
        or "grpc"
    ).lower()
    headers = _parse_otlp_headers(
        os.getenv("TRANSFORMERS_TEST_OTEL_STAGING_HEADERS")
        or os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
    )
    # Staging is best-effort: any failure building/attaching the mirror must
    # only warn, never break the primary export or the test run. (Runtime
    # export failures are already swallowed by the SDK's BatchSpanProcessor.)
    try:
        exporter = _build_staging_exporter(endpoint, protocol, headers)
        add_span_processor(BatchSpanProcessor(exporter))
    except Exception as error:
        print(
            f"OTEL STAGING WARNING could not attach mirror to {endpoint} "
            f"(protocol={protocol}); primary export unaffected: {error!r}",
            file=sys.stderr,
            flush=True,
        )
        return

    print(
        f"OTEL STAGING mirroring spans to {endpoint} (protocol={protocol})",
        file=sys.stderr,
        flush=True,
    )


def _wrap_active_tracer_exporters() -> None:
    if os.getenv("TRANSFORMERSCI_OTEL_DEBUG") != "1":
        return

    try:
        from .debug_exporter import install_debug_logging
    except ImportError as error:
        print(
            f"OTEL DEBUG could not import debug_exporter: {error!r}",
            file=sys.stderr,
            flush=True,
        )
        return

    patched = install_debug_logging()
    if patched == 0:
        print(
            "OTEL DEBUG no OTLP exporter classes available to patch "
            "(opentelemetry-exporter-otlp not installed?)",
            file=sys.stderr,
            flush=True,
        )
        return

    print(
        f"OTEL DEBUG patched export() on {patched} OTLP exporter class(es)",
        file=sys.stderr,
        flush=True,
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    # The staging mirror is strictly best-effort. Guard the whole call so no
    # staging-related error can abort session start (which would break the
    # primary trace export and the test run for prod).
    try:
        _install_staging_span_processor()
    except Exception as error:  # pragma: no cover - defensive belt-and-suspenders
        print(
            f"OTEL STAGING WARNING staging mirror setup failed; "
            f"primary export unaffected: {error!r}",
            file=sys.stderr,
            flush=True,
        )
    _wrap_active_tracer_exporters()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Any:
    if psutil is None or metrics_file_path(item.config) is None:
        yield
        return

    sampler = ResourceSampler()
    sampler.start()
    outcome = yield
    metrics = sampler.stop()
    write_resource_record(item, metrics)
    return outcome
