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
"""Pytest plugin that enriches CI runs with resource samples and a span mirror.

Loaded into the test session itself (producer side), it does two largely
independent jobs via pytest hooks:

- Per-test resource sampling — :class:`ResourceSampler` polls CPU/memory (and GPU
  via torch, when present) around each test, and :func:`write_resource_record`
  appends one JSONL row per test to a metrics file. The trace exporter later
  reads this file to produce resource panels, keeping bulky samples out of the
  spans themselves.
- Staging span mirror — :func:`_install_staging_span_processor` attaches a
  SECOND span processor so every span is shipped to a staging backend on top of
  the primary export, using its own endpoint/headers/protocol so staging can
  differ from prod (this is where the gRPC lowercase-``authorization`` fix
  lives).

Hooks: ``pytest_addoption`` (config), ``pytest_sessionstart`` (install the
mirror), and ``pytest_runtest_protocol`` (sample around each test).
"""

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


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    # Make OTel span/trace ids immune to the suite's reproducibility seeding.
    # OTel's default id generator pulls from the GLOBAL random module, which
    # transformers' set_seed()/random.seed() reseed on nearly every test — making
    # span_ids deterministic and COLLIDE across xdist workers (and tests) that
    # share a trace_id. Tempo then overwrites the duplicate (trace_id, span_id)
    # keys, silently dropping ~⅔–¾ of every shard trace's spans with no error.
    # Runs in pytest_configure so it is in place before the first span. See
    # transformersci.otel.id_generator for the full write-up.
    from . import id_generator

    id_generator.install()


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


# Exporter construction lives in ._export so the pytest staging mirror and the
# non-pytest in-process tracer (._instrument) share one code path (and one copy
# of the HTTP signal-path / gRPC lowercase-header handling). The private aliases
# below preserve this module's existing names so tests and monkeypatching of
# ``_build_staging_exporter`` keep working.
from ._export import STAGING_EXPORT_DISABLED  # noqa: E402
from ._export import STAGING_EXPORT_TIMEOUT_SECONDS  # noqa: E402,F401
from ._export import build_exporter as _build_staging_exporter  # noqa: E402
from ._export import parse_otlp_headers as _parse_otlp_headers  # noqa: E402


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
    if STAGING_EXPORT_DISABLED:
        # Hardcoded kill-switch (see _export.STAGING_EXPORT_DISABLED): the staging
        # mirror is off, so never attach the second span processor.
        return
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
