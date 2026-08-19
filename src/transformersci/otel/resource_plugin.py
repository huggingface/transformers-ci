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
mirror), ``pytest_runtest_protocol`` (sample around each test), and
``pytest_runtest_logreport`` (stamp each span with the real worker-measured
execution time so dashboards are not misled by xdist queue-wait overhead).
"""

from __future__ import annotations

import gc
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
    group.addoption(
        "--resource-gc-probe",
        action="store_true",
        default=False,
        help=(
            "Also record CUDA memory after a gc.collect() at the end of each test, "
            "which distinguishes uncollected garbage from a live reference. Off by "
            "default: collecting mid-run perturbs what is being measured."
        ),
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

    # Emit ONE span per test: drop pytest-opentelemetry's ::setup/::call/::teardown
    # phase spans and fixture spans (~83% of a shard trace, all unused by our
    # dashboards). Keeps shard traces under Tempo's read-path limit so the biggest
    # jobs (e.g. tests_torch) don't 500 on fetch and vanish. See
    # transformersci.otel.span_pruning for the full write-up.
    from . import span_pruning

    span_pruning.install()


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
    """Per-test CPU/RSS/CUDA sampling.

    ``gc_probe`` adds one extra measurement: after the raw end-of-test reading,
    run ``gc.collect()`` and read CUDA memory again. That second number is what
    separates the two reasons a test leaves device memory behind — see
    ``cuda_delta_after_gc_bytes`` in :meth:`stop`. It is opt-in because
    collecting mid-run perturbs the very behaviour we are measuring.
    """

    def __init__(self, gc_probe: bool = False) -> None:
        if psutil is None:  # pragma: no cover
            raise RuntimeError("psutil is required for pytest resource monitoring")

        self.gc_probe = gc_probe
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

        metrics: dict[str, float | int] = {
            "cpu_time_seconds": max(0.0, end_cpu_time - self.start_cpu_time),
            "rss_delta_bytes": end_rss_bytes - self.start_rss_bytes,
            "rss_end_bytes": end_rss_bytes,
            "rss_peak_bytes": self.peak_rss_bytes,
            "cuda_end_allocated_bytes": end_cuda_allocated_bytes,
            "cuda_peak_allocated_bytes": self.peak_cuda_allocated_bytes,
            # Device memory this test hands to the NEXT test in the same process.
            # The absolute end/peak numbers cannot say who is responsible — every
            # test after a leaky one looks huge. The delta attributes it. This is
            # the signal behind the daily OOM groups: a test that retains a
            # multi-GB model leaves the rest of its file to fail on a full card,
            # asking for tens of MiB.
            "cuda_start_allocated_bytes": self.start_cuda_allocated_bytes,
            "cuda_delta_bytes": end_cuda_allocated_bytes
            - self.start_cuda_allocated_bytes,
        }
        if self.gc_probe and self.cuda_available:
            # Why a test retained memory, not just that it did:
            #   delta > 0 and after_gc ~ 0  → unreferenced garbage CPython had not
            #       collected yet. A `cleanup(torch_device, gc_collect=True)` in
            #       tearDown fixes it.
            #   delta > 0 and after_gc > 0  → something still HOLDS a reference
            #       (a class attribute, or a failed test's traceback pinning the
            #       frame that owns the model). No tearDown cleanup can free that,
            #       so the fix has to drop the reference itself.
            gc.collect()
            after_gc = self._cuda_allocated_bytes()
            metrics["cuda_end_after_gc_bytes"] = after_gc
            metrics["cuda_delta_after_gc_bytes"] = (
                after_gc - self.start_cuda_allocated_bytes
            )
        return metrics


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


def gc_probe_enabled(config: pytest.Config | None = None) -> bool:
    """Whether to take the post-``gc.collect()`` CUDA reading. Env var mirrors the
    flag so a CI job can turn it on without changing its pytest invocation."""
    if config is not None and config.getoption("resource_gc_probe", default=False):
        return True
    return (os.getenv("PYTEST_RESOURCE_GC_PROBE") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


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


# Accumulated worker-side phase durations (setup + call + teardown) keyed by
# nodeid.  Populated by pytest_runtest_logreport; entries are removed once the
# teardown phase report arrives and the span attribute has been stamped.
_worker_durations: dict[str, float] = {}

# CUDA bytes allocated when each test's setup began, keyed by nodeid. Lives in
# the WORKER process (see pytest_runtest_makereport) because that is where the
# allocations are: under xdist the controller holds no CUDA memory at all.
_cuda_at_setup: dict[str, int] = {}

# The user_property carrying retained device memory from worker to controller.
CUDA_DELTA_PROPERTY = "cuda_delta_bytes"

# Same channel, for the post-gc.collect() reading when the probe is on. It has to
# ride the span like the raw delta does: the resource JSONL it also lands in has
# no transport out of a GitHub runner, so a probe run that only wrote there would
# produce a number nobody can read.
CUDA_DELTA_AFTER_GC_PROPERTY = "cuda_delta_after_gc_bytes"

# The two ABSOLUTE readings the delta is computed from. The delta alone cannot
# be read: it is a net change over the test's window, so a negative one is
# ambiguous — `-17.5 GiB` is equally "inherited 17.5, ended at 0" and "inherited
# 20, ended at 2.5". Prod is full of them (6767 series on 2026-08-18; gpt_oss's
# `test_model_outputs_02` reads -17.50 GiB while allocating nothing itself — it
# fails on an ImportError and merely happens to straddle the release of an
# earlier test's 18.49 GiB).
#
# `inherited` is also the only thing that answers "what did this test start
# with", which is the question an OOM actually turns on: a test asking for a few
# MiB on a card someone else filled.
CUDA_INHERITED_PROPERTY = "cuda_inherited_bytes"
CUDA_RETAINED_PROPERTY = "cuda_retained_bytes"


def _cuda_allocated_now() -> int | None:
    """CUDA bytes currently allocated, or ``None`` when there is no CUDA to read
    (no torch, CPU-only job, or a torch that raises on a driverless box)."""
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.memory_allocated())
    except Exception:  # pragma: no cover - defensive: never break a test run
        return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> Any:
    """Attach retained device memory to the test's report, worker-side.

    This is the transport, and it has to be this hook. The span is stamped on the
    *controller* (``pytest_runtest_logreport``), but under pytest-xdist the
    controller process never allocates CUDA memory — the worker does. So the
    worker measures here and hands the number over on ``report.user_properties``,
    which xdist serialises into the controller's copy of the report. It is the
    same worker→controller hop ``report.duration`` already rides.

    Measured from the start of *setup* to the end of *teardown*, so a fixture that
    allocates is attributed to the test that used it. No ``gc.collect()`` — the
    number we want is what the next test actually inherits, not what it would
    inherit if something collected first.
    """
    outcome = yield
    if call.when == "setup":
        allocated = _cuda_allocated_now()
        if allocated is not None:
            _cuda_at_setup[item.nodeid] = allocated
        return
    if call.when != "teardown":
        return
    start = _cuda_at_setup.pop(item.nodeid, None)
    end = _cuda_allocated_now()
    if start is None or end is None:
        return
    try:
        report = outcome.get_result()
    except Exception:  # pragma: no cover - a failed report is not ours to fix
        return
    report.user_properties.append((CUDA_DELTA_PROPERTY, end - start))
    # Both sides of that subtraction, so the delta stops being ambiguous: `end`
    # is what the next test really inherits (never negative), `start` is what
    # this one was handed.
    report.user_properties.append((CUDA_INHERITED_PROPERTY, start))
    report.user_properties.append((CUDA_RETAINED_PROPERTY, end))
    if not gc_probe_enabled(item.config):
        return
    # Opt-in second reading: how much of that delta survives a collection. It is
    # what separates "a tearDown would fix this" from "something holds a live
    # reference and no tearDown can". Taken after the raw number is already
    # recorded, so enabling the probe cannot change what `cuda_delta_bytes` says.
    gc.collect()
    after_gc = _cuda_allocated_now()
    if after_gc is not None:
        report.user_properties.append((CUDA_DELTA_AFTER_GC_PROPERTY, after_gc - start))


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Stamp the open test span with the real worker-measured execution time.

    Under pytest-xdist the per-test span is opened/closed on the *controller*
    by pytest-opentelemetry's ``pytest_runtest_protocol`` hookwrapper, but its
    wall-clock duration includes xdist queue-wait time and IPC overhead — not
    just the test's execution time.  ``report.duration`` is measured on the
    *worker* for each phase (setup / call / teardown), so summing the three
    phases gives the real time the worker spent on the test.

    We set ``pytest.worker_duration_seconds`` on the still-open protocol span
    (``pytest_runtest_logreport`` is called while still inside the hookwrapper's
    ``yield``) so dashboards can use it instead of the misleading span duration.
    In non-xdist runs the two values are identical; the attribute is still set
    for consistency.
    """
    _worker_durations[report.nodeid] = (
        _worker_durations.get(report.nodeid, 0.0) + report.duration
    )

    if report.when != "teardown":
        return

    total = _worker_durations.pop(report.nodeid, 0.0)

    try:
        from opentelemetry import trace
    except ImportError:  # pragma: no cover
        return

    span = trace.get_current_span()
    if not span.is_recording():
        return

    span.set_attribute("pytest.worker_duration_seconds", total)

    # Device memory this test leaves behind for the next test in its process,
    # measured in the worker and carried here on the report (see
    # pytest_runtest_makereport). Absent for CPU jobs and for any run without
    # torch/CUDA, in which case no attribute is set at all — the exporter then
    # emits no series rather than a misleading zero.
    properties = dict(report.user_properties)
    retained = properties.get(CUDA_DELTA_PROPERTY)
    if isinstance(retained, int):
        span.set_attribute("pytest.cuda_delta_bytes", retained)
    # Additive: an older exporter ignores these, and a run without them keeps
    # emitting exactly the series it did before.
    for prop, attribute in (
        (CUDA_INHERITED_PROPERTY, "pytest.cuda_inherited_bytes"),
        (CUDA_RETAINED_PROPERTY, "pytest.cuda_retained_bytes"),
    ):
        value = properties.get(prop)
        if isinstance(value, int):
            span.set_attribute(attribute, value)
    # Only present when the gc probe ran. Its whole point is to be readable
    # somewhere, so it rides the span rather than only the untransported JSONL.
    after_gc = properties.get(CUDA_DELTA_AFTER_GC_PROPERTY)
    if isinstance(after_gc, int):
        span.set_attribute("pytest.cuda_delta_after_gc_bytes", after_gc)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Any:
    if psutil is None or metrics_file_path(item.config) is None:
        yield
        return

    sampler = ResourceSampler(gc_probe=gc_probe_enabled(item.config))
    sampler.start()
    outcome = yield
    metrics = sampler.stop()
    write_resource_record(item, metrics)
    return outcome
