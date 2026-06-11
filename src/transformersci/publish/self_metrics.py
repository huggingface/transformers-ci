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
"""Self-observability metrics for the publisher batch job.

The publisher is a cron job that runs and exits, so there is no long-lived
endpoint for Prometheus to scrape. Instead each cycle writes a Prometheus
text-exposition file (``*.prom``) that the already-running ``node-exporter``
picks up via its textfile collector — the standard pattern for batch jobs. The
metrics land in the ``prometheus-infra`` instance alongside the trace exporter's
self-metrics and surface on the CI Health dashboard.

:func:`render_self_metrics` builds the text body (pure, unit-testable) and
:func:`write_self_metrics` publishes it atomically (temp file + ``os.replace``)
so node-exporter never reads a half-written file. Writing is best-effort: a
metrics failure must never fail an actual publish.
"""

from __future__ import annotations

import os
import resource
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_METRICS_FILE = "/textfile/ci_publisher.prom"

# name -> (prom type, HELP text). Only the keys present in the values dict are
# emitted, so a failed cycle (which has no throughput numbers) still publishes
# the health gauges it does have.
_METRICS: dict[str, tuple[str, str]] = {
    "ci_publisher_up": (
        "gauge",
        "1 if the publisher reached the metrics-writing step this cycle.",
    ),
    "ci_publisher_last_run_success": (
        "gauge",
        "1 if the last cycle built (and synced) without error, else 0.",
    ),
    "ci_publisher_last_run_timestamp_seconds": (
        "gauge",
        "Unix time the last cycle finished.",
    ),
    "ci_publisher_run_duration_seconds": (
        "gauge",
        "Wall-clock duration of the last cycle.",
    ),
    "ci_publisher_peak_rss_bytes": (
        "gauge",
        "Peak resident memory of the publisher process this cycle.",
    ),
    "ci_publisher_traces_published": (
        "gauge",
        "Traces shaped and written in the last cycle.",
    ),
    "ci_publisher_traces_per_second": (
        "gauge",
        "Traces processed per second in the last cycle.",
    ),
    "ci_publisher_test_rows_published": (
        "gauge",
        "Test rows written in the last cycle.",
    ),
    "ci_publisher_run_rollups_published": (
        "gauge",
        "Run-rollup rows written in the last cycle.",
    ),
    "ci_publisher_dataset_test_rows": (
        "gauge",
        "Total test rows in the published dataset (all partitions).",
    ),
    "ci_publisher_dataset_run_rollups": (
        "gauge",
        "Total run-rollup rows in the published dataset.",
    ),
    "ci_publisher_dataset_partitions": (
        "gauge",
        "Number of day partitions in the published dataset.",
    ),
    "ci_publisher_dataset_bytes": (
        "gauge",
        "Total bytes of the published dataset on disk (staging mirror).",
    ),
}


def metrics_path() -> Path:
    return Path(os.getenv("PUBLISH_METRICS_FILE", DEFAULT_METRICS_FILE))


def peak_rss_bytes() -> int:
    """Peak resident set size of this process, in bytes.

    ``ru_maxrss`` is kilobytes on Linux but bytes on macOS — normalize so the
    metric is always bytes regardless of where the cycle ran.
    """
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maxrss) if sys.platform == "darwin" else int(maxrss) * 1024


def directory_bytes(path: Path) -> int:
    """Total size in bytes of all files under ``path`` (0 if it doesn't exist)."""
    total = 0
    if not path.exists():
        return 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                pass
    return total


def _format_value(value: float) -> str:
    """Full-precision Prometheus float rendering.

    ``%g`` would collapse large values to 6 significant figures (a unix
    timestamp became ``1.78117e+09``, losing ~hundreds of seconds and breaking
    ``time() - last_run``). Emit whole numbers as plain integers and everything
    else via ``repr`` (shortest round-trippable form).
    """
    f = float(value)
    if f.is_integer():
        return str(int(f))
    return repr(f)


def render_self_metrics(values: dict[str, float]) -> str:
    """Render the Prometheus text-exposition body for the given metric values.

    Emits HELP/TYPE plus one sample line for each known metric present in
    ``values``; unknown keys are ignored so callers can't smuggle untyped lines.
    """
    lines: list[str] = []
    for name, (mtype, help_text) in _METRICS.items():
        if name not in values:
            continue
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {mtype}")
        lines.append(f"{name} {_format_value(values[name])}")
    return "\n".join(lines) + "\n"


def write_self_metrics(values: dict[str, float], path: Path | None = None) -> None:
    """Publish the metrics body atomically so node-exporter never reads a
    partial file. Best-effort: any failure is swallowed (metrics must not break
    a publish), but the temp file is cleaned up on error."""
    target = path or metrics_path()
    body = render_self_metrics(values)
    try:
        # Temp file shares the destination dir (atomic rename) but must NOT end
        # in .prom, or node-exporter would scrape the half-written temp.
        fd, tmp_name = tempfile.mkstemp(
            dir=str(target.parent), prefix=target.name + ".", suffix=".tmp"
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(body)
            # mkstemp creates 0600; node-exporter runs as an unprivileged user
            # and must be able to read the file, or its textfile collector
            # reports node_textfile_scrape_error and drops every sample.
            os.chmod(tmp, 0o644)
            os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    except OSError as error:
        print(
            f"[ci-data-publisher] could not write metrics to {target}: {error}",
            file=sys.stderr,
            flush=True,
        )


def collect_values(
    *,
    success: bool,
    duration_seconds: float,
    stats: dict,
    manifest: dict | None,
    dataset_bytes: int,
) -> dict[str, float]:
    """Assemble the metric values dict from one cycle's outcome."""
    values: dict[str, float] = {
        "ci_publisher_up": 1,
        "ci_publisher_last_run_success": 1 if success else 0,
        "ci_publisher_last_run_timestamp_seconds": round(time.time(), 3),
        "ci_publisher_run_duration_seconds": round(duration_seconds, 3),
        "ci_publisher_peak_rss_bytes": peak_rss_bytes(),
        "ci_publisher_dataset_bytes": dataset_bytes,
    }
    traces = stats.get("traces")
    if traces is not None:
        values["ci_publisher_traces_published"] = traces
        values["ci_publisher_traces_per_second"] = (
            traces / duration_seconds if duration_seconds > 0 else 0
        )
    if "test_rows" in stats:
        values["ci_publisher_test_rows_published"] = stats["test_rows"]
    if "run_rollups" in stats:
        values["ci_publisher_run_rollups_published"] = stats["run_rollups"]
    if manifest is not None:
        values["ci_publisher_dataset_test_rows"] = manifest.get("total_test_rows", 0)
        values["ci_publisher_dataset_run_rollups"] = manifest.get(
            "total_run_rollups", 0
        )
        values["ci_publisher_dataset_partitions"] = manifest.get("partition_count", 0)
    return values
