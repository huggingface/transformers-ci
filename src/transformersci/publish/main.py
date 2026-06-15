git# Copyright 2026 The HuggingFace Inc. team.
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
"""One publish cycle: Tempo window -> shaped tables + raw traces -> HF bucket.

Invoked once per run (the docker sidecar's cron fires it hourly). Each cycle:

1. fetch the lookback window of settled traces from Tempo,
2. shape them into ``test_rows`` / ``run_rollups`` and group by UTC day,
3. (re)write the affected ``daily/<date>/`` partitions in the staging dir,
   alongside the raw trace JSON,
4. refresh the data card (``README.md``) and manifest (``current_view.json``),
5. with ``--sync``, push the staging dir to the HF bucket via ``hf sync``.

Re-deriving whole day partitions and overwriting them is idempotent: settled
traces are immutable, so a day stabilises once it ages past the window and is
never rewritten again. ``hf sync`` only uploads what actually changed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import self_metrics
from .data_card import render_data_card
from .manifest import write_manifest
from .tables import (
    RUN_ROLLUP_COLUMNS,
    TEST_ROW_COLUMNS,
    RunRollupAccumulator,
    StreamingPartitionWriter,
    group_by_day,
    shape_trace_rows,
    write_parquet,
)
from .tempo_window import iter_window_traces

DEFAULT_STAGING_DIR = "/staging"
DEFAULT_BUCKET_URI = "hf://buckets/huggingface/transformers-ci-telemetry"
DEFAULT_ROW_BATCH = 2000
# Days of raw trace JSON to keep in the local staging mirror. The raw traces are
# the bulk of the mirror and the only thing that grows unbounded; everything
# older than this (and so already settled past the 24h publish window) is pruned
# locally after a successful sync. The small Parquet partitions are always kept,
# so the manifest's headline row counts stay exact, and `hf sync --no-delete`
# means the bucket retains every uploaded trace regardless.
DEFAULT_TRACE_RETENTION_DAYS = 2


def _trace_retention_days() -> int:
    raw = os.getenv("PUBLISH_TRACE_RETENTION_DAYS", "")
    try:
        return max(1, int(raw)) if raw else DEFAULT_TRACE_RETENTION_DAYS
    except ValueError:
        return DEFAULT_TRACE_RETENTION_DAYS


def prune_old_trace_partitions(staging_dir: str, keep_days: int) -> int:
    """Delete ``daily/<day>/traces/`` for partitions older than ``keep_days``.

    Returns the number of day partitions whose raw traces were removed. Only the
    bulky raw JSON is dropped; the Parquet files (and the partition dir) stay, so
    derived data and the manifest's row counts are untouched. Call only after a
    successful sync, so the pruned traces are safely in the bucket first.
    """
    daily = Path(staging_dir) / "daily"
    if not daily.is_dir():
        return 0
    cutoff = (
        datetime.now(tz=timezone.utc).date() - timedelta(days=keep_days)
    ).isoformat()
    pruned = 0
    for day_dir in daily.iterdir():
        # Day dirs are named YYYY-MM-DD, so lexicographic order == chronological.
        if not day_dir.is_dir() or day_dir.name >= cutoff:
            continue
        traces = day_dir / "traces"
        if traces.is_dir():
            shutil.rmtree(traces, ignore_errors=True)
            pruned += 1
    return pruned


def _row_batch_size() -> int:
    """Rows buffered per day before a Parquet row group is flushed.

    Bounds the publisher's peak memory: smaller = flatter memory, more (smaller)
    row groups. Override with ``PUBLISH_ROW_BATCH`` on the box if needed.
    """
    raw = os.getenv("PUBLISH_ROW_BATCH", "")
    try:
        return int(raw) if raw else DEFAULT_ROW_BATCH
    except ValueError:
        return DEFAULT_ROW_BATCH


def _log(message: str) -> None:
    print(f"[ci-data-publisher] {message}", file=sys.stderr, flush=True)


def _write_raw_trace(staging: Path, day: str, trace_id: str, trace: dict) -> None:
    trace_dir = staging / "daily" / day / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / f"{trace_id}.json").write_text(
        json.dumps(trace, separators=(",", ":")), encoding="utf-8"
    )


def run_cycle(staging_dir: str, bucket_uri: str, *, stats: dict | None = None) -> dict:
    """Build all partitions for the current window into ``staging_dir``.

    Streams traces one at a time: each trace's raw JSON is written to its day
    partition immediately and the trace is then dropped, so peak memory stays
    flat regardless of how many (large) traces the window holds — this is what
    keeps the sidecar under its memory cap. Only the small derived rows and
    rollup aggregates are retained until the per-day Parquet is written.

    If ``stats`` is passed it is populated with this cycle's throughput counts
    (``traces`` / ``test_rows`` / ``run_rollups`` / ``days``) for self-metrics;
    the return value stays the manifest so existing callers are unaffected.
    """
    staging = Path(staging_dir)
    staging.mkdir(parents=True, exist_ok=True)

    rollups = RunRollupAccumulator()
    # Test rows are streamed straight to per-day Parquet in bounded batches
    # rather than accumulated for the whole window — each row can carry a full
    # untruncated stacktrace, and holding them all is what OOM'd the sidecar.
    test_writer = StreamingPartitionWriter(
        staging / "daily", TEST_ROW_COLUMNS, batch_size=_row_batch_size()
    )
    n_traces = 0

    for trace in iter_window_traces():
        rows = shape_trace_rows(trace)
        if not rows:
            continue
        n_traces += 1
        rollups.add(trace)
        day = str(rows[0]["date"])
        trace_id = str(rows[0]["trace_id"])
        if not day or day == "unknown":
            continue
        test_writer.add(day, rows)
        if trace_id:
            _write_raw_trace(staging, day, trace_id, trace)
        # `trace` and its shaped `rows` are now free to be reclaimed.

    test_counts = test_writer.close()
    n_rows = sum(test_counts.values())
    _log(f"fetched {n_traces} trace(s) from Tempo over the publish window")

    rollup_rows = rollups.rows()
    roll_by_day = group_by_day(rollup_rows)
    days = sorted(
        d for d in (set(test_counts) | set(roll_by_day)) if d and d != "unknown"
    )
    for day in days:
        day_rolls = roll_by_day.get(day, [])
        if day_rolls:
            day_dir = staging / "daily" / day
            day_dir.mkdir(parents=True, exist_ok=True)
            write_parquet(
                day_rolls, RUN_ROLLUP_COLUMNS, day_dir / "run_rollups.parquet"
            )
    _log(
        f"wrote {n_rows} test row(s), {len(rollup_rows)} rollup(s) "
        f"across {len(days)} day partition(s)"
    )

    # Data card + manifest reflect the whole bucket, not just this window.
    (staging / "README.md").write_text(render_data_card(bucket_uri), encoding="utf-8")
    manifest = write_manifest(staging)
    _log(
        f"manifest: {manifest['partition_count']} partition(s), "
        f"{manifest['total_test_rows']} total test rows"
    )
    if stats is not None:
        stats.update(
            {
                "traces": n_traces,
                "test_rows": n_rows,
                "run_rollups": len(rollup_rows),
                "days": len(days),
            }
        )
    return manifest


def sync_to_bucket(staging_dir: str, bucket_uri: str) -> None:
    """Push the staging dir to the HF bucket with the ``hf`` CLI.

    Auth comes from ``HF_TOKEN`` in the environment, which ``hf`` reads.
    """
    cmd = ["hf", "sync", staging_dir, bucket_uri]
    _log("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staging-dir",
        default=os.getenv("PUBLISH_STAGING_DIR", DEFAULT_STAGING_DIR),
        help="local dir mirroring the bucket contents (default: $PUBLISH_STAGING_DIR or /staging)",
    )
    parser.add_argument(
        "--bucket-uri",
        default=os.getenv("HF_BUCKET_URI", DEFAULT_BUCKET_URI),
        help="hf:// destination synced with `hf sync` (default: $HF_BUCKET_URI)",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="after building, push the staging dir to the bucket via `hf sync`",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build partitions locally and never sync (overrides --sync)",
    )
    args = parser.parse_args(argv)

    # Time the whole cycle and always publish self-metrics — even on failure, so
    # the dashboard can tell a crashed/stalled publisher from an idle one. The
    # write is best-effort and never masks the real outcome.
    started = time.monotonic()
    stats: dict = {}
    manifest: dict | None = None
    success = False
    try:
        manifest = run_cycle(args.staging_dir, args.bucket_uri, stats=stats)
        if args.dry_run:
            _log("--dry-run: built locally, not syncing")
        elif args.sync:
            sync_to_bucket(args.staging_dir, args.bucket_uri)
            _log("synced to bucket")
            # Reclaim local disk only after the sync succeeded, so the bucket
            # holds the traces before we drop the local copies.
            keep = _trace_retention_days()
            pruned = prune_old_trace_partitions(args.staging_dir, keep)
            if pruned:
                _log(
                    f"pruned raw traces for {pruned} partition(s) older than "
                    f"{keep}d (kept Parquet; bucket retains them)"
                )
        success = True
    finally:
        self_metrics.write_self_metrics(
            self_metrics.collect_values(
                success=success,
                duration_seconds=time.monotonic() - started,
                stats=stats,
                manifest=manifest,
                dataset_bytes=self_metrics.directory_bytes(Path(args.staging_dir)),
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
