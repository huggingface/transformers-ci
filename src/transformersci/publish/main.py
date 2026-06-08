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
import subprocess
import sys
from pathlib import Path

from .data_card import render_data_card
from .manifest import write_manifest
from .tables import (
    RUN_ROLLUP_COLUMNS,
    TEST_ROW_COLUMNS,
    RunRollupAccumulator,
    group_by_day,
    shape_trace_rows,
    write_parquet,
)
from .tempo_window import iter_window_traces

DEFAULT_STAGING_DIR = "/staging"
DEFAULT_BUCKET_URI = "hf://buckets/huggingface/transformers-ci-telemetry"


def _log(message: str) -> None:
    print(f"[ci-data-publisher] {message}", file=sys.stderr, flush=True)


def _write_raw_trace(staging: Path, day: str, trace_id: str, trace: dict) -> None:
    trace_dir = staging / "daily" / day / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / f"{trace_id}.json").write_text(
        json.dumps(trace, separators=(",", ":")), encoding="utf-8"
    )


def run_cycle(staging_dir: str, bucket_uri: str) -> dict:
    """Build all partitions for the current window into ``staging_dir``.

    Streams traces one at a time: each trace's raw JSON is written to its day
    partition immediately and the trace is then dropped, so peak memory stays
    flat regardless of how many (large) traces the window holds — this is what
    keeps the sidecar under its memory cap. Only the small derived rows and
    rollup aggregates are retained until the per-day Parquet is written.
    """
    staging = Path(staging_dir)
    staging.mkdir(parents=True, exist_ok=True)

    test_rows_by_day: dict[str, list[dict]] = {}
    rollups = RunRollupAccumulator()
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
        test_rows_by_day.setdefault(day, []).extend(rows)
        if trace_id:
            _write_raw_trace(staging, day, trace_id, trace)
        # `trace` (potentially many MB) is now free to be reclaimed.

    _log(f"fetched {n_traces} trace(s) from Tempo over the publish window")

    rollup_rows = rollups.rows()
    roll_by_day = group_by_day(rollup_rows)
    days = sorted(
        d for d in (set(test_rows_by_day) | set(roll_by_day)) if d and d != "unknown"
    )
    n_rows = 0
    for day in days:
        day_dir = staging / "daily" / day
        day_dir.mkdir(parents=True, exist_ok=True)
        day_rows = test_rows_by_day.get(day, [])
        if day_rows:
            write_parquet(day_rows, TEST_ROW_COLUMNS, day_dir / "test_rows.parquet")
            n_rows += len(day_rows)
        day_rolls = roll_by_day.get(day, [])
        if day_rolls:
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

    run_cycle(args.staging_dir, args.bucket_uri)

    if args.dry_run:
        _log("--dry-run: built locally, not syncing")
        return 0
    if args.sync:
        sync_to_bucket(args.staging_dir, args.bucket_uri)
        _log("synced to bucket")
    return 0


if __name__ == "__main__":
    sys.exit(main())
