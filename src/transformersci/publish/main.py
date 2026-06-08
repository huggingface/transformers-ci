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
    build_run_rollups,
    build_test_rows,
    group_by_day,
    group_traces_by_day,
    write_parquet,
)
from .tempo_window import fetch_window

DEFAULT_STAGING_DIR = "/staging"
DEFAULT_BUCKET_URI = "hf://buckets/huggingface/transformers-ci-telemetry"


def _log(message: str) -> None:
    print(f"[ci-data-publisher] {message}", file=sys.stderr, flush=True)


def write_day_partition(
    staging: Path,
    day: str,
    test_rows: list[dict],
    rollups: list[dict],
    raw_traces: list[tuple[str, dict]],
) -> None:
    """Write (overwrite) one ``daily/<day>/`` partition."""
    day_dir = staging / "daily" / day
    day_dir.mkdir(parents=True, exist_ok=True)

    if test_rows:
        write_parquet(test_rows, TEST_ROW_COLUMNS, day_dir / "test_rows.parquet")
    if rollups:
        write_parquet(rollups, RUN_ROLLUP_COLUMNS, day_dir / "run_rollups.parquet")

    if raw_traces:
        trace_dir = day_dir / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        for trace_id, trace in raw_traces:
            (trace_dir / f"{trace_id}.json").write_text(
                json.dumps(trace, separators=(",", ":")), encoding="utf-8"
            )


def run_cycle(staging_dir: str, bucket_uri: str) -> dict:
    """Build all partitions for the current window into ``staging_dir``."""
    staging = Path(staging_dir)
    staging.mkdir(parents=True, exist_ok=True)

    traces = fetch_window()
    _log(f"fetched {len(traces)} trace(s) from Tempo over the publish window")

    test_rows = build_test_rows(traces)
    rollups = build_run_rollups(traces)
    test_by_day = group_by_day(test_rows)
    roll_by_day = group_by_day(rollups)
    traces_by_day = group_traces_by_day(traces)

    days = sorted(set(test_by_day) | set(roll_by_day) | set(traces_by_day))
    days = [d for d in days if d and d != "unknown"]
    for day in days:
        write_day_partition(
            staging,
            day,
            test_by_day.get(day, []),
            roll_by_day.get(day, []),
            traces_by_day.get(day, []),
        )
    _log(
        f"wrote {len(test_rows)} test row(s), {len(rollups)} rollup(s) "
        f"across {len(days)} day partition(s)"
    )

    # Data card + manifest reflect the whole bucket, not just this window.
    (staging / "README.md").write_text(
        render_data_card(bucket_uri), encoding="utf-8"
    )
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
