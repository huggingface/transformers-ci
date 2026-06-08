"""Build ``current_view.json`` — the dataset's machine-readable manifest.

Lets a consumer discover what's in the bucket without listing it: the schema
version, when it was last refreshed, the available day partitions with their
row counts, and a few headline stats. Rebuilt from the staging tree each cycle
so it always reflects what was actually synced.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .tables import SCHEMA_VERSION


def _count_parquet_rows(path: Path) -> int:
    try:
        import pyarrow.parquet as pq

        return pq.ParquetFile(str(path)).metadata.num_rows
    except Exception:
        return 0


def build_manifest(staging_dir, *, now: float | None = None) -> dict:
    """Scan ``<staging>/daily/*`` and return the manifest dict."""
    staging = Path(staging_dir)
    daily = staging / "daily"
    updated = datetime.fromtimestamp(
        now if now is not None else datetime.now(tz=timezone.utc).timestamp(),
        tz=timezone.utc,
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    partitions: list[dict] = []
    total_tests = 0
    total_runs = 0
    if daily.is_dir():
        for day_dir in sorted(p for p in daily.iterdir() if p.is_dir()):
            test_rows = _count_parquet_rows(day_dir / "test_rows.parquet")
            run_rows = _count_parquet_rows(day_dir / "run_rollups.parquet")
            trace_dir = day_dir / "traces"
            trace_count = (
                sum(1 for _ in trace_dir.glob("*.json")) if trace_dir.is_dir() else 0
            )
            partitions.append(
                {
                    "date": day_dir.name,
                    "test_rows": test_rows,
                    "run_rollups": run_rows,
                    "traces": trace_count,
                }
            )
            total_tests += test_rows
            total_runs += run_rows

    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": updated,
        "partition_count": len(partitions),
        "total_test_rows": total_tests,
        "total_run_rollups": total_runs,
        "partitions": partitions,
    }


def write_manifest(staging_dir, *, now: float | None = None) -> dict:
    """Build the manifest and write it to ``<staging>/current_view.json``."""
    manifest = build_manifest(staging_dir, now=now)
    path = Path(staging_dir) / "current_view.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
