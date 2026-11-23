"""Refresh the benchmark snapshot outside the stdlib-only HTTP server.

Run with the ``benchmarks`` extra installed. A failed refresh leaves the last
complete snapshot intact. The request path never downloads or parses Parquet.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

LOGGER = logging.getLogger(__name__)

DATASET = "hf-internal-testing/transformers-perf"


def finite(value):
    """JSON has no infinity; an unbounded confidence interval is null at that end."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    return value


def refresh(path: Path, dataset: str = DATASET) -> bool:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, snapshot_download
    from tperf.dashboard.build import build

    # One immutable revision for both folders, including cache hits on restart.
    revision = HfApi().repo_info(dataset, repo_type="dataset").sha
    try:
        previous = json.loads(path.read_text())
    except (OSError, ValueError):
        previous = {}
    if previous.get("revision") == revision and previous.get("dataset") == dataset:
        return False
    root = Path(
        snapshot_download(
            dataset,
            repo_type="dataset",
            revision=revision,
            allow_patterns=["data/*/*.parquet", "processors/*/*.parquet"],
        )
    )

    def rows(folder):
        return [
            row
            for file in sorted((root / folder).glob("*/*.parquet"))
            for row in pq.read_table(file).to_pylist()
        ]

    payload = finite(
        {
            "dataset": dataset,
            "revision": revision,
            "updated_at": time.time(),
            "pages": build(rows("data"), rows("processors")),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, allow_nan=False, separators=(",", ":")))
    temp.replace(path)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--path",
        type=Path,
        default=os.environ.get("TCI_BENCHMARK_SNAPSHOT", "/benchmarks/snapshot.json"),
    )
    parser.add_argument(
        "--dataset", default=os.environ.get("TCI_BENCHMARK_DATASET", DATASET)
    )
    parser.add_argument("--interval", type=int, default=900)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    while True:
        try:
            if refresh(args.path, args.dataset):
                LOGGER.info("Refreshed benchmarks from %s", args.dataset)
        except Exception:
            LOGGER.exception("Benchmark refresh failed; retaining last snapshot")
            if args.once:
                raise
        if args.once:
            return
        time.sleep(max(60, args.interval))


if __name__ == "__main__":
    main()
