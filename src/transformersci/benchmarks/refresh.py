"""Refresh the benchmark snapshot outside the stdlib-only HTTP server.

Run with the ``benchmarks`` extra installed. A failed refresh leaves the last
complete snapshot intact. The request path never downloads or parses Parquet.

Daily memory reports (``memory/<month>/<group>/`` in the dataset, one per suite
and day) are copied next to the snapshot, in ``memory/<group>/``, for the days
the pinned analysis links (its MEMORY_DAYS); older ones are deleted, so the
directory stays bounded.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import time
from pathlib import Path

LOGGER = logging.getLogger(__name__)

DATASET = "hf-internal-testing/transformers-perf"
# Copied as published; report.json is rendered here from the summaries (the site's page around it)
MEMORY_FILES = (
    "report.html",
    "host-flamegraph.html",
    "profile.json",
    "gpu_summary.json",
    "host_summary.json",
    "gpu_snapshot.pickle.gz",
    "python_summary.json",
    "python.folded",
)
MEMORY_REPORT = re.compile(
    r"(memory/[0-9]{4}-[0-9]{2}/([A-Za-z0-9][A-Za-z0-9._-]*))/report\.html"
)


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

    files = HfApi().list_repo_files(dataset, repo_type="dataset", revision=revision)
    reports = {m.group(2): m.group(1) for m in map(MEMORY_REPORT.fullmatch, files) if m}
    pages = build(rows("data"), rows("processors"), reports)
    memory = pages.pop("_memory", {})
    payload = finite(
        {
            "dataset": dataset,
            "revision": revision,
            "updated_at": time.time(),
            "pages": pages,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    copy_memory_reports(dataset, revision, memory, path.parent / "memory")
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, allow_nan=False, separators=(",", ":")))
    temp.replace(path)
    return True


def copy_memory_reports(dataset, revision, memory, dest):
    """The linked reports into dest/<group>/ (before the snapshot that links them
    is written), then every other group out."""
    from huggingface_hub import snapshot_download

    if memory:
        wanted = [f"{d}/{name}" for d in memory.values() for name in MEMORY_FILES]
        root = Path(
            snapshot_download(
                dataset,
                repo_type="dataset",
                revision=revision,
                allow_patterns=wanted,
            )
        )
        for group, directory in memory.items():
            target = dest / group
            target.mkdir(parents=True, exist_ok=True)
            for name in MEMORY_FILES:
                source = root / directory / name
                if source.is_file():
                    temp = target / (name + ".tmp")
                    shutil.copyfile(source, temp)
                    temp.replace(target / name)
            render_memory_report(target)
    if dest.is_dir():
        for old in dest.iterdir():
            if old.name not in memory:
                shutil.rmtree(old, ignore_errors=True)


def render_memory_report(target):
    """report.json ({title, css, body}) for the exporter to put inside the site's
    page, rendered by the pinned transformers-perf from the published summaries:
    no script, every value escaped. Reports published before the summaries were
    keep their standalone report.html only."""
    try:
        from tperf.memory.report import load_summaries, render_fragment
    except ImportError:
        return
    if not (target / "profile.json").is_file():
        return
    profile, gpu, host = load_summaries(target)
    try:  # the python pass (py-spy): transformers-perf after 1190a10 (pinned at acef2ad)
        from tperf.memory.report import load_python_summary
    except ImportError:
        load_python_summary = None
    python = load_python_summary(target) if load_python_summary else None
    links = {}
    if (target / "host-flamegraph.html").is_file():
        links["flamegraph"] = "host-flamegraph.html"
    if (target / "gpu_snapshot.pickle.gz").is_file():
        links["memory_viz"] = "memory-viz.html"
    if (target / "python.folded").is_file():
        links["python_stacks"] = "python.folded"
    fragment = (
        render_fragment(profile, gpu, host, links, python)
        if python is not None
        else render_fragment(profile, gpu, host, links)
    )
    temp = target / "report.json.tmp"
    temp.write_text(json.dumps(fragment))
    temp.replace(target / "report.json")


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
