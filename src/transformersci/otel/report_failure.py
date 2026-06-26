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
"""Emit a synthetic failing "test" span for a CI failure that produced no span.

A pytest-xdist worker that OOMs or segfaults dies mid-test, so the test it was
running emits no span at all. The failure is then invisible to the trace-derived
dashboards: the job shows ``failed=0`` (green) even though GitHub marks it
``failure``. The reusable test workflow already detects this in its
``Check for test worker crashes`` step; this CLI lets that step record the
failure into the *same* trace pipeline, so Tempo — and every downstream consumer
(the dashboard Fail column, the ``/run`` and ``/failure`` pages, the telemetry
publisher) — shows it like any other failing test, with no exporter-side change.

Invoke it THROUGH ``configure-ci-otel`` so the OTEL env (service.name, run id,
job, pr, repo) is set up identically to the pytest run that just crashed:

    configure-ci-otel --suite "$JOB_NAME" --service-name pytest-observability \\
      --protocol http --otlp-endpoint "$OTEL_EXPORTER_OTLP_ENDPOINT" \\
      --token "$OTEL_TOKEN" --staging-endpoint 10.90.52.50:5317 \\
      --staging-protocol grpc \\
      -- report-ci-failure --crash-log tests_output.txt \\
         --message "pytest-xdist worker crashed (likely OOM)"

It is a no-op (exit 0, a stderr note) when no OTLP endpoint is configured, so it
is always safe to call.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Mapping, Sequence

from . import instrument

# When a worker dies with --max-worker-restart=0, pytest-xdist names the test it
# was running. Quoting varies across xdist versions / the ydshieh fork, so accept
# an optional surrounding quote and capture up to the next quote/whitespace.
# Examples:
#   [gw3] worker 'gw3' crashed while running 'tests/models/x/test_y.py::T::test_z'
#   worker gw3 crashed while running tests/models/x/test_y.py::T::test_z
_CRASH_NODEID_PATTERN = re.compile(r"crashed while running ['\"]?([^'\"\s]+)")

# Bound the excerpt recorded on the span so a noisy log can't bloat it; the tail
# holds the crash itself.
_MAX_OUTPUT_CHARS = 8000


def parse_crashed_nodeids(text: str) -> list[str]:
    """Return the distinct test nodeids xdist reported as crashed, in order."""
    found: list[str] = []
    seen: set[str] = set()
    for match in _CRASH_NODEID_PATTERN.finditer(text):
        nodeid = match.group(1).strip()
        if nodeid and nodeid not in seen:
            seen.add(nodeid)
            found.append(nodeid)
    return found


def resolve_job(explicit: str | None, env: Mapping[str, str]) -> str:
    """Job name for the synthetic trace; the resource attrs carry the real one."""
    return (
        explicit
        or env.get("TRANSFORMERS_TEST_OTEL_JOB")
        or env.get("TRANSFORMERS_TEST_OTEL_SUITE")
        or "unknown"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Emit a synthetic failing test span for a CI job failure (e.g. a "
            "pytest-xdist worker OOM) that produced no test span. Run it through "
            "configure-ci-otel so the OTEL resource attributes are set."
        )
    )
    parser.add_argument(
        "--job",
        "--suite",
        dest="job",
        help="Job name (defaults to TRANSFORMERS_TEST_OTEL_JOB / _SUITE).",
    )
    parser.add_argument(
        "--nodeid",
        action="append",
        default=[],
        help="Failing test nodeid; repeatable. Overrides --crash-log parsing.",
    )
    parser.add_argument(
        "--crash-log",
        help="Path to captured pytest output to scan for crashed test nodeids.",
    )
    parser.add_argument(
        "--message",
        default=(
            "CI job failed with no test failure span — likely a pytest-xdist "
            "worker crash (OOM/segfault)."
        ),
        help="Exception message recorded on the span (shown on /failure).",
    )
    parser.add_argument(
        "--exception-type",
        default="WorkerCrash",
        help="exception.type recorded on the span (the /failure heading).",
    )
    args = parser.parse_args(argv)

    env = os.environ
    job = resolve_job(args.job, env)

    output = ""
    nodeids = list(args.nodeid)
    if not nodeids and args.crash_log:
        try:
            with open(args.crash_log, encoding="utf-8", errors="replace") as handle:
                output = handle.read()
        except OSError as error:
            print(
                f"report-ci-failure: cannot read {args.crash_log!r}: {error}",
                file=sys.stderr,
                flush=True,
            )
        else:
            nodeids = parse_crashed_nodeids(output)

    if not nodeids:
        # The caller only runs this once a crash is already detected, but the
        # exact test couldn't be pinned — record a job-level failure so it is
        # still visible and counted rather than silently green.
        nodeids = [f"{job}::worker_crash"]

    if not instrument.is_configured(env):
        print(
            "report-ci-failure: OTEL not configured (no OTLP endpoint); nothing "
            f"emitted. Would have reported {len(nodeids)} failure(s) for job "
            f"{job!r}: {', '.join(nodeids)}",
            file=sys.stderr,
            flush=True,
        )
        return 0

    excerpt = output[-_MAX_OUTPUT_CHARS:] if output else None
    with instrument.run(job) as run:
        for nodeid in nodeids:
            with run.step(
                nodeid, attributes={"transformers.failure.source": "ci_worker_crash"}
            ) as step:
                step.set_exit_code(
                    1,
                    command=args.message,
                    output=excerpt,
                    exception_type=args.exception_type,
                )
            print(
                f"report-ci-failure: recorded {args.exception_type} span for "
                f"{nodeid!r} (job {job!r}).",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
