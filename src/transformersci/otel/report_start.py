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
"""Emit a synthetic "run start" span so a CI run/job is visible to the
trace-derived dashboards the moment the job begins — before any test has
finished and produced its first span.

The exporter discovers a run/job from the resource attributes (run id, pr,
repo, job) carried on *any* span, not only test spans; it only *counts* a span
as a test when it is tagged ``pytest.span_type="test"`` with a matching nodeid
(see ``extract_trace_rows``). A bare ``instrument.run`` root span therefore
makes the run and its job discoverable — so the live spinner can light up and
the PR page's ``$latest_run_id`` resolves to the new run — while contributing
zero tests to any rollup: the rollup skips traces that produced no test rows.

Without this, nothing is emitted until the first test finishes, so a just-started
run keeps showing the *previous* run on the PR page until then.

Invoke it THROUGH ``configure-ci-otel`` so the OTEL resource attributes match
the pytest run that is about to start, as the first step before pytest:

    configure-ci-otel --suite "$JOB_NAME" --service-name pytest-observability \\
      --protocol http --otlp-endpoint "$OTEL_EXPORTER_OTLP_ENDPOINT" \\
      --token "$OTEL_TOKEN" --staging-endpoint 10.90.52.50:5317 \\
      --staging-protocol grpc \\
      -- report-ci-start

It is a no-op (exit 0, a stderr note) when no OTLP endpoint is configured, so it
is always safe to call.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from . import instrument
from .report_failure import resolve_job


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Emit a synthetic run-start span so a CI run/job shows up on the live "
            "dashboards before its first test span. Run it through "
            "configure-ci-otel so the OTEL resource attributes are set."
        )
    )
    parser.add_argument(
        "--job",
        "--suite",
        dest="job",
        help="Job name (defaults to TRANSFORMERS_TEST_OTEL_JOB / _SUITE).",
    )
    args = parser.parse_args(argv)

    env = os.environ
    job = resolve_job(args.job, env)

    if not instrument.is_configured(env):
        print(
            "report-ci-start: OTEL not configured (no OTLP endpoint); nothing "
            f"emitted. Would have marked the start of job {job!r}.",
            file=sys.stderr,
            flush=True,
        )
        return 0

    # The root span alone carries the run's resource attributes (run id, pr,
    # repo, job) but no ``pytest.span_type="test"`` tag, so the exporter
    # discovers the run/job without counting it as a test.
    with instrument.run(job):
        pass
    print(f"report-ci-start: emitted run-start span for job {job!r}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
