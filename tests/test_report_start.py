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
"""Tests for the ``report-ci-start`` CLI and the exporter contract it relies on."""

from __future__ import annotations

from transformersci.otel import report_start, trace_exporter


def make_tag(key: str, value: str) -> dict:
    return {"key": key, "value": value}


def test_main_noop_without_otel(monkeypatch, capsys):
    # No OTLP endpoint configured -> no-op, exit 0, and instrument.run untouched.
    for key in ("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TRANSFORMERS_TEST_OTEL_JOB", "tests_processors")

    def _boom(_job):  # instrument.run must not be called when unconfigured
        raise AssertionError("instrument.run should not run when OTEL is off")

    monkeypatch.setattr(report_start.instrument, "run", _boom)

    rc = report_start.main([])
    assert rc == 0
    err = capsys.readouterr().err
    assert "not configured" in err
    assert "tests_processors" in err


def test_main_emits_start_span(monkeypatch, capsys):
    # With OTEL "configured", a single run span is opened (and closed) for the
    # resolved job, with no test steps.
    opened: list[str] = []
    steps: list[str] = []

    class FakeRun:
        def __init__(self, job):
            self.job = job

        def __enter__(self):
            opened.append(self.job)
            return self

        def __exit__(self, *a):
            return False

        def step(self, nodeid, attributes=None):  # pragma: no cover - must not run
            steps.append(nodeid)

    monkeypatch.setattr(report_start.instrument, "is_configured", lambda env: True)
    monkeypatch.setattr(report_start.instrument, "run", lambda job: FakeRun(job))

    rc = report_start.main(["--job", "tests_torch"])
    assert rc == 0
    assert opened == ["tests_torch"]
    assert steps == []  # a start span opens no test steps
    assert "tests_torch" in capsys.readouterr().out


def test_job_flag_falls_back_to_env(monkeypatch):
    opened: list[str] = []

    class FakeRun:
        def __init__(self, job):
            self.job = job

        def __enter__(self):
            opened.append(self.job)
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(report_start.instrument, "is_configured", lambda env: True)
    monkeypatch.setattr(report_start.instrument, "run", lambda job: FakeRun(job))
    monkeypatch.setenv("TRANSFORMERS_TEST_OTEL_SUITE", "tests_generate")
    monkeypatch.delenv("TRANSFORMERS_TEST_OTEL_JOB", raising=False)

    assert report_start.main([]) == 0
    assert opened == ["tests_generate"]


def test_start_span_is_discoverable_but_counts_zero_tests():
    # The contract that makes report-ci-start safe: the exporter reads the run
    # identity (run id, pr, job) from the resource tags of *any* span, but only
    # counts a span as a test when it carries pytest.span_type="test". A bare
    # root span (what instrument.run emits) therefore yields a populated
    # trace_info with NO test rows — so the run is discoverable yet adds zero
    # tests to the rollup (which skips zero-row traces entirely).
    root_span = {
        "operationName": "tests_processors",  # the job name, NOT a pytest nodeid
        "processID": "pytest-process",
        "startTime": 5_000_000,
        "duration": 0,
        "tags": [make_tag("otel.status_code", "UNSET")],
    }
    trace = {
        "traceID": "trace-start",
        "spans": [root_span],
        "processes": {
            "pytest-process": {
                "serviceName": "pytest-observability",
                "tags": [
                    make_tag("transformers.test.provider", "github_actions"),
                    make_tag("transformers.test.run.id", "99999:1"),
                    make_tag("transformers.test.job", "tests_processors"),
                    make_tag("vcs.change.id", "46774"),
                    make_tag("vcs.repository.name", "huggingface/transformers"),
                ],
            }
        },
    }

    trace_info, rows = trace_exporter.extract_trace_rows(trace)

    assert rows == []  # not counted as a test
    assert trace_info["run_id"] == "99999:1"
    assert trace_info["pr"] == "46774"
    assert trace_info["test_job"] == "tests_processors"
    assert trace_info["repository"] == "huggingface/transformers"

    # And the rollup adds nothing for a zero-row trace: no run/job metrics.
    rollup = trace_exporter.extract_run_rollup_metrics([trace])
    assert not [ln for ln in rollup if ln.startswith("pytest_run_total_tests")]
    assert not [ln for ln in rollup if ln.startswith("pytest_run_job_total_tests")]
