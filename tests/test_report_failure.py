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
from __future__ import annotations

import pytest

from transformersci.otel import report_failure


CRASH_LOG = """\
tests/models/foo/test_bar.py::TestBar::test_a PASSED
[gw3] worker 'gw3' crashed while running 'tests/models/foo/test_bar.py::TestBar::test_oom'
replacing crashed worker gw3
INTERNALERROR> worker gw3 crashed and worker restarting disabled
"""


def test_parse_crashed_nodeids_quoted():
    assert report_failure.parse_crashed_nodeids(CRASH_LOG) == [
        "tests/models/foo/test_bar.py::TestBar::test_oom"
    ]


def test_parse_crashed_nodeids_unquoted_and_dedup():
    text = (
        "worker gw1 crashed while running tests/x.py::test_one\n"
        "worker gw1 crashed while running tests/x.py::test_one\n"
        "worker gw2 crashed while running tests/y.py::test_two\n"
    )
    assert report_failure.parse_crashed_nodeids(text) == [
        "tests/x.py::test_one",
        "tests/y.py::test_two",
    ]


def test_parse_crashed_nodeids_none():
    assert report_failure.parse_crashed_nodeids("all good, nothing crashed") == []


# Serial `pytest -v` output as `script` captures it (CRLF), cut off by a SIGKILL
# while the last test runs. Captured log lines may sit between a nodeid and its
# outcome.
KILLED_LOG = (
    "tests/models/foo/test_bar.py::FooTest::test_a PASSED"
    "                                [ 97%]\r\n"
    "tests/models/foo/test_bar.py::FooTest::test_b SKIPPED (not on GPU)"
    "                  [ 98%]\r\n"
    "tests/models/foo/test_bar.py::FooIntegrationTest::test_export \r\n"
    "WARNING  transformers.foo:logging.py:340 ERROR-ish message\r\n"
)


def test_parse_running_nodeid_killed_mid_test():
    assert (
        report_failure.parse_running_nodeid(KILLED_LOG)
        == "tests/models/foo/test_bar.py::FooIntegrationTest::test_export"
    )


def test_parse_running_nodeid_killed_between_tests():
    text = KILLED_LOG.replace(
        "test_export \r\n", "test_export FAILED                         [ 99%]\r\n"
    )
    assert report_failure.parse_running_nodeid(text) is None


def test_parse_running_nodeid_session_finished():
    # A finished session lists nodeids in its summaries; none of them is running.
    text = (
        "tests/x.py::T::test_one PASSED                                [100%]\n"
        "=============================== warnings summary ==============================\n"
        "tests/x.py::T::test_one\n"
        "  some warning\n"
        "===================== 1 passed, 1 warning in 3.21s =====================\n"
    )
    assert report_failure.parse_running_nodeid(text) is None


def test_parse_running_nodeid_no_test_started():
    assert (
        report_failure.parse_running_nodeid("collecting ... collected 3 items\n")
        is None
    )


def test_resolve_job_precedence():
    assert (
        report_failure.resolve_job("explicit", {"TRANSFORMERS_TEST_OTEL_JOB": "x"})
        == "explicit"
    )
    assert report_failure.resolve_job(None, {"TRANSFORMERS_TEST_OTEL_JOB": "x"}) == "x"
    assert (
        report_failure.resolve_job(None, {"TRANSFORMERS_TEST_OTEL_SUITE": "y"}) == "y"
    )
    assert report_failure.resolve_job(None, {}) == "unknown"


def test_main_noop_without_otel(monkeypatch, capsys):
    # No OTLP endpoint configured -> no-op, exit 0, and it reports what it would
    # have emitted (job-level fallback when no nodeid is found).
    for key in ("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TRANSFORMERS_TEST_OTEL_JOB", "tests_processors")
    rc = report_failure.main(["--message", "boom"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "not configured" in err
    assert "tests_processors::worker_crash" in err


def test_main_emits_parsed_nodeids(monkeypatch, tmp_path, capsys):
    # With OTEL "configured", instrument.run is exercised; stub it to capture
    # which nodeids and exit code/exception type get emitted.
    log = tmp_path / "tests_output.txt"
    log.write_text(CRASH_LOG, encoding="utf-8")

    recorded: list[tuple[str, int, str]] = []

    class FakeStep:
        def __init__(self, nodeid):
            self.nodeid = nodeid

        def set_exit_code(
            self, rc, *, command=None, output=None, exception_type="CheckFailed"
        ):
            recorded.append((self.nodeid, rc, exception_type))

    class FakeRun:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def step(self, nodeid, attributes=None):
            from contextlib import contextmanager

            @contextmanager
            def cm():
                yield FakeStep(nodeid)

            return cm()

    monkeypatch.setattr(report_failure.instrument, "is_configured", lambda env: True)
    monkeypatch.setattr(report_failure.instrument, "run", lambda job: FakeRun())
    monkeypatch.setenv("TRANSFORMERS_TEST_OTEL_JOB", "tests_processors")

    rc = report_failure.main(["--crash-log", str(log)])
    assert rc == 0
    assert recorded == [
        ("tests/models/foo/test_bar.py::TestBar::test_oom", 1, "WorkerCrash")
    ]


def test_main_job_level_fallback(monkeypatch, tmp_path):
    recorded = []

    class FakeStep:
        def __init__(self, nodeid):
            self.nodeid = nodeid

        def set_exit_code(
            self, rc, *, command=None, output=None, exception_type="CheckFailed"
        ):
            recorded.append(self.nodeid)

    class FakeRun:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def step(self, nodeid, attributes=None):
            from contextlib import contextmanager

            @contextmanager
            def cm():
                yield FakeStep(nodeid)

            return cm()

    monkeypatch.setattr(report_failure.instrument, "is_configured", lambda env: True)
    monkeypatch.setattr(report_failure.instrument, "run", lambda job: FakeRun())

    rc = report_failure.main(["--job", "tests_torch"])
    assert rc == 0
    assert recorded == ["tests_torch::worker_crash"]


def test_main_oom_killed_kind(monkeypatch):
    # No crash log / no nodeid + --kind oom_killed -> a job-level OOMKilled span.
    recorded: list[tuple[str, str]] = []

    class FakeStep:
        def __init__(self, nodeid):
            self.nodeid = nodeid

        def set_exit_code(
            self, rc, *, command=None, output=None, exception_type="CheckFailed"
        ):
            recorded.append((self.nodeid, exception_type))

    class FakeRun:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def step(self, nodeid, attributes=None):
            from contextlib import contextmanager

            @contextmanager
            def cm():
                yield FakeStep(nodeid)

            return cm()

    monkeypatch.setattr(report_failure.instrument, "is_configured", lambda env: True)
    monkeypatch.setattr(report_failure.instrument, "run", lambda job: FakeRun())

    rc = report_failure.main(["--job", "tests_processors", "--kind", "oom_killed"])
    assert rc == 0
    assert recorded == [("tests_processors::oom_killed", "OOMKilled")]


def test_main_oom_killed_names_running_test(monkeypatch, tmp_path):
    # --kind oom_killed + a `pytest -v` log -> the span names the killed test.
    log = tmp_path / "test_outputs.txt"
    log.write_text(KILLED_LOG, encoding="utf-8")
    recorded: list[tuple[str, str]] = []

    class FakeStep:
        def __init__(self, nodeid):
            self.nodeid = nodeid

        def set_exit_code(
            self, rc, *, command=None, output=None, exception_type="CheckFailed"
        ):
            recorded.append((self.nodeid, exception_type))

    class FakeRun:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def step(self, nodeid, attributes=None):
            from contextlib import contextmanager

            @contextmanager
            def cm():
                yield FakeStep(nodeid)

            return cm()

    monkeypatch.setattr(report_failure.instrument, "is_configured", lambda env: True)
    monkeypatch.setattr(report_failure.instrument, "run", lambda job: FakeRun())

    rc = report_failure.main(
        ["--job", "run_models_gpu", "--kind", "oom_killed", "--crash-log", str(log)]
    )
    assert rc == 0
    assert recorded == [
        ("tests/models/foo/test_bar.py::FooIntegrationTest::test_export", "OOMKilled")
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
