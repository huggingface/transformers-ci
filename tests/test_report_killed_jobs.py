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

from transformersci.otel import report_killed_jobs as rkj


# The tail of job 110493403532's log (run 36884255812): the pod is OOM-killed
# while exaone4's export test runs, and every later step fails to exec.
KILLED_JOB_LOG = """\
2026-10-01T17:28:39.8988650Z tests/models/exaone4/test_modeling_exaone4.py::Exaone4ModelTest::test_training_overfit PASSED [ 97%]
2026-10-01T17:28:39.9057442Z tests/models/exaone4/test_modeling_exaone4.py::Exaone4ModelTest::test_vision_axial_rope SKIPPED [ 98%]
2026-10-01T17:33:47.8333912Z tests/models/exaone4/test_modeling_exaone4.py::Exaone4IntegrationTest::test_export_static_cache ::debug::{"message":"command terminated"}
2026-10-01T17:33:47.8372343Z ##[error]Error: failed to run script step: command terminated with non-zero exit code: error executing command [sh -e /__w/_temp/x.sh], exit code 137
2026-10-01T17:33:47.8446995Z ##[error]Process completed with exit code 1.
2026-10-01T17:33:48.2946141Z ##[error]Error: failed to run script step: Internal error occurred: error executing command in container: failed to exec in container
"""

FAILED_JOB_LOG = """\
2026-10-01T17:47:35.9651166Z ===== 2 failed, 178 passed, 137 skipped, 18 warnings in 1358.62s (0:22:38) =====
2026-10-01T17:47:52.7835659Z ##[error]Error: failed to run script step: command terminated with non-zero exit code: error executing command [sh -e /__w/_temp/y.sh], exit code 1
"""


def _job(job_id, name, conclusion="failure"):
    return {"id": job_id, "name": name, "conclusion": conclusion}


def test_parse_model_job():
    job = rkj.parse_model_job(
        _job(
            110493403532,
            "Model CI / run_models_gpu (aws-g5-4xlarge-cache, 0) / run_models_gpu (models/exaone4)",
        )
    )
    assert job == rkj.ModelJob(
        job_id=110493403532,
        suite="run_models_gpu",
        folder="models/exaone4",
        hardware="single-gpu",
    )


def test_parse_model_job_trainer_family_and_multi_gpu():
    job = rkj.parse_model_job(
        _job(
            7,
            "Trainer CI / run_trainer_and_fsdp_gpu (aws-g5-12xlarge-cache, 2) / run_trainer_and_fsdp_gpu (fsdp)",
        )
    )
    assert job is not None
    assert (job.suite, job.folder, job.hardware) == (
        "run_trainer_and_fsdp_gpu",
        "fsdp",
        "multi-gpu",
    )


@pytest.mark.parametrize(
    "name",
    [
        "Setup / setup (aws-g5-4xlarge-cache)",
        "Model CI / Collated Reports / Collated reports",
        "Pipelines CI / run_pipelines_torch_gpu (aws-g5-4xlarge-cache)",
    ],
)
def test_parse_model_job_ignores_other_jobs(name):
    assert rkj.parse_model_job(_job(1, name)) is None


def test_pytest_output_from_killed_job_log():
    output = rkj.pytest_output_from_job_log(KILLED_JOB_LOG)
    assert output is not None
    assert "##[error]" not in output
    assert "::debug::" not in output
    assert output.splitlines()[0].startswith("tests/models/exaone4/")
    assert (
        rkj.parse_running_nodeid(output)
        == "tests/models/exaone4/test_modeling_exaone4.py::Exaone4IntegrationTest::test_export_static_cache"
    )


def test_pytest_output_from_job_log_not_killed():
    assert rkj.pytest_output_from_job_log(FAILED_JOB_LOG) is None


@pytest.mark.parametrize(
    ("ci_event", "slug"),
    [
        ("Daily CI", "daily"),
        ("Nightly CI", "nightly"),
        ("Past CI - pytorch-1.13", "past-ci-pytorch-1.13"),
        ("", "daily"),
    ],
)
def test_ci_event_slug_matches_the_model_job(ci_event, slug):
    assert rkj.ci_event_slug(ci_event) == slug


def test_report_command_and_env():
    job = rkj.ModelJob(
        job_id=42,
        suite="run_models_gpu",
        folder="models/exaone4",
        hardware="single-gpu",
    )
    wrapper = ["configure-ci-otel", "--service-name", "pytest-observability"]
    command = rkj.report_command(wrapper, job, "/tmp/log.txt")
    assert command[: len(wrapper)] == wrapper
    assert command[len(wrapper) : len(wrapper) + 4] == [
        "--suite",
        "run_models_gpu",
        "--",
        "report-ci-failure",
    ]
    assert command[command.index("--kind") + 1] == "oom_killed"
    assert command[command.index("--crash-log") + 1] == "/tmp/log.txt"

    env = rkj.report_env(
        {"OTEL_RESOURCE_ATTRIBUTES": "stale=1", "KEEP": "x"}, job, "Daily CI"
    )
    assert env["KEEP"] == "x"
    assert env["OTEL_RESOURCE_ATTRIBUTES"] == (
        "transformers.test.ci_event=daily,"
        "transformers.test.hardware=single-gpu,"
        "cicd.pipeline.task.run.id=42"
    )


def test_main_reports_only_killed_model_jobs(monkeypatch):
    jobs = [
        _job(
            1,
            "Model CI / run_models_gpu (aws-g5-4xlarge-cache, 0) / run_models_gpu (models/exaone4)",
        ),
        _job(
            2,
            "Model CI / run_models_gpu (aws-g5-12xlarge-cache, 0) / run_models_gpu (models/exaone4)",
        ),
        _job(3, "Setup / setup (aws-g5-4xlarge-cache)"),
    ]
    logs = {1: KILLED_JOB_LOG, 2: FAILED_JOB_LOG}
    calls = []

    class Result:
        returncode = 0

    def fake_run(command, env, check):
        crash_log = command[command.index("--crash-log") + 1]
        with open(crash_log, encoding="utf-8") as handle:
            calls.append((command, env["OTEL_RESOURCE_ATTRIBUTES"], handle.read()))
        return Result()

    monkeypatch.setenv("GITHUB_TOKEN", "t")
    monkeypatch.setattr(
        rkj, "list_failed_jobs", lambda repo, run_id, attempt, token: jobs
    )
    monkeypatch.setattr(rkj, "fetch_job_log", lambda repo, job_id, token: logs[job_id])
    monkeypatch.setattr(rkj.subprocess, "run", fake_run)

    rc = rkj.main(
        ["--repo", "o/r", "--run-id", "9", "--attempt", "2", "--", "configure-ci-otel"]
    )
    assert rc == 0
    assert len(calls) == 1
    command, attributes, crash_log = calls[0]
    assert command[:4] == ["configure-ci-otel", "--suite", "run_models_gpu", "--"]
    assert "cicd.pipeline.task.run.id=1" in attributes
    assert "transformers.test.hardware=single-gpu" in attributes
    assert "test_export_static_cache" in crash_log


def test_main_requires_wrapper(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    with pytest.raises(SystemExit):
        rkj.main(["--repo", "o/r", "--run-id", "9"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
