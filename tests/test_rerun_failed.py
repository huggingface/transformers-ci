from __future__ import annotations

import io
import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from transformersci.otel import rerun_actions, rerun_failed, rerun_revision
from transformersci.otel import trace_exporter
from transformersci.status import metrics as status_metrics

ROOT = Path(__file__).resolve().parents[1]
HEAD = "c" * 40
NODE = "tests/models/bert/test_modeling_bert.py::TestBert::test_x[fp16]"


def _series(run_id: str, started: int, event: str) -> dict:
    return {
        "metric": {"run_id": run_id, "ci_event": event},
        "value": [0, str(started)],
    }


class FakeRepo:
    """GitHub REST paths under the repository root, as ``api`` sees them."""

    def __init__(self) -> None:
        self.pull = {"state": "open", "merged_at": None, "head": {"sha": HEAD}}
        self.runs: dict[str, dict] = {}
        self.listed: dict[str, list[dict]] = {}
        self.comments: list[dict] = []

    def __call__(self, path: str) -> object:
        if path.startswith("pulls/"):
            return self.pull
        if path.startswith("issues/"):
            return self.comments
        if path.startswith("actions/workflows/"):
            workflow = path.split("/")[2]
            status = re.search(r"status=(\w+)", path).group(1)
            return {
                "workflow_runs": [
                    r for r in self.listed.get(workflow, []) if r["status"] == status
                ]
            }
        if path.startswith("actions/runs/"):
            return self.runs[path.rsplit("/", 1)[1]]
        raise AssertionError(path)


def _source(run_id: str, gpu: bool, status: str = "completed", **extra) -> dict:
    return {
        "id": int(run_id),
        "path": ".github/workflows/"
        + ("self-comment-ci.yml" if gpu else "pr-ci-caller.yml")
        + "@main",
        "event": "issue_comment" if gpu else "pull_request",
        "status": status,
        "run_attempt": 1,
        "head_sha": "d" * 40 if gpu else HEAD,
        "conclusion": "failure" if status == "completed" else "",
        "html_url": f"https://github.com/huggingface/transformers/actions/runs/{run_id}",
        **extra,
    }


def _rows(run_id: str) -> list[dict]:
    gpu = run_id.startswith("2")
    return [
        {
            "pr": "42",
            "status_code": "ERROR",
            "test_nodeid": NODE,
            "test_job": "run_models_gpu" if gpu else "tests_torch",
            "hardware": "single-gpu" if gpu else "cpu",
        },
        {
            "pr": "42",
            "status_code": "ERROR",
            "test_nodeid": "tests_torch::oom_killed",
            "test_job": "tests_torch",
            "hardware": "cpu",
        },
        {
            "pr": "42",
            "status_code": "ERROR",
            "test_nodeid": "tests/utils/test_x.py::test_y",
            "test_job": "check_code_quality",
            "hardware": "cpu",
        },
    ]


def _query(starts: list[dict]):
    def query(expr: str) -> list[dict]:
        if "pytest_run_info" in expr:
            return [
                {"metric": {"run_id": "110:1", "commit_sha": "a" * 40}},
                {"metric": {"run_id": "210:1", "commit_sha": "b" * 40}},
            ]
        assert 'ci_event!~"rerun-failed-.*"' in expr
        return starts

    return query


@pytest.fixture
def repo() -> FakeRepo:
    repo = FakeRepo()
    repo.runs = {
        "110": _source("110", False),
        "120": _source("120", False, "in_progress"),
        "210": _source("210", True),
        "220": _source("220", True, "in_progress"),
    }
    return repo


def _snapshot(repo: FakeRepo, rows=_rows) -> dict:
    starts = [
        _series("110:1", 110, "none"),
        _series("120:1", 120, "none"),
        _series("210:1", 210, "pr-comment"),
        _series("220:1", 220, "pr-comment"),
    ]
    return rerun_failed.snapshot("42", query=_query(starts), api=repo, get_rows=rows)


def test_snapshot_uses_latest_completed_per_lane_and_keeps_active_runs(repo) -> None:
    result = _snapshot(repo)
    cpu, gpu = result["lanes"]["cpu"], result["lanes"]["gpu"]
    assert result["pr_state"] == "open" and result["head_sha"] == HEAD
    assert cpu["source_run"]["run_id"] == "110:1"
    assert gpu["source_run"]["run_id"] == "210:1"
    assert cpu["source_run"]["commit"] == "a" * 40
    assert gpu["source_run"]["commit"] == "b" * 40
    assert [r["run_id"] for r in cpu["active_runs"]] == ["120:1"]
    assert [r["run_id"] for r in gpu["active_runs"]] == ["220:1"]
    assert cpu["complete"] and gpu["complete"]
    first, synthetic, unsupported = cpu["tests"]
    assert first["model"] == "bert" and first["eligible"] is True
    assert synthetic["eligible"] is False and "job-level" in synthetic["reason"]
    assert unsupported["eligible"] is False
    assert "check_code_quality" in unsupported["reason"]
    # Identical node IDs in distinct lanes/environments are distinct selections.
    assert gpu["tests"][0]["nodeid"] == first["nodeid"]
    assert gpu["tests"][0]["key"] != first["key"]
    assert re.fullmatch(r"[0-9a-f]{16}", result["version"])
    assert _snapshot(repo)["version"] == result["version"]


def test_queued_gpu_run_is_matched_to_its_pr_through_the_comment(repo) -> None:
    created = "2026-10-08T10:00:30Z"
    mine = _source(
        "230",
        True,
        "queued",
        run_attempt=1,
        created_at=created,
        triggering_actor={"login": "maintainer"},
    )
    replied = _source("240", True, "queued", run_attempt=1, created_at=created)
    other = _source(
        "250",
        True,
        "queued",
        run_attempt=1,
        created_at=created,
        triggering_actor={"login": "someone-else"},
    )
    repo.listed["self-comment-ci.yml"] = [mine, replied, other]
    repo.comments = [
        {
            "body": "run-slow: bert",
            "user": {"login": "maintainer"},
            "created_at": "2026-10-08T10:00:10Z",
        },
        {
            "body": "## Nvidia CI\n\n[Workflow Run ⚙️](https://github.com/huggingface/transformers/actions/runs/240)",
            "user": {"login": "github-actions[bot]"},
            "created_at": "2026-10-08T10:01:00Z",
        },
        {
            "body": "run-slow: llama",
            "user": {"login": "maintainer"},
            "created_at": "2026-10-08T09:00:00Z",
        },
    ]
    active = _snapshot(repo)["lanes"]["gpu"]["active_runs"]
    assert [r["run_id"] for r in active] == ["220:1", "230:1", "240:1"]


def test_queued_cpu_run_is_found_by_head_sha_before_telemetry(repo) -> None:
    repo.listed["pr-ci-caller.yml"] = [
        _source("130", False, "queued", head_sha=HEAD, run_attempt=2),
        _source("140", False, "queued", head_sha="d" * 40),
    ]
    active = _snapshot(repo)["lanes"]["cpu"]["active_runs"]
    assert [r["run_id"] for r in active] == ["120:1", "130:2"]


def test_failed_run_without_failing_telemetry_is_incomplete(repo) -> None:
    result = _snapshot(repo, rows=lambda run_id: [])
    cpu = result["lanes"]["cpu"]
    assert cpu["complete"] is False
    assert "telemetry has no failed tests" in cpu["notes"][0]


def test_plan_groups_by_environment_and_refuses_what_it_did_not_list(repo) -> None:
    snap = _snapshot(repo)
    cpu, gpu = snap["lanes"]["cpu"]["tests"], snap["lanes"]["gpu"]["tests"]
    lanes = rerun_failed.plan(snap, [cpu[0]["key"], gpu[0]["key"]])
    assert set(lanes) == {"cpu", "gpu"}
    assert json.loads(lanes["cpu"]["selection"]) == {
        "v": 1,
        "groups": [{"job": "tests_torch", "hardware": "cpu", "tests": [NODE]}],
    }
    assert (
        json.loads(lanes["gpu"]["selection"])["groups"][0]["hardware"] == "single-gpu"
    )
    assert [r["run_id"] for r in lanes["gpu"]["active_runs"]] == ["220:1"]
    assert set(rerun_failed.plan(snap, [cpu[0]["key"]])) == {"cpu"}

    for keys, code in (
        (["0" * 24], "stale_selection"),
        ([cpu[1]["key"]], "unsupported_selection"),
        ([], "invalid_selection"),
        ("abc", "invalid_selection"),
        ([cpu[0]["key"], cpu[0]["key"]], "invalid_selection"),
    ):
        with pytest.raises(rerun_failed.SelectionError) as error:
            rerun_failed.plan(snap, keys)
        assert error.value.code == code

    snap["lanes"]["cpu"]["complete"] = False
    snap["lanes"]["cpu"]["notes"] = ["truncated"]
    with pytest.raises(rerun_failed.SelectionError) as error:
        rerun_failed.plan(snap, [cpu[0]["key"]])
    assert error.value.code == "incomplete_source"


def _python_steps(workflow: str) -> list[str]:
    data = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())
    return [
        step["run"]
        for job in data["jobs"].values()
        for step in job["steps"]
        if step.get("shell") == "python3 {0}"
    ]


@pytest.mark.parametrize("workflow", ["rerun-failed-cpu.yml", "rerun-failed-gpu.yml"])
def test_workflow_python_steps_compile(workflow: str) -> None:
    steps = _python_steps(workflow)
    assert len(steps) == 2
    for source in steps:
        compile(source, workflow, "exec")


def test_workflow_allowlists_match_the_exporter() -> None:
    cpu_plan = _python_steps("rerun-failed-cpu.yml")[0]
    namespace: dict = {}
    exec(cpu_plan.split("NODEID =", 1)[0], namespace)
    assert set(namespace["ENVIRONMENTS"]) == rerun_failed.CPU_JOBS
    gpu_plan = _python_steps("rerun-failed-gpu.yml")[0]
    namespace = {}
    exec(gpu_plan.split("NODEID =", 1)[0], namespace)
    assert set(namespace["JOBS"]) == rerun_failed.GPU_JOBS
    assert namespace["RUNNERS"] == rerun_failed.GPU_RUNNERS
    caller = (ROOT / ".github/workflows/pr-ci_dynamic_caller_example.yml").read_text()
    assert set(re.findall(r'job_name:\s+"(\w+)"', caller)) == rerun_failed.CPU_JOBS


def test_no_node_id_reaches_a_shell() -> None:
    for workflow in ("rerun-failed-cpu.yml", "rerun-failed-gpu.yml"):
        text = (ROOT / ".github/workflows" / workflow).read_text()
        assert not re.search(r"\beval\b", text)
        assert "inputs.selection" in text
        # Only the planning step reads the selection, from an env variable.
        assert text.count("${{ inputs.selection }}") == 1


def _run_workflow_plan(workflow: str, selection: dict, tmp_path: Path) -> dict:
    output = tmp_path / "out"
    output.write_text("")
    result = subprocess.run(
        ["python3", "-c", _python_steps(workflow)[0]],
        env={"SELECTION": json.dumps(selection), "GITHUB_OUTPUT": str(output)},
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(output.read_text().split("=", 1)[1])


def test_workflow_plans_accept_the_exporter_selection(repo, tmp_path) -> None:
    snap = _snapshot(repo)
    lanes = rerun_failed.plan(
        snap,
        [
            snap["lanes"]["cpu"]["tests"][0]["key"],
            snap["lanes"]["gpu"]["tests"][0]["key"],
        ],
    )
    cpu = _run_workflow_plan(
        "rerun-failed-cpu.yml", json.loads(lanes["cpu"]["selection"]), tmp_path
    )["include"]
    assert cpu[0]["image"] == "huggingface/transformers-torch-light"
    assert json.loads(cpu[0]["tests"]) == [NODE]
    gpu = _run_workflow_plan(
        "rerun-failed-gpu.yml", json.loads(lanes["gpu"]["selection"]), tmp_path
    )["include"]
    assert gpu[0]["runner"] == "aws-g5-4xlarge-cache"
    with pytest.raises(AssertionError, match="not allowed"):
        _run_workflow_plan(
            "rerun-failed-cpu.yml",
            {
                "v": 1,
                "groups": [{"job": "check_x", "hardware": "cpu", "tests": [NODE]}],
            },
            tmp_path,
        )
    with pytest.raises(AssertionError, match="bad node id"):
        _run_workflow_plan(
            "rerun-failed-cpu.yml",
            {
                "v": 1,
                "groups": [
                    {"job": "tests_torch", "hardware": "cpu", "tests": ["-p evil"]}
                ],
            },
            tmp_path,
        )


@pytest.mark.parametrize("lane", ["cpu", "gpu"])
def test_caller_run_name_files_the_run_under_its_pr(lane: str) -> None:
    caller = yaml.safe_load(
        (ROOT / f"docs/rerun-failed/transformers-rerun-failed-{lane}.yml").read_text()
    )
    assert caller["name"] == f"Rerun failed tests ({lane.upper()})"
    assert caller["jobs"]["rerun"]["uses"].endswith(f"rerun-failed-{lane}.yml@main")
    title = (
        caller["run-name"]
        .replace("${{ inputs.pr_number }}", "42")
        .replace("${{ inputs.correlation_id }}", f"0123456789abcdef-{lane}")
    )
    state = {
        "event": "workflow_dispatch",
        "workflow_path": f".github/workflows/rerun-failed-{lane}.yml",
        "display_title": title,
        "head_branch": "main",
    }
    assert status_metrics.pr_label(state) == "42"
    # Anything else dispatched on main stays a branch run.
    assert (
        status_metrics.pr_label({**state, "workflow_path": ".github/workflows/x.yml"})
        == "main"
    )
    assert rerun_actions.LANE_TITLES[lane] in title
    assert f"0123456789abcdef-{lane}" in title


def test_reruns_never_count_as_a_badge_stream() -> None:
    assert not trace_exporter._badge_event_matches("pr-ci", "rerun-failed-cpu")
    assert not trace_exporter._badge_event_matches("run-slow", "rerun-failed-gpu")
    assert trace_exporter._badge_event_matches("pr-ci", "none")
    assert trace_exporter.checks_out_own_commit("rerun-failed-gpu")
    assert trace_exporter.checks_out_own_commit("pr-comment")
    assert not trace_exporter.checks_out_own_commit("none")


class FakeSerge:
    """serge's /dashboard/* API."""

    def __init__(self, role: str = "write") -> None:
        self.role = role
        self.calls: list[tuple[str, dict]] = []
        self.cancel_reply: tuple[int, dict] = (
            200,
            {"github_status": 202, "run_status": "in_progress"},
        )

    def call(self, operation: str, body: dict) -> tuple[int, dict]:
        self.calls.append((operation, body))
        if operation == "permission":
            return 200, {"can_write": self.role == "write"}
        if operation == "runs/cancel":
            return self.cancel_reply
        if operation == "workflows/dispatch":
            return 200, {"dispatched": True}
        raise AssertionError(operation)


def test_writes_go_through_serge_on_behalf_of_the_actor() -> None:
    serge = FakeSerge()
    reads = []
    github = rerun_actions.GitHub(
        "maintainer", serge, read=lambda path: reads.append(path) or (200, {"ok": 1})
    )
    assert github.get("pulls/42") == {"ok": 1} and reads == ["pulls/42"]
    assert github.request("POST", "actions/runs/120/cancel") == (202, None)
    serge.cancel_reply = (200, {"github_status": 409})
    assert github.request("POST", "actions/runs/120/cancel") == (409, None)
    # Already completed: serge made no call; the worker re-reads, as on a 409.
    serge.cancel_reply = (200, {"github_status": None, "run_status": "completed"})
    assert github.request("POST", "actions/runs/120/cancel") == (409, None)
    serge.cancel_reply = (403, {"detail": "workflow_not_cancellable"})
    assert github.request("POST", "actions/runs/120/cancel")[0] == 403
    status, _ = github.request(
        "POST",
        "actions/workflows/rerun-failed-cpu.yml/dispatches",
        {"ref": "main", "inputs": {"pr_number": "42"}},
    )
    assert status == 204
    assert serge.calls[0] == ("runs/cancel", {"actor": "maintainer", "run_id": "120"})
    assert serge.calls[-1] == (
        "workflows/dispatch",
        {
            "actor": "maintainer",
            "workflow": "rerun-failed-cpu.yml",
            "inputs": {"pr_number": "42"},
        },
    )
    with pytest.raises(ValueError):
        github.request("POST", "pulls/42/merge")
    assert rerun_actions.can_write(serge, "maintainer")
    assert not rerun_actions.can_write(FakeSerge(role="read"), "someone")


def test_serge_client_sends_the_token_and_repository(monkeypatch) -> None:
    sent = {}

    class Response(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class Opener:
        def open(self, request, timeout):
            sent["url"] = request.full_url
            sent["auth"] = request.get_header("Authorization")
            sent["body"] = json.loads(request.data)
            return Response(b'{"can_write": true}')

    monkeypatch.setattr(rerun_actions, "build_opener", lambda *a: Opener())
    status, payload = rerun_actions.Serge("http://serge.local", "s3cret").call(
        "permission", {"actor": "maintainer"}
    )
    assert (status, payload) == (200, {"can_write": True})
    assert sent == {
        "url": "http://serge.local/dashboard/permission",
        "auth": "Bearer s3cret",
        "body": {"repository": "huggingface/transformers", "actor": "maintainer"},
    }


class FakeGitHub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.runs = {"120": {"status": "in_progress"}}
        self.cancel_status = 202
        self.dispatch_status = 204
        self.dispatched: list[dict] = []
        self.find_after_dispatch = True

    def request(self, method: str, path: str, body: dict | None = None):
        self.calls.append((method, path))
        if method == "POST" and path.endswith("/cancel"):
            run = path.split("/")[2]
            if self.cancel_status == 202:
                self.runs[run] = {"status": "completed"}
            return self.cancel_status, None
        if method == "POST" and path.endswith("/dispatches"):
            self.dispatched.append(body)
            return self.dispatch_status, None
        return 200, self.get(path)

    def get(self, path: str) -> object:
        if path == "pulls/42":
            return {"state": "open", "head": {"sha": HEAD}}
        if path.startswith("commits/"):
            return {
                "sha": path.split("/")[1],
                "parents": [{"sha": "e" * 40}, {"sha": HEAD}],
            }
        if "/attempts/" in path:
            return _source(path.split("/")[2], path.split("/")[2] == "210")
        if path.startswith("actions/runs/"):
            return self.runs[path.rsplit("/", 1)[1]]
        if path.startswith("actions/workflows/"):
            runs = []
            if self.find_after_dispatch:
                for body in self.dispatched:
                    cid = body["inputs"]["correlation_id"]
                    runs.append(
                        {
                            "id": 900,
                            "display_title": f"Rerun failed tests · PR #42 · {cid}",
                            "html_url": "https://github.com/x/actions/runs/900",
                            "status": "completed",
                            "conclusion": "success",
                        }
                    )
            return {"workflow_runs": runs}
        raise AssertionError(path)


def _record(repo, db, key="idem-0001") -> dict:
    snap = _snapshot(repo)
    lanes = rerun_failed.plan(snap, [snap["lanes"]["cpu"]["tests"][0]["key"]])
    record = rerun_actions.new_record(
        pr="42", actor="maintainer", idempotency_key=key, snap=snap, lanes=lanes
    )
    record, created = db.create(record)
    assert created
    return record


def test_action_cancels_confirmed_runs_then_dispatches_once(repo, tmp_path) -> None:
    db = rerun_actions.Store(str(tmp_path / "actions.sqlite3"))
    record = _record(repo, db)
    assert [c["run_id"] for c in record["cancel"]] == ["120:1"]
    github = FakeGitHub()
    rerun_actions.run_action(record["id"], db=db, github=github, sleep=lambda s: None)
    done = db.get(record["id"])
    assert done["state"] == "completed", done["events"]
    assert done["cancel"][0]["result"] == "cancelled"
    assert len(github.dispatched) == 1
    inputs = github.dispatched[0]["inputs"]
    assert inputs["pr_number"] == "42" and inputs["head_sha"] == HEAD
    assert inputs["tested_sha"] == "a" * 40
    assert inputs["source_run"] == "110:1"
    assert json.loads(inputs["selection"])["groups"][0]["tests"] == [NODE]
    assert done["lanes"]["cpu"]["run"]["id"] == "900"
    cancel_index = github.calls.index(("POST", "actions/runs/120/cancel"))
    dispatch_index = next(
        i for i, c in enumerate(github.calls) if c[1].endswith("/dispatches")
    )
    assert cancel_index < dispatch_index
    # A second drive (e.g. after a restart) changes nothing.
    rerun_actions.run_action(record["id"], db=db, github=github, sleep=lambda s: None)
    assert len(github.dispatched) == 1


def test_failed_cancellation_dispatches_nothing(repo, tmp_path) -> None:
    db = rerun_actions.Store(str(tmp_path / "actions.sqlite3"))
    record = _record(repo, db)
    github = FakeGitHub()
    github.cancel_status = 403
    rerun_actions.run_action(record["id"], db=db, github=github, sleep=lambda s: None)
    done = db.get(record["id"])
    assert done["state"] == "failed"
    assert "nothing was dispatched" in done["error"]
    assert github.dispatched == []


def test_run_that_will_not_stop_dispatches_nothing(repo, tmp_path, monkeypatch) -> None:
    db = rerun_actions.Store(str(tmp_path / "actions.sqlite3"))
    record = _record(repo, db)
    github = FakeGitHub()
    original = github.request

    def stubborn(method, path, body=None):
        if path.endswith("/cancel"):
            return 202, None  # accepted, but the run never reaches completed
        return original(method, path, body)

    github.request = stubborn
    clock = iter(range(0, 10_000, 100))
    monkeypatch.setattr(rerun_actions.time, "time", lambda: next(clock))
    rerun_actions.run_action(record["id"], db=db, github=github, sleep=lambda s: None)
    done = db.get(record["id"])
    assert done["state"] == "failed" and "still active" in done["error"]
    assert github.dispatched == []


def test_restart_after_dispatch_request_never_dispatches_again(repo, tmp_path) -> None:
    db = rerun_actions.Store(str(tmp_path / "actions.sqlite3"))
    record = _record(repo, db)
    record["cancel"] = []
    record["state"] = "dispatching"
    record["lanes"]["cpu"]["dispatch"] = "requested"  # crashed mid-call
    db.save(record)
    github = FakeGitHub()
    github.find_after_dispatch = False
    rerun_actions.run_action(record["id"], db=db, github=github, sleep=lambda s: None)
    done = db.get(record["id"])
    assert github.dispatched == []
    assert done["state"] == "failed" and "unknown" in done["error"]


def test_store_is_idempotent_and_allows_one_open_action_per_pr(repo, tmp_path) -> None:
    db = rerun_actions.Store(str(tmp_path / "actions.sqlite3"))
    first = _record(repo, db)
    again, created = db.create({**first, "id": "f" * 16})
    assert not created and again["id"] == first["id"]
    snap = _snapshot(repo)
    lanes = rerun_failed.plan(snap, [snap["lanes"]["cpu"]["tests"][0]["key"]])
    second = rerun_actions.new_record(
        pr="42", actor="other", idempotency_key="idem-0002", snap=snap, lanes=lanes
    )
    with pytest.raises(rerun_actions.Busy):
        db.create(second)
    first["state"] = "dispatched"
    db.save(first)
    assert db.create(second)[1] is True
    assert db.count_since("pr", "42", 0) == 2
    view = rerun_actions.public_view(db.get(first["id"]))
    assert "selection" not in json.dumps(view)


class _Handler(trace_exporter.MetricsHandler):
    """Drive the POST route without a socket."""

    def __init__(self, body: dict, headers: dict) -> None:
        raw = json.dumps(body).encode()
        self.headers = {"Content-Length": str(len(raw)), **headers}
        self.rfile = io.BytesIO(raw)
        self.replies: list[tuple[int, dict]] = []

    def _reply_json(self, status: int, body: dict) -> None:
        self.replies.append((status, body))


@pytest.fixture
def enabled(monkeypatch, tmp_path, repo):
    db = rerun_actions.Store(str(tmp_path / "actions.sqlite3"))
    serge = FakeSerge()
    started: list[str] = []
    monkeypatch.setattr(rerun_actions, "dispatch_enabled", lambda: (True, ""))
    monkeypatch.setattr(rerun_actions, "store", lambda: db)
    monkeypatch.setattr(rerun_actions, "default_serge", lambda: serge)
    monkeypatch.setattr(rerun_actions, "start", started.append)
    monkeypatch.setattr(
        trace_exporter.MetricsHandler,
        "_rerun_snapshot",
        lambda self, pr, api: _snapshot(repo),
    )
    monkeypatch.setattr(
        trace_exporter.MetricsHandler, "_action_user", lambda self: "maintainer"
    )
    return {"db": db, "serge": serge, "started": started, "snap": _snapshot(repo)}


def _post(body: dict, headers: dict | None = None) -> _Handler:
    handler = _Handler(body, headers if headers is not None else {"X-TCI-Action": "1"})
    handler._serve_rerun_failed_action()
    return handler


def _body(snap: dict, **extra) -> dict:
    return {
        "pr": "42",
        "version": snap["version"],
        "keys": [snap["lanes"]["cpu"]["tests"][0]["key"]],
        "confirm_active": [],
        "idempotency_key": "idem-0001",
        **extra,
    }


def test_post_requires_confirmation_of_the_exact_active_runs(enabled) -> None:
    snap = enabled["snap"]
    status, reply = _post(_body(snap)).replies[-1]
    assert status == 409 and reply["status"] == "confirm"
    assert [r["run_id"] for r in reply["active_runs"]] == ["120:1"]
    assert enabled["started"] == []

    status, reply = _post(_body(snap, confirm_active=["120:1"])).replies[-1]
    assert status == 202 and reply["action"]["state"] == "prepared"
    assert enabled["started"] == [reply["action"]["id"]]
    # The same request again returns the same action.
    status, again = _post(_body(snap, confirm_active=["120:1"])).replies[-1]
    assert status == 200 and again["action"]["id"] == reply["action"]["id"]
    # A second, different request for the PR waits for the first.
    status, busy = _post(
        _body(snap, confirm_active=["120:1"], idempotency_key="idem-0002")
    ).replies[-1]
    assert status == 409 and busy["status"] == "busy"


def test_post_refuses_unsafe_requests(enabled) -> None:
    snap = enabled["snap"]
    assert _post(_body(snap), headers={}).replies[-1][0] == 403  # no CSRF header
    assert _post(_body(snap, version="0" * 16)).replies[-1][1]["status"] == "stale"
    assert (
        _post(_body(snap, keys=["f" * 24])).replies[-1][1]["status"]
        == "stale_selection"
    )
    assert _post(_body(snap, pr="x")).replies[-1][0] == 400
    enabled["serge"].role = "read"
    assert _post(_body(snap)).replies[-1] == (403, {"status": "no_write_access"})


def test_post_refuses_a_pr_that_is_no_longer_open(enabled, repo) -> None:
    repo.pull["state"] = "closed"
    status, reply = _post(_body(enabled["snap"])).replies[-1]
    assert status == 409 and reply["status"] == "pr_not_open"


def test_post_is_off_until_enabled(monkeypatch) -> None:
    monkeypatch.delenv("PYTEST_TRACE_EXPORTER_RERUN_DISPATCH", raising=False)
    assert rerun_actions.dispatch_enabled()[0] is False
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_RERUN_DISPATCH", "1")
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_RERUN_DB", "/tmp/x.sqlite3")
    monkeypatch.delenv("PYTEST_TRACE_EXPORTER_SERGE_DASHBOARD_TOKEN", raising=False)
    assert rerun_actions.dispatch_enabled() == (
        False,
        "The serge connection is not configured.",
    )
    monkeypatch.delenv("PYTEST_TRACE_EXPORTER_RERUN_DISPATCH")
    handler = _Handler({}, {"X-TCI-Action": "1"})
    handler._serve_rerun_failed_action()
    assert handler.replies[-1][1]["status"] == "disabled"


def test_picker_posts_only_keys_and_shows_the_exact_warning() -> None:
    page = rerun_failed.PAGE_HTML
    assert "Are you sure? This will cancel ongoing runs" in page
    assert "'X-TCI-Action':'1'" in page
    assert "/rerun-failed/data?pr=" in page
    assert "/rerun-failed/actions/" in page
    # Upstream text is only ever set as text, never parsed as HTML.
    assert "innerHTML" not in page


def _rate_limited(url: str) -> trace_exporter.HTTPError:
    from email.message import Message

    headers = Message()
    headers["X-RateLimit-Remaining"] = "0"
    headers["X-RateLimit-Reset"] = "2000"
    return trace_exporter.HTTPError(url, 403, "Forbidden", headers, io.BytesIO(b"{}"))


def test_github_reads_are_cached_and_fresh_reads_refill_the_cache(monkeypatch) -> None:
    calls: list[str] = []
    payloads = iter([{"n": 1}, {"n": 2}, {"n": 3}])

    def fake_get(url: str) -> object:
        calls.append(url)
        return next(payloads)

    monkeypatch.setattr(trace_exporter, "_github_api_get", fake_get)
    monkeypatch.setattr(trace_exporter, "_rerun_github_cache", {})
    clock = [100.0]
    monkeypatch.setattr(trace_exporter.time, "monotonic", lambda: clock[0])

    assert trace_exporter._rerun_github_api("pulls/42") == {"n": 1}
    assert trace_exporter._rerun_github_api("pulls/42") == {"n": 1}
    assert len(calls) == 1
    # The action reads fresh, and the picker's reload then sees that read.
    assert trace_exporter._rerun_github_api("pulls/42", fresh=True) == {"n": 2}
    assert trace_exporter._rerun_github_api("pulls/42") == {"n": 2}
    clock[0] += trace_exporter.RERUN_GITHUB_CACHE_SECONDS
    assert trace_exporter._rerun_github_api("pulls/42") == {"n": 3}
    assert len(calls) == 3


def test_a_spent_github_budget_is_reported_with_its_reset(monkeypatch) -> None:
    def spent(url: str) -> object:
        raise _rate_limited(url)

    monkeypatch.setattr(trace_exporter, "_github_api_get", spent)
    monkeypatch.setattr(trace_exporter, "_rerun_github_cache", {})
    monkeypatch.setattr(trace_exporter.time, "time", lambda: 1900.0)
    handler = _Handler({}, {})
    monkeypatch.setattr(trace_exporter, "prometheus_base_url", lambda: "http://prom")
    handler._serve_rerun_failed_data({"pr": ["42"]})
    assert handler.replies == [
        (503, {"status": "github_rate_limited", "retry_after": 100})
    ]


def test_other_snapshot_failures_stay_source_unavailable(monkeypatch) -> None:
    def broken(self, pr, api):
        raise ValueError("Prometheus unavailable")

    monkeypatch.setattr(trace_exporter.MetricsHandler, "_rerun_snapshot", broken)
    handler = _Handler({}, {})
    handler._serve_rerun_failed_data({"pr": ["42"]})
    assert handler.replies == [(503, {"status": "source_unavailable"})]


def test_rerun_opener_is_built_in_the_parent_realm() -> None:
    """The panel's helper iframe is re-created on every refresh; listeners
    created in its realm stop firing, so the overlay could not be closed."""
    content = (ROOT / "dashboard/pytest-observability-pr-dashboard.json").read_text()
    assert "w.tciOpenRerunFailed=new w.Function('pr'," in content
    assert "w.tciOpenRerunFailed=function" not in content


def test_repository_checks_explain_how_to_fix_them() -> None:
    assert rerun_failed._ineligible_reason(
        "cpu", "utils/checkers.py::docstrings", "check_repository_consistency", "cpu"
    ) == (
        "repository check, not a test: run python utils/checkers.py docstrings "
        "locally and push the fix"
    )
    assert (
        rerun_failed._ineligible_reason("cpu", "tests_torch", "tests_torch", "cpu")
        == "job-level failure, not a single test"
    )


@pytest.mark.parametrize("lane", ["cpu", "gpu"])
def test_revision_validation_preserves_source_base(lane):
    requested = []
    tested = "a" * 40 if lane == "cpu" else "b" * 40

    def api(path):
        requested.append(path)
        if path == "pulls/42":
            return {
                "state": "open",
                "head": {"sha": HEAD},
                "merge_commit_sha": "f" * 40,
            }
        if path == "actions/runs/110/attempts/2":
            return _source("110", lane == "gpu", run_attempt=2)
        if path == f"commits/{tested}":
            return {"sha": tested, "parents": [{"sha": "e" * 40}, {"sha": HEAD}]}
        raise AssertionError(path)

    rerun_revision.validate(
        api, pr="42", lane=lane, head_sha=HEAD, tested_sha=tested, source_run="110:2"
    )
    assert requested == ["pulls/42", "actions/runs/110/attempts/2", f"commits/{tested}"]


@pytest.mark.parametrize(
    "problem",
    [
        "closed",
        "new_head",
        "wrong_attempt",
        "wrong_id",
        "active",
        "wrong_workflow",
        "wrong_event",
        "wrong_pr",
        "wrong_cpu_head",
        "wrong_merge_parent",
        "not_merge",
        "wrong_commit",
    ],
)
def test_revision_validation_rejects_bad_provenance(problem):
    pull = {"state": "open", "head": {"sha": HEAD}}
    run = _source("110", False)
    commit = {"sha": "a" * 40, "parents": [{"sha": "e" * 40}, {"sha": HEAD}]}
    if problem == "closed":
        pull["state"] = "closed"
    elif problem == "new_head":
        pull["head"]["sha"] = "d" * 40
    elif problem == "wrong_attempt":
        run["run_attempt"] = 2
    elif problem == "wrong_id":
        run["id"] = 111
    elif problem == "active":
        run["status"] = "in_progress"
    elif problem == "wrong_workflow":
        run["path"] = ".github/workflows/other.yml"
    elif problem == "wrong_event":
        run["event"] = "push"
    elif problem == "wrong_pr":
        run["pull_requests"] = [{"number": 43}]
    elif problem == "wrong_cpu_head":
        run["head_sha"] = "d" * 40
    elif problem == "wrong_merge_parent":
        commit["parents"][1]["sha"] = "d" * 40
    elif problem == "not_merge":
        commit["parents"] = [{"sha": HEAD}]
    elif problem == "wrong_commit":
        commit["sha"] = "d" * 40
    with pytest.raises(ValueError):
        rerun_revision.validate(
            lambda path: pull
            if path.startswith("pulls/")
            else run
            if path.startswith("actions/")
            else commit,
            pr="42",
            lane="cpu",
            head_sha=HEAD,
            tested_sha="a" * 40,
            source_run="110:1",
        )


@pytest.mark.parametrize("shas", [[], [""], ["a" * 40, "b" * 40], ["a" * 40, ""]])
def test_selection_rejects_missing_or_ambiguous_tested_commit(repo, shas):
    snap = _snapshot(repo)
    commits = rerun_failed.tested_commits(
        "42",
        lambda _: [{"metric": {"run_id": "110:1", "commit_sha": sha}} for sha in shas],
    )
    snap["lanes"]["cpu"]["source_run"]["commit"] = commits.get("110:1", "")
    with pytest.raises(rerun_failed.SelectionError, match="unambiguous tested commit"):
        rerun_failed.plan(snap, [snap["lanes"]["cpu"]["tests"][0]["key"]])


def test_invalid_source_stops_before_cancellation(repo, tmp_path):
    db = rerun_actions.Store(str(tmp_path / "actions.sqlite3"))
    record = _record(repo, db)
    github = FakeGitHub()
    original = github.get
    github.get = (
        lambda path: {"state": "open", "head": {"sha": "d" * 40}}
        if path == "pulls/42"
        else original(path)
    )
    rerun_actions.run_action(record["id"], db=db, github=github, sleep=lambda _: None)
    assert db.get(record["id"])["state"] == "failed"
    assert github.calls == []


def test_pr_moving_during_cancellation_prevents_dispatch(repo, tmp_path):
    db = rerun_actions.Store(str(tmp_path / "actions.sqlite3"))
    record = _record(repo, db)
    github = FakeGitHub()
    original = github.get
    github.get = (
        lambda path: {"state": "open", "head": {"sha": "d" * 40}}
        if path == "pulls/42" and github.calls
        else original(path)
    )
    rerun_actions.run_action(record["id"], db=db, github=github, sleep=lambda _: None)
    assert db.get(record["id"])["state"] == "failed"
    assert github.dispatched == []
    assert github.calls == [("POST", "actions/runs/120/cancel")]


def test_dispatch_keeps_distinct_cpu_gpu_source_revisions(repo, tmp_path):
    snap = _snapshot(repo)
    lanes = rerun_failed.plan(
        snap, [snap["lanes"][lane]["tests"][0]["key"] for lane in ("cpu", "gpu")]
    )
    db = rerun_actions.Store(str(tmp_path / "actions.sqlite3"))
    record = rerun_actions.new_record(
        pr="42", actor="maintainer", idempotency_key="both", snap=snap, lanes=lanes
    )
    record["cancel"] = []
    db.create(record)
    github = FakeGitHub()
    rerun_actions.run_action(record["id"], db=db, github=github, sleep=lambda _: None)
    assert db.get(record["id"])["state"] == "completed"
    assert [body["inputs"]["tested_sha"] for body in github.dispatched] == [
        "a" * 40,
        "b" * 40,
    ]
    assert [body["inputs"]["source_run"] for body in github.dispatched] == [
        "110:1",
        "210:1",
    ]


@pytest.mark.parametrize("lane", ["cpu", "gpu"])
def test_workflows_checkout_original_merge_after_main_moves(lane, tmp_path):
    import os

    origin = tmp_path / "origin"
    origin.mkdir()

    def git(*args, cwd=origin):
        return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()

    git("init", "-b", "main")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.org")
    (origin / "base").write_text("base")
    git("add", ".")
    git("commit", "-m", "base")
    git("checkout", "-b", "pr")
    (origin / "pr").write_text("PR")
    git("add", ".")
    git("commit", "-m", "PR")
    head = git("rev-parse", "HEAD")
    git("checkout", "main")
    git("merge", "--no-ff", "pr", "-m", "original merge")
    tested = git("rev-parse", "HEAD")
    # Keep the original object available, but point the PR merge ref elsewhere.
    git("update-ref", "refs/pull/42/merge", tested)
    (origin / "main-new").write_text("main advanced")
    git("add", ".")
    git("commit", "-m", "new base")
    current = git("rev-parse", "HEAD")
    git("update-ref", "refs/pull/42/merge", current)
    checkout = tmp_path / "checkout"
    git("clone", str(origin), str(checkout), cwd=tmp_path)
    workflow = yaml.safe_load(
        (ROOT / f".github/workflows/rerun-failed-{lane}.yml").read_text()
    )
    steps = workflow["jobs"]["run"]["steps"]
    if lane == "cpu":
        assert steps[0]["with"]["ref"] == "${{ inputs.tested_sha }}"
        # Match actions/checkout's SHA fetch with depth 2.
        git("fetch", "--depth=2", "origin", tested, cwd=checkout)
        git("checkout", "--detach", tested, cwd=checkout)
        script = steps[1]["run"]
    else:
        script = steps[0]["run"]
    result = subprocess.run(
        ["bash", "-e", "-c", script],
        cwd=checkout,
        env={
            **os.environ,
            "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig"),
            "PR_NUMBER": "42",
            "HEAD_SHA": head,
            "TESTED_SHA": tested,
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert git("rev-parse", "HEAD", cwd=checkout) == tested != current
    assert not (checkout / "main-new").exists()


@pytest.mark.parametrize("lane", ["cpu", "gpu"])
def test_caller_revision_contract(lane):
    caller = yaml.safe_load(
        (ROOT / f"docs/rerun-failed/transformers-rerun-failed-{lane}.yml").read_text()
    )
    reusable = yaml.safe_load(
        (ROOT / f".github/workflows/rerun-failed-{lane}.yml").read_text()
    )
    # PyYAML's YAML 1.1 loader parses the key 'on' as True.
    for name in ("head_sha", "tested_sha", "source_run"):
        assert caller[True]["workflow_dispatch"]["inputs"][name]["required"]
        assert reusable[True]["workflow_call"]["inputs"][name]["required"]
        assert caller["jobs"]["rerun"]["with"][name] == "${{ inputs." + name + " }}"
    assert caller["permissions"]["actions"] == "read"
    assert caller["concurrency"]["cancel-in-progress"] is True
