from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from transformersci.otel import github_app, rerun_actions, rerun_failed
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


@pytest.mark.skipif(shutil.which("openssl") is None, reason="needs openssl")
@pytest.mark.parametrize("pkcs8", [False, True])
def test_app_jwt_signature_verifies_with_openssl(tmp_path: Path, pkcs8: bool) -> None:
    raw = tmp_path / "raw.pem"
    subprocess.run(
        ["openssl", "genrsa", "-out", str(raw), "2048"], check=True, capture_output=True
    )
    key = tmp_path / "key.pem"
    convert = (
        ["openssl", "pkcs8", "-topk8", "-nocrypt"]
        if pkcs8
        else ["openssl", "rsa", "-traditional"]
    )
    converted = subprocess.run(
        [*convert, "-in", str(raw), "-out", str(key)], capture_output=True
    )
    if converted.returncode and not pkcs8:  # LibreSSL: PKCS#1 is its default
        subprocess.run(
            ["openssl", "rsa", "-in", str(raw), "-out", str(key)],
            check=True,
            capture_output=True,
        )
    pem = key.read_text()
    assert ("BEGIN RSA PRIVATE KEY" in pem) != pkcs8
    token = github_app.app_jwt("123", github_app.parse_private_key(pem), now=1000)
    header, payload, signature = token.split(".")
    claims = json.loads(github_app.base64.urlsafe_b64decode(payload + "=="))
    assert claims == {"iat": 940, "exp": 1540, "iss": "123"}
    pub = tmp_path / "pub.pem"
    subprocess.run(
        ["openssl", "rsa", "-in", str(key), "-pubout", "-out", str(pub)],
        check=True,
        capture_output=True,
    )
    (tmp_path / "msg").write_bytes(f"{header}.{payload}".encode())
    (tmp_path / "sig").write_bytes(
        github_app.base64.urlsafe_b64decode(signature + "=" * (-len(signature) % 4))
    )
    verified = subprocess.run(
        [
            "openssl",
            "dgst",
            "-sha256",
            "-verify",
            str(pub),
            "-signature",
            str(tmp_path / "sig"),
            str(tmp_path / "msg"),
        ],
        capture_output=True,
        text=True,
    )
    assert verified.stdout.strip() == "Verified OK"


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
    github = FakeGitHub()
    github.request_permission = "write"
    original = github.request

    def request(method, path, body=None):
        if "/permission" in path:
            return 200, {"role_name": github.request_permission}
        return original(method, path, body)

    github.request = request
    github.get = repo  # snapshot reads go to the fake repository
    started: list[str] = []
    monkeypatch.setattr(rerun_actions, "dispatch_enabled", lambda: (True, ""))
    monkeypatch.setattr(rerun_actions, "store", lambda: db)
    monkeypatch.setattr(rerun_actions, "default_github", lambda: github)
    monkeypatch.setattr(rerun_actions, "start", started.append)
    monkeypatch.setattr(
        trace_exporter.MetricsHandler,
        "_rerun_snapshot",
        lambda self, pr, api: _snapshot(repo),
    )
    monkeypatch.setattr(
        trace_exporter.MetricsHandler, "_action_user", lambda self: "maintainer"
    )
    return {"db": db, "github": github, "started": started, "snap": _snapshot(repo)}


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
    enabled["github"].request_permission = "read"
    assert _post(_body(snap)).replies[-1] == (403, {"status": "no_write_access"})


def test_post_refuses_a_pr_that_is_no_longer_open(enabled, repo) -> None:
    repo.pull["state"] = "closed"
    status, reply = _post(_body(enabled["snap"])).replies[-1]
    assert status == 409 and reply["status"] == "pr_not_open"


def test_post_is_off_until_enabled(monkeypatch) -> None:
    monkeypatch.delenv("PYTEST_TRACE_EXPORTER_RERUN_DISPATCH", raising=False)
    assert rerun_actions.dispatch_enabled()[0] is False
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
