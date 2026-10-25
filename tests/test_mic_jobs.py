from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from io import BytesIO
import json

import pytest

from transformersci.otel import mic_jobs as jobs

ACTOR = {"user_id": 42, "login": "maintainer", "scope": "mic:submit mic:read"}


def options(**updates):
    return {
        "request_id": "a" * 32,
        "fleet_revision": "b" * 40,
        "fleets": ["control", "prototypes"],
        "profiles": ["cpu", "cuda"],
        "transformers_ref": "main",
        "fixtures_only": False,
        "upload_repo": None,
        **updates,
    }


def github(repository, path):
    if repository is None:
        return {"id": ACTOR["user_id"]}
    return {"sha": "c" * 40 if repository == jobs.REPOSITORY else "d" * 40}


class Serge:
    def __init__(self, allowed=True, uncertain=False):
        self.allowed, self.uncertain, self.calls = allowed, uncertain, []

    def call(self, operation, body):
        self.calls.append((operation, body))
        if operation == "permission":
            return 200, {"can_write": self.allowed}
        if self.uncertain:
            raise TimeoutError("The workflow may have been dispatched")
        return 200, {"dispatched": True}


@pytest.fixture
def service(tmp_path, monkeypatch):
    db = jobs.Store(tmp_path / "jobs.sqlite3")
    serge = Serge()
    monkeypatch.setattr(jobs.rerun_actions, "default_serge", lambda: serge)
    monkeypatch.setattr(jobs, "github_json", github)
    return db, serge


@pytest.mark.parametrize(
    "patch",
    [
        {"fleet_revision": "main"},
        {"transformers_ref": "feature;command"},
        {"profiles": ["cuda"]},
        {"fixtures_only": "false"},
        {"fleets": []},
        {"request_id": "bad"},
        {"workflow": "other.yml"},
        {"upload_repo": "another-org/data"},
    ],
)
def test_unpinned_or_executable_client_parameters_are_rejected(patch):
    with pytest.raises(jobs.auth.AuthError):
        jobs.request_options(options(**patch))


def test_dispatch_is_authorized_pinned_and_idempotent_across_restart(service, tmp_path):
    db, serge = service
    first = jobs.submit(db, ACTOR, options())
    assert first["status"] == "queued"
    assert first["mic_sha"] == "c" * 40 and first["transformers_sha"] == "d" * 40
    second = jobs.submit(jobs.Store(tmp_path / "jobs.sqlite3"), ACTOR, options())
    assert first["id"] == second["id"]
    dispatched = [
        body for operation, body in serge.calls if operation == "workflows/dispatch"
    ]
    assert len(dispatched) == 1
    body = dispatched[0]
    assert body["repository"] == jobs.REPOSITORY and body["workflow"] == jobs.WORKFLOW
    assert body["actor"] == ACTOR["login"]
    assert body["inputs"]["fleets"] == "control,prototypes"
    assert body["inputs"]["fixtures_only"] == "false"
    with pytest.raises(jobs.auth.AuthError) as error:
        jobs.submit(db, ACTOR, options(fixtures_only=True))
    assert error.value.status == 409


def test_concurrent_retries_dispatch_once(service):
    db, serge = service
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(lambda _: jobs.submit(db, ACTOR, options()), range(2)))
    assert records[0]["id"] == records[1]["id"]
    assert sum(operation == "workflows/dispatch" for operation, _ in serge.calls) == 1


def test_ambiguous_dispatch_is_not_repeated(service):
    db, serge = service
    serge.uncertain = True
    first = jobs.submit(db, ACTOR, options())
    assert first["status"] == "dispatch-uncertain"
    assert jobs.submit(db, ACTOR, options())["id"] == first["id"]
    assert sum(operation == "workflows/dispatch" for operation, _ in serge.calls) == 1


def test_model_snapshot_is_pinned_authorized_and_bound_to_dispatch(service):
    db, serge = service
    workspace = {"repo": "huggingface-mic/demo-mic", "revision": "e" * 40}
    result = jobs.submit(db, ACTOR, options(fleets=["prototypes"], workspace=workspace))
    assert result["workspace"] == workspace
    dispatched = [body for op, body in serge.calls if op == "workflows/dispatch"]
    assert dispatched[0]["inputs"]["workspace_repo"] == workspace["repo"]
    assert dispatched[0]["inputs"]["workspace_revision"] == "e" * 40


@pytest.mark.parametrize(
    "workspace",
    [
        {"repo": "unapproved/demo-mic", "revision": "e" * 40},
        {"repo": "huggingface-mic/demo-mic", "revision": "main"},
        {
            "repo": "huggingface-mic/demo-mic",
            "revision": "e" * 40,
            "command": "arbitrary",
        },
        {"repo": "huggingface-mic/demo", "revision": "e" * 40},
    ],
)
def test_unapproved_or_mutable_model_snapshots_are_rejected(workspace):
    with pytest.raises(jobs.auth.AuthError):
        jobs.request_options(options(fleets=["prototypes"], workspace=workspace))


def test_model_input_cannot_silently_drop_checkpoints_or_repeat_in_two_fleets():
    workspace = {"repo": "huggingface-mic/demo-mic", "revision": "e" * 40}
    for patch in ({"fixtures_only": True, "fleets": ["prototypes"]}, {}):
        with pytest.raises(jobs.auth.AuthError):
            jobs.request_options(options(workspace=workspace, **patch))


def test_no_write_permission_means_no_dispatch(service):
    db, serge = service
    serge.allowed = False
    with pytest.raises(jobs.auth.AuthError) as error:
        jobs.submit(db, ACTOR, options())
    assert error.value.status == 403
    assert not any(operation == "workflows/dispatch" for operation, _ in serge.calls)


def test_stale_login_cannot_inherit_a_different_accounts_permissions(service):
    db, serge = service
    with pytest.raises(jobs.auth.AuthError) as error:
        jobs.submit(db, {**ACTOR, "user_id": 999}, options())
    assert error.value.status == 403 and serge.calls == []


def test_job_and_artifact_ownership_are_private(service):
    db, _ = service
    record = jobs.submit(db, ACTOR, options())
    with pytest.raises(jobs.auth.AuthError) as error:
        db.get(record["id"], 99)
    assert error.value.status == 404


def test_refresh_correlates_exact_run_and_requires_both_profiles(service):
    db, _ = service
    record = jobs.submit(db, ACTOR, options())
    artifacts = [{"id": 1, "name": "mic-baselines-cpu", "expired": False}]

    def read(repo, path):
        if "workflows/" in path:
            return {
                "workflow_runs": [
                    {"id": 99, "display_title": "MIC baselines · wrong"},
                    {"id": 100, "display_title": "MIC baselines · " + record["id"]},
                ]
            }
        if "artifacts" in path:
            return {"artifacts": artifacts}
        return {"status": "completed", "conclusion": "success"}

    updated = jobs.refresh(db, record, read)
    assert updated["run_id"] == 100 and updated["status"] != "completed"
    artifacts.append({"id": 2, "name": "mic-baselines-cuda", "expired": False})
    assert jobs.refresh(db, updated, read)["status"] == "completed"


def test_remote_upload_is_dispatched_and_requires_a_publication_receipt(service):
    db, serge = service
    record = jobs.submit(db, ACTOR, options(upload_repo="huggingface-mic/candidates"))
    assert serge.calls[-1][1]["inputs"]["upload_repo"] == "huggingface-mic/candidates"
    artifacts = [
        {"id": 1, "name": "mic-baselines-cpu"},
        {"id": 2, "name": "mic-baselines-cuda"},
    ]

    def read(repo, path):
        if "workflows/" in path:
            return {
                "workflow_runs": [
                    {"id": 100, "display_title": "MIC baselines · " + record["id"]}
                ]
            }
        if "artifacts" in path:
            return {"artifacts": artifacts}
        return {"status": "completed", "conclusion": "success"}

    updated = jobs.refresh(db, record, read)
    assert updated["status"] != "completed"
    artifacts.append({"id": 3, "name": "mic-baselines-publication"})
    updated = jobs.refresh(db, updated, read)
    assert updated["status"] == "completed" and updated["publication_artifact"] == {
        "id": 3
    }


class Handler:
    def __init__(self, path, body=None):
        self.path = path
        self.headers = Message()
        payload = json.dumps(body).encode() if body is not None else b""
        self.headers["Authorization"] = "Bearer session"
        self.headers["Content-Type"] = "application/json"
        self.headers["Content-Length"] = str(len(payload))
        self.rfile = BytesIO(payload)

    def _reply_json(self, status, body):
        self.reply = status, body

    def _send(self, status, content_type, payload, **kwargs):
        self.reply = status, payload


def test_api_rejects_anonymous_and_insufficient_cli_scopes(service, monkeypatch):
    db, _ = service
    monkeypatch.setenv("TCI_MIC_JOBS_ENABLED", "1")
    monkeypatch.setattr(jobs, "store", lambda: db)
    monkeypatch.setattr(jobs.auth, "identity", lambda *a, **kw: None)
    handler = Handler("/mic-jobs", options())
    assert jobs.handle(handler, "POST") and handler.reply[0] == 401
    monkeypatch.setattr(
        jobs.auth, "identity", lambda *a, **kw: {**ACTOR, "scope": "profile:read"}
    )
    jobs.handle(handler, "POST")
    assert handler.reply[0] == 403
    monkeypatch.setattr(jobs.auth, "identity", lambda *a, **kw: ACTOR)
    jobs.handle(handler, "POST")
    assert handler.reply[0] == 202


def test_artifact_api_only_serves_the_owned_runs_artifact(service, monkeypatch):
    db, _ = service
    record = jobs.submit(db, ACTOR, options())
    record.update(status="completed", artifacts=[{"id": 12, "profile": "cpu"}])
    db.save(record)
    monkeypatch.setenv("TCI_MIC_JOBS_ENABLED", "1")
    monkeypatch.setattr(jobs, "store", lambda: db)
    monkeypatch.setattr(jobs.auth, "identity", lambda *a, **kw: ACTOR)
    downloads = []
    monkeypatch.setattr(
        jobs, "download_artifact", lambda id: downloads.append(id) or b"zip"
    )
    handler = Handler(f"/mic-jobs/{record['id']}/artifacts/999")
    jobs.handle(handler, "GET")
    assert handler.reply[0] == 404 and not downloads
    handler = Handler(f"/mic-jobs/{record['id']}/artifacts/12")
    jobs.handle(handler, "GET")
    assert handler.reply == (200, b"zip") and downloads == [12]


def test_job_service_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("TCI_MIC_JOBS_ENABLED", raising=False)
    handler = Handler("/mic-jobs", options())
    jobs.handle(handler, "POST")
    assert handler.reply == (503, {"error": "mic_jobs_not_enabled"})


def test_verified_progress_is_bound_to_run_and_survives_stale_poll_updates(
    service, monkeypatch
):
    db, _ = service
    record = jobs.submit(db, ACTOR, options())
    stale = dict(record)
    monkeypatch.setattr(
        jobs,
        "github_json",
        lambda repo, path: {
            "display_title": "MIC baselines · " + record["id"],
            "event": "workflow_dispatch",
            "path": ".github/workflows/cloud-baselines.yml",
        },
    )
    event = {
        "job_id": record["id"],
        "profile": "cuda",
        "stage": "recording",
        "completed": 1,
        "total": 10,
        "detail": "Demo/default",
        "sequence": 2,
        "run_id": 123,
        "run_attempt": 1,
    }
    db.progress(event)
    db.progress({**event, "sequence": 1, "completed": 0})
    db.save(stale)
    updated = db.get(record["id"], 42)
    assert updated["progress"]["cuda"]["completed"] == 1
    assert updated["status"] == "queued"  # progress never marks a workflow successful
    monkeypatch.setattr(
        jobs,
        "github_json",
        lambda repo, path: {
            "display_title": "another job",
            "event": "workflow_dispatch",
            "path": ".github/workflows/cloud-baselines.yml",
        },
    )
    with pytest.raises(jobs.auth.AuthError):
        db.progress({**event, "run_id": 456, "sequence": 3})


def test_cli_credentials_cannot_impersonate_the_progress_relay(service, monkeypatch):
    db, _ = service
    record = jobs.submit(db, ACTOR, options())
    monkeypatch.setenv("TCI_MIC_JOBS_ENABLED", "1")
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_SERGE_DASHBOARD_TOKEN", "service-secret")
    handler = Handler("/mic-jobs/progress", {"job_id": record["id"]})
    jobs.handle(handler, "POST")
    assert handler.reply[0] == 401
