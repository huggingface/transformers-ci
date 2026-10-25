"""Authenticated, durable MIC workflow dispatch through Serge; no GPU in the web pod."""

from __future__ import annotations

import json
import hmac
import os
from pathlib import Path
import re
import secrets
import sqlite3
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

from transformersci import github_app
from . import dashboard_auth as auth, rerun_actions

REPOSITORY = "huggingface/transformers-mic"
WORKFLOW = "cloud-baselines.yml"
SHA = re.compile(r"^[a-f0-9]{40}$")
ID = re.compile(r"^[a-f0-9]{32}$")
TERMINAL = {"completed", "failed", "cancelled", "unknown"}
MAX_ARCHIVE_BYTES = 256_000_000


def request_options(body):
    if not isinstance(body, dict) or set(body) - {"workspace"} != {
        "request_id",
        "fleet_revision",
        "fleets",
        "profiles",
        "transformers_ref",
        "fixtures_only",
        "upload_repo",
    }:
        raise auth.AuthError(400, "invalid_collection_request")
    if not isinstance(body["request_id"], str) or not ID.fullmatch(body["request_id"]):
        raise auth.AuthError(400, "invalid_request_id")
    if not isinstance(body["fleet_revision"], str) or not SHA.fullmatch(
        body["fleet_revision"]
    ):
        raise auth.AuthError(400, "immutable_fleet_revision_required")
    if body["fleets"] not in (["control"], ["prototypes"], ["control", "prototypes"]):
        raise auth.AuthError(400, "invalid_fleets")
    if body["profiles"] != ["cpu", "cuda"] or type(body["fixtures_only"]) is not bool:
        raise auth.AuthError(400, "invalid_collection_scope")
    if not isinstance(body["transformers_ref"], str) or (
        body["transformers_ref"] != "main"
        and not SHA.fullmatch(body["transformers_ref"])
    ):
        raise auth.AuthError(400, "immutable_framework_revision_or_main_required")
    destination = body["upload_repo"]
    if destination is not None:
        allowed = os.getenv("TCI_MIC_UPLOAD_NAMESPACES", "huggingface-mic").split(",")
        if (
            not isinstance(destination, str)
            or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", destination
            )
            or destination.split("/")[0] not in allowed
        ):
            raise auth.AuthError(403, "upload_namespace_not_allowed")
    if "workspace" in body:
        workspace = body["workspace"]
        allowed = os.getenv("TCI_MIC_INPUT_NAMESPACES", "huggingface-mic").split(",")
        if (
            not isinstance(workspace, dict)
            or set(workspace) != {"repo", "revision"}
            or not isinstance(workspace["repo"], str)
            or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*-mic",
                workspace["repo"],
            )
            or workspace["repo"].split("/")[0] not in allowed
            or not isinstance(workspace["revision"], str)
            or not SHA.fullmatch(workspace["revision"])
            or len(body["fleets"]) != 1
            or body["fixtures_only"]
        ):
            raise auth.AuthError(400, "invalid_model_workspace")
    return body


class Store:
    def __init__(self, path):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS mic_jobs (
            id TEXT PRIMARY KEY, actor_id INTEGER NOT NULL, request_id TEXT NOT NULL,
            request TEXT NOT NULL, record TEXT NOT NULL, created REAL NOT NULL,
            UNIQUE(actor_id, request_id))""")

    def get(self, job_id, actor_id):
        with self.lock:
            row = self.db.execute(
                "SELECT record FROM mic_jobs WHERE id=? AND actor_id=?",
                (job_id, actor_id),
            ).fetchone()
        if not row:
            raise auth.AuthError(404, "job_not_found")
        return json.loads(row[0])

    def existing(self, actor_id, options):
        with self.lock:
            row = self.db.execute(
                "SELECT request, record FROM mic_jobs WHERE actor_id=? AND request_id=?",
                (actor_id, options["request_id"]),
            ).fetchone()
        if row:
            if json.loads(row[0]) != options:
                raise auth.AuthError(409, "request_id_reused_with_different_options")
            return json.loads(row[1])

    def save(self, record):
        with self.lock:
            current = self.db.execute(
                "SELECT record FROM mic_jobs WHERE id=?", (record["id"],)
            ).fetchone()
            if current:
                persisted = json.loads(current[0])
                for key in ("progress_run_verified", "run_id"):
                    if key in persisted:
                        record.setdefault(key, persisted[key])
                record["progress"] = {
                    **record.get("progress", {}),
                    **json.loads(current[0]).get("progress", {}),
                }
            self.db.execute(
                "UPDATE mic_jobs SET record=? WHERE id=?",
                (json.dumps(record), record["id"]),
            )

    def progress(self, body):
        with self.lock:
            row = self.db.execute(
                "SELECT record FROM mic_jobs WHERE id=?", (body["job_id"],)
            ).fetchone()
            if not row:
                raise auth.AuthError(404, "job_not_found")
            record = json.loads(row[0])
        if record.get("progress_run_verified") != body["run_id"]:
            run = github_json(REPOSITORY, f"actions/runs/{body['run_id']}")
            if (
                run.get("display_title") != "MIC baselines · " + record["id"]
                or run.get("event") != "workflow_dispatch"
                or str(run.get("path", "")).split("@", 1)[0]
                != ".github/workflows/" + WORKFLOW
            ):
                raise auth.AuthError(403, "run_does_not_belong_to_job")
        with self.lock:
            record = json.loads(
                self.db.execute(
                    "SELECT record FROM mic_jobs WHERE id=?", (body["job_id"],)
                ).fetchone()[0]
            )
            if record.get("run_id", body["run_id"]) != body["run_id"]:
                raise auth.AuthError(403, "run_does_not_belong_to_job")
            record["run_id"] = body["run_id"]
            record["progress_run_verified"] = body["run_id"]
            progress = record.setdefault("progress", {})
            old = progress.get(body["profile"], {})
            if (body["run_attempt"], body["sequence"]) > (
                old.get("run_attempt", 0),
                old.get("sequence", -1),
            ):
                progress[body["profile"]] = {**body, "updated": time.time()}
                self.db.execute(
                    "UPDATE mic_jobs SET record=? WHERE id=?",
                    (json.dumps(record), record["id"]),
                )

    def create(self, actor, options, github_get):
        with self.lock:
            if previous := self.existing(actor["user_id"], options):
                return previous, False
            now = time.time()
            rows = self.db.execute(
                "SELECT actor_id, created, record FROM mic_jobs"
            ).fetchall()
            if (
                sum(
                    uid == actor["user_id"] and created > now - 3600
                    for uid, created, _ in rows
                )
                >= 3
            ):
                raise auth.AuthError(429, "rate_limited")
            if (
                sum(
                    json.loads(raw)["status"] not in TERMINAL
                    and created > now - 8 * 3600
                    for _, created, raw in rows
                )
                >= 2
            ):
                raise auth.AuthError(429, "gpu_capacity_busy")
            # Only known repositories and the default MIC code are executable.
            framework = github_get(
                "huggingface/transformers", "commits/" + options["transformers_ref"]
            )["sha"]
            mic = github_get(REPOSITORY, "commits/main")["sha"]
            if not SHA.fullmatch(framework) or not SHA.fullmatch(mic):
                raise auth.AuthError(502, "invalid_github_revision")
            record = {
                "id": secrets.token_hex(16),
                "actor_id": actor["user_id"],
                "actor": actor["login"],
                "fleet_revision": options["fleet_revision"],
                "transformers_sha": framework,
                "mic_sha": mic,
                "fleets": options["fleets"],
                "profiles": options["profiles"],
                "fixtures_only": options["fixtures_only"],
                "upload_repo": options["upload_repo"],
                "status": "dispatching",
                "created": now,
                "artifacts": [],
            }
            if "workspace" in options:
                record["workspace"] = options["workspace"]
            self.db.execute(
                "INSERT INTO mic_jobs VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record["id"],
                    actor["user_id"],
                    options["request_id"],
                    json.dumps(options),
                    json.dumps(record),
                    now,
                ),
            )
            return record, True


_stores = {}
_lock = threading.Lock()


def store():
    root = os.getenv("PYTEST_TRACE_EXPORTER_RUN_STORE", "")
    if not root:
        raise auth.AuthError(503, "persistent_job_store_required")
    path = str(Path(root) / "mic-jobs.sqlite3")
    with _lock:
        if path not in _stores:
            _stores[path] = Store(path)
        return _stores[path]


def github_json(repository, path):
    resource = f"repos/{repository}/{path}" if repository else path
    request = Request(
        f"https://api.github.com/{resource}",
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "transformers-ci-mic-jobs",
            "Authorization": "Bearer "
            + (github_app.read_token(("PYTEST_GITHUB_TOKEN",)) or ""),
        },
    )
    with urlopen(request, timeout=15) as response:
        return json.loads(response.read(4_000_000))


def permission(actor):
    if github_json(None, "users/" + actor["login"]).get("id") != actor["user_id"]:
        raise auth.AuthError(403, "github_identity_changed_sign_in_again")
    status, response = rerun_actions.default_serge().call(
        "permission", {"repository": REPOSITORY, "actor": actor["login"]}
    )
    if status != 200:
        raise auth.AuthError(503, "serge_unavailable")
    if response.get("can_write") is not True:
        raise auth.AuthError(403, "repository_write_access_required")


def submit(db, actor, options):
    options = request_options(options)
    permission(actor)  # authentication never implies GPU spending permission
    record, created = db.create(actor, options, github_json)
    if created:
        inputs = {
            key: record[key]
            for key in ("fleet_revision", "transformers_sha", "mic_sha")
        }
        inputs.update(
            job_id=record["id"],
            fleets=",".join(record["fleets"]),
            fixtures_only=str(record["fixtures_only"]).lower(),
            upload_repo=record["upload_repo"] or "",
        )
        if "workspace" in record:
            inputs.update(
                workspace_repo=record["workspace"]["repo"],
                workspace_revision=record["workspace"]["revision"],
            )
        try:
            status, _ = rerun_actions.default_serge().call(
                "workflows/dispatch",
                {
                    "repository": REPOSITORY,
                    "actor": actor["login"],
                    "workflow": WORKFLOW,
                    "inputs": inputs,
                },
            )
            record["status"] = "queued" if status == 200 else "dispatch-uncertain"
        except OSError:
            record["status"] = "dispatch-uncertain"
        # Intent was persisted before dispatch. A retry never creates another run.
        db.save(record)
    return record


def refresh(db, record, github_get=github_json):
    if record["status"] in TERMINAL:
        return record
    if "run_id" not in record:
        runs = github_get(
            REPOSITORY,
            f"actions/workflows/{WORKFLOW}/runs?event=workflow_dispatch&per_page=100",
        )["workflow_runs"]
        matching = [
            run
            for run in runs
            if run.get("display_title") == "MIC baselines · " + record["id"]
        ]
        if len(matching) > 1:
            record["status"] = "unknown"
        elif matching:
            record["run_id"] = matching[0]["id"]
        elif time.time() - record["created"] > 600:
            record["status"] = "unknown"
    if "run_id" in record:
        run = github_get(REPOSITORY, f"actions/runs/{record['run_id']}")
        record["run_url"] = (
            f"https://github.com/{REPOSITORY}/actions/runs/{record['run_id']}"
        )
        if run["status"] != "completed":
            record["status"] = "running" if run["status"] == "in_progress" else "queued"
            workflow_jobs = github_get(
                REPOSITORY, f"actions/runs/{record['run_id']}/jobs?per_page=100"
            ).get("jobs", [])
            record["workflow_jobs"] = [
                {
                    "name": job["name"],
                    "status": job["status"],
                    "step": next(
                        (
                            step["name"]
                            for step in job.get("steps", [])
                            if step["status"] == "in_progress"
                        ),
                        "",
                    ),
                }
                for job in workflow_jobs
            ]
        else:
            artifacts = github_get(
                REPOSITORY, f"actions/runs/{record['run_id']}/artifacts?per_page=100"
            )["artifacts"]
            record["artifacts"] = [
                {
                    "id": artifact["id"],
                    "profile": artifact["name"].removeprefix("mic-baselines-"),
                }
                for artifact in artifacts
                if artifact["name"] in ("mic-baselines-cpu", "mic-baselines-cuda")
                and not artifact.get("expired")
            ]
            publication = [
                artifact
                for artifact in artifacts
                if artifact["name"] == "mic-baselines-publication"
                and not artifact.get("expired")
            ]
            if len(publication) == 1:
                record["publication_artifact"] = {"id": publication[0]["id"]}
            if run.get("conclusion") == "success":
                if sorted(artifact["profile"] for artifact in record["artifacts"]) == [
                    "cpu",
                    "cuda",
                ] and (not record["upload_repo"] or "publication_artifact" in record):
                    record["status"] = "completed"
                elif (
                    time.time()
                    - (record.setdefault("artifact_wait_started", time.time()))
                    > 180
                ):
                    record["status"] = "failed"
                    record["error"] = "missing_profile_artifacts"
            else:
                record["status"] = (
                    "cancelled" if run.get("conclusion") == "cancelled" else "failed"
                )
    if time.time() - record["created"] > 8 * 3600 and record["status"] not in TERMINAL:
        record["status"] = "unknown"
    db.save(record)
    return record


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


def download_artifact(artifact_id):
    """GitHub's signed storage redirect must never receive the App token."""
    request = Request(
        f"https://api.github.com/repos/{REPOSITORY}/actions/artifacts/{artifact_id}/zip",
        headers={
            "Authorization": "Bearer "
            + (github_app.read_token(("PYTEST_GITHUB_TOKEN",)) or ""),
            "User-Agent": "transformers-ci-mic-jobs",
        },
    )
    try:
        response = build_opener(NoRedirect()).open(request, timeout=30)
    except HTTPError as error:
        if error.code not in (301, 302, 303, 307, 308):
            raise
        location = error.headers.get("Location", "")
        parsed = urlsplit(location)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or not parsed.hostname.endswith(
                (
                    ".blob.core.windows.net",
                    ".actions.githubusercontent.com",
                    ".githubusercontent.com",
                )
            )
        ):
            raise auth.AuthError(502, "invalid_artifact_storage_location") from None
        response = urlopen(
            Request(location), timeout=30
        )  # deliberately no Authorization
    with response:
        payload = response.read(MAX_ARCHIVE_BYTES + 1)
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise auth.AuthError(413, "artifact_too_large")
    return payload


def handle(handler, method):
    path = urlsplit(handler.path).path
    if path != "/mic-jobs" and not path.startswith("/mic-jobs/"):
        return False
    handler._request_route = "/mic-jobs"
    try:
        if os.getenv("TCI_MIC_JOBS_ENABLED") != "1":
            raise auth.AuthError(503, "mic_jobs_not_enabled")
        if path == "/mic-jobs/progress" and method == "POST":
            configured = rerun_actions.serge_token()
            supplied = handler.headers.get("Authorization", "")
            if not configured or not hmac.compare_digest(
                supplied, "Bearer " + configured
            ):
                raise auth.AuthError(401, "invalid_progress_service_token")
            body = auth._body(handler)
            required = {
                "job_id",
                "profile",
                "stage",
                "completed",
                "total",
                "detail",
                "sequence",
                "run_id",
                "run_attempt",
            }
            if (
                set(body) != required
                or body.get("profile") not in ("cpu", "cuda", "publish")
                or not isinstance(body["job_id"], str)
                or not ID.fullmatch(body["job_id"])
            ):
                raise auth.AuthError(400, "invalid_progress")
            for key in ("completed", "total", "sequence", "run_id", "run_attempt"):
                if type(body[key]) is not int or body[key] < 0:
                    raise auth.AuthError(400, "invalid_progress")
            store().progress(body)
            handler._reply_json(200, {"status": "ok"})
            return True
        if handler.headers.get("Authorization"):
            actor = auth.identity(handler.headers, cli=True)
            needed = "mic:submit" if method == "POST" else "mic:read"
            if actor and needed not in actor["scope"].split():
                raise auth.AuthError(403, "insufficient_scope")
        else:
            cfg = auth.config()
            actor = (
                auth._browser_post(handler, cfg)
                if method == "POST" and cfg
                else auth.identity(handler.headers)
            )
        if not actor:
            raise auth.AuthError(401, "unauthorized")
        db = store()
        if path == "/mic-jobs" and method == "POST":
            handler._reply_json(202, submit(db, actor, auth._body(handler)))
        elif method == "GET":
            parts = path.strip("/").split("/")
            if len(parts) not in (2, 4) or not ID.fullmatch(parts[1]):
                raise auth.AuthError(404, "job_not_found")
            record = refresh(db, db.get(parts[1], actor["user_id"]))
            if len(parts) == 2:
                handler._reply_json(200, record)
            else:
                if (
                    parts[2] != "artifacts"
                    or not parts[3].isdigit()
                    or int(parts[3])
                    not in (
                        {item["id"] for item in record["artifacts"]}
                        | {record.get("publication_artifact", {}).get("id")}
                    )
                ):
                    raise auth.AuthError(404, "artifact_not_found")
                handler._send(
                    200,
                    "application/zip",
                    download_artifact(int(parts[3])),
                    cache_control="no-store",
                )
        else:
            raise auth.AuthError(405, "method_not_allowed")
    except auth.AuthError as error:
        handler._reply_json(error.status, {"error": error.error})
    except (OSError, ValueError, KeyError, sqlite3.Error):
        handler._reply_json(503, {"error": "mic_service_unavailable"})
    return True
