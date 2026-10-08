"""Durable rerun actions: cancel the confirmed source runs, then dispatch at
most one targeted workflow run per lane.

The exporter holds no GitHub write credential: serge makes the writes with its
App (``/dashboard/*`` on serge, behind a shared service token, checking the
acting user's write access itself). GitHub reads use the exporter's own token.

An action is a row in a SQLite database on the exporter's persistent volume, so
a restart resumes it rather than repeating it. Two database rules carry the
safety: an idempotency key is UNIQUE (a retried POST returns the first action),
and at most one action per PR may sit in a pre-dispatch state (a second click
gets 409, not a second cancellation). A lane's dispatch is recorded as requested
*before* the GitHub call, so a restart that cannot find the run reports the
outcome as unknown instead of dispatching again.

States: prepared -> cancelling -> dispatching -> dispatched -> completed, or
failed from any of them (a lane that dispatched still lists its run).
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener, urlopen

from . import rerun_failed

OPEN_STATES = ("prepared", "cancelling", "dispatching")
LIVE_STATES = (*OPEN_STATES, "dispatched")
CANCEL_DEADLINE_SECONDS = 300
FIND_RUN_SECONDS = 90
WATCH_INTERVAL_SECONDS = 30
WATCH_DEADLINE_SECONDS = 8 * 3600
MAX_ACTIONS_PER_HOUR = 6
MAX_LIVE_GPU_ACTIONS = 3
WRITE_ROLES = frozenset({"admin", "maintain", "write"})
LANE_TITLES = {"cpu": "CPU", "gpu": "GPU"}


class Busy(Exception):
    """Another action for this PR has not dispatched yet."""


def dispatch_enabled() -> tuple[bool, str]:
    if os.getenv("PYTEST_TRACE_EXPORTER_RERUN_DISPATCH", "").strip() != "1":
        return False, "Dispatch is not enabled yet."
    if not (serge_url() and serge_token()):
        return False, "The serge connection is not configured."
    if not database_path():
        return False, "No persistent store for rerun actions."
    return True, ""


def serge_url() -> str:
    return os.getenv("PYTEST_TRACE_EXPORTER_SERGE_URL", "").strip().rstrip("/")


def serge_token() -> str:
    return os.getenv("PYTEST_TRACE_EXPORTER_SERGE_DASHBOARD_TOKEN", "").strip()


def database_path() -> str:
    explicit = os.getenv("PYTEST_TRACE_EXPORTER_RERUN_DB", "").strip()
    if explicit:
        return explicit
    store = os.getenv("PYTEST_TRACE_EXPORTER_RUN_STORE", "").rstrip("/")
    return f"{store}/rerun-actions.sqlite3" if store else ""


class Store:
    def __init__(self, path: str) -> None:
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS actions (
                id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                pr TEXT NOT NULL,
                actor TEXT NOT NULL,
                state TEXT NOT NULL,
                created REAL NOT NULL,
                record TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS actions_one_open_per_pr
                ON actions (pr) WHERE state IN ('prepared', 'cancelling', 'dispatching');
            CREATE INDEX IF NOT EXISTS actions_by_pr ON actions (pr, created);
            """
        )

    def create(self, record: dict) -> tuple[dict, bool]:
        """Insert ``record``; (existing action, False) for a repeated
        idempotency key. Raises Busy when the PR has an open action."""
        with self._lock:
            row = self._db.execute(
                "SELECT record FROM actions WHERE idempotency_key = ?",
                (record["idempotency_key"],),
            ).fetchone()
            if row:
                return json.loads(row[0]), False
            try:
                self._db.execute(
                    "INSERT INTO actions VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        record["id"],
                        record["idempotency_key"],
                        record["pr"],
                        record["actor"],
                        record["state"],
                        record["created"],
                        json.dumps(record),
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise Busy(str(error)) from error
            return record, True

    def save(self, record: dict) -> None:
        record["updated"] = time.time()
        with self._lock:
            self._db.execute(
                "UPDATE actions SET state = ?, record = ? WHERE id = ?",
                (record["state"], json.dumps(record), record["id"]),
            )

    def get(self, action_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute(
                "SELECT record FROM actions WHERE id = ?", (action_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def by_idempotency_key(self, key: str) -> dict | None:
        with self._lock:
            row = self._db.execute(
                "SELECT record FROM actions WHERE idempotency_key = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def latest_for_pr(self, pr: str) -> dict | None:
        with self._lock:
            row = self._db.execute(
                "SELECT record FROM actions WHERE pr = ? ORDER BY created DESC LIMIT 1",
                (pr,),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def ids_in(self, states: tuple[str, ...]) -> list[str]:
        marks = ",".join("?" * len(states))
        with self._lock:
            rows = self._db.execute(
                f"SELECT id FROM actions WHERE state IN ({marks})", states
            ).fetchall()
        return [row[0] for row in rows]

    def count_since(self, column: str, value: str, since: float) -> int:
        if column not in ("actor", "pr"):
            raise ValueError(column)
        with self._lock:
            return self._db.execute(
                f"SELECT COUNT(*) FROM actions WHERE {column} = ? AND created >= ?",
                (value, since),
            ).fetchone()[0]

    def live_gpu_actions(self) -> int:
        count = 0
        for action_id in self.ids_in(LIVE_STATES):
            record = self.get(action_id) or {}
            count += "gpu" in record.get("lanes", {})
        return count


_store: Store | None = None
_store_lock = threading.Lock()


def store() -> Store:
    global _store
    with _store_lock:
        if _store is None:
            _store = Store(database_path())
        return _store


class Serge:
    """serge's dashboard API: the GitHub writes, made with serge's App."""

    def __init__(self, url: str, token: str) -> None:
        self.url = url
        self.token = token

    def call(self, operation: str, body: dict) -> tuple[int, dict]:
        request = Request(
            f"{self.url}/dashboard/{operation}",
            data=json.dumps({"repository": rerun_failed.REPOSITORY, **body}).encode(),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "User-Agent": "transformersci-rerun-failed",
            },
        )
        try:
            # In-cluster Service: never through a proxy.
            with build_opener(ProxyHandler({})).open(request, timeout=30) as response:
                status, raw = response.status, response.read()
        except HTTPError as error:
            status, raw = error.code, error.read()
        try:
            payload = json.loads(raw) if raw else {}
        except ValueError:
            payload = {}
        return status, payload if isinstance(payload, dict) else {}


def _github_read(path: str) -> tuple[int, object]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "transformersci-rerun-failed",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.getenv("PYTEST_GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        f"https://api.github.com/repos/{rerun_failed.REPOSITORY}/{path}",
        headers=headers,
    )
    try:
        with urlopen(request, timeout=15) as response:
            status, raw = response.status, response.read()
    except HTTPError as error:
        status, raw = error.code, error.read()
    try:
        return status, json.loads(raw) if raw else None
    except ValueError:
        return status, None


class GitHub:
    """The repository as one action's worker sees it: reads with the
    exporter's token, the two writes through serge on behalf of ``actor``.
    Writes answer in GitHub's own status codes, so the worker cannot tell the
    difference: a cancel is 202 (accepted) or 409 (finishing or finished), a
    dispatch 204."""

    def __init__(
        self,
        actor: str,
        serge: Serge,
        read: Callable[[str], tuple[int, object]] = _github_read,
    ) -> None:
        self.actor = actor
        self.serge = serge
        self._read = read

    def request(
        self, method: str, path: str, body: dict | None = None
    ) -> tuple[int, object]:
        if method == "GET":
            return self._read(path)
        parts = path.split("/")
        if (
            method == "POST"
            and parts[:2] == ["actions", "runs"]
            and parts[3:] == ["cancel"]
        ):
            status, payload = self.serge.call(
                "runs/cancel", {"actor": self.actor, "run_id": parts[2]}
            )
            if status != 200:
                return status, payload
            return payload.get("github_status") or 409, None
        if (
            method == "POST"
            and parts[:2] == ["actions", "workflows"]
            and parts[3:] == ["dispatches"]
        ):
            status, payload = self.serge.call(
                "workflows/dispatch",
                {
                    "actor": self.actor,
                    "workflow": parts[2],
                    "inputs": (body or {}).get("inputs") or {},
                },
            )
            return (204, None) if status == 200 else (status, payload)
        raise ValueError(f"no write path for {method} {path}")

    def get(self, path: str) -> object:
        status, payload = self.request("GET", path)
        if status != 200:
            raise ValueError(f"GitHub GET {path.split('?')[0]} returned {status}")
        return payload


def default_serge() -> Serge:
    if not (serge_url() and serge_token()):
        raise ValueError("the serge connection is not configured")
    return Serge(serge_url(), serge_token())


def default_github(actor: str) -> GitHub:
    return GitHub(actor, default_serge())


def can_write(serge: Serge, login: str) -> bool:
    """Does ``login`` have write access to the repository? The same bar as
    run-slow, which only maintainers and collaborators can trigger. serge
    answers with its App and enforces it again on every write."""
    status, payload = serge.call("permission", {"actor": login})
    return status == 200 and payload.get("can_write") is True


def new_record(
    *,
    pr: str,
    actor: str,
    idempotency_key: str,
    snap: dict,
    lanes: dict[str, dict],
) -> dict:
    action_id = secrets.token_hex(8)
    now = time.time()
    return {
        "id": action_id,
        "idempotency_key": idempotency_key,
        "pr": pr,
        "actor": actor,
        "state": "prepared",
        "created": now,
        "updated": now,
        "snapshot_version": snap["version"],
        "head_sha": snap["head_sha"],
        "lanes": {
            lane: {
                "count": data["count"],
                "selection": data["selection"],
                "source_run": data["source_run"],
                "correlation_id": f"{action_id}-{lane}",
                "workflow": rerun_failed.RERUN_WORKFLOWS[lane],
                "dispatch": "pending",
                "run": None,
            }
            for lane, data in lanes.items()
        },
        "cancel": [
            {
                "run_id": run["run_id"],
                "url": run["url"],
                "lane": lane,
                "result": "pending",
            }
            for lane, data in lanes.items()
            for run in data["active_runs"]
        ],
        "error": "",
        "events": [{"t": now, "text": f"Requested by {actor}"}],
    }


def public_view(record: dict) -> dict:
    """What GET /rerun-failed/actions/{id} shows: no selection payloads."""
    return {
        "id": record["id"],
        "pr": record["pr"],
        "actor": record["actor"],
        "state": record["state"],
        "error": record.get("error", ""),
        "created": record["created"],
        "updated": record.get("updated", record["created"]),
        "head_sha": record["head_sha"],
        "cancel": record["cancel"],
        "lanes": {
            lane: {
                "count": data["count"],
                "source_run": data["source_run"],
                "dispatch": data["dispatch"],
                "run": data["run"],
            }
            for lane, data in record["lanes"].items()
        },
        "events": record["events"][-20:],
    }


def _event(record: dict, text: str) -> None:
    record["events"].append({"t": time.time(), "text": text})


def _fail(db: Store, record: dict, error: str) -> None:
    record["state"] = "failed"
    record["error"] = error
    _event(record, error)
    db.save(record)


def _run_state(github: GitHub, run_id: str) -> dict:
    payload = github.get(f"actions/runs/{run_id.split(':', 1)[0]}")
    return payload if isinstance(payload, dict) else {}


def _cancel(db: Store, github: GitHub, record: dict, sleep: Callable) -> bool:
    for target in record["cancel"]:
        if target["result"] != "pending":
            continue
        run = _run_state(github, target["run_id"])
        if run.get("status") == "completed":
            target["result"] = "already completed"
        else:
            status, _ = github.request(
                "POST", f"actions/runs/{target['run_id'].split(':', 1)[0]}/cancel"
            )
            if status == 202:
                target["result"] = "requested"
            elif status == 409:
                # GitHub refuses to cancel a run that is finishing; re-read it.
                again = _run_state(github, target["run_id"])
                target["result"] = (
                    "already completed"
                    if again.get("status") == "completed"
                    else "refused"
                )
            else:
                target["result"] = f"refused ({status})"
        _event(record, f"Cancel run {target['run_id']}: {target['result']}")
        db.save(record)
        if target["result"].startswith("refused"):
            _fail(
                db,
                record,
                f"Could not cancel run {target['run_id']}; nothing was dispatched.",
            )
            return False
    deadline = record.setdefault(
        "cancel_deadline", time.time() + CANCEL_DEADLINE_SECONDS
    )
    while True:
        waiting = [
            t
            for t in record["cancel"]
            if t["result"] == "requested"
            and _run_state(github, t["run_id"]).get("status") != "completed"
        ]
        if not waiting:
            for target in record["cancel"]:
                if target["result"] == "requested":
                    target["result"] = "cancelled"
            db.save(record)
            return True
        if time.time() > deadline:
            ids = ", ".join(t["run_id"] for t in waiting)
            _fail(
                db,
                record,
                f"Run {ids} was still active after cancellation; nothing was dispatched.",
            )
            return False
        sleep(5)


def _find_run(github: GitHub, lane: dict) -> dict | None:
    payload = github.get(
        f"actions/workflows/{lane['workflow']}/runs?event=workflow_dispatch&per_page=30"
    )
    runs = payload.get("workflow_runs") if isinstance(payload, dict) else None
    for run in runs or []:
        if isinstance(run, dict) and lane["correlation_id"] in str(
            run.get("display_title") or ""
        ):
            return {
                "id": str(run.get("id")),
                "url": str(run.get("html_url") or ""),
                "status": str(run.get("status") or ""),
                "conclusion": str(run.get("conclusion") or ""),
            }
    return None


def _dispatch(db: Store, github: GitHub, record: dict, sleep: Callable) -> None:
    failures = []
    for name, lane in sorted(record["lanes"].items()):
        if lane["dispatch"] == "requested" and lane["run"] is None:
            # A restart between the request and finding its run: look, never resend.
            lane["run"] = _find_run(github, lane)
            lane["dispatch"] = "dispatched" if lane["run"] else "unknown"
            if lane["run"] is None:
                failures.append(f"{LANE_TITLES[name]} dispatch outcome unknown")
            db.save(record)
            continue
        if lane["dispatch"] != "pending":
            continue
        lane["dispatch"] = "requested"
        db.save(record)
        status, payload = github.request(
            "POST",
            f"actions/workflows/{lane['workflow']}/dispatches",
            {
                "ref": "main",
                "inputs": {
                    "pr_number": record["pr"],
                    "head_sha": record["head_sha"],
                    "selection": lane["selection"],
                    "correlation_id": lane["correlation_id"],
                    "source_run": (lane["source_run"] or {}).get("run_id", ""),
                },
            },
        )
        if status not in (200, 204):
            lane["dispatch"] = "failed"
            failures.append(f"{LANE_TITLES[name]} dispatch refused ({status})")
            _event(record, failures[-1])
            db.save(record)
            continue
        run_id = payload.get("workflow_run_id") if isinstance(payload, dict) else None
        deadline = time.time() + FIND_RUN_SECONDS
        while True:
            found = _find_run(github, lane)
            if found or time.time() > deadline:
                break
            sleep(5)
        if found is None and run_id:
            found = {
                "id": str(run_id),
                "url": f"https://github.com/{rerun_failed.REPOSITORY}/actions/runs/{run_id}",
                "status": "queued",
                "conclusion": "",
            }
        lane["run"] = found
        lane["dispatch"] = "dispatched" if found else "unknown"
        if found:
            _event(record, f"{LANE_TITLES[name]} run {found['id']} dispatched")
        else:
            failures.append(f"{LANE_TITLES[name]} run was not found after dispatch")
            _event(record, failures[-1])
        db.save(record)
    if failures:
        _fail(db, record, "; ".join(failures))
    else:
        record["state"] = "dispatched"
        db.save(record)


def _watch(db: Store, github: GitHub, record: dict, sleep: Callable) -> None:
    deadline = record["created"] + WATCH_DEADLINE_SECONDS
    while time.time() < deadline:
        pending = False
        for lane in record["lanes"].values():
            run = lane["run"]
            if not run or run["status"] == "completed":
                continue
            state = _run_state(github, run["id"])
            run["status"] = str(state.get("status") or run["status"])
            run["conclusion"] = str(state.get("conclusion") or "")
            pending |= run["status"] != "completed"
        db.save(record)
        if not pending:
            if record["state"] == "dispatched":
                record["state"] = "completed"
                db.save(record)
            return
        sleep(WATCH_INTERVAL_SECONDS)


def run_action(
    action_id: str,
    *,
    db: Store | None = None,
    github: GitHub | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Drive one action to a final state. Safe to call again after a crash."""
    db = db or store()
    record = db.get(action_id)
    if record is None:
        return
    github = github or default_github(record["actor"])
    try:
        if record["state"] == "prepared":
            record["state"] = "cancelling" if record["cancel"] else "dispatching"
            db.save(record)
        if record["state"] == "cancelling":
            if not _cancel(db, github, record, sleep):
                return
            record["state"] = "dispatching"
            db.save(record)
        if record["state"] == "dispatching":
            _dispatch(db, github, record, sleep)
        if record["state"] in ("dispatched", "failed"):
            _watch(db, github, record, sleep)
    except Exception as error:  # an unexpected GitHub shape: stop, never retry blind
        if record["state"] in OPEN_STATES:
            _fail(db, record, f"Stopped: {type(error).__name__}: {error}")


_running: set[str] = set()
_running_lock = threading.Lock()


def start(action_id: str) -> None:
    with _running_lock:
        if action_id in _running:
            return
        _running.add(action_id)

    def work() -> None:
        try:
            run_action(action_id)
        finally:
            with _running_lock:
                _running.discard(action_id)

    threading.Thread(target=work, name=f"rerun-{action_id}", daemon=True).start()


def resume_all() -> None:
    """Pick up actions a restart interrupted."""
    if not dispatch_enabled()[0]:
        return
    try:
        ids = store().ids_in(LIVE_STATES)
    except sqlite3.Error:
        return
    for action_id in ids:
        start(action_id)
