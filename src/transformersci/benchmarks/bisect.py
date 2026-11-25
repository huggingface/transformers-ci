"""Bisect jobs for the benchmarks page: which transformers commit moved a daily series.

A Hugging Face member presses "Bisect" on a day-over-day step of a daily series; the job is queued here, and
`tperf bisect worker` on the bench host claims it, runs it and reports back. The dashboard never reaches the bench
host: the worker polls.

  POST /benchmarks/bisect                 browser session (or a CLI token with bench:submit); body
                                          {"series": <series key>, "good": <sha>, "bad": <sha>}: two adjacent daily
                                          points of the series. The job (model, harness config, bench arguments,
                                          expected change) is built from the snapshot, never from the request.
  GET  /benchmarks/bisect/next            worker token: claim the oldest queued job (204: none)
  POST /benchmarks/bisect/<id>/report     worker token: {"kind": "step", "step"} | {"kind": "done", "result"}
  GET  /benchmarks/bisect[/<id>]          the jobs, and one job's steps (?format=json for JSON)

Membership is checked at login, and sessions last 8 hours, so a submit re-checks active membership of the
organization through the GitHub App: an ex-member's live session does not spend GPU time.

Each job posts one Slack message to TCI_BENCHMARK_BISECT_SLACK_CHANNEL when queued, and its steps and result as
replies in that message's thread.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlsplit

from transformersci import github_app
from transformersci.otel import dashboard_auth as auth
from transformersci.otel import slack_api

log = logging.getLogger(__name__)

ENABLED_ENV = "TCI_BENCHMARK_BISECT_ENABLED"
TOKEN_ENV = "TCI_BENCHMARK_BISECT_TOKEN"  # the worker's service token
CHANNEL_ENV = "TCI_BENCHMARK_BISECT_SLACK_CHANNEL"
ORG = "huggingface"
SCOPE = "bench:submit"
MAX_PENDING = 3  # queued or running, all series together
LOST_SECONDS = (
    3 * 3600
)  # a running job with no report for this long is failed: the worker died
SHA = re.compile(r"[0-9a-f]{40}")
ID = re.compile(r"[0-9a-f]{12}")
ENDINGS = {
    "culprit": "found the commit",
    "range": "narrowed to a range",
    "not_reproduced": "did not reproduce: not a transformers change",
    "too_noisy": "too noisy to tell",
    "failed": "failed",
}


def enabled() -> bool:
    return os.getenv(ENABLED_ENV) == "1"


class Error(Exception):
    def __init__(self, status: int, error: str):
        super().__init__(error)
        self.status, self.error = status, error


class Store:
    def __init__(self, path: Path):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(
            str(path), check_same_thread=False, isolation_level=None
        )
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS bisect_jobs (id TEXT PRIMARY KEY, created REAL, updated REAL, series TEXT,"
            " requester TEXT, requester_id INTEGER, state TEXT, job TEXT, steps TEXT, result TEXT, slack_ts TEXT)"
        )

    def _job(self, row) -> dict:
        return {
            **json.loads(row["job"]),
            "id": row["id"],
            "created": row["created"],
            "updated": row["updated"],
            "requester": row["requester"],
            "state": row["state"],
            "steps": json.loads(row["steps"]),
            "result": json.loads(row["result"]) if row["result"] else None,
            "slack_ts": row["slack_ts"],
        }

    def get(self, job_id: str) -> dict | None:
        row = self.db.execute(
            "SELECT * FROM bisect_jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return self._job(row) if row else None

    def recent(self, limit: int = 50) -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM bisect_jobs ORDER BY created DESC LIMIT ?", (limit,)
        )
        return [self._job(r) for r in rows]

    def create(self, job: dict, login: str, user_id: int) -> dict:
        with self.lock:
            self.expire_lost()
            pending = self.db.execute(
                "SELECT series FROM bisect_jobs WHERE state IN ('queued', 'running')"
            ).fetchall()
            if any(r["series"] == job["series"] for r in pending):
                raise Error(409, "series_already_bisecting")
            if len(pending) >= MAX_PENDING:
                raise Error(429, "bisect_queue_full")
            job_id, now = uuid.uuid4().hex[:12], time.time()
            self.db.execute(
                "INSERT INTO bisect_jobs VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, '[]', NULL, NULL)",
                (job_id, now, now, job["series"], login, user_id, json.dumps(job)),
            )
            return self.get(job_id)

    def claim(self) -> dict | None:
        with self.lock:
            self.expire_lost()
            row = self.db.execute(
                "SELECT * FROM bisect_jobs WHERE state = 'queued' ORDER BY created LIMIT 1"
            ).fetchone()
            if not row:
                return None
            self.db.execute(
                "UPDATE bisect_jobs SET state = 'running', updated = ? WHERE id = ?",
                (time.time(), row["id"]),
            )
            return self.get(row["id"])

    def add_step(self, job_id: str, step: dict) -> dict:
        with self.lock:
            job = self._running(job_id)
            self.db.execute(
                "UPDATE bisect_jobs SET steps = ?, updated = ? WHERE id = ?",
                (json.dumps([*job["steps"], step]), time.time(), job_id),
            )
            return self.get(job_id)

    def finish(self, job_id: str, result: dict) -> dict:
        with self.lock:
            self._running(job_id)
            self.db.execute(
                "UPDATE bisect_jobs SET state = 'done', result = ?, updated = ? WHERE id = ?",
                (json.dumps(result), time.time(), job_id),
            )
            return self.get(job_id)

    def set_slack_ts(self, job_id: str, ts: str) -> None:
        self.db.execute(
            "UPDATE bisect_jobs SET slack_ts = ? WHERE id = ?", (ts, job_id)
        )

    def _running(self, job_id: str) -> dict:
        job = self.get(job_id)
        if not job:
            raise Error(404, "not_found")
        if job["state"] != "running":
            raise Error(409, "job_not_running")
        return job

    def expire_lost(self) -> None:
        result = json.dumps(
            {"state": "failed", "reason": "the worker stopped reporting"}
        )
        self.db.execute(
            "UPDATE bisect_jobs SET state = 'done', result = ?, updated = ? WHERE state = 'running' AND updated < ?",
            (result, time.time(), time.time() - LOST_SECONDS),
        )


_stores: dict[str, Store] = {}
_stores_lock = threading.Lock()


def store() -> Store:
    root = os.getenv("PYTEST_TRACE_EXPORTER_RUN_STORE", "")
    if not root:
        raise Error(503, "persistent_job_store_required")
    path = str(Path(root) / "bench-bisect.sqlite3")
    with _stores_lock:
        if path not in _stores:
            _stores[path] = Store(Path(path))
        return _stores[path]


def build_job(snapshot: dict, series_key: str, good: str, bad: str) -> dict:
    """The job for two adjacent daily points of a series, from the snapshot only."""
    page = snapshot.get("pages", {}).get("main.html") or {}
    series = next(
        (s for s in page.get("series", []) if s.get("key") == series_key), None
    )
    if not series:
        raise Error(404, "unknown_series")
    if not series.get("bisect"):
        raise Error(422, "series_cannot_be_bisected")
    points = series["points"]
    commits = [p["commit"] for p in points]
    if (
        good not in commits
        or bad not in commits
        or commits.index(bad) != commits.index(good) + 1
    ):
        raise Error(422, "not_adjacent_daily_points")
    before, after = points[commits.index(good)], points[commits.index(bad)]
    a, b = (p["values"].get("e2e", [None])[0] for p in (before, after))
    if not a or not b:
        raise Error(422, "no_e2e_values")
    return {
        "series": series_key,
        "model": series["model"],
        "scenario_label": series["scenario"],
        "gpu": series["gpu"],
        "scenario": series["bisect"]["config"],
        "bench_args": series["bisect"]["bench_args"],
        "good": good,
        "bad": bad,
        "good_date": before["date"][:10],
        "bad_date": after["date"][:10],
        "metric": "e2e",
        "rounds": 4,
        "min_change": 1.0,
        "expected": round((b / a - 1) * 100, 2),
    }


def snapshot() -> dict:
    from transformersci.benchmarks import page

    path = Path(os.environ.get("TCI_BENCHMARK_SNAPSHOT", "/benchmarks/snapshot.json"))
    try:
        return page._read_snapshot(str(path), path.stat().st_mtime_ns)
    except (OSError, ValueError):
        raise Error(503, "benchmark_snapshot_unavailable") from None


def still_member(user: dict) -> None:
    """Active membership of ORG now, through the GitHub App (the user's own token is not kept after login)."""
    cfg = auth.config()
    if cfg and user["user_id"] in cfg.allowed_user_ids:
        return
    token = github_app.app_token()
    if not token:
        raise Error(503, "membership_check_unavailable")
    try:
        membership = auth.github_json(
            f"https://api.github.com/orgs/{ORG}/memberships/{quote(user['login'])}",
            token=token,
        )
    except HTTPError as exc:
        if exc.code == 404:
            raise Error(403, "account_not_allowed") from None
        raise Error(503, "membership_check_unavailable") from None
    except (URLError, OSError, auth.AuthError):
        raise Error(503, "membership_check_unavailable") from None
    if not isinstance(membership, dict) or membership.get("state") != "active":
        raise Error(403, "account_not_allowed")


def requester(handler) -> dict:
    """The member asking for a bisect: a CLI token granted bench:submit, or a browser session (CSRF-checked)."""
    try:
        if handler.headers.get("Authorization"):
            user = auth.identity(handler.headers, cli=True)
            if not user:
                raise Error(401, "unauthorized")
            if SCOPE not in user["scope"].split():
                raise Error(403, "insufficient_scope")
            return user
        cfg = auth.config()
        if not cfg:
            raise Error(503, "auth_not_configured")
        return auth._browser_post(handler, cfg)
    except auth.AuthError as exc:
        raise Error(exc.status, exc.error) from None


def worker(handler) -> None:
    expected = os.getenv(TOKEN_ENV, "")
    supplied = handler.headers.get("Authorization", "")
    if not expected or not hmac.compare_digest(supplied, "Bearer " + expected):
        raise Error(401, "unauthorized")


def body(handler, limit: int = 262144) -> dict:
    try:
        length = int(handler.headers.get("Content-Length", "0"))
        if not 0 < length <= limit:
            raise ValueError
        data = json.loads(handler.rfile.read(length))
        if not isinstance(data, dict):
            raise ValueError
        return data
    except (ValueError, TypeError):
        raise Error(400, "invalid_json") from None


def job_url(job: dict) -> str:
    base = (auth.config().url if auth.config() else "").rstrip("/")
    return f"{base}/benchmarks/bisect/{job['id']}"


def describe_step(step: dict) -> str:
    where = f"`{step['base'][:10]}..{step['head'][:10]}`"
    if step.get("change") is None:
        return f"{step['role']} {where}: {step['verdict']}"
    lo, hi = step["ci"]
    return f"{step['role']} {where}: {step['change']:+.2f}% [{lo:+.2f}, {hi:+.2f}] → {step['verdict']}"


def describe_result(job: dict) -> str:
    r = job["result"] or {}
    state = r.get("state", "failed")
    text = f"*Bisect {ENDINGS.get(state, state)}*"
    if state == "culprit" and r.get("culprit"):
        c = r["culprit"]
        pr = (
            f" (<https://github.com/huggingface/transformers/pull/{c['pr']}|#{c['pr']}>)"
            if c.get("pr")
            else ""
        )
        text += (
            f": <https://github.com/huggingface/transformers/commit/{c['sha']}|`{c['sha'][:10]}`> "
            f"{c.get('subject', '')}{pr}, {c['change']:+.2f}% [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}] vs its parent"
        )
    elif state == "range" and r.get("range"):
        text += f": `{r['range'][0][:10]}`..`{r['range'][1][:10]}`"
    if r.get("reason"):
        text += f"\n{r['reason']}"
    return text


def slack(job: dict, text: str, *, thread: bool = True) -> str | None:
    """Post to the bisect channel; never fails the request (Slack is a notification, the job page is the record)."""
    channel = os.getenv(CHANNEL_ENV, "")
    if not channel or (thread and not job.get("slack_ts")):
        return None
    try:
        reply = slack_api.message(
            channel, text, thread_ts=job["slack_ts"] if thread else None
        )
        return reply.get("ts") if isinstance(reply, dict) else None
    except Exception:
        log.warning("bisect %s: Slack post failed", job["id"], exc_info=True)
        return None


def announce(db: Store, job: dict) -> None:
    text = (
        f"*Bisect: {job['model']} · {job['scenario_label']} · {job['gpu']}*, requested by {job['requester']}\n"
        f"{job['good_date']} → {job['bad_date']}: e2e {job['expected']:+.2f}% "
        f"(`{job['good'][:10]}`..`{job['bad'][:10]}`) · <{job_url(job)}|job page>"
    )
    if ts := slack(job, text, thread=False):
        db.set_slack_ts(job["id"], ts)


def public(job: dict) -> dict:
    return {k: v for k, v in job.items() if k != "slack_ts"}


def handle(handler, method: str) -> bool:
    """Serve /benchmarks/bisect*; False for any other path (or when the feature is off)."""
    parts = urlsplit(handler.path)
    path = parts.path.rstrip("/")
    if not enabled() or not (
        path == "/benchmarks/bisect" or path.startswith("/benchmarks/bisect/")
    ):
        return False
    handler._request_route = "/benchmarks/bisect"
    try:
        db = store()
        tail = path.removeprefix("/benchmarks/bisect").strip("/")
        if method == "POST" and not tail:
            user = requester(handler)
            data = body(handler, 4096)
            series, good, bad = (
                str(data.get(k, "")) for k in ("series", "good", "bad")
            )
            if not series or not SHA.fullmatch(good) or not SHA.fullmatch(bad):
                raise Error(400, "invalid_request")
            job = build_job(snapshot(), series, good, bad)
            still_member(user)
            created = db.create(job, user["login"], user["user_id"])
            announce(db, created)
            handler._reply_json(
                201, {"id": created["id"], "url": f"/benchmarks/bisect/{created['id']}"}
            )
        elif method == "GET" and tail == "next":
            worker(handler)
            job = db.claim()
            if not job:
                handler._send(204, "text/plain", b"")
            else:
                handler._reply_json(200, public(job))
                slack(job, "Claimed by the bench host worker")
        elif (
            method == "POST"
            and tail.endswith("/report")
            and ID.fullmatch(tail.removesuffix("/report"))
        ):
            worker(handler)
            job_id, data = tail.removesuffix("/report"), body(handler)
            if data.get("kind") == "step" and isinstance(data.get("step"), dict):
                job = db.add_step(job_id, data["step"])
                slack(job, describe_step(data["step"]))
            elif data.get("kind") == "done" and isinstance(data.get("result"), dict):
                job = db.finish(job_id, data["result"])
                slack(job, describe_result(job))
            else:
                raise Error(400, "invalid_report")
            handler._reply_json(200, {"ok": True})
        elif method == "GET" and (not tail or ID.fullmatch(tail)):
            as_json = parse_qs(parts.query).get("format") == ["json"]
            theme = (
                "light" if parse_qs(parts.query).get("theme") == ["light"] else "dark"
            )
            if tail:
                job = db.get(tail)
                if not job:
                    raise Error(404, "not_found")
                if as_json:
                    handler._reply_json(200, public(job))
                else:
                    handler._send(
                        200, "text/html; charset=utf-8", job_page(job, theme).encode()
                    )
            elif as_json:
                handler._reply_json(200, {"jobs": [public(j) for j in db.recent()]})
            else:
                handler._send(
                    200,
                    "text/html; charset=utf-8",
                    jobs_page(db.recent(), theme).encode(),
                )
        else:
            raise Error(404, "not_found")
    except Error as exc:
        handler._reply_json(exc.status, {"error": exc.error})
    return True


def job_page(job: dict, theme: str = "dark") -> str:
    from transformersci.benchmarks.page import commit, document, esc, link, table

    state = (
        job["state"]
        if job["state"] != "done"
        else (job["result"] or {}).get("state", "failed")
    )
    rows = [
        [esc(s["role"]), commit(s["base"]), commit(s["head"]),
         esc("" if s.get("change") is None else f"{s['change']:+.2f}%"),
         esc("" if s.get("ci") is None else f"[{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}]"), esc(s["verdict"])]
        for s in job["steps"]
    ]  # fmt: skip
    body = (
        f"<p>{link('/benchmarks/bisect', 'All bisects')} · {link('/benchmarks/main', 'Daily main')}</p>"
        f"<h1>Bisect: {esc(job['model'])}</h1>"
        f'<p class="meta">{esc(job["scenario_label"])} · {esc(job["gpu"])} · requested by {esc(job["requester"])}</p>'
        f"<p>{esc(job['good_date'])} {commit(job['good'])} → {esc(job['bad_date'])} {commit(job['bad'])}: "
        f"e2e {job['expected']:+.2f}% on the daily chart</p>"
        f'<p class="headline">{esc(ENDINGS.get(state, state))}</p>'
    )
    if job["result"]:
        r = job["result"]
        if r.get("culprit"):
            c = r["culprit"]
            body += f"<p>Culprit {commit(c['sha'])} {esc(c.get('subject', ''))}</p>"
        if r.get("range"):
            body += f"<p>Range {commit(r['range'][0])} .. {commit(r['range'][1])}</p>"
        if r.get("reason"):
            body += f"<p>{esc(r['reason'])}</p>"
    body += (
        table(["Step", "Base", "Head", "Change", "95% CI", "Verdict"], rows)
        if rows
        else ""
    )
    if job["state"] != "done":
        body += '<p class="muted">Steps appear as the bench host measures them; reload to update.</p>'
    return document("Bisect", body, theme)


def jobs_page(jobs: list[dict], theme: str = "dark") -> str:
    from transformersci.benchmarks.page import document, esc, link, table

    rows = [
        [link(f"/benchmarks/bisect/{j['id']}", j["id"]), esc(j["model"]), esc(j["scenario_label"]),
         esc(f"{j['good_date']} → {j['bad_date']}"), esc(j["requester"]),
         esc(j["state"] if j["state"] != "done" else (j["result"] or {}).get("state", "failed"))]
        for j in jobs
    ]  # fmt: skip
    body = "<h1>Bisects</h1>" + (
        table(["Job", "Model", "Scenario", "Step", "Requested by", "State"], rows)
        if rows
        else '<p class="muted">No bisect yet.</p>'
    )
    return document("Bisects", body, theme)
