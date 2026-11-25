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

PR benches share the queue and the worker: `tperf ab` of a transformers PR's head against its merge-base with main,
on the uncompiled daily scenarios (one step per scenario). Asked for from `tperf bench <PR>` (a CLI token with
bench:submit), `@askserge bench <PR>` in Slack (a linked GitHub account), or a browser session:

  POST /benchmarks/pr-bench               {"pr": <number>}; the head commit is read from GitHub, the scenarios from
                                          the snapshot
  GET  /benchmarks/pr-bench/<id>          one PR bench (?format=json for JSON)

  GET  /benchmarks/queue                  members only (a dashboard session): the running job, the queued ones and
                                          the last 10 that ended

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
PR = re.compile(r"[1-9][0-9]{0,9}")
REPO = "huggingface/transformers"
PR_ROUNDS = 6  # 3 rounds gave a 95% CI of ±17% on gemma-3-1b; 6 gave ±3.7%
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
                raise Error(
                    409,
                    "pr_already_queued"
                    if job.get("kind") == "pr"
                    else "series_already_bisecting",
                )
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

    def active(self) -> dict[str, dict]:
        """Queued or running jobs by series key (PR benches are keyed pr:<n>)."""
        rows = self.db.execute(
            "SELECT * FROM bisect_jobs WHERE state IN ('queued', 'running') ORDER BY created"
        )
        return {r["series"]: self._job(r) for r in rows}

    def queue(self, finished: int = 10) -> tuple[list[dict], list[dict], list[dict]]:
        """(running, queued oldest first, the last `finished` that ended)."""
        with self.lock:
            self.expire_lost()
        rows = lambda sql, *a: [self._job(r) for r in self.db.execute(sql, a)]
        return (
            rows("SELECT * FROM bisect_jobs WHERE state = 'running' ORDER BY created"),
            rows("SELECT * FROM bisect_jobs WHERE state = 'queued' ORDER BY created"),
            rows(
                "SELECT * FROM bisect_jobs WHERE state = 'done' ORDER BY updated DESC LIMIT ?",
                finished,
            ),
        )

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


def active() -> dict[str, dict]:
    """Active jobs by series, or {} when the feature or its store is unavailable (the page must still render)."""
    if not enabled():
        return {}
    try:
        return store().active()
    except (Error, sqlite3.Error):
        log.warning("bisect: could not read active jobs", exc_info=True)
        return {}


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


def pr_scenarios(snapshot: dict) -> list[dict]:
    """The daily scenarios a PR bench runs: every daily series the worker can rerun, uncompiled."""
    page = snapshot.get("pages", {}).get("main.html") or {}
    scenarios = []
    for s in page.get("series", []):
        spec = s.get("bisect")
        if not spec or set(spec["config"]) != {"attn_implementation"}:
            continue
        scenarios.append(
            {
                "series": s["key"],
                "model": s["model"],
                "label": s["scenario"],
                "gpu": s["gpu"],
                "scenario": spec["config"],
                "bench_args": spec["bench_args"],
            }
        )
    if not scenarios:
        raise Error(503, "no_pr_bench_scenarios")
    return scenarios


def pull_request(pr: str) -> dict:
    """The PR's head commit and title, through the GitHub App; only open PRs are benched."""
    token = github_app.app_token()
    if not token:
        raise Error(503, "github_unavailable")
    try:
        pull = auth.github_json(
            f"https://api.github.com/repos/{REPO}/pulls/{pr}", token=token
        )
    except HTTPError as exc:
        raise Error(
            404 if exc.code == 404 else 503,
            "pr_not_found" if exc.code == 404 else "github_unavailable",
        ) from None
    except (URLError, OSError, auth.AuthError):
        raise Error(503, "github_unavailable") from None
    if not isinstance(pull, dict) or pull.get("state") != "open":
        raise Error(422, "pr_not_open")
    head = str((pull.get("head") or {}).get("sha", ""))
    if not SHA.fullmatch(head):
        raise Error(503, "github_unavailable")
    return {"head": head, "title": str(pull.get("title", ""))[:200]}


def build_pr_job(snapshot: dict, pr: str, pull: dict) -> dict:
    return {
        "kind": "pr",
        "series": f"pr:{pr}",
        "pr": int(pr),
        "title": pull["title"],
        "head": pull["head"],
        "scenarios": pr_scenarios(snapshot),
        "metric": "e2e",
        "rounds": PR_ROUNDS,
    }


def submit_pr(pr: str, user: dict) -> dict:
    """Queue a PR bench for a member (browser, CLI token, or Slack's linked GitHub login)."""
    if not PR.fullmatch(str(pr)):
        raise Error(400, "invalid_request")
    job = build_pr_job(snapshot(), str(pr), pull_request(str(pr)))
    still_member(user)
    db = store()
    created = db.create(job, user["login"], user.get("user_id", 0))
    announce(db, created)
    return created


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
    if cfg and user.get("user_id") and user["user_id"] in cfg.allowed_user_ids:
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
        if exc.code == 403:
            # The App lacks the organization's "Members: read": the caller's own membership check stands. Browser
            # sessions and CLI tokens (approved from a session) both last 8 hours, and each passed it at login.
            log.warning(
                "bisect: the GitHub App cannot read %s memberships; relying on the login check",
                ORG,
            )
            return
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
    return f"{base}{job_path(job)}"


def job_path(job: dict) -> str:
    kind = "pr-bench" if job.get("kind") == "pr" else "bisect"
    return f"/benchmarks/{kind}/{job['id']}"


def describe_step(step: dict) -> str:
    if step.get("scenario"):  # a PR bench: one step per scenario
        where = f"{step['model']} · {step['scenario']}"
    else:
        where = f"`{step['base'][:10]}..{step['head'][:10]}`"
    if step.get("change") is None:
        return f"{step.get('role', '')} {where}: {step['verdict']}".strip()
    lo, hi = step["ci"]
    return f"{step.get('role', '')} {where}: {step['change']:+.2f}% [{lo:+.2f}, {hi:+.2f}] → {step['verdict']}".strip()


def describe_result(job: dict) -> str:
    r = job["result"] or {}
    state = r.get("state", "failed")
    if job.get("kind") == "pr":
        text = f"*PR bench {'done' if state == 'done' else state}*: <{job_url(job)}|results>"
        return text + (f"\n{r['reason']}" if r.get("reason") else "")
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
    if job.get("kind") == "pr":
        text = (
            f"*PR bench: <https://github.com/{REPO}/pull/{job['pr']}|#{job['pr']}> {job['title']}*, "
            f"requested by {job['requester']}\n`{job['head'][:10]}` against its merge-base with main, "
            f"{len(job['scenarios'])} daily scenarios · <{job_url(job)}|job page>"
        )
        if ts := slack(job, text, thread=False):
            db.set_slack_ts(job["id"], ts)
        return
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
    if not enabled():
        return False
    if path == "/benchmarks/queue" and method == "GET":
        handler._request_route = "/benchmarks/queue"
        theme = "light" if parse_qs(parts.query).get("theme") == ["light"] else "dark"
        if not auth.identity(handler.headers):
            handler._send(200, "text/html; charset=utf-8", queue_signin(theme).encode())
            return True
        try:
            running, queued, finished = store().queue()
        except Error as exc:
            handler._reply_json(exc.status, {"error": exc.error})
            return True
        handler._send(
            200,
            "text/html; charset=utf-8",
            queue_page(running, queued, finished, theme).encode(),
        )
        return True
    if path == "/benchmarks/pr-bench" or path.startswith("/benchmarks/pr-bench/"):
        handle_pr(
            handler,
            method,
            path.removeprefix("/benchmarks/pr-bench").strip("/"),
            parts.query,
        )
        return True
    if not (path == "/benchmarks/bisect" or path.startswith("/benchmarks/bisect/")):
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
                    page_html = (
                        pr_page(job, theme)
                        if job.get("kind") == "pr"
                        else job_page(job, theme)
                    )
                    handler._send(200, "text/html; charset=utf-8", page_html.encode())
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


def handle_pr(handler, method: str, tail: str, query: str) -> None:
    handler._request_route = "/benchmarks/pr-bench"
    try:
        if method == "POST" and not tail:
            user = requester(handler)
            created = submit_pr(str(body(handler, 1024).get("pr", "")), user)
            handler._reply_json(201, {"id": created["id"], "url": job_path(created)})
        elif method == "GET" and ID.fullmatch(tail):
            job = store().get(tail)
            if not job or job.get("kind") != "pr":
                raise Error(404, "not_found")
            if parse_qs(query).get("format") == ["json"]:
                handler._reply_json(200, public(job))
            else:
                theme = "light" if parse_qs(query).get("theme") == ["light"] else "dark"
                handler._send(
                    200, "text/html; charset=utf-8", pr_page(job, theme).encode()
                )
        else:
            raise Error(404, "not_found")
    except Error as exc:
        handler._reply_json(exc.status, {"error": exc.error})


GOOD_VERDICTS = {"faster", "good", "not_reproduced", "no significant change"}
BAD_VERDICTS = {"slower", "bad", "reproduced", "confirmed"}
LIVE_CSS = """
.job-progress{margin:14px 0 18px;padding:12px 14px;border:1px solid var(--line);border-radius:6px;background:var(--panel)}
.job-progress .row{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
.job-progress .state{font-size:15px;font-weight:600;display:flex;align-items:center;gap:8px}
.job-progress .clock{color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
.job-bar{position:relative;height:8px;margin-top:10px;border-radius:99px;background:var(--line);overflow:hidden}
.job-bar>span{position:absolute;inset:0 auto 0 0;border-radius:99px;background:linear-gradient(90deg,var(--accent),#a78bfa,var(--accent));
  background-size:200% 100%;transition:width .8s cubic-bezier(.2,.8,.2,1)}
.job-progress[data-live] .job-bar>span{animation:job-flow 2.2s linear infinite}
.job-bar.indeterminate>span{width:35%;animation:job-slide 1.6s ease-in-out infinite,job-flow 2.2s linear infinite}
.job-progress[data-state=done] .job-bar>span{background:var(--good)}
.job-progress[data-state=failed] .job-bar>span{background:var(--bad)}
.pulse{width:10px;height:10px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 0 var(--accent);animation:job-pulse 1.4s infinite}
.job-progress[data-state=done] .pulse{background:var(--good);animation:none}
.job-progress[data-state=failed] .pulse{background:var(--bad);animation:none}
.job-progress[data-state=queued] .pulse{background:var(--muted)}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:11px;border:1px solid var(--line)}
.pill.good{color:var(--good);border-color:currentColor}.pill.bad{color:var(--bad);border-color:currentColor}
.measuring{display:inline-flex;align-items:center;gap:6px;color:var(--accent);font-size:12px}
.measuring i{width:5px;height:5px;border-radius:50%;background:currentColor;animation:job-dot 1.2s infinite}
.measuring i:nth-child(2){animation-delay:.2s}.measuring i:nth-child(3){animation-delay:.4s}
tr.fresh{animation:job-in 1.2s ease-out}
@keyframes job-flow{to{background-position:-200% 0}}
@keyframes job-slide{0%{left:-35%}100%{left:100%}}
@keyframes job-pulse{0%{box-shadow:0 0 0 0 color-mix(in srgb,var(--accent) 60%,transparent)}100%{box-shadow:0 0 0 10px transparent}}
@keyframes job-dot{0%,80%,100%{opacity:.25;transform:scale(.8)}40%{opacity:1;transform:scale(1.2)}}
@keyframes job-in{from{background:color-mix(in srgb,var(--accent) 25%,transparent)}to{background:transparent}}
@media (prefers-reduced-motion:reduce){.job-progress *,tr.fresh{animation:none!important;transition:none!important}}
"""
# Polls the page itself while the job runs and swaps #job in place (no reload): new rows flash, timers tick.
LIVE_SCRIPT = """
(()=>{const fmt=s=>{s=Math.max(0,Math.round(s));const h=Math.floor(s/3600),m=Math.floor(s%3600/60),x=s%60;
return (h?h+'h ':'')+(h||m?m+'m ':'')+x+'s'};
const tick=()=>document.querySelectorAll('[data-since]').forEach(e=>{e.textContent=fmt(Date.now()/1000-+e.dataset.since)});
tick();setInterval(tick,1000);
const poll=async()=>{const old=document.getElementById('job');if(!old||old.dataset.done)return;
try{const r=await fetch(location.href,{cache:'no-store'});if(!r.ok)throw 0;
const doc=new DOMParser().parseFromString(await r.text(),'text/html');const nu=doc.getElementById('job');
if(nu&&nu.dataset.version!==old.dataset.version){const n=old.querySelectorAll('tbody tr').length;
old.replaceWith(nu);nu.querySelectorAll('tbody tr').forEach((tr,i)=>{if(i>=n-1&&!tr.querySelector('.measuring'))tr.classList.add('fresh')});
tick();if(nu.dataset.done){document.title='\u2713 '+document.title;return}}}catch(e){}
setTimeout(poll,5000)};setTimeout(poll,5000)})();
"""


def pill(verdict: str) -> str:
    from transformersci.benchmarks.page import esc

    tone = (
        "good" if verdict in GOOD_VERDICTS else "bad" if verdict in BAD_VERDICTS else ""
    )
    return f'<span class="pill {tone}">{esc(verdict)}</span>'


def running_cell() -> str:
    return '<span class="measuring"><i></i><i></i><i></i> measuring on the bench host</span>'


def progress(job: dict, done: int, total: int | None, label: str) -> str:
    """The job's state, a progress bar (indeterminate when the step count is not known), and live timers."""
    from transformersci.benchmarks.page import esc

    r = job.get("result") or {}
    tone = "queued" if job["state"] == "queued" else "running" if job["state"] == "running" else (
        "failed" if r.get("state") == "failed" else "done")  # fmt: skip
    if job["state"] == "done":
        width, bar = 100, "job-bar"
    elif total:
        width, bar = max(4, round(100 * done / total)), "job-bar"
    else:
        width, bar = 0, "job-bar indeterminate"
    if job["state"] == "queued":
        width, bar = 0, "job-bar"
    count = (
        f"{done}/{total} scenarios"
        if total
        else f"{done} step{'s' if done != 1 else ''}"
    )
    if job["state"] == "done":
        clock = f"{count} · took {esc(duration(job['updated'] - job['created']))}"
    else:
        what = "queued" if job["state"] == "queued" else "running"
        clock = (f'{count} · {what} for <span data-since="{job["created"]:.0f}"></span>'
                 f' · last update <span data-since="{job["updated"]:.0f}"></span> ago')  # fmt: skip
    style = f' style="width:{width}%"' if "indeterminate" not in bar else ""
    live = "" if job["state"] == "done" else " data-live"
    return (
        f'<div class="job-progress" data-state="{tone}"{live}><div class="row">'
        f'<span class="state"><span class="pulse"></span>{esc(label)}</span><span class="clock">{clock}</span></div>'
        f'<div class="{bar}"><span{style}></span></div></div>'
    )


def duration(seconds: float) -> str:
    s = max(0, round(seconds))
    h, m = divmod(s // 60, 60)
    return f"{h}h {m}m" if h else f"{m}m {s % 60}s" if m else f"{s}s"


def live(job: dict, body: str) -> str:
    """Wrap a job page's body so the page updates itself until the job is done."""
    version = f"{job['state']}-{len(job['steps'])}-{job['updated']:.3f}"
    done = ' data-done="1"' if job["state"] == "done" else ""
    return (
        f'<style>{LIVE_CSS}</style><div id="job" data-version="{version}"{done}>{body}</div>'
        f"<script>{LIVE_SCRIPT}</script>"
    )


def pr_page(job: dict, theme: str = "dark") -> str:
    from transformersci.benchmarks.page import commit, document, esc, link, table

    done = {(s.get("model"), s.get("scenario")): s for s in job["steps"]}
    pending = [sc for sc in job["scenarios"] if (sc["model"], sc["label"]) not in done]
    current = pending[0] if pending and job["state"] == "running" else None
    rows = []
    for sc in job["scenarios"]:
        s = done.get((sc["model"], sc["label"]))
        rows.append(
            [esc(sc["model"]), esc(sc["label"]),
             esc("" if not s or s.get("change") is None else f"{s['change']:+.2f}%"),
             esc("" if not s or s.get("ci") is None else f"[{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}]"),
             pill(s["verdict"]) if s else ("" if job["state"] == "done" else (running_cell() if sc is current else pill("pending")))]
        )  # fmt: skip
    r = job["result"] or {}
    state = job["state"] if job["state"] != "done" else r.get("state", "failed")
    base = next((s["base"] for s in job["steps"] if s.get("base")), None)
    pr_link = link(
        f"https://github.com/{REPO}/pull/{job['pr']}", job["title"] or f"#{job['pr']}"
    )
    body = (
        f"<p>{link('/benchmarks/bisect', 'All jobs')} · {link('/benchmarks/main', 'Daily main')}</p>"
        f"<h1>PR bench: #{job['pr']}</h1>"
        f'<p class="meta">{pr_link}'
        f" · requested by {esc(job['requester'])}</p>"
        f"<p>Head {commit(job['head'])}"
        + (
            f" against merge-base {commit(base)}"
            if base
            else " against its merge-base with main"
        )
        + f", {job['metric']}, {job['rounds']} rounds per side, change = head vs base (lower is faster)</p>"
        + progress(job, len(done), len(job["scenarios"]), state)
    )
    if r.get("reason"):
        body += f"<p>{esc(r['reason'])}</p>"
    body += table(["Model", "Scenario", "Change", "95% CI", "Verdict"], rows)
    return document(f"PR bench #{job['pr']}", live(job, body), theme)


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
         esc("" if s.get("ci") is None else f"[{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}]"), pill(s["verdict"])]
        for s in job["steps"]
    ]  # fmt: skip
    if job["state"] == "running":
        rows.append(
            [esc("reproduce" if not rows else "step"), "", "", "", "", running_cell()]
        )
    body = (
        f"<p>{link('/benchmarks/bisect', 'All bisects')} · {link('/benchmarks/main', 'Daily main')}</p>"
        f"<h1>Bisect: {esc(job['model'])}</h1>"
        f'<p class="meta">{esc(job["scenario_label"])} · {esc(job["gpu"])} · requested by {esc(job["requester"])}</p>'
        f"<p>{esc(job['good_date'])} {commit(job['good'])} → {esc(job['bad_date'])} {commit(job['bad'])}: "
        f"e2e {job['expected']:+.2f}% on the daily chart</p>"
        + progress(job, len(job["steps"]), None, ENDINGS.get(state, state))
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
    return document("Bisect", live(job, body), theme)


def describe_job(job: dict) -> tuple[str, str]:
    """(what, detail) of a job for the tables."""
    if job.get("kind") == "pr":
        return (
            f"PR #{job['pr']} {job.get('title', '')}",
            f"{len(job['scenarios'])} scenarios",
        )
    return (
        f"Bisect {job['model']}",
        f"{job['scenario_label']} · {job['good_date']} → {job['bad_date']}",
    )


def job_state(job: dict) -> str:
    return (
        job["state"]
        if job["state"] != "done"
        else (job["result"] or {}).get("state", "failed")
    )


def queue_signin(theme: str = "dark") -> str:
    from transformersci.benchmarks.page import document, tabs

    body = (
        tabs("queue", theme) + "<h1>Bench queue</h1>"
        "<p>The bench queue is for Hugging Face organization members. "
        '<a href="/auth/github?next=/benchmarks/queue">Sign in with GitHub</a> to see it.</p>'
    )
    return document("Bench queue", body, theme)


def queue_page(
    running: list[dict], queued: list[dict], finished: list[dict], theme: str = "dark"
) -> str:
    from transformersci.benchmarks.page import document, esc, link, table, tabs

    def rows(jobs: list[dict], when: str) -> list[list[str]]:
        out = []
        for j in jobs:
            what, detail = describe_job(j)
            stamp = j["updated"] if when == "ended" else j["created"]
            out.append(
                [link(job_path(j), j["id"]), esc(what), esc(detail), esc(j["requester"]),
                 pill(job_state(j)) if when == "ended" else esc(f"{len(j['steps'])} steps"),
                 f'<span data-since="{stamp:.0f}"></span> ago']
            )  # fmt: skip
        return out

    body = tabs("queue", theme) + "<h1>Bench queue</h1>"
    body += (
        '<p class="meta">Bisects from the Daily main charts and PR benches (<code>tperf bench &lt;PR&gt;</code>, '
        "<code>/askserge bench &lt;PR&gt;</code>) run one at a time on the bench host. Up to "
        f"{MAX_PENDING} jobs wait or run at once.</p>"
    )
    body += "<h2>Running</h2>"
    if running:
        for j in running:
            what, detail = describe_job(j)
            total = len(j["scenarios"]) if j.get("kind") == "pr" else None
            body += (
                f"<p>{link(job_path(j), what)} · {esc(detail)} · requested by {esc(j['requester'])}</p>"
                + progress(j, len(j["steps"]), total, "running")
            )
    else:
        body += '<p class="muted">Nothing is running.</p>'
    body += "<h2>Queued</h2>"
    body += (
        table(
            ["Job", "What", "Detail", "Requested by", "Progress", "Queued"],
            rows(queued, "queued"),
        )
        if queued
        else '<p class="muted">Nothing is waiting.</p>'
    )
    body += "<h2>Last 10 finished</h2>"
    body += (
        table(
            ["Job", "What", "Detail", "Requested by", "Result", "Ended"],
            rows(finished, "ended"),
        )
        if finished
        else '<p class="muted">No job has finished yet.</p>'
    )
    version = "-".join(
        f"{j['id']}:{j['state']}:{len(j['steps'])}"
        for j in running + queued + finished[:1]
    )
    return document(
        "Bench queue",
        f'<style>{LIVE_CSS}</style><div id="job" data-version="{esc(version)}">{body}</div><script>{LIVE_SCRIPT}</script>',
        theme,
    )


def jobs_page(jobs: list[dict], theme: str = "dark") -> str:
    from transformersci.benchmarks.page import document, esc, link, table

    def what(j: dict) -> list[str]:
        if j.get("kind") == "pr":
            return [
                esc(f"PR #{j['pr']}"),
                esc(j.get("title", "")),
                esc(f"{len(j['scenarios'])} scenarios"),
            ]
        return [
            esc(j["model"]),
            esc(j["scenario_label"]),
            esc(f"{j['good_date']} → {j['bad_date']}"),
        ]

    rows = [
        [link(job_path(j), j["id"]), *what(j), esc(j["requester"]),
         esc(j["state"] if j["state"] != "done" else (j["result"] or {}).get("state", "failed"))]
        for j in jobs
    ]  # fmt: skip
    body = "<h1>Bisects and PR benches</h1>" + (
        table(["Job", "What", "Scenario", "Range", "Requested by", "State"], rows)
        if rows
        else '<p class="muted">No bisect yet.</p>'
    )
    return document("Bisects", body, theme)
