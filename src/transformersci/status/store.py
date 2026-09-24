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
"""Durable run/job state in SQLite.

One writer: the service runs as a single replica (the Deployment must use the
``Recreate`` strategy so two pods never share the file). ``apply`` records the
delivery id and the merged state in ONE transaction and returns only after the
commit, so a delivery acknowledged to GitHub survives a crash, and a delivery
seen twice — GitHub redelivers, and reconciliation re-reads — is a no-op.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterable
from pathlib import Path

from .reducer import COMPLETED, JobUpdate, RunUpdate, merge_job, merge_run

# Events whose run should name a PR; GitHub omits it for a PR from a fork.
PR_EVENTS = frozenset({"pull_request", "pull_request_target"})

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    repository TEXT NOT NULL,
    run_id INTEGER NOT NULL,
    attempt INTEGER NOT NULL,
    state TEXT NOT NULL,
    status TEXT NOT NULL,
    finished_at REAL,
    touched_at REAL NOT NULL,
    PRIMARY KEY (repository, run_id, attempt)
);
CREATE INDEX IF NOT EXISTS runs_by_status ON runs (status, finished_at);
CREATE TABLE IF NOT EXISTS jobs (
    repository TEXT NOT NULL,
    job_id INTEGER NOT NULL,
    run_id INTEGER NOT NULL,
    attempt INTEGER NOT NULL,
    state TEXT NOT NULL,
    status TEXT NOT NULL,
    finished_at REAL,
    touched_at REAL NOT NULL,
    PRIMARY KEY (repository, job_id)
);
CREATE INDEX IF NOT EXISTS jobs_by_run ON jobs (repository, run_id, attempt);
CREATE INDEX IF NOT EXISTS jobs_by_status ON jobs (status, finished_at);
CREATE TABLE IF NOT EXISTS deliveries (
    delivery_id TEXT PRIMARY KEY,
    received_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS deliveries_by_age ON deliveries (received_at);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# What a repair is: a change to any of these, or a record that did not exist.
_PROGRESS_FIELDS = ("status", "conclusion", "needs_lookup", "prs")


def _dump(state: dict) -> str:
    return json.dumps(state, sort_keys=True)


def _load(raw: str) -> dict:
    state = json.loads(raw)
    if "prs" in state:
        state["prs"] = tuple(state["prs"])
    return state


def _moved(before: dict | None, after: dict) -> bool:
    if before is None:
        return True
    return any(before.get(field) != after.get(field) for field in _PROGRESS_FIELDS)


def _pr_unknown(state: dict) -> bool:
    return not state.get("prs") and state.get("event") in PR_EVENTS


class Store:
    """Run/job state keyed by GitHub identity, safe to share across threads."""

    def __init__(self, path: str | Path) -> None:
        self._lock = threading.Lock()
        self._db = sqlite3.connect(
            str(path), check_same_thread=False, isolation_level=None
        )
        self._db.execute("PRAGMA journal_mode=WAL")
        # FULL: a commit is on disk before apply() returns and the delivery is
        # acknowledged.
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # -- writes ---------------------------------------------------------------

    def apply(
        self,
        updates: Iterable[RunUpdate | JobUpdate],
        *,
        delivery_id: str | None = None,
        authoritative: bool = False,
        now: float | None = None,
    ) -> bool:
        """Merge ``updates`` durably. Returns False, changing nothing, when
        ``delivery_id`` was already applied. A job update also merges the stub
        it implies for its run, so a job seen before its run still publishes."""
        applied, _runs, _jobs = self._apply(updates, delivery_id, authoritative, now)
        return applied

    def reconcile(
        self, updates: Iterable[RunUpdate | JobUpdate], *, now: float | None = None
    ) -> tuple[int, int]:
        """Merge updates read from the GitHub API (authoritative: they settle a
        conclusion two deliveries disagreed on). Returns how many runs and jobs
        it created or moved — i.e. what the webhooks had missed."""
        _applied, runs, jobs = self._apply(updates, None, True, now)
        return runs, jobs

    def _apply(
        self,
        updates: Iterable[RunUpdate | JobUpdate],
        delivery_id: str | None,
        authoritative: bool,
        now: float | None,
    ) -> tuple[bool, int, int]:
        now = time.time() if now is None else now
        changed_runs = changed_jobs = 0
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                if delivery_id is not None:
                    seen = self._db.execute(
                        "SELECT 1 FROM deliveries WHERE delivery_id = ?", (delivery_id,)
                    ).fetchone()
                    if seen:
                        self._db.execute("ROLLBACK")
                        return False, 0, 0
                    self._db.execute(
                        "INSERT INTO deliveries (delivery_id, received_at) VALUES (?, ?)",
                        (delivery_id, now),
                    )
                for update in updates:
                    if isinstance(update, JobUpdate):
                        changed_runs += self._merge_run(update.run_stub(), False, now)
                        changed_jobs += self._merge_job(update, authoritative, now)
                    else:
                        changed_runs += self._merge_run(update, authoritative, now)
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return True, changed_runs, changed_jobs

    def _merge_run(self, update: RunUpdate, authoritative: bool, now: float) -> bool:
        row = self._db.execute(
            "SELECT state FROM runs WHERE repository = ? AND run_id = ? AND attempt = ?",
            update.key,
        ).fetchone()
        before = _load(row[0]) if row else None
        state = merge_run(before, update, authoritative=authoritative)
        finished = (
            (state.get("updated_at") or now) if state["status"] == COMPLETED else None
        )
        self._db.execute(
            "INSERT INTO runs (repository, run_id, attempt, state, status, finished_at,"
            " touched_at) VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (repository, run_id, attempt) DO UPDATE SET"
            " state = excluded.state, status = excluded.status,"
            " finished_at = excluded.finished_at, touched_at = excluded.touched_at",
            (*update.key, _dump(state), state["status"], finished, now),
        )
        return _moved(before, state)

    def _merge_job(self, update: JobUpdate, authoritative: bool, now: float) -> bool:
        row = self._db.execute(
            "SELECT state FROM jobs WHERE repository = ? AND job_id = ?", update.key
        ).fetchone()
        before = _load(row[0]) if row else None
        state = merge_job(before, update, authoritative=authoritative)
        finished = (
            (state.get("completed_at") or now) if state["status"] == COMPLETED else None
        )
        self._db.execute(
            "INSERT INTO jobs (repository, job_id, run_id, attempt, state, status,"
            " finished_at, touched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (repository, job_id) DO UPDATE SET"
            " state = excluded.state, status = excluded.status,"
            " finished_at = excluded.finished_at, touched_at = excluded.touched_at",
            (
                update.repository,
                update.job_id,
                update.run_id,
                update.attempt,
                _dump(state),
                state["status"],
                finished,
                now,
            ),
        )
        return _moved(before, state)

    def mark_run(self, key: tuple[str, int, int], **fields: object) -> None:
        """Record reconciler bookkeeping (``jobs_synced``, ``pr_lookup_done``)
        on a run's state. Merges keep unknown fields, so these persist."""
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    "SELECT state FROM runs WHERE repository = ? AND run_id = ?"
                    " AND attempt = ?",
                    key,
                ).fetchone()
                if row:
                    state = _load(row[0])
                    state.update(fields)
                    self._db.execute(
                        "UPDATE runs SET state = ? WHERE repository = ? AND run_id = ?"
                        " AND attempt = ?",
                        (_dump(state), *key),
                    )
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?)"
                " ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def runs_needing_reconcile(self, *, touched_since: float) -> list[dict]:
        """Runs the reconciler must read back from GitHub: not completed, or
        completed but with jobs not yet listed since, with an active or disputed
        job, a disputed conclusion, or a PR GitHub did not name. Bounded to runs
        touched recently, so a run GitHub itself lost cannot be polled forever."""
        with self._lock:
            rows = self._db.execute(
                "SELECT r.state,"
                " EXISTS (SELECT 1 FROM jobs j WHERE j.repository = r.repository"
                "   AND j.run_id = r.run_id AND j.attempt = r.attempt"
                "   AND (j.status != ? OR json_extract(j.state, '$.needs_lookup')))"
                " FROM runs r WHERE r.touched_at >= ? ORDER BY r.touched_at",
                (COMPLETED, touched_since),
            ).fetchall()
        due = []
        for raw, open_jobs in rows:
            state = _load(raw)
            if (
                state["status"] != COMPLETED
                or state.get("needs_lookup")
                or not state.get("jobs_synced")
                or open_jobs
                or (_pr_unknown(state) and not state.get("pr_lookup_done"))
            ):
                due.append(state)
        return due

    def prune(self, *, now: float | None = None, retention_seconds: float) -> int:
        """Forget completed runs/jobs and delivery ids older than the retention.
        Active records are kept whatever their age: only reconciliation may
        decide they are over."""
        now = time.time() if now is None else now
        cutoff = now - retention_seconds
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                removed = 0
                for table in ("runs", "jobs"):
                    removed += self._db.execute(
                        f"DELETE FROM {table} WHERE status = ? AND finished_at < ?",
                        (COMPLETED, cutoff),
                    ).rowcount
                self._db.execute(
                    "DELETE FROM deliveries WHERE received_at < ?", (cutoff,)
                )
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return removed

    # -- reads ----------------------------------------------------------------

    def _select(self, table: str, since: float) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                f"SELECT state FROM {table} WHERE status != ? OR finished_at >= ?"
                " ORDER BY rowid",
                (COMPLETED, since),
            ).fetchall()
        return [_load(row[0]) for row in rows]

    def runs(self, *, completed_since: float = 0.0) -> list[dict]:
        """Every active run, and completed runs finished at/after the cutoff."""
        return self._select("runs", completed_since)

    def jobs(self, *, completed_since: float = 0.0) -> list[dict]:
        """Every active job, and completed jobs finished at/after the cutoff."""
        return self._select("jobs", completed_since)

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._db.execute(
                "SELECT 'runs', status = ?, COUNT(*) FROM runs GROUP BY 2"
                " UNION ALL SELECT 'jobs', status = ?, COUNT(*) FROM jobs GROUP BY 2",
                (COMPLETED, COMPLETED),
            ).fetchall()
            lookups = self._db.execute(
                "SELECT (SELECT COUNT(*) FROM runs WHERE json_extract(state, '$.needs_lookup'))"
                " + (SELECT COUNT(*) FROM jobs WHERE json_extract(state, '$.needs_lookup'))"
            ).fetchone()[0]
        counts = {
            "runs_active": 0,
            "runs_completed": 0,
            "jobs_active": 0,
            "jobs_completed": 0,
        }
        for table, completed, count in rows:
            counts[f"{table}_{'completed' if completed else 'active'}"] = count
        counts["needs_lookup"] = int(lookups or 0)
        return counts
