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
"""


def _dump(state: dict) -> str:
    return json.dumps(state, sort_keys=True)


def _load(raw: str) -> dict:
    state = json.loads(raw)
    if "prs" in state:
        state["prs"] = tuple(state["prs"])
    return state


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
        now = time.time() if now is None else now
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                if delivery_id is not None:
                    seen = self._db.execute(
                        "SELECT 1 FROM deliveries WHERE delivery_id = ?", (delivery_id,)
                    ).fetchone()
                    if seen:
                        self._db.execute("ROLLBACK")
                        return False
                    self._db.execute(
                        "INSERT INTO deliveries (delivery_id, received_at) VALUES (?, ?)",
                        (delivery_id, now),
                    )
                for update in updates:
                    if isinstance(update, JobUpdate):
                        self._merge_run(update.run_stub(), False, now)
                        self._merge_job(update, authoritative, now)
                    else:
                        self._merge_run(update, authoritative, now)
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return True

    def _merge_run(self, update: RunUpdate, authoritative: bool, now: float) -> None:
        row = self._db.execute(
            "SELECT state FROM runs WHERE repository = ? AND run_id = ? AND attempt = ?",
            update.key,
        ).fetchone()
        state = merge_run(
            _load(row[0]) if row else None, update, authoritative=authoritative
        )
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

    def _merge_job(self, update: JobUpdate, authoritative: bool, now: float) -> None:
        row = self._db.execute(
            "SELECT state FROM jobs WHERE repository = ? AND job_id = ?", update.key
        ).fetchone()
        state = merge_job(
            _load(row[0]) if row else None, update, authoritative=authoritative
        )
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
