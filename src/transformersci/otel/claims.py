"""Who picked up a failing test, so two people never open issues for the same one.

New issue *reserves* the test for whoever starts the draft; everyone else is
told who is drafting instead of getting a second draft. Publishing turns the
reservation into a claim with the issue's number; closing the draft releases
it, and an abandoned one lapses after ``RESERVE_SECONDS``. A published claim's
issue is re-read from GitHub every ``CHECK_SECONDS``. A closed issue still
holds the test: the page says it was closed (a close is not proof of a fix) and
offers to reopen it instead of a second issue. Reopening makes the person who
reopened it the claimant.

One row per test node id, in ``$RUN_STORE/claims.sqlite3``. The exporter runs
as a single replica, so a process lock around each SQLite transaction makes a
check-and-reserve atomic for every viewer. Without a run store (local runs,
tests) the claims live in memory.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from datetime import datetime

RESERVE_SECONDS = 15 * 60
CHECK_SECONDS = 5 * 60

_store: "Store | None" = None
_store_lock = threading.Lock()


def database_path() -> str:
    explicit = os.getenv("PYTEST_TRACE_EXPORTER_CLAIMS_DB", "").strip()
    if explicit:
        return explicit
    store = os.getenv("PYTEST_TRACE_EXPORTER_RUN_STORE", "").rstrip("/")
    return f"{store}/claims.sqlite3" if store else ":memory:"


class Store:
    def __init__(self, path: str) -> None:
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        if path != ":memory:":
            self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS claims (
                nodeid TEXT PRIMARY KEY,
                login TEXT NOT NULL,
                state TEXT NOT NULL,  -- 'drafting' or 'claimed'
                created REAL NOT NULL,
                expires REAL,         -- drafting only
                number INTEGER,       -- claimed only
                url TEXT
            )
            """
        )
        # Added after the first deploy: an existing database gains them in place.
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(claims)")}
        for name, kind in (
            ("issue_state", "TEXT"),  # 'open' or 'closed', from GitHub
            ("closed_at", "REAL"),
            ("checked_at", "REAL"),
        ):
            if name not in columns:
                self._db.execute(f"ALTER TABLE claims ADD COLUMN {name} {kind}")

    def _row(self, nodeid: str) -> dict | None:
        row = self._db.execute(
            "SELECT login, state, created, expires, number, url, issue_state"
            " FROM claims WHERE nodeid = ?",
            (nodeid,),
        ).fetchone()
        if not row:
            return None
        keys = ("login", "state", "created", "expires", "number", "url", "issue_state")
        return dict(zip(keys, row))

    def reserve(self, nodeid: str, login: str, now: float | None = None) -> dict:
        """Reserve ``nodeid`` for ``login``, or say who already has it.

        ``{"status": "ok"}``, ``{"status": "claimed", login, number, url}``,
        ``{"status": "closed", login, number, url}`` (reopen that issue instead)
        or ``{"status": "drafting", login, since}``. The same person may
        reserve again (a second tab, or publishing after a while): it refreshes.
        """
        now = time.time() if now is None else now
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                current = self._row(nodeid)
                if current and current["state"] == "claimed":
                    return {
                        "status": "closed"
                        if current["issue_state"] == "closed"
                        else "claimed",
                        "login": current["login"],
                        "number": current["number"],
                        "url": current["url"],
                    }
                if (
                    current
                    and current["login"] != login
                    and (current["expires"] or 0) > now
                ):
                    return {
                        "status": "drafting",
                        "login": current["login"],
                        "since": current["created"],
                    }
                created = (
                    current["created"] if current and current["login"] == login else now
                )
                self._db.execute(
                    "INSERT OR REPLACE INTO claims (nodeid, login, state, created, expires)"
                    " VALUES (?, ?, 'drafting', ?, ?)",
                    (nodeid, login, created, now + RESERVE_SECONDS),
                )
                return {"status": "ok"}
            finally:
                self._db.execute("COMMIT")

    def confirm(
        self, nodeid: str, login: str, number: int, url: str, now: float | None = None
    ) -> None:
        now = time.time() if now is None else now
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO claims"
                " (nodeid, login, state, created, number, url, issue_state, checked_at)"
                " VALUES (?, ?, 'claimed', ?, ?, ?, 'open', ?)",
                (nodeid, login, now, number, url, now),
            )

    def reopen(self, nodeid: str, login: str, now: float | None = None) -> dict:
        """Take a closed claim for ``login`` before its issue is reopened.

        ``{"status": "ok", number, url, previous}`` (``previous`` restores the
        claim with ``undo_reopen`` if GitHub refuses), or the claim's status
        when it is not closed: one of two people reopening at once wins.
        """
        now = time.time() if now is None else now
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                current = self._row(nodeid)
                if not current or current["state"] != "claimed":
                    return {"status": "invalid"}
                if current["issue_state"] != "closed":
                    return {
                        "status": "claimed",
                        "login": current["login"],
                        "number": current["number"],
                        "url": current["url"],
                    }
                previous = self._db.execute(
                    "SELECT login, created, closed_at, checked_at FROM claims"
                    " WHERE nodeid = ?",
                    (nodeid,),
                ).fetchone()
                self._db.execute(
                    "UPDATE claims SET login = ?, created = ?, issue_state = 'open',"
                    " closed_at = NULL, checked_at = ? WHERE nodeid = ?",
                    (login, now, now, nodeid),
                )
                return {
                    "status": "ok",
                    "number": current["number"],
                    "url": current["url"],
                    "previous": previous,
                }
            finally:
                self._db.execute("COMMIT")

    def undo_reopen(self, nodeid: str, number: int, previous: tuple) -> None:
        login, created, closed_at, checked_at = previous
        with self._lock:
            self._db.execute(
                "UPDATE claims SET login = ?, created = ?, issue_state = 'closed',"
                " closed_at = ?, checked_at = ?"
                " WHERE nodeid = ? AND number = ? AND state = 'claimed'",
                (login, created, closed_at, checked_at, nodeid, number),
            )

    def release(self, nodeid: str, login: str) -> None:
        """Drop ``login``'s own reservation; a published claim stays."""
        with self._lock:
            self._db.execute(
                "DELETE FROM claims WHERE nodeid = ? AND login = ? AND state = 'drafting'",
                (nodeid, login),
            )

    def snapshot(self, now: float | None = None) -> dict[str, dict]:
        """Every claim (open or closed issue) and live reservation, by node id."""
        now = time.time() if now is None else now
        with self._lock:
            rows = self._db.execute(
                "SELECT nodeid, login, state, created, expires, number, url,"
                " issue_state, closed_at FROM claims"
                " WHERE state = 'claimed' OR expires > ?",
                (now,),
            ).fetchall()
        keys = (
            "login",
            "state",
            "created",
            "expires",
            "number",
            "url",
            "issue_state",
            "closed_at",
        )
        return {row[0]: dict(zip(keys, row[1:])) for row in rows}

    def due(self, now: float | None = None) -> list[tuple[str, int]]:
        """Published claims whose issue was not read in the last CHECK_SECONDS."""
        now = time.time() if now is None else now
        with self._lock:
            return self._db.execute(
                "SELECT nodeid, number FROM claims WHERE state = 'claimed'"
                " AND number IS NOT NULL"
                " AND (checked_at IS NULL OR checked_at < ?)",
                (now - CHECK_SECONDS,),
            ).fetchall()

    def set_issue_state(
        self,
        nodeid: str,
        number: int,
        state: str,
        closed_at: float | None,
        now: float | None = None,
    ) -> None:
        now = time.time() if now is None else now
        with self._lock:
            # Keyed on the number too: a newer claim on the test is left alone.
            self._db.execute(
                "UPDATE claims SET issue_state = ?, closed_at = ?, checked_at = ?"
                " WHERE nodeid = ? AND number = ? AND state = 'claimed'",
                (state, closed_at, now, nodeid, number),
            )

    def refresh(self, read_issue, now: float | None = None) -> int:
        """Re-read due issues with ``read_issue(number) -> GitHub issue dict``.

        Returns how many were read; one failing read is skipped, not fatal.
        """
        read = 0
        for nodeid, number in self.due(now):
            try:
                issue = read_issue(number)
            except Exception:
                continue
            if not isinstance(issue, dict) or issue.get("state") not in (
                "open",
                "closed",
            ):
                continue
            closed_at = None
            if issue["state"] == "closed" and issue.get("closed_at"):
                try:
                    closed_at = datetime.fromisoformat(
                        str(issue["closed_at"]).replace("Z", "+00:00")
                    ).timestamp()
                except ValueError:
                    closed_at = None
            self.set_issue_state(nodeid, number, issue["state"], closed_at, now)
            read += 1
        return read


def store() -> Store:
    global _store
    with _store_lock:
        if _store is None:
            _store = Store(database_path())
        return _store
