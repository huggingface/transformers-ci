"""Who picked up a failing test, so two people never open issues for the same one.

New issue *reserves* the test for whoever starts the draft; everyone else is
told who is drafting instead of getting a second draft. Publishing turns the
reservation into a claim with the issue's number; closing the draft releases
it, and an abandoned one lapses after ``RESERVE_SECONDS``.

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

RESERVE_SECONDS = 15 * 60

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

    def _row(self, nodeid: str) -> dict | None:
        row = self._db.execute(
            "SELECT login, state, created, expires, number, url FROM claims"
            " WHERE nodeid = ?",
            (nodeid,),
        ).fetchone()
        if not row:
            return None
        keys = ("login", "state", "created", "expires", "number", "url")
        return dict(zip(keys, row))

    def reserve(self, nodeid: str, login: str, now: float | None = None) -> dict:
        """Reserve ``nodeid`` for ``login``, or say who already has it.

        ``{"status": "ok"}``, ``{"status": "claimed", login, number, url}`` or
        ``{"status": "drafting", login, since}``. The same person may reserve
        again (a second tab, or publishing after a while): it refreshes.
        """
        now = time.time() if now is None else now
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                current = self._row(nodeid)
                if current and current["state"] == "claimed":
                    return {
                        "status": "claimed",
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
                    "INSERT OR REPLACE INTO claims VALUES (?, ?, 'drafting', ?, ?, NULL, NULL)",
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
                "INSERT OR REPLACE INTO claims VALUES (?, ?, 'claimed', ?, NULL, ?, ?)",
                (nodeid, login, now, number, url),
            )

    def release(self, nodeid: str, login: str) -> None:
        """Drop ``login``'s own reservation; a published claim stays."""
        with self._lock:
            self._db.execute(
                "DELETE FROM claims WHERE nodeid = ? AND login = ? AND state = 'drafting'",
                (nodeid, login),
            )

    def snapshot(self, now: float | None = None) -> dict[str, dict]:
        """Every live claim and reservation, by node id."""
        now = time.time() if now is None else now
        with self._lock:
            rows = self._db.execute(
                "SELECT nodeid, login, state, created, expires, number, url FROM claims"
                " WHERE state = 'claimed' OR expires > ?",
                (now,),
            ).fetchall()
        keys = ("login", "state", "created", "expires", "number", "url")
        return {row[0]: dict(zip(keys, row[1:])) for row in rows}


def store() -> Store:
    global _store
    with _store_lock:
        if _store is None:
            _store = Store(database_path())
        return _store
