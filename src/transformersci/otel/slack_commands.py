"""Bounded Slack command execution with durable retry deduplication.

No response URLs or Slack credentials are stored. Users retrieve results with
/askserge result, keeping all replies private and avoiding outbound Slack access.
"""

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from . import rerun_actions

_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="slack-command")
_SLOTS = threading.BoundedSemaphore(4)
_LOG = logging.getLogger(__name__)


@contextmanager
def connect():
    root = os.getenv("PYTEST_TRACE_EXPORTER_RUN_STORE", "")
    if not root:
        raise ValueError("persistent run store required")
    path = Path(root) / "slack-commands.sqlite3"
    db = sqlite3.connect(path, timeout=1)
    db.execute("""CREATE TABLE IF NOT EXISTS commands (
        id TEXT PRIMARY KEY, owner TEXT NOT NULL, created REAL NOT NULL,
        result TEXT NOT NULL)""")
    try:
        with db:
            yield db
    finally:
        db.close()


def actor_for(team, user):
    try:
        mapping = json.loads(os.getenv("TCI_SLACK_ACTORS", "{}"))
        actor = mapping.get(f"{team}:{user}")
        if isinstance(actor, str) and re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9-]{0,38}", actor
        ):
            return actor
    except (ValueError, AttributeError):
        pass
    return None


def queue(fields, raw, execute, *, notify=None, fallback=None):
    team, user = fields["team_id"][0], fields["user_id"][0]
    owner = f"{team}:{user}"
    text = fields.get("text", [""])[0]
    parts = text.split()
    static_result = None
    operation, pr, actor = "reply", "", None
    if (
        len(parts) == 2
        and parts[0] == "result"
        and re.fullmatch(r"[a-f0-9]{24}", parts[1])
    ):
        with connect() as db:
            row = db.execute(
                "SELECT result FROM commands WHERE id=? AND owner=?", (parts[1], owner)
            ).fetchone()
        static_result = row[0] if row else "Command not found for your Slack account."
    elif (
        len(parts) == 2
        and parts[0] in {"pr", "rerun", "investigate"}
        and re.fullmatch(r"[1-9][0-9]{0,9}", parts[1])
    ):
        operation, pr = parts
        actor = actor_for(team, user)
        if operation != "pr" and not actor:
            static_result = "Your Slack account has no configured GitHub identity. Ask the dashboard administrator to link it."
    elif re.match(r"look at the traceback\s*:", text, re.I):
        operation = "traceback"
        pr = re.sub(
            r"^look at the traceback\s*:\s*", "", text, count=1, flags=re.I
        ).strip()
        actor = actor_for(team, user)
        if not pr:
            static_result = "Please paste a traceback or a link to the failing test on the dashboard."
        elif not actor:
            static_result = (
                "Ask the administrator to link your Slack account to GitHub first."
            )
    elif (
        len(parts) == 3
        and parts[0] == "decision"
        and re.fullmatch(r"[a-f0-9]{24}", parts[1])
        and parts[2] in {"approve", "decline"}
    ):
        operation, pr = "decision", " ".join(parts[1:])
        actor = actor_for(team, user)
        if not actor:
            static_result = "Your Slack account is no longer linked to GitHub."
    elif fallback is not None:
        static_result = fallback(text)
    else:
        return None
    if static_result is not None and notify is None:
        return static_result
    command_id = hashlib.sha256(raw).hexdigest()[:24]
    lookup = f"Use `/askserge result {command_id}` for the result."
    if not _SLOTS.acquire(blocking=False):
        return "The bot is busy. Try again shortly."
    try:
        with connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "DELETE FROM commands WHERE created < ?", (time.time() - 7 * 86400,)
            )
            existing = db.execute(
                "SELECT result FROM commands WHERE id=? AND owner=?",
                (command_id, owner),
            ).fetchone()
            if existing:
                _SLOTS.release()
                return existing[0] + "\n" + lookup
            count = db.execute(
                "SELECT COUNT(*) FROM commands WHERE owner=? AND created>?",
                (owner, time.time() - 3600),
            ).fetchone()[0]
            if count >= 20:
                if notify is None:
                    _SLOTS.release()
                    return "Command limit reached (20 per hour). Try again later."
                static_result = "Command limit reached (20 per hour). Try again later."
            db.execute(
                "INSERT INTO commands VALUES (?, ?, ?, ?)",
                (
                    command_id,
                    owner,
                    time.time(),
                    "Command accepted. If this remains pending after a server restart, check the dashboard before submitting again.",
                ),
            )
    except Exception:
        _SLOTS.release()
        raise

    def work():
        try:
            result = (
                static_result
                if static_result is not None
                else execute(operation, pr, actor, command_id)
            )
        except ValueError as error:
            result = str(error)
        except Exception:
            _LOG.exception("Slack command %s failed", command_id)
            result = "Command failed. Check the dashboard before retrying."
        try:
            with connect() as db:
                db.execute(
                    "UPDATE commands SET result=? WHERE id=?", (result, command_id)
                )
        except Exception:
            _LOG.exception("Could not save Slack command %s result", command_id)
        finally:
            try:
                if notify is not None:
                    notify(result)
            except Exception:
                _LOG.exception("Could not deliver Slack command %s result", command_id)
            finally:
                _SLOTS.release()

    _POOL.submit(work)
    return "Command accepted. " + lookup


def execute(
    operation,
    pr,
    actor,
    command_id,
    *,
    summary,
    submit_rerun,
    dashboard,
    serge,
    github=None,
):
    from .slack_bot import command_reply

    if operation == "pr" and github is not None:
        return pr_status(pr, github, dashboard)
    if operation == "pr":
        return command_reply(f"pr {pr}", summary, dashboard, serge)
    enabled, reason = rerun_actions.dispatch_enabled()
    if not enabled:
        return "Actions are disabled: " + reason
    if operation == "rerun":
        replies = []
        submit_rerun(
            {"pr": pr, "idempotency_key": "slack-" + command_id},
            actor,
            lambda code, body: replies.append((code, body)),
        )
        code, body = replies[0]
        if body.get("status") == "confirm":
            return f"Active CI runs would need cancellation. Review and confirm in <{dashboard}/pr-ci-view?pr={pr}|the dashboard>."
        if code < 300:
            action = body["action"]
            return f"Rerun accepted ({action['id']}). Follow <{dashboard}/pr-ci-view?pr={pr}|PR #{pr}>."
        return "Rerun refused: " + str(body.get("status", "unavailable"))
    code, body = rerun_actions.default_serge().call(
        "investigate",
        {
            "actor": actor,
            "pr": int(pr),
            "idempotency_key": "slack-" + command_id,
        },
    )
    if code >= 300:
        return "Investigation refused: " + str(body.get("detail", "unavailable"))
    path = body.get("url", "")
    if not re.fullmatch(
        r"/reviews/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/[0-9]+/[a-f0-9]+", path
    ):
        return "Serge returned an unexpected response. Check Serge before retrying."
    return f"Serge investigation started: <{serge}{path}|view investigation>. The result is a draft for review."


def pr_status(pr, github, dashboard):
    pull = github(f"pulls/{pr}")
    sha = (pull.get("head") or {}).get("sha", "")
    if not re.fullmatch(r"[a-f0-9]{40}", sha):
        return "GitHub did not return a valid PR head. Check the dashboard."
    checks = github(f"commits/{sha}/check-runs?per_page=100")
    statuses = github(f"commits/{sha}/status?per_page=100")
    counts = {"passed": 0, "failed": 0, "pending": 0, "other": 0}
    for check in checks.get("check_runs", []):
        conclusion = check.get("conclusion")
        key = (
            "pending"
            if check.get("status") != "completed"
            else (
                "passed"
                if conclusion == "success"
                else "failed"
                if conclusion
                in {
                    "failure",
                    "timed_out",
                    "cancelled",
                    "action_required",
                    "startup_failure",
                }
                else "other"
            )
        )
        counts[key] += 1
    for status in statuses.get("statuses", []):
        key = {
            "success": "passed",
            "failure": "failed",
            "error": "failed",
            "pending": "pending",
        }.get(status.get("state"), "other")
        counts[key] += 1
    detail = ", ".join(f"{count} {key}" for key, count in counts.items())
    if checks.get("total_count", 0) > 100 or statuses.get("total_count", 0) > 100:
        detail += " (first 100 checks/statuses only; see dashboard for the rest)"
    return f"<{dashboard}/pr-ci-view?pr={pr}|Transformers PR #{pr}> — {pull.get('state', 'unknown')}, head {sha[:8]}: {detail}."
