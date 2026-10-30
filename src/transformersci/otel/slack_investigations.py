"""Interactive ITF-style investigations: durable DM conversations and approval.

Only the dashboard's configured Tempo is queried. A pasted URL supplies IDs,
never a fetch destination. The Serge runner owns preparation and verification.
"""

import html
import logging
import os
import re
import threading
import time
import uuid
from urllib.parse import parse_qs, urlparse

from . import rerun_actions, slack_api, slack_commands
from transformersci.agentic import dashboard_failure_triage as triage

_LOG = logging.getLogger(__name__)
_STARTED = False
_LOCK = threading.Lock()


def init(db):
    db.execute("""CREATE TABLE IF NOT EXISTS investigations (
        id TEXT PRIMARY KEY, owner TEXT NOT NULL, actor TEXT NOT NULL,
        channel TEXT NOT NULL, job TEXT, cursor INTEGER NOT NULL DEFAULT -1,
        state TEXT NOT NULL, created REAL NOT NULL)""")

    columns = {row[1] for row in db.execute("PRAGMA table_info(investigations)")}
    for name, definition in {
        "thread_ts": "TEXT NOT NULL DEFAULT ''",
        "session_status": "TEXT NOT NULL DEFAULT ''",
        "session_updated": "REAL NOT NULL DEFAULT 0",
    }.items():
        if name not in columns:
            db.execute(f"ALTER TABLE investigations ADD COLUMN {name} {definition}")


def reply(owner, command_id, user, text, *, channel=None, thread_ts=None):
    with slack_commands.connect() as db:
        init(db)
        row = db.execute(
            "SELECT channel, thread_ts FROM investigations WHERE id=? AND owner=?",
            (command_id, owner),
        ).fetchone()
    channel = channel or slack_api.open_dm(user)
    if row:
        channel, thread_ts = row
    if thread_ts:
        return slack_api.message(channel, text, thread_ts=thread_ts)
    return slack_api.message(channel, text)


def sync_sessions():
    # Persist successful syncs separately from workflow state so Slack outages
    # cannot lose terminal transitions. Refresh long-running sessions every 20m.
    now = time.time()
    with slack_commands.connect() as db:
        init(db)
        db.execute(
            "UPDATE investigations SET state='expired' WHERE state='awaiting_approval' AND created<?",
            (now - 86400,),
        )
        rows = db.execute(
            "SELECT id, owner, channel, thread_ts, state, session_status, session_updated FROM investigations WHERE thread_ts!=''"
        ).fetchall()
    for command_id, owner, channel, thread_ts, state, previous, updated in rows:
        desired = (
            "processing"
            if state == "running"
            else "suspended"
            if state == "awaiting_approval"
            else "closed"
        )
        if desired == previous and (desired != "processing" or now - updated < 1200):
            continue
        try:
            slack_api.session(channel, thread_ts, desired, owner.split(":", 1)[1])
            with slack_commands.connect() as db:
                db.execute(
                    "UPDATE investigations SET session_status=?, session_updated=? WHERE id=?",
                    (desired, now, command_id),
                )
        except Exception:
            _LOG.exception("Could not sync Slack session %s", command_id)


def context_for(text, get_trace):
    text = html.unescape(text).strip()
    # Slack wraps links as <url|label>. Only interpret a standalone dashboard URL.
    text = re.sub(r"^<(https://[^>|]+)(?:\|[^>]+)?>", r"\1", text)
    url_match = re.match(r"^(https://\S+)", text)
    if not url_match:
        if len(text) > 40000:
            raise ValueError("Traceback is too long; use a dashboard link.")
        return "Failure evidence supplied by a maintainer (untrusted):\n\n" + text
    url = url_match.group(1).rstrip(",")
    parsed_url = urlparse(url)
    public = urlparse(os.getenv("TCI_SLACK_PUBLIC_URL", ""))
    if (parsed_url.scheme, parsed_url.netloc) != (public.scheme, public.netloc):
        raise ValueError("Use a link to this dashboard, or paste the traceback itself.")
    parsed = triage.parse_dashboard_url(url)
    query = parse_qs(parsed_url.query)
    trace_id = parsed.get("trace_id") or (query.get("trace_id") or [""])[0]
    nodeid = parsed.get("test_nodeid") or (query.get("test_nodeid") or [None])[0]
    if not re.fullmatch(r"[a-fA-F0-9]{32}", trace_id):
        raise ValueError(
            "Use the failing test's dashboard link containing a trace_id, or paste its traceback."
        )
    failure = get_trace(trace_id, nodeid)
    if not failure:
        raise ValueError(
            "No failure found in that trace. Please paste the traceback or use a failing test link."
        )
    fingerprint = triage.failure_fingerprint(
        failure.get("nodeid") or nodeid or "?",
        failure.get("exception_type") or "",
        failure.get("exception_message") or "",
    )
    return triage.render_serge_context(
        failure,
        {**parsed, "trace_id": trace_id},
        fingerprint,
        os.getenv("TCI_SLACK_PUBLIC_URL", ""),
        trace_chars=30000,
    )


def begin(owner, user, actor, command_id, evidence, get_trace):
    if not actor:
        return "Ask the administrator to map your Slack account to your GitHub account first."
    if not (rerun_actions.serge_url() and rerun_actions.serge_token()):
        return "The Serge connection is not configured."
    channel = slack_api.open_dm(user)
    with slack_commands.connect() as db:
        init(db)
        row = db.execute(
            "SELECT job FROM investigations WHERE id=? AND owner=?", (command_id, owner)
        ).fetchone()
        if row:
            return (
                "This investigation has already been accepted. Follow its updates here."
            )
    context = context_for(evidence, get_trace)
    root = slack_api.message(
        channel,
        "I’m collecting the failure evidence. I’ll send phase updates here and ask before creating a PR.",
    )
    thread_ts = root["ts"]
    # Persist the server's deterministic ID before dispatch. If the response is
    # lost or this process restarts, polling can still discover the accepted job.
    job = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"slack-task:huggingface/transformers:{actor}:slack-{command_id}",
    ).hex
    with slack_commands.connect() as db:
        db.execute(
            "INSERT INTO investigations(id, owner, actor, channel, job, state, created, thread_ts) VALUES (?, ?, ?, ?, ?, 'running', ?, ?)",
            (command_id, owner, actor, channel, job, time.time(), thread_ts),
        )
    sync_sessions()
    try:
        code, result = rerun_actions.default_serge().call(
            "traceback",
            {
                "actor": actor,
                "context": context,
                "idempotency_key": "slack-" + command_id,
            },
        )
    except Exception:
        start()
        return "Serge has not confirmed receipt yet. I’m checking its status; please do not submit the same investigation again."
    if code >= 300 or result.get("id") != job:
        with slack_commands.connect() as db:
            db.execute(
                "UPDATE investigations SET state='error' WHERE id=?", (command_id,)
            )
        return "Serge could not start the investigation: " + str(
            result.get("detail", "unexpected response")
        )
    start()
    return "Investigation started. I’ll reproduce the failure and prepare and verify a fix where the task configuration supports it."


def approval_blocks(command_id, text):
    return [
        {"type": "section", "text": {"type": "mrkdwn", "text": text[:2900]}},
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Create PR"},
                    "style": "primary",
                    "action_id": "serge_approve",
                    "value": command_id,
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "No PR"},
                    "action_id": "serge_decline",
                    "value": command_id,
                },
            ],
        },
    ]


def poll_once():
    sync_sessions()
    with slack_commands.connect() as db:
        init(db)
        rows = db.execute(
            "SELECT id, actor, channel, job, cursor, created, thread_ts FROM investigations WHERE state='running'"
        ).fetchall()
    for command_id, actor, channel, job, cursor, created, thread_ts in rows:

        def message(text, blocks=None):
            if thread_ts:
                return slack_api.message(channel, text, blocks, thread_ts=thread_ts)
            return slack_api.message(channel, text, blocks)

        try:
            code, data = rerun_actions.default_serge().call(
                "traceback/status", {"actor": actor, "id": job, "after": cursor}
            )
            if code >= 300:
                if (
                    code == 404 and time.time() - created > 300
                ) or time.time() - created > 86400:
                    message(
                        "I could not recover this investigation from Serge. Check the task dashboard before starting it again.",
                    )
                    with slack_commands.connect() as db:
                        db.execute(
                            "UPDATE investigations SET state='error' WHERE id=?",
                            (command_id,),
                        )
                continue
            for step in data.get("steps", []):
                phase = str(step.get("text", ""))
                if phase not in {"done", "awaiting_approval"}:
                    labels = {
                        "clone": "Checking out the code",
                        "preflight": "Checking the test environment",
                        "reproduce": "Reproducing the failure",
                        "llm": "Investigating the cause and preparing a fix",
                        "history": "Checking previous failures and fixes",
                        "guidance": "Checking maintainer guidance",
                        "normalize": "Validating the patch",
                        "apply": "Applying the proposed fix",
                        "commit": "Preparing the candidate branch",
                        "verify": "Verifying the fix",
                    }
                    message(
                        labels.get(phase, "Investigation phase: " + phase[:250]),
                    )
                cursor = max(cursor, step["seq"])
                with slack_commands.connect() as db:
                    db.execute(
                        "UPDATE investigations SET cursor=? WHERE id=?",
                        (cursor, command_id),
                    )
            state = data.get("status")
            if state == "running":
                continue
            result = data.get("result") or {}
            if state == "awaiting_approval":
                proposal = result.get("proposal") or {}
                branch = result.get("branch", "")
                # Plain text fields are escaped before inclusion in Slack mrkdwn.
                title = html.escape(str(proposal.get("title", "Proposed fix")))
                summary = html.escape(str(proposal.get("body", ""))[:1800])
                verdict = html.escape(str(result.get("verify_verdict") or "not run"))
                text = f"{title}\n{summary}\nVerification: {verdict}.\n"
                if result.get("expectation_only"):
                    text += "This changes test expectations; a passing verification is not independent proof of a fix.\n"
                text += f"<https://github.com/huggingface/transformers/tree/{branch}|Review candidate branch>\nWould you like me to create the PR? Approval expires 24 hours after the investigation started."
                message(text, approval_blocks(command_id, text))
            else:
                detail = result.get("message") or data.get("error") or state
                message("Investigation finished: " + str(detail)[:3000])
            with slack_commands.connect() as db:
                db.execute(
                    "UPDATE investigations SET state=? WHERE id=?",
                    (state or "error", command_id),
                )
        except Exception:
            _LOG.exception("Could not update Slack investigation %s", command_id)

    sync_sessions()


def pending_decision(owner, channel, text, thread_ts=""):
    with slack_commands.connect() as db:
        init(db)
        rows = db.execute(
            "SELECT id FROM investigations WHERE owner=? AND channel=? AND thread_ts=? AND state='awaiting_approval' AND created>?",
            (owner, channel, thread_ts, time.time() - 86400),
        ).fetchall()
    if len(rows) != 1:
        return None
    decision = "approve" if text in {"yes", "create pr"} else "decline"
    return f"decision {rows[0][0]} {decision}"


def decide(owner, actor, channel, command_id, decision):
    with slack_commands.connect() as db:
        init(db)
        row = db.execute(
            "SELECT actor, channel, job, state FROM investigations WHERE id=? AND owner=?",
            (command_id, owner),
        ).fetchone()
    if not row or row[0] != actor or row[1] != channel:
        return "This approval belongs to another investigation or account."
    code, data = rerun_actions.default_serge().call(
        "traceback/decision", {"actor": actor, "id": row[2], "decision": decision}
    )
    if code >= 300:
        return "Cannot apply that decision: " + str(data.get("detail", "unavailable"))
    with slack_commands.connect() as db:
        db.execute(
            "UPDATE investigations SET state=? WHERE id=?", (data["status"], command_id)
        )
    sync_sessions()
    if data["status"] == "published":
        return "PR created: " + str((data.get("result") or {}).get("url", ""))
    return "No PR will be created. The candidate branch is retained for reference."


def start():
    global _STARTED
    if not (
        os.getenv("TCI_SLACK_BOT_TOKEN")
        and os.getenv("PYTEST_TRACE_EXPORTER_RUN_STORE")
    ):
        return
    with _LOCK:
        if _STARTED:
            return
        _STARTED = True

    def run():
        while True:
            try:
                poll_once()
            except Exception:
                _LOG.exception("Slack investigation polling failed")
            time.sleep(5)

    threading.Thread(target=run, daemon=True, name="slack-investigations").start()
