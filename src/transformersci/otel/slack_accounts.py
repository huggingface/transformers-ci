"""Verified, persistent Slack/GitHub linking with confirmation in both accounts."""

import hashlib
import html
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import time
from urllib.parse import urlsplit

from . import dashboard_auth as auth, slack_api, slack_commands

TTL = 900
LOGIN = r"[A-Za-z0-9][A-Za-z0-9-]{0,38}"


def init(db):
    db.execute("""CREATE TABLE IF NOT EXISTS slack_accounts (
        owner TEXT PRIMARY KEY, team TEXT NOT NULL, github_id INTEGER NOT NULL,
        login TEXT NOT NULL, created REAL NOT NULL, UNIQUE(team, github_id))""")
    db.execute("""CREATE TABLE IF NOT EXISTS slack_link_grants (
        id TEXT PRIMARY KEY, token TEXT UNIQUE NOT NULL, owner TEXT NOT NULL,
        expires REAL NOT NULL, github_id INTEGER, login TEXT,
        confirmed INTEGER NOT NULL DEFAULT 0)""")


def manual():
    try:
        values = json.loads(os.getenv("TCI_SLACK_ACTORS", "{}"))
        return {
            owner: login
            for owner, login in values.items()
            if re.fullmatch(r"T[A-Z0-9]+:[UW][A-Z0-9]+", owner)
            and isinstance(login, str)
            and re.fullmatch(LOGIN, login)
        }
    except (ValueError, AttributeError, TypeError):
        return {}


def allowed(owner):
    return bool(re.fullmatch(r"T[A-Z0-9]+:[UW][A-Z0-9]+", owner)) and owner.split(":")[
        0
    ] in os.getenv("TCI_SLACK_TEAM_IDS", "").split(",")


def lookup(owner, *, refresh=False):
    configured = manual().get(owner)
    if configured:
        return configured
    if not os.getenv("PYTEST_TRACE_EXPORTER_RUN_STORE"):
        return None
    with slack_commands.connect() as db:
        init(db)
        row = db.execute(
            "SELECT github_id, login FROM slack_accounts WHERE owner=?", (owner,)
        ).fetchone()
    if not row:
        return None
    if not refresh:
        return row[1]
    # Usernames can be renamed/reassigned. Resolve the verified immutable ID
    # before authorizing work; never grant another user's permissions on rename.
    user = auth.github_json(f"https://api.github.com/user/{row[0]}")
    if (
        not isinstance(user, dict)
        or user.get("id") != row[0]
        or not re.fullmatch(LOGIN, str(user.get("login", "")))
    ):
        raise ValueError(
            "Could not verify your linked GitHub account. Try again later."
        )
    with slack_commands.connect() as db:
        changed = db.execute(
            "UPDATE slack_accounts SET login=? WHERE owner=? AND github_id=?",
            (user["login"], owner, row[0]),
        ).rowcount
    if not changed:
        raise ValueError(
            "Your GitHub account was disconnected. Send `connect` to link it again."
        )
    return user["login"]


def owners_for(login, github_id):
    mappings = manual()
    owners = {
        owner
        for owner, name in mappings.items()
        if name.lower() == login.lower() and allowed(owner)
    }
    if github_id is not None:
        with slack_commands.connect() as db:
            init(db)
            owners.update(
                row[0]
                for row in db.execute(
                    "SELECT owner FROM slack_accounts WHERE github_id=?", (github_id,)
                )
                if row[0] not in mappings and allowed(row[0])
            )
    return sorted(owners)


def prompt(owner):
    if not allowed(owner) or not auth.config():
        return "GitHub account linking is not configured. Contact the dashboard administrator."
    existing = lookup(owner)
    if existing:
        return f"Your Slack account is linked to GitHub @{existing}. Send `disconnect` before connecting a different account."
    token, grant = secrets.token_urlsafe(32), secrets.token_hex(16)
    with slack_commands.connect() as db:
        init(db)
        db.execute("DELETE FROM slack_link_grants WHERE expires<?", (time.time(),))
        db.execute(
            "INSERT INTO slack_link_grants(id,token,owner,expires) VALUES (?,?,?,?)",
            (
                grant,
                hashlib.sha256(token.encode()).hexdigest(),
                owner,
                time.time() + TTL,
            ),
        )
    # Fragment stays out of HTTP access logs and referrers. The page saves it in
    # per-tab storage before starting the existing GitHub OAuth flow.
    return (
        f"Connect your GitHub account to use investigations and CI reruns: <{auth.config().url}/auth/slack#{token}|Connect GitHub>. "
        "Sign in, then confirm in this Slack DM. The link expires in 15 minutes. "
        "After connecting, retry your command. You can still use `help` and `pr <number>` without linking."
    )


def propose(token, identity):
    if not isinstance(token, str) or not re.fullmatch(r"[A-Za-z0-9_-]{43}", token):
        raise auth.AuthError(400, "invalid_or_expired_link")
    with slack_commands.connect() as db:
        init(db)
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT id,owner,github_id,login,confirmed FROM slack_link_grants WHERE token=? AND expires>?",
            (hashlib.sha256(token.encode()).hexdigest(), time.time()),
        ).fetchone()
        if not row or not allowed(row[1]) or row[4]:
            raise auth.AuthError(400, "invalid_or_expired_link")
        if row[2] is not None and row[2] != identity["user_id"]:
            raise auth.AuthError(409, "link_already_claimed")
        if row[1] in manual():
            raise auth.AuthError(409, "managed_account")
        db.execute(
            "UPDATE slack_link_grants SET github_id=?,login=? WHERE id=?",
            (identity["user_id"], identity["login"], row[0]),
        )
    channel = slack_api.open_dm(row[1].split(":")[1])
    text = f"Connect this Slack account to GitHub @{identity['login']}? Only confirm if you just signed in to connect it."
    slack_api.message(
        channel,
        text,
        [
            {"type": "section", "text": {"type": "mrkdwn", "text": html.escape(text)}},
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "Confirm GitHub connection",
                        },
                        "action_id": "serge_link_confirm",
                        "value": row[0],
                    }
                ],
            },
        ],
    )


def confirm(owner, grant):
    if not re.fullmatch(r"[a-f0-9]{32}", grant) or not allowed(owner):
        return "Invalid connection request. Send `connect` to start again."
    with slack_commands.connect() as db:
        init(db)
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT github_id,login,confirmed FROM slack_link_grants WHERE id=? AND owner=? AND expires>?",
            (grant, owner, time.time()),
        ).fetchone()
        if not row or row[0] is None:
            return "This connection request expired or belongs to another Slack account. Send `connect` to start again."
        configured = manual()
        if owner in configured:
            return "Your account is managed by the administrator; its mapping has not changed."
        if any(
            name.lower() == row[1].lower()
            and other != owner
            and other.split(":")[0] == owner.split(":")[0]
            for other, name in configured.items()
        ):
            return "That GitHub account already has a managed Slack connection. Contact the administrator."
        existing = db.execute(
            "SELECT github_id FROM slack_accounts WHERE owner=?", (owner,)
        ).fetchone()
        if existing and existing[0] != row[0]:
            return "Your Slack account is already connected. Send `disconnect` before changing accounts."
        if row[2]:
            return "This connection request has already been used. Send `connect` to check your connection."
        try:
            db.execute(
                "INSERT INTO slack_accounts(owner,team,github_id,login,created) VALUES (?,?,?,?,?) ON CONFLICT(owner) DO UPDATE SET login=excluded.login",
                (owner, owner.split(":")[0], row[0], row[1], time.time()),
            )
        except sqlite3.IntegrityError:
            return "That GitHub account is already linked to another Slack account in this workspace. Disconnect it there first."
        db.execute("UPDATE slack_link_grants SET confirmed=1 WHERE owner=?", (owner,))
    return f"Connected to GitHub @{row[1]}. Retry your investigation or CI command. Send `disconnect` to unlink."


def disconnect(owner):
    if owner in manual():
        return "Your account is managed by the administrator. Ask them to remove its configured mapping."
    with slack_commands.connect() as db:
        init(db)
        db.execute("DELETE FROM slack_accounts WHERE owner=?", (owner,))
        db.execute("DELETE FROM slack_link_grants WHERE owner=?", (owner,))
    return "GitHub account disconnected. Send `connect` to link an account again."


def handle(handler, method):
    path = urlsplit(handler.path).path
    if path not in {"/auth/slack", "/auth/slack/app.js", "/auth/slack/connect"}:
        return False
    handler._request_route = "/auth/slack"
    headers = {
        "Cache-Control": "no-store",
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Content-Security-Policy": "default-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
    }
    try:
        if method == "GET" and path in {"/auth/slack", "/auth/slack/app.js"}:
            asset = "slack_connect.js" if path.endswith(".js") else "slack_connect.html"
            handler._send(
                200,
                "text/javascript"
                if asset.endswith(".js")
                else "text/html; charset=utf-8",
                Path(__file__).with_name(asset).read_bytes(),
                headers=headers,
            )
            return True
        cfg = auth.config()
        if not cfg or not os.getenv("TCI_SLACK_BOT_TOKEN"):
            raise auth.AuthError(503, "linking_unavailable")
        if method != "POST" or path != "/auth/slack/connect":
            raise auth.AuthError(405, "method_not_allowed")
        user = auth._browser_post(handler, cfg)
        auth.store(cfg).limit(f"slack-link:{user['user_id']}", 5)
        propose(auth._body(handler).get("token"), user)
        handler._send(
            200, "application/json", b'{"status":"confirm_in_slack"}', headers=headers
        )
    except auth.AuthError as error:
        handler._send(
            error.status,
            "application/json",
            json.dumps({"error": error.error}).encode(),
            headers=headers,
        )
    except Exception:
        handler._send(
            503, "application/json", b'{"error":"linking_unavailable"}', headers=headers
        )
    return True
