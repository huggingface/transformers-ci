"""Signed, workspace-restricted Slack commands for the dashboard.

Only local telemetry is read on the request path: Slack requires a prompt ack.
Slack identity is deliberately not treated as GitHub authorization.
"""

import hashlib
import hmac
import json
import os
import re
import time
from urllib.parse import parse_qs, urlparse


def verified(body: bytes, timestamp: str, signature: str, secret: str) -> bool:
    if not re.fullmatch(r"[0-9]{1,12}", timestamp) or not re.fullmatch(
        r"v0=[a-f0-9]{64}", signature
    ):
        return False
    try:
        if abs(time.time() - int(timestamp)) > 300:
            return False
    except (ValueError, TypeError):
        return False
    expected = (
        "v0="
        + hmac.new(
            secret.encode(), b"v0:" + timestamp.encode() + b":" + body, hashlib.sha256
        ).hexdigest()
    )
    return hmac.compare_digest(expected, signature)


def command_reply(text: str, summary, dashboard: str, serge: str) -> str:
    parts = text.split()
    if not parts or parts == ["help"]:
        return (
            "*Investigate a failure and prepare a fix*\n"
            "Mention `@askserge look at the traceback: <traceback or dashboard link>`, "
            "or send `look at the traceback: ...` in this DM. "
            "I’ll investigate, prepare a fix, and run the configured verification. "
            "You’ll receive phase updates in a private thread, and I’ll ask before creating a PR.\n\n"
            "*Commands*\n"
            "• `/askserge pr <number>` — Show CI check status for the PR’s current commit and link to its dashboard.\n"
            "• `/askserge rerun <number>` — Rerun failed CI for that PR. If active runs need cancellation, confirm in the dashboard first.\n"
            "• `/askserge investigate <number>` — Ask Serge for a draft review of the PR; returns a link to the review.\n"
            "• `/askserge result <id>` — Retrieve a saved command result using the ID from its receipt.\n"
            "• `/askserge dashboard` — Open the CI dashboard.\n"
            "• `/askserge serge` — Open Serge.\n"
            "• `/askserge help` — Show this guide.\n\n"
            "In a DM, omit `/askserge` (for example, `pr 123`). "
            "In a channel, mention `@askserge pr 123`; replies arrive privately.\n"
            "Investigations and CI reruns require a linked GitHub account and repository permission. "
            "For a proposed fix, use *Create PR* / *No PR*, or reply *yes* / *no* in its thread."
        )
    if text.strip().lower() in {"yes", "create pr", "no", "no pr"}:
        return "Use the Create PR or No PR button on the investigation you mean, or reply yes/no in the investigation’s thread."
    if parts == ["dashboard"]:
        return f"<{dashboard}/pr-table|Open CI dashboard>"
    if parts == ["serge"]:
        return f"<{serge}|Open Serge>" if serge else "Serge URL is not configured."
    if (
        len(parts) == 2
        and parts[0] == "pr"
        and re.fullmatch(r"[1-9][0-9]{0,9}", parts[1])
    ):
        pr = parts[1]
        link = f"<{dashboard}/pr-ci-view?pr={pr}|Transformers PR #{pr}>"
        run = summary(pr)
        if run is None:
            return f"{link}: no telemetry in the current dashboard snapshot. Open the dashboard for full CI status."

        def count(key):
            value = run.get(key)
            return str(int(value)) if isinstance(value, (int, float)) else "unknown"

        return (
            f"{link}: latest slow-test telemetry snapshot — {count('failed_tests')} failed / "
            f"{count('total_tests')} tests observed. This is not the overall CI conclusion."
        )
    return "Unknown command. Use `/askserge help`."


def handle(handler, summary, execute=None, investigation=None) -> bool:
    path = urlparse(handler.path).path
    if path not in {"/slack/commands", "/slack/events", "/slack/actions"}:
        return False
    handler._request_route = path
    secret = os.getenv("TCI_SLACK_SIGNING_SECRET", "")
    teams = set(filter(None, os.getenv("TCI_SLACK_TEAM_IDS", "").split(",")))
    dashboard = os.getenv("TCI_SLACK_PUBLIC_URL", "").rstrip("/")
    if not secret or not teams or not dashboard:
        handler._reply_json(503, {"error": "slack_not_configured"})
        return True
    try:
        length = int(handler.headers.get("Content-Length", "0"))
        if not 0 < length <= 65536 or handler.headers.get("Transfer-Encoding"):
            raise ValueError
    except ValueError:
        handler._reply_json(400, {"error": "invalid_body"})
        return True
    expected_type = (
        "application/json"
        if path == "/slack/events"
        else "application/x-www-form-urlencoded"
    )
    if handler.headers.get_content_type() != expected_type:
        handler._reply_json(415, {"error": "invalid_content_type"})
        return True
    raw = handler.rfile.read(length)
    if not verified(
        raw,
        handler.headers.get("X-Slack-Request-Timestamp", ""),
        handler.headers.get("X-Slack-Signature", ""),
        secret,
    ):
        handler._reply_json(401, {"error": "invalid_signature"})
        return True
    try:
        if path == "/slack/events":
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError
            if payload.get("type") == "url_verification" and isinstance(
                payload.get("challenge"), str
            ):
                handler._reply_json(200, {"challenge": payload["challenge"]})
                return True
            if not os.getenv("TCI_SLACK_APP_ID") or payload.get(
                "api_app_id"
            ) != os.getenv("TCI_SLACK_APP_ID"):
                handler._reply_json(403, {"error": "app_not_allowed"})
                return True
            event = payload.get("event") or {}
            direct = (
                event.get("type") == "message" and event.get("channel_type") == "im"
            )
            if (
                (event.get("type") != "app_mention" and not direct)
                or event.get("bot_id")
                or event.get("subtype")
            ):
                handler._reply_json(200, {})
                return True
            bot = os.getenv("TCI_SLACK_BOT_USER_ID", "")
            if (
                not bot
                or event.get("user") == bot
                or (
                    not direct
                    and not str(event.get("text", "")).startswith(f"<@{bot}>")
                )
            ):
                handler._reply_json(200, {})
                return True
            fields = {
                "team_id": [payload.get("team_id", "")],
                "user_id": [event.get("user", "")],
                "channel_id": [event.get("channel", "")],
                "thread_ts": [event.get("thread_ts", "")],
                "text": [
                    event.get("text", "").strip()
                    if direct
                    else event["text"][len(bot) + 3 :].strip()
                ],
            }
        else:
            fields = parse_qs(
                raw.decode("utf-8"), keep_blank_values=True, max_num_fields=40
            )
            if any(len(values) != 1 for values in fields.values()):
                raise ValueError
            if path == "/slack/actions":
                payload = json.loads(fields.get("payload", [""])[0])
                if payload.get("type") != "block_actions" or payload.get(
                    "api_app_id"
                ) != os.getenv("TCI_SLACK_APP_ID"):
                    raise ValueError
                actions = payload.get("actions", [])
                if len(actions) != 1 or actions[0].get("action_id") not in {
                    "serge_approve",
                    "serge_decline",
                }:
                    raise ValueError
                decision = actions[0]["action_id"].removeprefix("serge_")
                fields = {
                    "team_id": [payload["team"]["id"]],
                    "user_id": [payload["user"]["id"]],
                    "channel_id": [payload["channel"]["id"]],
                    "thread_ts": [(payload.get("message") or {}).get("thread_ts", "")],
                    "text": [f"decision {actions[0]['value']} {decision}"],
                }
            elif fields.get("command") != ["/askserge"]:
                raise ValueError
    except (ValueError, UnicodeError, KeyError, TypeError, AttributeError):
        handler._reply_json(400, {"error": "invalid_payload"})
        return True
    team, user = fields.get("team_id", [""])[0], fields.get("user_id", [""])[0]
    if (
        not isinstance(team, str)
        or team not in teams
        or not isinstance(user, str)
        or not re.fullmatch(r"[UW][A-Z0-9]+", user)
    ):
        handler._reply_json(403, {"error": "workspace_or_user_not_allowed"})
        return True
    if path == "/slack/events":
        event_id = payload.get("event_id")
        if not isinstance(event_id, str) or not re.fullmatch(
            r"Ev[A-Za-z0-9]+", event_id
        ):
            handler._reply_json(400, {"error": "invalid_event_id"})
            return True
        raw = f"event:{team}:{event_id}".encode()
    from . import slack_api, slack_commands, slack_investigations

    if path == "/slack/events" and direct:
        text = fields["text"][0].strip().lower()
        if text in {"yes", "create pr", "no", "no pr"}:
            try:
                command = slack_investigations.pending_decision(
                    f"{team}:{user}",
                    fields["channel_id"][0],
                    text,
                    fields["thread_ts"][0],
                )
            except Exception:
                handler._reply_json(503, {"error": "command_store_unavailable"})
                return True
            if command:
                fields["text"] = [command]
    dm = [None]
    reply_id = [None]

    def channel():
        if dm[0] is None:
            dm[0] = slack_api.open_dm(user)
        return dm[0]

    def run(operation, argument, actor, command_id):
        reply_id[0] = command_id
        if path != "/slack/commands" or operation == "traceback":
            channel()  # Establish the private reply channel before starting work.
        if operation == "traceback" and investigation is not None:
            return investigation(f"{team}:{user}", user, actor, command_id, argument)
        if operation == "decision":
            command, decision = argument.split()
            reply_id[0] = command
            return slack_investigations.decide(
                f"{team}:{user}",
                actor,
                fields.get("channel_id", [""])[0],
                command,
                decision,
            )
        return execute(operation, argument, actor, command_id)

    def fallback(text):
        return command_reply(
            text,
            summary,
            dashboard,
            os.getenv("TCI_SLACK_SERGE_PUBLIC_URL", "").rstrip("/"),
        )

    if execute is not None:
        try:
            reply = slack_commands.queue(
                fields,
                raw,
                run,
                notify=(
                    lambda text: slack_investigations.reply(
                        f"{team}:{user}",
                        reply_id[0],
                        user,
                        text,
                        channel=channel(),
                        thread_ts=fields.get("thread_ts", [""])[0]
                        if fields.get("channel_id", [""])[0] == channel()
                        else None,
                    )
                )
                if path != "/slack/commands"
                else None,
                fallback=fallback,
            )
        except Exception:
            handler._reply_json(503, {"error": "command_store_unavailable"})
            return True
    else:
        reply = fallback(fields.get("text", [""])[0])
    if path != "/slack/commands":
        handler._reply_json(
            503 if reply == "The bot is busy. Try again shortly." else 200, {}
        )
    else:
        handler._reply_json(
            200,
            {
                "response_type": "ephemeral",
                "text": reply,
                "unfurl_links": False,
                "unfurl_media": False,
            },
        )
    return True
