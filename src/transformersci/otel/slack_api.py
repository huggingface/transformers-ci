"""Small Slack client; credentials only go to Slack, never to supplied URLs."""

import json
import os
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def call(method, body):
    if method not in {
        "conversations.open",
        "chat.postMessage",
        "agents.sessions.setStatus",
    }:
        raise ValueError("unsupported Slack method")
    token = os.getenv("TCI_SLACK_BOT_TOKEN", "")
    if not token:
        raise ValueError("Slack bot token not configured")
    proxy = os.getenv("TCI_SLACK_PROXY", "")
    opener = build_opener(ProxyHandler({"https": proxy} if proxy else {}), NoRedirect())
    request = Request(
        "https://slack.com/api/" + method,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
        },
    )
    with opener.open(request, timeout=15) as response:
        result = json.loads(response.read())
    if not result.get("ok"):
        raise RuntimeError(
            "Slack API request failed: " + str(result.get("error", "unknown"))
        )
    return result


def open_dm(user):
    return call("conversations.open", {"users": user})["channel"]["id"]


def message(channel, text, blocks=None, *, thread_ts=None):
    body = {
        "channel": channel,
        "text": text,
        "unfurl_links": False,
        "unfurl_media": False,
    }
    if thread_ts:
        body["thread_ts"] = thread_ts
    if blocks is not None:
        body["blocks"] = blocks
    return call("chat.postMessage", body)


def session(channel, thread_ts, status, user):
    return call(
        "agents.sessions.setStatus",
        {
            "channel_id": channel,
            "thread_ts": thread_ts,
            "status": status,
            "title": "Serge failure investigation",
            "initiator_user_id": user,
        },
    )
