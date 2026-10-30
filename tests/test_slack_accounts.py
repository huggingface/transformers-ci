import hashlib
import io
import json
import re
from email.message import Message
from types import SimpleNamespace

import pytest

from transformersci.otel import (
    dashboard_auth as auth,
    slack_accounts as links,
    slack_commands,
    slack_bot,
)


@pytest.fixture
def configured(monkeypatch, tmp_path):
    monkeypatch.setenv("TCI_AUTH_URL", "https://dashboard.test")
    monkeypatch.setenv("TCI_AUTH_GITHUB_CLIENT_ID", "app")
    monkeypatch.setenv("TCI_AUTH_GITHUB_CLIENT_SECRET", "secret")
    monkeypatch.setenv("TCI_AUTH_DATABASE", str(tmp_path / "auth.sqlite3"))
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_RUN_STORE", str(tmp_path))
    monkeypatch.setenv("TCI_SLACK_TEAM_IDS", "T1")
    monkeypatch.setenv("TCI_SLACK_ACTORS", "{}")
    monkeypatch.setenv("TCI_SLACK_BOT_TOKEN", "fake")
    monkeypatch.setattr(links.slack_api, "open_dm", lambda user: "D" + user)
    messages = []
    monkeypatch.setattr(links.slack_api, "message", lambda *args: messages.append(args))
    return messages


def grant(owner="T1:U1"):
    text = links.prompt(owner)
    token = re.search(r"/auth/slack#([A-Za-z0-9_-]+)", text)[1]
    with slack_commands.connect() as db:
        row = db.execute(
            "SELECT id,token FROM slack_link_grants WHERE token=?",
            (hashlib.sha256(token.encode()).hexdigest(),),
        ).fetchone()
    assert row[1] != token
    return token, row[0]


def person(uid=42, login="octocat"):
    return {"user_id": uid, "login": login}


def test_link_needs_both_accounts_and_survives_restart(configured, monkeypatch):
    token, gid = grant()
    assert links.lookup("T1:U1") is None
    links.propose(token, person())
    assert links.lookup("T1:U1") is None
    assert configured[-1][0] == "DU1"
    assert configured[-1][2][1]["elements"][0]["value"] == gid
    assert "another" in links.confirm("T1:U2", gid)
    assert links.lookup("T1:U1") is None
    assert "Connected" in links.confirm("T1:U1", gid)
    assert slack_commands.actor_for("T1", "U1") == "octocat"
    assert links.owners_for("renamed", 42) == ["T1:U1"]
    monkeypatch.setattr(auth, "github_json", lambda url: {"id": 42, "login": "renamed"})
    assert links.lookup("T1:U1", refresh=True) == "renamed"
    assert links.lookup("T1:U1") == "renamed"
    assert "already been used" in links.confirm("T1:U1", gid)
    links.disconnect("T1:U1")
    assert links.lookup("T1:U1") is None
    assert "expired" in links.confirm("T1:U1", gid)


def test_replay_expiry_and_claim_cannot_be_rebound(configured, monkeypatch):
    token, gid = grant()
    links.propose(token, person())
    with pytest.raises(auth.AuthError, match="link_already_claimed"):
        links.propose(token, person(43, "other"))
    now = links.time.time()
    monkeypatch.setattr(links.time, "time", lambda: now + links.TTL + 1)
    with pytest.raises(auth.AuthError, match="invalid_or_expired_link"):
        links.propose(token, person())
    assert "expired" in links.confirm("T1:U1", gid)
    assert links.lookup("T1:U1") is None


def test_one_github_identity_per_workspace_and_manual_precedence(
    configured, monkeypatch
):
    token, first = grant()
    links.propose(token, person())
    links.confirm("T1:U1", first)
    token, second = grant("T1:U2")
    links.propose(token, person())
    assert "another Slack account" in links.confirm("T1:U2", second)
    monkeypatch.setenv("TCI_SLACK_ACTORS", '{"T1:U1":"managed"}')
    assert links.lookup("T1:U1") == "managed"
    assert links.owners_for("octocat", 42) == []
    assert links.owners_for("managed", 99) == ["T1:U1"]
    assert "administrator" in links.disconnect("T1:U1")
    assert "managed" in links.confirm("T1:U1", first)


def test_grant_cannot_replace_an_existing_link(configured):
    a, first = grant()
    b, second = grant()
    links.propose(a, person())
    links.propose(b, person(43, "other"))
    links.confirm("T1:U1", first)
    assert "already connected" in links.confirm("T1:U1", second)
    assert links.lookup("T1:U1") == "octocat"


def test_refresh_fails_closed_on_wrong_stable_identity(configured, monkeypatch):
    token, gid = grant()
    links.propose(token, person())
    links.confirm("T1:U1", gid)
    monkeypatch.setattr(auth, "github_json", lambda url: {"id": 99, "login": "octocat"})
    with pytest.raises(ValueError):
        links.lookup("T1:U1", refresh=True)


def test_browser_routes_require_session_csrf_and_confirmation(configured):
    db = auth.store(auth.config())
    uid = db.user({"id": 42, "login": "octocat", "name": "Octocat"})
    session = db.new_session(uid, "browser", "Web")
    token, gid = grant()

    def call(headers):
        body = json.dumps(
            {"token": token, "login": "attacker", "user_id": 999}
        ).encode()
        h = Message()
        for key, value in {
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            **headers,
        }.items():
            h[key] = value
        replies = []
        handler = SimpleNamespace(
            path="/auth/slack/connect",
            headers=h,
            rfile=io.BytesIO(body),
            _send=lambda *args, **kwargs: replies.append((args, kwargs)),
        )
        assert links.handle(handler, "POST")
        return replies[0]

    valid = {
        "Origin": "https://dashboard.test",
        "X-TCI-Action": "1",
        "Cookie": f"{auth.COOKIE}={session}",
    }
    assert call({})[0][0] == 403
    assert call({"Origin": "https://dashboard.test", "X-TCI-Action": "1"})[0][0] == 401
    assert call({**valid, "Origin": "https://evil.test"})[0][0] == 403
    result = call(valid)
    assert result[0][0] == 200
    assert result[1]["headers"]["Referrer-Policy"] == "no-referrer"
    assert links.lookup("T1:U1") is None
    links.confirm("T1:U1", gid)
    assert links.lookup("T1:U1") == "octocat"


def test_failed_dm_delivery_can_be_retried_without_rebinding(configured, monkeypatch):
    token, gid = grant()
    original = links.slack_api.message
    monkeypatch.setattr(
        links.slack_api, "message", lambda *args: (_ for _ in ()).throw(TimeoutError())
    )
    with pytest.raises(TimeoutError):
        links.propose(token, person())
    assert links.lookup("T1:U1") is None
    monkeypatch.setattr(links.slack_api, "message", original)
    links.propose(token, person())
    assert "Connected" in links.confirm("T1:U1", gid)


def test_first_help_prompts_and_mutation_never_runs_unlinked(configured, monkeypatch):
    monkeypatch.setattr(slack_commands, "_POOL", SimpleNamespace(submit=lambda f: f()))
    calls = []
    fields = {"team_id": ["T1"], "user_id": ["U1"], "text": ["help"]}
    reply = slack_commands.queue(
        fields,
        b"help",
        lambda *args: calls.append(args),
        fallback=lambda text: "Help guide",
        onboarding=True,
    )
    assert "Help guide" in reply and "Connect GitHub" in reply
    fields["text"] = ["rerun 123"]
    assert "Connect GitHub" in slack_commands.queue(
        fields, b"rerun", lambda *args: calls.append(args), onboarding=True
    )
    assert calls == []
    for i in range(20):
        slack_commands.queue(
            fields, str(i).encode(), lambda *args: calls.append(args), onboarding=True
        )
    assert calls == []
    with slack_commands.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM slack_link_grants").fetchone()[0] == 20


def test_signed_action_routes_link_confirmation(configured, monkeypatch):
    from test_slack_bot import event_handler

    monkeypatch.setattr(slack_commands, "_POOL", SimpleNamespace(submit=lambda f: f()))
    token, gid = grant()
    links.propose(token, person())
    payload = {
        "type": "block_actions",
        "api_app_id": "A1",
        "team": {"id": "T1"},
        "user": {"id": "U1"},
        "channel": {"id": "DU1"},
        "actions": [{"action_id": "serge_link_confirm", "value": gid}],
    }
    handler, replies = event_handler(monkeypatch, payload, "/slack/actions")
    # event_handler uses a fixed clock; preserve grant expiry under that clock.
    assert slack_bot.handle(
        handler,
        lambda _: None,
        execute=lambda *args: pytest.fail("must not execute task"),
    )
    assert replies[0][0] == 200
    assert links.lookup("T1:U1") == "octocat"
