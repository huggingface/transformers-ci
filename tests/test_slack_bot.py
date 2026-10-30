import hashlib
import hmac
import io
import json
from email.message import Message
from types import SimpleNamespace
from urllib.parse import urlencode

import pytest

from transformersci.otel import slack_bot, slack_commands, rerun_actions


def signed_handler(
    monkeypatch, *, team="T1", text="help", timestamp="1000", signature=None
):
    monkeypatch.setenv("TCI_SLACK_SIGNING_SECRET", "secret")
    monkeypatch.setenv("TCI_SLACK_TEAM_IDS", "T1")
    monkeypatch.setenv("TCI_SLACK_PUBLIC_URL", "https://dashboard.example")
    monkeypatch.setattr(slack_bot.time, "time", lambda: 1000)
    body = urlencode(
        dict(team_id=team, user_id="U1", command="/askserge", text=text)
    ).encode()
    headers = Message()
    headers["Content-Type"] = "application/x-www-form-urlencoded"
    headers["Content-Length"] = str(len(body))
    headers["X-Slack-Request-Timestamp"] = timestamp
    headers["X-Slack-Signature"] = (
        signature
        or "v0="
        + hmac.new(
            b"secret", b"v0:" + timestamp.encode() + b":" + body, hashlib.sha256
        ).hexdigest()
    )
    replies = []
    return SimpleNamespace(
        path="/slack/commands",
        headers=headers,
        rfile=io.BytesIO(body),
        _reply_json=lambda *args: replies.append(args),
    ), replies


@pytest.mark.parametrize(
    "overrides,code",
    [
        ({}, 200),
        ({"signature": "v0=bad"}, 401),
        ({"timestamp": "1"}, 401),
        ({"timestamp": "nonsense"}, 401),
        ({"team": "T2"}, 403),
    ],
)
def test_signed_requests(monkeypatch, overrides, code):
    handler, replies = signed_handler(monkeypatch, **overrides)
    assert slack_bot.handle(handler, lambda pr: None)
    assert replies[0][0] == code
    if code == 200:
        assert replies[0][1]["response_type"] == "ephemeral"


def test_disabled_and_body_limits(monkeypatch):
    handler, replies = signed_handler(monkeypatch)
    monkeypatch.delenv("TCI_SLACK_TEAM_IDS")
    slack_bot.handle(handler, lambda pr: None)
    assert replies[0][0] == 503
    handler, replies = signed_handler(monkeypatch)
    handler.headers.replace_header("Content-Length", "70000")
    slack_bot.handle(handler, lambda pr: None)
    assert replies[0][0] == 400


@pytest.fixture
def commands(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_RUN_STORE", str(tmp_path))
    monkeypatch.setenv("TCI_SLACK_ACTORS", json.dumps({"T1:U1": "maintainer"}))
    monkeypatch.setattr(slack_commands, "_POOL", SimpleNamespace(submit=lambda f: f()))
    return {"team_id": ["T1"], "user_id": ["U1"], "text": ["rerun 42"]}


def test_queue_deduplication_owner_and_identity(commands):
    calls = []

    def run(*args):
        calls.append(args)
        return "started"

    first = slack_commands.queue(commands, b"request", run)
    assert "accepted" in first
    assert "started" in slack_commands.queue(commands, b"request", run)
    assert len(calls) == 1
    assert calls[0][:3] == ("rerun", "42", "maintainer")
    key = calls[0][3]
    commands["text"] = [f"result {key}"]
    assert slack_commands.queue(commands, b"poll", run) == "started"
    commands["user_id"] = ["U2"]
    assert "not found" in slack_commands.queue(commands, b"poll", run)
    commands["text"] = ["investigate 42"]
    assert "no configured GitHub identity" in slack_commands.queue(
        commands, b"new", run
    )
    assert len(calls) == 1


def test_queue_rate_limit_and_failures(commands):
    def broken(*args):
        raise RuntimeError("internal secret")

    for i in range(20):
        assert "accepted" in slack_commands.queue(commands, str(i).encode(), broken)
    assert "limit reached" in slack_commands.queue(commands, b"21", broken)
    with slack_commands.connect() as db:
        assert (
            "internal secret"
            not in db.execute("SELECT result FROM commands").fetchone()[0]
        )


def test_rerun_confirmation_and_shared_path(monkeypatch):
    monkeypatch.setattr(rerun_actions, "dispatch_enabled", lambda: (True, ""))
    seen = []

    def submit(body, actor, reply):
        seen.append((body, actor))
        reply(409, {"status": "confirm"})

    result = slack_commands.execute(
        "rerun",
        "42",
        "alice",
        "a" * 24,
        summary=lambda pr: None,
        submit_rerun=submit,
        dashboard="https://dashboard",
        serge="https://serge",
    )
    assert "Review and confirm" in result
    assert "confirm_active" not in seen[0][0]
    assert seen[0][1] == "alice"


def test_investigation_dispatch(monkeypatch):
    monkeypatch.setattr(rerun_actions, "dispatch_enabled", lambda: (True, ""))
    calls = []

    def call(operation, body):
        calls.append((operation, body))
        return 200, {"url": "/reviews/huggingface/transformers/42/abcd"}

    monkeypatch.setattr(
        rerun_actions, "default_serge", lambda: SimpleNamespace(call=call)
    )
    result = slack_commands.execute(
        "investigate",
        "42",
        "alice",
        "a" * 24,
        summary=lambda pr: None,
        submit_rerun=None,
        dashboard="https://dashboard",
        serge="https://serge",
    )
    assert "https://serge/reviews/" in result
    assert calls[0][1]["actor"] == "alice"


def test_pr_snapshot_does_not_claim_ci_success():
    reply = slack_bot.command_reply(
        "pr 42",
        lambda pr: {"total_tests": 30, "failed_tests": 2},
        "https://dashboard",
        "",
    )
    assert "2 failed / 30" in reply
    assert "not the overall CI conclusion" in reply


def event_handler(monkeypatch, payload, path="/slack/events"):
    handler, replies = signed_handler(monkeypatch)
    monkeypatch.setenv("TCI_SLACK_APP_ID", "A1")
    monkeypatch.setenv("TCI_SLACK_BOT_USER_ID", "UBOT")
    handler.path = path
    raw = (
        json.dumps(payload).encode()
        if path == "/slack/events"
        else urlencode({"payload": json.dumps(payload)}).encode()
    )
    handler.rfile = io.BytesIO(raw)
    handler.headers.replace_header(
        "Content-Type",
        "application/json"
        if path == "/slack/events"
        else "application/x-www-form-urlencoded",
    )
    handler.headers.replace_header("Content-Length", str(len(raw)))
    handler.headers.replace_header(
        "X-Slack-Signature",
        "v0=" + hmac.new(b"secret", b"v0:1000:" + raw, hashlib.sha256).hexdigest(),
    )
    return handler, replies


def test_mentions_and_dms_start_once_and_reply_privately(monkeypatch, commands):
    from transformersci.otel import slack_api

    messages, investigations = [], []
    monkeypatch.setattr(slack_api, "open_dm", lambda user: "DPRIVATE")
    monkeypatch.setattr(
        slack_api, "message", lambda channel, text: messages.append((channel, text))
    )
    payload = {
        "api_app_id": "A1",
        "team_id": "T1",
        "type": "event_callback",
        "event_id": "Ev1",
        "event": {
            "type": "app_mention",
            "user": "U1",
            "channel": "CPUBLIC",
            "text": "<@UBOT> look at the traceback: ValueError: boom",
        },
    }

    def investigate(*args):
        investigations.append(args)
        return "started"

    for _ in range(2):
        handler, replies = event_handler(monkeypatch, payload)
        slack_bot.handle(handler, lambda pr: None, lambda *args: "ok", investigate)
        assert replies == [(200, {})]
    assert len(investigations) == 1
    assert investigations[0][:3] == ("T1:U1", "U1", "maintainer")
    assert investigations[0][-1] == "ValueError: boom"
    assert messages == [("DPRIVATE", "started")]
    payload["event_id"] = "Ev2"
    payload["event"] = {
        "type": "message",
        "channel_type": "im",
        "user": "U1",
        "text": "help",
    }
    handler, _ = event_handler(monkeypatch, payload)
    slack_bot.handle(handler, lambda pr: None, lambda *args: "ok", investigate)
    assert "look at the traceback" in messages[-1][1]


def test_signed_url_challenge_and_wrong_app(monkeypatch):
    handler, replies = event_handler(
        monkeypatch, {"type": "url_verification", "challenge": "proof"}
    )
    slack_bot.handle(handler, lambda pr: None)
    assert replies == [(200, {"challenge": "proof"})]
    handler, replies = event_handler(
        monkeypatch, {"api_app_id": "OTHER", "team_id": "T1", "event": {}}
    )
    slack_bot.handle(handler, lambda pr: None)
    assert replies[0][0] == 403


def test_link_extraction_never_fetches_external_destination(monkeypatch):
    from transformersci.otel import slack_investigations as si

    monkeypatch.setenv("TCI_SLACK_PUBLIC_URL", "https://dashboard.example")
    calls = []

    def failure(trace, nodeid):
        calls.append((trace, nodeid))
        return {
            "nodeid": nodeid,
            "exception_type": "ValueError",
            "exception_message": "boom",
            "exception_stacktrace": "Traceback",
        }

    with pytest.raises(ValueError, match="this dashboard"):
        si.context_for("https://private.internal/secret", failure)
    assert not calls
    trace = "a" * 32
    result = si.context_for(
        f"<https://dashboard.example/test?trace_id={trace}&test_nodeid=tests/x.py::test_x|trace>",
        failure,
    )
    assert calls == [(trace, "tests/x.py::test_x")]
    assert "ValueError" in result and "Traceback" in result


def test_phase_poll_and_approval_are_durable_and_owned(monkeypatch, commands):
    from transformersci.otel import slack_investigations as si

    messages, calls = [], []
    monkeypatch.setattr(si.slack_api, "message", lambda *args: messages.append(args))

    def call(operation, body):
        calls.append((operation, body))
        return 200, {
            "status": "awaiting_approval",
            "steps": [{"seq": 1, "text": "clone"}, {"seq": 2, "text": "verify"}],
            "result": {
                "branch": "serge/slack-abcd",
                "verify_verdict": "fixed",
                "proposal": {"title": "Fix", "body": "Because"},
            },
        }

    monkeypatch.setattr(
        rerun_actions, "default_serge", lambda: SimpleNamespace(call=call)
    )
    with slack_commands.connect() as db:
        si.init(db)
        db.execute(
            "INSERT INTO investigations(id,owner,actor,channel,job,state,created) VALUES (?,?,?,?,?,?,?)",
            ("a" * 24, "T1:U1", "maintainer", "D1", "b" * 32, "running", 1000),
        )
    si.poll_once()
    assert len(messages) == 3
    assert "Would you like" in messages[-1][1]
    assert messages[-1][2][1]["elements"][0]["action_id"] == "serge_approve"
    si.poll_once()
    assert len(messages) == 3
    assert "another" in si.decide("T1:U2", "maintainer", "D1", "a" * 24, "approve")
    assert "another" in si.decide("T1:U1", "maintainer", "DOTHER", "a" * 24, "approve")
    assert len(calls) == 1


def test_dashboard_fix_it_uses_authenticated_mapping(monkeypatch, commands):
    from transformersci.otel import trace_exporter as te

    queued = []
    monkeypatch.setenv("TCI_SLACK_BOT_TOKEN", "fake")
    monkeypatch.setenv("TCI_SLACK_TEAM_IDS", "T1")
    monkeypatch.setenv("TCI_SLACK_PUBLIC_URL", "https://dashboard.example")
    monkeypatch.setattr(
        slack_commands,
        "queue",
        lambda fields, raw, execute, **kwargs: (
            queued.append(fields) or "Command accepted."
        ),
    )
    replies = []
    body = json.dumps(
        {
            "trace_id": "a" * 32,
            "test_nodeid": "tests/x.py::test_x",
            "idempotency_key": "12345678",
        }
    ).encode()
    handler = SimpleNamespace(
        headers={"Content-Length": str(len(body))},
        rfile=io.BytesIO(body),
        _action_allowed=lambda: True,
        _action_user=lambda: "maintainer",
        _reply_json=lambda *args: replies.append(args),
    )
    te.MetricsHandler._serve_fix_it(handler)
    assert replies[0][0] == 202
    assert queued[0]["user_id"] == ["U1"]
    assert "trace_id=" in queued[0]["text"][0]
    handler._action_user = lambda: "unmapped"
    handler.rfile = io.BytesIO(body)
    te.MetricsHandler._serve_fix_it(handler)
    assert replies[-1][0] == 409
    assert len(queued) == 1
    handler._action_allowed = lambda: False
    te.MetricsHandler._serve_fix_it(handler)
    assert len(queued) == 1


def test_pr_status_counts_current_head_checks():
    seen = []

    def github(path):
        seen.append(path)
        if path == "pulls/42":
            return {"state": "open", "head": {"sha": "a" * 40}}
        if "check-runs" in path:
            return {
                "check_runs": [
                    {"status": "completed", "conclusion": "failure"},
                    {"status": "in_progress"},
                ],
                "total_count": 2,
            }
        return {"statuses": [{"state": "success"}], "total_count": 1}

    result = slack_commands.pr_status("42", github, "https://dashboard")
    assert "1 passed, 1 failed, 1 pending" in result
    assert all("a" * 40 in path for path in seen[1:])


def test_lost_dispatch_response_keeps_a_recoverable_job(monkeypatch, commands):
    from transformersci.otel import slack_investigations as si

    monkeypatch.setattr(si.slack_api, "open_dm", lambda user: "D1")
    monkeypatch.setattr(si.slack_api, "message", lambda *args: {"ts": "123.456"})
    monkeypatch.setattr(si.slack_api, "session", lambda *args: None)
    monkeypatch.setattr(si, "start", lambda: None)
    monkeypatch.setattr(rerun_actions, "serge_url", lambda: "http://serge")
    monkeypatch.setattr(rerun_actions, "serge_token", lambda: "fake")

    def lost(*args):
        raise TimeoutError

    monkeypatch.setattr(
        rerun_actions, "default_serge", lambda: SimpleNamespace(call=lost)
    )
    result = si.begin(
        "T1:U1", "U1", "maintainer", "a" * 24, "ValueError: boom", lambda *args: None
    )
    assert "checking its status" in result
    with slack_commands.connect() as db:
        job, state = db.execute("SELECT job,state FROM investigations").fetchone()
    assert len(job) == 32 and state == "running"


def test_yes_no_requires_one_pending_investigation_in_same_dm(commands):
    import time
    from transformersci.otel import slack_investigations as si

    with slack_commands.connect() as db:
        si.init(db)
        db.execute(
            "INSERT INTO investigations(id,owner,actor,channel,job,state,created) VALUES (?,?,?,?,?,?,?)",
            (
                "a" * 24,
                "T1:U1",
                "maintainer",
                "D1",
                "b" * 32,
                "awaiting_approval",
                time.time(),
            ),
        )
    assert si.pending_decision("T1:U1", "D1", "yes") == f"decision {'a' * 24} approve"
    assert si.pending_decision("T1:U1", "D1", "no") == f"decision {'a' * 24} decline"
    assert si.pending_decision("T1:U2", "D1", "yes") is None
    assert si.pending_decision("T1:U1", "D2", "yes") is None
    with slack_commands.connect() as db:
        db.execute(
            "INSERT INTO investigations(id,owner,actor,channel,job,state,created) VALUES (?,?,?,?,?,?,?)",
            (
                "c" * 24,
                "T1:U1",
                "maintainer",
                "D1",
                "d" * 32,
                "awaiting_approval",
                time.time(),
            ),
        )
    assert si.pending_decision("T1:U1", "D1", "yes") is None


def test_native_sessions_retry_heartbeat_and_close(monkeypatch, commands):
    from transformersci.otel import slack_investigations as si

    now = [1000.0]
    monkeypatch.setattr(si.time, "time", lambda: now[0])
    calls = []
    fail = [True]

    def session(*args):
        calls.append(args)
        if fail[0]:
            raise TimeoutError

    monkeypatch.setattr(si.slack_api, "session", session)
    with slack_commands.connect() as db:
        si.init(db)
        db.execute(
            "INSERT INTO investigations(id,owner,actor,channel,job,state,created,thread_ts) VALUES (?,?,?,?,?,?,?,?)",
            (
                "a" * 24,
                "T1:U1",
                "maintainer",
                "D1",
                "b" * 32,
                "running",
                now[0],
                "123.456",
            ),
        )
    si.sync_sessions()
    fail[0] = False
    si.sync_sessions()
    si.sync_sessions()
    assert len(calls) == 2
    assert calls[-1] == ("D1", "123.456", "processing", "U1")
    now[0] += 1201
    si.sync_sessions()
    assert len(calls) == 3
    with slack_commands.connect() as db:
        db.execute("UPDATE investigations SET state='awaiting_approval'")
    si.sync_sessions()
    assert calls[-1][2] == "suspended"
    now[0] += 86400
    si.sync_sessions()
    assert calls[-1][2] == "closed"


def test_approval_replies_are_scoped_to_threads(commands):
    import time
    from transformersci.otel import slack_investigations as si

    with slack_commands.connect() as db:
        si.init(db)
        for key, thread in [("a", "1.1"), ("b", "2.2")]:
            db.execute(
                "INSERT INTO investigations(id,owner,actor,channel,state,created,thread_ts) VALUES (?,?,?,?,?,?,?)",
                (
                    key * 24,
                    "T1:U1",
                    "maintainer",
                    "D1",
                    "awaiting_approval",
                    time.time(),
                    thread,
                ),
            )
    assert si.pending_decision("T1:U1", "D1", "yes") is None
    assert (
        si.pending_decision("T1:U1", "D1", "yes", "1.1")
        == f"decision {'a' * 24} approve"
    )
    assert (
        si.pending_decision("T1:U1", "D1", "no", "2.2")
        == f"decision {'b' * 24} decline"
    )
    assert si.pending_decision("T1:U2", "D1", "yes", "1.1") is None


def test_native_thread_updates_and_terminal_state(monkeypatch, commands):
    import time
    from transformersci.otel import slack_investigations as si

    messages, statuses = [], []
    monkeypatch.setattr(
        si.slack_api, "message", lambda *args, **kw: messages.append((args, kw))
    )
    monkeypatch.setattr(si.slack_api, "session", lambda *args: statuses.append(args))
    monkeypatch.setattr(
        rerun_actions,
        "default_serge",
        lambda: SimpleNamespace(
            call=lambda *args: (
                200,
                {
                    "status": "no_fix",
                    "steps": [{"seq": 1, "text": "verify"}],
                    "result": {"message": "No verified fix"},
                },
            )
        ),
    )
    with slack_commands.connect() as db:
        si.init(db)
        db.execute(
            "INSERT INTO investigations(id,owner,actor,channel,job,state,created,thread_ts) VALUES (?,?,?,?,?,?,?,?)",
            (
                "a" * 24,
                "T1:U1",
                "maintainer",
                "D1",
                "b" * 32,
                "running",
                time.time(),
                "123.456",
            ),
        )
    si.poll_once()
    si.poll_once()
    assert len(messages) == 2
    assert all(kw == {"thread_ts": "123.456"} for _, kw in messages)
    assert "No verified fix" in messages[-1][0][1]
    assert [call[2] for call in statuses] == ["processing", "closed"]
    si.reply("T1:U1", "a" * 24, "U1", "receipt", channel="D1")
    assert messages[-1][1] == {"thread_ts": "123.456"}


def test_native_schema_migrates_existing_investigations(commands):
    from transformersci.otel import slack_investigations as si

    with slack_commands.connect() as db:
        db.execute(
            "CREATE TABLE investigations (id TEXT PRIMARY KEY, owner TEXT, actor TEXT, channel TEXT, job TEXT, cursor INTEGER DEFAULT -1, state TEXT, created REAL)"
        )
        db.execute("INSERT INTO investigations(id,state) VALUES ('old','running')")
        si.init(db)
        si.init(db)
        assert db.execute(
            "SELECT id,thread_ts,session_status,session_updated FROM investigations"
        ).fetchone() == ("old", "", "", 0)
