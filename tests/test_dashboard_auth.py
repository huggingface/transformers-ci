"""Exercise OAuth/session boundaries and the browser-to-CLI handoff."""

from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import pytest

from transformersci.otel import dashboard_auth as auth, trace_exporter


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setenv("TCI_AUTH_URL", "https://dashboard.test")
    monkeypatch.setenv("TCI_AUTH_GITHUB_CLIENT_ID", "app-client")
    monkeypatch.setenv("TCI_AUTH_GITHUB_CLIENT_SECRET", "app-secret")
    monkeypatch.setenv("TCI_AUTH_DATABASE", str(tmp_path / "auth.sqlite3"))
    monkeypatch.setenv("TCI_AUTH_ALLOWED_ORGS", "")
    return auth.store(auth.config())


def account(db, identifier=42, login="octocat"):
    return db.user({"id": identifier, "login": login, "name": "A <user>"})


def error(call, message):
    with pytest.raises(auth.AuthError) as caught:
        call()
    assert caught.value.error == message


def test_sessions_survive_restart_store_only_hashes_and_follow_stable_id(configured):
    db = configured
    token = db.new_session(account(db), "browser", "Web browser")
    assert db.identity(token, "browser")["user_id"] == 42
    assert db.identity(token, "cli") is None
    assert db.identity("wrong", "browser") is None
    assert token not in json.dumps(
        [tuple(row) for row in db.db.execute("SELECT * FROM sessions")]
    )
    account(db, login="renamed")
    restarted = auth.Store(auth.config().database)
    user = restarted.identity(token, "browser")
    assert user["login"] == "renamed"
    restarted.revoke(999, user["id"])
    assert restarted.identity(token, "browser")
    restarted.revoke(42, user["id"])
    assert db.identity(token, "browser") is None
    assert Path(auth.config().database).stat().st_mode & 0o777 == 0o600


def test_expired_session_is_rejected(configured, monkeypatch):
    token = configured.new_session(account(configured), "browser", "Browser")
    now = auth.time.time()
    monkeypatch.setattr(auth.time, "time", lambda: now + auth.SESSION_SECONDS + 1)
    assert configured.identity(token, "browser") is None


@pytest.mark.parametrize(
    "destination",
    ["https://evil.test", "//evil.test", "/\\evil", "/\nLocation:evil", "", None],
)
def test_return_paths_cannot_redirect_offsite(destination):
    assert auth.local_path(destination) == "/profile"


def test_oauth_state_is_bound_to_browser_single_use_and_expires(
    configured, monkeypatch
):
    state, browser, challenge = configured.start_login("/auth/cli?user_code=ABC")
    assert len(challenge) == 43
    error(
        lambda: configured.finish_login(state, "another browser"), "invalid_oauth_state"
    )
    assert (
        configured.finish_login(state, browser)["destination"]
        == "/auth/cli?user_code=ABC"
    )
    error(lambda: configured.finish_login(state, browser), "invalid_oauth_state")
    state, browser, _ = configured.start_login("/profile")
    now = auth.time.time()
    monkeypatch.setattr(auth.time, "time", lambda: now + 601)
    error(lambda: configured.finish_login(state, browser), "invalid_oauth_state")


def test_device_pending_throttle_approve_and_one_time_exchange(configured, monkeypatch):
    db = configured
    uid = account(db)
    grant = db.device()
    assert grant["device_code"] not in json.dumps(
        [tuple(row) for row in db.db.execute("SELECT * FROM devices")]
    )
    error(lambda: db.poll(grant["device_code"]), "authorization_pending")
    error(lambda: db.poll(grant["device_code"]), "slow_down")
    db.decide(uid, grant["user_code"].lower(), "approve")
    now = auth.time.time()
    monkeypatch.setattr(auth.time, "time", lambda: now + 11)
    result = db.poll(grant["device_code"])
    assert result["scope"] == auth.SCOPES
    assert db.identity(result["access_token"], "cli")["user_id"] == uid
    assert db.identity(result["access_token"], "browser") is None
    error(lambda: db.poll(grant["device_code"]), "expired_token")
    error(
        lambda: db.decide(uid, grant["user_code"], "approve"), "invalid_or_expired_code"
    )


def test_device_denial_expiry_and_rate_limits(configured, monkeypatch):
    db = configured
    uid = account(db)
    grant = db.device()
    db.decide(uid, grant["user_code"], "deny")
    error(lambda: db.poll(grant["device_code"]), "access_denied")
    now = auth.time.time()
    monkeypatch.setattr(auth.time, "time", lambda: now + 901)
    error(lambda: db.poll(grant["device_code"]), "expired_token")
    for _ in range(10):
        error(lambda: db.decide(uid, "unknown", "approve"), "invalid_or_expired_code")
    error(lambda: db.decide(uid, "unknown", "approve"), "rate_limited")


def test_concurrent_cli_exchange_mints_one_session(configured):
    db = configured
    uid = account(db)
    grant = db.device()
    db.decide(uid, grant["user_code"], "approve")

    def poll():
        try:
            return db.poll(grant["device_code"])
        except auth.AuthError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: poll(), range(2)))
    assert sum(result is not None for result in results) == 1
    assert (
        db.db.execute("SELECT count(*) FROM sessions WHERE kind='cli'").fetchone()[0]
        == 1
    )


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


@pytest.fixture
def http(configured):
    server = ThreadingHTTPServer(("127.0.0.1", 0), trace_exporter.MetricsHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()

    def call(path, *, body=None, headers=None):
        request = Request(
            f"http://127.0.0.1:{server.server_port}" + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        try:
            response = build_opener(NoRedirect()).open(request)
        except HTTPError as error:
            response = error
        payload = response.read()
        result = (
            json.loads(payload)
            if response.headers.get_content_type() == "application/json"
            else payload.decode()
        )
        return response.code, result, response.headers

    yield call
    server.shutdown()
    server.server_close()
    worker.join()


def browser(db, uid=42):
    token = db.new_session(account(db, uid), "browser", "Browser")
    return {
        "Cookie": f"{auth.COOKIE}={token}",
        "Origin": "https://dashboard.test",
        "X-TCI-Action": "1",
    }


def test_http_login_pkce_callback_and_profile(http, configured, monkeypatch):
    status, _, headers = http("/auth/github?next=%2Fprofile")
    assert status == 302
    query = parse_qs(urlsplit(headers["Location"]).query)
    assert query["client_id"] == ["app-client"]
    assert query["code_challenge_method"] == ["S256"]
    assert query["redirect_uri"] == ["https://dashboard.test/auth/callback"]
    assert "HttpOnly" in headers["Set-Cookie"] and "Secure" in headers["Set-Cookie"]
    state_cookie = headers["Set-Cookie"].split(";", 1)[0]
    calls = []

    def github(url, **kwargs):
        calls.append((url, kwargs))
        return (
            {"access_token": "github-token"}
            if "access_token" in url
            else {"id": 42, "login": "octocat", "name": "Octocat"}
        )

    monkeypatch.setattr(auth, "github_json", github)
    callback = "/auth/callback?state=" + query["state"][0] + "&code=github-code"
    assert http(callback)[0] == 400  # stolen state without browser cookie
    status, _, headers = http(callback, headers={"Cookie": state_cookie})
    assert status == 302 and headers["Location"] == "/profile"
    assert calls[0][1]["data"]["code_verifier"]
    session = headers["Set-Cookie"].split(";", 1)[0]
    status, profile, headers = http("/auth/me", headers={"Cookie": session})
    assert status == 200 and profile["id"] == 42
    assert "github-token" not in json.dumps(profile)
    assert headers["Cache-Control"] == "no-store"
    assert headers["Referrer-Policy"] == "no-referrer"
    assert http(callback, headers={"Cookie": state_cookie})[0] == 400


def test_http_device_login_revocation_and_csrf(http, configured):
    status, grant, _ = http(
        "/auth/device", body={"client_id": "tmic", "scope": "admin"}
    )
    assert status == 200
    assert grant["verification_uri_complete"].startswith(
        "https://dashboard.test/auth/cli?"
    )
    headers = browser(configured)
    body = {"user_code": grant["user_code"], "decision": "approve"}
    assert http("/auth/cli/decision", body=body)[0] == 403
    assert (
        http(
            "/auth/cli/decision",
            body=body,
            headers={**headers, "Origin": "https://evil.test"},
        )[0]
        == 403
    )
    assert (
        http(
            "/auth/cli/decision",
            body=body,
            headers={"Origin": "https://dashboard.test", "X-TCI-Action": "1"},
        )[0]
        == 401
    )
    assert http("/auth/cli/decision", body=body, headers=headers)[0] == 200
    status, result, _ = http(
        "/auth/token",
        body={
            "device_code": grant["device_code"],
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        },
    )
    assert status == 200 and "admin" not in result["scope"]
    bearer = {"Authorization": "Bearer " + result["access_token"]}
    status, profile, _ = http("/auth/me", headers=bearer)
    assert status == 200 and profile["login"] == "octocat" and "sessions" not in profile
    assert auth.browser_login(bearer) is None
    # No CLI credential may authorize existing browser write actions.
    assert (
        http(
            "/auth/logout",
            body={},
            headers={**bearer, "Origin": "https://dashboard.test", "X-TCI-Action": "1"},
        )[0]
        == 401
    )
    assert http("/auth/revoke", body={}, headers=bearer)[0] == 200
    assert http("/auth/me", headers=bearer)[0] == 401


def test_profile_sessions_are_private_and_revoke_only_owned_sessions(http, configured):
    alice, bob = browser(configured), browser(configured, 43)
    _, profile, _ = http("/auth/me", headers=alice)
    _, other, _ = http("/auth/me", headers=bob)
    assert len(profile["sessions"]) == 1
    assert (
        http(
            "/auth/sessions/revoke",
            body={"session_id": other["session_id"]},
            headers=alice,
        )[0]
        == 200
    )
    assert http("/auth/me", headers=bob)[0] == 200
    status, _, headers = http("/auth/logout", body={}, headers=alice)
    assert status == 200 and "Max-Age=0" in headers["Set-Cookie"]
    assert http("/auth/me", headers=alice)[0] == 401


def test_grafana_cookie_does_not_authenticate_and_profile_is_standalone(
    http, monkeypatch
):
    assert http("/auth/me", headers={"Cookie": "grafana_session=anything"})[0] == 401
    for path in ("/profile", "/auth/cli?user_code=ABC", "/profile/app.js"):
        status, page, headers = http(path)
        assert status == 200 and headers["X-Frame-Options"] == "DENY"
        assert "grafana" not in page.lower()
    monkeypatch.delenv("TCI_AUTH_GITHUB_CLIENT_SECRET")
    assert http("/auth/github")[0] == 503
    assert http("/profile")[0] == 200


def test_config_rejects_insecure_or_non_origin_urls(monkeypatch, configured):
    for url in (
        "http://dashboard.test",
        "https://evil.test/path",
        "https://user@dashboard.test",
    ):
        monkeypatch.setenv("TCI_AUTH_URL", url)
        error(auth.config, "invalid_auth_configuration")


def test_membership_gate_is_active_only_and_accepts_explicit_account_ids(
    monkeypatch, configured
):
    monkeypatch.setenv("TCI_AUTH_ALLOWED_ORGS", "huggingface")
    cfg = auth.config()
    user = {"id": 42, "login": "octocat"}
    urls = []

    def membership(state):
        def github(url, **kw):
            urls.append(url)
            return {"state": state, "organization": {"login": "huggingface"}}

        return github

    monkeypatch.setattr(auth, "github_json", membership("pending"))
    error(lambda: auth.require_membership(cfg, user, "token"), "account_not_allowed")
    monkeypatch.setattr(auth, "github_json", membership("active"))
    auth.require_membership(cfg, user, "token")
    assert urls[-1] == "https://api.github.com/user/memberships/orgs/huggingface"
    monkeypatch.setenv("TCI_AUTH_ALLOWED_USER_IDS", "42")
    monkeypatch.setattr(
        auth,
        "github_json",
        lambda *a, **kw: pytest.fail("explicit grant should not request membership"),
    )
    auth.require_membership(auth.config(), user, "token")


def test_membership_gate_separates_non_members_from_missing_app_permission(
    monkeypatch, configured
):
    monkeypatch.setenv("TCI_AUTH_ALLOWED_ORGS", "huggingface")
    cfg = auth.config()
    user = {"id": 42, "login": "octocat"}

    def status(code):
        def github(url, **kw):
            raise HTTPError(url, code, "", {}, None)

        return github

    monkeypatch.setattr(auth, "github_json", status(404))
    error(lambda: auth.require_membership(cfg, user, "token"), "account_not_allowed")
    monkeypatch.setattr(auth, "github_json", status(403))
    error(
        lambda: auth.require_membership(cfg, user, "token"),
        "github_app_cannot_read_membership",
    )
    monkeypatch.setattr(auth, "github_json", status(500))
    error(
        lambda: auth.require_membership(cfg, user, "token"),
        "github_membership_check_failed",
    )


def test_github_client_never_follows_token_redirects():
    error(
        lambda: auth.NoRedirect().redirect_request(
            None, None, 302, "", {}, "https://evil.test"
        ),
        "unexpected_github_redirect",
    )
