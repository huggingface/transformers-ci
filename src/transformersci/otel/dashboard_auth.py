"""Dashboard-owned GitHub App login and revocable browser/CLI sessions.

No Grafana dependency. GitHub credentials are used only during login; opaque
dashboard credentials are stored as hashes. GPU authorization remains a separate
repository-permission check at submission time.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from http.cookies import SimpleCookie, CookieError
from pathlib import Path
import re
import secrets
import sqlite3
import threading
import time
from urllib.error import HTTPError
from urllib.parse import parse_qs, quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, ProxyHandler

COOKIE = "__Host-tci_session"
STATE_COOKIE = "__Host-tci_oauth"
SESSION_SECONDS = 8 * 3600
DEVICE_SECONDS = 900
SCOPES = "profile:read mic:read mic:submit"


class AuthError(Exception):
    def __init__(self, status: int, error: str):
        self.status, self.error = status, error
        super().__init__(error)


@dataclass(frozen=True)
class Config:
    url: str
    client_id: str
    client_secret: str
    database: str
    allowed_orgs: tuple[str, ...] = ("huggingface",)
    allowed_user_ids: tuple[int, ...] = ()

    @property
    def callback(self):
        return self.url + "/auth/callback"


def config():
    url = os.getenv("TCI_AUTH_URL", "").rstrip("/")
    client = os.getenv("TCI_AUTH_GITHUB_CLIENT_ID", "")
    secret = os.getenv("TCI_AUTH_GITHUB_CLIENT_SECRET", "")
    root = os.getenv("PYTEST_TRACE_EXPORTER_RUN_STORE", "")
    database = os.getenv("TCI_AUTH_DATABASE", "") or (
        str(Path(root) / "auth.sqlite3") if root else ""
    )
    if not all((url, client, secret, database)):
        return None
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.path
        or parsed.query
        or parsed.fragment
        or parsed.username
    ):
        raise AuthError(503, "invalid_auth_configuration")
    orgs = tuple(
        value.strip().lower()
        for value in os.getenv("TCI_AUTH_ALLOWED_ORGS", "huggingface").split(",")
        if value.strip()
    )
    try:
        users = tuple(
            int(value.strip())
            for value in os.getenv("TCI_AUTH_ALLOWED_USER_IDS", "").split(",")
            if value.strip()
        )
    except ValueError:
        raise AuthError(503, "invalid_auth_configuration") from None
    if any(not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,38}", org) for org in orgs):
        raise AuthError(503, "invalid_auth_configuration")
    return Config(url, client, secret, database, orgs, users)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def cookie(headers, name):
    try:
        cookies = SimpleCookie(headers.get("Cookie", ""))
        return cookies[name].value if name in cookies else ""
    except CookieError:
        return ""


def cookie_value(name, value, age):
    return f"{name}={value}; Path=/; Secure; HttpOnly; SameSite=Lax; Max-Age={age}"


def local_path(value):
    # Never redirect to an origin supplied by a request, including //evil,
    # backslash URLs and header control characters.
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or value.startswith("//")
        or re.search(r"[\\\x00-\x20\x7f]", value)
    ):
        return "/profile"
    return value


class Store:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        os.chmod(path, 0o600)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY, login TEXT NOT NULL, name TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, token TEXT UNIQUE NOT NULL,
                user_id INTEGER NOT NULL, kind TEXT NOT NULL, label TEXT NOT NULL,
                scope TEXT NOT NULL, created REAL NOT NULL, expires REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS oauth (
                state TEXT PRIMARY KEY, browser TEXT NOT NULL, verifier TEXT NOT NULL,
                destination TEXT NOT NULL, expires REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS devices (
                token TEXT PRIMARY KEY, code TEXT UNIQUE NOT NULL,
                created REAL NOT NULL, expires REAL NOT NULL,
                last_poll REAL NOT NULL DEFAULT 0, interval INTEGER NOT NULL DEFAULT 5,
                decision TEXT NOT NULL DEFAULT 'pending', user_id INTEGER
            );
            CREATE TABLE IF NOT EXISTS attempts (
                bucket TEXT NOT NULL, created REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS attempts_bucket ON attempts(bucket, created);
        """)

    def limit(self, bucket, maximum, window=60):
        now = time.time()
        with self.lock:
            self.db.execute("DELETE FROM attempts WHERE created < ?", (now - 3600,))
            count = self.db.execute(
                "SELECT count(*) FROM attempts WHERE bucket=? AND created>?",
                (bucket, now - window),
            ).fetchone()[0]
            if count >= maximum:
                raise AuthError(429, "rate_limited")
            self.db.execute("INSERT INTO attempts VALUES (?, ?)", (bucket, now))
            for table in ("oauth", "devices", "sessions"):
                self.db.execute(f"DELETE FROM {table} WHERE expires < ?", (now - 3600,))

    def new_session(self, user_id, kind, label):
        token = secrets.token_urlsafe(32)
        now = time.time()
        scope = SCOPES if kind == "cli" else "browser"
        with self.lock:
            self.db.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    secrets.token_hex(16),
                    digest(token),
                    user_id,
                    kind,
                    label,
                    scope,
                    now,
                    now + SESSION_SECONDS,
                ),
            )
        return token

    def identity(self, token, kind):
        with self.lock:
            row = self.db.execute(
                """SELECT s.*, u.login, u.name FROM sessions s
                JOIN users u ON s.user_id=u.id WHERE s.token=? AND s.kind=? AND s.expires>?""",
                (digest(token), kind, time.time()),
            ).fetchone()
            return dict(row) if row else None

    def start_login(self, destination):
        state, browser, verifier = (secrets.token_urlsafe(32) for _ in range(3))
        with self.lock:
            self.db.execute(
                "INSERT INTO oauth VALUES (?, ?, ?, ?, ?)",
                (
                    digest(state),
                    digest(browser),
                    verifier,
                    local_path(destination),
                    time.time() + 600,
                ),
            )
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )
        return state, browser, challenge

    def finish_login(self, state, browser):
        with self.lock:
            row = self.db.execute(
                "SELECT * FROM oauth WHERE state=?", (digest(state),)
            ).fetchone()
            if (
                not row
                or row["expires"] <= time.time()
                or not secrets.compare_digest(row["browser"], digest(browser))
            ):
                raise AuthError(400, "invalid_oauth_state")
            self.db.execute("DELETE FROM oauth WHERE state=?", (digest(state),))
            return dict(row)

    def user(self, data):
        if (
            type(data.get("id")) is not int
            or data["id"] <= 0
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}", data.get("login", ""))
        ):
            raise AuthError(502, "invalid_github_identity")
        with self.lock:
            self.db.execute(
                "INSERT INTO users VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET login=excluded.login, name=excluded.name",
                (data["id"], data["login"], str(data.get("name") or "")[:200]),
            )
        return data["id"]

    def device(self):
        self.limit("device-start", 60)
        token = secrets.token_urlsafe(32)
        code = "-".join(secrets.token_hex(3).upper() for _ in range(2))
        now = time.time()
        with self.lock:
            self.db.execute(
                "INSERT INTO devices(token, code, created, expires) VALUES (?, ?, ?, ?)",
                (digest(token), code, now, now + DEVICE_SECONDS),
            )
        return {
            "device_code": token,
            "user_code": code,
            "expires_in": DEVICE_SECONDS,
            "interval": 5,
        }

    def decide(self, user_id, code, decision):
        self.limit(f"decision:{user_id}", 10)
        if decision not in ("approve", "deny"):
            raise AuthError(400, "invalid_decision")
        with self.lock:
            cursor = self.db.execute(
                "UPDATE devices SET decision=?, user_id=? WHERE code=? AND decision='pending' AND expires>?",
                (decision, user_id, str(code).upper(), time.time()),
            )
            if cursor.rowcount != 1:
                raise AuthError(400, "invalid_or_expired_code")

    def poll(self, token):
        now = time.time()
        with self.lock:
            row = self.db.execute(
                "SELECT * FROM devices WHERE token=?", (digest(token),)
            ).fetchone()
            if not row or row["expires"] <= now or row["decision"] == "consumed":
                raise AuthError(400, "expired_token")
            if now < row["last_poll"] + row["interval"]:
                self.db.execute(
                    "UPDATE devices SET interval=interval+5, last_poll=? WHERE token=?",
                    (now, digest(token)),
                )
                raise AuthError(400, "slow_down")
            self.db.execute(
                "UPDATE devices SET last_poll=? WHERE token=?", (now, digest(token))
            )
            if row["decision"] == "pending":
                raise AuthError(400, "authorization_pending")
            if row["decision"] == "deny":
                raise AuthError(400, "access_denied")
            self.db.execute(
                "UPDATE devices SET decision='consumed' WHERE token=?", (digest(token),)
            )
            access = self.new_session(row["user_id"], "cli", "tmic")
            return {
                "access_token": access,
                "token_type": "Bearer",
                "expires_in": SESSION_SECONDS,
                "scope": SCOPES,
            }

    def profile(self, identity):
        with self.lock:
            sessions = [
                dict(row)
                for row in self.db.execute(
                    "SELECT id, kind, label, scope, created, expires FROM sessions WHERE user_id=? AND expires>? ORDER BY created DESC",
                    (identity["user_id"], time.time()),
                )
            ]
        return {
            "id": identity["user_id"],
            "login": identity["login"],
            "name": identity["name"],
            "github_url": "https://github.com/" + identity["login"],
            "session_id": identity["id"],
            "scope": identity["scope"],
            "sessions": sessions,
        }

    def revoke(self, user_id, session_id):
        with self.lock:
            self.db.execute(
                "DELETE FROM sessions WHERE user_id=? AND id=?", (user_id, session_id)
            )


_stores = {}
_stores_lock = threading.Lock()


def store(cfg):
    with _stores_lock:
        if cfg.database not in _stores:
            _stores[cfg.database] = Store(cfg.database)
        return _stores[cfg.database]


def identity(headers, *, cli=False):
    cfg = config()
    if not cfg:
        return None
    if cli:
        scheme, _, token = headers.get("Authorization", "").partition(" ")
        return (
            store(cfg).identity(token, "cli")
            if scheme.lower() == "bearer" and token
            else None
        )
    token = cookie(headers, COOKIE)
    return store(cfg).identity(token, "browser") if token else None


def browser_login(headers):
    user = identity(headers)
    return user["login"] if user else None


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise AuthError(502, "unexpected_github_redirect")


def github_json(url, *, data=None, token=None):
    headers = {"Accept": "application/json", "User-Agent": "transformers-ci-dashboard"}
    if token:
        headers["Authorization"] = "Bearer " + token
    if data is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    request = Request(
        url,
        data=urlencode(data).encode() if data is not None else None,
        headers=headers,
    )
    proxy = os.getenv("TCI_AUTH_GITHUB_PROXY", "")
    with build_opener(
        ProxyHandler({"https": proxy} if proxy else {}), NoRedirect()
    ).open(request, timeout=15) as response:
        result = json.loads(response.read(65536))
    if not isinstance(result, (dict, list)) or "error" in result:
        raise AuthError(401, "github_authorization_failed")
    return result


def require_membership(cfg, user, token):
    """Preserve the HF-member gate using the user's private memberships.

    With a GitHub App user token, GitHub answers membership only when the App
    has the organization "Members: read" permission: 404 means not a member,
    403 means the App cannot read memberships (or is blocked by the org), which
    is reported separately so a missing permission does not look like a refusal.
    Only active memberships count.
    """
    if user["id"] in cfg.allowed_user_ids or not (
        cfg.allowed_orgs or cfg.allowed_user_ids
    ):
        return
    for org in cfg.allowed_orgs:
        try:
            membership = github_json(
                f"https://api.github.com/user/memberships/orgs/{quote(org)}",
                token=token,
            )
        except HTTPError as exc:
            if exc.code == 404:
                continue
            if exc.code == 403:
                raise AuthError(503, "github_app_cannot_read_membership") from None
            raise AuthError(502, "github_membership_check_failed") from None
        if not isinstance(membership, dict):
            raise AuthError(502, "invalid_github_memberships")
        if membership.get("state") == "active":
            return
    raise AuthError(403, "account_not_allowed")


def _body(handler):
    try:
        length = int(handler.headers.get("Content-Length", "0"))
        if (
            not 0 < length <= 4096
            or handler.headers.get_content_type() != "application/json"
        ):
            raise ValueError
        body = json.loads(handler.rfile.read(length))
        if not isinstance(body, dict):
            raise ValueError
        return body
    except (ValueError, TypeError):
        raise AuthError(400, "invalid_json") from None


def _browser_post(handler, cfg):
    if (
        handler.headers.get("Origin") != cfg.url
        or handler.headers.get("X-TCI-Action") != "1"
    ):
        raise AuthError(403, "forbidden")
    user = identity(handler.headers)
    if not user:
        raise AuthError(401, "unauthorized")
    return user


def handle(handler, method):
    """Return True for auth/profile routes, including when disabled."""
    parsed = urlsplit(handler.path)
    path, query = parsed.path, parse_qs(parsed.query)
    if not (path.startswith("/auth/") or path in ("/profile", "/profile/app.js")):
        return False
    handler._request_route = "/auth"  # never label telemetry with codes or tokens
    headers = {
        "Cache-Control": "no-store",
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Content-Security-Policy": "default-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
    }

    def send(status, body, extra=None):
        handler._send(
            status,
            "application/json",
            json.dumps(body).encode(),
            headers={**headers, **(extra or {})},
        )

    def redirect(location, extra=None):
        handler._send(
            302,
            "text/plain",
            b"",
            headers={**headers, "Location": location, **(extra or {})},
        )

    try:
        cfg = config()
        if path in ("/profile", "/auth/cli", "/profile/app.js") and method == "GET":
            asset = "profile.js" if path.endswith("app.js") else "profile.html"
            handler._send(
                200,
                "text/javascript"
                if asset.endswith("js")
                else "text/html; charset=utf-8",
                Path(__file__).with_name(asset).read_bytes(),
                headers=headers,
            )
            return True
        if path == "/auth/me" and method == "GET" and not cfg:
            send(503, {"error": "auth_not_configured"})
            return True
        if not cfg:
            raise AuthError(503, "auth_not_configured")
        db = store(cfg)
        if path == "/auth/github" and method == "GET":
            db.limit("oauth-start", 60)
            state, browser, challenge = db.start_login(
                query.get("next", ["/profile"])[0]
            )
            redirect(
                "https://github.com/login/oauth/authorize?"
                + urlencode(
                    {
                        "client_id": cfg.client_id,
                        "redirect_uri": cfg.callback,
                        "state": state,
                        "code_challenge": challenge,
                        "code_challenge_method": "S256",
                    }
                ),
                {"Set-Cookie": cookie_value(STATE_COOKIE, browser, 600)},
            )
        elif path == "/auth/callback" and method == "GET":
            pending = db.finish_login(
                query.get("state", [""])[0], cookie(handler.headers, STATE_COOKIE)
            )
            code = query.get("code", [""])[0]
            if not code or len(code) > 512:
                raise AuthError(400, "github_authorization_failed")
            result = github_json(
                "https://github.com/login/oauth/access_token",
                data={
                    "client_id": cfg.client_id,
                    "client_secret": cfg.client_secret,
                    "code": code,
                    "redirect_uri": cfg.callback,
                    "code_verifier": pending["verifier"],
                },
            )
            token = result.get("access_token") if isinstance(result, dict) else None
            if not isinstance(token, str) or not token:
                raise AuthError(401, "github_authorization_failed")
            user = github_json("https://api.github.com/user", token=token)
            if not isinstance(user, dict):
                raise AuthError(502, "invalid_github_identity")
            user_id = db.user(user)
            require_membership(cfg, user, token)
            # Rotate an existing session when signing in again.
            previous = identity(handler.headers)
            if previous:
                db.revoke(previous["user_id"], previous["id"])
            session = db.new_session(user_id, "browser", "Web browser")
            redirect(
                pending["destination"],
                {"Set-Cookie": cookie_value(COOKIE, session, SESSION_SECONDS)},
            )
        elif path == "/auth/me" and method == "GET":
            user = identity(
                handler.headers, cli=bool(handler.headers.get("Authorization"))
            )
            if not user:
                raise AuthError(401, "unauthorized")
            profile = db.profile(user)
            if user["kind"] == "cli":
                profile.pop("sessions")
            send(200, profile)
        elif path == "/auth/device" and method == "POST":
            body = _body(handler)
            if body.get("client_id") != "tmic":
                raise AuthError(400, "invalid_client")
            result = db.device()
            result["verification_uri"] = cfg.url + "/auth/cli"
            result["verification_uri_complete"] = (
                result["verification_uri"]
                + "?"
                + urlencode({"user_code": result["user_code"]})
            )
            send(200, result)
        elif path == "/auth/token" and method == "POST":
            body = _body(handler)
            if (
                body.get("grant_type") != "urn:ietf:params:oauth:grant-type:device_code"
                or not isinstance(body.get("device_code"), str)
                or len(body["device_code"]) > 128
            ):
                raise AuthError(400, "invalid_grant")
            send(200, db.poll(body["device_code"]))
        elif path == "/auth/cli/decision" and method == "POST":
            user = _browser_post(handler, cfg)
            body = _body(handler)
            db.decide(user["user_id"], body.get("user_code"), body.get("decision"))
            send(200, {"status": "ok"})
        elif path == "/auth/sessions/revoke" and method == "POST":
            user = _browser_post(handler, cfg)
            body = _body(handler)
            db.revoke(user["user_id"], str(body.get("session_id", "")))
            send(200, {"status": "ok"})
        elif path == "/auth/logout" and method == "POST":
            user = _browser_post(handler, cfg)
            db.revoke(user["user_id"], user["id"])
            send(200, {"status": "ok"}, {"Set-Cookie": cookie_value(COOKIE, "", 0)})
        elif path == "/auth/revoke" and method == "POST":
            user = identity(handler.headers, cli=True)
            if not user:
                raise AuthError(401, "unauthorized")
            db.revoke(user["user_id"], user["id"])
            send(200, {"status": "ok"})
        else:
            raise AuthError(404, "not_found")
    except AuthError as error:
        send(error.status, {"error": error.error})
    except (OSError, ValueError, sqlite3.Error):
        # Never include upstream bodies, OAuth codes, or credentials in replies.
        send(503, {"error": "authentication_unavailable"})
    return True
