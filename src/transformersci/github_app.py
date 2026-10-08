"""GitHub reads as the dashboard's read-only GitHub App, with no third-party dependency.

The dashboard's GitHub reads (trace-exporter, ci-data-publisher, ci-github-status)
used a personal access token, whose 5,000/h budget is per *user* and so shared
with everything else that person runs. An App installation has its own budget.

``read_token`` returns an installation token when ``TRANSFORMERSCI_GITHUB_APP_ID``
and ``TRANSFORMERSCI_GITHUB_APP_PRIVATE_KEY`` are set, and otherwise (or while the
App cannot mint one: not installed yet, key rotated) the first fallback token
found, so a deploy can land before the App exists.

The exporter image is a bare ``python:3.12-slim`` with no ``cryptography``, so
the RS256 app JWT is signed here: RSASSA-PKCS1-v1_5 over SHA-256 (RFC 8017
§8.2), with RSA blinding so the private exponentiation does not time the key.
Only the PEM shapes GitHub issues are read: PKCS#1 ``RSA PRIVATE KEY`` and
PKCS#8 ``PRIVATE KEY``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import sys
import threading
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from urllib.request import Request, urlopen

API_URL = "https://api.github.com"
APP_ID_ENV = "TRANSFORMERSCI_GITHUB_APP_ID"
PRIVATE_KEY_ENV = "TRANSFORMERSCI_GITHUB_APP_PRIVATE_KEY"
# Optional: the installation is otherwise looked up from this repository.
INSTALLATION_ID_ENV = "TRANSFORMERSCI_GITHUB_APP_INSTALLATION_ID"
REPOSITORY = "huggingface/transformers"
# Read-only on purpose: these reads serve a public dashboard. Writes go
# through serge's /dashboard API, never through this App.
PERMISSIONS = {
    "actions": "read",
    "contents": "read",
    "issues": "read",
    "metadata": "read",
    "pull_requests": "read",
}
# Mint a new token this long before the current one expires (they live 1h).
REFRESH_MARGIN_SECONDS = 300
# After a failed mint, use the fallback token for this long before retrying.
RETRY_AFTER_FAILURE_SECONDS = 300
# DigestInfo prefix for SHA-256 (RFC 8017 §9.2, note 1).
_SHA256_PREFIX = bytes.fromhex("3031300d060960864801650304020105000420")
_RSA_OID = bytes.fromhex("2a864886f70d010101")


class PrivateKeyError(ValueError):
    """The private key could not be read."""


def _der_item(data: bytes, offset: int) -> tuple[int, bytes, int]:
    """(tag, content, next offset) of the DER item at ``offset``."""
    if offset + 2 > len(data):
        raise PrivateKeyError("truncated DER")
    tag, length = data[offset], data[offset + 1]
    offset += 2
    if length & 0x80:
        count = length & 0x7F
        if not 0 < count <= 4 or offset + count > len(data):
            raise PrivateKeyError("bad DER length")
        length = int.from_bytes(data[offset : offset + count], "big")
        offset += count
    if offset + length > len(data):
        raise PrivateKeyError("truncated DER")
    return tag, data[offset : offset + length], offset + length


def _der_sequence(data: bytes) -> list[tuple[int, bytes]]:
    tag, content, _ = _der_item(data, 0)
    if tag != 0x30:
        raise PrivateKeyError("expected a DER sequence")
    items, offset = [], 0
    while offset < len(content):
        tag, value, offset = _der_item(content, offset)
        items.append((tag, value))
    return items


def parse_private_key(pem: str) -> tuple[int, int, int]:
    """(n, e, d) of an RSA private key in PEM form."""
    match = re.search(
        r"-----BEGIN (RSA )?PRIVATE KEY-----(.+?)-----END (?:RSA )?PRIVATE KEY-----",
        pem,
        re.S,
    )
    if not match:
        raise PrivateKeyError("no PEM private key")
    der = base64.b64decode("".join(match.group(2).split()))
    if not match.group(1):  # PKCS#8 wraps the PKCS#1 key in an OCTET STRING
        items = _der_sequence(der)
        if len(items) < 3 or items[2][0] != 0x04 or _RSA_OID not in items[1][1]:
            raise PrivateKeyError("not an RSA PKCS#8 key")
        der = items[2][1]
    items = _der_sequence(der)
    if len(items) < 4 or any(tag != 0x02 for tag, _ in items[:4]):
        raise PrivateKeyError("not an RSA PKCS#1 key")
    n, e, d = (int.from_bytes(value, "big") for _, value in items[1:4])
    return n, e, d


def rsa_sha256_sign(key: tuple[int, int, int], message: bytes) -> bytes:
    n, e, d = key
    size = (n.bit_length() + 7) // 8
    digest = _SHA256_PREFIX + hashlib.sha256(message).digest()
    if size < len(digest) + 11:
        raise PrivateKeyError("RSA key too small")
    encoded = b"\x00\x01" + b"\xff" * (size - len(digest) - 3) + b"\x00" + digest
    m = int.from_bytes(encoded, "big")
    while True:
        r = secrets.randbelow(n - 2) + 2
        try:
            r_inv = pow(r, -1, n)
            break
        except ValueError:
            continue
    blinded = pow(m * pow(r, e, n) % n, d, n)
    return (blinded * r_inv % n).to_bytes(size, "big")


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def app_jwt(app_id: str, key: tuple[int, int, int], now: float | None = None) -> str:
    now = int(time.time() if now is None else now)
    header = _b64url(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
    # iat backdated for clock drift; GitHub refuses exp more than 10 min ahead.
    payload = _b64url(
        json.dumps({"iat": now - 60, "exp": now + 540, "iss": app_id}).encode()
    )
    signing_input = f"{header}.{payload}".encode()
    return f"{header}.{payload}.{_b64url(rsa_sha256_sign(key, signing_input))}"


def _request_json(
    url: str, jwt: str, opener: Callable[..., object], body: dict | None = None
) -> object:
    request = Request(
        url,
        data=None if body is None else json.dumps(body).encode(),
        method="GET" if body is None else "POST",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {jwt}",
            "User-Agent": "transformersci-dashboard",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with opener(request, timeout=10) as response:
        return json.loads(response.read())


class AppTokens:
    """Read-only installation tokens, cached until shortly before they expire."""

    def __init__(
        self,
        app_id: str,
        private_key: str,
        *,
        installation_id: str = "",
        repository: str = REPOSITORY,
        opener: Callable[..., object] = urlopen,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.app_id = app_id
        self.key = parse_private_key(private_key)
        self.installation_id = installation_id
        self.repository = repository
        self._open, self._clock = opener, clock
        self._lock = threading.Lock()
        self._token = ""
        self._expires = 0.0

    @classmethod
    def from_env(cls) -> AppTokens | None:
        app_id = os.getenv(APP_ID_ENV, "").strip()
        key = os.getenv(PRIVATE_KEY_ENV, "")
        if not (app_id.isdigit() and key.strip()):
            return None
        installation = os.getenv(INSTALLATION_ID_ENV, "").strip()
        return cls(
            app_id,
            key,
            installation_id=installation if installation.isdigit() else "",
        )

    def token(self) -> str:
        with self._lock:
            now = self._clock()
            if self._token and now < self._expires - REFRESH_MARGIN_SECONDS:
                return self._token
            jwt = app_jwt(self.app_id, self.key, now=now)
            if not self.installation_id:
                found = _request_json(
                    f"{API_URL}/repos/{self.repository}/installation", jwt, self._open
                )
                installation = found.get("id") if isinstance(found, dict) else None
                if not isinstance(installation, int):
                    raise ValueError("GitHub returned no installation id")
                self.installation_id = str(installation)
            payload = _request_json(
                f"{API_URL}/app/installations/{self.installation_id}/access_tokens",
                jwt,
                self._open,
                body={"permissions": PERMISSIONS},
            )
            token = payload.get("token") if isinstance(payload, dict) else None
            if not isinstance(token, str) or not token:
                raise ValueError("GitHub returned no installation token")
            try:
                expires = datetime.fromisoformat(
                    str(payload.get("expires_at") or "").replace("Z", "+00:00")
                ).timestamp()
            except ValueError:
                expires = now + 1800
            self._token, self._expires = token, expires
            return token


_app: AppTokens | None = None
_app_loaded = False
_app_failed_until = 0.0
_app_state_lock = threading.Lock()


def _log(message: str) -> None:
    print(f"[github-app] {message}", file=sys.stderr, flush=True)


def app_token() -> str:
    """An installation token, or "" when the App is not configured or cannot
    mint one right now (logged, and retried after a pause)."""
    global _app, _app_loaded, _app_failed_until
    with _app_state_lock:
        if not _app_loaded:
            _app_loaded = True
            try:
                _app = AppTokens.from_env()
            except PrivateKeyError as error:
                _log(f"{PRIVATE_KEY_ENV} is unusable ({error}); using the fallback")
            if _app is not None:
                _log(f"reading GitHub as App {_app.app_id}")
        app = _app
        if app is None or time.time() < _app_failed_until:
            return ""
    try:
        return app.token()
    except Exception as error:
        with _app_state_lock:
            _app_failed_until = time.time() + RETRY_AFTER_FAILURE_SECONDS
        _log(
            f"could not mint a token ({type(error).__name__}: {error}); using the "
            f"fallback for {RETRY_AFTER_FAILURE_SECONDS}s"
        )
        return ""


def configured() -> bool:
    """Both App settings are present (the key is only parsed on first use)."""
    return os.getenv(APP_ID_ENV, "").strip().isdigit() and bool(
        os.getenv(PRIVATE_KEY_ENV, "").strip()
    )


def read_token(fallback_envs: Sequence[str]) -> str:
    """The App's token if it can mint one, else the first non-empty fallback."""
    token = app_token()
    if token:
        return token
    for name in fallback_envs:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def reset_for_tests() -> None:
    global _app, _app_loaded, _app_failed_until
    with _app_state_lock:
        _app, _app_loaded, _app_failed_until = None, False, 0.0
