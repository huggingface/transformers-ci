"""GitHub App installation tokens with no third-party dependency.

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
import threading
import time
from urllib.request import Request, urlopen

API_URL = "https://api.github.com"
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


class AppTokens:
    """Installation tokens for one app installation, narrowed to one
    repository and the permissions the rerun action needs, cached until
    shortly before they expire."""

    PERMISSIONS = {
        "actions": "write",
        "issues": "read",
        "metadata": "read",
        "pull_requests": "read",
    }

    def __init__(
        self, app_id: str, installation_id: str, private_key: str, repository: str
    ) -> None:
        self.app_id = app_id
        self.installation_id = installation_id
        self.key = parse_private_key(private_key)
        self.repository = repository
        self._lock = threading.Lock()
        self._token = ""
        self._expires = 0.0

    @classmethod
    def from_env(cls, repository: str) -> AppTokens | None:
        app_id = os.getenv("PYTEST_TRACE_EXPORTER_GITHUB_APP_ID", "").strip()
        installation = os.getenv(
            "PYTEST_TRACE_EXPORTER_GITHUB_APP_INSTALLATION_ID", ""
        ).strip()
        key = os.getenv("PYTEST_TRACE_EXPORTER_GITHUB_APP_PRIVATE_KEY", "")
        if not (app_id.isdigit() and installation.isdigit() and key.strip()):
            return None
        return cls(app_id, installation, key, repository)

    def token(self) -> str:
        with self._lock:
            if self._token and time.time() < self._expires - 300:
                return self._token
            body = json.dumps(
                {
                    "repositories": [self.repository.split("/", 1)[1]],
                    "permissions": self.PERMISSIONS,
                }
            ).encode()
            request = Request(
                f"{API_URL}/app/installations/{self.installation_id}/access_tokens",
                data=body,
                method="POST",
                headers={
                    "Accept": "application/vnd.github+json",
                    "Authorization": f"Bearer {app_jwt(self.app_id, self.key)}",
                    "User-Agent": "transformersci-rerun-failed",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
            with urlopen(request, timeout=10) as response:
                payload = json.loads(response.read())
            token = payload.get("token") if isinstance(payload, dict) else None
            if not isinstance(token, str) or not token:
                raise ValueError("GitHub returned no installation token")
            expires = str(payload.get("expires_at") or "")
            try:
                from datetime import datetime

                self._expires = datetime.fromisoformat(
                    expires.replace("Z", "+00:00")
                ).timestamp()
            except ValueError:
                self._expires = time.time() + 1800
            self._token = token
            return token
