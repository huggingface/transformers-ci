"""The dashboard's read-only GitHub App and its fallback token."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from transformersci import github_app
from transformersci.status.reconcile import GitHubClient


@pytest.fixture(scope="module")
def pem(tmp_path_factory) -> str:
    if shutil.which("openssl") is None:
        pytest.skip("needs openssl")
    path = tmp_path_factory.mktemp("key") / "key.pem"
    subprocess.run(
        ["openssl", "genrsa", "-out", str(path), "2048"],
        check=True,
        capture_output=True,
    )
    return path.read_text()


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in (
        github_app.APP_ID_ENV,
        github_app.PRIVATE_KEY_ENV,
        github_app.INSTALLATION_ID_ENV,
        "PAT",
    ):
        monkeypatch.delenv(name, raising=False)
    github_app.reset_for_tests()
    yield
    github_app.reset_for_tests()


class FakeGitHub:
    """Answers the installation lookup and the token mint."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[str, str, dict | None]] = []
        self.fail = fail
        self.minted = 0

    def __call__(self, request, timeout=None):
        body = json.loads(request.data) if request.data else None
        self.calls.append((request.get_method(), request.full_url, body))
        if self.fail:
            raise OSError("GitHub unreachable")
        if request.full_url.endswith("/installation"):
            payload = {"id": 77}
        else:
            self.minted += 1
            payload = {
                "token": f"ghs_{self.minted}",
                "expires_at": "2026-10-08T15:00:00Z",
            }
        return io.BytesIO(json.dumps(payload).encode())


@pytest.mark.skipif(shutil.which("openssl") is None, reason="needs openssl")
@pytest.mark.parametrize("pkcs8", [False, True])
def test_app_jwt_signature_verifies_with_openssl(tmp_path: Path, pkcs8: bool) -> None:
    raw = tmp_path / "raw.pem"
    subprocess.run(
        ["openssl", "genrsa", "-out", str(raw), "2048"], check=True, capture_output=True
    )
    key = tmp_path / "key.pem"
    convert = (
        ["openssl", "pkcs8", "-topk8", "-nocrypt"]
        if pkcs8
        else ["openssl", "rsa", "-traditional"]
    )
    converted = subprocess.run(
        [*convert, "-in", str(raw), "-out", str(key)], capture_output=True
    )
    if converted.returncode and not pkcs8:  # LibreSSL: PKCS#1 is its default
        subprocess.run(
            ["openssl", "rsa", "-in", str(raw), "-out", str(key)],
            check=True,
            capture_output=True,
        )
    pem = key.read_text()
    assert ("BEGIN RSA PRIVATE KEY" in pem) != pkcs8
    token = github_app.app_jwt("123", github_app.parse_private_key(pem), now=1000)
    header, payload, signature = token.split(".")
    claims = json.loads(github_app.base64.urlsafe_b64decode(payload + "=="))
    assert claims == {"iat": 940, "exp": 1540, "iss": "123"}
    pub = tmp_path / "pub.pem"
    subprocess.run(
        ["openssl", "rsa", "-in", str(key), "-pubout", "-out", str(pub)],
        check=True,
        capture_output=True,
    )
    (tmp_path / "msg").write_bytes(f"{header}.{payload}".encode())
    (tmp_path / "sig").write_bytes(
        github_app.base64.urlsafe_b64decode(signature + "=" * (-len(signature) % 4))
    )
    verified = subprocess.run(
        [
            "openssl",
            "dgst",
            "-sha256",
            "-verify",
            str(pub),
            "-signature",
            str(tmp_path / "sig"),
            str(tmp_path / "msg"),
        ],
        capture_output=True,
        text=True,
    )
    assert verified.stdout.strip() == "Verified OK"


def test_tokens_are_read_only_cached_and_refreshed(pem: str) -> None:
    github = FakeGitHub()
    expires = 1791471600.0  # 2026-10-08T15:00:00Z
    clock = [expires - 3600]
    tokens = github_app.AppTokens("123", pem, opener=github, clock=lambda: clock[0])

    assert tokens.token() == "ghs_1"
    assert tokens.token() == "ghs_1"
    lookup, mint = github.calls
    assert lookup[:2] == (
        "GET",
        "https://api.github.com/repos/huggingface/transformers/installation",
    )
    assert mint[:2] == (
        "POST",
        "https://api.github.com/app/installations/77/access_tokens",
    )
    assert mint[2] == {"permissions": github_app.PERMISSIONS}
    assert set(github_app.PERMISSIONS.values()) == {"read"}

    clock[0] = expires - github_app.REFRESH_MARGIN_SECONDS
    assert tokens.token() == "ghs_2"
    assert len(github.calls) == 3  # the installation id is looked up once


def test_read_token_falls_back_until_the_app_is_configured(monkeypatch) -> None:
    monkeypatch.setenv("PAT", "ghp_personal")
    assert not github_app.configured()
    assert github_app.read_token(("MISSING", "PAT")) == "ghp_personal"


def test_read_token_prefers_the_app(monkeypatch, pem: str) -> None:
    monkeypatch.setenv("PAT", "ghp_personal")
    monkeypatch.setenv(github_app.APP_ID_ENV, "123")
    monkeypatch.setenv(github_app.PRIVATE_KEY_ENV, pem)
    assert github_app.configured()
    tokens = github_app.AppTokens("123", pem, opener=FakeGitHub())
    monkeypatch.setattr(
        github_app.AppTokens, "from_env", classmethod(lambda cls: tokens)
    )
    assert github_app.read_token(("PAT",)) == "ghs_1"


def test_a_failing_app_falls_back_and_retries_later(monkeypatch, pem: str) -> None:
    monkeypatch.setenv("PAT", "ghp_personal")
    github = FakeGitHub(fail=True)
    tokens = github_app.AppTokens("123", pem, opener=github)
    monkeypatch.setattr(
        github_app.AppTokens, "from_env", classmethod(lambda cls: tokens)
    )
    now = [1000.0]
    monkeypatch.setattr(github_app.time, "time", lambda: now[0])

    assert github_app.read_token(("PAT",)) == "ghp_personal"
    assert github_app.read_token(("PAT",)) == "ghp_personal"
    assert len(github.calls) == 1  # paused, not retried on every read
    github.fail = False
    now[0] += github_app.RETRY_AFTER_FAILURE_SECONDS
    assert github_app.read_token(("PAT",)) == "ghs_1"


def test_an_unusable_key_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("PAT", "ghp_personal")
    monkeypatch.setenv(github_app.APP_ID_ENV, "123")
    monkeypatch.setenv(github_app.PRIVATE_KEY_ENV, "not a key")
    assert github_app.configured()
    assert github_app.read_token(("PAT",)) == "ghp_personal"


def test_status_client_asks_a_token_provider_per_request() -> None:
    seen: list[str] = []
    tokens = iter(["ghs_a", "ghs_b"])

    def opener(request, timeout=None):
        seen.append(request.get_header("Authorization"))
        response = io.BytesIO(b"{}")
        response.headers = {}
        return response

    client = GitHubClient(lambda: next(tokens), opener=opener)
    client.get("/repos/huggingface/transformers/actions/runs/1")
    client.get("/repos/huggingface/transformers/actions/runs/2")
    assert seen == ["Bearer ghs_a", "Bearer ghs_b"]
