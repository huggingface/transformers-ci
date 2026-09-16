"""The relore client/daemon version check.

The interesting behaviour is not "does it reinstall" — it is *when it refuses
to*. relore publishes no tags and the daemon does not report its build commit,
so ``main`` is the only installable ref: a reinstall converges only when ``main``
is where the deployment is. Both directions in which it cannot converge have to
stop rather than retry, and neither may fail the nightly.
"""

from __future__ import annotations

import urllib.error

import pytest

from transformersci.agentic import relore_version as rv


@pytest.fixture(autouse=True)
def _no_ambient_api(monkeypatch):
    monkeypatch.delenv("RELORE_API", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)


def _versions(monkeypatch, *, server, client, after_install=None):
    """Wire the two probes; record whether an install was attempted."""
    calls = {"install": 0}
    seen = iter([client] + ([after_install] if after_install is not None else []))

    def installed():
        try:
            return next(seen)
        except StopIteration:
            return after_install if after_install is not None else client

    monkeypatch.setattr(rv, "daemon_version", lambda api, **kw: server)
    monkeypatch.setattr(rv, "installed_version", installed)

    def install(**kwargs):
        calls["install"] += 1
        return True, "reinstalled from main"

    monkeypatch.setattr(rv, "install", install)
    return calls


# -- the happy path --------------------------------------------------------


def test_matching_versions_do_not_reinstall(monkeypatch):
    calls = _versions(monkeypatch, server="0.3.17", client="0.3.17")
    result = rv.ensure_current("http://relore.example")
    assert result.ok
    assert calls["install"] == 0
    assert not result.reinstalled


def test_client_behind_is_upgraded(monkeypatch):
    calls = _versions(
        monkeypatch, server="0.3.17", client="0.3.12", after_install="0.3.17"
    )
    result = rv.ensure_current("http://relore.example")
    assert result.ok and result.reinstalled
    assert calls["install"] == 1
    assert "0.3.12 -> 0.3.17" in result.detail


def test_missing_client_is_installed(monkeypatch):
    calls = _versions(monkeypatch, server="0.3.17", client=None, after_install="0.3.17")
    result = rv.ensure_current("http://relore.example")
    assert result.ok and result.reinstalled
    assert calls["install"] == 1


# -- the directions a reinstall cannot fix ---------------------------------


def test_client_newer_does_not_reinstall(monkeypatch):
    # The deployment is behind, not the client. `main` is the only installable
    # ref, so reinstalling can only make the gap wider.
    calls = _versions(monkeypatch, server="0.3.12", client="0.3.17")
    result = rv.ensure_current("http://relore.example")
    assert not result.ok
    assert calls["install"] == 0
    assert "NEWER" in result.detail and "Redeploy relore" in result.detail


def test_an_overshooting_upgrade_is_reported_not_retried(monkeypatch):
    # main moved past the deployment: the upgrade lands newer than the daemon.
    # One attempt, then stop — a second would install the same commit.
    calls = _versions(
        monkeypatch, server="0.3.12", client="0.3.9", after_install="0.3.17"
    )
    result = rv.ensure_current("http://relore.example")
    assert not result.ok
    assert calls["install"] == 1
    assert "no reinstall converges" in result.detail


def test_a_failed_install_is_reported(monkeypatch):
    _versions(monkeypatch, server="0.3.17", client="0.3.12")
    monkeypatch.setattr(rv, "install", lambda **kw: (False, "pip install failed: boom"))
    result = rv.ensure_current("http://relore.example")
    assert not result.ok
    assert "pip install failed" in result.detail


def test_no_install_flag_reports_without_installing(monkeypatch):
    calls = _versions(monkeypatch, server="0.3.17", client="0.3.12")
    result = rv.ensure_current("http://relore.example", allow_install=False)
    assert not result.ok
    assert calls["install"] == 0


def test_unrankable_versions_still_try_the_one_action_that_might_help(monkeypatch):
    calls = _versions(
        monkeypatch, server="0.3.17", client="0.4.0rc1", after_install="0.3.17"
    )
    result = rv.ensure_current("http://relore.example")
    assert calls["install"] == 1
    assert result.ok


# -- a daemon that is not there --------------------------------------------


def test_an_unreachable_daemon_is_not_a_failure(monkeypatch):
    monkeypatch.setattr(rv, "daemon_version", lambda api, **kw: None)
    called = {"n": 0}
    monkeypatch.setattr(
        rv,
        "install",
        lambda **kw: (called.__setitem__("n", called["n"] + 1), (True, ""))[1],
    )
    result = rv.ensure_current("http://relore.example")
    assert not result.ok
    assert called["n"] == 0  # nothing to match against; do not guess
    assert "continuing without it" in result.detail


# -- the probe itself ------------------------------------------------------


class _Response:
    def __init__(self, headers):
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_daemon_version_reads_the_header_from_metrics(monkeypatch):
    seen = {}

    def urlopen(request, timeout=None):
        seen["url"] = request.full_url
        return _Response({rv.SERVER_HEADER: "0.3.17"})

    monkeypatch.setattr(rv.urllib.request, "urlopen", urlopen)
    assert rv.daemon_version("https://relore.example/") == "0.3.17"
    # /metrics, not /api/v1/status: unauthenticated and outside the version gate,
    # so it answers a client of any version including none.
    assert seen["url"] == "https://relore.example/metrics"


def test_daemon_version_reads_the_header_off_an_error_response(monkeypatch):
    # The gate stamps the header on refusals too, so a non-2xx still identifies
    # the daemon.
    def urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 503, "nope", {rv.SERVER_HEADER: "0.3.17"}, None
        )

    monkeypatch.setattr(rv.urllib.request, "urlopen", urlopen)
    assert rv.daemon_version("https://relore.example") == "0.3.17"


def test_daemon_version_is_none_when_nothing_answers(monkeypatch):
    def urlopen(request, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(rv.urllib.request, "urlopen", urlopen)
    assert rv.daemon_version("https://relore.example") is None


def test_a_response_without_the_header_is_not_a_relore_daemon(monkeypatch):
    monkeypatch.setattr(
        rv.urllib.request, "urlopen", lambda request, timeout=None: _Response({})
    )
    assert rv.daemon_version("https://relore.example") is None


# -- the entry point -------------------------------------------------------


@pytest.mark.parametrize("ok", [True, False])
def test_main_always_exits_zero(monkeypatch, capsys, ok):
    # relore is auxiliary here. A version skew degrades retrieval; it must never
    # fail a nightly that has real work to do.
    monkeypatch.setattr(
        rv, "ensure_current", lambda api, allow_install=True: rv.Result(ok, "detail")
    )
    assert rv.main([]) == 0
    assert "relore: detail" in capsys.readouterr().out


def test_main_annotates_a_failure_in_actions(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(
        rv,
        "ensure_current",
        lambda api, allow_install=True: rv.Result(False, "deployment is behind"),
    )
    rv.main([])
    out = capsys.readouterr().out
    assert "::warning title=relore unavailable::deployment is behind" in out


def test_main_does_not_annotate_a_success(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(
        rv, "ensure_current", lambda api, allow_install=True: rv.Result(True, "agree")
    )
    rv.main([])
    assert "::warning" not in capsys.readouterr().out
