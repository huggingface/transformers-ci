"""Keep the `relore` client in step with the daemon it will talk to.

`relore` ships client and daemon as one version and refuses a request from any
other with **426 Upgrade Required**, rather than answering it with a contract the
caller does not have. That is the right trade — an old client would otherwise get
a complete-looking reply missing whatever it did not know to ask for — but it
means a client pinned at install time goes dead the moment the daemon is
redeployed, and relore is redeployed often.

serge solves this by pinning the client in its images, because a task pod's
egress allowlist has no PyPI and cannot self-repair. **A GitHub Actions runner
can**, so here the answer is to check and reinstall rather than to pin: one probe
before the nightly uses relore, and a reinstall when the client is the stale end.

How the version is read
-----------------------
From the ``x-relore-version`` header, on ``/metrics``. That endpoint is
unauthenticated and deliberately outside the version gate (it is the
deployment's own probe and must not fail on a skew it cannot fix), so it answers
whatever the client's state is — which is exactly what a version check needs.
``relore status`` is *inside* the gate, so a drifted client cannot run it: it
fails with the 426 instead. The 426 body does name both versions, but reading the
header needs no client at all and works when none is installed yet.

What this can and cannot fix
----------------------------
`relore` publishes no tags and no PyPI release, and the daemon does not report
the commit it was built from — so the only installable ref is ``main``. That
bounds what self-repair can do, and the bound is worth stating because it decides
what this module does in each direction:

* **Client older than the daemon.** Reinstall from ``main`` and re-read. If
  ``main`` is where the daemon is, this fixes it. If ``main`` has moved past the
  deployment, the reinstall overshoots and the client is now *newer* — so the
  result is re-checked after installing, and a second mismatch is reported rather
  than retried. Reinstalling again cannot converge.
* **Client newer than the daemon.** The deployment is behind, not the client.
  Reinstalling would make it worse. Report and stop.

Both give-up paths are the same outcome for the caller: no usable relore this
run. Which is fine — every consumer of relore here is auxiliary context or an
extra defer reason, never a gate (transformers-ci#118). A retrieval outage must
never silently shrink the night's work, so nothing in this module raises and the
entry point always exits 0.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

#: Where the client comes from. No tags, no PyPI — see the module docstring.
INSTALL_SPEC = "relore @ git+https://github.com/huggingface/relore"

#: The response header every relore daemon stamps on every reply.
SERVER_HEADER = "x-relore-version"

#: Default deployment. Overridden by ``RELORE_API``, like the client's own.
DEFAULT_API = "https://ghlore.huggingface.tech"

PROBE_TIMEOUT_SECONDS = 15
INSTALL_TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class Result:
    """What the check concluded, in a form the caller can log and move on from.

    ``ok`` is the only field a consumer needs: it means a client is installed and
    its version equals the daemon's, so a relore call will not be refused.
    """

    ok: bool
    detail: str
    client: str | None = None
    server: str | None = None
    reinstalled: bool = False


def daemon_version(api: str, *, timeout: int = PROBE_TIMEOUT_SECONDS) -> str | None:
    """The deployed daemon's version, or ``None`` if nothing answered.

    ``/metrics`` rather than ``/api/v1/status``: unauthenticated, outside the
    version gate, and the one endpoint guaranteed to answer a client of any
    version (including none).
    """
    url = api.rstrip("/") + "/metrics"
    request = urllib.request.Request(url, headers={"accept": "text/plain"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            version = response.headers.get(SERVER_HEADER)
    except urllib.error.HTTPError as exc:
        # A non-2xx still carries the header — the gate stamps it on refusals too.
        version = exc.headers.get(SERVER_HEADER) if exc.headers else None
    except Exception:
        return None
    return (version or "").strip() or None


def installed_version() -> str | None:
    """The installed client's version, or ``None`` when it is not installed.

    Read by running the client rather than by importing it: the entry point is
    what the workflow will actually invoke, so this proves the thing on PATH is
    the thing that got upgraded — an import could resolve to a different
    site-packages than the console script.
    """
    try:
        proc = subprocess.run(
            ["relore", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    # `argparse` prints "relore 0.3.17".
    parts = (proc.stdout or proc.stderr).strip().split()
    return parts[-1] if parts else None


def install(*, spec: str = INSTALL_SPEC) -> tuple[bool, str]:
    """``pip install --upgrade`` the client from ``main``. Never raises."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "--no-input", spec],
            capture_output=True,
            text=True,
            timeout=INSTALL_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.SubprocessError as exc:
        return False, f"pip install did not complete: {exc}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        return False, "pip install failed: " + " / ".join(tail)
    return True, "reinstalled from main"


def ensure_current(api: str | None = None, *, allow_install: bool = True) -> Result:
    """Probe, reinstall if that can help, and report. Never raises.

    The reinstall is attempted in exactly one direction — client behind daemon —
    and its outcome is re-read rather than assumed, because ``main`` can be ahead
    of the deployment and the fix would then leave the client newer.
    """
    api = (api or os.environ.get("RELORE_API") or DEFAULT_API).strip()

    server = daemon_version(api)
    if server is None:
        return Result(
            False,
            f"no relore daemon answered at {api} (down, or the runner is off the "
            "network that reaches it); continuing without it",
        )

    client = installed_version()
    if client == server:
        return Result(True, f"client and daemon agree at {server}", client, server)

    if client is None:
        if not allow_install:
            return Result(
                False, f"no relore client installed; daemon is {server}", None, server
            )
        ok, detail = install()
        if not ok:
            return Result(False, detail, None, server)
        client = installed_version()
        if client == server:
            return Result(True, f"installed {client}", client, server, True)
        return Result(
            False,
            f"installed {client} but the daemon is {server}; main has moved past "
            "the deployment, so no reinstall converges. Redeploy relore.",
            client,
            server,
            True,
        )

    if _behind(client, server) is False:
        # Client newer: the deployment is behind and reinstalling makes it worse.
        return Result(
            False,
            f"client {client} is NEWER than the daemon ({server}): the deployment "
            "is behind, not the client. Redeploy relore; continuing without it.",
            client,
            server,
        )

    if not allow_install:
        return Result(
            False,
            f"client {client} != daemon {server} (install not allowed)",
            client,
            server,
        )

    ok, detail = install()
    if not ok:
        return Result(False, detail, client, server)
    upgraded = installed_version()
    if upgraded == server:
        return Result(True, f"upgraded {client} -> {upgraded}", upgraded, server, True)
    return Result(
        False,
        f"upgraded {client} -> {upgraded} but the daemon is {server}; main has "
        "moved past the deployment, so no reinstall converges. Redeploy relore.",
        upgraded,
        server,
        True,
    )


def _behind(client: str, server: str) -> bool | None:
    """``True`` when ``client`` sorts before ``server``; ``None`` if unrankable.

    Mirrors ``relore.wire._parts``: a version that is not all-numeric dotted
    parts cannot be ordered, and an unrankable pair is treated as "behind" by the
    caller so the one action that might help is still tried.
    """

    def parts(version: str) -> tuple[int, ...] | None:
        bits = version.split(".")
        if not bits or not all(bit.isdigit() for bit in bits):
            return None
        return tuple(int(bit) for bit in bits)

    mine, theirs = parts(client), parts(server)
    if mine is None or theirs is None:
        return None
    return mine < theirs


def main(argv: list[str] | None = None) -> int:
    """Always exits 0. relore is auxiliary here; it must never fail a nightly."""
    parser = argparse.ArgumentParser(
        prog="relore-ensure-current",
        description=(
            "Check the installed relore client against the deployed daemon's "
            "x-relore-version header and reinstall when the client is the stale "
            "end. Always exits 0 — relore is auxiliary context, never a gate."
        ),
    )
    parser.add_argument(
        "--api", default=None, help=f"default: $RELORE_API, else {DEFAULT_API}"
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="report the mismatch without reinstalling",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    result = ensure_current(args.api, allow_install=not args.no_install)

    if args.json:
        print(json.dumps(result.__dict__, indent=2))
    else:
        print(f"relore: {result.detail}")

    # A mismatch is a warning, not a failure: the consumers degrade to no
    # retrieval. Surfaced as a workflow annotation so it is visible without
    # reading the step log, and so a deployment that is behind gets noticed.
    if not result.ok and os.environ.get("GITHUB_ACTIONS"):
        print(f"::warning title=relore unavailable::{result.detail}")
    return 0
