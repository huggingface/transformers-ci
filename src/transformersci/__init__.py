from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    try:
        __version__ = _pkg_version("transformers-ci")
    except PackageNotFoundError:  # not installed (e.g. running from a raw source tree)
        __version__ = "0.0.0+unknown"
except Exception:  # pragma: no cover - importlib.metadata always present on 3.10+
    __version__ = "0.0.0+unknown"

try:
    from ._build_info import BUILD_SHA
except Exception:  # pragma: no cover - stub is always importable
    BUILD_SHA = ""


def _runtime_git_sha() -> str:
    """Best-effort git short SHA resolved at *runtime* from the package's own
    source tree. This is the fallback for source/editable installs where the
    build hook never ran (``BUILD_SHA`` empty); it returns "" when the package
    lives in a wheel with no ``.git`` — which is exactly the runner case the
    build-time stamp already covers."""
    import os
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return ""


def exporter_version() -> str:
    """Version of the installed transformers-ci package, with the git short SHA
    when available (e.g. ``0.1.0+g011834c``).

    The SHA is the single source of truth for *which build* of the producer
    package a CI runner installed — a static ``version`` in ``pyproject.toml``
    can't tell two builds apart, so the SHA is what answers "is fix X live on the
    runners?". It comes from the build-time stamp (``BUILD_SHA``, set when the
    wheel is built — the runner path, where no ``.git`` exists at runtime), and
    falls back to a runtime ``git`` lookup so source/editable checkouts also
    report it. See ``docs/verify-span-id-collision-fix.md``.
    """
    sha = BUILD_SHA or _runtime_git_sha()
    if sha:
        return f"{__version__}+g{sha}"
    return __version__
