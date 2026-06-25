"""Build shim: all package metadata lives in ``pyproject.toml``.

The only reason this file exists is to stamp the git short SHA into the wheel at
build time. ``pyproject.toml``'s ``version`` is static, so it can't tell two
builds apart; the SHA captured here is what lets the installed package report
*which build* it is (see ``transformersci.exporter_version`` and
``docs/verify-span-id-collision-fix.md``).
"""

from __future__ import annotations

import os
import subprocess

from setuptools import setup
from setuptools.command.build_py import build_py


def _resolve_build_sha() -> str:
    # Prefer an explicit override: CI (or an sdist build with no .git in the
    # sandbox) can stamp the SHA deterministically via this env var.
    sha = os.environ.get("TRANSFORMERSCI_BUILD_SHA", "").strip()
    if sha:
        return sha
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        # No git available (e.g. building from a bare sdist) — leave it empty;
        # exporter_version() degrades to the plain version string.
        return ""


class build_py_with_sha(build_py):
    """Overwrite the checked-in ``_build_info.py`` stub in the build tree with
    the SHA captured at build time. The source tree is left untouched."""

    def run(self) -> None:
        super().run()
        sha = _resolve_build_sha()
        target = os.path.join(self.build_lib, "transformersci", "_build_info.py")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(
                '"""Generated at build time by setup.py. Do not edit."""\n'
                f'BUILD_SHA = "{sha}"\n'
            )
        if sha:
            self.announce(f"transformers-ci: stamped BUILD_SHA={sha}", level=2)


setup(cmdclass={"build_py": build_py_with_sha})
