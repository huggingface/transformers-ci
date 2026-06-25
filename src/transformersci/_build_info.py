"""Build-time provenance for the transformers-ci package.

This is a *stub*: in a source checkout or editable install ``BUILD_SHA`` is
empty because no wheel build ran. When a wheel is built (``uv pip install .`` /
``python -m build``), ``setup.py``'s ``build_py`` hook overwrites this file in
the build tree with the git short SHA captured at build time, so the installed
package can report exactly which build a CI runner is running. See
``transformersci.exporter_version`` and
``docs/verify-span-id-collision-fix.md``.
"""

BUILD_SHA = ""
