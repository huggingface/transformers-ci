"""Expand GitHub-style emoji shortcodes (``:rotating_light:``) to Unicode glyphs.

PR titles and commit subjects come from GitHub, which keeps the shortcodes the
author typed, and Grafana renders them verbatim. Both the trace-exporter and
ci-github-status run on a bare ``python`` image with no packages installed, so
the table is vendored (:mod:`transformersci._emoji_shortcodes`) instead of
importing the optional ``emoji`` package, which silently made this a no-op.

To regenerate the table after upgrading ``emoji``, rerun the snippet that
produced it (fully-qualified glyphs, English names + aliases, no skin tones)::

    for glyph, data in emoji.EMOJI_DATA.items():
        if data.get("status") != emoji.STATUS["fully_qualified"]: continue
        for a in [data.get("en")] + data.get("alias", []):
            if a and "skin_tone" not in a: table.setdefault(a[1:-1], glyph)
"""

from __future__ import annotations

import re

from transformersci._emoji_shortcodes import SHORTCODES

_SHORTCODE_RE = re.compile(r":([^\s:]+):")


def emojize(text: str) -> str:
    """Replace every known ``:shortcode:`` in *text*; unknown ones stay as typed."""
    if not text or ":" not in text:
        return text
    return _SHORTCODE_RE.sub(lambda m: SHORTCODES.get(m.group(1), m.group(0)), text)
