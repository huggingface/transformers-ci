"""The shared loader every dashboard carries in a hidden srcdoc iframe."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASHBOARDS = sorted((ROOT / "dashboard").glob("*.json"))


def _contents(path: Path) -> list[str]:
    def walk(panels: list) -> list:
        out = []
        for panel in panels:
            out.append(panel)
            out.extend(walk(panel.get("panels", [])))
        return out

    data = json.loads(path.read_text())
    return [
        (panel.get("options") or {}).get("content") or ""
        for panel in walk(data.get("panels", []))
    ]


def _guard(content: str) -> str:
    start = content.index("if(!w.tciDeployGuard)")
    return content[start : content.index("`)();}", start)]


def test_every_loader_carries_the_same_deploy_guard() -> None:
    """A redeploy leaves the exporter with no backend for a while; the guard
    replaces the ALB's bare 502/503/504 page in exporter iframes with a notice
    and reloads them until the exporter answers."""
    guards = {}
    for path in DASHBOARDS:
        loaders = [c for c in _contents(path) if "if(!w.tciFullNav)" in c]
        assert len(loaders) == 1, path.name
        guards[path.name] = _guard(loaders[0])
    assert len(set(guards.values())) == 1, guards.keys()


def test_parent_realm_functions_have_no_backslashes() -> None:
    """Code passed to ``new w.Function(`...`)`` is a template literal first:
    ``\\b`` and ``\\/`` are consumed as string escapes and the regex breaks,
    silently, because the loader swallows nothing and logs nothing."""
    for path in DASHBOARDS:
        for content in _contents(path):
            for body in re.findall(r"new w\.Function\([^`]*`([^`]*)`", content):
                assert "\\" not in body, path.name
