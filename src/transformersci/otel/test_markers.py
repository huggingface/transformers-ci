"""Which of a test's decorators the Test page shows as labels (@slow, @is_flaky).

Traces do not record decorators, so they are read from the test's source at the
commit that ran it: on the method, on its class, or on the base class that
defines an inherited method (often a mixin imported from a shared test file).
"""

from __future__ import annotations

import ast
import base64
import os
import posixpath
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable

from transformersci.otel import related_issues

# Decorator name -> label. ``is_flaky`` retries the test in transformers; the
# pytest marker spellings are accepted too.
LABELS = {"slow": "slow", "is_flaky": "flaky", "flaky": "flaky"}
ORDER = ("slow", "flaky")
# One test file plus at most this many base-class files.
MAX_EXTRA_FILES = 3
_files: OrderedDict[tuple[str, str], tuple[float, ast.Module | None]] = OrderedDict()
_lock = threading.Lock()


def _decorator_labels(decorators: list[ast.expr]) -> set[str]:
    found = set()
    for node in decorators:
        target = node.func if isinstance(node, ast.Call) else node
        # slow / is_flaky / pytest.mark.slow / pytest.mark.flaky
        name = (
            target.id
            if isinstance(target, ast.Name)
            else target.attr
            if isinstance(target, ast.Attribute)
            else ""
        )
        if name in LABELS:
            found.add(LABELS[name])
    return found


def _module(path: str, ref: str, github_get: Callable[[str], object]):
    """The parsed file at ``ref``, cached (for good at a commit, briefly on main)."""
    key = (path, ref)
    now = time.monotonic()
    with _lock:
        cached = _files.get(key)
        if cached and cached[0] > now:
            return cached[1]
    base = os.getenv("PYTEST_GITHUB_API_URL", "https://api.github.com").rstrip("/")
    tree = None
    try:
        item = github_get(
            f"{base}/repos/{related_issues.REPOSITORY}/contents/{path}?ref={ref}"
        )
        if isinstance(item, dict) and item.get("encoding") == "base64":
            raw = base64.b64decode(item["content"], validate=False)
            if len(raw) <= 2 * 1024 * 1024:
                tree = ast.parse(raw.decode("utf-8"))
    except (OSError, ValueError, KeyError, TypeError, UnicodeError, SyntaxError):
        tree = None
    ttl = 86400 if re.fullmatch(r"[0-9a-f]{40}", ref) else 600
    with _lock:
        _files[key] = (now + (ttl if tree is not None else 60), tree)
        _files.move_to_end(key)
        while len(_files) > 128:
            _files.popitem(last=False)
    return tree


def _imports(tree: ast.Module, path: str) -> dict[str, str]:
    """Names imported from other test files -> their repository path."""
    found = {}
    package = posixpath.dirname(path)
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if node.level:
            parent = package
            for _ in range(node.level - 1):
                parent = posixpath.dirname(parent)
            target = posixpath.join(parent, *node.module.split(".")) + ".py"
        elif node.module.startswith("tests."):
            target = "/".join(node.module.split(".")) + ".py"
        else:
            continue
        if not target.startswith("tests/") or ".." in target.split("/"):
            continue
        for alias in node.names:
            found[alias.asname or alias.name] = target
    return found


def _classes(tree: ast.Module) -> dict[str, ast.ClassDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def markers(nodeid: str, commit: str, github_get: Callable[[str], object]) -> list[str]:
    """``["slow", "flaky"]`` or a subset, for a ``tests/…py::[Class::]test`` node."""
    parts = nodeid.split("::")
    path = parts[0]
    if not re.fullmatch(r"tests/[\w./-]+\.py", path) or ".." in path.split("/"):
        return []
    name = parts[-1].split("[", 1)[0]
    if len(parts) < 2 or not re.fullmatch(r"[A-Za-z_]\w*", name):
        return []
    ref = commit if re.fullmatch(r"[0-9a-f]{40}", commit) else "main"
    tree = _module(path, ref, github_get)
    if tree is None:
        return []
    labels: set[str] = set()
    if len(parts) == 2:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == name:
                    labels |= _decorator_labels(node.decorator_list)
        return [label for label in ORDER if label in labels]
    # Walk the class and its bases until the method's definition is found;
    # decorators on every class on that path count (unittest inherits them).
    pending = [(path, tree, parts[1])]
    extra_files = 0
    seen = set()
    while pending:
        file_path, module, class_name = pending.pop(0)
        if (file_path, class_name) in seen:
            continue
        seen.add((file_path, class_name))
        cls = _classes(module).get(class_name)
        if cls is None:
            continue
        labels |= _decorator_labels(cls.decorator_list)
        method = next(
            (
                node
                for node in cls.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name
            ),
            None,
        )
        if method is not None:
            labels |= _decorator_labels(method.decorator_list)
            break
        imported = _imports(module, file_path)
        for base in cls.bases:
            if not isinstance(base, ast.Name):
                continue
            if base.id in _classes(module):
                pending.append((file_path, module, base.id))
            elif base.id in imported and extra_files < MAX_EXTRA_FILES:
                other = _module(imported[base.id], ref, github_get)
                extra_files += 1
                if other is not None:
                    pending.append((imported[base.id], other, base.id))
    return [label for label in ORDER if label in labels]
