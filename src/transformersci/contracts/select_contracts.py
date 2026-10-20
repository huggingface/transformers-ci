"""Select the contracts a Transformers change can affect, from the files it changes.

A contract's code is found from its fixtures, not declared by hand: the
model_type and architectures in each fixture's config.json, and the tokenizer,
feature extractor, image processor, and processor classes in its files or in
the Auto mappings for that model_type, each resolved to its directory under
src/transformers/models/. A changed file then selects:

  src/transformers/models/<dir>/...        the contracts whose code lives in <dir>
  registry-only Auto/import additions    only the registered models (requires base diff)
  any other file under src/transformers/   every contract (shared runtime code)
  setup.py, pyproject.toml                 every contract (dependencies)
  anything else (tests, docs, utils, ...)  nothing
"""

import ast
import json
import subprocess
from pathlib import Path

SHARED = ("setup.py", "pyproject.toml")
PROCESSOR_FILES = (
    "tokenizer_config.json",
    "preprocessor_config.json",
    "processor_config.json",
)
CLASS_KEYS = (
    "tokenizer_class",
    "processor_class",
    "feature_extractor_type",
    "image_processor_type",
)


def module_dir(name):
    """models/<dir> holding a Transformers class, or None if it lives in shared code."""
    import transformers

    cls = getattr(transformers, name, None) or getattr(
        transformers, name.removesuffix("Fast"), None
    )
    if cls is None:
        return None
    parts = cls.__module__.split(".")
    return parts[2] if len(parts) > 3 and parts[1] == "models" else None


def code_dirs(fixture_folder):
    """The model directories whose code a fixture's usage path runs."""
    from transformers.models.auto import (
        configuration_auto,
        feature_extraction_auto,
        image_processing_auto,
        processing_auto,
        tokenization_auto,
    )

    folder = Path(fixture_folder)
    config = json.loads((folder / "config.json").read_text())
    model_type = config["model_type"]
    names = set(config.get("architectures") or [])
    names.add(configuration_auto.CONFIG_MAPPING_NAMES[model_type])
    # Composite models run their sub-models' code too (MusicGen Melody: T5 text encoder, EnCodec audio codec).
    stack = [v for v in config.values() if isinstance(v, dict)]
    while stack:
        sub = stack.pop()
        stack.extend(v for v in sub.values() if isinstance(v, dict))
        if sub.get("model_type") in configuration_auto.CONFIG_MAPPING_NAMES:
            names.add(configuration_auto.CONFIG_MAPPING_NAMES[sub["model_type"]])
    for mapping in (
        tokenization_auto.TOKENIZER_MAPPING_NAMES,
        feature_extraction_auto.FEATURE_EXTRACTOR_MAPPING_NAMES,
        image_processing_auto.IMAGE_PROCESSOR_MAPPING_NAMES,
        processing_auto.PROCESSOR_MAPPING_NAMES,
    ):
        value = mapping.get(model_type)
        names.update(
            v
            for v in (value if isinstance(value, (tuple, list)) else [value])
            if isinstance(v, str)
        )
    for file in PROCESSOR_FILES:
        if (folder / file).exists():
            values = json.loads((folder / file).read_text())
            names.update(
                values[k] for k in CLASS_KEYS if isinstance(values.get(k), str)
            )
    return sorted({d for d in map(module_dir, names) if d})


def _entries(value):
    if isinstance(value, ast.Dict):
        pairs = list(zip(value.keys, value.values))
    elif (
        isinstance(value, ast.Call)
        and (
            isinstance(value.func, ast.Name)
            and value.func.id == "OrderedDict"
            or isinstance(value.func, ast.Subscript)
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == "OrderedDict"
        )
        and len(value.args) == 1
        and isinstance(value.args[0], (ast.List, ast.Tuple))
    ):
        pairs = []
        for item in value.args[0].elts:
            if isinstance(item, ast.Starred):
                continue  # Unchanged splices are checked by the final AST comparison.
            if not isinstance(item, (ast.List, ast.Tuple)) or len(item.elts) != 2:
                return None
            pairs.append(item.elts)
    else:
        return None
    if not all(
        isinstance(key, ast.Constant) and isinstance(key.value, str) for key, _ in pairs
    ):
        return None
    result = {key.value: ast.dump(val) for key, val in pairs}
    return result


def _registration_value(node):
    """Registrations contain names and optional backend availability checks."""
    for part in ast.walk(node):
        if isinstance(part, ast.Call):
            if not (
                isinstance(part.func, ast.Name)
                and part.func.id.startswith("is_")
                and part.func.id.endswith("_available")
                and not part.args
                and not part.keywords
            ):
                return False
        elif not isinstance(
            part, (ast.Constant, ast.Name, ast.Load, ast.Tuple, ast.List, ast.IfExp)
        ):
            return False
    return True


def registry_models(path, before, after):
    """Model directories from provably additive Auto registrations, else None.

    Remove only new mapping entries/imports and require the remaining AST to be
    identical. Changes to existing keys, helpers, imports or runtime logic must
    keep the full-fleet fallback.
    """
    if not path.startswith("src/transformers/models/auto/") and path not in {
        "src/transformers/models/__init__.py",
        "src/transformers/__init__.py",
    }:
        return None
    try:
        old, new = ast.parse(before), ast.parse(after)
    except SyntaxError:
        return None
    models = set()
    if path in {"src/transformers/models/__init__.py", "src/transformers/__init__.py"}:
        old_imports = {
            ast.dump(node) for node in ast.walk(old) if isinstance(node, ast.ImportFrom)
        }
        for node in ast.walk(new):
            if (
                not isinstance(node, ast.If)
                or not isinstance(node.test, ast.Name)
                or node.test.id != "TYPE_CHECKING"
            ):
                continue
            retained = []
            for item in node.body:
                if (
                    isinstance(item, ast.ImportFrom)
                    and item.level == 1
                    and item.module
                    and ast.dump(item) not in old_imports
                ):
                    parts = item.module.split(".")
                    if path == "src/transformers/__init__.py":
                        if len(parts) < 2 or parts[0] != "models":
                            return None
                        models.add(parts[1])
                    else:
                        models.add(parts[0])
                else:
                    retained.append(item)
            node.body = retained
    else:
        assignments = {
            node.targets[0].id: node.value
            for node in old.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        }
        for node in new.body:
            if (
                not isinstance(node, ast.Assign)
                or len(node.targets) != 1
                or not isinstance(node.targets[0], ast.Name)
            ):
                continue
            name = node.targets[0].id
            if (
                not (
                    name.endswith(("_MAPPING_NAMES", "_MAPPING"))
                    or name == "SPECIAL_MODEL_TYPE_TO_MODULE_NAME"
                )
                or name not in assignments
            ):
                continue
            previous, current = _entries(assignments[name]), _entries(node.value)
            if (
                previous is None
                or current is None
                or any(current.get(k) != v for k, v in previous.items())
            ):
                continue
            added = set(current) - set(previous)
            if not added:
                continue
            values = (
                dict(zip((key.value for key in node.value.keys), node.value.values))
                if isinstance(node.value, ast.Dict)
                else {
                    item.elts[0].value: item.elts[1]
                    for item in node.value.args[0].elts
                    if not isinstance(item, ast.Starred)
                }
            )
            if any(not _registration_value(values[key]) for key in added):
                continue
            keys = (
                [key.value for key in node.value.keys]
                if isinstance(node.value, ast.Dict)
                else [
                    item.elts[0].value
                    for item in node.value.args[0].elts
                    if not isinstance(item, ast.Starred)
                ]
            )
            if any(keys.count(key) != 1 for key in added):
                continue
            models.update(added)
            if isinstance(node.value, ast.Dict):
                pairs = [
                    (k, v)
                    for k, v in zip(node.value.keys, node.value.values)
                    if k.value not in added
                ]
                node.value.keys = [k for k, _ in pairs]
                node.value.values = [v for _, v in pairs]
            else:
                node.value.args[0].elts = [
                    item
                    for item in node.value.args[0].elts
                    if isinstance(item, ast.Starred) or item.elts[0].value not in added
                ]
    return models if models and ast.dump(old) == ast.dump(new) else None


def changed_sources(paths, base):
    """Read the same merge-base diff as CI, falling back to path-only selection."""
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "--verify", "--end-of-options", f"{base}^{{commit}}"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        ancestor = subprocess.check_output(
            ["git", "merge-base", revision, "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return {}
    sources = {}
    for path in paths:
        if not path.startswith("src/transformers/models/auto/") and path not in {
            "src/transformers/models/__init__.py",
            "src/transformers/__init__.py",
        }:
            continue
        try:
            before = subprocess.check_output(
                ["git", "show", f"{ancestor}:{path}"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            sources[path] = (before, Path(path).read_text())
        except (subprocess.CalledProcessError, OSError, UnicodeError):
            pass
    return sources


def select(changed_files, contract_dirs, sources=None, model_type_to_module=None):
    """{contract: reason} for the contracts the changed files can affect."""
    selected = {}
    for path in changed_files:
        if path.startswith(
            (
                "utils/contracts/",
                ".github/workflows/contracts",
                "tests/contracts/fleet",
                "tests/contracts/prototype-fleet",
            )
        ):
            return {name: f"contract infrastructure ({path})" for name in contract_dirs}
        if path.startswith("tests/contracts/expectations/"):
            name = path.split("/")[3]
            if name in contract_dirs:
                selected[name] = f"expectations ({path})"
            continue
        if path in SHARED:
            return {name: f"dependencies ({path})" for name in contract_dirs}
        if not path.startswith("src/transformers/"):
            continue
        if sources and path in sources:
            models = registry_models(path, *sources[path])
            if models is not None:
                resolve = model_type_to_module or (lambda name: name.replace("-", "_"))
                models = {resolve(name) for name in models}
                for name, dirs in contract_dirs.items():
                    if models.intersection(dirs):
                        selected.setdefault(name, f"model registration ({path})")
                continue
        parts = path.split("/")
        if len(parts) > 4 and parts[2] == "models" and parts[3] != "auto":
            for name, dirs in contract_dirs.items():
                if parts[3] in dirs:
                    selected.setdefault(name, f"models/{parts[3]} ({path})")
        else:
            return {name: f"shared code ({path})" for name in contract_dirs}
    return selected
