"""Regenerate data/mic-coverage.json: how much of Transformers the MIC fleet covers.

The exporter has no Hub or GitHub egress, so the MIC page reads this snapshot.
Rerun it after a fleet update:

    python -m transformersci.contracts.coverage

- contracts: every contract in the fleet's control and prototype tables.
- covered: the distinct ``model_type`` of those contracts' fixtures.
- model types: the keys of ``CONFIG_MAPPING_NAMES`` on Transformers ``main``.
- reachable: covered, plus the model types whose real config the generic
  shrink rule builds and runs (transformers-mic's architecture sweep).
"""

import ast
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from . import fleet

OUTPUT = Path(__file__).parent / "data/mic-coverage.json"
TRANSFORMERS = "huggingface/transformers"
AUTO = "src/transformers/models/auto"
MIC = "huggingface/transformers-mic"
SWEEP = "experiments/results/sweep.json"


def _get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "transformersci"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode()


def _head(repo):
    return json.loads(_get(f"https://api.github.com/repos/{repo}/commits/main"))["sha"]


def model_types(*sources):
    """Every key CONFIG_MAPPING_NAMES is built or updated with across ``sources``."""
    names = set()
    nodes = (node for source in sources for node in ast.walk(ast.parse(source)))
    for node in nodes:
        if isinstance(node, ast.Assign):
            if not any(
                getattr(t, "id", "") == "CONFIG_MAPPING_NAMES" for t in node.targets
            ):
                continue
        elif not (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", "") == "update"
            and getattr(node.func.value, "id", "") == "CONFIG_MAPPING_NAMES"
        ):
            continue
        for item in ast.walk(node):
            if isinstance(item, ast.Tuple) and len(item.elts) == 2:
                key = item.elts[0]
            elif isinstance(item, ast.Dict):
                names.update(k.value for k in item.keys if isinstance(k, ast.Constant))
                continue
            else:
                continue
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                names.add(key.value)
    return sorted(names)


def fleet_fixtures(revision):
    base = f"https://huggingface.co/datasets/{fleet.REPOSITORY}/resolve/{revision}"
    contracts, fixtures = set(), set()
    for name in ("control", "prototypes"):
        rows = fleet._rows("contracts", _get(f"{base}/{name}/contracts.csv"))
        contracts |= {f"{name}/{row['contract']}" for row in rows}
        fixtures |= {(row["mic_repo"], row["revision"]) for row in rows}
    return contracts, sorted(fixtures)


def fixture_model_type(repo, revision):
    config = json.loads(
        _get(f"https://huggingface.co/{repo}/resolve/{revision}/config.json")
    )
    return config["model_type"]


def build():
    revision = fleet.resolve_revision()
    contracts, fixtures = fleet_fixtures(revision)
    covered = sorted({fixture_model_type(*pin) for pin in fixtures})
    transformers = _head(TRANSFORMERS)
    auto = f"https://raw.githubusercontent.com/{TRANSFORMERS}/{transformers}/{AUTO}"
    types = model_types(
        *(
            _get(f"{auto}/{name}")
            for name in ("auto_mappings.py", "configuration_auto.py")
        )
    )
    mic = _head(MIC)
    sweep = json.loads(_get(f"https://raw.githubusercontent.com/{MIC}/{mic}/{SWEEP}"))
    shrinkable = sorted(
        name
        for name, row in sweep["rows"].items()
        if row.get("stage") == "ok" and row.get("finite")
    )
    known = set(types)
    return {
        "format": "mic-coverage-v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fleet": {"repo": fleet.REPOSITORY, "revision": revision},
        "contracts": len(contracts),
        "fixtures": len(fixtures),
        "covered": [name for name in covered if name in known],
        "transformers": {"revision": transformers, "model_types": len(types)},
        "sweep": {
            "repo": MIC,
            "revision": mic,
            "transformers": sweep.get("transformers", ""),
            "shrinkable": [name for name in shrinkable if name in known],
        },
    }


def main():
    data = build()
    OUTPUT.write_text(json.dumps(data, indent=2) + "\n")
    reachable = set(data["covered"]) | set(data["sweep"]["shrinkable"])
    total = data["transformers"]["model_types"]
    print(
        f"{data['contracts']} contracts, {len(data['covered'])}/{total} model types "
        f"covered, {len(reachable)} reachable"
    )


if __name__ == "__main__":
    main()
