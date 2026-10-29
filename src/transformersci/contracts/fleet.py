"""Resolve the shared Hub fleet to an immutable dataset snapshot.

The dataset stores each fleet as CSV tables under ``<fleet>/``, written by
transformers-mic's ``tmic export``; this reads them back into the JSON lock the
runner validates. Revisions from before the tables only have the JSON lock.
"""

import csv
import io
import json
import re
import tempfile
from pathlib import Path

REPOSITORY = "huggingface-mic/transformers-fleet"
FILES = {"control": "fleet-lock.json", "prototypes": "prototype-fleet.lock"}
TABLES = {
    "contracts": (
        "contract",
        "target",
        "mic_repo",
        "revision",
        "contract_sha256",
        "fixture_sha256",
        "checkpoints",
    ),
    "environment": ("package", "version"),
    "known_failures": ("contract", "target", "issue", "match", "skip_outputs"),
}


def resolve_revision(revision="main"):
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    from huggingface_hub import HfApi

    return HfApi().dataset_info(REPOSITORY, revision=revision).sha


def _rows(table, text):
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != TABLES[table]:
        raise ValueError(
            f"{table}.csv must have the columns {', '.join(TABLES[table])}"
        )
    return list(reader)


def _split(cell):
    return [part.strip() for part in cell.split(";") if part.strip()]


def from_tables(tables):
    """The fleet lock for {table: CSV text}; mirrors transformers-mic's fleet.from_tables."""
    contracts = {}
    for row in _rows("contracts", tables["contracts"]):
        contract = contracts.setdefault(row["contract"], {"fixtures": {}})
        fixture = {"revision": row["revision"], "files_sha256": row["fixture_sha256"]}
        if row["target"]:
            fixture["target"] = row["target"]
        contract["fixtures"][row["mic_repo"]] = fixture
        if row["contract_sha256"]:
            if "integration" in contract:
                raise ValueError(
                    f"{row['contract']}: more than one row carries the contract"
                )
            contract["integration"] = {
                "repo": row["mic_repo"],
                "revision": row["revision"],
                "files_sha256": row["contract_sha256"],
            }
            contract["checkpoints"] = dict(
                pin.rsplit("@", 1) for pin in _split(row["checkpoints"])
            )
        elif row["checkpoints"]:
            raise ValueError(
                f"{row['contract']}: checkpoints belong on the row carrying the contract"
            )
    for name, contract in contracts.items():
        if "integration" not in contract:
            raise ValueError(f"{name}: no row carries the contract (contract_sha256)")
        contracts[name] = {
            "integration": contract["integration"],
            "fixtures": contract["fixtures"],
            "checkpoints": contract["checkpoints"],
        }
    known = {}
    for row in _rows("known_failures", tables["known_failures"]):
        entry = {"issue": row["issue"]}
        if row["match"]:
            entry["match"] = row["match"]
        if row["skip_outputs"]:
            entry["skip_outputs"] = _split(row["skip_outputs"])
        known[f"{row['contract']}/{row['target']}"] = (
            entry["issue"] if len(entry) == 1 else entry
        )
    lock = {
        "framework": "transformers",
        "environment": {
            row["package"]: row["version"]
            for row in _rows("environment", tables["environment"])
        },
        "contracts": contracts,
    }
    if known:
        lock["known_failures"] = known
    return lock


def lock_path(fleet="control", revision="main"):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    resolved = resolve_revision(revision)

    def fetch(filename):
        return Path(
            hf_hub_download(
                REPOSITORY, filename, repo_type="dataset", revision=resolved
            )
        )

    try:
        tables = {table: fetch(f"{fleet}/{table}.csv").read_text() for table in TABLES}
    except EntryNotFoundError:  # a revision from before the tables
        return fetch(FILES[fleet]), resolved
    lock = from_tables(tables)
    try:
        legacy = json.loads(fetch(FILES[fleet]).read_text())
    except EntryNotFoundError:
        legacy = None
    if legacy is not None:
        # While tmic still writes both, a JSON-only update must not be shadowed by stale tables.
        legacy.pop("comment", None)
        if legacy != lock:
            raise ValueError(
                f"{REPOSITORY}@{resolved}: the {fleet} tables and {FILES[fleet]} disagree"
            )
    path = Path(tempfile.gettempdir()) / "mic-fleet" / resolved / FILES[fleet]
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(lock, indent=2) + "\n")
    partial.replace(path)
    return path, resolved


def read_revision(path="tests/contracts.lock"):
    """Read the framework's sole fleet configuration: an immutable dataset SHA."""
    revision = Path(path).read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"{path} must contain one immutable dataset commit SHA")
    return revision
