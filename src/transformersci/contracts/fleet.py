"""Resolve the shared Hub fleet to an immutable dataset snapshot."""

import re
from pathlib import Path

REPOSITORY = "huggingface-mic/transformers-fleet"
FILES = {"control": "fleet-lock.json", "prototypes": "prototype-fleet.lock"}


def resolve_revision(revision="main"):
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    from huggingface_hub import HfApi

    return HfApi().dataset_info(REPOSITORY, revision=revision).sha


def lock_path(fleet="control", revision="main"):
    from huggingface_hub import hf_hub_download

    resolved = resolve_revision(revision)
    path = hf_hub_download(
        REPOSITORY, FILES[fleet], repo_type="dataset", revision=resolved
    )
    return Path(path), resolved


def read_revision(path="tests/contracts.lock"):
    """Read the framework's sole fleet configuration: an immutable dataset SHA."""
    revision = Path(path).read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"{path} must contain one immutable dataset commit SHA")
    return revision
