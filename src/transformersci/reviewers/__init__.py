"""Reviewer assignment — who should review a pull request, and asking them.

`resolver` maps a file to its owners using the reviewed repository's
`.github/scripts/codeowners_for_review_action`; that file stays in the repo being reviewed,
because who owns what is that repo's decision. `assign` ranks the owners of a PR's changed files
and requests reviews from the top ones. It is exposed as the ``assign-reviewers`` console script
and driven from the reusable ``assign-reviewers.yml`` workflow in this repo.

`resolver` is also imported on its own by consumers that check their own tree offline, such as
`utils/check_reviewers.py` in huggingface/transformers — so it is importable without PyGithub, and
`assign` is deliberately not imported here.
"""

from . import resolver
from .resolver import (
    CODEOWNERS_PATH,
    owners_for_file,
    read_local_file,
    resolution_source,
)


__all__ = [
    "CODEOWNERS_PATH",
    "owners_for_file",
    "read_local_file",
    "resolution_source",
    "resolver",
]
