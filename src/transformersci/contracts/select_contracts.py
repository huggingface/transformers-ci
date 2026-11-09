"""Whether a Transformers change can select any model integration contract.

Contract selection itself is transformers-mic's (`tmic check --changed-files`, its
select_contracts.py): a model directory selects the contracts whose fixtures use
it, other library code and dependency files select all of them, and files outside
the library select none. This module keeps only the path-level gate PR CI runs
before the contracts job, without transformers-mic, a checkout or fixture
downloads, so it must stay a superset of those rules.
"""

SHARED = ("setup.py", "pyproject.toml")
LOCK = "tests/contracts.lock"
INFRASTRUCTURE = (
    "utils/contracts/",
    ".github/workflows/contracts",
    "tests/contracts/fleet",
    "tests/contracts/prototype-fleet",
)
EXPECTATIONS = "tests/contracts/expectations/"


def may_select(changed_files):
    """Whether tmic's selection can pick any contract, decided from the paths alone."""
    return any(
        path == LOCK
        or path in SHARED
        or path.startswith((*INFRASTRUCTURE, EXPECTATIONS, "src/transformers/"))
        for path in changed_files
    )
