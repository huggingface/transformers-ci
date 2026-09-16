"""Has a human already opened a pull request that fixes this failure group?

The `--max-groups` cap exists to bound agent work. Spending a slot on a group a
maintainer is already fixing is the most expensive failure in the loop: it costs
a dispatch slot, a task session, GPU minutes, and produces a pull request for
that same person to close. Deferring costs one lookup, and a deferred group does
not consume a slot at all — it goes to the tracking issue for a human to confirm.

Why this is two calls and not one
---------------------------------
`relore inflight N` is the verb built for "what already claims to close N", and
it is the only one that reports a claimant's **state** — open, draft, merged,
closed-by-whom. But it needs a number, and an integration-failure group has
none: it is a model, a failure mode, a terminal exception and a set of failing
test ids.

So the test ids are the way in. :func:`find_open_claims` searches the index by
test id to find the threads about this failure, then asks `inflight` about each
candidate. `search` alone cannot end the question — its hits carry kind, trust,
age and author but **not** open-vs-closed, and "already claimed" means nothing
if the claim was closed unmerged last March.

What counts as a claim
----------------------
An **open** pull request whose author is not one of our own bots. Three
exclusions, each load-bearing:

* **Merged** is not a claim. A merged fix that did not stop the failure is
  genuinely new information and must still be dispatched — the same rule
  :func:`~...integration_failure_triage.rejected_attempts_reason` follows.
* **Closed unmerged** is not a claim either; that is a human having declined,
  which is the *other* defer reason's job and is keyed on our fingerprint.
* **Our own bots** are not humans. serge's prior attempts are already handled by
  `rejected_attempts_reason`, keyed on the group fingerprint, which is a far
  more precise match than "mentions the same test".

Drafts DO count, and are labelled as such in the reason. Someone with a draft
open is someone mid-fix; dispatching alongside them duplicates the work. The
label lets the human reading the tracking issue judge it.

Failing soft is the whole contract
----------------------------------
A daemon that is down, a version skew, a timeout, an empty result and an
unparseable answer all mean the same thing here: **no reason, dispatch as
today**. A retrieval outage must never silently shrink the night's work, so
nothing in this module raises and every path returns a value. The cost of a
false negative is one wasted dispatch; the cost of a false *defer* is a real
failure nobody works on, which is silent. When unsure, dispatch.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass

#: Per-call wall clock for the client. Short on purpose: this runs once per
#: candidate group inside a nightly that has real work to do, and an answer that
#: arrives late is worth less than dispatching.
DEFAULT_TIMEOUT_SECONDS = 20

#: How many search hits become `inflight` candidates. The search is ranked, so
#: the claim — if there is one — is at the top. Three bounds the fan-out at
#: 1 + 3 calls per candidate group.
DEFAULT_MAX_CANDIDATES = 3

#: How many test ids go into one search. They AND with each other in relore, so
#: more is not better: two or three distinctive ids find the thread, and a whole
#: group's worth finds nothing.
DEFAULT_MAX_TESTS = 3

#: Accounts whose pull requests are not a *human* claim. `*[bot]` is caught
#: separately; this is for machines that present as ordinary users, which is the
#: same trap relore's own section 6.2 bot list exists for.
DEFAULT_BOT_ACCOUNTS = (
    "sergereview[bot]",
    "HuggingFaceDocBuilderDev",
    "github-actions[bot]",
)


@dataclass(frozen=True)
class Claim:
    """One open pull request that claims to close a thread about this failure."""

    number: int
    title: str
    author: str
    url: str
    draft: bool = False
    #: The thread the claim was found through — what `inflight` was asked about.
    via: int | None = None

    def describe(self) -> str:
        draft = " (draft)" if self.draft else ""
        return f"#{self.number}{draft} by @{self.author}"


def _relore(args: list[str], *, timeout: int) -> dict | None:
    """One `relore --json` call. ``None`` on any failure, never raises.

    The client is installed and matched to the daemon by
    ``relore-ensure-current`` (a workflow step); if it is missing or skewed this
    returns ``None`` and the caller dispatches, which is the safe direction.
    """
    try:
        proc = subprocess.run(
            ["relore", "--json", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout or "{}")
    except (json.JSONDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _bot_accounts() -> frozenset[str]:
    extra = (os.environ.get("ITF_RELORE_BOT_ACCOUNTS") or "").split(",")
    names = [*DEFAULT_BOT_ACCOUNTS, *(n.strip() for n in extra if n.strip())]
    return frozenset(n.lower() for n in names)


def _is_bot(author: str) -> bool:
    name = (author or "").strip().lower()
    return not name or name.endswith("[bot]") or name in _bot_accounts()


def group_tests(target: dict, *, limit: int = DEFAULT_MAX_TESTS) -> list[str]:
    """The failing test ids for a group, deduped and capped, order preserved."""
    seen: dict[str, None] = {}
    for failure in target.get("failures") or []:
        test = str(failure.get("test") or "").strip()
        if test:
            seen.setdefault(test, None)
    return list(seen)[:limit]


def find_open_claims(
    tests: list[str],
    *,
    repo: str,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[Claim]:
    """Open, non-bot pull requests claiming to close a thread about ``tests``.

    Empty on every failure path — see the module docstring: when unsure,
    dispatch.
    """
    if not tests or not repo:
        return []

    args = ["search", "--repo", repo, "--limit", str(max_candidates)]
    for test in tests:
        args += ["--test", test]
    found = _relore(args, timeout=timeout)
    if not found:
        return []

    # Candidate threads, best-ranked first, deduped: several hits routinely come
    # from different comments on the same pull request.
    candidates: list[int] = []
    for hit in found.get("hits") or []:
        number = hit.get("number")
        if isinstance(number, int) and number not in candidates:
            candidates.append(number)
        if len(candidates) >= max_candidates:
            break

    claims: list[Claim] = []
    seen: set[int] = set()
    for number in candidates:
        payload = _relore(
            ["inflight", str(number), "--repo", repo], timeout=timeout
        )
        if not payload:
            continue
        for raw in payload.get("claims") or []:
            claim = _as_open_claim(raw, via=number)
            if claim is not None and claim.number not in seen:
                seen.add(claim.number)
                claims.append(claim)
    return claims


def _as_open_claim(raw: dict, *, via: int) -> Claim | None:
    """One `inflight` claim row, if it is an open pull request by a human."""
    if not isinstance(raw, dict):
        return None
    if raw.get("type") != "pr":
        return None
    if raw.get("state") != "open" or raw.get("merged"):
        return None
    author = str(raw.get("author") or "")
    if _is_bot(author):
        return None
    number = raw.get("number")
    if not isinstance(number, int):
        return None
    return Claim(
        number=number,
        title=str(raw.get("title") or ""),
        author=author,
        url=str(raw.get("url") or ""),
        draft=bool(raw.get("draft")),
        via=via,
    )


def superseded_by_human_reason(
    target: dict,
    *,
    repo: str,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Why this group should not be dispatched, or ``""``.

    The string is what lands in ``defer_reason``, so it names the claiming pull
    requests and the thread they were found through — a human reading the
    tracking issue has to be able to check the call, because a false defer is a
    failure nobody works on and nothing else will say so.
    """
    tests = group_tests(target)
    if not tests:
        return ""
    claims = find_open_claims(
        tests, repo=repo, max_candidates=max_candidates, timeout=timeout
    )
    if not claims:
        return ""
    shown = ", ".join(c.describe() for c in claims[:3])
    more = f" +{len(claims) - 3} more" if len(claims) > 3 else ""
    via = claims[0].via
    return (
        f"a human has an open PR claiming this fix ({shown}{more}"
        f"{f', found via #{via}' if via else ''}) — confirm before dispatching again"
    )
