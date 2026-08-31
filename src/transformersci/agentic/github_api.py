# Copyright 2026 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared GitHub REST helpers used to back the "GitHub is the ledger" pattern.

Both Serge dispatchers use open PRs and marker-carrying issues as their durable
state (no server-side DB of their own). These are the scheme-agnostic
primitives for that: list open PRs, match a PR by an HTML marker or branch,
find/create/patch a marker-carrying issue, and search issues+PRs for prior
human work. The *marker text* and *fingerprint scheme* belong to the caller.

Every call is best-effort: on any GitHub error it prints a warning and returns
an empty / ``None`` result so a missing token or a transient API failure never
crashes a triage run — the worst case is "acted as if there was no prior work".
"""

from __future__ import annotations

import datetime
import json
import urllib.error
import urllib.parse
import urllib.request

_API = "https://api.github.com"


def gh_headers(github_token: str | None) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    return headers


def compare_commits(
    repo: str, base: str, head: str, github_token: str | None = None
) -> dict | None:
    """``GET /repos/{repo}/compare/{base}...{head}``, or ``None`` when the range
    cannot be resolved.

    Used to size and enumerate the commits between two daily-CI runs. A daily
    run's commit is **not guaranteed to still exist** — a force-push or a
    never-landed merge-queue commit leaves a sha that GitHub answers with 422
    ``No commit found`` (observed for ``918dbf1``, the 2026-08-13 run). That is a
    normal outcome here, not an error: the caller widens the bracket to a
    neighbouring day or drops it."""
    if "/" not in repo:
        return None
    owner, name = repo.split("/", 1)
    url = f"{_API}/repos/{owner}/{name}/compare/{urllib.parse.quote(base)}...{urllib.parse.quote(head)}"
    req = urllib.request.Request(url, headers=gh_headers(github_token))
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code not in (404, 422):
            print(
                f"      warning: could not compare {base}...{head}: {e.code}",
                flush=True,
            )
        return None
    except (urllib.error.URLError, ValueError) as e:
        print(f"      warning: could not compare {base}...{head}: {e}", flush=True)
        return None


def list_open_pulls(repo: str, github_token: str | None) -> list[dict]:
    """All open PRs for ``repo`` (paginated). Returns ``[]`` on error so the
    caller treats 'could not check' the same as 'no existing PR'."""
    if "/" not in repo:
        return []
    owner, name = repo.split("/", 1)
    headers = gh_headers(github_token)

    pulls_all: list[dict] = []
    page = 1
    while True:
        params = urllib.parse.urlencode(
            {"state": "open", "per_page": 100, "page": page}
        )
        url = f"{_API}/repos/{owner}/{name}/pulls?{params}"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                pulls = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            print(
                f"      warning: could not query open PRs for task state: {e.code} {detail}",
                flush=True,
            )
            return pulls_all
        except urllib.error.URLError as e:
            print(
                f"      warning: could not query open PRs for task state: {e.reason}",
                flush=True,
            )
            return pulls_all
        if not pulls:
            return pulls_all
        pulls_all.extend(pulls)
        page += 1


def list_recent_pulls(
    repo: str,
    github_token: str | None,
    *,
    lookback_days: int = 90,
    max_pages: int = 25,
) -> list[dict]:
    """Recent PRs for ``repo`` in **every** state (open, closed, merged).

    ``list_open_pulls`` cannot answer "has this been attempted before?": the
    moment a PR is closed it vanishes from that listing, so a caller using open
    PRs as its ledger re-does work it already tried. Widening that call to
    ``state=all`` is not an option on a busy repository — it would page through
    every PR ever opened — so this walks newest-updated-first and stops at
    ``lookback_days``, which bounds the fetch to a window that still covers any
    attempt recent enough to matter.

    Sorted by ``updated``, so a stale closed PR nobody has touched falls out of
    the window even if it is newer than the cutoff by creation date. That is the
    intended trade: this answers "recently attempted", not "ever attempted".

    Best-effort, like ``list_open_pulls``: on any error it returns what it has so
    a caller treats "could not check" as "nothing found" rather than crashing the
    run.
    """
    if "/" not in repo:
        return []
    owner, name = repo.split("/", 1)
    headers = gh_headers(github_token)
    cutoff = (
        datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(days=lookback_days)
    ).isoformat()

    pulls_all: list[dict] = []
    for page in range(1, max_pages + 1):
        params = urllib.parse.urlencode(
            {
                "state": "all",
                "per_page": 100,
                "page": page,
                "sort": "updated",
                "direction": "desc",
            }
        )
        url = f"{_API}/repos/{owner}/{name}/pulls?{params}"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                pulls = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            print(
                f"      warning: could not query recent PRs for task state: "
                f"{e.code} {detail}",
                flush=True,
            )
            return pulls_all
        except urllib.error.URLError as e:
            print(
                f"      warning: could not query recent PRs for task state: {e.reason}",
                flush=True,
            )
            return pulls_all
        if not pulls:
            return pulls_all
        pulls_all.extend(pulls)
        # The listing is newest-updated-first, so once a page ends older than the
        # cutoff every later page is older still.
        if (pulls[-1].get("updated_at") or "") < cutoff:
            return pulls_all
    return pulls_all


def match_pr(
    pulls: list[dict], marker: str, branch_prefix: str | None = None
) -> int | None:
    """Return the number of an open PR carrying ``marker`` in its body (or a
    head branch starting with ``branch_prefix``), else ``None``."""
    for pr in pulls:
        body = pr.get("body") or ""
        head_ref = (pr.get("head") or {}).get("ref") or ""
        if marker in body or (branch_prefix and head_ref.startswith(branch_prefix)):
            return int(pr["number"])
    return None


def find_open_issue_by_marker(
    repo: str, marker: str, github_token: str | None
) -> int | None:
    """Number of the open issue whose body carries ``marker``, else ``None``.

    The issues endpoint also lists PRs, so anything with a ``pull_request``
    field is skipped. Best-effort → ``None`` on error / missing token."""
    if "/" not in repo or not github_token:
        return None
    owner, name = repo.split("/", 1)
    headers = gh_headers(github_token)
    page = 1
    while True:
        params = urllib.parse.urlencode(
            {"state": "open", "per_page": 100, "page": page}
        )
        url = f"{_API}/repos/{owner}/{name}/issues?{params}"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                items = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            print(
                f"      warning: could not list issues: {getattr(e, 'reason', e)}",
                flush=True,
            )
            return None
        if not items:
            return None
        for it in items:
            if it.get("pull_request"):
                continue
            if marker in (it.get("body") or ""):
                return int(it["number"])
        page += 1


def create_issue(
    repo: str, title: str, body: str, github_token: str | None
) -> int | None:
    """POST a new issue; return its number, or ``None`` on error / missing token."""
    if "/" not in repo or not github_token:
        return None
    owner, name = repo.split("/", 1)
    url = f"{_API}/repos/{owner}/{name}/issues"
    data = json.dumps({"title": title, "body": body}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={**gh_headers(github_token), "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return int(json.loads(resp.read().decode("utf-8"))["number"])
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        print(
            f"      warning: could not create issue: {getattr(e, 'reason', e)}",
            flush=True,
        )
        return None


def update_issue_body(
    repo: str, issue_number: int, body: str, github_token: str | None
) -> bool:
    """PATCH an existing issue's body in place. Best-effort: returns ``True`` on
    success, ``False`` on any error (the run continues without a refresh)."""
    if "/" not in repo or not github_token:
        return False
    owner, name = repo.split("/", 1)
    url = f"{_API}/repos/{owner}/{name}/issues/{issue_number}"
    data = json.dumps({"body": body}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="PATCH",
        headers={**gh_headers(github_token), "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30):
            return True
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        reason = getattr(e, "reason", e)
        print(f"      warning: could not refresh issue: {reason}", flush=True)
        return False


def search_issues(
    repo: str, terms: list[str], github_token: str | None, *, per_page: int = 30
) -> list[dict]:
    """GitHub Search API over issues **and** PRs in ``repo`` matching ANY of the
    quoted ``terms`` (OR-ed). Used as the "is a human already working on this?"
    check. Returns the raw ``items`` list, or ``[]`` on error / no terms.

    Each term is phrase-quoted so a nodeid or symbol matches literally rather
    than as separate words."""
    if "/" not in repo or not github_token:
        return []
    terms = [t for t in terms if t and t.strip()]
    if not terms:
        return []
    ors = " OR ".join(f'"{t}"' for t in terms)
    q = f"repo:{repo} in:title,body ({ors})"
    params = urllib.parse.urlencode({"q": q, "per_page": per_page})
    url = f"{_API}/search/issues?{params}"
    req = urllib.request.Request(url, headers=gh_headers(github_token))
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        print(
            f"      warning: could not search existing work: {getattr(e, 'reason', e)}",
            flush=True,
        )
        return []
    return data.get("items") or []


_NON_HUMAN_LOGINS = {"HuggingFaceDocBuilderDev"}


def is_bot_login(login: str | None) -> bool:
    """Whether a comment author is a bot rather than a reviewer.

    GitHub App accounts carry a ``[bot]`` suffix; ``HuggingFaceDocBuilderDev`` is
    a plain user account that posts the doc-preview comment on every PR, so it
    has to be named."""
    login = login or ""
    return login.endswith("[bot]") or login in _NON_HUMAN_LOGINS


def get_file_text(
    repo: str, path: str, github_token: str | None = None, ref: str = "main"
) -> str | None:
    """The text of one file in ``repo`` at ``ref``, or ``None`` when absent.

    A 404 is an ordinary answer here, not a fault: callers ask "does this model
    have a ``modular_*.py`` at all?", and most models predate modular. So a 404
    returns ``None`` silently and only other failures warn.
    """
    url = (
        f"{_API}/repos/{repo}/contents/{urllib.parse.quote(path)}"
        f"?ref={urllib.parse.quote(ref)}"
    )
    headers = gh_headers(github_token)
    # Raw bytes, not the base64 JSON envelope: the contents API inlines at most
    # 1MB of base64 and silently omits `content` above that, while the raw media
    # type has no such cap. A big modular_*.py is exactly the case that would
    # trip it.
    headers["Accept"] = "application/vnd.github.raw"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"      warning: could not fetch {repo}:{path}: {e.code}", flush=True)
    except urllib.error.URLError as e:
        print(f"      warning: could not fetch {repo}:{path}: {e.reason}", flush=True)
    return None


def _get_json(url: str, headers: dict[str, str], what: str, timeout: int = 30):
    """One best-effort GET returning parsed JSON, or ``None`` on any error."""
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        print(f"      warning: could not query {what}: {e.code} {detail}", flush=True)
    except urllib.error.URLError as e:
        print(f"      warning: could not query {what}: {e.reason}", flush=True)
    return None


def list_pr_review_feedback(
    repo: str,
    number: int,
    github_token: str | None,
    *,
    max_items: int = 6,
    timeout: int = 30,
) -> list[dict]:
    """What humans said on one PR: inline review comments, review verdicts, and
    conversation comments — newest first, bots dropped.

    Three endpoints, because the useful sentence turns up in all three and none
    of them is a superset of the others: the *specific* objection is usually an
    inline comment (``/pulls/N/comments``), the verdict that blocks the PR is a
    review state that often has an empty body (``/pulls/N/reviews``), and a
    close reason is a conversation comment (``/issues/N/comments``). A
    ``CHANGES_REQUESTED`` review is kept even with no body — the state itself is
    the signal.

    Only the ``max_items`` newest items are returned; this feeds a prompt, not an
    audit. Best-effort like every other call here: any error degrades to fewer
    items, never an exception.

    Each item is ``{pr, kind, author, created_at, state, path, line, body}`` with
    ``kind`` in ``{"inline", "review", "comment"}``."""
    if "/" not in repo:
        return []
    owner, name = repo.split("/", 1)
    headers = gh_headers(github_token)
    base = f"{_API}/repos/{owner}/{name}"
    items: list[dict] = []

    inline = (
        _get_json(
            f"{base}/pulls/{number}/comments?per_page=100",
            headers,
            f"review comments on PR #{number}",
            timeout,
        )
        or []
    )
    for c in inline:
        if is_bot_login((c.get("user") or {}).get("login")):
            continue
        items.append(
            {
                "pr": number,
                "kind": "inline",
                "author": (c.get("user") or {}).get("login") or "?",
                "created_at": c.get("created_at") or "",
                "state": "",
                "path": c.get("path"),
                "line": c.get("line") or c.get("original_line"),
                "body": c.get("body") or "",
            }
        )

    reviews = (
        _get_json(
            f"{base}/pulls/{number}/reviews?per_page=100",
            headers,
            f"reviews on PR #{number}",
            timeout,
        )
        or []
    )
    for r in reviews:
        if is_bot_login((r.get("user") or {}).get("login")):
            continue
        state = (r.get("state") or "").upper()
        body = r.get("body") or ""
        if not body and state != "CHANGES_REQUESTED":
            continue
        items.append(
            {
                "pr": number,
                "kind": "review",
                "author": (r.get("user") or {}).get("login") or "?",
                "created_at": r.get("submitted_at") or "",
                "state": state,
                "path": None,
                "line": None,
                "body": body,
            }
        )

    convo = (
        _get_json(
            f"{base}/issues/{number}/comments?per_page=100",
            headers,
            f"conversation comments on PR #{number}",
            timeout,
        )
        or []
    )
    for c in convo:
        if is_bot_login((c.get("user") or {}).get("login")):
            continue
        items.append(
            {
                "pr": number,
                "kind": "comment",
                "author": (c.get("user") or {}).get("login") or "?",
                "created_at": c.get("created_at") or "",
                "state": "",
                "path": None,
                "line": None,
                "body": c.get("body") or "",
            }
        )

    items.sort(key=lambda i: i.get("created_at") or "", reverse=True)
    return items[:max_items]
