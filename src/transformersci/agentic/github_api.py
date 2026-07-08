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
