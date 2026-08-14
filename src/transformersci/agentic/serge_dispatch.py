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

"""Shared Serge dispatch client — ``POST /tasks`` + OIDC, scheme-agnostic.

These are the low-level primitives every Grafana/CI → Serge hand-off reuses:
mint a GitHub-Actions OIDC bearer (``aud=serge``), build the task payload, POST
it, and poll the resulting job's status. They carry **no** fingerprint / marker
/ tracking-issue scheme of their own — the caller owns that and passes the
``instruction`` and ``branch_prefix`` in. See
:mod:`transformersci.agentic.integration_failure_triage` and
:mod:`transformersci.agentic.dashboard_failure_triage` for the two callers.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class SergeDispatchError(Exception):
    """A single ``POST /tasks`` failed. Raised (not ``SystemExit``) so a
    fan-out loop can record the failure and continue to the next task."""


def build_task_payload(
    repo: str,
    base_ref: str,
    instruction: str,
    context: str,
    title: str | None,
    *,
    branch_prefix: str,
    existing_pr: int | None = None,
    tracking_issue: int | None = None,
    slack_channel: str | None = None,
    notify_pr_created: bool = True,
    notify_task_finished: bool = False,
    test_links: dict[str, list[dict[str, str]]] | None = None,
) -> dict:
    """Assemble the ``POST /tasks`` body.

    When ``existing_pr`` is set the task updates that PR (``existing_pr`` mode);
    otherwise it opens a new PR on a branch under ``branch_prefix``. The caller
    supplies both ``instruction`` (the trusted task directive) and ``context``
    (the untrusted failure report), keeping this function free of any
    per-source wording.

    ``test_links`` (node-id → ``[{label, url}]``, built by
    :mod:`transformersci.agentic.pr_evidence`) is where each failing test can be
    watched. Serge renders the entries for the group its patch fixes and knows
    nothing about what they point at; an older Serge ignores the field."""
    if existing_pr is not None:
        output: dict = {"mode": "existing_pr", "pr_number": existing_pr}
    else:
        output = {"mode": "new_pr", "branch_prefix": branch_prefix}
    if title:
        output["title"] = title
    payload = {
        "repo": repo,
        "base_ref": base_ref,
        "instruction": instruction,
        "context": context,
        "output": output,
    }
    # A no_fix task opens no PR, so a PR-driven reconciler never links it. Tell
    # Serge the tracking issue so it comments the outcome there directly.
    if tracking_issue is not None:
        payload["tracking_issue"] = tracking_issue
    # Omitted rather than sent empty: no links is the same as no field, and an
    # absent key keeps the payload readable in the action log.
    if test_links:
        payload["test_links"] = test_links
    notifications: dict[str, str | bool] = {
        "pr_created": notify_pr_created,
        "task_finished": notify_task_finished,
    }
    if slack_channel:
        notifications["slack_channel"] = slack_channel
    if slack_channel or notify_task_finished or not notify_pr_created:
        payload["notifications"] = notifications
    return payload


def dispatch_to_serge(
    serge_url: str, token: str, payload: dict, timeout: int = 240
) -> dict:
    """POST the task to Serge. Returns the parsed 202 response body."""
    url = serge_url.rstrip("/") + "/tasks"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise SergeDispatchError(
            f"Serge POST /tasks failed: {e.code} {e.reason}\n{detail}"
        )
    except urllib.error.URLError as e:
        raise SergeDispatchError(f"could not reach Serge at {url}: {e.reason}")


def poll_serge_status(
    serge_url: str, token: str, repo: str, job_id: str, timeout: int = 15
) -> str | None:
    """Best-effort GET of a task's status from Serge, OIDC-authorized with the
    same bearer used to dispatch (``GET /tasks/{owner}/{repo}/{job_id}/status``).

    Returns the status string (``running`` / ``published`` / ``no_fix`` /
    ``error`` / …) or ``None`` on any error — the caller treats ``None`` as
    "unknown, try again later" and never fails the run over it."""
    detail = poll_serge_task(serge_url, token, repo, job_id, timeout=timeout)
    status = detail.get("status") if detail else None
    return str(status) if status else None


def poll_serge_task(
    serge_url: str, token: str, repo: str, job_id: str, timeout: int = 15
) -> dict:
    """Best-effort GET of a task's machine-readable status payload.

    Unlike :func:`poll_serge_status`, this returns the whole JSON body, including
    ``error``. Callers that need to distinguish retryable provider failures
    from permanent task failures can inspect that text.
    """
    url = f"{serge_url.rstrip('/')}/tasks/{repo}/{job_id}/status"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8") or "{}")
    except (urllib.error.HTTPError, urllib.error.URLError, ValueError):
        return {}
    return body if isinstance(body, dict) else {}


def mint_serge_oidc_token() -> str | None:
    """Re-mint a fresh Serge-audience OIDC token from the GitHub Actions token
    service (the same exchange the workflow's mint step does in bash). Used to
    refresh the bearer during a long reconcile poll, since the token minted at
    start can expire before every task finishes. Returns ``None`` outside
    Actions (no request env) so callers fall back to the initial token."""
    base = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL")
    rtok = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    if not base or not rtok:
        return None
    req = urllib.request.Request(
        f"{base}&audience=serge", headers={"Authorization": f"Bearer {rtok}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8")).get("value") or None
    except (urllib.error.HTTPError, urllib.error.URLError, ValueError):
        return None
