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
"""Verify a GitHub webhook delivery and turn it into state updates.

Only ``workflow_run`` and ``workflow_job`` are understood; anything else is
acknowledged and ignored. A delivery is accepted only when

* its ``X-Hub-Signature-256`` is the HMAC-SHA256 of the raw body under the
  shared secret (checked before the body is parsed);
* its repository is on the explicit allow-list AND GitHub reports it public.
  The dashboard is served anonymously, so a private repository's runs must
  never be stored, even if someone adds it to the list by mistake;
* its workflow is one of the configured CI workflows, so a docs or security
  workflow on the same PR cannot become the PR page's "latest run".
"""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from datetime import datetime

from .reducer import JobUpdate, RunUpdate

SUPPORTED_EVENTS = frozenset({"workflow_run", "workflow_job"})

# merge_group runs carry the PR only in their ref,
# e.g. "gh-readonly-queue/main/pr-48976-0123abcd".
_MERGE_GROUP_PR = re.compile(r"(?:^|/)pr-(\d+)(?:-|$)")


class Ignored(Exception):
    """A well-formed delivery this service deliberately does not store."""


class Rejected(Exception):
    """A delivery that must not be trusted (bad signature, malformed body)."""


@dataclass(frozen=True)
class Filters:
    repositories: frozenset[str]
    workflows: frozenset[str]


def verify_signature(secret: bytes, body: bytes, header: str | None) -> bool:
    """Constant-time check of GitHub's ``sha256=<hex>`` signature header."""
    if not secret or not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(secret, body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header[len("sha256=") :])


def parse_timestamp(value: object) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _int(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Rejected(f"{what} is missing or not an integer")
    return value


def _check_repository(payload: dict, filters: Filters) -> str:
    repository = payload.get("repository")
    if not isinstance(repository, dict):
        raise Rejected("no repository")
    name = _text(repository.get("full_name"))
    if name not in filters.repositories:
        raise Ignored(f"repository {name!r} is not on the allow-list")
    if repository.get("private") is not False:
        raise Ignored(f"repository {name!r} is not public")
    return name


def _check_workflow(workflow: str, filters: Filters) -> None:
    if workflow not in filters.workflows:
        raise Ignored(f"workflow {workflow!r} is not followed")


def pull_request_numbers(
    pull_requests: object, event: str, head_branch: str
) -> tuple[int, ...]:
    """PR numbers of a run. GitHub leaves ``pull_requests`` empty for a PR from
    a fork, so an empty result means "unknown" and is enriched later — it never
    discards the event. A merge-queue run names its PR in its ref."""
    numbers: set[int] = set()
    if isinstance(pull_requests, list):
        for item in pull_requests:
            number = item.get("number") if isinstance(item, dict) else None
            if isinstance(number, int) and not isinstance(number, bool):
                numbers.add(number)
    if not numbers and event == "merge_group":
        match = _MERGE_GROUP_PR.search(head_branch)
        if match:
            numbers.add(int(match.group(1)))
    return tuple(sorted(numbers))


def parse_workflow_run(payload: dict, filters: Filters) -> RunUpdate:
    repository = _check_repository(payload, filters)
    run = payload.get("workflow_run")
    if not isinstance(run, dict):
        raise Rejected("workflow_run event without a workflow_run")
    workflow = _text(run.get("name"))
    _check_workflow(workflow, filters)
    event = _text(run.get("event"))
    head_branch = _text(run.get("head_branch"))
    return RunUpdate(
        repository=repository,
        run_id=_int(run.get("id"), "workflow_run.id"),
        attempt=_int(run.get("run_attempt", 1), "workflow_run.run_attempt"),
        status=_text(run.get("status")),
        workflow=workflow,
        workflow_path=_text(run.get("path")),
        event=event,
        head_sha=_text(run.get("head_sha")),
        head_branch=head_branch,
        head_repository=_text((run.get("head_repository") or {}).get("full_name"))
        if isinstance(run.get("head_repository"), dict)
        else "",
        prs=pull_request_numbers(run.get("pull_requests"), event, head_branch),
        conclusion=_text(run.get("conclusion")),
        created_at=parse_timestamp(run.get("created_at")),
        started_at=parse_timestamp(run.get("run_started_at")),
        updated_at=parse_timestamp(run.get("updated_at")),
        html_url=_text(run.get("html_url")),
        display_title=_text(run.get("display_title")),
    )


def parse_workflow_job(payload: dict, filters: Filters) -> JobUpdate:
    repository = _check_repository(payload, filters)
    job = payload.get("workflow_job")
    if not isinstance(job, dict):
        raise Rejected("workflow_job event without a workflow_job")
    workflow = _text(job.get("workflow_name"))
    _check_workflow(workflow, filters)
    return JobUpdate(
        repository=repository,
        run_id=_int(job.get("run_id"), "workflow_job.run_id"),
        attempt=_int(job.get("run_attempt", 1), "workflow_job.run_attempt"),
        job_id=_int(job.get("id"), "workflow_job.id"),
        status=_text(job.get("status")),
        name=_text(job.get("name")),
        workflow=workflow,
        head_sha=_text(job.get("head_sha")),
        head_branch=_text(job.get("head_branch")),
        conclusion=_text(job.get("conclusion")),
        created_at=parse_timestamp(job.get("created_at")),
        started_at=parse_timestamp(job.get("started_at")),
        completed_at=parse_timestamp(job.get("completed_at")),
        html_url=_text(job.get("html_url")),
    )


def parse_delivery(
    event: str, payload: object, filters: Filters
) -> RunUpdate | JobUpdate:
    """The update a verified delivery carries; raises Ignored or Rejected."""
    if event not in SUPPORTED_EVENTS:
        raise Ignored(f"event {event!r} is not followed")
    if not isinstance(payload, dict):
        raise Rejected("payload is not a JSON object")
    if event == "workflow_run":
        return parse_workflow_run(payload, filters)
    return parse_workflow_job(payload, filters)
