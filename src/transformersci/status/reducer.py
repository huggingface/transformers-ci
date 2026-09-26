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
"""Merge one GitHub update into the current state of a run or job.

GitHub delivers webhooks at least once and in no guaranteed order, so the rules
here decide what a late or repeated event may change:

* status only moves forward (queued -> in_progress -> completed): an older
  ``queued`` or ``in_progress`` event arriving after ``completed`` changes
  nothing but can still fill a field the state lacks;
* a conclusion is kept verbatim, whatever GitHub calls it (``cancelled``,
  ``timed_out``, ``action_required``, ...). No status or conclusion is ever
  inferred as success;
* two ``completed`` events that disagree on the conclusion keep the first and
  mark the record ``needs_lookup``; only an authoritative update (an API read)
  replaces a terminal conclusion.

Pure functions over plain dicts, so the store and the tests share them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Every status GitHub uses for "not started yet" ranks as queued.
STATUS_RANK = {
    "requested": 1,
    "pending": 1,
    "waiting": 1,
    "queued": 1,
    "in_progress": 2,
    "completed": 3,
}
COMPLETED = "completed"


def status_rank(status: str) -> int:
    """1 queued, 2 in progress, 3 completed; 0 for a status GitHub has not
    documented, which therefore never overrides a known one."""
    return STATUS_RANK.get(status, 0)


@dataclass(frozen=True)
class RunUpdate:
    """One observation of a workflow run attempt."""

    repository: str
    run_id: int
    attempt: int
    status: str
    workflow: str = ""
    workflow_path: str = ""
    event: str = ""
    head_sha: str = ""
    head_branch: str = ""
    # owner/name the head commit lives in: a fork for a PR from a fork, which
    # is how reconciliation finds the PR GitHub left out of ``pull_requests``.
    head_repository: str = ""
    prs: tuple[int, ...] = ()
    conclusion: str = ""
    created_at: float | None = None
    started_at: float | None = None
    updated_at: float | None = None
    html_url: str = ""
    # GitHub's title for the run: the PR title on a pull_request run, which is
    # all the PR page's header has before the exporter has seen a trace.
    display_title: str = ""

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.repository, self.run_id, self.attempt)


@dataclass(frozen=True)
class JobUpdate:
    """One observation of a job of a run attempt."""

    repository: str
    run_id: int
    attempt: int
    job_id: int
    status: str
    name: str = ""
    workflow: str = ""
    head_sha: str = ""
    head_branch: str = ""
    conclusion: str = ""
    created_at: float | None = None
    started_at: float | None = None
    completed_at: float | None = None
    html_url: str = ""

    @property
    def key(self) -> tuple[str, int]:
        return (self.repository, self.job_id)

    def run_stub(self) -> RunUpdate:
        """What a job event says about its run: that it exists and has got at
        least as far as the job — but a finished job never finishes the run."""
        run_status = "in_progress" if status_rank(self.status) >= 2 else self.status
        return RunUpdate(
            repository=self.repository,
            run_id=self.run_id,
            attempt=self.attempt,
            status=run_status,
            workflow=self.workflow,
            head_sha=self.head_sha,
            head_branch=self.head_branch,
        )


# Fields that describe the run/job rather than its progress: a later event may
# fill them in when empty but never blanks them.
_RUN_FILL = (
    "workflow",
    "workflow_path",
    "event",
    "head_sha",
    "head_branch",
    "head_repository",
    "created_at",
    "started_at",
    "html_url",
    "display_title",
)
_JOB_FILL = (
    "name",
    "workflow",
    "head_sha",
    "head_branch",
    "created_at",
    "started_at",
    "html_url",
)


def _fill(state: dict, update: dict, fields: tuple[str, ...]) -> None:
    for field in fields:
        if not state.get(field) and update.get(field):
            state[field] = update[field]


def _advance(state: dict, update: dict, authoritative: bool) -> bool:
    """Apply ``update``'s progress to ``state`` in place; return whether the
    two disagree on a terminal conclusion that only a lookup can settle."""
    current, incoming = status_rank(state["status"]), status_rank(update["status"])
    if incoming > current:
        state["status"] = update["status"]
        state["conclusion"] = update["conclusion"]
        return False
    if incoming < current or incoming == 0:
        return False
    if state["status"] != COMPLETED:
        state["status"] = update["status"]
        return False
    if update["conclusion"] == state["conclusion"]:
        return False
    if authoritative and update["conclusion"]:
        state["conclusion"] = update["conclusion"]
        return False
    return bool(update["conclusion"])


def merge_run(
    current: dict | None, update: RunUpdate, *, authoritative: bool = False
) -> dict:
    """The run's state after ``update``. ``current`` is not modified."""
    incoming = asdict(update)
    incoming["prs"] = tuple(sorted(set(update.prs)))
    if current is None:
        state = dict(incoming)
        state["needs_lookup"] = False
        return state
    state = dict(current)
    conflict = _advance(state, incoming, authoritative)
    if status_rank(incoming["status"]) >= status_rank(current["status"]):
        if (incoming.get("updated_at") or 0) > (state.get("updated_at") or 0):
            state["updated_at"] = incoming["updated_at"]
    _fill(state, incoming, _RUN_FILL)
    state["prs"] = tuple(sorted(set(state.get("prs") or ()) | set(incoming["prs"])))
    state["needs_lookup"] = (
        False
        if authoritative and not conflict
        else bool(current.get("needs_lookup")) or conflict
    )
    return state


def merge_job(
    current: dict | None, update: JobUpdate, *, authoritative: bool = False
) -> dict:
    """The job's state after ``update``. ``current`` is not modified."""
    incoming = asdict(update)
    if current is None:
        state = dict(incoming)
        state["needs_lookup"] = False
        return state
    state = dict(current)
    conflict = _advance(state, incoming, authoritative)
    if state["status"] == COMPLETED and incoming["status"] == COMPLETED:
        if not state.get("completed_at") or (
            authoritative and incoming["completed_at"]
        ):
            state["completed_at"] = incoming["completed_at"] or state.get(
                "completed_at"
            )
    _fill(state, incoming, _JOB_FILL)
    state["needs_lookup"] = (
        False
        if authoritative and not conflict
        else bool(current.get("needs_lookup")) or conflict
    )
    return state
