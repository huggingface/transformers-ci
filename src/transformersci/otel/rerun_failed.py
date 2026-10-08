"""Failure snapshot, selection and dispatch plan for targeted PR reruns.

The picker (``GET /rerun-failed/data``) and the action API
(``POST /rerun-failed``) build the same snapshot, so a selection key coming back
from the browser can only name a test the server itself found in a lane's latest
completed source run. Keep discovery independent of Grafana's latest-run
variable: that variable combines CPU and GPU into one newest run.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from datetime import datetime, timedelta

REPOSITORY = "huggingface/transformers"
SOURCE_WORKFLOWS = {
    "cpu": "pr-ci-caller.yml",
    "gpu": "self-comment-ci.yml",
}
# The transformers callers (workflow_dispatch) that run a targeted selection,
# and the ci_event each stamps into its telemetry.
RERUN_WORKFLOWS = {
    "cpu": "rerun-failed-cpu.yml",
    "gpu": "rerun-failed-gpu.yml",
}
RERUN_EVENTS = {"cpu": "rerun-failed-cpu", "gpu": "rerun-failed-gpu"}
ACTIVE_STATUSES = ("queued", "in_progress", "waiting", "requested", "pending")
MAX_CANDIDATES = 20
MAX_TESTS_PER_LANE = 100
# GitHub caps one workflow_dispatch input at 65,535 characters.
MAX_SELECTION_BYTES = 48_000

# PR CI jobs a CPU rerun can reproduce. The image, install and marker of each
# live in the reusable workflow (rerun-failed-cpu.yml), which must not take them
# from its caller; a test pins the two lists together. A job missing here is
# shown but not selectable.
CPU_JOBS = frozenset(
    {
        "tests_torch",
        "tests_generate",
        "tests_tokenization",
        "tests_processors",
        "pipelines_torch",
        "tests_custom_tokenizers",
        "examples_torch",
        "tests_exotic_models",
        "tests_repo_utils",
        "tests_non_model",
        "tests_training_ci",
        "tests_tensor_parallel_ci",
        "tests_fsdp_ci",
        "tests_peft_integration",
    }
)
# run-slow's GPU jobs. Only run_models_gpu emits telemetry today.
GPU_JOBS = frozenset({"run_models_gpu"})
GPU_RUNNERS = {
    "single-gpu": "aws-g5-4xlarge-cache",
    "multi-gpu": "aws-g5-12xlarge-cache",
}

_NODEID = re.compile(r"^(?:tests|examples)/[^\s\x00-\x1f]+\.py::[^\x00-\x1f]+$")
_RUN_ID = re.compile(r"^[0-9]+(?::[0-9]+)?$")
_KEY = re.compile(r"^[0-9a-f]{24}$")
_SHA = re.compile(r"^[0-9a-f]{40}$")
_RUN_SLOW_COMMENT = re.compile(r"^\s*run[-_ ]slow", re.IGNORECASE)


class SelectionError(ValueError):
    """A selection the server refuses; ``code`` is the API status."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(detail or code)
        self.code = code


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def selection_key(lane: str, run_id: str, test: dict) -> str:
    """Opaque, stable id of one failure in one source run."""
    basis = "\0".join(
        (lane, run_id, test["nodeid"], test["job"], test["hardware"])
    ).encode()
    return hashlib.sha256(basis).hexdigest()[:24]


def source_candidates(
    pr: str, query: Callable[[str], list[dict]]
) -> dict[str, tuple[list[str], int]]:
    """Newest telemetry run IDs per source lane, excluding targeted reruns,
    with how many candidates there were in all."""
    expr = (
        'last_over_time(pytest_run_start_time_seconds{pr="'
        + pr
        + '",ci_event!~"rerun-failed-.*"}[90d])'
    )
    candidates: dict[str, list[tuple[float, str]]] = {"cpu": [], "gpu": []}
    for item in query(expr):
        metric = item.get("metric") if isinstance(item, dict) else None
        value = item.get("value") if isinstance(item, dict) else None
        if (
            not isinstance(metric, dict)
            or not isinstance(value, list)
            or len(value) < 2
        ):
            continue
        run_id = metric.get("run_id")
        if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
            continue
        lane = "gpu" if metric.get("ci_event") == "pr-comment" else "cpu"
        try:
            started = float(value[1])
        except (ValueError, TypeError):
            continue
        candidates[lane].append((started, run_id))
    result = {}
    for lane, items in candidates.items():
        unique = sorted(set(items), reverse=True)
        result[lane] = ([run_id for _, run_id in unique[:MAX_CANDIDATES]], len(unique))
    return result


def tested_commits(pr: str, query: Callable[[str], list[dict]]) -> dict[str, str]:
    expr = f'last_over_time(pytest_run_info{{pr="{pr}"}}[90d])'
    result = {}
    for item in query(expr):
        metric = item.get("metric") if isinstance(item, dict) else None
        if isinstance(metric, dict):
            run_id, sha = metric.get("run_id"), metric.get("commit_sha")
            if (
                isinstance(run_id, str)
                and _RUN_ID.fullmatch(run_id)
                and isinstance(sha, str)
            ):
                result[run_id] = sha if _SHA.fullmatch(sha) else ""
    return result


def _source_run(run: object, lane: str) -> bool:
    if not isinstance(run, dict):
        return False
    path = str(run.get("path") or "").split("@", 1)[0]
    if path != f".github/workflows/{SOURCE_WORKFLOWS[lane]}":
        return False
    if lane == "cpu":
        # GitHub's workflow-run API can return pull_requests=[] even for a real
        # pull_request run. The candidate came from PR-stamped telemetry; verify
        # its workflow and event here, not that unreliable response field.
        return run.get("event") in {"pull_request", "merge_group"}
    # issue_comment workflow payloads have no pull_requests association. The
    # candidate came from a PR-stamped trace; the workflow path is its second
    # independent source check.
    return run.get("event") == "issue_comment"


def _model(nodeid: str) -> str:
    path_parts = nodeid.split("::", 1)[0].split("/")
    if len(path_parts) >= 4 and path_parts[:2] == ["tests", "models"]:
        return path_parts[2]
    if len(path_parts) >= 3:
        return path_parts[1]
    return path_parts[-1].removesuffix(".py")


def _ineligible_reason(lane: str, nodeid: str, job: str, hardware: str) -> str:
    if not _NODEID.fullmatch(nodeid):
        return "job-level failure, not a single test"
    if lane == "cpu" and (job not in CPU_JOBS or hardware != "cpu"):
        return f"no targeted environment for {job} on {hardware}"
    if lane == "gpu" and (job not in GPU_JOBS or hardware not in GPU_RUNNERS):
        return f"no targeted environment for {job} on {hardware}"
    return ""


def _test(row: object, pr: str, lane: str, run_id: str) -> dict | None:
    if not isinstance(row, dict) or str(row.get("pr")) != pr:
        return None
    if row.get("status_code") != "ERROR":
        return None
    nodeid = str(row.get("test_nodeid") or "")
    job = str(row.get("test_job") or "")
    hardware = str(row.get("hardware") or "")
    if not nodeid or not job or not hardware:
        return None
    reason = _ineligible_reason(lane, nodeid, job, hardware)
    test = {
        "nodeid": nodeid,
        "job": job,
        "hardware": hardware,
        "model": _model(nodeid),
        "eligible": not reason,
        "reason": reason,
        "trace_id": str(row.get("trace_id") or ""),
    }
    test["key"] = selection_key(lane, run_id, test)
    return test


def _run_summary(run: dict, run_id: str, commit: str = "") -> dict:
    return {
        "run_id": run_id,
        "url": str(run.get("html_url") or ""),
        "status": str(run.get("status") or ""),
        "conclusion": str(run.get("conclusion") or ""),
        "commit": commit,
        "created_at": str(run.get("created_at") or ""),
        "completed_at": str(run.get("updated_at") or ""),
    }


def _listed_runs(api: Callable[[str], object], workflow: str, filters: str) -> list:
    runs = []
    for status in ACTIVE_STATUSES:
        payload = api(
            f"actions/workflows/{workflow}/runs?status={status}&per_page=100{filters}"
        )
        if not isinstance(payload, dict) or not isinstance(
            payload.get("workflow_runs"), list
        ):
            raise ValueError("GitHub run list incomplete")
        runs.extend(r for r in payload["workflow_runs"] if isinstance(r, dict))
    return runs


def _is_comment_trigger(comment: dict, run: dict) -> bool:
    """Is ``comment`` the ``run-slow`` comment that created ``run``?

    Either the bot's "Nvidia CI" reply links the run, or the comment is a
    run-slow request by the run's triggering actor made just before it."""
    body = str(comment.get("body") or "")
    if f"/actions/runs/{run.get('id')}" in body:
        return True
    actor = (run.get("triggering_actor") or {}).get("login")
    author = (comment.get("user") or {}).get("login")
    created, commented = (
        _timestamp(run.get("created_at")),
        _timestamp(comment.get("created_at")),
    )
    return bool(
        actor
        and actor == author
        and created
        and commented
        and _RUN_SLOW_COMMENT.match(body)
        and created - timedelta(seconds=120)
        <= commented
        <= created + timedelta(seconds=5)
    )


def active_runs(
    pr: str, lane: str, head_sha: str, api: Callable[[str], object]
) -> list[dict]:
    """Queued and running source runs for ``pr`` straight from GitHub.

    Telemetry only knows a run once its first span lands; a queued run-slow run
    has none for minutes, and GitHub's issue_comment run names no PR, so the PR
    is read from the comment that triggered it."""
    workflow = SOURCE_WORKFLOWS[lane]
    if lane == "cpu":
        runs = _listed_runs(api, workflow, f"&event=pull_request&head_sha={head_sha}")
        mine = [r for r in runs if r.get("head_sha") == head_sha]
    else:
        runs = _listed_runs(api, workflow, "&event=issue_comment")
        mine = []
        if runs:
            created = [t for t in (_timestamp(r.get("created_at")) for r in runs) if t]
            since = (min(created) - timedelta(minutes=10)) if created else None
            suffix = f"&since={since.strftime('%Y-%m-%dT%H:%M:%SZ')}" if since else ""
            comments = api(f"issues/{pr}/comments?per_page=100{suffix}")
            if not isinstance(comments, list):
                raise ValueError("GitHub comment list incomplete")
            comments = [c for c in comments if isinstance(c, dict)]
            mine = [r for r in runs if any(_is_comment_trigger(c, r) for c in comments)]
    seen, result = set(), []
    for run in mine:
        run_id = f"{run.get('id')}:{run.get('run_attempt') or 1}"
        if _RUN_ID.fullmatch(run_id) and run_id not in seen:
            seen.add(run_id)
            result.append(_run_summary(run, run_id))
    return result


def snapshot_version(lanes: dict) -> str:
    basis = [
        [
            lane,
            (data.get("source_run") or {}).get("run_id", ""),
            (data.get("source_run") or {}).get("commit", ""),
            sorted(t["key"] for t in data.get("tests", [])),
        ]
        for lane, data in sorted(lanes.items())
    ]
    return hashlib.sha256(json.dumps(basis).encode()).hexdigest()[:16]


def snapshot(
    pr: str,
    *,
    query: Callable[[str], list[dict]],
    api: Callable[[str], object],
    get_rows: Callable[[str], list[dict]],
) -> dict:
    """Source failures and active runs per lane, with GitHub workflow identity
    checked, plus the PR's current state. ``api`` GETs a path under the
    repository's REST root."""
    if not re.fullmatch(r"[1-9][0-9]*", pr):
        raise ValueError("invalid PR number")
    pull = api(f"pulls/{pr}")
    if not isinstance(pull, dict):
        raise ValueError("GitHub PR response incomplete")
    head = pull.get("head") if isinstance(pull.get("head"), dict) else {}
    head_sha = str(head.get("sha") or "")
    if not _SHA.fullmatch(head_sha):
        raise ValueError("GitHub PR has no head commit")
    lanes = {}
    commits = tested_commits(pr, query)
    for lane, (candidates, total) in source_candidates(pr, query).items():
        completed = None
        active: dict[str, dict] = {}
        notes = []
        for run_id in candidates:
            run = api(f"actions/runs/{run_id.split(':', 1)[0]}")
            if not _source_run(run, lane):
                continue
            summary = _run_summary(run, run_id, commits.get(run_id, ""))
            if summary["status"] in ACTIVE_STATUSES:
                active[run_id] = summary
            elif summary["status"] == "completed":
                completed = summary
                break
        if completed is None and total > len(candidates):
            notes.append(
                f"Only the newest {len(candidates)} of {total} runs were inspected."
            )
        for summary in active_runs(pr, lane, head_sha, api):
            active.setdefault(summary["run_id"], summary)
        tests = []
        if completed is not None:
            seen = set()
            for row in get_rows(completed["run_id"]):
                test = _test(row, pr, lane, completed["run_id"])
                if test is not None and test["key"] not in seen:
                    tests.append(test)
                    seen.add(test["key"])
            if not tests and completed["conclusion"] == "failure":
                notes.append(
                    "GitHub reports this run failed, but its telemetry has no failed tests."
                )
            if len(tests) > MAX_TESTS_PER_LANE:
                notes.append(
                    f"Showing {MAX_TESTS_PER_LANE} of {len(tests)} failures; "
                    "re-run the full CI instead."
                )
                tests = tests[:MAX_TESTS_PER_LANE]
        lanes[lane] = {
            "source_run": completed,
            "active_runs": sorted(active.values(), key=lambda r: r["run_id"]),
            "tests": tests,
            "complete": not notes,
            "notes": notes,
        }
    return {
        "pr": pr,
        "pr_state": "merged" if pull.get("merged_at") else str(pull.get("state") or ""),
        "head_sha": head_sha,
        "version": snapshot_version(lanes),
        "lanes": lanes,
    }


def plan(snap: dict, keys: object) -> dict[str, dict]:
    """The per-lane dispatch payload for ``keys``; raises SelectionError.

    Every key must name an eligible failure in this snapshot, so a selection
    can only narrow what the server found, never broaden it."""
    if (
        not isinstance(keys, list)
        or not keys
        or len(keys) > 2 * MAX_TESTS_PER_LANE
        or not all(isinstance(k, str) and _KEY.fullmatch(k) for k in keys)
        or len(set(keys)) != len(keys)
    ):
        raise SelectionError("invalid_selection")
    by_key = {
        test["key"]: (lane, test)
        for lane, data in snap["lanes"].items()
        for test in data["tests"]
    }
    lanes: dict[str, dict[tuple[str, str], list[str]]] = {}
    for key in keys:
        found = by_key.get(key)
        if found is None:
            raise SelectionError(
                "stale_selection", "a selected test is no longer listed"
            )
        lane, test = found
        if not test["eligible"]:
            raise SelectionError("unsupported_selection", test["reason"])
        group = lanes.setdefault(lane, {}).setdefault(
            (test["job"], test["hardware"]), []
        )
        group.append(test["nodeid"])
    result = {}
    for lane, groups in lanes.items():
        data = snap["lanes"][lane]
        if not data["complete"]:
            raise SelectionError("incomplete_source", " ".join(data["notes"]))
        selection = {
            "v": 1,
            "groups": [
                {"job": job, "hardware": hardware, "tests": sorted(tests)}
                for (job, hardware), tests in sorted(groups.items())
            ],
        }
        encoded = json.dumps(selection, separators=(",", ":"))
        if len(encoded) > MAX_SELECTION_BYTES:
            raise SelectionError("selection_too_large")
        result[lane] = {
            "selection": encoded,
            "count": sum(len(g["tests"]) for g in selection["groups"]),
            "source_run": data["source_run"],
            "active_runs": data["active_runs"],
        }
    return result


PAGE_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Re-run failed tests</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#181b1f;color:#d8d9da;font:14px/1.5 system-ui,sans-serif}
main{max-width:960px;margin:0 auto;padding:20px}h1{font-size:20px;margin:0 36px 4px 0}h2{font-size:17px;margin:22px 0 7px}
.close{position:fixed;top:8px;right:12px;z-index:2;background:#242b35;color:#d8d9da;border:1px solid #657083;border-radius:4px;width:28px;height:28px;padding:0;font:20px/20px system-ui;cursor:pointer}
.muted{color:#9699a0}.intro{margin:0 0 20px}.card{border:1px solid #33363c;background:#15171a;border-radius:8px;padding:16px;margin:16px 0}
.meta{display:flex;flex-wrap:wrap;gap:12px;margin:3px 0 12px;font-size:12px;color:#a9abb1}a{color:#79baff}
.group{border-top:1px solid #303238;padding:10px 0}.group summary{display:flex;align-items:center;gap:10px;cursor:pointer;list-style:none}
.group summary::-webkit-details-marker{display:none}.group summary:before{content:'▸';color:#a9abb1}.group[open] summary:before{content:'▾'}
.group summary strong{min-width:100px}.group .count{color:#9699a0;font-size:12px}.tests{margin:8px 0 0 24px;display:grid;gap:7px}
.test{display:flex;gap:8px;align-items:flex-start;overflow-wrap:anywhere}.test code{font-size:12px}.test small{color:#9699a0;margin-left:5px}
input[type=checkbox]{accent-color:#72aaff;margin-top:4px}.warning{border-color:#a06734;background:#2b2118}.warning strong{color:#ffce92}
.note{border-color:#6b5a2a;background:#26221a}.error{border-color:#8f3d3d;background:#2a1a1a}.ok{border-color:#2f6b45;background:#17251c}
.runs-list{margin:6px 0 0;padding-left:18px}.actions{display:flex;gap:10px;margin-top:12px}
footer{position:sticky;bottom:0;display:flex;align-items:center;justify-content:space-between;gap:16px;background:#181b1f;border-top:1px solid #33363c;padding:14px 0}
button{border:1px solid #5777a5;border-radius:5px;background:#30538a;color:white;padding:8px 14px;font:inherit;cursor:pointer}button:disabled{opacity:.52;cursor:not-allowed}
button.secondary{background:#242b35;border-color:#657083}
</style></head><body><button id="close" class="close" type="button" aria-label="Close">×</button><main><h1>Re-run failed tests</h1><p class="intro muted">Choose exact failures from each lane’s latest completed run. A re-run tests the PR’s current merge commit and runs only the selected tests: at most one CPU and one GPU workflow run.</p>
<div id="message" role="status">Loading failures…</div><div id="action"></div><div id="confirm"></div><div id="runs"></div>
<footer><span id="selected">0 tests selected</span><button id="submit" disabled>Run selected tests</button></footer>
</main><script>
const params=new URLSearchParams(location.search),pr=params.get('pr'),runs=document.getElementById('runs'),msg=document.getElementById('message'),selected=document.getElementById('selected'),submit=document.getElementById('submit'),actionBox=document.getElementById('action'),confirmBox=document.getElementById('confirm');
function closePicker(){if(parent!==window)parent.postMessage({type:'tci-rerun-close'},location.origin);else history.back()}
document.getElementById('close').addEventListener('click',closePicker);document.addEventListener('keydown',e=>{if(e.key==='Escape')closePicker()});
const labels={cpu:'CPU · PR CI',gpu:'GPU · run-slow'},lanes={cpu:'CPU',gpu:'GPU'};
let snapshot=null,dispatch={enabled:false,reason:''},pending=null,polling=null,busy=false;
function el(tag,cls,content){const n=document.createElement(tag);if(cls)n.className=cls;if(content!==undefined)n.textContent=content;return n}
function link(parent,href,label){const a=el('a','',label);if(/^https:\/\/github\.com\//.test(href||''))a.href=href;a.target='_blank';a.rel='noopener noreferrer';parent.append(a);return a}
function runLink(run){return run.url||('https://github.com/huggingface/transformers/actions/runs/'+String(run.run_id||run.id).split(':')[0])}
function checkedKeys(){return [...runs.querySelectorAll('.test input:checked')].map(c=>c.dataset.key)}
function update(){const counts={cpu:0,gpu:0};for(const c of runs.querySelectorAll('.test input:checked'))counts[c.dataset.lane]++;const n=counts.cpu+counts.gpu;
 selected.textContent=n+' test'+(n===1?'':'s')+' selected'+(n?' ('+counts.cpu+' CPU · '+counts.gpu+' GPU)':'');
 submit.disabled=busy||!dispatch.enabled||!n;submit.title=dispatch.enabled?'':dispatch.reason}
function showLane(lane,data){const card=el('section','card'),heading=el('h2','',labels[lane]);card.append(heading);
 for(const note of data.notes||[]){card.append(el('div','card note',note))}
 const source=data.source_run;if(!source){card.append(el('p','muted','No completed source run found.'));runs.append(card);return}
 const meta=el('div','meta');link(meta,runLink(source),'Run '+source.run_id);if(source.commit&&/^[0-9a-f]{40}$/.test(source.commit)){const c=el('span');c.textContent='Tested commit ';link(c,'https://github.com/huggingface/transformers/commit/'+source.commit,source.commit.slice(0,12));meta.append(c)}
 if(source.completed_at)meta.append(el('span','',source.completed_at));card.append(meta);
 if(data.active_runs.length){const warning=el('div','card warning');warning.append(el('strong','','Ongoing runs'));warning.append(el('p','muted','Re-running tests from this lane cancels them, after you confirm:'));for(const active of data.active_runs){link(warning,runLink(active),'Run '+active.run_id+' ('+active.status+')');warning.append(' ')}card.append(warning)}
 const eligible=data.tests.filter(t=>t.eligible),ineligible=data.tests.filter(t=>!t.eligible);
 if(!data.tests.length)card.append(el('p','muted','No failed tests found in this run.'));
 const groups=new Map();for(const test of eligible){if(!groups.has(test.model))groups.set(test.model,[]);groups.get(test.model).push(test)}
 for(const [model,tests] of [...groups].sort((a,b)=>a[0].localeCompare(b[0]))){const details=el('details','group'),summary=el('summary'),master=document.createElement('input');master.type='checkbox';master.checked=true;master.setAttribute('aria-label','Select all '+model+' tests');master.addEventListener('click',e=>e.stopPropagation());summary.append(master,el('strong','',model),el('span','count',tests.length+' failed'));details.append(summary);
  const list=el('div','tests'),checks=[];for(const test of tests){const row=el('label','test'),check=document.createElement('input');check.type='checkbox';check.checked=true;check.dataset.key=test.key;check.dataset.lane=lane;checks.push(check);const body=el('span');body.append(el('code','',test.nodeid),el('small','',test.job+' · '+test.hardware));row.append(check,body);list.append(row);check.addEventListener('change',()=>{master.checked=checks.every(x=>x.checked);master.indeterminate=!master.checked&&checks.some(x=>x.checked);update()})}
  master.addEventListener('change',()=>{for(const check of checks)check.checked=master.checked;master.indeterminate=false;update()});details.append(list);card.append(details)}
 if(ineligible.length){const details=el('details','group'),summary=el('summary');summary.append(el('strong','','Not selectable'),el('span','count',ineligible.length+' failure'+(ineligible.length===1?'':'s')));details.append(summary);const list=el('div','tests');for(const test of ineligible){const row=el('div','test'),body=el('span');body.append(el('code','',test.nodeid),el('small','',test.reason));row.append(body);list.append(row)}details.append(list);card.append(details)}
 runs.append(card)}
const stateText={prepared:'Preparing',cancelling:'Cancelling ongoing runs…',dispatching:'Starting targeted runs…',dispatched:'Running',completed:'Finished',failed:'Stopped'};
function showAction(action){actionBox.replaceChildren();if(!action)return;const card=el('section','card '+(action.state==='failed'?'error':action.state==='completed'?'ok':'note'));
 card.append(el('strong','',stateText[action.state]||action.state));const meta=el('div','meta');meta.append(el('span','','Requested by '+action.actor),el('span','',new Date(action.created*1000).toLocaleString()));card.append(meta);
 if(action.error)card.append(el('p','',action.error));
 if(action.cancel.length){const ul=el('ul','runs-list');for(const c of action.cancel){const li=el('li');link(li,c.url,'Run '+c.run_id);li.append(' — '+c.result);ul.append(li)}card.append(el('div','muted','Cancellations'),ul)}
 const ul=el('ul','runs-list');for(const [lane,data] of Object.entries(action.lanes)){const li=el('li');li.append(lanes[lane]+': '+data.count+' test'+(data.count===1?'':'s')+' — ');if(data.run){link(li,data.run.url,'run '+data.run.id);li.append(' '+data.run.status+(data.run.conclusion?' · '+data.run.conclusion:''))}else li.append(data.dispatch);ul.append(li)}card.append(el('div','muted','Re-runs'),ul);
 actionBox.append(card);const live=['prepared','cancelling','dispatching','dispatched'].includes(action.state),wasBusy=busy;busy=['prepared','cancelling','dispatching'].includes(action.state);update();
 if(wasBusy&&!busy&&!pending)load();
 clearTimeout(polling);if(live)polling=setTimeout(()=>poll(action.id),5000)}
function poll(id){fetch('/rerun-failed/actions/'+encodeURIComponent(id),{cache:'no-store'}).then(r=>r.ok?r.json():Promise.reject()).then(d=>showAction(d.action)).catch(()=>{polling=setTimeout(()=>poll(id),10000)})}
function showConfirm(active){confirmBox.replaceChildren();const card=el('section','card warning');card.append(el('strong','','Are you sure? This will cancel ongoing runs'));const ul=el('ul','runs-list');for(const run of active){const li=el('li');li.append(lanes[run.lane]+': ');link(li,runLink(run),'Run '+run.run_id);li.append(' ('+run.status+')');ul.append(li)}card.append(ul);
 const row=el('div','actions'),yes=el('button','','Cancel these runs and re-run'),no=el('button','secondary','Keep them running');yes.type=no.type='button';
 yes.addEventListener('click',()=>{confirmBox.replaceChildren();send(active.map(r=>r.run_id))});no.addEventListener('click',()=>{confirmBox.replaceChildren();pending=null;busy=false;update()});row.append(yes,no);card.append(row);confirmBox.append(card);yes.focus()}
function fail(text,signIn){msg.replaceChildren(el('span','',text));if(signIn){msg.append(' ');const a=el('a','','Sign in');a.href='/login';a.target='_top';msg.append(a)}busy=false;update()}
function send(confirmActive){busy=true;update();msg.textContent='Submitting…';
 fetch('/rerun-failed',{method:'POST',cache:'no-store',headers:{'Content-Type':'application/json','X-TCI-Action':'1'},body:JSON.stringify({pr,version:snapshot.version,keys:pending.keys,confirm_active:confirmActive,idempotency_key:pending.idem})})
 .then(async r=>{let d={};try{d=await r.json()}catch(e){}return [r.status,d]}).then(([status,d])=>{
  if(status===200||status===202){msg.textContent='PR #'+pr;pending=null;showAction(d.action);return}
  if(status===409&&d.status==='confirm'){msg.textContent='PR #'+pr;showConfirm(d.active_runs);return}
  if(status===409&&d.status==='busy'){msg.textContent='Another re-run of this PR is still starting.';pending=null;showAction(d.action);return}
  if(status===409&&d.status==='stale'){pending=null;fail('The failure list changed. Reloading…');load();return}
  if(status===401)return fail('Sign in to re-run tests.',true);
  if(status===403)return fail(d.status==='no_write_access'?'Re-running needs write access to huggingface/transformers.':'Request refused.');
  if(status===429)return fail('Too many re-runs recently. Try again later.');
  if(d.status==='github_rate_limited'){pending=null;return fail('GitHub API rate limit reached. Try again later.')}
  pending=null;fail(d.detail||'Could not start the re-run ('+(d.status||status)+').')}).catch(()=>fail('Could not reach the server. Please try again.'))}
submit.addEventListener('click',()=>{const keys=checkedKeys();if(!keys.length||!snapshot)return;pending={keys,idem:(crypto.randomUUID?crypto.randomUUID():String(Date.now())+Math.random())};send([])});
function load(){fetch('/rerun-failed/data?pr='+encodeURIComponent(pr),{cache:'no-store'}).then(async r=>{if(!r.ok){const d=await r.json().catch(()=>({}));if(d.status==='github_rate_limited')throw Error('GitHub API rate limit reached'+(d.retry_after>=0?'; try again in '+Math.max(1,Math.ceil(d.retry_after/60))+' min.':'. Try again later.'));throw Error('Could not load failures. Please refresh and try again.')}return r.json()}).then(data=>{snapshot=data;dispatch=data.dispatch||dispatch;msg.textContent='PR #'+pr+(data.pr_state&&data.pr_state!=='open'?' is '+data.pr_state+': re-runs are only for open PRs.':'');
 if(data.pr_state!=='open')dispatch={enabled:false,reason:'The PR is not open.'};else if(!dispatch.enabled&&dispatch.reason)msg.textContent='PR #'+pr+' · '+dispatch.reason;
 runs.replaceChildren();showLane('cpu',data.lanes.cpu);showLane('gpu',data.lanes.gpu);showAction(data.latest_action);update()}).catch(e=>{msg.textContent=e.message})}
if(!/^[1-9][0-9]*$/.test(pr||'')){msg.textContent='Open this page from a PR dashboard.'}else load();
</script></body></html>"""
