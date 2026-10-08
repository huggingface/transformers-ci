"""Read-only failure snapshot for targeted PR reruns.

The selection and dispatch APIs will use the same source run IDs and exact
node IDs. Keep discovery here independent of Grafana's latest-run variable:
that variable combines CPU and GPU into one newest run.
"""

from __future__ import annotations

import re
from collections.abc import Callable

REPOSITORY = "huggingface/transformers"
SOURCE_WORKFLOWS = {
    "cpu": "pr-ci-caller.yml",
    "gpu": "self-comment-ci.yml",
}
ACTIVE_STATUSES = frozenset(
    {"queued", "in_progress", "requested", "waiting", "pending"}
)
_NODEID = re.compile(r"^(?:tests|examples)/[^\s\x00-\x1f]+\.py::[^\x00-\x1f]+$")
_RUN_ID = re.compile(r"^[0-9]+(?::[0-9]+)?$")


def source_candidates(
    pr: str, query: Callable[[str], list[dict]]
) -> dict[str, list[str]]:
    """Newest telemetry run IDs per source lane, excluding targeted reruns."""
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
    return {
        lane: [run_id for _, run_id in sorted(items, reverse=True)[:20]]
        for lane, items in candidates.items()
    }


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
                result[run_id] = sha if re.fullmatch(r"[0-9a-f]{40}", sha) else ""
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


def _test(row: object, pr: str) -> dict | None:
    if not isinstance(row, dict) or str(row.get("pr")) != pr:
        return None
    if row.get("status_code") != "ERROR":
        return None
    nodeid = str(row.get("test_nodeid") or "")
    job = str(row.get("test_job") or "")
    hardware = str(row.get("hardware") or "")
    if not nodeid or not job or not hardware:
        return None
    path_parts = nodeid.split("::", 1)[0].split("/")
    if len(path_parts) >= 4 and path_parts[:2] == ["tests", "models"]:
        model = path_parts[2]
    elif len(path_parts) >= 3:
        model = path_parts[1]
    else:
        model = path_parts[-1].removesuffix(".py")
    return {
        "nodeid": nodeid,
        "job": job,
        "hardware": hardware,
        "model": model,
        "eligible": bool(_NODEID.fullmatch(nodeid)),
        "trace_id": str(row.get("trace_id") or ""),
    }


def snapshot(
    pr: str,
    *,
    query: Callable[[str], list[dict]],
    get_run: Callable[[str], dict],
    get_rows: Callable[[str], list[dict]],
) -> dict:
    """Source failures and active runs, with GitHub workflow identity checked."""
    if not re.fullmatch(r"[1-9][0-9]*", pr):
        raise ValueError("invalid PR number")
    lanes = {}
    commits = tested_commits(pr, query)
    for lane, candidates in source_candidates(pr, query).items():
        completed = None
        active = []
        for run_id in candidates:
            run = get_run(run_id.split(":", 1)[0])
            if not _source_run(run, lane):
                continue
            status = str(run.get("status") or "")
            summary = {
                "run_id": run_id,
                "url": str(run.get("html_url") or ""),
                "status": status,
                "commit": commits.get(run_id, ""),
                "completed_at": str(run.get("updated_at") or ""),
            }
            if status in ACTIVE_STATUSES:
                active.append(summary)
            elif status == "completed" and completed is None:
                completed = summary
                break
        tests = []
        if completed is not None:
            seen = set()
            for row in get_rows(completed["run_id"]):
                test = _test(row, pr)
                if test is None:
                    continue
                key = (test["nodeid"], test["job"], test["hardware"])
                if key not in seen:
                    tests.append(test)
                    seen.add(key)
        lanes[lane] = {"source_run": completed, "active_runs": active, "tests": tests}
    return {"pr": pr, "lanes": lanes}


PAGE_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Re-run failed tests</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#0b0c0e;color:#d8d9da;font:14px/1.5 system-ui,sans-serif}
main{max-width:960px;margin:0 auto;padding:24px}h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:22px 0 7px}
.muted{color:#9699a0}.intro{margin:0 0 20px}.card{border:1px solid #33363c;background:#15171a;border-radius:8px;padding:16px;margin:16px 0}
.meta{display:flex;flex-wrap:wrap;gap:12px;margin:3px 0 12px;font-size:12px;color:#a9abb1}a{color:#79baff}
.group{border-top:1px solid #303238;padding:10px 0}.group summary{display:flex;align-items:center;gap:10px;cursor:pointer;list-style:none}
.group summary::-webkit-details-marker{display:none}.group summary:before{content:'▸';color:#a9abb1}.group[open] summary:before{content:'▾'}
.group summary strong{min-width:100px}.group .count{color:#9699a0;font-size:12px}.tests{margin:8px 0 0 24px;display:grid;gap:7px}
.test{display:flex;gap:8px;align-items:flex-start;overflow-wrap:anywhere}.test code{font-size:12px}.test small{color:#9699a0;margin-left:5px}
input[type=checkbox]{accent-color:#72aaff;margin-top:4px}.warning{border-color:#a06734;background:#2b2118}.warning strong{color:#ffce92}
footer{position:sticky;bottom:0;display:flex;align-items:center;justify-content:space-between;gap:16px;background:#0b0c0e;border-top:1px solid #33363c;padding:14px 0}
button{border:1px solid #5777a5;border-radius:5px;background:#30538a;color:white;padding:8px 14px;font:inherit}button:disabled{opacity:.52;cursor:not-allowed}
</style></head><body><main><h1>Re-run failed tests</h1><p class="intro muted">Choose exact failures from each lane’s latest completed run. This preview cannot launch tests yet.</p>
<div id="message" role="status">Loading failures…</div><div id="runs"></div>
<footer><span id="selected">0 tests selected</span><button disabled title="Dispatch is not connected yet">Run selected tests (coming soon)</button></footer>
</main><script>
const params=new URLSearchParams(location.search),pr=params.get('pr'),runs=document.getElementById('runs'),msg=document.getElementById('message'),selected=document.getElementById('selected');
const labels={cpu:'CPU · PR CI',gpu:'GPU · run-slow'};
function el(tag,cls,content){const n=document.createElement(tag);if(cls)n.className=cls;if(content!==undefined)n.textContent=content;return n}
function link(parent,href,label){const a=el('a','',label);a.href=href;a.target='_blank';a.rel='noopener noreferrer';parent.append(a)}
function runLink(run){return 'https://github.com/huggingface/transformers/actions/runs/'+run.run_id.split(':')[0]}
function update(){const checked=runs.querySelectorAll('.test input:checked').length;selected.textContent=checked+' test'+(checked===1?'':'s')+' selected'}
function showLane(lane,data){const card=el('section','card'),heading=el('h2','',labels[lane]);card.append(heading);
 const source=data.source_run;if(!source){card.append(el('p','muted','No completed source run found.'));runs.append(card);return}
 const meta=el('div','meta');link(meta,runLink(source),'Run '+source.run_id);if(source.commit&&/^[0-9a-f]{40}$/.test(source.commit)){const c=el('span');c.textContent='Tested commit ';link(c,'https://github.com/huggingface/transformers/commit/'+source.commit,source.commit.slice(0,12));meta.append(c)}
 if(source.completed_at)meta.append(el('span','',source.completed_at));card.append(meta);
 if(data.active_runs.length){const warning=el('div','card warning');warning.append(el('strong','','Are you sure? This will cancel ongoing runs'));warning.append(el('p','muted','Confirmation will be required when dispatch is connected. Active runs: '));for(const active of data.active_runs){link(warning,runLink(active),'Run '+active.run_id);warning.append(' ')}card.append(warning)}
 const eligible=data.tests.filter(t=>t.eligible),ineligible=data.tests.filter(t=>!t.eligible);
 if(!data.tests.length)card.append(el('p','muted','No failed tests found in this run.'));
 const groups=new Map();for(const test of eligible){if(!groups.has(test.model))groups.set(test.model,[]);groups.get(test.model).push(test)}
 for(const [model,tests] of [...groups].sort((a,b)=>a[0].localeCompare(b[0]))){const details=el('details','group'),summary=el('summary'),master=document.createElement('input');master.type='checkbox';master.setAttribute('aria-label','Select all '+model+' tests');master.addEventListener('click',e=>e.stopPropagation());summary.append(master,el('strong','',model),el('span','count',tests.length+' failed'));details.append(summary);
  const list=el('div','tests'),checks=[];for(const test of tests){const row=el('label','test'),check=document.createElement('input');check.type='checkbox';checks.push(check);const body=el('span');body.append(el('code','',test.nodeid),el('small','',test.job+' · '+test.hardware));row.append(check,body);list.append(row);check.addEventListener('change',()=>{master.checked=checks.every(x=>x.checked);master.indeterminate=!master.checked&&checks.some(x=>x.checked);update()})}
  master.addEventListener('change',()=>{for(const check of checks)check.checked=master.checked;master.indeterminate=false;update()});details.append(list);card.append(details)}
 if(ineligible.length){card.append(el('p','muted',ineligible.length+' job-level failure'+(ineligible.length===1?' is':'s are')+' unavailable for exact test reruns.'))}
 runs.append(card)}
if(!/^[1-9][0-9]*$/.test(pr||'')){msg.textContent='Open this page from a PR dashboard.'}else fetch('/rerun-failed/data?pr='+encodeURIComponent(pr),{credentials:'same-origin',cache:'no-store'}).then(async r=>{if(r.status===401)throw Error('Sign in to Grafana to view and select failed tests.');if(!r.ok)throw Error('Could not load failures. Please refresh and try again.');return r.json()}).then(data=>{msg.textContent='PR #'+pr;showLane('cpu',data.lanes.cpu);showLane('gpu',data.lanes.gpu)}).catch(e=>{msg.textContent=e.message});
</script></body></html>"""
