"""The Test page, rendered by the exporter (``GET /test``).

Everything the Grafana Test page shows across six panels, from one trace fetch:
the test's identity and links, how to reproduce it, its timing and device
memory, its failure, and (loaded after the page shows) related issues and Serge
Actions. This module only renders; ``trace_exporter`` gathers the data.

All dynamic text is escaped here; the small client script only sets
``textContent``.
"""

from __future__ import annotations

import html
import json
import time
from urllib.parse import urlencode

from transformersci.otel.related_issues import GITHUB_MARK_PATH

HARDWARE_LABELS = {
    "cpu": "CPU",
    "gpu": "GPU",
    "single-gpu": "GPU",
    "multi-gpu": "xGPU",
}

GITHUB_MARK = (
    "<svg class='gh' viewBox='0 0 16 16' width='11' height='11' aria-hidden='true' "
    f"fill='currentColor'><path d='{GITHUB_MARK_PATH}'/></svg>"
)

STYLE = """
:root{color-scheme:dark;--bg:#111217;--panel:#181b1f;--line:#2c3038;--soft:#22252b;
--text:#d8d9da;--muted:#8e9197;--link:#6ab0ff;--red:#ff7383;--green:#73bf69;--amber:#f2cc60}
*{box-sizing:border-box}
html,body{margin:0;background:var(--panel);color:var(--text);font:13px/1.5 Inter,system-ui,sans-serif}
a{color:var(--link);text-decoration:none}a:hover{text-decoration:underline}
.gh{vertical-align:-1px;margin-left:3px}
.page{width:100%;margin:0 auto;padding:16px 16px 24px}
.card{min-width:0}
.head{display:flex;gap:16px;align-items:flex-start;padding:0 0 14px;margin:0 0 16px;border-bottom:1px solid var(--line)}
.head-main{flex:1 1 auto;min-width:0}
.facts{flex:none;display:grid;grid-template-columns:auto auto;gap:4px 14px;align-items:baseline;margin:0;padding:8px 12px;border:1px solid var(--line);border-radius:5px;font-size:12px}
.facts dt{color:var(--muted)}.facts dd{margin:0;font-weight:500;white-space:nowrap}
.facts .sep{color:var(--muted);margin:0 5px}
.facts .bad{color:var(--red)}.facts .good{color:var(--green)}.facts .unknown{color:var(--muted)}
@media (max-width:900px){.head{flex-direction:column}}
h1{font:600 18px/1.3 Inter,system-ui,sans-serif;margin:0 0 10px;overflow-wrap:anywhere}
h2{font:600 14px/1.3 Inter,system-ui,sans-serif;margin:0 0 10px}
.chips{list-style:none;margin:0;padding:0;display:flex;flex-wrap:wrap;gap:6px}
.chips li{display:inline-flex;align-items:center;gap:6px;max-width:100%;padding:3px 8px;
border:1px solid rgba(204,204,220,.15);border-radius:4px;background:rgba(204,204,220,.04);
font-size:12px;line-height:18px}
.chips .k{color:var(--muted);font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.05em}
.chips .v{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.status-ERROR .v{color:var(--red);font-weight:600}.status-OK .v,.status-UNSET .v{color:var(--green);font-weight:600}
.repro{display:flex;gap:6px;margin:12px 0 0}
.repro code{flex:1 1 auto;min-width:0;padding:7px 12px;border:1px solid rgba(204,204,220,.15);
border-radius:4px;background:rgba(204,204,220,.04);font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;
white-space:pre-wrap;overflow-wrap:anywhere}
button{background:#242b35;color:var(--text);border:1px solid #657083;border-radius:4px;
padding:4px 12px;cursor:pointer;font:inherit}
.grid{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,.75fr);gap:24px;align-items:start}
.side{position:sticky;top:12px;display:flex;flex-direction:column;gap:12px;max-height:calc(100dvh - 24px);min-width:0}
.box{border:1px solid var(--line);border-radius:5px;background:var(--panel);min-width:0}
.box h2{display:flex;align-items:center;gap:8px;margin:0;padding:10px 12px;border-bottom:1px solid var(--line)}
.box h2 img{width:20px;height:20px}
#issues{display:flex;flex-direction:column;flex:1 1 auto;min-height:180px}
#issues .status{margin:0;padding:12px}
#issues ul{flex:1;overflow-y:auto;min-height:0;padding:12px}
@media (max-width:1100px){.grid{grid-template-columns:minmax(0,1fr)}.side{position:static;max-height:none}#issues{max-height:75dvh}}
.exc{color:var(--amber);font:600 15px/1.3 Inter,system-ui,sans-serif;margin:0 0 2px}
.where{color:var(--muted);font-size:12px;margin:0 0 4px;overflow-wrap:anywhere}
.gh-labels{display:flex;flex-wrap:wrap;gap:5px;margin:4px 0 6px}
.gh-label{display:inline-block;padding:2px 8px;border-radius:999px;font:600 11px/1.5 system-ui,sans-serif;white-space:nowrap}
.label{margin:14px 0 4px;color:var(--muted);font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.05em}
pre{margin:0;white-space:pre-wrap;word-break:break-word;background:#0b0c0e;border:1px solid var(--soft);
border-radius:6px;padding:12px;font:12.5px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;
max-height:70vh;overflow:auto}
pre.msg{color:#ff8a80;max-height:none}
.detail+.detail{margin-top:18px;padding-top:18px;border-top:1px solid var(--line)}
.empty{color:var(--muted)}
.passed{color:var(--green);font-weight:600}
#issues ul{list-style:none;margin:0}
#issues li{border:1px solid var(--line);border-radius:4px;padding:10px;margin:0 0 10px;background:var(--bg)}
#issues li:last-child{margin-bottom:0}
#issues .meta{color:var(--muted);font-size:12px;margin:1px 0}
#issues .snippet{color:#aeb7c2;font-size:12px;overflow-wrap:anywhere;display:-webkit-box;
-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
iframe.serge{width:100%;height:64px;border:0;display:block}
"""

SCRIPT = r"""
const ctx=JSON.parse(document.getElementById('ctx').textContent);
const AGO_UNITS=[['y',31536000],['mo',2592000],['d',86400],['h',3600],['m',60]];
function ago(seconds){
  if(seconds<45)return 'just now';
  for(const [suffix,size] of AGO_UNITS){
    const n=Math.floor(seconds/size);
    if(n>=1)return n+suffix+' ago';
  }
  return 'just now';
}
document.querySelectorAll('[data-ts]').forEach(el=>{
  el.textContent=ago(Date.now()/1000-Number(el.dataset.ts));
});
const copy=document.getElementById('copy');
if(copy)copy.addEventListener('click',async()=>{
  const text=document.getElementById('command').textContent;
  try{await navigator.clipboard.writeText(text);}catch(error){
    const area=document.createElement('textarea');area.value=text;document.body.append(area);
    area.select();document.execCommand('copy');area.remove();
  }
  copy.textContent='Copied';setTimeout(()=>{copy.textContent='Copy';},1500);
});
const issues=document.getElementById('issues');
const statusNode=issues.querySelector('.status'),list=issues.querySelector('ul');
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function loadIssues(){
  if(!ctx.test_nodeid){statusNode.textContent='No test selected.';return;}
  const url='/related-issues?'+new URLSearchParams({format:'json',trace_id:ctx.trace_id,test_nodeid:ctx.test_nodeid});
  try{
    let data;
    // The exporter is busy only when saturated: back off quietly, then give up.
    for(const delay of [1000,2000,4000,0]){
      const response=await fetch(url,{cache:'no-store'});
      if(!response.ok)throw new Error('Lookup failed');
      data=await response.json();
      if(data.status!=='busy'||!delay)break;
      await sleep(delay);
    }
    if(data.status!=='ok'){statusNode.textContent='Related issues are temporarily unavailable.';return;}
    statusNode.textContent='No related issues found.';statusNode.hidden=data.hits.length>0;
    for(const hit of data.hits){
      const li=document.createElement('li'),a=document.createElement('a');
      a.href=hit.url;a.target='_blank';a.rel='noopener noreferrer';
      a.textContent=(hit.type==='pr'?'PR':'Issue')+' #'+hit.number+' · '+hit.title;
      const meta=document.createElement('div');meta.className='meta';
      meta.textContent=[hit.author,hit.age,hit.trust].filter(Boolean).join(' · ');
      const snippet=document.createElement('div');snippet.className='snippet';
      snippet.textContent=hit.snippet;snippet.title=hit.snippet;
      li.append(a,meta,snippet);list.append(li);
    }
  }catch(error){statusNode.textContent='Related issues are temporarily unavailable.';}
}
loadIssues();
// Same pills as the PR page's GitHub labels.
const MARKER_COLORS={slow:'fbca04',flaky:'8250df'};
async function loadMarkers(){
  const box=document.getElementById('markers');
  if(!box||!ctx.test_nodeid)return;
  try{
    const params={test_nodeid:ctx.test_nodeid};
    if(ctx.commit)params.commit=ctx.commit;
    const response=await fetch('/test/markers?'+new URLSearchParams(params));
    if(!response.ok)return;
    const data=await response.json();
    for(const name of data.markers||[]){
      const color=MARKER_COLORS[name]||'6b7280';
      const red=parseInt(color.slice(0,2),16),green=parseInt(color.slice(2,4),16),blue=parseInt(color.slice(4,6),16);
      const badge=document.createElement('span');badge.className='gh-label';badge.textContent=name;
      badge.title=name==='slow'?'Marked @slow: runs only with RUN_SLOW=1':'Marked @is_flaky: retried on failure';
      badge.style.backgroundColor='#'+color;
      badge.style.color=(red*299+green*587+blue*114)/1000>145?'#111217':'#fff';
      box.append(badge);
    }
    box.hidden=!box.children.length;
  }catch(error){}
}
loadMarkers();
"""


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {rest:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def format_bytes(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    sign = ("+" if value > 0 else "-" if value < 0 else "") if signed else ""
    size = abs(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            text = f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
            return sign + text
        size /= 1024
    return sign + f"{size:.1f} GiB"  # pragma: no cover - loop always returns


def compact_ago(seconds: float) -> str:
    """Server-side first paint of the client's relative time ("16h ago")."""
    if seconds < 45:
        return "just now"
    for suffix, size in (
        ("y", 31536000),
        ("mo", 2592000),
        ("d", 86400),
        ("h", 3600),
        ("m", 60),
    ):
        if seconds >= size:
            return f"{int(seconds // size)}{suffix} ago"
    return "just now"


def memory_leak(data: dict) -> tuple[str, str, str]:
    """(text, css class, tooltip) for the Memory leak cell.

    A leak is device memory the test left allocated beyond what it inherited
    (``pytest.cuda_delta_bytes`` > 0). Without the reading (CPU jobs, older
    plugin) it is unknown, never "No".
    """
    delta = data.get("cuda_delta_bytes")
    inherited = data.get("cuda_inherited_bytes")
    retained = data.get("cuda_retained_bytes")
    parts = [
        f"{label} {format_bytes(value)}"
        for label, value in (("inherited", inherited), ("left for next test", retained))
        if value is not None
    ]
    if delta is None:
        return "—", "unknown", "No device-memory reading for this test"
    parts.append(f"net change {format_bytes(delta, signed=True)}")
    return (
        (format_bytes(delta, signed=True), "bad", " · ".join(parts))
        if delta > 0
        else ("No", "good", " · ".join(parts))
    )


def reproduce_command(nodeid: str) -> str:
    quoted = "'" + nodeid.replace("'", "'\"'\"'") + "'"
    if nodeid.startswith("utils/checkers.py::"):
        name = nodeid.split("::")[-1]
        return "make '" + name.replace("'", "'\"'\"'") + "'"
    return "pytest -svx " + quoted


def _link(url: str, text: str, *, github: bool = False, top: bool = False) -> str:
    esc = html.escape
    if not url:
        return f"<span class='v'>{esc(text)}</span>"
    target = (
        " target='_blank' rel='noopener noreferrer'"
        if github
        else " target='_top'"
        if top
        else ""
    )
    mark = GITHUB_MARK if github else ""
    return f"<a class='v' href='{esc(url)}'{target} title='{esc(text)}'>{esc(text)}{mark}</a>"


def render_test_page(data: dict) -> str:
    """Render the page. ``data`` keys (all optional except ``test_nodeid``):

    trace_id, test_nodeid, status, module, job, pr, run_id, hardware,
    runner_type, runner_name, file_url, run_url, job_url, pr_url,
    duration_seconds, started_at (epoch s), cuda_inherited_bytes,
    cuda_retained_bytes, cuda_delta_bytes, details (failure dicts with
    exception_type, exception_message, exception_stacktrace, github_url,
    job_log_url, run_url), has_trace.
    """
    esc = html.escape
    node = str(data.get("test_nodeid") or "")
    trace_id = str(data.get("trace_id") or "")
    status = str(data.get("status") or "")
    job = str(data.get("job") or "")
    pr = str(data.get("pr") or "")
    run_id = str(data.get("run_id") or "")
    hardware = str(data.get("hardware") or "")
    details = list(data.get("details") or [])

    job_page = (
        "/failure/job-page?"
        + urlencode(
            {
                "trace_id": trace_id,
                "test_nodeid": node,
                "job": job,
                "run_id": run_id,
                "pr": pr,
                "status": status,
            }
        )
        if job and run_id
        else ""
    )
    runner = " · ".join(
        part
        for part in (
            str(data.get("runner_type") or ""),
            HARDWARE_LABELS.get(hardware, hardware),
        )
        if part
    )
    chips = [
        ("status", "Status", _link("", status) if status else ""),
        (
            "module",
            "Module",
            _link(
                str(data.get("file_url") or ""),
                str(data.get("module") or ""),
                github=True,
            )
            if data.get("module")
            else "",
        ),
        ("job", "Job", _link(job_page, job, top=True) if job else ""),
        (
            "runner",
            "Runner",
            _link(
                str(data.get("job_url") or data.get("run_url") or ""),
                runner,
                github=True,
            )
            if runner
            else "",
        ),
        (
            "pr",
            "PR",
            _link(str(data.get("pr_url") or ""), pr, github=True) if pr else "",
        ),
        (
            "run",
            "Run",
            _link(str(data.get("run_url") or ""), run_id, github=True)
            if run_id
            else "",
        ),
    ]

    out = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>{esc(node.split('::')[-1] or 'Test')} · Test</title>",
        f"<style>{STYLE}</style></head><body><div class='page'>",
    ]
    # Header: identity, links, reproduce.
    out.append(
        f"<section class='head'><div class='head-main'><h1>{esc(node) or 'No test selected'}</h1>"
        # Filled by the script from /test/markers (@slow, @is_flaky).
        "<div id='markers' class='gh-labels' hidden></div>"
    )
    chip_html = [
        f"<li class='{'status-' + esc(status) if key == 'status' else ''}'>"
        f"<span class='k'>{esc(label)}</span>{value}</li>"
        for key, label, value in chips
        if value
    ]
    if chip_html:
        out.append("<ul class='chips'>" + "".join(chip_html) + "</ul>")
    if node:
        out.append(
            "<div class='repro'>"
            f"<code id='command'>{esc(reproduce_command(node))}</code>"
            "<button id='copy' type='button' aria-label='Copy command'>Copy</button></div>"
        )
    out.append("</div>")

    # Top right: when and how long it ran, and whether it leaked device memory.
    when = []
    started = data.get("started_at")
    if started:
        moment = time.gmtime(float(started))
        stamp = time.strftime("%Y-%m-%d %H:%M UTC", moment)
        when.append(f"{time.strftime('%b', moment)} {moment.tm_mday}")
        when.append(
            f"<span data-ts='{float(started):.0f}' title='{esc(stamp)}'>"
            f"{esc(compact_ago(time.time() - float(started)))}</span>"
        )
    when.append(esc(format_duration(data.get("duration_seconds"))))
    leak, leak_class, leak_title = memory_leak(data)
    out.append(
        "<dl class='facts'><dt>Duration</dt><dd>"
        + "<span class='sep'>·</span>".join(when)
        + "</dd>"
        f"<dt>Memory leak</dt><dd class='{leak_class}' title='{esc(leak_title)}'>"
        f"{esc(leak)}</dd></dl>"
    )
    out.append("</section>")

    # Failure | related issues, half the width each.
    out.append("<div class='grid'><section class='card'>")
    if details:
        for detail in details:
            out.append("<div class='detail'>")
            out.append(
                f"<div class='exc'>{esc(detail.get('exception_type') or 'Failure')}</div>"
            )
            where = []
            if detail.get("github_url"):
                where.append(
                    _link(
                        str(detail["github_url"]),
                        str(detail.get("test_nodeid") or node),
                        github=True,
                    )
                )
            for key, label in (
                ("job_log_url", "GitHub job log"),
                ("run_url", "GitHub run"),
            ):
                if detail.get(key):
                    where.append(_link(str(detail[key]), label, github=True))
                    break
            if where:
                out.append("<div class='where'>" + " · ".join(where) + "</div>")
            if detail.get("exception_message"):
                out.append("<div class='label'>Message</div>")
                out.append(
                    f"<pre class='msg'>{esc(str(detail['exception_message']))}</pre>"
                )
            if detail.get("exception_stacktrace"):
                out.append("<div class='label'>Traceback</div>")
                out.append(f"<pre>{esc(str(detail['exception_stacktrace']))}</pre>")
            out.append("</div>")
    elif not data.get("has_trace"):
        out.append(
            "<p class='empty'>"
            + ("Trace not found." if trace_id else "No trace selected.")
            + "</p>"
        )
    elif status == "ERROR":
        out.append(
            "<p class='empty'>The test failed, but its span carries no exception.</p>"
        )
    else:
        out.append("<p class='passed'>This test passed in this run.</p>")
    out.append("</section>")
    serge_query = urlencode(
        {
            "trace_id": trace_id,
            "latest_trace": "",
            "test_nodeid": node,
            "status": status,
            "module": str(data.get("module") or ""),
            "job": job,
            "pr": pr,
            "run_id": run_id,
        }
    )
    out.append(
        "<aside class='side'><section class='box' id='serge'>"
        "<h2><img src='/serge-actions/logo.png' alt=''>Agentic Actions</h2>"
        f"<iframe class='serge' title='Serge actions' src='/serge-actions?{esc(serge_query)}'></iframe>"
        "</section>"
        "<section class='box' id='issues'><h2>Potential related issues</h2>"
        "<p class='status empty' role='status' aria-live='polite'>Searching related issues…</p>"
        "<ul></ul></section></aside>"
    )
    out.append("</div></div>")
    context = {
        "trace_id": trace_id,
        "test_nodeid": node,
        "commit": str(data.get("commit") or ""),
    }
    # json.dumps escapes neither "<" nor "/", so close-tag injection is
    # neutralised by escaping "<" explicitly.
    out.append(
        "<script type='application/json' id='ctx'>"
        + json.dumps(context).replace("<", "\\u003c")
        + "</script>"
    )
    out.append(f"<script>{SCRIPT}</script></body></html>")
    return "".join(out)
