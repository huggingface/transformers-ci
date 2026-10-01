"""Serge Actions panel for the pytest Test dashboard.

A static shell for now: the three actions are shown to signed-in Grafana users
and stay disabled until the authenticated action API behind them exists.
Hiding them from anonymous viewers is presentation only, never the
authorization gate.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable

# (id, label, what Serge will do, wired yet). Rendered with textContent only.
ACTIONS = (
    ("new-issue", "New issue", "Serge writes a GitHub issue for this failure.", False),
    ("fix-it", "Fix it!", "Serge reproduces the failure and opens a fix PR.", False),
    ("wdyt", "WDYT?", "Serge takes a quick look and tells you what it thinks.", True),
)

# Fix PRs Serge already opened for a test: its PR bodies quote the failing node id.
REPOSITORY = "huggingface/transformers"
SERGE_AUTHOR = "app/sergereview"
FIX_PR_TTL_SECONDS = 600.0
FIX_PR_ERROR_TTL_SECONDS = 60.0  # GitHub search allows 30 requests a minute
FIX_PR_CACHE_SIZE = 512
FIX_PR_LIMIT = 3
_fix_cache: dict[str, tuple[float, dict]] = {}
_fix_lock = threading.Lock()


def fix_pr_query(nodeid: str) -> str:
    return f'repo:{REPOSITORY} is:pr author:{SERGE_AUTHOR} in:body "{nodeid.replace(chr(34), "")}"'


def _fix_pr(item: object, nodeid: str) -> dict | None:
    # Phrase search is fuzzy (test_foo matches test_foo_bar): keep exact quotes only.
    if not isinstance(item, dict) or not re.search(
        re.escape(nodeid) + r"(?!\w)", str(item.get("body") or "")
    ):
        return None
    number, url = item.get("number"), item.get("html_url")
    prefix = f"https://github.com/{REPOSITORY}/pull/"
    if type(number) is not int or url != f"{prefix}{number}":
        return None
    pull = item.get("pull_request") or {}
    if isinstance(pull, dict) and pull.get("merged_at"):
        state = "merged"
    elif item.get("state") == "open":
        state = "draft" if item.get("draft") else "open"
    else:
        state = "closed"
    return {
        "number": number,
        "url": url,
        "title": str(item.get("title") or "")[:300],
        "state": state,
        "created_at": str(item.get("created_at") or ""),
    }


def fix_prs(nodeid: str, search: Callable[[str], list]) -> dict:
    """Serge PRs that name this test, newest first; cached per node id."""
    now = time.monotonic()
    with _fix_lock:
        cached = _fix_cache.get(nodeid)
        if cached and cached[0] > now:
            return cached[1]
    try:
        prs = [
            pr for item in search(fix_pr_query(nodeid)) if (pr := _fix_pr(item, nodeid))
        ]
        result, ttl = {"status": "ok", "prs": prs[:FIX_PR_LIMIT]}, FIX_PR_TTL_SECONDS
    except Exception:
        result, ttl = {"status": "unavailable", "prs": []}, FIX_PR_ERROR_TTL_SECONDS
    with _fix_lock:
        if len(_fix_cache) >= FIX_PR_CACHE_SIZE:
            _fix_cache.pop(next(iter(_fix_cache)))
        _fix_cache[nodeid] = (now + ttl, result)
    return result


# A pixel-art homage drawn here (no third-party image): rainbow trail in
# 5-unit segments that alternate up/down, Pop-Tart body, grey head.
_RAINBOW = ("#ff0000", "#ff9900", "#ffff00", "#33ff00", "#0099ff", "#6633ff")
_TRAIL = "".join(
    f'<g class="w{seg % 2}">'
    + "".join(
        f'<rect x="{seg * 5}" y="{3 + 2 * i}" width="5" height="2" fill="{c}"/>'
        for i, c in enumerate(_RAINBOW)
    )
    + "</g>"
    for seg in range(6)
)
_CAT = (
    '<rect x="28" y="8" width="3" height="2" fill="#999"/>'  # tail
    + "".join(
        f'<rect x="{x}" y="14" width="2" height="2" fill="#999"/>'
        for x in (32, 35, 41, 44)
    )  # legs
    + '<rect x="31" y="2" width="13" height="12" rx="1" fill="#ffcc99" stroke="#000"'
    ' stroke-width=".5"/>'
    + '<rect x="32" y="3" width="11" height="10" rx="1" fill="#ff99ff"/>'
    + "".join(
        f'<rect x="{x}" y="{y}" width="1" height="1" fill="#ff3399"/>'
        for x, y in ((34, 5), (38, 4), (35, 9), (40, 7), (33, 11), (41, 11))
    )  # sprinkles
    + '<rect x="40" y="4" width="2" height="2" fill="#999"/>'  # ears
    + '<rect x="47" y="4" width="2" height="2" fill="#999"/>'
    + '<rect x="40" y="6" width="9" height="7" fill="#999"/>'  # head
    + '<rect x="42" y="8" width="1" height="1" fill="#000"/>'  # eyes
    + '<rect x="46" y="8" width="1" height="1" fill="#000"/>'
    + '<rect x="41" y="10" width="1" height="1" fill="#ff9999"/>'  # cheeks
    + '<rect x="47" y="10" width="1" height="1" fill="#ff9999"/>'
    + '<rect x="43" y="11" width="3" height="1" fill="#000"/>'  # mouth
)
NYAN_SVG = (
    '<svg class="nyan" viewBox="0 0 50 18" shape-rendering="crispEdges"'
    ' aria-hidden="true">' + _TRAIL + '<g class="cat">' + _CAT + "</g></svg>"
)

PAGE_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<style>
body{margin:0;padding:8px 10px;background:#181b1f;color:#d8d9da;
font:13px/1.45 system-ui,sans-serif}
.row{display:flex;gap:6px;flex-wrap:wrap}
button{flex:1 1 0;min-width:0;background:#242b35;color:#d8d9da;
border:1px solid #657083;border-radius:4px;padding:5px 8px;font:inherit;cursor:pointer;
white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
button:disabled{opacity:.55;cursor:not-allowed}
p{margin:6px 0 0;color:#8e9197;font-size:12px}a{color:#6ab0ff;text-decoration:none}
.sky{position:relative;height:24px;margin:6px 0 0;overflow:hidden}
.nyan{position:absolute;top:0;height:24px;width:67px;left:calc(100% - 67px);
animation:fly 7s linear infinite}
.nyan .cat,.nyan .w0,.nyan .w1{animation:bob .4s steps(1) infinite}
.nyan .w1{animation-delay:-.2s}
@keyframes fly{from{left:-67px}to{left:100%}}
@keyframes bob{0%{transform:translateY(0)}50%{transform:translateY(1px)}}
button.thinking{color:#fff;border-color:transparent;background:linear-gradient(90deg,
#ff0000,#ff9900,#ffff00,#33ff00,#0099ff,#6633ff,#ff0000);background-size:300% 100%;
animation:shimmer 1.2s linear infinite;text-shadow:0 1px 2px #000}
@keyframes shimmer{to{background-position:-150% 0}}
body.thinking .nyan{animation-duration:1.4s}
@media (prefers-reduced-motion:reduce){.nyan,.nyan *,button.thinking{animation:none}}
</style></head><body>
<div class="row" id="actions" hidden></div>
<p id="fix" hidden></p>
<div class="sky">__NYAN__</div>
<p id="note" role="status" aria-live="polite"></p>
<script>
const ACTIONS=__ACTIONS__;
const row=document.getElementById('actions'),note=document.getElementById('note');
function signIn(){
  note.textContent='';const a=document.createElement('a');
  a.href='/login';a.target='_top';a.textContent='Sign in';
  note.append(a,' to ask Serge about this failure.');
}
fetch('/api/user',{credentials:'same-origin',cache:'no-store'}).then(response=>{
  if(!response.ok){signIn();return;}
  for(const [id,label,help,enabled] of ACTIONS){
    const b=document.createElement('button');b.type='button';b.id=id;
    b.textContent=label;b.title=help;b.disabled=!enabled;row.append(b);
  }
  row.hidden=false;
  document.getElementById('wdyt').addEventListener('click',wdyt);
}).catch(signIn);

const context=new URLSearchParams(location.search);
// Public: a fix PR Serge already opened for this test, from its PR bodies on GitHub.
const FIX_STATES={open:'open',draft:'draft',merged:'merged ✓',closed:'closed'};
if(context.get('test_nodeid'))fetch('/serge-actions/fix-prs?test_nodeid='+
  encodeURIComponent(context.get('test_nodeid')),{cache:'no-store'})
  .then(response=>response.json()).then(data=>{
    const prs=data.prs||[];if(!prs.length)return;
    const fix=document.getElementById('fix');
    fix.append('🔧 Serge opened ');
    prs.forEach((pr,i)=>{
      if(i)fix.append(', ');
      const a=document.createElement('a');a.href=pr.url;a.target='_blank';
      a.rel='noopener noreferrer';a.title=pr.title;a.textContent='PR #'+pr.number;
      fix.append(a,' ('+(FIX_STATES[pr.state]||pr.state)+')');
    });
    fix.append(' to fix this.');fix.hidden=false;
  }).catch(()=>{});
// Like the traceback panel: a link without var-trace_id uses the test's latest
// failing trace, which the dashboard resolves into latest_trace.
if(!context.get('trace_id')&&/^[0-9a-f]{32}$/i.test(context.get('latest_trace')||''))
  context.set('trace_id',context.get('latest_trace'));
const MESSAGES={disabled:'WDYT? is not configured on this server yet.',
  unauthorized:'Your Grafana session expired. Sign in again.',
  rate_limited:'You asked a lot this hour. Try again later.',
  busy:'Serge is busy with other questions. Try again in a minute.',
  unavailable:'Serge could not answer right now. Try again in a minute.',
  forbidden:'Request refused.',invalid:'This test has no trace to look at.'};
let run=null;  // the in-flight question: {view, done}
async function wdyt(){
  const button=document.getElementById('wdyt');
  if(run){run.view.show();return;}  // reopen the popup of a running question
  const view=popup();run={view};
  button.classList.add('thinking');document.body.classList.add('thinking');
  button.textContent='Thinking…';
  let result={status:'unavailable'};
  try{
    const response=await fetch('/serge-actions/wdyt',{method:'POST',credentials:'same-origin',
      headers:{'Content-Type':'application/json','X-TCI-Action':'1'},
      body:JSON.stringify(Object.fromEntries(['trace_id','test_nodeid','status','job',
        'module','pr','run_id'].map(k=>[k,context.get(k)||''])))});
    if(!(response.headers.get('Content-Type')||'').includes('ndjson')){
      result=await response.json();
    }else{
      // NDJSON: one event per line, the last one {"event":"done",...}.
      const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='';
      for(;;){
        const {value,done}=await reader.read();
        buffer+=decoder.decode(value||new Uint8Array(),{stream:!done});
        let cut;
        while((cut=buffer.indexOf('\n'))>=0){
          const line=buffer.slice(0,cut).trim();buffer=buffer.slice(cut+1);
          if(!line)continue;
          const event=JSON.parse(line);
          if(event.event==='done')result=event;else view.progress(event);
        }
        if(done)break;
      }
    }
  }catch(error){}
  button.classList.remove('thinking');document.body.classList.remove('thinking');
  button.textContent='WDYT?';run=null;
  view.finish(result);
}

// Tiny Markdown subset -> DOM nodes (never innerHTML): paragraphs, "- " bullets,
// **bold**, `code`, and #N linked when N is a related thread the model was given
// (links: number -> server-built GitHub URL), so an invented number stays text.
function linked(doc,parent,text,links){
  for(const part of text.split(/(#\d{1,7}\b)/)){
    if(!part)continue;
    const url=part[0]==='#'&&links[part.slice(1)];
    if(url){
      const a=doc.createElement('a');a.href=url;a.target='_blank';a.rel='noopener noreferrer';
      a.textContent=part;a.style.cssText='color:#6ab0ff;text-decoration:none';parent.append(a);
    }else parent.append(doc.createTextNode(part));
  }
}
function inline(doc,parent,text,links){
  for(const part of text.split(/(\*\*[^*]+\*\*|`[^`]+`)/)){
    if(!part)continue;
    if(part.startsWith('**')&&part.endsWith('**')&&part.length>4){
      const b=doc.createElement('strong');linked(doc,b,part.slice(2,-2),links);parent.append(b);
    }else if(part.startsWith('`')&&part.endsWith('`')&&part.length>2){
      const c=doc.createElement('code');c.textContent=part.slice(1,-1);
      c.style.cssText='background:rgba(204,204,220,.1);padding:0 4px;border-radius:3px;'+
        'font:12px ui-monospace,monospace';parent.append(c);
    }else linked(doc,parent,part,links);
  }
}
function markdown(doc,text,links={}){
  const root=doc.createElement('div');let list=null;
  for(const raw of text.split('\n')){
    const line=raw.trim();
    if(!line){list=null;continue;}
    const bullet=line.match(/^[-*] (.*)$/);
    if(bullet){
      if(!list){list=doc.createElement('ul');list.style.cssText='margin:6px 0;padding-left:20px';
        root.append(list);}
      const li=doc.createElement('li');li.style.margin='3px 0';inline(doc,li,bullet[1],links);list.append(li);
    }else{
      list=null;const p=doc.createElement('p');p.style.margin='6px 0';
      inline(doc,p,line.replace(/^#+ /,''),links);root.append(p);
    }
  }
  return root;
}

// The billed cost of a fresh answer. HF reports usage with a delay, so poll the
// exporter (which reads the org's billing API) until it shows up.
async function cost(sessionId,node){
  const say=text=>{node.textContent=text;};
  for(let attempt=0;attempt<36;attempt++){
    let data={status:'unavailable'};
    try{
      const response=await fetch('/serge-actions/wdyt/cost?session_id='+
        encodeURIComponent(sessionId),{method:'POST',credentials:'same-origin',
        headers:{'X-TCI-Action':'1'}});
      data=await response.json();
    }catch(error){}
    if(data.status==='reported'){
      const usd=data.cost_usd;
      say('You just spent $'+(usd<0.01?usd.toFixed(4):usd.toFixed(2))+'. Go easy on the AI ;)');
      return;
    }
    if(data.status==='forbidden'){say('Cost not visible: this key cannot read the org billing.');return;}
    if(data.status==='unknown'||data.status==='unauthorized'){say('');return;}
    if(!node.isConnected&&attempt>2)return;  // popup closed: stop asking
    await new Promise(resolve=>setTimeout(resolve,5000));
  }
  say('Cost not reported yet. HF billing can lag by a few minutes.');
}

function popup(){
  let doc=document,host=document.body;
  try{if(parent.document.body){doc=parent.document;host=doc.body;}}catch(error){}
  doc.getElementById('tci-wdyt')?.remove();
  const el=(tag,css,text)=>{const n=doc.createElement(tag);if(css)n.style.cssText=css;
    if(text!==undefined)n.textContent=text;return n;};
  const overlay=el('div','position:fixed;inset:0;z-index:1100;background:rgba(0,0,0,.55);'+
    'display:flex;align-items:center;justify-content:center;padding:16px');overlay.id='tci-wdyt';
  const box=el('div','background:#181b1f;color:#d8d9da;border:1px solid rgba(204,204,220,.2);'+
    'border-radius:6px;max-width:640px;width:100%;max-height:80vh;overflow:auto;'+
    'padding:16px 20px;font:14px/1.5 Inter,system-ui,sans-serif;box-shadow:0 8px 32px #000');
  box.setAttribute('role','dialog');box.setAttribute('aria-modal','true');
  box.setAttribute('aria-label','Serge thinks');
  const title=el('div','font-weight:600;font-size:16px;margin:0 0 8px','✨🦄 Serge thinks…');
  const test=el('div','color:#8e9197;font:12px ui-monospace,monospace;overflow-wrap:anywhere;'+
    'margin:0 0 8px',context.get('test_nodeid')||'');
  // Live progress: finished steps get a check, the current one pulses.
  const steps=el('ul','list-style:none;margin:0 0 8px;padding:0;color:#8e9197;font-size:13px');
  steps.setAttribute('aria-live','polite');
  const answer=el('div','margin:6px 0 0');let streamed='';
  const foot=el('div','color:#8e9197;font-size:12px;margin:10px 0 0');
  const style=el('style');
  style.textContent='@keyframes tci-pulse{50%{opacity:.35}}'+
    '@media (prefers-reduced-motion:reduce){#tci-wdyt *{animation:none!important}}';
  box.append(style,title,test,steps,answer,foot);
  let current=null;
  const step=text=>{
    if(current){current.firstChild.textContent='✓ ';current.firstChild.style.animation='';}
    const li=el('li','margin:2px 0'),mark=el('span','animation:tci-pulse 1s infinite','● ');
    li.append(mark,doc.createTextNode(text));steps.append(li);current=li;
  };
  step('Starting');
  const actions=el('div','display:flex;gap:8px;justify-content:flex-end;margin:14px 0 0');
  const button=(label,help,disabled)=>{const b=el('button','background:#242b35;color:#d8d9da;'+
    'border:1px solid #657083;border-radius:4px;padding:5px 12px;font:inherit;cursor:'+
    (disabled?'not-allowed':'pointer')+';opacity:'+(disabled?'.55':'1'),label);
    b.type='button';b.title=help;b.disabled=disabled;return b;};
  const deeper=button('Deeper investigation','Runs a full Serge investigation.',true);
  const close=button('Close','',false);actions.append(deeper,close);box.append(actions);
  overlay.append(box);
  const onKey=event=>{if(event.key==='Escape')hide();};
  const show=()=>{if(!overlay.isConnected){host.append(overlay);doc.addEventListener('keydown',onKey);}
    close.focus();};
  const hide=()=>{overlay.remove();doc.removeEventListener('keydown',onKey);};
  close.addEventListener('click',hide);
  overlay.addEventListener('click',event=>{if(event.target===overlay)hide();});
  show();
  let thinkingShown=false;
  return {show,
    progress(event){
      if(event.event==='step')step(event.text);
      else if(event.event==='thinking'&&!thinkingShown){thinkingShown=true;step('Reasoning…');}
      else if(event.event==='delta'){
        if(!answer.dataset.streaming){answer.dataset.streaming='1';step('Writing the answer');}
        streamed+=event.text;answer.replaceChildren(markdown(doc,streamed));
      }
    },
    finish(data){
      if(current){current.firstChild.textContent='✓ ';current.firstChild.style.animation='';}
      title.textContent='✨🦄 Serge thinks';
      answer.replaceChildren();
      if(data.status==='ok'){
        const links={};
        for(const hit of data.related||[])
          if(/^https:\/\/github\.com\//.test(hit.url))links[hit.number]=hit.url;
        answer.append(markdown(doc,data.answer,links));
        foot.textContent=(data.cached?'Answered earlier by ':'Quick opinion from ')+data.model+
          ', based only on this page. It can be wrong.';
        if(!data.cached&&data.session_id){
          const spent=el('div','margin:4px 0 0','Counting the cost…');foot.append(spent);
          cost(data.session_id,spent);
        }
      }else{
        current=null;step(MESSAGES[data.status]||MESSAGES.unavailable);
        current.firstChild.textContent='✗ ';current.style.color='#f2495c';
      }
      show();
    }};
}
</script></body></html>"""

PAGE_HTML = PAGE_HTML.replace("__ACTIONS__", json.dumps(ACTIONS)).replace(
    "__NYAN__", NYAN_SVG
)

# CI Health: clear WDYT?'s answer cache so a demo shows the whole live run.
ADMIN_HTML = """<!doctype html><html><head><meta charset="utf-8">
<style>
body{margin:0;padding:8px 10px;background:#181b1f;color:#d8d9da;
font:13px/1.45 system-ui,sans-serif;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
button{background:#242b35;color:#d8d9da;border:1px solid #657083;border-radius:4px;
padding:5px 12px;font:inherit;cursor:pointer}button:disabled{opacity:.55;cursor:wait}
span{color:#8e9197;font-size:12px}
</style></head><body>
<button id="clear" type="button">Clear WDYT? cache</button>
<span id="note" role="status" aria-live="polite">Next WDYT? on any test asks the model again.</span>
<script>
const button=document.getElementById('clear'),note=document.getElementById('note');
const MESSAGES={unauthorized:'Sign in to clear the cache.',forbidden:'Request refused.'};
button.addEventListener('click',async()=>{
  button.disabled=true;
  try{
    const response=await fetch('/serge-actions/wdyt/cache',{method:'POST',
      credentials:'same-origin',headers:{'X-TCI-Action':'1'}});
    const data=await response.json();
    note.textContent=data.status==='ok'?
      'Cleared '+data.cleared+' cached answer'+(data.cleared===1?'':'s')+'.':
      (MESSAGES[data.status]||'Could not clear the cache.');
  }catch(error){note.textContent='Could not clear the cache.';}
  button.disabled=false;
});
</script></body></html>"""
