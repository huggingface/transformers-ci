"""Serge Actions panel for the pytest Test dashboard.

The panel discovers the dashboard-owned GitHub session and directs anonymous
viewers to GitHub sign-in. Presentation is never the authorization gate: every
action authenticates again on the server.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from collections.abc import Callable

# (id, label, what Serge will do, wired yet). Rendered with textContent only.
ACTIONS = (
    ("new-issue", "New issue", "Draft a GitHub issue for this failure.", True),
    ("fix-it", "Fix it!", "Serge reproduces the failure and opens a fix PR.", False),
    ("wdyt", "WDYT?", "Serge takes a quick look and tells you what it thinks.", True),
    (
        "who-can-help",
        "Who can help",
        "Find people connected to this failing line and explain where they can help.",
        True,
    ),
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


# Serge's logo (huggingface/serge docs/assets/serge-logo.png), served at
# /serge-actions/logo.png for this page, the WDYT? popup and the Test page.
LOGO_PNG = (Path(__file__).parent / "serge-logo.png").read_bytes()

# Shown before a button's label when the viewer is signed out.
LOCK_SVG = (
    "<svg viewBox='0 0 16 16' width='11' height='11' fill='currentColor' "
    "aria-hidden='true' style='vertical-align:-1px;margin-right:5px'><path d='M4 7V5a4 4 0 0 1 8 "
    "0v2h.5A1.5 1.5 0 0 1 14 8.5v5a1.5 1.5 0 0 1-1.5 1.5h-9A1.5 1.5 0 0 1 2 13.5v-5A1.5 "
    "1.5 0 0 1 3.5 7H4zm2 0h4V5a2 2 0 1 0-4 0v2z'/></svg>"
)

PAGE_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<style>
body{margin:0;padding:8px 10px;background:#181b1f;color:#d8d9da;
font:13px/1.45 system-ui,sans-serif}
.row{display:flex;gap:6px;flex-wrap:wrap}
button{flex:1 1 100px;min-width:0;background:#242b35;color:#d8d9da;
border:1px solid #657083;border-radius:4px;padding:5px 8px;font:inherit;cursor:pointer;
white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
button:disabled{opacity:.55;cursor:not-allowed}
p{margin:6px 0 0;color:#8e9197;font-size:12px}a{color:#6ab0ff;text-decoration:none}
button.locked{color:#aeb7c2}
button.thinking{border-color:#6ab0ff;color:#6ab0ff}
button.thinking:is(#who-can-help,#new-issue,#work-on-it)::before{content:'';display:inline-block;
width:10px;height:10px;margin-right:6px;border:2px solid #657083;border-top-color:#6ab0ff;
border-radius:50%;vertical-align:-2px;animation:serge-spin .8s linear infinite}
@keyframes serge-spin{to{transform:rotate(360deg)}}
@media(prefers-reduced-motion:reduce){button.thinking::before{animation:none}}
</style></head><body>
<div class="row" id="actions"></div>
<p id="fix" hidden></p>
<p id="probable" hidden></p>
<p id="note" role="status" aria-live="polite"></p>
<script>
// Project health asks for one action (work=1), Work on it: New issue with
// Serge's WDYT? answer in the draft and related threads in place of people.
const WORK=new URLSearchParams(location.search).get('work')==='1';
// Work on a batch (Project health): {group,signal,tests:[{trace_id,test_nodeid}]}
// in the fragment, too long for a query string. One issue for all of them.
const PACKET=(()=>{try{return WORK&&location.hash.length>1?
  JSON.parse(decodeURIComponent(location.hash.slice(1))):null;}catch(error){return null;}})();
const ACTIONS=WORK?[['work-on-it','Work on it',
  "Draft the failure with Serge's take, then work on it with your coding agent or open a GitHub issue.",true]]:__ACTIONS__,LOCK=__LOCK__;
const row=document.getElementById('actions'),note=document.getElementById('note');
// Every button shows at once; signed out, each carries a lock and goes to the
// dashboard GitHub sign-in, returning to the current page afterward.
function signIn(){
  let back='/';
  try{
    const here=top.location;back=here.pathname+here.search;
  }catch(error){}
  top.location.href='/auth/github?next='+encodeURIComponent(back);
}
const buttons={};
function fitFrame(){
  try{if(window.frameElement)window.frameElement.style.height=
    Math.ceil(document.body.getBoundingClientRect().height+2)+'px';}catch(error){}
}
window.addEventListener('resize',fitFrame);
for(const [id,label,help] of ACTIONS){
  const b=document.createElement('button');b.type='button';b.id=id;b.title=help;
  b.textContent=label;b.disabled=true;buttons[id]=b;row.append(b);
}
function render(signedIn){
  for(const [id,label,help,enabled] of ACTIONS){
    const b=buttons[id];b.replaceChildren();
    if(signedIn){
      b.textContent=label;b.title=help;b.disabled=!enabled;b.classList.remove('locked');
      b.onclick=id==='wdyt'?wdyt:id==='who-can-help'?whoCanHelp:
        id==='new-issue'?()=>newIssue(false):id==='work-on-it'?()=>newIssue(true):null;
    }else{
      b.insertAdjacentHTML('afterbegin',LOCK);b.append(label);
      b.title='Sign in to use Serge. '+help;b.disabled=false;b.classList.add('locked');
      b.onclick=signIn;
    }
  }
  fitFrame();
}
fetch('/auth/me',{credentials:'same-origin',cache:'no-store'})
  .then(response=>render(response.ok)).catch(()=>render(false));

const context=new URLSearchParams(location.search);
// Project health's Slow rows have no failure trace: actions get the test's
// timings instead, and the server answers from those (kind=slow).
function slowTarget(){
  if(context.get('kind')!=='slow')return null;
  let durations=[];try{durations=JSON.parse(context.get('durations')||'[]');}catch(error){}
  return {kind:'slow',test_nodeid:context.get('test_nodeid')||'',job:context.get('job')||'',
    runner:context.get('runner')||'',first_day:context.get('first_day')||'',durations};
}
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
    fix.append(' to fix this.');fix.hidden=false;fitFrame();
  }).catch(()=>{});
// Like the traceback panel: a link without var-trace_id uses the test's latest
// failing trace, which the dashboard resolves into latest_trace.
if(!context.get('trace_id')&&/^[0-9a-f]{24,32}$/i.test(context.get('latest_trace')||''))
  context.set('trace_id',context.get('latest_trace'));
if(context.get('test_nodeid'))fetch('/serge-actions/probable-fix?'+
  new URLSearchParams({trace_id:context.get('trace_id')||'',
    test_nodeid:context.get('test_nodeid')}),{cache:'no-store'})
  .then(response=>response.json()).then(data=>{
    if(data.status!=='ok'||!data.pr)return;
    const p=document.getElementById('probable'),pr=data.pr;
    const a=document.createElement('a');a.href=pr.url;a.target='_blank';
    a.rel='noopener noreferrer';a.title=pr.title;
    a.textContent='PR #'+pr.number;
    // Same rule and words as Project health's Probable fix column.
    const verdict={open:'open',merged_after:'merged, no failure on main since',
      merged_before:'merged, but the test failed on main after it'}[pr.verdict]||'';
    p.append(pr.verdict==='merged_before'?'🔎 Tried in ':'🔎 Probable fix: ',a,
      ' ('+verdict+') · '+pr.title);
    p.hidden=false;fitFrame();
  }).catch(()=>{});
const MESSAGES={disabled:'WDYT? is not configured on this server yet.',
  unauthorized:'Your dashboard session expired. Sign in again.',
  rate_limited:'You asked a lot this hour. Try again later.',
  busy:'Serge is busy with other questions. Try again in a minute.',
  unavailable:'Serge could not answer right now. Try again in a minute.',
  forbidden:'Request refused.',invalid:'This test has no trace to look at.'};
async function actionResponse(response,progress){
  if(!(response.headers.get('Content-Type')||'').includes('ndjson'))return response.json();
  const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='',result={status:'unavailable'};
  for(;;){
    const {value,done}=await reader.read();
    buffer+=decoder.decode(value||new Uint8Array(),{stream:!done});
    let cut;
    while((cut=buffer.indexOf('\n'))>=0){
      const line=buffer.slice(0,cut).trim();buffer=buffer.slice(cut+1);
      if(!line)continue;
      const event=JSON.parse(line);
      if(event.event==='done')result=event;else progress(event);
    }
    if(done)break;
  }
  return result;
}
let run=null;  // the in-flight question: {view, done}
async function wdyt(){
  const button=document.getElementById('wdyt');
  if(run){run.view.show();return;}  // reopen the popup of a running question
  const view=popup();run={view};
  button.classList.add('thinking');button.textContent='Thinking…';
  let result={status:'unavailable'};
  try{
    const slow=slowTarget();
    const response=await fetch('/serge-actions/wdyt'+(slow?'?kind=slow':''),{method:'POST',credentials:'same-origin',
      headers:{'Content-Type':'application/json','X-TCI-Action':'1'},
      body:JSON.stringify(slow||Object.fromEntries(['trace_id','test_nodeid','status','job',
        'module','pr','run_id'].map(k=>[k,context.get(k)||''])))});
    result=await actionResponse(response,event=>view.progress(event));
  }catch(error){}
  button.classList.remove('thinking');button.textContent='WDYT?';run=null;
  view.finish(result);
}

let helpersRun=null;
async function whoCanHelp(){
  if(helpersRun){helpersRun.view.show();return;}
  const button=buttons['who-can-help'];
  button.classList.add('thinking');button.textContent='Looking…';
  const view=helpersPopup();helpersRun={view};
  let result={status:'unavailable'};
  try{
    const slow=slowTarget();
    const response=await fetch('/serge-actions/who-can-help'+(slow?'?kind=slow':''),{
      method:'POST',credentials:'same-origin',
      headers:{'Content-Type':'application/json','X-TCI-Action':'1'},
      body:JSON.stringify(slow||{trace_id:context.get('trace_id')||'',
        test_nodeid:context.get('test_nodeid')||''})});
    result=await actionResponse(response,event=>view.progress(event));
  }catch(error){}
  button.classList.remove('thinking');button.textContent='Who can help';helpersRun=null;
  view.finish(result);
}

let issueRun=null;
async function newIssue(work){
  if(issueRun){issueRun.view.show();return;}
  const id=work?'work-on-it':'new-issue',button=buttons[id],view=issuePopup(work);
  issueRun={view};
  button.classList.add('thinking');button.textContent='Drafting…';
  let result={status:'unavailable'};
  try{
    const draft=PACKET?{packet:PACKET}:slowTarget()||
      {trace_id:context.get('trace_id')||'',test_nodeid:context.get('test_nodeid')||''};
    if(work)draft.work=true;
    const response=await fetch('/serge-actions/new-issue/draft',{
      method:'POST',credentials:'same-origin',
      headers:{'Content-Type':'application/json','X-TCI-Action':'1'},
      body:JSON.stringify(draft)});
    result=await actionResponse(response,event=>view.progress(event));
  }catch(error){}
  button.classList.remove('thinking');
  button.textContent=ACTIONS.find(action=>action[0]===id)[1];issueRun=null;
  view.finish(result);
}

function actionSteps(doc,host,initial){
  const style=doc.createElement('style');
  style.textContent='@keyframes tci-popup-spin{to{transform:rotate(360deg)}}'+
    '.tci-popup-progress{list-style:none;margin:0 0 12px;padding:0;color:#aeb7c2;font-size:13px}'+
    '.tci-popup-progress li{display:flex;align-items:center;gap:8px;margin:4px 0}'+
    '.tci-popup-progress .active{display:inline-block;width:12px;height:12px;box-sizing:border-box;'+
    'border:2px solid #657083;border-top-color:#6ab0ff;border-radius:50%;'+
    'animation:tci-popup-spin .8s linear infinite}'+
    '@media(prefers-reduced-motion:reduce){.tci-popup-progress .active{animation:none}}';
  const list=doc.createElement('ul');list.className='tci-popup-progress';
  list.setAttribute('role','status');list.setAttribute('aria-live','polite');
  host.append(style,list);let active=[];
  // Several steps can run at once (parallel); each is {mark,label}.
  const settle=(entry,ok)=>{
    entry.mark.className='';entry.mark.textContent=ok?'✓':'×';
    entry.mark.style.cssText='display:inline-block;width:12px;color:'+(ok?'#73bf69':'#f2495c');
  };
  const finish=ok=>{for(const entry of active)settle(entry,ok);active=[];};
  const step=(text,parallel)=>{
    if(!parallel)finish(true);
    const item=doc.createElement('li'),mark=doc.createElement('span'),label=doc.createTextNode(text);
    mark.className='active';mark.setAttribute('aria-hidden','true');
    item.append(mark,label);list.append(item);active.push({mark,label});
  };
  const find=text=>active.find(entry=>entry.label.data===text);
  const done=text=>{const entry=find(text);
    if(entry){settle(entry,true);active=active.filter(other=>other!==entry);}};
  const retext=(text,to)=>{const entry=find(text);if(entry)entry.label.data=to;};
  const progress=event=>{
    if(event.event==='step')step(event.text,event.parallel);
    else if(event.event==='step_done')done(event.text);
    else if(event.event==='step_text')retext(event.text,event.to);
  };
  step(initial);
  return {step,finish,progress};
}

// Work on it (Project health) is "Working on the failure": once the draft is
// ready its steps fold away and two choices follow, a coding agent (copy a
// prompt, no issue) or a GitHub issue (the editable form). Either way the
// dashboard records who is on the test; GitHub's assignees play no part.
// New issue (Test page) goes straight to the form.
function issuePopup(work){
  const heading=work?'Working on the failure':'New issue';
  let doc=document,host=document.body;
  for(let w=window;w.parent!==w;){
    try{w=w.parent;if(w.document.body){doc=w.document;host=doc.body;}}catch(error){break;}
  }
  doc.getElementById('tci-new-issue')?.remove();
  const el=(tag,text)=>{const n=doc.createElement(tag);if(text!==undefined)n.textContent=text;return n;};
  const post=(path,body)=>fetch(path,{method:'POST',credentials:'same-origin',
    headers:{'Content-Type':'application/json','X-TCI-Action':'1'},body:JSON.stringify(body)});
  const overlay=el('div');overlay.id='tci-new-issue';
  overlay.style.cssText='position:fixed;inset:0;z-index:1100;background:rgba(0,0,0,.55);'+
    'display:flex;align-items:center;justify-content:center;padding:16px';
  const box=el('div');box.style.cssText='background:#181b1f;color:#d8d9da;'+
    'border:1px solid #657083;border-radius:6px;max-width:760px;width:100%;'+
    'max-height:85vh;overflow:auto;padding:16px 20px;font:14px/1.5 system-ui,sans-serif';
  box.setAttribute('role','dialog');box.setAttribute('aria-modal','true');
  box.setAttribute('aria-label',heading);
  const title=el('h2',heading);title.style.margin='0 0 8px';
  const progressHost=el('div'),steps=actionSteps(doc,progressHost,'Starting issue draft');
  const content=el('div');
  const actions=el('div');actions.style.cssText='display:flex;gap:8px;justify-content:flex-end;margin-top:14px';
  const buttonCss='background:#242b35;color:#d8d9da;border:1px solid #657083;'+
    'border-radius:4px;padding:6px 12px;font:inherit;cursor:pointer';
  const close=el('button','Close');close.type='button';
  const publish=el('button','Publish issue');publish.type='submit';publish.hidden=true;
  for(const button of [close,publish])button.style.cssText=buttonCss;
  actions.append(close,publish);box.append(title,progressHost,content,actions);
  overlay.append(box);host.append(overlay);
  // An open draft reserves the test (one issue per test); closing it unpublished
  // frees the test for the next person. Working on it is not a draft: it stays.
  let heldDraft=null;
  const release=()=>{
    if(!heldDraft)return;
    const draftId=heldDraft;heldDraft=null;
    fetch('/serge-actions/new-issue/release',{method:'POST',credentials:'same-origin',keepalive:true,
      headers:{'Content-Type':'application/json','X-TCI-Action':'1'},
      body:JSON.stringify({draft_id:draftId})}).catch(()=>{});
  };
  const hide=()=>{release();overlay.remove();doc.removeEventListener('keydown',onKey);};
  const onKey=event=>{if(event.key==='Escape')hide();};
  const show=()=>{if(!overlay.isConnected){host.append(overlay);doc.addEventListener('keydown',onKey);}
    close.focus();};
  doc.addEventListener('keydown',onKey);close.onclick=hide;
  overlay.onclick=event=>{if(event.target===overlay)hide();};close.focus();
  const link=(text,url)=>{const a=el('a',text);a.href=url;a.target='_blank';
    a.rel='noopener noreferrer';return a;};

  // The editable title + body, published as a GitHub issue.
  function issueForm(data,host,done){
    const form=el('form');form.id='tci-new-issue-form';form.style.margin='0';
    publish.setAttribute('form',form.id);
    const note=el('p','Draft based on the Transformers bug report template. Edit both fields before publishing. The transformers-ci dashboard App creates the issue and names you in it.');
    note.style.cssText='color:#aeb7c2;margin:0 0 12px';
    const titleLabel=el('label','Title');titleLabel.style.display='block';
    const titleInput=el('input');titleInput.type='text';titleInput.value=data.title;
    titleInput.maxLength=200;titleInput.style.cssText='display:block;box-sizing:border-box;width:100%;'+
      'margin:4px 0 12px;padding:8px;background:#242b35;color:#d8d9da;border:1px solid #657083';
    titleLabel.append(titleInput);
    const bodyLabel=el('label','Body (Markdown)');bodyLabel.style.display='block';
    const bodyInput=el('textarea');bodyInput.value=data.body;bodyInput.maxLength=30000;
    bodyInput.rows=18;bodyInput.style.cssText='display:block;box-sizing:border-box;width:100%;'+
      'margin:4px 0 12px;padding:8px;background:#242b35;color:#d8d9da;'+
      'border:1px solid #657083;font:12px/1.5 ui-monospace,monospace;resize:vertical';
    bodyLabel.append(bodyInput);
    const status=el('p','');status.setAttribute('role','status');
    form.append(note,titleLabel,bodyLabel,status);host.replaceChildren(form);
    publish.hidden=false;publish.disabled=false;titleInput.focus();
    form.onsubmit=async event=>{
      event.preventDefault();
      if(!titleInput.value.trim()||!bodyInput.value.trim()){
        status.textContent='Title and body are required.';return;
      }
      publish.disabled=true;publish.textContent='Publishing…';status.textContent='';
      try{
        const response=await post('/serge-actions/new-issue/publish',
          {draft_id:data.draft_id,title:titleInput.value,body:bodyInput.value});
        const result=await response.json();
        if(['claimed','drafting','closed','working'].includes(result.status)){
          heldDraft=null;publish.hidden=true;done(heldBy(doc,result));return;
        }
        if(result.status==='ok'){
          heldDraft=null;publish.hidden=true;
          const p=el('p','Published ');p.append(link('issue #'+result.number,result.url));
          if(work)p.append('. You are on it now: Fix failures says so.');
          done(p);return;
        }
        status.textContent=result.status==='invalid'?'Draft expired. Create a new draft.':
          'Could not publish the issue. Check GitHub permissions and try again.';
      }catch(error){status.textContent='Could not publish the issue. Try again.';}
      publish.disabled=false;publish.textContent='Publish issue';
    };
  }

  // Coding agent: the dashboard records the work and hands back a prompt.
  async function agent(data,choice,panel){
    choice.disabled=true;panel.replaceChildren(el('p','Recording that you are on it…'));
    publish.hidden=true;
    let result={status:'unavailable'};
    try{
      result=await (await post('/serge-actions/new-issue/agent',
        {draft_id:data.draft_id,title:data.title,body:data.body})).json();
    }catch(error){}
    choice.disabled=false;
    if(['claimed','drafting','closed','working'].includes(result.status)){
      panel.replaceChildren(heldBy(doc,result));return;
    }
    if(result.status!=='ok'){
      panel.replaceChildren(el('p',result.status==='invalid'?'Draft expired. Create a new draft.':
        'Could not record the work right now. Try again.'));return;
    }
    const text=el('textarea');text.readOnly=true;text.value=result.prompt;text.rows=14;
    text.style.cssText='display:block;box-sizing:border-box;width:100%;margin:6px 0 0;padding:8px;'+
      'background:#242b35;color:#d8d9da;border:1px solid #657083;'+
      'font:12px/1.5 ui-monospace,monospace;resize:vertical';
    const shown=el('details'),summary=el('summary','Show the prompt');
    summary.style.cursor='pointer';shown.append(summary,text);
    const said=el('p');
    let copied=false;
    try{await (doc.defaultView||window).navigator.clipboard.writeText(result.prompt);copied=true;}
    catch(error){
      try{text.select();copied=doc.execCommand('copy');}catch(other){}
    }
    said.textContent=copied?'✓ Prompt copied: paste it into your coding agent. You are on it now: Fix failures says so.':
      'You are on it now. Copy the prompt below into your coding agent.';
    if(!copied){shown.open=true;}
    const again=el('button','Copy again');again.type='button';again.style.cssText=buttonCss;
    again.onclick=async()=>{
      try{await (doc.defaultView||window).navigator.clipboard.writeText(result.prompt);
        again.textContent='Copied ✓';}catch(error){shown.open=true;text.select();}
    };
    panel.replaceChildren(said,again,shown);
    if(!copied)text.select();
  }

  // Already on it: since when and how, and a way to stop.
  function myWork(data){
    const worker=data.working,p=el('p');
    p.style.cssText='margin:0 0 12px;padding:8px 10px;border:1px solid #3d71d9;border-radius:4px;background:#1b2433';
    const minutes=Math.round((Date.now()/1000-worker.since)/60);
    const since=minutes<1?'just now':minutes<120?minutes+' min ago':Math.round(minutes/60)+' h ago';
    p.append('You are working on this since '+since+', ');
    if(worker.via==='issue'&&worker.url)p.append('in ',link('#'+worker.number,worker.url),'. ');
    else p.append('with a coding agent. ');
    const stop=el('button','Stop working on it');stop.type='button';stop.style.cssText=buttonCss;
    stop.onclick=async()=>{
      stop.disabled=true;
      try{
        const result=await (await post('/serge-actions/new-issue/stop',{draft_id:data.draft_id})).json();
        if(result.status==='ok'){p.replaceChildren('You stopped working on this: it is free for others.');return;}
      }catch(error){}
      stop.disabled=false;stop.textContent='Could not stop: try again';
    };
    p.append(stop);
    return p;
  }

  function choices(data){
    const grid=el('div');
    grid.style.cssText='display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:4px 0 12px';
    const choice=(icon,name,help)=>{
      const b=el('button');b.type='button';
      b.style.cssText='display:flex;flex-direction:column;align-items:flex-start;gap:4px;text-align:left;'+
        'min-height:96px;padding:14px 16px;background:#242b35;color:#d8d9da;border:1px solid #657083;'+
        'border-radius:6px;font:inherit;cursor:pointer';
      const head=el('span',icon+' '+name);head.style.cssText='font-size:17px;font-weight:600';
      const sub=el('span',help);sub.style.cssText='color:#aeb7c2;font-size:13px';
      b.append(head,sub);return b;
    };
    const panel=el('div');
    const pressed=b=>{for(const other of grid.children){
      other.style.borderColor=other===b?'#6ab0ff':'#657083';
      other.setAttribute('aria-pressed',String(other===b));}};
    const coding=choice('🤖','Coding agent','Work on it on my own: copy a prompt for my coding agent.');
    coding.onclick=()=>{pressed(coding);agent(data,coding,panel);};
    const existing=data.working?.via==='issue'&&data.working.url;
    const issue=choice('🐙','GitHub issue',existing?'Already open as #'+data.working.number+': open it on GitHub.':
      'Create an issue on GitHub: review the draft, then publish it.');
    issue.onclick=()=>{
      pressed(issue);
      if(existing){(doc.defaultView||window).open(data.working.url,'_blank','noopener');return;}
      issueForm(data,panel,node=>panel.replaceChildren(node));
    };
    grid.append(coding,issue);
    return [grid,panel];
  }

  return {show,progress:steps.progress,finish(data){
    const ok=data.status==='ok'||data.status==='closed';
    steps.finish(ok);
    content.replaceChildren();
    if(data.status==='closed'){title.textContent='Reopen issue';box.setAttribute('aria-label','Reopen issue');}
    if(['claimed','drafting','closed','working'].includes(data.status)){
      content.append(heldBy(doc,data));return;
    }
    if(data.status!=='ok'){
      const messages={invalid:'The selected failure is unavailable.',
        unauthorized:'Your dashboard session expired. Sign in again.',
        rate_limited:'You asked a lot this hour. Try again later.',
        busy:'Serge is busy. Try again shortly.',
        unavailable:'Could not draft the issue right now. Try again shortly.'};
      content.textContent=messages[data.status]||messages.unavailable;return;
    }
    heldDraft=data.draft_id;
    if(!work){issueForm(data,content,node=>content.replaceChildren(node));return;}
    // The steps are done: fold them into one line that still opens.
    const folded=el('details'),summary=el('summary');
    const count=progressHost.querySelectorAll('li').length;
    summary.textContent='✓ Draft ready: '+data.title+' ('+count+' steps)';
    summary.style.cssText='cursor:pointer;color:#aeb7c2;font-size:13px;margin:0 0 12px';
    progressHost.replaceWith(folded);folded.append(summary,progressHost);
    if(data.working)content.append(myWork(data));
    const lead=el('p','How do you want to work on it?');lead.style.cssText='margin:0 0 8px;font-weight:600';
    content.append(lead,...choices(data));
  }};
}

// Someone else holds this test: a published issue, or a draft in progress.
// A closed issue is offered for reopening: closed is not proof it was fixed.
function heldBy(doc,data){
  const p=doc.createElement('p'),who=doc.createElement('a');
  who.href='https://github.com/'+encodeURIComponent(data.login);who.target='_blank';
  who.rel='noopener noreferrer';who.textContent='@'+data.login;
  const issue=doc.createElement('a');issue.href=data.url;issue.target='_blank';
  issue.rel='noopener noreferrer';issue.textContent='#'+data.number;
  if(data.status==='closed'){
    p.append('This test already has an issue, ',issue,', picked up by ',who,
      ', and it was closed. Reopen it rather than opening a second one.');
    return reopenIssue(doc,data,p);
  }
  if(data.status==='working'){
    const minutes=Math.round((Date.now()/1000-data.since)/60);
    p.append(who,' is working on this test'+(data.via==='issue'?' in ':' with a coding agent'));
    if(data.via==='issue'&&data.url)p.append(issue);
    p.append(', since '+(minutes<1?'just now':minutes<120?minutes+' min ago':Math.round(minutes/60)+' h ago')+'.');
  }else if(data.status==='claimed'){
    p.append('This test already has an open issue, ',issue,'. Comment there, or use Pick up on Fix failures, instead of opening a second one.');
  }else{
    const minutes=Math.round((Date.now()/1000-data.since)/60);
    p.append(who,' started an issue for this test '+(minutes<1?'just now':minutes+' min ago')+'. '+
      'It frees up if they close it without publishing.');
  }
  return p;
}
function reopenIssue(doc,data,p){
  const box=doc.createElement('div'),note=doc.createElement('p');
  note.setAttribute('role','status');
  const button=doc.createElement('button');button.type='button';
  button.textContent='Reopen #'+data.number;
  button.style.cssText='background:#242b35;color:#d8d9da;border:1px solid #657083;'+
    'border-radius:4px;padding:6px 12px;font:inherit;cursor:pointer';
  button.onclick=async()=>{
    button.disabled=true;button.textContent='Reopening…';note.textContent='';
    let result={status:'unavailable'};
    try{
      const response=await fetch('/serge-actions/new-issue/reopen',{
        method:'POST',credentials:'same-origin',
        headers:{'Content-Type':'application/json','X-TCI-Action':'1'},
        body:JSON.stringify({test_nodeid:context.get('test_nodeid')||''})});
      result=await response.json();
    }catch(error){}
    if(result.status==='ok'){
      const link=doc.createElement('a');link.href=result.url;link.target='_blank';
      link.rel='noopener noreferrer';link.textContent='#'+result.number;
      box.replaceChildren(doc.createElement('p'));
      box.firstChild.append('Reopened ',link,'. You are on it now: the issue says so.');
      return;
    }
    if(result.status==='claimed'){box.replaceChildren(heldBy(doc,result));return;}
    button.disabled=false;button.textContent='Reopen #'+data.number;
    note.textContent=result.status==='unauthorized'?'Your dashboard session expired. Sign in again.':
      'Could not reopen the issue. Check GitHub permissions and try again.';
  };
  box.append(p,button,note);
  return box;
}

function helperRefs(doc,parent,value,evidenceUrl,prLinks){
  const text=String(value||'');
  const pattern=/https:\/\/github\.com\/[\w.-]+\/[\w.-]+\/(?:pull|issues)\/\d+|[\w.-]+\/[\w.-]+#\d+|(?:pull request|PR|issues?)\s*#\d+|@[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}[A-Za-z0-9])?|#\d{1,7}\b/gi;
  let last=0;
  for(const match of text.matchAll(pattern)){
    const ref=match[0],at=match.index;
    if(at>last)parent.append(doc.createTextNode(text.slice(last,at)));
    let url='';
    if(ref.startsWith('https://github.com/')){
      const pr=ref.match(/^https:\/\/github\.com\/huggingface\/transformers\/pull\/(\d+)$/i);
      url=pr&&prLinks?.[pr[1]]||ref;
    }
    else if(ref.startsWith('@'))url='https://github.com/'+ref.slice(1);
    else if(/^[\w.-]+\/[\w.-]+#\d+$/.test(ref)){
      const [repo,number]=ref.split('#');
      url=repo.toLowerCase()==='huggingface/transformers'&&prLinks?.[number]||
        (String(evidenceUrl||'').endsWith('/'+number)?evidenceUrl:
          'https://github.com/'+repo+'/issues/'+number);
    }
    else{
      const number=ref.match(/#(\d+)$/)?.[1];
      const route=/^(?:PR|pull request)/i.test(ref)?'pull':'issues';
      const isIssue=/^issues?\s/i.test(ref);
      url=!isIssue&&prLinks?.[number]||
        (ref.startsWith('#')&&String(evidenceUrl||'').endsWith('/'+number)?evidenceUrl:
          'https://github.com/huggingface/transformers/'+route+'/'+number);
    }
    const link=doc.createElement('a');link.href=url;link.target='_blank';
    link.rel='noopener noreferrer';link.textContent=ref;parent.append(link);
    last=at+ref.length;
  }
  if(last<text.length)parent.append(doc.createTextNode(text.slice(last)));
}

function helpersPopup(){
  let doc=document,host=document.body;
  for(let w=window;w.parent!==w;){
    try{w=w.parent;if(w.document.body){doc=w.document;host=doc.body;}}catch(error){break;}
  }
  doc.getElementById('tci-helpers')?.remove();
  const el=(tag,text)=>{const n=doc.createElement(tag);if(text)n.textContent=text;return n;};
  const overlay=el('div');overlay.id='tci-helpers';
  overlay.style.cssText='position:fixed;inset:0;z-index:1100;background:rgba(0,0,0,.55);'+
    'display:flex;align-items:center;justify-content:center;padding:16px';
  const box=el('div');box.style.cssText='background:#181b1f;color:#d8d9da;'+
    'border:1px solid #657083;border-radius:6px;max-width:640px;width:100%;'+
    'max-height:80vh;overflow:auto;padding:16px 20px;font:14px/1.5 system-ui,sans-serif';
  box.setAttribute('role','dialog');box.setAttribute('aria-modal','true');
  box.setAttribute('aria-label','Who can help');
  const title=el('h2','Who can help');title.style.margin='0 0 8px';
  const progressHost=el('div'),steps=actionSteps(doc,progressHost,'Starting helper lookup');
  const content=el('div');
  const close=el('button','Close');close.type='button';close.style.marginTop='14px';
  const hide=()=>{overlay.remove();doc.removeEventListener('keydown',onKey);};
  const onKey=event=>{if(event.key==='Escape')hide();};
  const show=()=>{if(!overlay.isConnected){host.append(overlay);doc.addEventListener('keydown',onKey);}
    close.focus();};
  close.onclick=hide;overlay.onclick=event=>{if(event.target===overlay)hide();};
  box.append(title,progressHost,content,close);overlay.append(box);host.append(overlay);
  doc.addEventListener('keydown',onKey);close.focus();
  return {show,progress(event){if(event.event==='step')steps.step(event.text);},finish(data){
    steps.finish(data.status==='ok');
    content.replaceChildren();
    if(data.status==='ok'){
      content.append(el('p','People connected to '+data.location+'; suggestions, not assignments.'));
      const list=el('ul');list.style.paddingLeft='20px';
      for(const person of data.people||[]){
        const item=el('li');item.style.margin='10px 0';
        const link=el('a','@'+person.login);
        link.href='https://github.com/'+encodeURIComponent(person.login);
        link.target='_blank';link.rel='noopener noreferrer';
        const evidence=el('a',person.evidence);
        const pr=String(person.url||'').match(/^https:\/\/github\.com\/huggingface\/transformers\/pull\/(\d+)$/i);
        evidence.href=pr&&data.pr_links?.[pr[1]]||person.url;
        evidence.target='_blank';evidence.rel='noopener noreferrer';
        item.append(link,' — ');
        helperRefs(doc,item,person.why,person.url,data.pr_links);
        item.append(' (',evidence,')');list.append(item);
      }
      content.append(list);
    }else{
      const messages={no_line:'The trace has no repository line to inspect.',
        no_people:'No people with enough evidence were found for this line.',
        invalid:'The selected failure is unavailable.',
        unauthorized:'Your dashboard session expired. Sign in again.',
        rate_limited:'You asked a lot this hour. Try again later.',
        busy:'Serge is busy. Try again shortly.',
        unavailable:'Could not find suggestions right now. Try again shortly.'};
      content.textContent=messages[data.status]||messages.unavailable;
    }
  }};
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
  // The top-most same-origin page: the popup centres on what the viewer sees,
  // not on an iframe taller than the screen (the Test page inside Grafana).
  let doc=document,host=document.body;
  for(let w=window;w.parent!==w;){
    try{w=w.parent;if(w.document.body){doc=w.document;host=doc.body;}}catch(error){break;}
  }
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
  const title=el('div','font-weight:600;font-size:16px;margin:0 0 8px;display:flex;'+
    'align-items:center;gap:8px');
  const logo=el('img','width:22px;height:22px');logo.src='/serge-actions/logo.png';logo.alt='';
  title.append(logo,'Serge thinks…');
  const test=el('div','color:#8e9197;font:12px ui-monospace,monospace;overflow-wrap:anywhere;'+
    'margin:0 0 8px',context.get('test_nodeid')||'');
  // Live progress: finished steps get a check, the current one pulses.
  const steps=el('ul','list-style:none;margin:0 0 8px;padding:0;color:#8e9197;font-size:13px');
  steps.setAttribute('aria-live','polite');
  const answer=el('div','margin:6px 0 0');let streamed='';
  const foot=el('div','color:#8e9197;font-size:12px;margin:10px 0 0');
  const style=el('style');
  style.textContent='';
  box.append(style,title,test,steps,answer,foot);
  let current=null;
  const step=text=>{
    if(current){current.firstChild.textContent='✓ ';}
    const li=el('li','margin:2px 0'),mark=el('span','','● ');
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
      if(current){current.firstChild.textContent='✓ ';}
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
    "__LOCK__", json.dumps(LOCK_SVG)
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
