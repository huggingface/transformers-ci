"""Serge Actions panel for the pytest Test dashboard.

A static shell for now: the three actions are shown to signed-in Grafana users
and stay disabled until the authenticated action API behind them exists.
Hiding them from anonymous viewers is presentation only, never the
authorization gate.
"""

from __future__ import annotations

import json

# (id, label, what Serge will do, wired yet). Rendered with textContent only.
ACTIONS = (
    ("new-issue", "New issue", "Serge writes a GitHub issue for this failure.", False),
    ("fix-it", "Fix it!", "Serge reproduces the failure and opens a fix PR.", False),
    ("wdyt", "WDYT?", "Serge takes a quick look and tells you what it thinks.", True),
)

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
async function wdyt(){
  const button=document.getElementById('wdyt');
  if(button.classList.contains('thinking'))return;
  button.classList.add('thinking');document.body.classList.add('thinking');
  button.textContent='Thinking…';
  let data;
  try{
    const response=await fetch('/serge-actions/wdyt',{method:'POST',credentials:'same-origin',
      headers:{'Content-Type':'application/json','X-TCI-Action':'1'},
      body:JSON.stringify(Object.fromEntries(['trace_id','test_nodeid','status','job',
        'module','pr','run_id'].map(k=>[k,context.get(k)||''])))});
    data=await response.json();
  }catch(error){data={status:'unavailable'};}
  button.classList.remove('thinking');document.body.classList.remove('thinking');
  button.textContent='WDYT?';
  popup(data);
}

// Tiny Markdown subset -> DOM nodes (never innerHTML): paragraphs, "- " bullets,
// **bold** and `code`.
function inline(doc,parent,text){
  for(const part of text.split(/(\*\*[^*]+\*\*|`[^`]+`)/)){
    if(!part)continue;
    if(part.startsWith('**')&&part.endsWith('**')&&part.length>4){
      const b=doc.createElement('strong');b.textContent=part.slice(2,-2);parent.append(b);
    }else if(part.startsWith('`')&&part.endsWith('`')&&part.length>2){
      const c=doc.createElement('code');c.textContent=part.slice(1,-1);
      c.style.cssText='background:rgba(204,204,220,.1);padding:0 4px;border-radius:3px;'+
        'font:12px ui-monospace,monospace';parent.append(c);
    }else parent.append(doc.createTextNode(part));
  }
}
function markdown(doc,text){
  const root=doc.createElement('div');let list=null;
  for(const raw of text.split('\n')){
    const line=raw.trim();
    if(!line){list=null;continue;}
    const bullet=line.match(/^[-*] (.*)$/);
    if(bullet){
      if(!list){list=doc.createElement('ul');list.style.cssText='margin:6px 0;padding-left:20px';
        root.append(list);}
      const li=doc.createElement('li');li.style.margin='3px 0';inline(doc,li,bullet[1]);list.append(li);
    }else{
      list=null;const p=doc.createElement('p');p.style.margin='6px 0';
      inline(doc,p,line.replace(/^#+ /,''));root.append(p);
    }
  }
  return root;
}

function popup(data){
  let doc=document,host=document.body;
  try{if(parent.document.body){doc=parent.document;host=doc.body;}}catch(error){}
  doc.getElementById('tci-wdyt')?.remove();
  const overlay=doc.createElement('div');overlay.id='tci-wdyt';
  overlay.style.cssText='position:fixed;inset:0;z-index:1100;background:rgba(0,0,0,.55);'+
    'display:flex;align-items:center;justify-content:center;padding:16px';
  const box=doc.createElement('div');box.setAttribute('role','dialog');
  box.setAttribute('aria-modal','true');box.setAttribute('aria-label','Serge thinks');
  box.style.cssText='background:#181b1f;color:#d8d9da;border:1px solid rgba(204,204,220,.2);'+
    'border-radius:6px;max-width:640px;width:100%;max-height:80vh;overflow:auto;'+
    'padding:16px 20px;font:14px/1.5 Inter,system-ui,sans-serif;box-shadow:0 8px 32px #000';
  const title=doc.createElement('div');title.textContent='✨🦄 Serge thinks…';
  title.style.cssText='font-weight:600;font-size:16px;margin:0 0 8px';
  const test=doc.createElement('div');test.textContent=context.get('test_nodeid')||'';
  test.style.cssText='color:#8e9197;font:12px ui-monospace,monospace;overflow-wrap:anywhere;'+
    'margin:0 0 8px';
  box.append(title,test);
  if(data.status==='ok'){
    box.append(markdown(doc,data.answer));
    const foot=doc.createElement('div');
    foot.textContent='Quick opinion from '+data.model+', based only on this page. It can be wrong.';
    foot.style.cssText='color:#8e9197;font-size:12px;margin:10px 0 0';box.append(foot);
  }else{
    const p=doc.createElement('p');p.textContent=MESSAGES[data.status]||MESSAGES.unavailable;
    box.append(p);
  }
  const actions=doc.createElement('div');
  actions.style.cssText='display:flex;gap:8px;justify-content:flex-end;margin:14px 0 0';
  const button=(label,help,disabled)=>{const b=doc.createElement('button');b.type='button';
    b.textContent=label;b.title=help;b.disabled=disabled;
    b.style.cssText='background:#242b35;color:#d8d9da;border:1px solid #657083;'+
      'border-radius:4px;padding:5px 12px;font:inherit;cursor:'+(disabled?'not-allowed':'pointer')+
      ';opacity:'+(disabled?'.55':'1');return b;};
  const deeper=button('Deeper investigation','Runs a full Serge investigation.',true);
  const close=button('Close','',false);actions.append(deeper,close);box.append(actions);
  overlay.append(box);host.append(overlay);
  const dismiss=()=>{overlay.remove();doc.removeEventListener('keydown',onKey);};
  const onKey=event=>{if(event.key==='Escape')dismiss();};
  close.addEventListener('click',dismiss);doc.addEventListener('keydown',onKey);
  overlay.addEventListener('click',event=>{if(event.target===overlay)dismiss();});
  close.focus();
}
</script></body></html>"""

PAGE_HTML = PAGE_HTML.replace("__ACTIONS__", json.dumps(ACTIONS)).replace(
    "__NYAN__", NYAN_SVG
)
