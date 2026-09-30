"""Serge Actions panel for the pytest Test dashboard.

A static shell for now: the three actions are shown to signed-in Grafana users
and stay disabled until the authenticated action API behind them exists.
Hiding them from anonymous viewers is presentation only, never the
authorization gate.
"""

from __future__ import annotations

import json

# (id, label, what Serge will do). Rendered with textContent only.
ACTIONS = (
    ("new-issue", "New issue", "Serge writes a GitHub issue for this failure."),
    ("fix-it", "Fix it!", "Serge reproduces the failure and opens a fix PR."),
    ("wdyt", "WDYT?", "Serge takes a quick look and tells you what it thinks."),
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

PAGE_HTML = """<!doctype html><html><head><meta charset="utf-8">
<style>
body{margin:0;padding:8px 10px;background:#181b1f;color:#d8d9da;
font:13px/1.45 system-ui,sans-serif}
.row{display:flex;gap:6px;flex-wrap:wrap}
button{flex:1 1 0;min-width:0;background:#242b35;color:#d8d9da;
border:1px solid #657083;border-radius:4px;padding:5px 8px;font:inherit;cursor:pointer;
white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
button:disabled{opacity:.55;cursor:not-allowed}
p{margin:6px 0 0;color:#8e9197;font-size:12px}a{color:#6ab0ff;text-decoration:none}
.sky{position:relative;height:24px;margin:0 0 6px;overflow:hidden}
.nyan{position:absolute;top:0;height:24px;width:67px;left:calc(100% - 67px);
animation:fly 7s linear infinite}
.nyan .cat,.nyan .w0,.nyan .w1{animation:bob .4s steps(1) infinite}
.nyan .w1{animation-delay:-.2s}
@keyframes fly{from{left:-67px}to{left:100%}}
@keyframes bob{0%{transform:translateY(0)}50%{transform:translateY(1px)}}
@media (prefers-reduced-motion:reduce){.nyan,.nyan *{animation:none}}
</style></head><body>
<div class="sky">__NYAN__</div>
<div class="row" id="actions" hidden></div>
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
  for(const [id,label,help] of ACTIONS){
    const b=document.createElement('button');b.type='button';b.id=id;
    b.textContent=label;b.title=help+' (coming soon)';b.disabled=true;row.append(b);
  }
  row.hidden=false;note.textContent='Coming soon.';
}).catch(signIn);
</script></body></html>"""

PAGE_HTML = PAGE_HTML.replace("__ACTIONS__", json.dumps(ACTIONS)).replace(
    "__NYAN__", NYAN_SVG
)
