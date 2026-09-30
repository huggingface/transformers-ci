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
</style></head><body>
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

PAGE_HTML = PAGE_HTML.replace("__ACTIONS__", json.dumps(ACTIONS))
