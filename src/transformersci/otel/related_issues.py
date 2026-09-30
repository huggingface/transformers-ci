"""Read-only Relore suggestions for the public test dashboard.

The HTTP adapter intentionally pins the wire version it was tested against.
A version mismatch is unavailable, never an empty search or an automatic upgrade.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from http.client import HTTPException
from urllib.error import URLError
from urllib.request import Request, urlopen

REPOSITORY = "huggingface/transformers"
WIRE_VERSION = "0.3.17"
MAX_RESPONSE_BYTES = 512 * 1024
_cache: OrderedDict[tuple[str, str, str], tuple[float, dict]] = OrderedDict()
_inflight: set[tuple[str, str, str]] = set()
_lock = threading.Lock()


def search_payload(nodeid: str, details: list[dict[str, str]]) -> dict:
    # Leave signals in the text: explicit tests/errors filters would AND them
    # across expansion legs and hide reports of the same error in another test.
    context = "\n".join(
        f"{item.get('exception_type', '')}: {item.get('exception_message', '')[:2500]}"
        for item in details[:2]
    )
    return {
        "repos": [REPOSITORY],
        "kind": "failure",
        "query": f"{nodeid}\n{context}".strip(),
        "expand": True,
        "compact": True,
        "limit": 10,
    }


def search(base_url: str, payload: dict) -> list[dict]:
    request = Request(
        base_url.rstrip("/") + "/api/v1/search",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "x-relore-client": WIRE_VERSION},
        method="POST",
    )
    with urlopen(request, timeout=8) as response:
        if response.headers.get("x-relore-version") != WIRE_VERSION:
            raise ValueError("Relore wire version mismatch")
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("Relore response too large")
    result = json.loads(raw)
    if not isinstance(result, dict) or not isinstance(result.get("hits"), list):
        raise ValueError("Invalid Relore search response")
    hits = []
    seen = set()
    for hit in result["hits"]:
        if not isinstance(hit, dict):
            raise ValueError("Invalid Relore hit")
        number = hit.get("number")
        kind = hit.get("type")
        if (
            hit.get("repo") != REPOSITORY
            or type(number) is not int
            or number <= 0
            or kind not in ("issue", "pr")
            or number in seen
        ):
            continue
        seen.add(number)
        route = "pull" if kind == "pr" else "issues"
        hits.append(
            {
                "number": number,
                "type": kind,
                "url": f"https://github.com/{REPOSITORY}/{route}/{number}",
                **{
                    key: str(hit.get(key) or "")[:limit]
                    for key, limit in (
                        ("title", 300),
                        ("snippet", 700),
                        ("author", 100),
                        ("age", 100),
                        ("trust", 100),
                    )
                },
            }
        )
        if len(hits) == 5:
            break
    return hits


def lookup(
    trace_id: str,
    nodeid: str,
    load_details: Callable[[], list[dict[str, str]]],
) -> dict:
    """Bound cache size and concurrent work independently of dashboard traffic."""
    base_url = os.getenv("PYTEST_TRACE_EXPORTER_RELORE_URL", "").strip()
    if not base_url:
        return {"status": "unavailable", "hits": []}
    key = (base_url, trace_id, nodeid)
    with _lock:
        cached = _cache.get(key)
        if cached and cached[0] > time.monotonic():
            _cache.move_to_end(key)
            return cached[1]
        if key in _inflight or len(_inflight) >= 2:
            return {"status": "busy", "hits": []}
        _inflight.add(key)
    result = None
    try:
        details = load_details() if trace_id else []
        hits = search(base_url, search_payload(nodeid, details))
        result = {
            "status": "ok",
            "hits": hits,
            "context": "failure" if details else "test-only",
        }
        ttl = 300 if details else 30
    except (OSError, URLError, ValueError, HTTPException):
        result = {"status": "unavailable", "hits": []}
        ttl = 15
    finally:
        # Removal below is atomic with publishing the cache. Unexpected errors
        # still release capacity, then propagate to the caller.
        with _lock:
            _inflight.discard(key)
            if result is not None:
                _cache[key] = (time.monotonic() + ttl, result)
                _cache.move_to_end(key)
                while len(_cache) > 256:
                    _cache.popitem(last=False)
    return result


PAGE_HTML = """<!doctype html><html><head><meta charset="utf-8">
<style>
html,body{height:100%}
body{margin:0;padding:10px;box-sizing:border-box;display:flex;flex-direction:column;
background:#181b1f;color:#d8d9da;font:13px/1.45 system-ui,sans-serif}body>*{flex:none}
p{margin:0 0 6px;color:#aeb7c2}ul{list-style:none;padding:0;margin:0}
#results{flex:1 1 0;min-height:72px;overflow-y:auto;border-bottom:1px solid #343b45}
#results li{padding:6px 0;border-top:1px solid #343b45}a{color:#6ab0ff;text-decoration:none}
a:hover{text-decoration:underline}.meta{color:#8e9197;font-size:12px;margin:1px 0}
.snippet{color:#aeb7c2;font-size:12px;overflow-wrap:anywhere;display:-webkit-box;
-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
button{background:#242b35;color:#d8d9da;
border:1px solid #657083;border-radius:4px;padding:4px 10px;cursor:pointer}
</style></head><body>
<h3 style="font-size:14px;margin:0 0 6px">Potential related issues</h3>
<p id="status" role="status" aria-live="polite">Searching related issues…</p>
<ul id="results"></ul><button id="retry" hidden>Retry</button>
<script>
const statusNode=document.getElementById('status'), results=document.getElementById('results');
const retry=document.getElementById('retry');
async function load(){
  retry.hidden=true; results.replaceChildren(); statusNode.hidden=false;
  statusNode.textContent='Searching related issues…';
  const url=new URL(location.href);url.searchParams.set('format','json');
  try {
    const response=await fetch(url,{cache:'no-store'});
    if(!response.ok)throw new Error('Lookup failed');
    const data=await response.json();
    if(data.status!=='ok'){
      statusNode.textContent=data.status==='busy'?'A search is already running. Try again shortly.':
        'Related issues are temporarily unavailable.';retry.hidden=false;return;
    }
    statusNode.textContent='No related issues found.';statusNode.hidden=data.hits.length>0;
    for(const hit of data.hits){
      const li=document.createElement('li'),a=document.createElement('a');
      a.href=hit.url;a.target='_blank';a.rel='noopener noreferrer';
      a.textContent=(hit.type==='pr'?'PR':'Issue')+' #'+hit.number+' · '+hit.title;
      const meta=document.createElement('div');meta.className='meta';
      meta.textContent=[hit.author,hit.age,hit.trust].filter(Boolean).join(' · ');
      const snippet=document.createElement('div');snippet.className='snippet';snippet.textContent=hit.snippet;snippet.title=hit.snippet;
      li.append(a,meta,snippet);results.append(li);
    }
  }catch(error){statusNode.textContent='Related issues are temporarily unavailable.';retry.hidden=false;}
}
retry.addEventListener('click',load);load();
</script></body></html>"""


# Match the PR dashboard's Branch header and metadata chips. All dynamic text
# comes from URLSearchParams and textContent, never JS/HTML interpolation.
SUMMARY_HTML = PAGE_HTML.replace(
    "</head><body>",
    """<style>
.tci-hd{font:600 18px/1.3 Inter,system-ui,sans-serif;margin:0 0 10px;
display:flex;align-items:center;gap:8px;flex-wrap:wrap;overflow-wrap:anywhere}
.tci-meta{list-style:none;margin:0;padding:0;display:flex;flex-wrap:wrap;gap:6px}
.tci-meta li{display:inline-flex;align-items:center;gap:6px;min-width:0;max-width:100%;
padding:3px 8px;border:1px solid rgba(204,204,220,.15);border-radius:4px;
background:rgba(204,204,220,.04);font-size:12px;line-height:18px}
.tci-meta .k{flex:none;color:#8e9197;font-size:10px;font-weight:600;
text-transform:uppercase;letter-spacing:.05em}
.tci-meta .v{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.cmdLabel{font-size:14px;margin:12px 0 6px}.cmdRow{display:flex;gap:6px;align-items:stretch;
margin:0 0 12px}.cmdRow button{flex:none}.cmd{display:block;flex:1 1 auto;min-width:0;
padding:8px 12px;border:1px solid rgba(204,204,220,.15);border-radius:4px;
background:rgba(204,204,220,.04);font:13px/1.45 ui-monospace,monospace;
white-space:pre-wrap;overflow-wrap:anywhere}
</style></head><body>
<div class="tci-hd" id="test-heading"></div><ul class="tci-meta" id="test-meta"></ul>
<h3 class="cmdLabel">Reproduce locally</h3>
<div class="cmdRow"><code class="cmd" id="command"></code>
<button id="copy" type="button" aria-label="Copy command">Copy</button></div>
<script>
const context=new URLSearchParams(location.search);
const node=context.get('test_nodeid')||'';
document.getElementById('test-heading').textContent=node;
for(const [key,label] of [['status','Status'],['module','Module'],['job','Job'],['pr','PR'],['run_id','Run']]){
  const value=context.get(key);if(!value)continue;
  const li=document.createElement('li'),k=document.createElement('span'),v=document.createElement('span');
  k.className='k';k.textContent=label;v.className='v';v.textContent=value;v.title=value;
  li.append(k,v);document.getElementById('test-meta').append(li);
}
const shellQuote=value=>"'"+value.replaceAll("'","'\\"'\\"'")+"'";
document.getElementById('command').textContent=node.startsWith('utils/checkers.py::')?
  'make '+shellQuote(node.split('::').pop()):'pytest -svx '+shellQuote(node);
const copyButton=document.getElementById('copy');
copyButton.addEventListener('click',async()=>{
  const text=document.getElementById('command').textContent;
  try{await navigator.clipboard.writeText(text);}catch(error){
    const area=document.createElement('textarea');area.value=text;document.body.append(area);
    area.select();document.execCommand('copy');area.remove();
  }
  copyButton.textContent='Copied';setTimeout(()=>{copyButton.textContent='Copy';},1500);
});
</script>""",
)
