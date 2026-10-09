"""The ``/pr-search`` page: a "Find a PR" box that sits in Grafana's top bar.

Every dashboard carries a hidden text panel whose iframe appends
``<iframe id="tci-pr-search" src="/pr-search">`` to Grafana's ``<body>``, fixed over
the slot of Grafana's own "Search..." box (which only finds dashboard
titles). This page is that iframe. It is same-origin with Grafana (the ingress
routes ``/pr-search`` here), so it hides Grafana's box, sizes its own frame, and
queries Prometheus through Grafana's ``/api/ds/query`` like any panel does.
It also replaces Grafana's menu and breadcrumbs with the Hugging Face emoji
and the current dashboard's navigation links.
Until it loads the frame is 0x0, so a down exporter leaves Grafana's box alone.
"""

from __future__ import annotations

PR_DASHBOARD_URL = "/d/pytest-observability-by-pr/pytest-observability-branch"

SEARCH_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Find a PR</title>
<style>
html,body{margin:0;background:transparent;font:14px Inter,Helvetica,Arial,sans-serif;overflow:hidden}
#box{box-sizing:border-box;width:100%;height:32px;padding:0 8px 0 30px;border:1px solid var(--line);
 border-radius:4px;background:var(--bg) no-repeat 8px center;color:var(--fg);outline:none;font:inherit}
#box:focus{border-color:#3d71d9;box-shadow:0 0 0 1px #3d71d9}
#box::placeholder{color:var(--dim)}
#list{display:none;margin-top:4px;border:1px solid var(--line);border-radius:4px;background:var(--panel);
 max-height:396px;overflow-y:auto;box-shadow:0 8px 24px rgba(0,0,0,.4)}
#list.open{display:block}
a.row{display:grid;grid-template-columns:64px 1fr;gap:0 10px;padding:6px 10px;color:var(--fg);text-decoration:none;
 border-bottom:1px solid var(--line)}
a.row:last-child{border-bottom:0}
a.row.sel,a.row:hover{background:var(--hover)}
.num{color:#6e9fff;font-variant-numeric:tabular-nums}
.title{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.meta{grid-column:2;color:var(--dim);font-size:12px}
.st-open{color:#73bf69}.st-merged{color:#8f7ee7}.st-closed{color:var(--dim)}
.msg{padding:8px 10px;color:var(--dim)}
.sk{padding:8px 10px;border-bottom:1px solid var(--line)}
.sk:last-child{border-bottom:0}
.sk i{display:block;height:10px;margin:4px 0;border-radius:3px;background:linear-gradient(90deg,var(--hover) 25%,var(--line) 50%,var(--hover) 75%);
 background-size:200% 100%;animation:shimmer 1.2s linear infinite}
.sk i+i{width:40%;height:8px}
@keyframes shimmer{from{background-position:200% 0}to{background-position:-200% 0}}
</style></head>
<body>
<input id="box" type="search" autocomplete="off" spellcheck="false"
 placeholder="Find a PR: number, title or author" aria-label="Find a PR">
<div id="list" role="listbox"></div>
<script>
(function () {
  var PR_URL = "__PR_URL__";
  var WIDTH = 460, MAXH = 396, MIN_CHARS = 3;
  var fe = window.frameElement, pd = parent.document;
  var box = document.getElementById("box"), list = document.getElementById("list");
  var rows = [], sel = -1, seq = 0, timer = null;

  // Hide Grafana's own box; this one takes its place. Grafana 13 renders the
  // wide box as the command-palette trigger and, below the lg breakpoint, an
  // icon button labelled "Search...".
  if (!pd.getElementById("tci-pr-search-style")) {
    var st = pd.createElement("style");
    st.id = "tci-pr-search-style";
    st.textContent = '[data-testid="data-testid Command palette trigger"],' +
      'header button[aria-label="Search..."]{visibility:hidden!important}';
    pd.head.appendChild(st);
  }

  function theme() {
    var cs = parent.getComputedStyle(pd.body), light = /light/.test(pd.body.className) ||
      (parent.grafanaBootData && parent.grafanaBootData.user && parent.grafanaBootData.user.lightTheme);
    var r = document.documentElement.style;
    r.setProperty("--fg", cs.color);
    r.setProperty("--bg", light ? "#fff" : "#111217");
    r.setProperty("--panel", light ? "#fff" : "#181b1f");
    r.setProperty("--hover", light ? "rgba(36,41,46,.08)" : "rgba(204,204,220,.08)");
    r.setProperty("--line", light ? "rgba(36,41,46,.2)" : "rgba(204,204,220,.2)");
    r.setProperty("--dim", light ? "rgba(36,41,46,.6)" : "rgba(204,204,220,.6)");
    box.style.backgroundImage = "url(\\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' " +
      "width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='" + encodeURIComponent(cs.color) +
      "' stroke-width='2'><circle cx='11' cy='11' r='7'/><path d='m20 20-3.5-3.5'/></svg>\\")";
  }

  // Keep this customization in the shared loader rather than each dashboard.
  // Clone dashboard links: moving React-owned nodes breaks later renders.
  function header() {
    var h = pd.querySelector('header'), crumbs = h && h.querySelector('nav[aria-label="Breadcrumbs"]');
    if (!crumbs) return;
    if (!pd.getElementById('tci-header-style')) {
      var style = pd.createElement('style');
      style.id = 'tci-header-style';
      style.textContent = '.main-view:has(#tci-header-nav) main{padding-left:0!important}' +
        'header:has(#tci-header-nav){left:0!important;width:100%!important}' +
        '[data-testid="data-testid navigation mega-menu"],button[aria-label="Open menu"],' +
        'button[aria-label="Toggle menu"],button[aria-label="Close menu"],button[aria-label="Main menu"],' +
        'header nav[aria-label="Breadcrumbs"],body:has(#tci-header-nav) [data-testid="data-testid Dashboard link"]{display:none!important}' +
        'body:has(#tci-header-nav) [data-testid="data-testid new share link-button"],' +
        '[data-tci-empty-controls="true"]{display:none!important}' +
        'body:has(#tci-header-nav) [data-testid="data-testid Sidebar container"]{display:none!important}' +
        'body:has(#tci-header-nav) [data-testid="data-testid DashboardSidebarSplitter primary body"]{padding-right:0!important}' +
        '#tci-header-nav{display:flex;align-items:center;gap:8px;min-width:0;overflow-x:auto;white-space:nowrap}' +
        '#tci-header-nav a{display:inline-flex;align-items:center;box-sizing:border-box;height:32px;flex:none;color:inherit;text-decoration:none;border:1px solid rgba(128,128,128,.4);border-radius:4px;padding:0 8px;font:600 12px/18px Inter,system-ui,sans-serif}' +
        '#tci-header-nav a:hover{background:rgba(128,128,128,.15)}' +
        '#tci-header-nav a:focus-visible{outline:2px solid #6e9fff;outline-offset:-2px}' +
        '#tci-header-nav a[aria-current="page"]{border-color:#6e9fff}' +
        '#tci-header-nav .tci-home{display:flex;align-items:center;gap:8px;font-size:14px;border:0;padding:0 4px;line-height:32px}' +
        '#tci-header-nav .tci-home .tci-emoji{font-size:26px}' +
        '@media(max-width:1250px){header:has(#tci-header-nav){height:128px!important}' +
        '.main-view:has(#tci-header-nav)>div:not([data-testid="data-testid navigation mega-menu"]){padding-top:128px!important}' +
        '#tci-header-nav{position:absolute;top:44px;left:16px;right:16px;max-width:none}' +
        '#tci-header-nav .tci-home{position:fixed;top:4px;left:16px}' +
        'header img[alt="Grafana"]{display:none!important}}';
      pd.head.appendChild(style);
    }
    var nav = pd.getElementById('tci-header-nav');
    if (!nav) {
      nav = pd.createElement('nav');
      nav.id = 'tci-header-nav';
      nav.setAttribute('aria-label', 'CI dashboards');
      crumbs.parentNode.insertBefore(nav, crumbs);
    }
    var links = Array.from(pd.querySelectorAll('[data-testid="data-testid Dashboard link"]'));
    var signature = JSON.stringify(links.map(function (a) { return [a.textContent, a.getAttribute('href')]; }));
    if (nav.dataset.links !== signature) {
      nav.dataset.links = signature;
      nav.replaceChildren();
      var home = pd.createElement('a');
      home.href = '/';
      home.className = 'tci-home';
      var emoji = pd.createElement('span');
      emoji.className = 'tci-emoji';
      emoji.textContent = '\\uD83E\\uDD17';
      emoji.setAttribute('aria-hidden', 'true');
      home.appendChild(emoji);
      home.appendChild(pd.createTextNode('Transformers CI'));
      home.setAttribute('aria-label', 'Transformers CI home');
      nav.appendChild(home);
      links.forEach(function (a) {
        var copy = pd.createElement('a');
        copy.href = a.getAttribute('href');
        copy.textContent = a.textContent;
        copy.title = a.title || a.textContent;
        if (a.target) copy.target = a.target;
        if (a.rel) copy.rel = a.rel;
        if (a.pathname === parent.location.pathname) copy.setAttribute('aria-current', 'page');
        nav.appendChild(copy);
      });
    }
    // Remove the empty controls row, but retain rows with variables or time controls.
    var controls = pd.querySelector('[data-testid="data-testid dashboard controls"]');
    if (controls && controls.parentElement) {
      var keep = Array.from(controls.querySelectorAll('button,input,select,[role="combobox"]')).some(function (el) {
        return !el.closest('[data-testid="data-testid new share link-button"],[data-testid="data-testid Dashboard link container"]');
      });
      controls.parentElement.dataset.tciEmptyControls = keep ? 'false' : 'true';
    }
  }

  // Follow Grafana's header: gone in kiosk mode and on pages without one.
  function place() {
    header();
    var h = pd.querySelector("header"), show = !!(h && h.getBoundingClientRect().height > 0);
    fe.style.display = show ? "block" : "none";
    var narrow = parent.innerWidth <= 1250;
    var nav = pd.getElementById('tci-header-nav');
    fe.style.top = narrow ? '88px' : (nav ? nav.getBoundingClientRect().top : 8) + 'px';
    fe.style.left = narrow ? '16px' : 'auto';
    fe.style.right = narrow ? 'auto' : '160px';
    fe.style.transform = 'none';
    fe.style.width = Math.min(WIDTH, Math.max(160, parent.innerWidth - (narrow ? 32 : 850))) + "px";
    if (!list.classList.contains("open")) fe.style.height = "32px";
  }
  function grow() {
    var h = list.classList.contains("open") ? 36 + Math.min(MAXH, list.scrollHeight) + 2 : 32;
    fe.style.height = h + "px";
  }
  function close() { list.classList.remove("open"); sel = -1; grow(); }

  // One regex word per whitespace-separated token; every token must match.
  function re(s) { return s.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\\\$&"); }
  function str(s) { return s.replace(/\\\\/g, "\\\\\\\\").replace(/"/g, '\\\\"'); }
  function matcher(word) {
    var w = str(re(word)), parts = [
      'last_over_time(pytest_pr_info{title=~"(?i).*' + w + '.*"}[90d])',
      'last_over_time(ci_github_run_title_info{title=~"(?i).*' + w + '.*"}[90d])',
      'last_over_time(pytest_pr_info{author=~"(?i).*' + w + '.*"}[90d])'];
    var n = word.replace(/^#/, "");
    if (/^[0-9]+$/.test(n)) parts.push(
      'last_over_time(pytest_pr_info{pr=~"' + n + '.*"}[90d])',
      'last_over_time(ci_github_run_title_info{pr=~"' + n + '.*"}[90d])');
    return "group by (pr) (" + parts.join(" or ") + ")";
  }
  // Two steps: the matchers are cheap and give the PR numbers; the 90d subquery
  // for each PR's latest title is not, so it only runs over those PRs.
  function matches(q) {
    return q.split(/\\s+/).filter(Boolean).map(matcher).join(" and on (pr) ");
  }
  function details(prs) {
    var sel = 'pr=~"' + prs.join("|") + '"';
    return {
      hits: 'topk by (pr) (1, max by (pr, title) (max_over_time(timestamp(pytest_pr_info{' + sel + ',title!=""})[90d:1h]))' +
        ' or on (pr) max by (pr, title) (max_over_time(timestamp(ci_github_run_title_info{' + sel + ',title!=""})[90d:1h])))',
      author: 'max by (pr, author) (last_over_time(pytest_pr_info{' + sel + ',author!=""}[90d]))',
      state: "max by (pr) (last_over_time(pytest_pr_state{" + sel + "}[90d]))"
    };
  }
  function query(exprs) {
    var ds = {type: "prometheus", uid: "prometheus"};
    var body = {from: "now-5m", to: "now", queries: Object.keys(exprs).map(function (k) {
      return {refId: k, datasource: ds, expr: exprs[k], instant: true, range: false, maxDataPoints: 1, intervalMs: 60000};
    })};
    return fetch("/api/ds/query", {method: "POST", credentials: "same-origin",
      headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)})
      .then(function (r) { return r.json(); })
      .then(function (j) {
        var res = j.results || {};
        Object.keys(res).forEach(function (k) { if (res[k].error) throw new Error(res[k].error); });
        return res;
      });
  }

  function series(result) {
    var out = [];
    ((result && result.frames) || []).forEach(function (f) {
      var fields = f.schema.fields, vals = f.data.values;
      for (var i = 0; i < fields.length; i++) {
        if (fields[i].type === "number") {
          var v = vals[i] && vals[i].length ? vals[i][vals[i].length - 1] : null;
          out.push({labels: fields[i].labels || {}, value: v});
        }
      }
    });
    return out;
  }

  function ago(sec) {
    var d = Date.now() / 1000 - sec;
    if (d < 3600) return Math.max(1, Math.round(d / 60)) + "m ago";
    if (d < 86400) return Math.round(d / 3600) + "h ago";
    return Math.round(d / 86400) + "d ago";
  }
  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) { return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]; });
  }

  // Placeholder rows while a query is in flight and nothing has come back yet.
  function loading() {
    var widths = [72, 58, 66];
    list.innerHTML = widths.map(function (w) {
      return '<div class="sk"><i style="width:' + w + '%"></i><i></i></div>';
    }).join("");
    list.classList.add("open");
    grow();
  }

  function render(msg) {
    if (msg) {
      list.innerHTML = '<div class="msg">' + esc(msg) + "</div>";
    } else {
      list.innerHTML = rows.map(function (r, i) {
        var meta = [r.author, r.state && '<span class="st-' + r.state + '">' + r.state + "</span>", ago(r.seen)]
          .filter(Boolean).join(" · ");
        return '<a class="row' + (i === sel ? " sel" : "") + '" role="option" target="_top" href="' +
          PR_URL + "?var-pr=" + encodeURIComponent(r.pr) + '"><span class="num">#' + esc(r.pr) +
          '</span><span class="title" title="' + esc(r.title) + '">' + esc(r.title) + '</span>' +
          '<span class="meta">' + meta + "</span></a>";
      }).join("");
    }
    list.classList.add("open");
    grow();
  }

  function search() {
    var q = box.value.trim(), mine = ++seq;
    if (q.length < MIN_CHARS) { rows = []; close(); return; }
    query({m: matches(q)})
      .then(function (res) {
        if (mine !== seq) return null;
        // Newest PR numbers first, capped so the second query stays small.
        var prs = series(res.m).map(function (s) { return s.labels.pr; })
          .sort(function (a, b) { return b - a; }).slice(0, 200);
        if (!prs.length) return {};
        return query(details(prs));
      })
      .then(function (res) {
        if (!res || mine !== seq) return;
        var author = {}, state = {};
        series(res.author).forEach(function (s) { author[s.labels.pr] = s.labels.author; });
        series(res.state).forEach(function (s) { state[s.labels.pr] = ["closed", "open", "merged"][s.value]; });
        rows = series(res.hits).map(function (s) {
          return {pr: s.labels.pr, title: s.labels.title, seen: s.value, author: author[s.labels.pr], state: state[s.labels.pr]};
        }).sort(function (a, b) { return b.seen - a.seen; }).slice(0, 20);
        sel = rows.length ? 0 : -1;
        render(rows.length ? "" : "No PR in the last 90 days matches.");
      })
      .catch(function (e) { if (mine === seq) { rows = []; render("Search failed: " + e.message); } });
  }

  function go(r, newTab) {
    var url = PR_URL + "?var-pr=" + encodeURIComponent(r.pr);
    if (newTab) parent.open(url, "_blank"); else parent.location.assign(url);
  }

  box.addEventListener("input", function () {
    clearTimeout(timer);
    rows = [];
    if (box.value.trim().length < MIN_CHARS) { seq++; close(); return; }
    loading();
    timer = setTimeout(search, 200);
  });
  box.addEventListener("focus", function () { if (box.value.trim() && rows.length) render(); });
  box.addEventListener("keydown", function (e) {
    if (e.key === "Escape") { box.blur(); close(); return; }
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      if (!rows.length) return;
      e.preventDefault();
      sel = (sel + (e.key === "ArrowDown" ? 1 : rows.length - 1)) % rows.length;
      render();
      var el = list.children[sel]; if (el) el.scrollIntoView({block: "nearest"});
      return;
    }
    if (e.key === "Enter") {
      var n = box.value.trim().replace(/^#/, "");
      if (sel >= 0 && rows[sel]) go(rows[sel], e.metaKey || e.ctrlKey);
      else if (/^[0-9]+$/.test(n)) go({pr: n}, e.metaKey || e.ctrlKey);
    }
  });
  pd.addEventListener("mousedown", close);
  parent.addEventListener("resize", place);

  theme();
  place();
  setInterval(place, 1000);
})();
</script>
</body></html>
""".replace("__PR_URL__", PR_DASHBOARD_URL)
