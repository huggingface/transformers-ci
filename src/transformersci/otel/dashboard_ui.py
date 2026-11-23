"""Shared Grafana-compatible styling and navigation for standalone dashboard pages."""

from html import escape

# Shared with the navigation injected into Grafana by pr_search.py. Keep the
# hit area above Grafana's empty command-palette toolbar, which otherwise
# intercepts Profile clicks and hover even when our search box is elsewhere.
NAV_CSS = """
.tci-nav{position:relative;z-index:2;display:flex;flex:none;align-items:center;gap:8px;white-space:nowrap}
.tci-nav a{display:inline-flex;align-items:center;box-sizing:border-box;height:32px;flex:none;color:inherit;text-decoration:none;border:1px solid rgba(128,128,128,.4);border-radius:4px;padding:0 8px;font:600 12px/18px Inter,system-ui,sans-serif}
.tci-nav a:hover{background:rgba(128,128,128,.15);text-decoration:none}
.tci-nav a:focus-visible{outline:2px solid #6e9fff;outline-offset:-2px}
.tci-nav a[aria-current="page"]{border-color:#6e9fff}
.tci-nav .tci-lock{width:11px;height:11px;margin-left:5px;opacity:.75}
.tci-nav .tci-home{display:flex;align-items:center;gap:8px;font-size:14px;border:0;padding:0 4px;line-height:32px}
.tci-nav .tci-emoji{font-size:26px}
"""
CSS = (
    """
@font-face{font-family:Inter;src:url('/public/fonts/inter/Inter-Regular.woff2') format('woff2');font-style:normal;font-weight:400;font-display:swap}
@font-face{font-family:Inter;src:url('/public/fonts/inter/Inter-Medium.woff2') format('woff2');font-style:normal;font-weight:500 700;font-display:swap}
:root{color-scheme:dark;--bg:#111217;--panel:#181b1f;--fg:#ccccdc;--text:#ccccdc;--muted:rgba(204,204,220,.65);--line:rgba(204,204,220,.12);--accent:#6e9fff;--good:#73bf69;--bad:#f2495c;--hover:rgba(204,204,220,.08)}
:root[data-theme=light]{color-scheme:light;--bg:#f4f5f5;--panel:#fff;--fg:#24292e;--text:#24292e;--muted:rgba(36,41,46,.65);--line:rgba(36,41,46,.12);--accent:#1f60c4;--good:#26762c;--bad:#c4162a;--hover:rgba(36,41,46,.08)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 Inter,Helvetica,Arial,sans-serif}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.tci-site-header{background:var(--panel);border-bottom:1px solid var(--line);min-height:48px;padding:7px 28px;display:flex;align-items:center}
.tci-site-header .tci-nav{flex-wrap:wrap;white-space:normal}
main{width:100%;margin:0;padding:12px 24px 32px}h1{font-size:20px;font-weight:500;line-height:28px;margin:8px 0 2px}h2{font-size:16px;font-weight:600;margin:0}h3{font-size:14px;font-weight:500;margin:0}p{margin:4px 0}.muted,small{color:var(--muted)}
button,select,input{font:inherit;color:var(--fg);background:var(--bg);border:1px solid rgba(128,128,128,.4);border-radius:2px;padding:5px 8px;min-height:32px;max-width:100%}button{cursor:pointer;background:var(--panel)}button:hover{background:var(--hover)}:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;vertical-align:top;border-bottom:1px solid var(--line);padding:6px 8px}th{color:var(--muted);font-weight:600}tbody tr:hover{background:var(--hover)}.scroll{overflow-x:auto}
@media(max-width:600px){.tci-site-header{padding:8px 16px}main{padding:12px 16px 24px}}
"""
    + NAV_CSS
)

MENU = (
    ("Activity", "/"),
    ("Fix failures", "/d/pytest-test-health/fix-failures"),
    ("MIC failures", "/d/mic-failures/mic-failures"),
    ("Monitoring", "/d/pytest-observability-daily/monitoring"),
    (
        "CI Health",
        "/login/github?redirectTo=%2Fd%2Fpytest-observability-stack-health%2Fci-health",
    ),
    ("Benchmarks", "/benchmarks"),
    ("Profile", "/profile"),
)
LOCK = '<svg class="tci-lock" viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M4 7V5a4 4 0 1 1 8 0v2h.5A1.5 1.5 0 0 1 14 8.5v5A1.5 1.5 0 0 1 12.5 15h-9A1.5 1.5 0 0 1 2 13.5v-5A1.5 1.5 0 0 1 3.5 7H4zm2 0h4V5a2 2 0 1 0-4 0v2z"/></svg>'


def header(active, theme="dark"):
    links = [
        '<a class="tci-home" href="/" aria-label="Transformers CI home"><span class="tci-emoji" aria-hidden="true">🤗</span>Transformers CI</a>'
    ]
    for label, path in MENU:
        if label in ("Benchmarks", "Profile") and theme == "light":
            path += "?theme=light"
        current = ' aria-current="page"' if label == active else ""
        icon = LOCK if label == "CI Health" else ""
        links.append(f'<a href="{escape(path, quote=True)}"{current}>{label}{icon}</a>')
    return (
        '<header class="tci-site-header"><nav class="tci-nav" aria-label="CI dashboards">'
        + "".join(links)
        + "</nav></header>"
    )
