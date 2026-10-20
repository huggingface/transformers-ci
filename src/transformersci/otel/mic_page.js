(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const params = new URLSearchParams(location.search);
  if (params.get('theme') === 'light') document.body.classList.add('light');
  if (['1', '7', '14', '30'].includes(params.get('days'))) byId('days').value = params.get('days');
  byId('search').value = params.get('q') || '';
  let data = {rows: []}, request = 0;
  function element(tag, text, parent) {
    const node = document.createElement(tag);
    if (text) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  }
  function link(parent, text, href) {
    if (!href) return;
    const a = element('a', text, parent);
    a.href = href; a.target = '_blank'; a.rel = 'noopener';
  }
  function render() {
    const filter = byId('search').value.toLowerCase();
    const rows = data.rows.filter(r => Object.values(r).join(' ').toLowerCase().includes(filter));
    byId('rows').replaceChildren();
    rows.forEach(r => {
      const tr = element('tr', '', byId('rows'));
      let td = element('td', '', tr);
      element('strong', r.contract, td); element('small', r.target, td);
      td = element('td', r.variant || '—', tr); element('small', r.platform || '—', td);
      td = element('td', /^\d+$/.test(r.pr) ? `PR #${r.pr}` : r.pr || '—', tr);
      element('small', r.run_id, td);
      if (r.started_at) element('small', new Date(r.started_at * 1000).toLocaleString(), td);
      if (r.commit_sha) element('small', r.commit_sha.slice(0, 12), td);
      td = element('td', '', tr);
      const detail = element('details', '', td);
      element('summary', r.failure.split('\n')[0] || r.status, detail);
      element('pre', r.failure || 'Open the trace for failure details.', detail);
      if (r.selected_by) element('small', `Selected by: ${r.selected_by}`, detail);
      if (r.known_failure) element('small', `Known failure: ${r.known_failure}`, detail);
      td = element('td', '', tr); td.className = 'links';
      link(td, 'Trace', r.trace_url); link(td, 'CI run', r.run_url); link(td, 'Baseline', r.baseline_url);
    });
    byId('state').className = '';
    byId('state').textContent = data.rows.length
      ? `${rows.length} shown · ${data.total} failure observations in ${data.days} day(s).` + (data.total > data.limit ? ` Showing the latest ${data.limit}; narrow the window for earlier results.` : '')
      : 'No MIC failures were reported in this window.';
    const next = new URLSearchParams({days: byId('days').value});
    if (byId('search').value) next.set('q', byId('search').value);
    history.replaceState(null, '', `${location.pathname}?${next}`);
  }
  async function load() {
    const id = ++request;
    byId('state').textContent = 'Loading MIC failures…';
    byId('refresh').disabled = true;
    try {
      const response = await fetch(`/mic/data?days=${byId('days').value}`);
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'MIC data is unavailable.');
      if (id !== request) return;
      data = result; render();
    } catch (error) {
      if (id !== request) return;
      byId('rows').replaceChildren();
      byId('state').className = 'error';
      byId('state').textContent = `Could not load MIC failures: ${error.message}`;
    } finally { if (id === request) byId('refresh').disabled = false; }
  }
  byId('days').addEventListener('change', load);
  byId('search').addEventListener('input', render);
  byId('refresh').addEventListener('click', load);
  load();
})();
