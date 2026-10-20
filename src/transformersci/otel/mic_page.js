(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const params = new URLSearchParams(location.search);
  if (window.tci) tci.applyTheme();
  const state = {
    days: ['7', '14', '30'].includes(params.get('days')) ? params.get('days') : '7',
    scope: ['all', 'main', 'prs'].includes(params.get('scope')) ? params.get('scope') : 'all',
    variant: params.get('variant') || '', platform: params.get('platform') || '',
    failure_type: params.get('failure_type') || '',
  };
  byId('search').value = params.get('q') || '';
  let data = {rows: [], options: {}}, request = 0, loading = false, errorText = '';
  function element(tag, text, parent) {
    const node = document.createElement(tag);
    if (text) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  }
  function link(parent, text, href) {
    if (!href) return element('span', text, parent);
    const a = element('a', text, parent);
    a.href = href;
    a.target = href.startsWith('/') ? '_top' : '_blank';
    a.rel = 'noopener';
    return a;
  }
  function ago(ts) {
    const seconds = Math.max(0, Date.now() / 1000 - ts);
    for (const [suffix, size] of [['d', 86400], ['h', 3600], ['m', 60]]) {
      if (seconds >= size) return `${Math.floor(seconds / size)}${suffix} ago`;
    }
    return 'just now';
  }
  function press(id) {
    byId(id).querySelectorAll('button').forEach(button => {
      button.setAttribute('aria-pressed', String(button.dataset.v === state[id]));
    });
  }
  function options(id) {
    byId(id).replaceChildren();
    const values = [...new Set(['', ...(data.options?.[id] || []), ...(state[id] ? [state[id]] : [])])];
    values.forEach(value => {
      const button = element('button', value || 'All', byId(id));
      button.type = 'button'; button.dataset.v = value;
    });
    press(id);
  }
  function render() {
    const filter = byId('search').value.toLowerCase();
    const rows = data.rows.filter(r => Object.values(r).join(' ').toLowerCase().includes(filter));
    const opened = new Set([...byId('rows').querySelectorAll('details[open]')].map(node => node.dataset.case));
    byId('rows').replaceChildren();
    byId('days-head').textContent = `Last ${state.days} days`;
    rows.forEach(r => {
      const tr = element('tr', '', byId('rows'));
      let td = element('td', '', tr); td.className = 'model';
      if (r.target === 'runner') element('span', 'MIC runner', td);
      else {
        link(td, r.source_model || 'Source model unavailable', r.source_model_url);
        const slim = element('small', '', td);
        link(slim, `${r.fixture_repo?.split('/').pop() || 'Slim model'} (slim)`, r.fixture_repo_url);
      }
      td = element('td', r.weights || 'unknown', tr);
      if (!r.weights || r.weights === 'unknown') td.title = 'Weight strategy is unavailable for this historical run.';
      td = element('td', r.variant || '—', tr); element('small', r.platform || '—', td);
      td = element('td', '', tr); td.className = 'pr';
      link(td, r.pr_title || (r.pr_number ? `PR #${r.pr_number}` : r.pr || '—'), r.pr_url);
      if (r.pr_number && r.pr_title) element('small', `#${r.pr_number}`, td);
      td = element('td', '', tr);
      element('span', String(r.fails || 1), td).className = 'num';
      const strip = element('span', '', td); strip.className = 'strip';
      (r.days || []).forEach((count, index) => {
        const day = new Date((data.first_day + index * 86400) * 1000).toLocaleDateString('en', {month: 'short', day: 'numeric', timeZone: 'UTC'});
        const cell = element('i', '', strip); if (count) cell.className = 'f';
        cell.title = `${day}: ${count} failing run${count === 1 ? '' : 's'}`;
      });
      td = element('td', '', tr); td.className = 'date';
      if (r.started_at) {
        const stamp = new Date(r.started_at * 1000);
        element('span', stamp.toLocaleString('en-US'), td).title = stamp.toISOString();
        element('small', `(${ago(r.started_at)})`, td);
      } else td.textContent = '—';
      td = element('td', '', tr); td.className = 'failure';
      const kind = element('span', r.failure_kind || 'Baseline assertion', td);
      kind.className = 'badge' + (r.failure_kind === 'Runtime traceback' ? ' runtime' : r.failure_kind === 'Setup error' ? ' setup' : '');
      const detail = element('details', '', td);
      detail.dataset.case = `${r.run_id}:${r.nodeid}`;
      detail.open = opened.has(detail.dataset.case);
      element('summary', r.failure.split('\n')[0] || 'Failure details unavailable', detail);
      element('pre', r.failure, detail);
      if (r.selected_by) element('small', `Selected by: ${r.selected_by}`, detail);
      if (r.known_failure) element('small', `Known failure: ${r.known_failure}`, detail);
      td = element('td', '', tr); td.className = 'links';
      link(td, 'Trace', r.trace_url); link(td, 'Run', r.run_url);
      if (r.baseline_url) link(td, 'Baseline', r.baseline_url);
    });
    byId('state').className = errorText ? 'error' : '';
    byId('state').textContent = errorText || (data.rows.length
      ? `${rows.length} cases shown · ${data.total} failure observations in ${data.days} day(s).` + (data.cases > data.limit ? ` Showing the latest ${data.limit}; narrow the filters or window for earlier results.` : '')
      : 'No MIC failures were reported with these filters.');
    const next = new URLSearchParams(state);
    if (byId('search').value) next.set('q', byId('search').value);
    history.replaceState(null, '', `${location.pathname}?${next}`);
  }
  async function load(quiet = false) {
    const id = ++request;
    loading = true;
    if (!quiet) byId('progress').hidden = false;
    try {
      const response = await fetch(`/mic/data?${new URLSearchParams(state)}`);
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'MIC data is unavailable.');
      if (id !== request) return;
      data = result; errorText = ''; options('variant'); options('platform'); options('failure_type'); render();
    } catch (error) {
      if (id !== request) return;
      errorText = `Could not refresh MIC failures: ${error.message}`;
      render();
    } finally {
      if (id === request) { loading = false; byId('progress').hidden = true; }
    }
  }
  ['scope', 'variant', 'platform', 'failure_type', 'days'].forEach(id => {
    byId(id).addEventListener('click', event => {
      const button = event.target.closest('button'); if (!button) return;
      state[id] = button.dataset.v; press(id); load();
    });
  });
  byId('search').addEventListener('input', render);
  document.addEventListener('visibilitychange', () => { if (!document.hidden && !loading) load(true); });
  press('scope'); press('days'); options('variant'); options('platform'); options('failure_type'); load();
  setInterval(() => { if (!document.hidden && !loading) load(true); }, 30000);
})();
