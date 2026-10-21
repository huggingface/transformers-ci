(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const params = new URLSearchParams(location.search);
  if (window.tci) tci.applyTheme();
  const state = {
    days: ['7', '14', '30'].includes(params.get('days')) ? params.get('days') : '7',
    scope: ['all', 'main', 'prs'].includes(params.get('scope')) ? params.get('scope') : 'all',
    variant: params.getAll('variant').filter(Boolean), platform: params.getAll('platform').filter(Boolean),
    failure_type: params.getAll('failure_type').filter(Boolean),
    sticky_only: params.get('sticky_only') !== 'false',
    sort: ['fails', 'alpha', 'alpha_desc'].includes(params.get('sort')) ? params.get('sort') : 'fails',
  };
  byId('sticky-only').checked = state.sticky_only;
  byId('search').value = params.get('q') || '';
  let data = {rows: [], options: {}}, request = 0, loading = false, errorText = '';
  const openGroups = new Set();
  let groupNames = [];
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
      button.setAttribute('aria-pressed', String(button.dataset.v === state[id] || (id === 'sort' && button.dataset.v === 'alpha' && state.sort === 'alpha_desc')));
    });
  }
  function options(id) {
    byId(id).replaceChildren();
    const values = [...new Set(['', ...(data.options?.[id] || []), ...state[id]])];
    values.forEach(value => {
      const label = element('label', '', byId(id));
      const input = element('input', '', label);
      input.type = 'checkbox'; input.value = value;
      input.checked = value ? state[id].includes(value) : state[id].length === 0;
      element('span', value || 'All', label);
    });
  }
  function queryParams() {
    const next = new URLSearchParams();
    Object.entries(state).forEach(([key, value]) => {
      if (Array.isArray(value)) value.forEach(item => next.append(key, item));
      else next.set(key, String(value));
    });
    return next;
  }
  function outcomeBar(parent, counts, label) {
    const total = counts.pass + counts.fail + counts.skipped;
    const percent = total ? 100 * counts.fail / total : null;
    const metric = element('span', '', parent); metric.className = 'model-metric';
    element('span', `${label}: ${percent == null ? '—' : `${percent.toFixed(1)}% failed`}`, metric);
    const bar = element('span', '', metric); bar.className = 'model-meter';
    const description = `${label}: ${counts.pass} passed, ${counts.fail} failed, ${counts.skipped} skipped`;
    metric.title = description; bar.setAttribute('role', 'img'); bar.setAttribute('aria-label', description);
    for (const kind of ['pass', 'fail', 'skipped']) {
      if (!counts[kind]) continue;
      const part = element('span', '', bar); part.className = kind; part.style.flex = String(counts[kind]);
    }
  }
  function foldControl() {
    const button = byId('fold'); button.hidden = !groupNames.length;
    const collapse = groupNames.some(name => openGroups.has(name));
    const label = collapse ? 'Collapse all groups' : 'Expand all groups';
    button.title = label; button.setAttribute('aria-label', label);
    const path = collapse ? 'M4.5 2.5L8 6l3.5-3.5M4.5 13.5L8 10l3.5 3.5' : 'M4.5 6L8 2.5 11.5 6M4.5 10L8 13.5 11.5 10';
    button.innerHTML = `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${path}"/></svg>`;
  }
  function render() {
    const summary = data.summary;
    byId('failure-rate').textContent = summary?.rate == null ? '—' : `${summary.rate.toFixed(1)}%`;
    byId('failure-counts').textContent = summary
      ? `${summary.failed.toLocaleString()} failed / ${summary.total.toLocaleString()} recorded checks · last ${data.days} days · ${data.scope === 'main' ? 'main' : data.scope === 'prs' ? 'PRs' : 'all branches'} · all variants and platforms`
      : 'Loading recorded MIC outcomes…';
    const filter = byId('search').value.toLowerCase();
    const rows = data.rows.filter(r => Object.values(r).join(' ').toLowerCase().includes(filter));
    const opened = new Set([...byId('rows').querySelectorAll('details[open]')].map(node => node.dataset.case));
    byId('rows').replaceChildren();
    byId('days-head').textContent = `Last ${state.days} days`;
    const groups = new Map();
    rows.forEach(r => {
      const key = r.source_model || r.fixture_repo || (r.target === 'runner' ? 'MIC runner' : r.contract);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(r);
    });
    const byName = (a, b) => a[0].localeCompare(b[0], undefined, {sensitivity: 'base'});
    const failingRuns = members => members.reduce((sum, r) => sum + (r.fails || 0), 0);
    const ordered = [...groups].sort(state.sort === 'alpha' ? byName : state.sort === 'alpha_desc' ? (a, b) => -byName(a, b) : (a, b) => failingRuns(b[1]) - failingRuns(a[1]) || byName(a, b));
    groupNames = ordered.map(([name]) => name); foldControl(); press('sort');
    byId('sort').querySelector('[data-v="alpha"]').textContent = state.sort === 'alpha_desc' ? 'Z–A' : 'A–Z';
    ordered.forEach(([name, members]) => {
      const open = openGroups.has(name);
      const summaryRow = element('tr', '', byId('rows')); summaryRow.className = 'model-summary';
      const summaryCell = element('td', '', summaryRow); summaryCell.colSpan = 8;
      const line = element('div', '', summaryCell); line.className = 'model-summary-line';
      const toggle = element('button', open ? '▾' : '▸', line); toggle.type = 'button'; toggle.className = 'model-toggle';
      toggle.setAttribute('aria-expanded', String(open)); toggle.setAttribute('aria-label', `${open ? 'Collapse' : 'Expand'} ${name} errors`);
      toggle.addEventListener('click', () => { if (open) openGroups.delete(name); else openGroups.add(name); render(); });
      const r = members[0];
      const model = link(line, name, r.target === 'runner' ? '' : r.fixture_repo_url); model.className = 'model-name'; model.title = name;
      element('span', `${members.length} case${members.length === 1 ? '' : 's'}`, line).className = 'sub';
      const latest = {pass: 0, fail: 0, skipped: 0}, history = {pass: 0, fail: 0, skipped: 0};
      members.forEach(row => {
        const status = row.latest?.status;
        if (status) latest[status === 'pass' ? 'pass' : ['fail', 'error'].includes(status) ? 'fail' : 'skipped'] += 1;
        (row.daily || []).forEach(day => { for (const key of ['pass', 'fail', 'skipped']) history[key] += day[key]; });
      });
      const stateLabel = element('span', latest.fail ? 'Failing' : latest.pass ? 'Passing' : 'No passing check', line);
      stateLabel.className = latest.fail ? 'latest-fail' : latest.pass ? 'latest-pass' : 'sub';
      const meters = element('span', '', line); meters.className = 'model-meters';
      outcomeBar(meters, latest, 'Latest'); outcomeBar(meters, history, `${state.days} days`);
      if (!open) return;
      members.forEach(r => {
        const tr = element('tr', '', byId('rows')); tr.className = 'case-row'; tr.hidden = !open;
        let td = element('td', r.weights || 'unknown', tr);
        if (!r.weights || r.weights === 'unknown') td.title = 'Weight strategy is unavailable for this historical run.';
        td = element('td', r.variant || '—', tr); element('small', r.platform || '—', td);
        td = element('td', '', tr); td.className = 'pr';
        link(td, r.pr_title || (r.pr_number ? `PR #${r.pr_number}` : r.pr || '—'), r.pr_url);
        if (r.pr_number && r.pr_title) element('small', `#${r.pr_number}`, td);
        td = element('td', '', tr);
        element('span', String(r.fails || 1), td).className = 'num';
        const strip = element('span', '', td); strip.className = 'strip';
        (r.daily || []).forEach((counts, index) => {
          const day = new Date((data.first_day + index * 86400) * 1000).toLocaleDateString('en', {month: 'short', day: 'numeric', timeZone: 'UTC'});
          const cell = element('i', '', strip);
          const total = counts.pass + counts.fail + counts.skipped;
          for (const [kind, className] of [['fail', 'f'], ['pass', 'p'], ['skipped', 's']]) {
            if (!counts[kind]) continue;
            const part = element('span', '', cell); part.className = className;
            part.style.flex = String(counts[kind]);
          }
          cell.title = `${day} UTC: ${total ? `${counts.pass} passed, ${counts.fail} failed, ${counts.skipped} skipped` : 'no observed checks'}`;
        });
        td = element('td', '', tr); td.className = 'latest';
        const latest = r.latest;
        if (latest) {
          const names = {pass: 'Passed', fail: 'Failed', error: 'Error', xfail: 'Expected failure', unsupported: 'Unsupported'};
          const check = link(td, names[latest.status] || 'Unknown', latest.run_url);
          check.className = `latest-${latest.status}`;
          if (latest.started_at) {
            element('small', ago(latest.started_at), td).title = new Date(latest.started_at * 1000).toLocaleString('en-US');
          }
        } else element('span', 'Unknown', td);
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
        if (r.contract_url) {
          const contract = link(td, 'Contract', r.contract_url);
          contract.title = r.contract_revision ? 'Usage pinned to the failed contract revision' : 'Current usage; historical contract revision unavailable';
        }
      });
    });
    byId('state').className = errorText ? 'error' : '';
    byId('state').textContent = errorText || (data.rows.length
      ? `${groups.size} models · ${rows.length} cases shown · ${data.total} failure observations in ${data.days} day(s).` + (data.models > data.limit ? ` Showing the latest ${data.limit} models; narrow the filters or window for earlier results.` : '')
      : 'No MIC failures were reported with these filters.');
    const next = queryParams();
    if (byId('search').value) next.set('q', byId('search').value);
    history.replaceState(null, '', `${location.pathname}?${next}`);
  }
  async function load(quiet = false) {
    const id = ++request;
    loading = true;
    if (!quiet) byId('progress').hidden = false;
    try {
      const response = await fetch(`/mic/data?${queryParams()}`);
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
  ['scope', 'days'].forEach(id => {
    byId(id).addEventListener('click', event => {
      const button = event.target.closest('button'); if (!button) return;
      state[id] = button.dataset.v; press(id); load();
    });
  });
  ['variant', 'platform', 'failure_type'].forEach(id => {
    byId(id).addEventListener('change', event => {
      state[id] = event.target.value === '' ? [] : [...byId(id).querySelectorAll('input:checked')].map(input => input.value).filter(Boolean);
      load();
    });
  });
  byId('sticky-only').addEventListener('change', event => { state.sticky_only = event.target.checked; load(); });
  byId('sort').addEventListener('click', event => {
    const button = event.target.closest('button'); if (!button) return;
    state.sort = button.dataset.v === 'alpha' ? (state.sort === 'alpha' ? 'alpha_desc' : 'alpha') : 'fails';
    render();
  });
  byId('fold').addEventListener('click', () => {
    if (groupNames.some(name => openGroups.has(name))) openGroups.clear();
    else groupNames.forEach(name => openGroups.add(name));
    render();
  });
  byId('search').addEventListener('input', render);
  document.addEventListener('visibilitychange', () => { if (!document.hidden && !loading) load(true); });
  press('scope'); press('days'); options('variant'); options('platform'); options('failure_type'); load();
  setInterval(() => { if (!document.hidden && !loading) load(true); }, 30000);
})();
