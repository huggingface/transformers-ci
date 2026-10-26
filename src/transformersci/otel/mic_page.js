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
    sort: ['fails', 'recent', 'alpha', 'alpha_desc'].includes(params.get('sort')) ? params.get('sort') : 'fails',
    group: params.get('group') === 'pr' ? 'pr' : 'model',
  };
  byId('sticky-only').checked = state.sticky_only;
  byId('search').value = params.get('q') || '';
  let data = {rows: [], options: {}}, request = 0, loading = false, errorText = '';
  const openGroups = {model: new Set(), pr: new Set()};
  const expanded = () => openGroups[state.group];
  let groupNames = [], rootGroupNames = [];
  const quantity = (count, singular, plural = `${singular}s`) => `${count} ${count === 1 ? singular : plural}`;
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
  function groupInfo(row, by = state.group) {
    if (by === 'pr') {
      const identity = row.pr_number || row.pr || 'Unknown branch';
      const name = row.pr_number ? `#${row.pr_number}${row.pr_title ? ` · ${row.pr_title}` : ''}` : identity;
      return {key: `${row.repository || 'huggingface/transformers'}:${identity}`, name, sortName: row.pr_title || name, href: row.pr_url};
    }
    const name = row.source_model || row.fixture_repo || (row.target === 'runner' ? 'MIC runner' : row.contract) || 'Unknown model';
    return {key: name, name, sortName: name, href: row.target === 'runner' ? '' : row.fixture_repo_url};
  }
  function groupRows(rows, by) {
    const groups = new Map();
    rows.forEach(row => {
      const info = groupInfo(row, by);
      if (!groups.has(info.key)) groups.set(info.key, {...info, members: []});
      groups.get(info.key).members.push(row);
    });
    return groups;
  }
  function sortGroups(groups) {
    const byName = (a, b) => a.sortName.localeCompare(b.sortName, undefined, {sensitivity: 'base'}) || a.key.localeCompare(b.key);
    const failures = group => group.members.reduce((sum, row) => sum + (row.fails || 0), 0);
    const recent = group => group.members.reduce((stamp, row) => Math.max(stamp, row.latest?.started_at || row.started_at || 0), 0);
    return [...groups].sort(state.sort === 'alpha' ? byName : state.sort === 'alpha_desc' ? (a, b) => -byName(a, b) : state.sort === 'recent' ? (a, b) => recent(b) - recent(a) || byName(a, b) : (a, b) => failures(b) - failures(a) || byName(a, b));
  }
  function groupSummary({name, href, members}, key, level, label) {
    const open = expanded().has(key);
    const row = element('tr', '', byId('rows')); row.className = `model-summary group-level-${level}`;
    const cell = element('td', '', row); cell.colSpan = 7;
    const line = element('div', '', cell); line.className = 'model-summary-line';
    const toggle = element('button', open ? '▾' : '▸', line); toggle.type = 'button'; toggle.className = 'model-toggle';
    toggle.setAttribute('aria-expanded', String(open)); toggle.setAttribute('aria-label', `${open ? 'Collapse' : 'Expand'} ${name} errors`);
    toggle.addEventListener('click', () => { if (open) expanded().delete(key); else expanded().add(key); render(); });
    const nameLink = link(line, name, href); nameLink.className = 'model-name'; nameLink.title = name;
    element('span', label, line).className = 'sub';
    const latest = {pass: 0, fail: 0, skipped: 0}, history = {pass: 0, fail: 0, skipped: 0};
    members.forEach(member => {
      const status = member.latest?.status;
      if (status) latest[status === 'pass' ? 'pass' : ['fail', 'error'].includes(status) ? 'fail' : 'skipped'] += 1;
      (member.daily || []).forEach(day => { for (const kind of ['pass', 'fail', 'skipped']) history[kind] += day[kind]; });
    });
    const status = element('span', latest.fail ? 'Failing' : latest.pass ? 'Passing' : 'No passing check', line);
    status.className = latest.fail ? 'latest-fail' : latest.pass ? 'latest-pass' : 'sub';
    const meters = element('span', '', line); meters.className = 'model-meters';
    outcomeBar(meters, latest, 'Latest'); outcomeBar(meters, history, `${state.days} days`);
    return open;
  }
  function foldControl() {
    const button = byId('fold'); button.hidden = !groupNames.length;
    const collapse = rootGroupNames.some(name => expanded().has(name));
    const label = collapse ? 'Collapse all groups' : 'Expand all groups';
    button.title = label; button.setAttribute('aria-label', label);
    const path = collapse ? 'M4.5 2.5L8 6l3.5-3.5M4.5 13.5L8 10l3.5 3.5' : 'M4.5 6L8 2.5 11.5 6M4.5 10L8 13.5 11.5 10';
    button.innerHTML = `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${path}"/></svg>`;
  }
  function render() {
    const summary = data.summary;
    const current = summary?.now;
    byId('failing-now').textContent = current?.rate == null ? '—' : `${current.rate.toFixed(1)}%`;
    byId('failing-now-counts').textContent = current
      ? `${current.failing.toLocaleString()} of ${current.cases.toLocaleString()} cases fail their latest check`
      : '';
    byId('failure-rate-label').textContent = data.sticky_only ? 'Sticky failure rate' : 'Global failure rate';
    byId('failure-rate').textContent = summary?.rate == null ? '—' : `${summary.rate.toFixed(1)}%`;
    byId('failure-counts').textContent = summary
      ? `${summary.failed.toLocaleString()} failed / ${summary.total.toLocaleString()} recorded checks · last ${data.days} days · ${data.scope === 'main' ? 'main' : data.scope === 'prs' ? 'PRs' : 'all branches'} · ${data.sticky_only ? 'matching sticky cases' : 'all variants and platforms'}`
      : 'Loading recorded MIC outcomes…';
    const filter = byId('search').value.toLowerCase();
    const rows = data.rows.filter(r => Object.values(r).join(' ').toLowerCase().includes(filter));
    const opened = new Set([...byId('rows').querySelectorAll('details[open]')].map(node => node.dataset.case));
    byId('rows').replaceChildren();
    byId('days-head').textContent = `Last ${state.days} days`;
    // On main every model has one branch: list its cases under the model.
    const flat = state.group === 'model' && data.scope === 'main';
    byId('identity-head').textContent = state.group === 'pr' ? 'PR / model / weights' : flat ? 'Model / weights' : 'Model / PR / weights';
    press('group');
    const groups = groupRows(rows, state.group), ordered = sortGroups(groups.values());
    const secondary = state.group === 'pr' ? 'model' : 'pr';
    ordered.forEach(primary => {
      primary.id = JSON.stringify([primary.key]);
      primary.children = flat ? [] : sortGroups(groupRows(primary.members, secondary).values());
      primary.children.forEach(child => { child.id = JSON.stringify([primary.key, child.key]); });
    });
    rootGroupNames = ordered.map(primary => primary.id);
    groupNames = ordered.flatMap(primary => [primary.id, ...primary.children.map(child => child.id)]);
    foldControl(); press('sort');
    byId('sort').querySelector('[data-v="alpha"]').textContent = state.sort === 'alpha_desc' ? 'Z–A' : 'A–Z';
    byId('sort').querySelector('[data-v="alpha"]').title = `Alphabetical by ${state.group === 'pr' ? 'PR title' : 'model name'}; click again to reverse`;
    ordered.forEach(primary => {
      const cases = quantity(primary.members.length, 'case');
      const label = flat ? cases : `${quantity(primary.children.length, secondary === 'pr' ? 'PR / branch' : 'model', secondary === 'pr' ? 'PRs / branches' : 'models')} · ${cases}`;
      if (!groupSummary(primary, primary.id, 0, label)) return;
      if (flat) { primary.members.forEach(caseRow); return; }
      primary.children.forEach(child => {
        if (!groupSummary(child, child.id, 1, quantity(child.members.length, 'case'))) return;
        child.members.forEach(caseRow);
      });
    });
    function caseRow(r) {
      const tr = element('tr', '', byId('rows')); tr.className = 'case-row';
      let td = element('td', r.weights || 'unknown', tr);
      if (!r.weights || r.weights === 'unknown') td.title = 'Weight strategy is unavailable for this historical run.';
      td = element('td', r.variant || '—', tr); element('small', r.platform || '—', td);
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
    }
    byId('state').className = errorText ? 'error' : '';
    byId('state').textContent = errorText || (data.rows.length
      ? `${groups.size} ${state.group === 'pr' ? 'PRs / branches' : 'models'} · ${rows.length} cases shown · ${data.total} failure observations in ${data.days} day(s).` + ((data.groups ?? data.models) > data.limit ? ` Showing the latest ${data.limit} groups; narrow the filters or window for earlier results.` : '')
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
  ['scope', 'days', 'group'].forEach(id => {
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
    state.sort = button.dataset.v === 'alpha' ? (state.sort === 'alpha' ? 'alpha_desc' : 'alpha') : button.dataset.v;
    render();
  });
  byId('fold').addEventListener('click', () => {
    if (rootGroupNames.some(name => expanded().has(name))) expanded().clear();
    else groupNames.forEach(name => expanded().add(name));
    render();
  });
  byId('search').addEventListener('input', render);
  document.addEventListener('visibilitychange', () => { if (!document.hidden && !loading) load(true); });
  press('scope'); press('days'); options('variant'); options('platform'); options('failure_type'); load();
  setInterval(() => { if (!document.hidden && !loading) load(true); }, 30000);
})();
