/* The overview PR list queries Grafana's Prometheus datasource without navigating Grafana. */
(() => {
  'use strict';
  const mineButton = document.getElementById('mine');
  const openButton = document.getElementById('open');
  const dayButtons = [...document.querySelectorAll('[data-days]')];
  const status = document.getElementById('status');
  const rows = document.getElementById('rows');
  const empty = document.getElementById('empty');
  const {stored, save} = tci;
  const storage = (key) => stored(key) === 'true';
  let mine = storage('tci-my-stuff');
  let hideClosed = storage('tci-hide-closed-prs');
  const storedDays = Number(stored('tci-pr-days'));
  let days = [7, 14, 21].includes(storedDays) ? storedDays : 7;
  let minePrs = null;
  let mineUntil = 0;
  let queryConfig = null;
  const categories = new Map();
  let sequence = 0;
  let active = null;
  let signInPending = false;

  tci.applyTheme();

  function switches() {
    mineButton.setAttribute('aria-checked', String(mine));
    openButton.setAttribute('aria-checked', String(hideClosed));
    dayButtons.forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.days) === days)));
  }
  function message(value, error = false) {
    status.textContent = value;
    status.classList.toggle('error', error);
  }
  function fail(value) {
    rows.replaceChildren();
    empty.hidden = false;
    empty.textContent = value;
    message(value, true);
  }
  function signInPrompt() {
    message('Sign in to use My Stuff: ');
    const link = document.createElement('a');
    link.href = '/login/github?redirectTo=' + encodeURIComponent('/pr-table/auth-complete');
    link.target = '_blank';
    link.rel = 'noopener';
    link.textContent = 'Sign in with GitHub';
    status.append(link);
  }
  function signIn() {
    mine = false;
    save('tci-my-stuff', false);
    signInPending = true;
    switches();
    signInPrompt();
    refresh();
  }
  async function json(response) {
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
    return data;
  }
  async function candidates(signal) {
    if (!mine) return null;
    if (minePrs && Date.now() < mineUntil) return minePrs;
    const response = await fetch('/pr-search/mine', {credentials: 'same-origin', signal});
    if (response.status === 401) throw new Error('SignInRequired');
    const data = await json(response);
    if (!Array.isArray(data.prs)) throw new Error('Could not load your PR activity');
    minePrs = data.prs.filter(pr => typeof pr === 'string' && /^[1-9][0-9]*$/.test(pr));
    mineUntil = Date.now() + 300000;
    return minePrs;
  }
  async function query(expressions, signal) {
    const datasource = {type: 'prometheus', uid: 'prometheus'};
    const queries = Object.entries(expressions).map(([refId, expr]) => ({
      refId, expr, datasource, format: 'table', instant: true, range: false,
      maxDataPoints: 1, intervalMs: 60000
    }));
    const response = await fetch('/api/ds/query', {method: 'POST', credentials: 'same-origin', signal,
      headers: {'Content-Type': 'application/json'}, body: JSON.stringify({from: 'now-5m', to: 'now', queries})});
    const data = await json(response);
    for (const result of Object.values(data.results || {})) {
      if (result.error || result.status !== 200) throw new Error(result.error || 'Grafana query failed');
    }
    return data.results || {};
  }
  function series(result) {
    const values = [];
    for (const frame of result?.frames || []) {
      const fields = frame.schema?.fields || [];
      for (let i = 0; i < fields.length; i++) {
        if (fields[i].type !== 'number') continue;
        const column = frame.data?.values?.[i] || [];
        if (column.length) values.push({labels: fields[i].labels || {}, value: column[column.length - 1]});
      }
    }
    return values;
  }
  function topExpression(prs) {
    const original = queryConfig.top;
    const prefix = 'query_result(topk(20, ';
    if (!original.startsWith(prefix) || !original.endsWith('))')) throw new Error('Invalid PR query configuration');
    const base = original.slice(prefix.length, -2);
    const scope = prs ? (prs.length ? prs.join('|') : '0') : '[0-9]+';
    const activity = base.replaceAll('${pr_scope:raw}', scope).replaceAll('[7d]', `[${days}d]`);
    const visible = hideClosed
      ? `(${activity}) unless on (pr) (max by (pr) (last_over_time(pytest_pr_state[${days}d])) != 1)`
      : activity;
    return `topk(20, (${visible}) > (time() - ${days * 86400}))`;
  }
  function targets(prs) {
    const scope = prs.join('|');
    return Object.fromEntries(Object.entries(queryConfig.targets).map(([key, expr]) =>
      [key, expr.replaceAll('$top_prs', scope).replace(/\[7d(:?)\]/g, `[${days}d$1]`)]));
  }
  function text(value) { return value == null ? '' : String(value); }
  function cell(value, className = '') {
    const element = document.createElement('td');
    element.textContent = text(value);
    if (className) element.className = className;
    return element;
  }
  function relative(seconds) {
    if (!Number.isFinite(seconds)) return '';
    const minutes = Math.max(0, Math.round((Date.now() / 1000 - seconds) / 60));
    if (minutes < 60) return `${minutes}m ago`;
    if (minutes < 1440) return `${Math.round(minutes / 60)}h ago`;
    return `${Math.round(minutes / 1440)}d ago`;
  }
  function age(days) {
    if (!Number.isFinite(days)) return '';
    if (days < 1 / 24) return 'Now';
    if (days < 1) return `${Math.max(1, Math.floor(days * 24))}h`;
    return `${Math.floor(days)}d`;
  }
  function rate(value) {
    if (!Number.isFinite(value)) return '';
    if (value === 0) return '0%';
    if (value < 1) return '<1%';
    return `${Math.round(value)}%`;
  }
  function rateStyle(value) {
    if (!Number.isFinite(value)) return '';
    if (value >= 10) return 'rate-high';
    if (value >= 1) return 'rate-medium';
    return value > 0 ? 'rate-low' : 'rate-zero';
  }
  function categoryStyle(value) {
    return {'New model': 'new-model', Bugfix: 'bugfix', Feature: 'feature', 'CI/infra': 'ci',
      Docs: 'docs', Performance: 'performance', Refactor: 'refactor', Tests: 'tests', Other: 'other'}[value] || '';
  }
  function paintCategory(cell, value) {
    cell.className = value ? `category-cell ${categoryStyle(value)}` : '';
    if (value === 'New model') {
      const stamp = document.createElement('span');
      stamp.className = 'model-badge';
      stamp.textContent = 'NEW MODEL!';
      cell.replaceChildren(stamp);
    } else cell.textContent = text(value);
  }
  function currentCategory(pr, cell) {
    const cached = categories.get(pr);
    if (cached && cached.until > Date.now()) { if (cached.value) paintCategory(cell, cached.value); return; }
    fetch('/pr-search/labels?' + new URLSearchParams({pr, repository: 'huggingface/transformers'}),
      {credentials: 'same-origin'})
      .then(json)
      .then(data => {
        const value = data.category || '';
        categories.set(pr, {value, until: Date.now() + 60000});
        if (value && cell.isConnected) paintCategory(cell, value);
      })
      .catch(() => categories.set(pr, {value: '', until: Date.now() + 15000}));
  }
  function render(result, ordered) {
    const byPr = new Map(ordered.map(pr => [pr, {pr}]));
    for (const [name, response] of Object.entries(result)) {
      for (const item of series(response)) {
        const row = byPr.get(item.labels.pr);
        if (!row) continue;
        row[name] = name === 'Title' ? item.labels.title : name === 'Category' ? item.labels.category : item.value;
      }
    }
    const fragment = document.createDocumentFragment();
    for (const pr of ordered) {
      const row = byPr.get(pr);
      const tr = document.createElement('tr');
      const live = cell('');
      if (row.Active === 1) { const spinner = document.createElement('span'); spinner.className = 'live'; spinner.title = 'CI is running'; live.append(spinner); }
      tr.append(live);
      const number = cell('');
      const link = document.createElement('a');
      link.href = `/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=${encodeURIComponent(pr)}`;
      link.target = '_top';
      link.textContent = pr;
      number.append(link);
      const category = cell('');
      paintCategory(category, row.Category);
      tr.append(number, cell(row.Title), category, cell(age(row.Age)),
        cell(relative(row.Activity / 1000)), cell(row.Runs), cell(row.Failures),
        cell(rate(row.Rate), rateStyle(row.Rate)));
      const state = row.State === 1 ? 'open' : row.State === 2 ? 'merged' : row.State === 0 ? 'closed' : '';
      tr.append(cell(state, state));
      fragment.append(tr);
      currentCategory(pr, category);
    }
    rows.replaceChildren(fragment);
    empty.hidden = ordered.length > 0;
    empty.textContent = mine ? 'No PRs with recent CI activity in your PR activity.' : 'No recent PR activity.';
  }
  async function refresh() {
    if (active) active.abort();
    const controller = new AbortController();
    active = controller;
    const current = ++sequence;
    message('Loading…');
    try {
      queryConfig ||= await fetch('/pr-table/queries', {signal: controller.signal}).then(json);
      const prs = await candidates(controller.signal);
      const top = await query({Top: topExpression(prs)}, controller.signal);
      const ordered = series(top.Top).sort((a, b) => b.value - a.value).map(item => item.labels.pr)
        .filter(pr => /^[1-9][0-9]*$/.test(pr)).slice(0, 20);
      if (!ordered.length) { render({}, []); if (signInPending) signInPrompt(); else message(''); return; }
      const details = await query(targets(ordered), controller.signal);
      if (current !== sequence) return;
      render(details, ordered);
      if (signInPending) signInPrompt();
      else message('');
    } catch (error) {
      if (error.name === 'AbortError' || current !== sequence) return;
      if (error.message === 'SignInRequired') { signIn(); return; }
      fail(error.message || 'Could not load PRs');
    } finally {
      if (current === sequence) active = null;
    }
  }
  mineButton.addEventListener('click', async () => {
    if (!mine) {
      mine = true;
      save('tci-my-stuff', true);
      switches();
      try {
        const response = await fetch('/pr-search/mine', {credentials: 'same-origin'});
        if (response.status === 401) {
          signIn();
          return;
        }
        const data = await json(response);
        minePrs = data.prs.filter(pr => typeof pr === 'string' && /^[1-9][0-9]*$/.test(pr));
        mineUntil = Date.now() + 300000;
      } catch (error) { fail(error.message || 'Could not load your PR activity'); return; }
    } else { mine = false; save('tci-my-stuff', false); signInPending = false; }
    switches();
    refresh();
  });
  openButton.addEventListener('click', () => {
    hideClosed = !hideClosed;
    save('tci-hide-closed-prs', hideClosed);
    switches();
    refresh();
  });
  dayButtons.forEach(button => button.addEventListener('click', () => {
    const selected = Number(button.dataset.days);
    if (selected === days) return;
    days = selected;
    save('tci-pr-days', days);
    switches();
    refresh();
  }));
  switches();
  window.addEventListener('storage', event => {
    if (event.key !== 'tci-pr-auth-complete' || !signInPending) return;
    signInPending = false;
    mine = true;
    save('tci-my-stuff', true);
    switches();
    refresh();
  });
  refresh();
  setInterval(refresh, 10000);
})();
