(() => {
  'use strict';
  const rows = document.getElementById('rows'), empty = document.getElementById('empty'), status = document.getElementById('status');
  tci.applyTheme();
  // Choices persist per viewer, like Last 20 PRs' toggles.
  const pick = (key, choices) => { const value = Number(tci.stored(key)); return choices.includes(value) ? value : choices[0]; };
  const state = {
    days: pick('tci-branch-days', [7, 14, 21]),
    limit: pick('tci-branch-limit', [10, 20, 30]),
    main: tci.stored('tci-branch-main') === 'true',
  };
  const mainButton = document.getElementById('main');
  const dayButtons = [...document.querySelectorAll('[data-days]')], limitButtons = [...document.querySelectorAll('[data-limit]')];
  function controls() {
    mainButton.setAttribute('aria-checked', String(state.main));
    dayButtons.forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.days) === state.days)));
    limitButtons.forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.limit) === state.limit)));
  }
  function choose(key, value) {
    state[key] = value;
    tci.save({days: 'tci-branch-days', limit: 'tci-branch-limit', main: 'tci-branch-main'}[key], value);
    controls();
    refresh(true);
  }
  mainButton.addEventListener('click', () => choose('main', !state.main));
  dayButtons.forEach(button => button.addEventListener('click', () => choose('days', Number(button.dataset.days))));
  limitButtons.forEach(button => button.addEventListener('click', () => choose('limit', Number(button.dataset.limit))));
  controls();
  const cell = (cls, text) => tci.node('td', cls, text);
  // Grafana's `s` unit, which the table used: 9.77 s, 2.47 mins, 3.27 hours, 3.23 days.
  function seconds(value) {
    if (value == null) return '';
    for (const [size, unit] of [[86400, 'days'], [3600, 'hours'], [60, 'mins']]) {
      if (value >= size) return `${(value / size).toPrecision(3)} ${unit}`;
    }
    return `${value.toPrecision(3)} s`;
  }
  // Grafana's dateTimeFromNow: "8 hours ago", "a day ago".
  function fromNow(ts) {
    const s = Math.max(0, Date.now() / 1000 - ts);
    if (s < 45) return 'a few seconds ago';
    for (const [limit, size, one, many] of [[2700, 60, 'a minute', 'minutes'], [79200, 3600, 'an hour', 'hours'], [2246400, 86400, 'a day', 'days']]) {
      if (s < limit) { const n = Math.round(s / size); return n <= 1 ? `${one} ago` : `${n} ${many} ago`; }
    }
    return `${Math.round(s / 2592000)} months ago`;
  }
  function rateCell(rate) {
    if (rate == null) return cell('');
    const td = cell(`rate ${rate === 0 ? 'rate-zero' : rate < 1 ? 'rate-low' : rate < 10 ? 'rate-medium' : 'rate-high'}`,
      rate === 0 ? '0%' : rate < 1 ? '<1%' : `${Number(rate.toPrecision(3))}%`);
    td.title = `${rate.toPrecision(3)}% of tests failed`;
    return td;
  }
  function render(data) {
    rows.replaceChildren();
    empty.hidden = data.rows.length > 0;
    empty.textContent = `No ${data.main_only ? 'main' : 'branch'} runs in the last ${data.days} days.`;
    for (const run of data.rows) {
      const tr = document.createElement('tr');
      const live = cell('');
      if (run.live) live.append(tci.spinner('In progress'));
      const commit = cell('');
      const link = document.createElement('a');
      link.href = `/d/pytest-observability-by-run/pytest-observability-run?var-run_id=${encodeURIComponent(run.run_id)}`;
      link.target = '_top';
      link.textContent = run.commit_message || run.run_id;
      link.title = run.commit_message || run.run_id;
      commit.append(link);
      const lanes = cell('');
      if (run.lanes.length) {
        const icons = tci.laneIcons(run.lanes);
        icons.title = run.lanes.map(hardware => tci.laneName(hardware)).join(' + ');
        lanes.append(icons);
      } else lanes.textContent = '-';
      tr.append(live, cell('', run.branch), commit, cell('num', run.jobs ?? ''), lanes,
        cell('num', run.total == null ? '' : Math.round(run.total)), cell('num', fromNow(run.started)),
        cell('num', seconds(run.duration)), cell('num', seconds(run.wall)), rateCell(run.rate));
      tr.children[6].title = new Date(run.started * 1000).toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
      rows.append(tr);
    }
  }
  let loading = false, wanted = '';
  // A click while a refresh is in flight must not lose to the older answer.
  async function refresh(force = false) {
    if ((loading && !force) || document.hidden) return;
    const query = new URLSearchParams({days: state.days, limit: state.limit, main: state.main ? '1' : '0'}).toString();
    wanted = query; loading = true;
    tci.status(status, 'Loading…');
    try {
      const response = await fetch(`/branch-runs/rows?${query}`, {credentials: 'same-origin', cache: 'no-store'});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      if (query !== wanted) return;
      render(data);
      tci.status(status, '');
    } catch (error) {
      tci.status(status, `Could not load runs (${error.message}). Retrying…`, true);
    } finally { loading = false; }
  }
  refresh();
  setInterval(refresh, 10000);
})();
