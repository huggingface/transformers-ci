(() => {
  'use strict';
  const RUNNERS = [['gpu', 'GPU'], ['cpu', 'CPU']];
  const CATEGORIES = [['failure', 'Failure']];
  // The Test page's Serge Actions (button id in /serge-actions, label).
  const ACTIONS = [['new-issue', 'New issue'], ['wdyt', 'WDYT?'], ['who-can-help', 'Who can help']];
  const DAY = 86400;
  const $ = id => document.getElementById(id);
  const esc = value => String(value).replace(/[&<>"']/g, ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[ch]));
  const params = new URLSearchParams(location.search);
  const state = {
    days: [7, 14, 30].includes(Number(params.get('days'))) ? Number(params.get('days')) : 14,
    cat: 'all', runner: 'all', sticky: false, q: '', group: 'none', sort: 'fails', unclaimed: false, mine: false,
  };
  let rows = [], generatedAt = 0, claims = {}, me = '';

  try {
    const body = parent.document.body;
    if (/light/.test(body.className) || parent.grafanaBootData?.user?.lightTheme) document.body.classList.add('light');
  } catch (_) {}

  function ago(ts) {
    const seconds = Math.max(0, (generatedAt || Date.now() / 1000) - ts);
    for (const [suffix, size] of [['d', DAY], ['h', 3600], ['m', 60]]) {
      if (seconds >= size) return `${Math.floor(seconds / size)}${suffix} ago`;
    }
    return 'just now';
  }
  const dayLabel = ts => new Date(ts * 1000).toLocaleDateString('en', {month: 'short', day: 'numeric', timeZone: 'UTC'});

  async function getJson(path) {
    const response = await fetch(`${path}?days=${state.days}`, {credentials: 'same-origin'});
    if (!response.ok) throw new Error(`${response.status}`);
    return response.json();
  }

  // ---- chart ---------------------------------------------------------------
  // Percentages of very different sizes (GPU ~0.1%, CPU ~0.0002%): two
  // significant digits, never rounded to zero.
  function rate(value) {
    if (value == null) return '–';
    const percent = 100 * value;
    if (percent === 0) return '0%';
    return `${percent >= 1 ? percent.toFixed(1) : Number(percent.toPrecision(2))}%`;
  }
  function niceMax(value) {
    if (!(value > 0)) return 1e-4;
    const power = 10 ** Math.floor(Math.log10(value));
    return [1, 2, 2.5, 5, 10].map(step => step * power).find(step => step >= value);
  }
  function drawRunner(trend, key, name) {
    const host = $(`chart-${key}`), svg = host.querySelector('svg'), hover = host.querySelector('.hover');
    const width = svg.clientWidth || 600, height = 180;
    const left = 58, right = 12, top = 10, bottom = 24;
    const plotW = width - left - right, plotH = height - top - bottom;
    const days = trend.series[key].days, n = trend.days;
    const top_ = niceMax(Math.max(0, ...days.map(day => day.rate ?? 0)) * 1.1);
    const x = i => left + (n === 1 ? plotW / 2 : (i * plotW) / (n - 1));
    const y = value => top + plotH * (1 - value / top_);
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    let out = '<g class="grid">';
    for (const tick of [0, 0.5, 1]) out += `<line x1="${left}" x2="${left + plotW}" y1="${y(tick * top_)}" y2="${y(tick * top_)}"/>`;
    out += '</g><g class="axis">';
    for (const tick of [0, 0.5, 1]) out += `<text x="${left - 6}" y="${y(tick * top_) + 4}" text-anchor="end">${rate(tick * top_)}</text>`;
    const step = Math.max(1, Math.ceil(n / (width < 500 ? 4 : 6)));
    for (let i = 0; i < n; i += step) out += `<text x="${x(i)}" y="${height - 6}" text-anchor="middle">${dayLabel(trend.first_day + i * DAY)}</text>`;
    out += '</g>';
    let path = '', pen = false;
    days.forEach((day, i) => {
      if (day.rate == null) { pen = false; return; }
      path += `${pen ? 'L' : 'M'}${x(i).toFixed(1)},${y(day.rate).toFixed(1)}`;
      pen = true;
    });
    out += `<path class="line" d="${path}" stroke="var(--${key})"/>`;
    days.forEach((day, i) => {
      if (day.rate != null) out += `<circle class="dot" cx="${x(i)}" cy="${y(day.rate)}" r="3" fill="var(--${key})"/>`;
    });
    out += `<line class="cross" y1="${top}" y2="${top + plotH}" visibility="hidden"/>`;
    out += `<rect x="${left}" y="${top}" width="${plotW}" height="${plotH}" fill="transparent" class="hit"/>`;
    svg.innerHTML = out;
    const cross = svg.querySelector('.cross'), hit = svg.querySelector('.hit');
    hit.addEventListener('mousemove', event => {
      const box = svg.getBoundingClientRect();
      const px = ((event.clientX - box.left) / box.width) * width;
      const i = Math.max(0, Math.min(n - 1, Math.round(((px - left) / plotW) * (n - 1))));
      cross.setAttribute('x1', x(i)); cross.setAttribute('x2', x(i)); cross.setAttribute('visibility', 'visible');
      const day = days[i];
      hover.innerHTML = `<b>${name} · ${dayLabel(trend.first_day + i * DAY)}</b>${day.runs
        ? `<div>${rate(day.rate)} of tests failing, averaged over ${day.runs} run${day.runs === 1 ? '' : 's'}</div>
           <div>${Math.round(day.failed).toLocaleString('en')} failed of ${Math.round(day.tests).toLocaleString('en')} tests</div>
           <div>${day.failing} of ${day.runs} runs had a failure</div>`
        : '<div>no merge run</div>'}`;
      hover.style.display = 'block';
      const chartBox = host.getBoundingClientRect();
      hover.style.left = `${Math.min(event.clientX - chartBox.left + 14, chartBox.width - hover.offsetWidth - 4)}px`;
      hover.style.top = `${event.clientY - chartBox.top + 12}px`;
    });
    hit.addEventListener('mouseleave', () => { hover.style.display = 'none'; cross.setAttribute('visibility', 'hidden'); });
  }
  function drawChart(trend) {
    for (const [key, name] of RUNNERS) drawRunner(trend, key, name);
    $('legend').innerHTML = RUNNERS.map(([key, name]) => {
      const series = trend.series[key];
      return `<span style="--c:var(--${key})">${name}: ${rate(series.rate)} of tests failing per merge run on average · ${series.failing} of ${series.runs} runs had a failure</span>`;
    }).join('');
    for (const [key] of RUNNERS) {
      const series = trend.series[key];
      $(`s-${key}`).textContent = rate(series.rate);
      $(`s-${key}-n`).textContent = `average over ${series.runs} runs · ${trend.days} days`;
    }
  }

  async function loadTrend() {
    $('chart-progress').hidden = false;
    try {
      const trend = await getJson('/project-health/trend');
      drawChart(trend);
      window.addEventListener('resize', () => drawChart(trend), {once: true});
    } catch (error) {
      document.querySelectorAll('.chart svg').forEach(svg => { svg.innerHTML = ''; });
      $('legend').textContent = `Could not load merge runs (${error.message}).`;
    } finally {
      $('chart-progress').hidden = true;
    }
  }

  // ---- table ---------------------------------------------------------------
  function testUrl(row) {
    const query = new URLSearchParams({
      orgId: '1', 'var-trace_id': row.trace_id, 'var-test_nodeid': row.nodeid,
      'var-run_id': row.run_id, 'var-pr': 'main', 'var-gh_run_id': String(row.run_id).split(':')[0],
      from: `now-${state.days}d`, to: 'now',
    });
    return `/d/pytest-test/test?${query}`;
  }

  function visible() {
    const q = state.q;
    const base = rows.filter(row => (state.runner === 'all' || row.runner === state.runner)
      && (!state.sticky || row.sticky)
      && (!state.unclaimed || !claims[row.nodeid])
      && (!state.mine || claims[row.nodeid]?.login === me)
      && (!q || `${row.model} ${row.nodeid} ${row.job} ${row.signal}`.toLowerCase().includes(q)));
    return {base, shown: base.filter(row => state.cat === 'all' || row.category === state.cat)};
  }

  function drawCategories(base) {
    $('cat').innerHTML = [['all', 'All'], ...CATEGORIES].map(([value, name]) => {
      const count = value === 'all' ? base.length : base.filter(row => row.category === value).length;
      return `<button data-v="${value}" aria-pressed="${state.cat === value}">${name}<small>${count}</small></button>`;
    }).join('');
  }

  // Group keys, as on the PR CI view. "Test" is the test class, "Function" the
  // test function without its parameters.
  const fn = row => row.test.split('::').pop().replace(/\[.*\]$/, '');
  const GROUP_KEYS = {
    model: row => row.model,
    test: row => row.test.includes('::') ? row.test.split('::')[0] : fn(row),
    function: fn,
    test_model: row => `${row.model} · ${fn(row)}`,
    error: row => row.signal,
  };
  // Sort toggle: most failing runs first, or alphabetical by name.
  const byName = (a, b) => `${a.model} ${a.test}`.localeCompare(`${b.model} ${b.test}`) || a.job.localeCompare(b.job);
  const byFails = (a, b) => b.fails - a.fails || b.bad_days - a.bad_days || b.last_seen - a.last_seen || byName(a, b);
  const sortRows = list => [...list].sort(state.sort === 'alpha' ? byName : byFails);
  const openGroups = new Set();
  let groupsSeen = '', groupNames = [];

  function claimHtml(claim) {
    if (!claim) return '<span class="sub">—</span>';
    const who = `<a href="https://github.com/${encodeURIComponent(claim.login)}" target="_blank" rel="noopener noreferrer">@${esc(claim.login)}</a>`;
    if (claim.state === 'claimed') {
      return `${who} · <a href="${esc(claim.url)}" target="_blank" rel="noopener noreferrer">#${claim.number}</a>`;
    }
    return `${who} <span class="drafting">drafting…</span>`;
  }

  function rowHtml(row, index) {
    return `<tr>
      <td><span class="badge k-${row.category}">${row.category.toUpperCase()}</span></td>
      <td class="test"><b>${esc(row.model)}</b><div>${esc(row.test)}</div></td>
      <td class="sig" data-i="${index}" tabindex="0">${esc(row.signal)}</td>
      <td><span class="rn">${row.runner}</span> <span class="job">${esc(row.job)}</span></td>
      <td class="num">${row.fails} fail${row.fails === 1 ? '' : 's'}${row.sticky ? '<span class="sticky">sticky</span>' : ''}<span class="strip">${row.days.map((count, i) =>
        `<i class="${count ? 'f' : ''}" title="${dayLabel(rowsFirstDay + i * DAY)}: ${count} failing run${count === 1 ? '' : 's'}"></i>`).join('')}</span></td>
      <td title="${new Date(row.last_seen * 1000).toISOString()}">${ago(row.last_seen)}</td>
      <td class="claim">${claimHtml(claims[row.nodeid])}</td>
      <td><span class="acts">${ACTIONS.map(([id, label], k) =>
        `<button type="button" class="act${k ? '' : ' pri'}" data-act="${id}" data-i="${index}">${label}</button>`).join('')}</span></td></tr>`;
  }

  function groupedHtml(shown) {
    const key = GROUP_KEYS[state.group];
    const groups = new Map();
    shown.forEach((row, index) => {
      const name = key(row) || '(unknown)';
      if (!groups.has(name)) groups.set(name, []);
      groups.get(name).push([row, index]);
    });
    const fails = members => members.reduce((sum, [row]) => sum + row.fails, 0);
    const ordered = [...groups].sort(state.sort === 'alpha'
      ? (a, b) => a[0].localeCompare(b[0])
      : (a, b) => fails(b[1]) - fails(a[1]) || b[1].length - a[1].length || a[0].localeCompare(b[0]));
    // A new grouping opens its small groups, as the PR CI view does; after that
    // the viewer's own choices stick.
    const seen = `${state.group}|${state.days}`;
    if (seen !== groupsSeen) {
      groupsSeen = seen; openGroups.clear();
      for (const [name, members] of ordered) if (members.length <= 5) openGroups.add(name);
    }
    groupNames = ordered.map(([name]) => name);
    return ordered.map(([name, members]) => {
      const open = openGroups.has(name);
      const errors = new Map();
      for (const [row] of members) errors.set(row.signal, (errors.get(row.signal) || 0) + 1);
      const summary = state.group === 'error' ? '' : ` · ${[...errors].sort((a, b) => b[1] - a[1]).slice(0, 3)
        .map(([error, count]) => `${count}× ${esc(error)}`).join(', ')}${errors.size > 3 ? ', …' : ''}`;
      const sticky = members.filter(([row]) => row.sticky).length;
      return `<tr class="grp" data-group="${esc(name)}" aria-expanded="${open}"><td colspan="8"><span class="caret">${open ? '▾' : '▸'}</span>${esc(name)}
        <span class="meta"> — ${members.length} test${members.length === 1 ? '' : 's'}, ${fails(members)} failing run${fails(members) === 1 ? '' : 's'}${sticky ? `, ${sticky} sticky` : ''}${summary}</span></td></tr>`
        + (open ? members.map(([row, index]) => rowHtml(row, index)).join('') : '');
    }).join('');
  }

  const FOLD_PATHS = {
    collapse: 'M4.5 2.5L8 6l3.5-3.5M4.5 13.5L8 10l3.5 3.5',
    expand: 'M4.5 6L8 2.5 11.5 6M4.5 10L8 13.5 11.5 10',
  };
  function drawFold() {
    const fold = $('fold');
    fold.hidden = state.group === 'none';
    const collapse = groupNames.some(name => openGroups.has(name));
    const label = collapse ? 'Collapse all groups' : 'Expand all groups';
    fold.title = label; fold.setAttribute('aria-label', label);
    fold.innerHTML = `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${FOLD_PATHS[collapse ? 'collapse' : 'expand']}"/></svg>`;
  }

  function drawRows() {
    const visibleRows = visible();
    const base = visibleRows.base, shown = sortRows(visibleRows.shown);
    drawCategories(base);
    $('days-head').textContent = `Last ${state.days} days`;
    $('rows').innerHTML = (state.group === 'none' ? shown.map(rowHtml).join('') : groupedHtml(shown))
      || `<tr><td colspan="8" class="state">${rows.length ? 'Nothing matches these filters.' : 'No failures on main in this window.'}</td></tr>`;
    $('count').textContent = `Showing ${shown.length} of ${rows.length}`;
    current = shown;
    drawFold();
  }

  let current = [], rowsFirstDay = 0;
  async function loadRows(quiet = false) {
    if (!quiet) $('rows-progress').hidden = false;
    try {
      const data = await getJson('/project-health/rows');
      rows = data.rows; generatedAt = data.generated_at; rowsFirstDay = data.first_day;
      claims = data.claims || {};
      drawRows();
      const fails = rows.filter(row => row.category === 'failure');
      $('s-fail').textContent = fails.length;
      $('s-fail-n').textContent = `${fails.filter(row => row.runner === 'GPU').length} GPU · ${fails.filter(row => row.runner === 'CPU').length} CPU · ${state.days} days`;
      $('s-sticky').textContent = fails.filter(row => row.sticky).length;
    } catch (error) {
      if (quiet) return;
      $('rows').innerHTML = `<tr><td colspan="8" class="state">Could not load failures (${esc(error.message)}).</td></tr>`;
    } finally {
      $('rows-progress').hidden = true;
    }
  }

  // ---- traceback popup -----------------------------------------------------
  const tip = $('tip'), frame = $('tip-frame');
  let hideTimer = null, tipRow = null;
  function showTip(cell) {
    clearTimeout(hideTimer);
    const row = current[Number(cell.dataset.i)];
    if (!row) return;
    if (row !== tipRow) {
      tipRow = row;
      $('tip-title').textContent = row.signal;
      $('tip-sub').textContent = `${row.model} · ${row.job} · last failure ${ago(row.last_seen)}`;
      $('tip-more').href = testUrl(row);
      frame.src = `/failure?${new URLSearchParams({trace_id: row.trace_id, test_nodeid: row.nodeid})}`;
    }
    tip.classList.add('show');
    const box = cell.getBoundingClientRect();
    const top = Math.min(box.bottom + 4, innerHeight - tip.offsetHeight - 8);
    tip.style.top = `${Math.max(8, top)}px`;
    tip.style.left = `${Math.max(8, Math.min(box.left, innerWidth - tip.offsetWidth - 8))}px`;
  }
  const hideTip = () => { hideTimer = setTimeout(() => tip.classList.remove('show'), 200); };
  $('rows').addEventListener('mouseover', event => { const cell = event.target.closest('td.sig'); if (cell) showTip(cell); });
  $('rows').addEventListener('mouseout', event => { if (event.target.closest('td.sig')) hideTip(); });
  $('rows').addEventListener('focusin', event => { const cell = event.target.closest('td.sig'); if (cell) showTip(cell); });
  tip.addEventListener('mouseenter', () => clearTimeout(hideTimer));
  tip.addEventListener('mouseleave', hideTip);
  document.addEventListener('keydown', event => { if (event.key === 'Escape') tip.classList.remove('show'); });

  // ---- actions -------------------------------------------------------------
  // Each row's buttons drive the Test page's own Serge Actions, unchanged: a
  // hidden /serge-actions frame per test, whose button we click. It handles
  // sign-in, the streamed steps and the dialogs (opened in the top-most page),
  // so the behaviour matches the Test page exactly. A frame is kept per test
  // because reloading it would abort a question still running in it.
  const FRAME_LIMIT = 8;
  const frames = new Map();
  function sergeFrame(row) {
    const key = `${row.nodeid}\n${row.job}`;
    let entry = frames.get(key);
    if (entry) { frames.delete(key); frames.set(key, entry); return entry; }
    const frame = document.createElement('iframe');
    frame.className = 'serge'; frame.title = 'Serge Actions'; frame.tabIndex = -1;
    frame.src = `/serge-actions?${new URLSearchParams({
      trace_id: row.trace_id, test_nodeid: row.nodeid, job: row.job, status: 'ERROR',
      module: row.nodeid.split('::')[0].split('/').pop(), pr: 'main', run_id: row.run_id,
    })}`;
    entry = {frame, loaded: new Promise(resolve => frame.addEventListener('load', resolve, {once: true}))};
    $('serge-frames').append(frame);
    frames.set(key, entry);
    while (frames.size > FRAME_LIMIT) {
      const [oldest, old] = frames.entries().next().value;
      old.frame.remove(); frames.delete(oldest);
    }
    return entry;
  }
  async function runAction(button) {
    const row = current[Number(button.dataset.i)];
    if (!row || button.getAttribute('aria-busy') === 'true') return;
    button.setAttribute('aria-busy', 'true');
    try {
      const {frame, loaded} = sergeFrame(row);
      await loaded;
      // The frame enables its buttons once it knows whether the viewer is signed in.
      for (let tries = 0; tries < 100; tries++) {
        const target = frame.contentDocument?.getElementById(button.dataset.act);
        if (target && !target.disabled) { target.click(); return; }
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    } finally {
      button.removeAttribute('aria-busy');
    }
  }
  $('rows').addEventListener('click', event => {
    const button = event.target.closest('button[data-act]');
    if (button) { runAction(button); return; }
    const header = event.target.closest('tr.grp');
    if (!header) return;
    const name = header.dataset.group;
    if (openGroups.has(name)) openGroups.delete(name); else openGroups.add(name);
    drawRows();
  });

  // ---- controls ------------------------------------------------------------
  function press(seg, value) {
    seg.querySelectorAll('button').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.v === String(value))));
  }
  $('cat').addEventListener('click', event => {
    const button = event.target.closest('button'); if (!button) return;
    state.cat = button.dataset.v; drawRows();
  });
  $('runner').addEventListener('click', event => {
    const button = event.target.closest('button'); if (!button) return;
    state.runner = button.dataset.v; press($('runner'), state.runner); drawRows();
  });
  $('days').addEventListener('click', event => {
    const button = event.target.closest('button'); if (!button || Number(button.dataset.v) === state.days) return;
    state.days = Number(button.dataset.v); press($('days'), state.days);
    params.set('days', String(state.days));
    history.replaceState(null, '', `${location.pathname}?${params}`);
    loadRows(); loadTrend();
  });
  $('group').addEventListener('click', event => {
    const button = event.target.closest('button'); if (!button) return;
    state.group = button.dataset.v; press($('group'), state.group); drawRows();
  });
  $('sort').addEventListener('click', event => {
    const button = event.target.closest('button'); if (!button) return;
    state.sort = button.dataset.v; press($('sort'), state.sort); drawRows();
  });
  $('fold').addEventListener('click', () => {
    if (groupNames.some(name => openGroups.has(name))) openGroups.clear();
    else groupNames.forEach(name => openGroups.add(name));
    drawRows();
  });
  $('unclaimed-only').addEventListener('change', event => { state.unclaimed = event.target.checked; drawRows(); });
  $('mine-only').addEventListener('change', event => { state.mine = event.target.checked; drawRows(); });
  $('sticky-only').addEventListener('change', event => { state.sticky = event.target.checked; drawRows(); });
  $('q').addEventListener('input', event => { state.q = event.target.value.trim().toLowerCase(); drawRows(); });

  press($('days'), state.days);
  loadTrend();
  loadRows();
  // Claims move while people work (Bug Days): refresh the rows quietly. The
  // server caches the Prometheus part, so this only re-reads the claims.
  setInterval(() => { if (!document.hidden) loadRows(true); }, 30000);
  fetch('/api/user', {credentials: 'same-origin', cache: 'no-store'})
    .then(response => response.ok ? response.json() : null)
    .then(user => { if (user?.login) { me = user.login; $('mine-toggle').hidden = false; } })
    .catch(() => {});
})();
