(() => {
  'use strict';
  const RUNNERS = [['gpu', 'GPU'], ['cpu', 'CPU']];
  const CATEGORIES = [['failure', 'Failure']];
  // The Test page's Serge Actions (button id in /serge-actions, label).
  const ACTIONS = [['new-issue', 'New issue'], ['wdyt', 'WDYT?'], ['who-can-help', 'Who can help']];
  const DAY = 86400;
  const $ = id => document.getElementById(id);
  const esc = value => String(value).replace(/[&<>"']/g, ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[ch]));
  const pct = (value, digits = 0) => value == null ? '–' : `${(100 * value).toFixed(digits)}%`;
  const params = new URLSearchParams(location.search);
  const state = {
    days: [7, 14, 30].includes(Number(params.get('days'))) ? Number(params.get('days')) : 14,
    cat: 'all', runner: 'all', sticky: false, q: '', group: 'none',
  };
  let rows = [], generatedAt = 0;

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
  function drawChart(trend) {
    const svg = $('svg'), width = svg.clientWidth || 900, height = 200;
    const left = 36, right = 64, top = 10, bottom = 24;
    const plotW = width - left - right, plotH = height - top - bottom;
    const n = trend.days, x = i => left + (n === 1 ? plotW / 2 : (i * plotW) / (n - 1));
    const y = share => top + plotH * (1 - share);
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    let out = '<g class="grid">';
    for (const tick of [0, 0.25, 0.5, 0.75, 1]) out += `<line x1="${left}" x2="${left + plotW}" y1="${y(tick)}" y2="${y(tick)}"/>`;
    out += '</g><g class="axis">';
    for (const tick of [0, 0.5, 1]) out += `<text x="${left - 6}" y="${y(tick) + 4}" text-anchor="end">${tick * 100}%</text>`;
    const step = Math.max(1, Math.ceil(n / 7));
    for (let i = 0; i < n; i += step) out += `<text x="${x(i)}" y="${height - 6}" text-anchor="middle">${dayLabel(trend.first_day + i * DAY)}</text>`;
    out += '</g>';
    const labels = [];
    for (const [key, name] of RUNNERS) {
      const days = trend.series[key].days;
      let path = '', pen = false, last = null;
      days.forEach((day, i) => {
        if (!day.runs) { pen = false; return; }
        const share = day.failing / day.runs;
        path += `${pen ? 'L' : 'M'}${x(i).toFixed(1)},${y(share).toFixed(1)}`;
        pen = true; last = [i, share];
      });
      out += `<path class="line" d="${path}" stroke="var(--${key})"/>`;
      days.forEach((day, i) => {
        if (day.runs) out += `<circle class="dot" cx="${x(i)}" cy="${y(day.failing / day.runs)}" r="3" fill="var(--${key})"/>`;
      });
      if (last) labels.push({key, name, x: x(last[0]) + 8, y: y(last[1]) + 4});
    }
    // Keep the two end labels apart when the lines finish close together.
    if (labels.length === 2 && Math.abs(labels[0].y - labels[1].y) < 13) {
      const [hi, lo] = labels[0].y <= labels[1].y ? labels : [labels[1], labels[0]];
      const mid = (hi.y + lo.y) / 2; hi.y = mid - 7; lo.y = mid + 7;
    }
    for (const label of labels) out += `<text class="label" x="${label.x}" y="${label.y}">${label.name}</text>`;
    out += `<line class="cross" id="cross" y1="${top}" y2="${top + plotH}" visibility="hidden"/>`;
    out += `<rect x="${left}" y="${top}" width="${plotW}" height="${plotH}" fill="transparent" id="hit"/>`;
    svg.innerHTML = out;

    const hover = $('hover'), cross = $('cross');
    $('hit').addEventListener('mousemove', event => {
      const box = svg.getBoundingClientRect();
      const px = ((event.clientX - box.left) / box.width) * width;
      const i = Math.max(0, Math.min(n - 1, Math.round(((px - left) / plotW) * (n - 1))));
      cross.setAttribute('x1', x(i)); cross.setAttribute('x2', x(i)); cross.setAttribute('visibility', 'visible');
      let html = `<b>${dayLabel(trend.first_day + i * DAY)}</b>`;
      for (const [key, name] of RUNNERS) {
        const day = trend.series[key].days[i];
        html += `<div><i style="background:var(--${key})"></i>${name}: ${day.runs
          ? `${day.failing} of ${day.runs} runs (${pct(day.failing / day.runs)}) · ${pct(day.tests ? day.failed / day.tests : 0, 4)} of tests`
          : 'no merge run'}</div>`;
      }
      hover.innerHTML = html; hover.style.display = 'block';
      const chartBox = $('chart').getBoundingClientRect();
      const hx = event.clientX - chartBox.left + 14;
      hover.style.left = `${Math.min(hx, chartBox.width - hover.offsetWidth - 4)}px`;
      hover.style.top = `${event.clientY - chartBox.top + 12}px`;
    });
    $('hit').addEventListener('mouseleave', () => { hover.style.display = 'none'; cross.setAttribute('visibility', 'hidden'); });

    $('legend').innerHTML = RUNNERS.map(([key, name]) => {
      const series = trend.series[key];
      const what = key === 'gpu' ? 'run_models_gpu, push' : 'every CPU job on main';
      return `<span style="--c:var(--${key})">${name} (${what}) · ${series.failing} of ${series.runs} runs · ${pct(series.test_rate, 4)} of tests failed</span>`;
    }).join('');
    for (const [key, name] of RUNNERS) {
      const series = trend.series[key];
      $(`s-${key}`).textContent = pct(series.share);
      $(`s-${key}-n`).textContent = `${series.failing} of ${series.runs} runs · ${trend.days} days`;
    }
  }

  async function loadTrend() {
    $('chart-progress').hidden = false;
    try {
      const trend = await getJson('/project-health/trend');
      drawChart(trend);
      window.addEventListener('resize', () => drawChart(trend), {once: true});
    } catch (error) {
      $('svg').innerHTML = '';
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
  const openGroups = new Set();
  let groupsSeen = '';

  function rowHtml(row, index) {
    return `<tr>
      <td><span class="badge k-${row.category}">${row.category.toUpperCase()}</span></td>
      <td class="test"><b>${esc(row.model)}</b><div>${esc(row.test)}</div></td>
      <td class="sig" data-i="${index}" tabindex="0">${esc(row.signal)}</td>
      <td><span class="rn">${row.runner}</span> <span class="job">${esc(row.job)}</span></td>
      <td class="num">${row.fails} fail${row.fails === 1 ? '' : 's'}${row.sticky ? '<span class="sticky">sticky</span>' : ''}<span class="strip">${row.days.map((count, i) =>
        `<i class="${count ? 'f' : ''}" title="${dayLabel(rowsFirstDay + i * DAY)}: ${count} failing run${count === 1 ? '' : 's'}"></i>`).join('')}</span></td>
      <td title="${new Date(row.last_seen * 1000).toISOString()}">${ago(row.last_seen)}</td>
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
    const ordered = [...groups].sort((a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]));
    // A new grouping opens its small groups, as the PR CI view does; after that
    // the viewer's own choices stick.
    const seen = `${state.group}|${state.days}`;
    if (seen !== groupsSeen) {
      groupsSeen = seen; openGroups.clear();
      for (const [name, members] of ordered) if (members.length <= 5) openGroups.add(name);
    }
    return ordered.map(([name, members]) => {
      const open = openGroups.has(name);
      const errors = new Map();
      for (const [row] of members) errors.set(row.signal, (errors.get(row.signal) || 0) + 1);
      const summary = state.group === 'error' ? '' : ` · ${[...errors].sort((a, b) => b[1] - a[1]).slice(0, 3)
        .map(([error, count]) => `${count}× ${esc(error)}`).join(', ')}${errors.size > 3 ? ', …' : ''}`;
      const sticky = members.filter(([row]) => row.sticky).length;
      return `<tr class="grp" data-group="${esc(name)}" aria-expanded="${open}"><td colspan="7"><span class="caret">${open ? '▾' : '▸'}</span>${esc(name)}
        <span class="meta"> — ${members.length} test${members.length === 1 ? '' : 's'}${sticky ? `, ${sticky} sticky` : ''}${summary}</span></td></tr>`
        + (open ? members.map(([row, index]) => rowHtml(row, index)).join('') : '');
    }).join('');
  }

  function drawRows() {
    const {base, shown} = visible();
    drawCategories(base);
    $('days-head').textContent = `Last ${state.days} days`;
    $('rows').innerHTML = (state.group === 'none' ? shown.map(rowHtml).join('') : groupedHtml(shown))
      || `<tr><td colspan="7" class="state">${rows.length ? 'Nothing matches these filters.' : 'No failures on main in this window.'}</td></tr>`;
    $('count').textContent = `Showing ${shown.length} of ${rows.length}`;
    current = shown;
  }

  let current = [], rowsFirstDay = 0;
  async function loadRows() {
    $('rows-progress').hidden = false;
    try {
      const data = await getJson('/project-health/rows');
      rows = data.rows; generatedAt = data.generated_at; rowsFirstDay = data.first_day;
      drawRows();
      const fails = rows.filter(row => row.category === 'failure');
      $('s-fail').textContent = fails.length;
      $('s-fail-n').textContent = `${fails.filter(row => row.runner === 'GPU').length} GPU · ${fails.filter(row => row.runner === 'CPU').length} CPU · ${state.days} days`;
      $('s-sticky').textContent = fails.filter(row => row.sticky).length;
    } catch (error) {
      $('rows').innerHTML = `<tr><td colspan="7" class="state">Could not load failures (${esc(error.message)}).</td></tr>`;
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
  $('sticky-only').addEventListener('change', event => { state.sticky = event.target.checked; drawRows(); });
  $('q').addEventListener('input', event => { state.q = event.target.value.trim().toLowerCase(); drawRows(); });

  press($('days'), state.days);
  loadTrend();
  loadRows();
})();
