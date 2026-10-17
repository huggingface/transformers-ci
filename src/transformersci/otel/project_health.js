(() => {
  'use strict';
  const RUNNERS = [['gpu', 'GPU'], ['cpu', 'CPU']];
  const CATEGORIES = [['failure', 'Failure']];
  const DAY = 86400;
  const $ = id => document.getElementById(id);
  const esc = value => String(value).replace(/[&<>"']/g, ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[ch]));
  const pct = (value, digits = 0) => value == null ? '–' : `${(100 * value).toFixed(digits)}%`;
  const params = new URLSearchParams(location.search);
  const state = {
    days: [7, 14, 30].includes(Number(params.get('days'))) ? Number(params.get('days')) : 14,
    cat: 'all', runner: 'all', sticky: false, q: '',
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

  function drawRows() {
    const {base, shown} = visible();
    drawCategories(base);
    $('days-head').textContent = `Last ${state.days} days`;
    $('rows').innerHTML = shown.map((row, index) => `<tr>
      <td><span class="badge k-${row.category}">${row.category.toUpperCase()}</span></td>
      <td class="test"><b>${esc(row.model)}</b><div>${esc(row.test)}</div></td>
      <td class="sig" data-i="${index}" tabindex="0">${esc(row.signal)}</td>
      <td><span class="rn">${row.runner}</span> <span class="job">${esc(row.job)}</span></td>
      <td class="num">${row.fails} fail${row.fails === 1 ? '' : 's'}${row.sticky ? '<span class="sticky">sticky</span>' : ''}<span class="strip">${row.days.map((count, i) =>
        `<i class="${count ? 'f' : ''}" title="${dayLabel(rowsFirstDay + i * DAY)}: ${count} failing run${count === 1 ? '' : 's'}"></i>`).join('')}</span></td>
      <td title="${new Date(row.last_seen * 1000).toISOString()}">${ago(row.last_seen)}</td></tr>`).join('')
      || `<tr><td colspan="6" class="state">${rows.length ? 'Nothing matches these filters.' : 'No failures on main in this window.'}</td></tr>`;
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
      $('rows').innerHTML = `<tr><td colspan="6" class="state">Could not load failures (${esc(error.message)}).</td></tr>`;
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
  $('sticky-only').addEventListener('change', event => { state.sticky = event.target.checked; drawRows(); });
  $('q').addEventListener('input', event => { state.q = event.target.value.trim().toLowerCase(); drawRows(); });

  press($('days'), state.days);
  loadTrend();
  loadRows();
})();
