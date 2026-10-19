// Monitoring: how main is doing. The stat tiles, what changed on main, and
// the share of failing tests per merge run; Fix failures (project_health.js)
// has the table of tests to pick up.
(() => {
  'use strict';
  const RUNNERS = [['gpu', 'GPU'], ['cpu', 'CPU']];
  const DAY = 86400;
  const $ = id => document.getElementById(id);
  const {esc} = tci;
  const params = new URLSearchParams(location.search);
  const state = {days: [7, 14, 30].includes(Number(params.get('days'))) ? Number(params.get('days')) : 14};

  tci.applyTheme();

  const dayLabel = ts => new Date(ts * 1000).toLocaleDateString('en', {month: 'short', day: 'numeric', timeZone: 'UTC'});

  async function getJson(path) {
    const response = await fetch(`${path}?days=${state.days}`, {credentials: 'same-origin'});
    if (!response.ok) throw new Error(`${response.status}`);
    return response.json();
  }

  // ---- chart ---------------------------------------------------------------
  // Per-thousand rates of very different sizes (GPU ~1‰, CPU ~0.002‰): two
  // significant digits, never rounded to zero.
  function rate(value) {
    if (value == null) return '–';
    const permille = 1000 * value;
    if (permille === 0) return '0‰';
    return `${permille >= 1 ? permille.toFixed(1) : Number(permille.toPrecision(2))}‰`;
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

  // ---- events --------------------------------------------------------------
  // Bars and the issue line are per UTC day; markers sit at their merge time.
  const EVENT_KINDS = {model: ['Model added', 'var(--ev-model)'], dependency: ['Dependency change', 'var(--ev-dep)']};
  const SERIES = [['prs_opened', 'PRs opened', 'var(--ev-opened)', 'PRs opened'], ['prs_merged', 'PRs merged', 'var(--ev-merged)', 'PRs merged'], ['issues_opened', 'Issues opened', 'var(--amber)', 'issues opened'], ['issues_closed', 'Issues closed', 'var(--ev-closed)', 'issues closed']];
  const prPage = pr => `/d/pytest-observability-by-pr/branch?var-pr=${encodeURIComponent(pr)}&from=now-90d&to=now`;
  function drawEvents(data) {
    const host = $('chart-events'), svg = host.querySelector('svg'), hover = host.querySelector('.hover');
    const width = svg.clientWidth || 900, height = 244;
    const left = 46, right = 12, lane = 58, bottom = 24;
    const top = lane + 6, plotW = width - left - right, plotH = height - top - bottom;
    const n = data.days, band = plotW / n, counts = data.counts;
    const dayX = i => left + band * (i + 0.5);
    const timeX = ts => left + ((ts - data.first_day) / DAY) * band;
    const peak = counts ? Math.max(1, ...counts.flatMap(day => SERIES.map(([key]) => day[key] ?? 0))) : 1;
    const top_ = Math.ceil(niceMax(peak * 1.1));
    const y = value => top + plotH * (1 - value / top_);
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    let out = '<g class="grid">';
    for (const tick of [0, 0.5, 1]) out += `<line x1="${left}" x2="${left + plotW}" y1="${y(tick * top_)}" y2="${y(tick * top_)}"/>`;
    out += '</g><g class="axis">';
    if (counts) for (const tick of [0, 0.5, 1]) out += `<text x="${left - 6}" y="${y(tick * top_) + 4}" text-anchor="end">${Math.round(tick * top_)}</text>`;
    const step = Math.max(1, Math.ceil(n / (width < 500 ? 4 : 8)));
    for (let i = 0; i < n; i += step) out += `<text x="${dayX(i)}" y="${height - 6}" text-anchor="middle">${dayLabel(data.first_day + i * DAY)}</text>`;
    out += '</g>';
    if (counts) {
      const barW = Math.max(2, Math.min(12, band * 0.32));
      counts.forEach((day, i) => {
        [['prs_opened', -barW - 1], ['prs_merged', 1]].forEach(([key, offset]) => {
          const value = day[key];
          if (!value) return;
          out += `<rect x="${dayX(i) + offset}" y="${y(value)}" width="${barW}" height="${y(0) - y(value)}" rx="1.5" fill="${SERIES.find(s => s[0] === key)[2]}"/>`;
        });
      });
      for (const [key, , color] of SERIES.filter(([key]) => key.startsWith('issues_'))) {
        let path = '';
        counts.forEach((day, i) => { if (day[key] != null) path += `${path ? 'L' : 'M'}${dayX(i).toFixed(1)},${y(day[key]).toFixed(1)}`; });
        out += `<path class="line" d="${path}" stroke="${color}"/>`;
        counts.forEach((day, i) => { if (day[key] != null) out += `<circle class="dot" cx="${dayX(i)}" cy="${y(day[key])}" r="2.5" fill="${color}"/>`; });
      }
    }
    out += `<line class="cross" y1="${top}" y2="${top + plotH}" visibility="hidden"/>`;
    out += `<rect x="${left}" y="${top}" width="${plotW}" height="${plotH}" fill="transparent" class="hit"/>`;
    // Markers stack upwards when they would overlap: four models merged within
    // a few hours of each other must stay four clickable dots.
    const levels = [];
    data.events.forEach((event, index) => {
      const cx = timeX(event.at);
      let level = levels.findIndex(last => cx - last > 13);
      if (level === -1) level = levels.length < 4 ? levels.length : levels.length - 1;
      levels[level] = cx;
      const cy = lane - 7 - level * 13, color = EVENT_KINDS[event.kind][1];
      const mark = event.kind === 'model'
        ? `<circle class="mark" cx="${cx}" cy="${cy}" r="5" fill="${color}" stroke="var(--panel)" stroke-width="1.5"/>`
        : `<rect class="mark" x="${cx - 4.5}" y="${cy - 4.5}" width="9" height="9" transform="rotate(45 ${cx} ${cy})" fill="${color}" stroke="var(--panel)" stroke-width="1.5"/>`;
      const body = `<line class="guide" x1="${cx}" x2="${cx}" y1="${cy + 5}" y2="${top + plotH}" stroke="${color}"/>${mark}`;
      out += event.pr
        ? `<a class="ev" data-i="${index}" href="${prPage(event.pr)}" target="_top">${body}</a>`
        : `<g class="ev" data-i="${index}">${body}</g>`;
    });
    svg.innerHTML = out;
    const place = (event, html) => {
      hover.innerHTML = html; hover.style.display = 'block';
      const chartBox = host.getBoundingClientRect();
      hover.style.left = `${Math.max(4, Math.min(event.clientX - chartBox.left + 14, chartBox.width - hover.offsetWidth - 4))}px`;
      hover.style.top = `${event.clientY - chartBox.top + 12}px`;
    };
    const eventLine = event => `<div class="evline"><i style="background:${EVENT_KINDS[event.kind][1]}"></i>${esc(event.title)}${event.pr ? ` <span class="sub">#${esc(event.pr)}</span>` : ''}</div>`;
    const cross = svg.querySelector('.cross'), hit = svg.querySelector('.hit');
    hit.addEventListener('mousemove', event => {
      const box = svg.getBoundingClientRect();
      const px = ((event.clientX - box.left) / box.width) * width;
      const i = Math.max(0, Math.min(n - 1, Math.floor((px - left) / band)));
      cross.setAttribute('x1', dayX(i)); cross.setAttribute('x2', dayX(i)); cross.setAttribute('visibility', 'visible');
      const start = data.first_day + i * DAY;
      const today = data.events.filter(item => item.at >= start && item.at < start + DAY);
      place(event, `<b>${dayLabel(start)}</b>${counts
        ? SERIES.map(([key, , color, inline]) => `<div><i style="background:${color}"></i>${counts[i][key] ?? '–'} ${inline}</div>`).join('')
        : ''}${today.map(eventLine).join('')}`);
    });
    hit.addEventListener('mouseleave', () => { hover.style.display = 'none'; cross.setAttribute('visibility', 'hidden'); });
    svg.querySelectorAll('.ev').forEach(node => {
      const item = data.events[Number(node.dataset.i)];
      node.addEventListener('mousemove', event => place(event,
        `<b>${EVENT_KINDS[item.kind][0]}</b>${eventLine(item)}<div class="sub">${new Date(item.at * 1000).toISOString().slice(0, 16).replace('T', ' ')} UTC${item.pr ? ' · click for the PR page' : ''}</div>`));
      node.addEventListener('mouseleave', () => { hover.style.display = 'none'; });
    });
    const total = key => counts ? counts.reduce((sum, day) => sum + (day[key] || 0), 0).toLocaleString('en') : '–';
    const kinds = kind => data.events.filter(event => event.kind === kind).length;
    $('ev-legend').innerHTML = SERIES.map(([key, name, color]) => `<span style="--c:${color}"><i class="${key.startsWith('issues_') ? 'ln' : ''}"></i>${name} <b>${total(key)}</b></span>`).join('')
      + `<span style="--c:var(--ev-model)"><i class="mk"></i>Models added <b>${kinds('model')}</b></span>`
      + `<span style="--c:var(--ev-dep)"><i class="dm"></i>Dependency changes <b>${kinds('dependency')}</b></span>`
      + (counts ? '' : `<span>PR and issue counts unavailable${data.counts_error ? ` (${esc(data.counts_error)})` : ''}</span>`);
  }

  async function loadEvents() {
    $('events-progress').hidden = false;
    try {
      const data = await getJson('/project-health/events');
      drawEvents(data);
      window.addEventListener('resize', () => drawEvents(data), {once: true});
    } catch (error) {
      $('chart-events').querySelector('svg').innerHTML = '';
      $('ev-legend').textContent = `Could not load events (${error.message}).`;
    } finally {
      $('events-progress').hidden = true;
    }
  }

  // ---- failing tests -------------------------------------------------------
  // The two count tiles come from Fix failures' rows (cached server side).
  async function loadCounts() {
    try {
      const data = await getJson('/project-health/rows');
      const fails = data.rows.filter(row => row.category === 'failure');
      $('s-fail').textContent = fails.length;
      $('s-fail-n').textContent = `${fails.filter(row => row.runner === 'GPU').length} GPU · ${fails.filter(row => row.runner === 'CPU').length} CPU · ${state.days} days`;
      $('s-sticky').textContent = fails.filter(row => row.sticky).length;
    } catch (_) {
      $('s-fail-n').textContent = 'unavailable';
    }
  }

  loadEvents();
  loadTrend();
  loadCounts();
})();
