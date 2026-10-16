/* Latest CPU and run-slow results in one panel, with the run table's grouping controls. */
(() => {
  'use strict';
  const pr = new URLSearchParams(location.search).get('pr');
  const runs = document.getElementById('runs');
  const failures = document.getElementById('failures');
  const jobs = document.getElementById('jobs');
  const jobsEmpty = document.getElementById('jobs-empty');
  const onlySpecific = document.getElementById('specific');
  const search = document.getElementById('search');
  const progress = document.getElementById('progress');
  const groupButtons = [...document.querySelectorAll('[data-group]')];
  let snapshot = null, loading = false;
  let group = 'none';
  try {
    const body = parent.document.body;
    if (/light/.test(body.className) || parent.grafanaBootData?.user?.lightTheme) document.body.classList.add('light');
  } catch (_) {}
  try {
    const saved = localStorage.getItem('tci-pr-specific-v2');
    onlySpecific.checked = saved === null ? true : saved === 'true';
  } catch (_) { onlySpecific.checked = true; }
  const node = (tag, cls, value) => {
    const item = document.createElement(tag);
    if (cls) item.className = cls;
    if (value != null) item.textContent = String(value);
    return item;
  };
  const link = (href, label) => {
    const item = node('a', '', label);
    item.href = href;
    item.target = /^https?:/.test(href) ? '_blank' : '_top';
    item.rel = 'noopener';
    return item;
  };
  const runLink = run => run.url || `https://github.com/huggingface/transformers/actions/runs/${String(run.run_id).split(':')[0]}`;
  const jobLink = (lane, job) => {
    const query = new URLSearchParams({'var-job': job.name, 'var-run_id': lane.run.run_id,
      'var-pr': pr, 'var-hardware': job.hardware, from: 'now-90d', to: 'now'});
    return `/d/pytest-observability-by-job/pytest-observability-job?${query}`;
  };
  const live = status => ['in_progress', 'queued', 'waiting', 'requested', 'pending'].includes(status);
  const statusClass = status => ['failure', 'timed_out', 'startup_failure'].includes(status) ? 'failure' : status === 'success' ? 'success' : live(status) ? 'running' : ['skipped', 'cancelled'].includes(status) ? status : 'unknown';
  const svgNode = tag => document.createElementNS('http://www.w3.org/2000/svg', tag);
  function statusIcon(status) {
    const kind = statusClass(status);
    const label = kind === 'unknown' ? 'Job status unavailable; test results were recorded' : status.replaceAll('_', ' ');
    const icon = svgNode('svg');
    icon.classList.add('status-icon');
    icon.setAttribute('viewBox', '0 0 16 16');
    icon.setAttribute('role', 'img');
    icon.setAttribute('aria-label', label);
    const title = svgNode('title'); title.textContent = label; icon.append(title);
    const circle = svgNode('circle');
    circle.setAttribute('cx', '8'); circle.setAttribute('cy', '8'); circle.setAttribute('r', '7');
    circle.setAttribute('fill', kind === 'success' ? '#3fb950' : kind === 'failure' ? '#f85149' : '#6e7681');
    icon.append(circle);
    if (kind === 'unknown') {
      const question = svgNode('text');
      question.setAttribute('x', '8'); question.setAttribute('y', '11.3'); question.setAttribute('text-anchor', 'middle');
      question.setAttribute('fill', '#fff'); question.setAttribute('font-size', '10'); question.setAttribute('font-weight', '700');
      question.textContent = '?'; icon.append(question);
    } else {
      const mark = svgNode('path');
      mark.setAttribute('d', kind === 'success' ? 'M4.5 8.3l2.1 2.1 4.9-5.1' : kind === 'failure' ? 'M5.4 5.4l5.2 5.2M10.6 5.4l-5.2 5.2' : 'M4.5 8h7');
      mark.setAttribute('fill', 'none'); mark.setAttribute('stroke', '#fff'); mark.setAttribute('stroke-width', '1.8');
      mark.setAttribute('stroke-linecap', 'round'); mark.setAttribute('stroke-linejoin', 'round'); icon.append(mark);
    }
    return icon;
  }
  const laneName = (lane, hardware) => lane.id === 'cpu' ? 'CPU' : lane.id === 'amd' ? (hardware === 'amd-2gpu' ? 'AMD xGPU' : 'AMD GPU') : (hardware === 'multi-gpu' ? 'Nvidia xGPU' : 'Nvidia GPU');
  function renderRuns() {
    runs.replaceChildren();
    for (const lane of snapshot.lanes) {
      const card = node('div', 'run');
      card.append(node('strong', '', lane.label));
      if (lane.run) {
        card.append(link(runLink(lane.run), `Run ${lane.run.run_id}`));
        const status = lane.run.status === 'completed' ? lane.run.conclusion : lane.run.status;
        if (live(status)) card.append(node('span', 'live'));
        card.append(node('small', status === 'success' ? 'good' : statusClass(status) === 'failure' ? 'bad' : 'busy', status.replaceAll('_', ' ')));
      } else card.append(node('small', '', 'No run'));
      runs.append(card);
    }
  }
  function testLink(lane, test) {
    if (!/^[0-9a-f]{32}$/.test(test.trace_id) || !lane.run) return null;
    const query = new URLSearchParams({orgId: '1', 'var-trace_id': test.trace_id,
      'var-test_nodeid': test.nodeid, 'var-run_id': lane.run.run_id, 'var-pr': pr,
      'var-gh_run_id': String(lane.run.run_id).split(':')[0]});
    return link(`/d/pytest-test/test?${query}`, test.nodeid);
  }
  function visibleFailures() {
    const listed = [];
    for (const lane of snapshot.lanes) {
      const filterable = lane.id !== 'cpu' && lane.comparison?.state === 'complete';
      for (const test of lane.failures) {
        if (onlySpecific.checked && filterable && test.specific !== true) continue;
        const text = `${test.nodeid} ${test.job} ${test.model || ''} ${test.exception_type || ''} ${test.exception_message || ''} ${lane.label}`.toLowerCase();
        if (search.value.trim().toLowerCase().split(/\s+/).filter(Boolean).some(term => !text.includes(term))) continue;
        listed.push({lane, test});
      }
    }
    return listed;
  }
  // Same summary as the run table's groups: the three commonest error types.
  function errorTypes(members) {
    const counts = new Map();
    for (const {test} of members) {
      if (test.exception_type) counts.set(test.exception_type, (counts.get(test.exception_type) || 0) + 1);
    }
    const sorted = [...counts].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
    if (!sorted.length) return '';
    const shown = sorted.slice(0, 3).map(([type, count]) => count > 1 ? `${type} ×${count}` : type);
    if (sorted.length > 3) shown.push(`+${sorted.length - 3} more`);
    return ' · ' + shown.join(', ');
  }
  function groupKey(item) {
    const {test} = item;
    if (group === 'test') return test.test_name || test.nodeid.split('::').at(-1).replace(/\[.*\]$/, '');
    if (group === 'model') return test.model || '';
    if (group === 'test_model') return `${test.model || ''} · ${test.test_name || test.nodeid.split('::').at(-1)}`;
    return test.job;
  }
  function table(items) {
    const element = node('table', 'test-table');
    const cols = node('colgroup');
    for (const width of ['70px', '', '115px', '92px', '70px']) {
      const col = node('col');
      if (width) col.style.width = width;
      cols.append(col);
    }
    element.append(cols);
    const head = node('thead');
    const headings = node('tr');
    for (const label of ['Status', 'Test', 'Job', 'Lane', 'Duration']) headings.append(node('th', '', label));
    head.append(headings); element.append(head);
    const body = node('tbody');
    for (const {lane, test} of items) {
      const row = node('tr');
      row.append(node('td', 'bad', 'FAIL'));
      const title = node('td', 'nodeid');
      title.append(testLink(lane, test) || node('span', '', test.nodeid));
      if (test.specific === true) title.append(node('span', 'specific', 'PR-specific'));
      else if (test.specific === false) title.append(node('span', 'shared', 'Shared or pre-existing'));
      const error = [test.exception_type, test.exception_message].filter(Boolean).join(': ');
      if (error) title.append(node('div', 'exc', error));
      if (/^[0-9a-f]{32}$/.test(test.trace_id)) {
        const toggle = node('button', 'tb', 'error ▸');
        toggle.type = 'button';
        toggle.dataset.src = `/failure?${new URLSearchParams({trace_id:test.trace_id,test_nodeid:test.nodeid})}`;
        title.append(toggle);
      }
      row.append(title, node('td', '', test.job), node('td', '', laneName(lane, test.hardware)));
      const duration = Number(test.duration_seconds || 0);
      row.append(node('td', 'dur', duration ? `${duration.toFixed(3)}s` : '—'));
      body.append(row);
    }
    element.append(body);
    return element;
  }
  function renderFailures() {
    if (!snapshot) return;
    const opened = new Set([...failures.querySelectorAll('details[open]')].map(item => item.dataset.key));
    failures.replaceChildren();
    const listed = visibleFailures();
    if (!listed.length) {
      let message = search.value ? 'No matching failures.' : 'No failing tests.';
      if (!search.value) {
        const failedRunSlow = snapshot.lanes.find(lane => lane.id !== 'cpu' && lane.jobs.some(job => statusClass(job.status) === 'failure') && !lane.failures.length);
        if (failedRunSlow) {
          const platform = failedRunSlow.id === 'amd' ? 'AMD' : 'Nvidia';
          const state = failedRunSlow.comparison?.state;
          message = state === 'pending' ? `${platform} jobs failed; test attribution is pending.`
            : state === 'complete' ? `${platform} jobs failed; no PR-specific failing tests were reported.`
            : `${platform} jobs failed; test attribution is unavailable.`;
        }
      }
      failures.append(node('div', 'notice', message));
      return;
    }
    failures.append(node('div', 'summary', `${listed.length} failing test${listed.length === 1 ? '' : 's'}${group === 'none' ? ' · longest first' : ''}`));
    if (group === 'none') {
      listed.sort((a, b) => Number(b.test.duration_seconds || 0) - Number(a.test.duration_seconds || 0));
      failures.append(table(listed));
      return;
    }
    const groups = new Map();
    for (const item of listed) {
      const key = groupKey(item) || '(unknown)';
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(item);
    }
    for (const [key, members] of [...groups].sort((a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]))) {
      const details = node('details');
      details.dataset.key = key;
      details.open = opened.has(key) || (!opened.size && members.length <= 5);
      const summary = node('summary', '', key);
      summary.append(node('span', 'meta', ` — ${members.length} test${members.length === 1 ? '' : 's'}${errorTypes(members)}`));
      details.append(summary, table(members.sort((a, b) => a.test.nodeid.localeCompare(b.test.nodeid))));
      failures.append(details);
    }
  }
  function renderJobs() {
    jobs.replaceChildren();
    const ordered = snapshot.lanes.flatMap(lane => lane.jobs.map(job => ({lane, job})));
    const rank = status => live(status) ? 0 : statusClass(status) === 'failure' ? 1 : status === 'success' ? 2 : 3;
    ordered.sort((a, b) => rank(a.job.status) - rank(b.job.status) || (b.job.fail || 0) - (a.job.fail || 0) || a.job.name.localeCompare(b.job.name));
    for (const {lane, job} of ordered) {
      const tr = node('tr');
      tr.append(node('td', '', laneName(lane, job.hardware)));
      const name = node('td');
      name.append(lane.run ? link(jobLink(lane, job), job.name) : node('span', '', job.name));
      tr.append(name);
      const status = node('td', 'status-cell');
      if (live(job.status)) {
        status.append(node('span', 'live'));
        status.append(node('span', 'running-text', job.status === 'in_progress' ? 'running' : job.status));
      }
      else status.append(statusIcon(job.status));
      tr.append(status, node('td', 'dur', job.pass ?? '—'), node('td', 'dur', job.fail ?? '—'));
      jobs.append(tr);
    }
    jobsEmpty.hidden = ordered.length > 0;
    jobsEmpty.textContent = ordered.length ? '' : 'No jobs found in the latest available runs.';
  }
  function render() { renderRuns(); renderFailures(); renderJobs(); }
  async function refresh() {
    if (loading || document.hidden) return;
    if (!/^[1-9][0-9]*$/.test(pr || '')) { runs.textContent = 'Open this view from a PR dashboard.'; progress.hidden = true; return; }
    loading = true; progress.hidden = false;
    try {
      const response = await fetch(`/pr-ci-view/data?pr=${encodeURIComponent(pr)}`, {credentials: 'same-origin', cache: 'no-store'});
      if (!response.ok || !response.headers.get('content-type')?.includes('application/json')) {
        throw new Error(`CI results are temporarily unavailable (HTTP ${response.status}). Retrying shortly…`);
      }
      const data = await response.json();
      snapshot = data; render();
    } catch (error) { runs.textContent = error.message || 'CI results are temporarily unavailable. Retrying shortly…'; }
    finally { loading = false; progress.hidden = true; }
  }
  onlySpecific.addEventListener('change', () => {
    try { localStorage.setItem('tci-pr-specific-v2', String(onlySpecific.checked)); } catch (_) {}
    renderFailures();
  });
  search.addEventListener('input', renderFailures);
  for (const button of groupButtons) button.addEventListener('click', () => {
    group = button.dataset.group;
    for (const candidate of groupButtons) candidate.setAttribute('aria-pressed', String(candidate === button));
    renderFailures();
  });
  groupButtons[0].setAttribute('aria-pressed', 'true');
  failures.addEventListener('click', event => {
    const button = event.target.closest('button.tb');
    if (!button) return;
    const row = button.closest('tr');
    const next = row.nextElementSibling;
    if (next?.classList.contains('tbrow')) { next.remove(); button.textContent = 'error ▸'; return; }
    const extra = node('tr', 'tbrow');
    const cell = node('td'); cell.colSpan = 5;
    const frame = node('iframe'); frame.title = 'Failure details'; frame.src = button.dataset.src;
    frame.addEventListener('load', () => { try { frame.style.height = `${Math.min(600, frame.contentDocument.documentElement.scrollHeight + 4)}px`; } catch (_) {} });
    cell.append(frame); extra.append(cell); row.after(extra); button.textContent = 'error ▾';
  });
  refresh();
  setInterval(refresh, 30000);
})();
