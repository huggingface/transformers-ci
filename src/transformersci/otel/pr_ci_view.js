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
  // Jobs picked in the Latest Run Jobs table; empty = every job.
  const pickedJobs = new Set();
  const jobKey = (lane, name, hardware) => `${lane.id}|${name}|${hardware || ''}`;
  const groupButtons = [...document.querySelectorAll('[data-group]')];
  let snapshot = null, snapshotText = '', loading = false;
  // Merge-queue history per failing node id; null until loaded, {} when unavailable.
  let history = null, historyIncomplete = false, historyKey = '', historyAt = 0;
  const expanded = new Set();
  const foldAll = document.getElementById('fold-all');
  // One-shot override for group <details> on the next render: true = open all, false = close all.
  let groupsForced = null;
  let group = 'none';
  try {
    const body = parent.document.body;
    if (/light/.test(body.className) || parent.grafanaBootData?.user?.lightTheme) document.body.classList.add('light');
  } catch (_) {}
  try {
    const saved = localStorage.getItem('tci-pr-hide-nonspecific-v1');
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
  const VERDICTS = {
    pr: ['v-pr', 'Likely this PR'],
    flaky: ['v-flaky', 'Flaky — not this PR'],
    main: ['v-main', 'Broken on main'],
    others: ['v-main', 'Fails for other PRs'],
    unknown: ['v-unknown', 'Not enough history'],
    none: ['v-none', 'No queue history'],
  };
  const VERDICT_RANK = {pr: 0, unknown: 1, none: 2, flaky: 3, others: 4, main: 5};
  const queueOf = test => (test.hardware === 'cpu' && history?.[test.nodeid]) || null;
  const verdictOf = test => queueOf(test)?.verdict || null;
  const age = seconds => seconds < 3600 ? `${Math.max(1, Math.round(seconds / 60))} min ago`
    : seconds < 172800 ? `${Math.round(seconds / 3600)} h ago` : `${Math.round(seconds / 86400)} d ago`;
  const runResult = run => run.status === 'passed' ? 'passed'
    : run.status === 'rerun' ? `passed after a rerun${run.exception_type ? ` (failed once: ${run.exception_type})` : ''}`
    : `failed${run.exception_type ? `: ${run.exception_type}` : ''}${run.reported_ok ? ' (final result unknown, before tci#275)' : ''}`;
  const runTitle = run => `#${run.pr} · ${runResult(run)} · ${age(run.age_seconds)}`;
  function strip(queue, width) {
    const element = node('div', 'mq');
    const runs = queue.runs.slice(0, width).reverse();
    for (let slot = 0; slot < width - runs.length; slot++) {
      const empty = node('i', 'x'); empty.title = 'No run'; element.append(empty);
    }
    for (const run of runs) {
      const square = node('i', run.status === 'passed' ? 'p' : run.status === 'rerun' ? 'a' : run.reported_ok ? 'r' : 'f');
      square.title = runTitle(run);
      element.append(square);
    }
    return element;
  }
  function rateText(queue) {
    if (!queue.ran_count) return 'not run in the queue';
    if (!queue.fail_count) return `0 / ${queue.ran_count} · none failed`;
    return `${queue.fail_count} / ${queue.ran_count} · ${queue.distinct_prs_failed} PR${queue.distinct_prs_failed === 1 ? '' : 's'}`;
  }
  // CPU: a chip. GPU: a card, two for multi-GPU runners; vendor colour, full name on hover.
  function laneIcon(lane, hardware) {
    const label = laneName(lane, hardware);
    const wrap = node('span', `lane-icon ${lane.id === 'cpu' ? 'cpu' : lane.id === 'amd' ? 'amd' : 'nvidia'}`);
    wrap.title = label; wrap.setAttribute('role', 'img'); wrap.setAttribute('aria-label', label);
    const multi = hardware === 'multi-gpu' || hardware === 'amd-2gpu';
    for (let index = 0; index < (lane.id !== 'cpu' && multi ? 2 : 1); index++) {
      const icon = svgNode('svg');
      icon.setAttribute('viewBox', '0 0 24 24'); icon.setAttribute('fill', 'none');
      icon.setAttribute('stroke', 'currentColor'); icon.setAttribute('stroke-width', '1.7');
      icon.setAttribute('stroke-linecap', 'round'); icon.setAttribute('aria-hidden', 'true');
      const path = svgNode('path');
      path.setAttribute('d', lane.id === 'cpu'
        ? 'M7 7h10v10H7zM10 10h4v4h-4zM10 3v4M14 3v4M10 17v4M14 17v4M3 10h4M3 14h4M17 10h4M17 14h4'
        : 'M3 7h18v10H3zM8 12m-2.5 0a2.5 2.5 0 1 0 5 0a2.5 2.5 0 1 0-5 0M14 10h4M14 14h4M5 17v3M9 17v3');
      icon.append(path); wrap.append(icon);
    }
    return wrap;
  }
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
  // Why a failure is hidden, or null. Every hidden failure is counted in the hidden note.
  function hiddenReason(lane, test) {
    const verdict = verdictOf(test);
    if (!onlySpecific.checked) return null;
    if (lane.id !== 'cpu' && lane.comparison?.state === 'complete' && test.specific !== true) return 'shared';
    if (['main', 'others', 'flaky'].includes(verdict)) return verdict;
    return null;
  }
  function visibleFailures() {
    const listed = [], hidden = [];
    for (const lane of snapshot.lanes) {
      for (const test of lane.failures) {
        if (pickedJobs.size && !pickedJobs.has(jobKey(lane, test.job, test.hardware))) continue;
        const text = `${test.nodeid} ${test.job} ${test.model || ''} ${test.exception_type || ''} ${test.exception_message || ''} ${lane.label}`.toLowerCase();
        if (search.value.trim().toLowerCase().split(/\s+/).filter(Boolean).some(term => !text.includes(term))) continue;
        const reason = hiddenReason(lane, test);
        if (reason) hidden.push({lane, test, reason});
        else listed.push({lane, test});
      }
    }
    return {listed, hidden};
  }
  const shortName = nodeid => nodeid.split('/').at(-1);
  function hiddenNote(hidden) {
    const note = node('div', 'hidden-note');
    const labels = {shared: 'run-slow not specific to this PR', flaky: 'flaky', main: 'broken on main', others: 'failing for other PRs too'};
    const counts = ['main', 'others', 'shared', 'flaky'].map(kind => [kind, hidden.filter(item => item.reason === kind)]).filter(([, items]) => items.length);
    note.append('Hidden: ');
    counts.forEach(([kind, items], index) => {
      if (index) note.append(', ');
      note.append(node('b', '', `${items.length} ${labels[kind]}`));
    });
    const names = hidden.slice(0, 3).map(({test}) => shortName(test.nodeid));
    const types = [...new Set(hidden.map(({test}) => test.exception_type).filter(Boolean))];
    note.append(` — ${names.join(', ')}${hidden.length > 3 ? ` +${hidden.length - 3} more` : ''}${types.length ? ` (${types.join(' / ')})` : ''}`);
    const queued = hidden.filter(item => item.reason !== 'shared').map(({test}) => queueOf(test));
    if (queued.length) {
      const failed = Math.max(...queued.map(queue => queue.fail_count));
      const ran = Math.max(...queued.map(queue => queue.ran_count));
      const prs = Math.max(...queued.map(queue => queue.distinct_prs_failed));
      note.append(` · up to ${failed} of the last ${ran} merge-queue runs failed, across ${prs} other PR${prs === 1 ? '' : 's'}`);
    }
    note.append(' · ');
    const show = node('button', '', 'show them');
    show.type = 'button';
    show.addEventListener('click', () => { setToggle(onlySpecific, false); renderFailures(); });
    note.append(show);
    return note;
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
  const WINDOW = 20;
  function historyRow(queue) {
    const row = node('tr', 'xp');
    const cell = node('td'); cell.colSpan = 3;
    cell.append(node('div', 'why', queue.reason));
    const failedRuns = queue.runs.filter(run => run.status !== 'passed');
    const passedRuns = queue.runs.filter(run => run.status === 'passed');
    if (failedRuns.length) {
      const list = node('table');
      for (const run of failedRuns.slice(0, 8)) {
        const line = node('tr');
        const who = node('td');
        const square = node('i', 'sq');
        square.style.background = run.status === 'passed' ? 'var(--green)' : run.status === 'rerun' ? 'var(--amber)' : 'var(--red)';
        who.append(square, link(`https://github.com/huggingface/transformers/pull/${run.pr}`, `#${run.pr}`));
        const result = node('td');
        if (run.status === 'passed') result.append(node('span', 'age', 'passed'));
        else {
          result.append(node('span', run.status === 'rerun' ? 'busy' : 'bad', [run.exception_type, run.exception_message].filter(Boolean).join(': ').slice(0, 160) || 'failed'));
          if (run.status === 'rerun') result.append(node('span', 'note', ' — passed after a rerun'));
          else if (run.reported_ok) result.append(node('span', 'note', ' — final result unknown (before tci#275)'));
        }
        const log = node('td'); log.append(link(run.log_url, 'log'));
        line.append(who, result, node('td', 'age', age(run.age_seconds)), log);
        list.append(line);
      }
      cell.append(list);
      if (failedRuns.length > 8) cell.append(node('div', 'meta-line', `+${failedRuns.length - 8} older failing runs in the strip`));
    }
    if (passedRuns.length) {
      const prs = [...new Set(passedRuns.map(run => run.pr))];
      const line = node('div', 'meta-line', `Passed in ${passedRuns.length} other run${passedRuns.length === 1 ? '' : 's'}, from ${prs.length} PR${prs.length === 1 ? '' : 's'}: `);
      prs.slice(0, 10).forEach((number, index) => {
        if (index) line.append(', ');
        line.append(link(`https://github.com/huggingface/transformers/pull/${number}`, `#${number}`));
      });
      if (prs.length > 10) line.append(` +${prs.length - 10} more`);
      cell.append(line);
    }
    row.append(cell);
    return row;
  }
  function table(items) {
    const element = node('table', 'test-table');
    const cols = node('colgroup');
    for (const width of ['', '200px', '150px']) {
      const col = node('col');
      if (width) col.style.width = width;
      cols.append(col);
    }
    element.append(cols);
    const head = node('thead');
    const headings = node('tr');
    for (const label of ['Test', `Merge queue · last ${WINDOW}`, 'Verdict']) headings.append(node('th', '', label));
    head.append(headings); element.append(head);
    const body = node('tbody');
    for (const {lane, test} of items) {
      const row = node('tr');
      row.dataset.nodeid = test.nodeid;
      const title = node('td', 'nodeid');
      const label = testLink(lane, test) || node('span', '', test.nodeid);
      label.textContent = '';
      test.nodeid.split('::').forEach((part, index) => {
        if (index) label.append('::', document.createElement('wbr'));
        label.append(part);
      });
      title.append(label);
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
      const duration = Number(test.duration_seconds || 0);
      title.append(node('div', 'meta-line', [test.job, laneName(lane, test.hardware), duration ? `${duration.toFixed(2)}s` : ''].filter(Boolean).join(' · ')));
      row.append(title);
      const queue = queueOf(test);
      const mq = node('td', 'mqcell'), verdict = node('td');
      if (lane.id !== 'cpu') {
        mq.append(node('span', 'meta-line', 'GPU · not run in the queue'));
        mq.title = 'Merge-queue runs are CPU only; this GPU test never runs there.';
      } else if (!queue) {
        mq.append(node('span', 'meta-line', history === null ? 'loading…' : '—'));
      } else if (!queue.fail_count) {
        mq.append(node('div', 'rate good', queue.ran_count ? '✓ passed' : 'not run in the queue'));
      } else {
        mq.append(strip(queue, WINDOW));
        mq.append(node('div', 'rate bad', rateText(queue)));
      }
      if (queue?.ran_count && !queue.fail_count) mq.title ||= `Passed in all ${queue.ran_count} recent merge-queue runs of other PRs`;
      if (queue && lane.id === 'cpu') {
        const [cls, text] = VERDICTS[queue.verdict] || VERDICTS.unknown;
        const pill = node('button', `verdict ${cls}`, text);
        pill.type = 'button'; pill.title = queue.reason || '';
        verdict.append(pill);
        mq.dataset.expand = verdict.dataset.expand = 'true';
      }
      row.append(mq, verdict);
      body.append(row);
      if (queue && lane.id === 'cpu' && expanded.has(test.nodeid)) body.append(historyRow(queue));
    }
    element.append(body);
    return element;
  }
  // Rows that can unfold: tests with merge-queue history, plus any groups.
  const foldable = () => visibleFailures().listed.filter(({lane, test}) => lane.id === 'cpu' && queueOf(test));
  function anyUnfolded() {
    return expanded.size > 0 || failures.querySelector('details[open], tr.tbrow') !== null;
  }
  function updateFoldAll() {
    const items = snapshot ? foldable() : [];
    foldAll.hidden = !items.length && !failures.querySelector('details');
    const fold = anyUnfolded();
    const label = fold ? 'Collapse all' : 'Expand all';
    foldAll.title = label; foldAll.setAttribute('aria-label', label);
    foldAll.replaceChildren();
    const icon = svgNode('svg');
    icon.setAttribute('viewBox', '0 0 16 16'); icon.setAttribute('fill', 'none'); icon.setAttribute('stroke', 'currentColor');
    icon.setAttribute('stroke-width', '1.6'); icon.setAttribute('stroke-linecap', 'round'); icon.setAttribute('stroke-linejoin', 'round');
    icon.setAttribute('aria-hidden', 'true');
    const path = svgNode('path');
    // Collapse: chevrons pointing together. Expand: chevrons pointing apart.
    path.setAttribute('d', fold ? 'M4.5 2.5L8 6l3.5-3.5M4.5 13.5L8 10l3.5 3.5' : 'M4.5 6L8 2.5 11.5 6M4.5 10L8 13.5 11.5 10');
    icon.append(path); foldAll.append(icon);
  }
  foldAll.addEventListener('click', () => {
    if (anyUnfolded()) { expanded.clear(); groupsForced = false; }
    else { for (const {test} of foldable()) expanded.add(test.nodeid); groupsForced = true; }
    renderFailures();
  });
  function renderFailures() {
    renderFailureList();
    groupsForced = null;
    updateFoldAll();
  }
  function renderFailureList() {
    if (!snapshot) return;
    const opened = new Set([...failures.querySelectorAll('details[open]')].map(item => item.dataset.key));
    failures.replaceChildren();
    const {listed, hidden} = visibleFailures();
    if (pickedJobs.size) {
      const banner = node('div', 'jobfilter', 'Only tests from ');
      [...pickedJobs].forEach((key, index) => {
        const [laneId, name, hardware] = key.split('|');
        if (index) banner.append(', ');
        banner.append(node('b', '', laneId === 'cpu' ? name : `${name} (${laneName({id: laneId}, hardware)})`));
      });
      banner.append(' · ');
      const clear = node('button', '', 'all jobs');
      clear.type = 'button';
      clear.addEventListener('click', () => { pickedJobs.clear(); renderJobs(); renderFailures(); });
      banner.append(clear);
      failures.append(banner);
    }
    if (!listed.length && !hidden.length) {
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
    failures.append(node('div', 'summary', `${listed.length} failing test${listed.length === 1 ? '' : 's'}${hidden.length ? ` · ${hidden.length} hidden as not this PR` : ''}${group === 'none' ? " · this PR's likely failures first" : ''}${historyIncomplete ? ' · merge-queue history may be incomplete' : ''}`));
    if (hidden.length) failures.append(hiddenNote(hidden));
    if (!listed.length) return;
    if (group === 'none') {
      const rank = item => VERDICT_RANK[verdictOf(item.test)] ?? 1;
      listed.sort((a, b) => rank(a) - rank(b) || Number(b.test.duration_seconds || 0) - Number(a.test.duration_seconds || 0));
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
      details.open = groupsForced ?? (opened.has(key) || (!opened.size && members.length <= 5));
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
    const failingJobs = new Set(snapshot.lanes.flatMap(lane => lane.failures.map(test => jobKey(lane, test.job, test.hardware))));
    for (const {lane, job} of ordered) {
      const tr = node('tr');
      const key = jobKey(lane, job.name, job.hardware);
      const pick = node('td');
      if (failingJobs.has(key)) {
        const box = node('input');
        box.type = 'checkbox'; box.checked = pickedJobs.has(key);
        box.setAttribute('aria-label', `Show only failing tests from ${job.name}`);
        box.title = 'Show only this job\'s failing tests';
        box.addEventListener('change', () => { if (box.checked) pickedJobs.add(key); else pickedJobs.delete(key); renderJobs(); renderFailures(); });
        pick.append(box);
      }
      if (pickedJobs.has(key)) tr.classList.add('picked');
      tr.append(pick);
      const name = node('td');
      name.append(lane.run ? link(jobLink(lane, job), job.name) : node('span', '', job.name));
      const status = node('td', 'status-cell');
      if (live(job.status)) {
        status.append(node('span', 'live'));
        status.append(node('span', 'running-text', job.status === 'in_progress' ? 'running' : job.status));
      }
      else status.append(statusIcon(job.status));
      const laneCell = node('td');
      laneCell.append(laneIcon(lane, job.hardware));
      tr.append(status, name, node('td', 'dur', job.pass ?? '—'), node('td', 'dur', job.fail ?? '—'), laneCell);
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
      const text = await response.text();
      // An unchanged refresh must not re-render: that would close open tracebacks and history rows.
      if (text !== snapshotText) { snapshotText = text; snapshot = JSON.parse(text); render(); }
      loadHistory();
    } catch (error) { runs.textContent = error.message || 'CI results are temporarily unavailable. Retrying shortly…'; }
    finally { loading = false; progress.hidden = true; }
  }
  const HISTORY_MAX_TESTS = 60, HISTORY_REFRESH_MS = 300000;
  async function loadHistory() {
    const nodeids = [...new Set(snapshot.lanes.filter(lane => lane.id === 'cpu')
      .flatMap(lane => lane.failures.map(test => test.nodeid)))].sort().slice(0, HISTORY_MAX_TESTS);
    const key = nodeids.join('\n');
    if (!nodeids.length) { history = {}; historyKey = key; renderFailures(); return; }
    if (key === historyKey && Date.now() - historyAt < HISTORY_REFRESH_MS) return;
    const query = new URLSearchParams({pr});
    for (const nodeid of nodeids) query.append('t', nodeid);
    try {
      const response = await fetch(`/pr-ci-view/queue-history?${query}`, {credentials: 'same-origin', cache: 'no-store'});
      const data = response.ok ? await response.json() : {};
      historyKey = key; historyAt = Date.now();
      const same = history !== null && JSON.stringify(data.tests || {}) === JSON.stringify(history) && Boolean(data.incomplete) === historyIncomplete;
      history = data.tests || {};
      historyIncomplete = Boolean(data.incomplete);
      if (same) return;
    } catch (_) { if (history !== null) return; history = {}; }
    renderFailures();
  }
  function setToggle(box, value) {
    box.checked = value;
    try { localStorage.setItem('tci-pr-hide-nonspecific-v1', String(value)); } catch (_) {}
  }
  onlySpecific.addEventListener('change', () => { setToggle(onlySpecific, onlySpecific.checked); renderFailures(); });
  search.addEventListener('input', renderFailures);
  for (const button of groupButtons) button.addEventListener('click', () => {
    group = button.dataset.group;
    for (const candidate of groupButtons) candidate.setAttribute('aria-pressed', String(candidate === button));
    renderFailures();
  });
  groupButtons[0].setAttribute('aria-pressed', 'true');
  failures.addEventListener('toggle', updateFoldAll, true);
  failures.addEventListener('click', event => {
    const expander = event.target.closest('[data-expand]');
    if (expander && !event.target.closest('a')) {
      const nodeid = expander.closest('tr').dataset.nodeid;
      if (expanded.has(nodeid)) expanded.delete(nodeid); else expanded.add(nodeid);
      renderFailures();
      return;
    }
    const button = event.target.closest('button.tb');
    if (!button) return;
    const row = button.closest('tr');
    const next = row.nextElementSibling;
    if (next?.classList.contains('tbrow')) { next.remove(); button.textContent = 'error ▸'; updateFoldAll(); return; }
    const extra = node('tr', 'tbrow');
    const cell = node('td'); cell.colSpan = 3;
    // The traceback page can take seconds; show progress instead of an empty box.
    const waiting = node('div', 'tb-loading');
    const bar = node('div', 'progress'); bar.append(node('span'));
    waiting.append(bar, node('div', 'tb-loading-text', 'Loading traceback…'));
    const frame = node('iframe'); frame.title = 'Failure details'; frame.hidden = true;
    frame.addEventListener('load', () => {
      waiting.remove(); frame.hidden = false;
      try { frame.style.height = `${Math.min(600, frame.contentDocument.documentElement.scrollHeight + 4)}px`; } catch (_) {}
    });
    frame.src = button.dataset.src;
    cell.append(waiting, frame); extra.append(cell); row.after(extra); button.textContent = 'error ▾';
    updateFoldAll();
  });
  refresh();
  setInterval(refresh, 30000);
})();
