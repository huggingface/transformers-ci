(() => {
  'use strict';
  const CATEGORIES = [['failure', 'Failure'], ['slow', 'Slow']];
  // The Test page's Serge Actions (button id in /serge-actions, label).
  // One action: a New issue draft that carries Serge's WDYT? answer and lists
  // related issues and PRs instead of people.
  const ACTIONS = [['work-on-it', 'Work on it', "Draft a GitHub issue with Serge's take and the related issues and PRs"]];
  const DAY = 86400;
  const $ = id => document.getElementById(id);
  const {esc} = tci;
  const params = new URLSearchParams(location.search);
  const state = {
    days: [7, 14, 30].includes(Number(params.get('days'))) ? Number(params.get('days')) : 14,
    cat: 'all', runner: 'all', sticky: true, q: '', group: 'model', sort: 'fails', unclaimed: true, mine: false, fix: false,
  };
  let rows = [], generatedAt = 0, claims = {}, me = '', fixIndexAt = 0;

  tci.applyTheme();

  function ago(ts) {
    const seconds = Math.max(0, (generatedAt || Date.now() / 1000) - ts);
    for (const [suffix, size] of [['d', DAY], ['h', 3600], ['m', 60]]) {
      if (seconds >= size) return `${Math.floor(seconds / size)}${suffix} ago`;
    }
    return 'just now';
  }
  // Slow timings are per UTC day, so their "seen" is a day, not a time.
  function daysAgo(ts) {
    const days = Math.floor((generatedAt || Date.now() / 1000) / DAY) - Math.floor(ts / DAY);
    return days <= 0 ? 'today' : days === 1 ? 'yesterday' : `${days}d ago`;
  }
  const dayLabel = ts => new Date(ts * 1000).toLocaleDateString('en', {month: 'short', day: 'numeric', timeZone: 'UTC'});

  async function getJson(path) {
    const response = await fetch(`${path}?days=${state.days}`, {credentials: 'same-origin'});
    if (!response.ok) throw new Error(`${response.status}`);
    return response.json();
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
      && (!state.unclaimed || ['free', 'closed'].includes(claimState(row)))
      && (!state.mine || claims[row.nodeid]?.login === me)
      && (!state.fix || hasProbableFix(row))
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
    error: row => row.category === 'slow' ? 'Slow (no error)' : row.signal,
  };
  // Sort toggle: most failing runs first, or alphabetical by name.
  const byName = (a, b) => `${a.model} ${a.test}`.localeCompare(`${b.model} ${b.test}`) || a.job.localeCompare(b.job);
  // Slow rows: longest first.
  const longer = (a, b) => a.category === 'slow' && b.category === 'slow' ? b.seconds - a.seconds : 0;
  const byFails = (a, b) => longer(a, b) || b.fails - a.fails || b.bad_days - a.bad_days || b.last_seen - a.last_seen || byName(a, b);
  const minutes = value => `${(value / 60).toFixed(1)} min`;
  const sortRows = list => [...list].sort(state.sort === 'alpha' ? byName : byFails);
  const openGroups = new Set();
  let groupsSeen = '', groupNames = [];

  // A closed issue is only "closed": it is not proof of a fix. Its row offers
  // Reopen instead of New issue, and says if the test failed after the close.
  function claimState(row) {
    const claim = claims[row.nodeid];
    if (!claim) return 'free';
    if (claim.state !== 'claimed') return 'drafting';
    return claim.issue_state === 'closed' ? 'closed' : 'on-it';
  }

  // A PR merged before the last failure did not fix it: not a probable fix.
  const hasProbableFix = row => Boolean(row.fix) && row.fix.verdict !== 'merged_before';

  // The probable fix, judged by when the test last failed: a PR merged before
  // that did not fix it (the Test page cannot tell, the table can).
  function fixHtml(row) {
    const fix = row.fix;
    if (!fix && !fixIndexAt) return '<span class="sub" title="Looking up recent PRs: about a minute after an update">looking up…</span>';
    if (!fix) return '<span class="sub">—</span>';
    const link = `<a href="${esc(fix.url)}" target="_blank" rel="noopener noreferrer" title="${esc(fix.title)}">#${fix.number}</a>`;
    const more = fix.others ? ` <span class="sub" title="${fix.others} more PR${fix.others === 1 ? '' : 's'} change this test file and name the model">+${fix.others}</span>` : '';
    if (fix.verdict === 'open') return `${link} <span class="open">open</span>${more}`;
    if (fix.verdict === 'merged_after') return `${link} <span class="ok" title="Merged ${ago(fix.merged_at)}, and the test has not failed since">merged, no failure since</span>${more}`;
    return `${link} <span class="sub" title="Merged ${ago(fix.merged_at)}, but the test failed after that">merged, still failing</span>${more}`;
  }

  function claimHtml(row) {
    const claim = claims[row.nodeid], state = claimState(row);
    if (state === 'free') return '<span class="sub">—</span>';
    const who = `<a href="https://github.com/${encodeURIComponent(claim.login)}" target="_blank" rel="noopener noreferrer">@${esc(claim.login)}</a>`;
    const issue = claim.url ? `<a href="${esc(claim.url)}" target="_blank" rel="noopener noreferrer">#${claim.number}</a>` : '';
    if (state === 'drafting') return `${who} <span class="drafting">is drafting…</span>`;
    if (state === 'on-it') return `${who} is on it · ${issue}`;
    const since = claim.closed_at && row.last_seen > claim.closed_at
      ? ' · <span class="again" title="The test failed after the issue was closed">failed since</span>' : '';
    return `${issue} <span class="closed">closed</span>${since} <span class="sub">· was ${who}</span>`;
  }

  // New issue is off while someone has the test: its issue is the place to talk.
  // Your own unpublished draft stays reachable, to reopen it.
  function newIssueBlocked(row) {
    const claim = claims[row.nodeid], state = claimState(row);
    if (state === 'on-it') return `@${claim.login} is on it in #${claim.number}`;
    if (state === 'drafting' && claim.login !== me) return `@${claim.login} is drafting an issue for this test`;
    return '';
  }

  function actionButton(id, label, help, k, index, row) {
    let blocked = newIssueBlocked(row);
    if (claimState(row) === 'closed') {
      // Same button: the dialog finds the closed issue and offers to reopen it.
      const claim = claims[row.nodeid];
      label = 'Reopen';
      help = `Reopen #${claim.number}, closed, instead of opening a second issue`;
    }
    if (row.category === 'slow' && !row.nodeid.startsWith('tests/')) blocked = 'Not a pytest test';
    if (blocked) return `<button type="button" class="act" disabled title="${esc(blocked)}">${label}</button>`;
    return `<button type="button" class="act${k ? '' : ' pri'}" data-act="${id}" data-i="${index}" title="${help}">${label}</button>`;
  }

  function rowHtml(row, index) {
    const slow = row.category === 'slow';
    const fails = slow ? `slowest ${row.fails} of ${row.days.length} days` : `${row.fails} fail${row.fails === 1 ? '' : 's'}`;
    const cell = slow ? 's' : 'f';
    const cellTitle = (count, i) => slow
      ? `${dayLabel(rowsFirstDay + i * DAY)}: ${row.durations[i] == null ? 'not among the slowest' : minutes(row.durations[i])}`
      : `${dayLabel(rowsFirstDay + i * DAY)}: ${count} failing run${count === 1 ? '' : 's'}`;
    return `<tr>
      <td class="test"><div class="tags"><span class="badge k-${row.category}">${row.category.toUpperCase()}</span><span class="rn">${row.runner}</span>${row.sticky ? '<span class="sticky">sticky</span>' : ''}<b>${esc(row.model)}</b><span class="job">${esc(row.job)}</span></div>
        <div class="nodeid">${esc(row.test)}</div></td>
      <td class="sig${slow ? ' slow' : ''}"><span class="sigtext" title="${esc(row.signal)}">${esc(row.signal)}</span><button type="button" class="why" data-why="${index}" aria-label="${slow ? 'Timings' : 'Traceback'} for ${esc(row.test)}" title="${slow ? 'Show the timings' : 'Show the traceback'}">?</button></td>
      <td class="days"><span class="num">${fails}</span><span class="strip">${row.days.map((count, i) =>
        `<i class="${count ? cell : ''}" title="${cellTitle(count, i)}"></i>`).join('')}</span></td>
      <td title="${slow ? 'Day of its latest timing (UTC)' : new Date(row.last_seen * 1000).toISOString()}">${slow ? daysAgo(row.last_seen) : ago(row.last_seen)}</td>
      <td class="claim">${claimHtml(row)}</td>
      <td class="fix">${fixHtml(row)}</td>
      <td><span class="acts">${ACTIONS.map(([id, label, help], k) =>
        actionButton(id, label, help, k, index, row)).join('')}</span></td></tr>`;
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
    const slowest = members => members.every(([row]) => row.category === 'slow') ? Math.max(...members.map(([row]) => row.seconds)) : null;
    const longest = (a, b) => slowest(a) != null && slowest(b) != null ? slowest(b) - slowest(a) : 0;
    const ordered = [...groups].sort(state.sort === 'alpha'
      ? (a, b) => a[0].localeCompare(b[0])
      : (a, b) => longest(a[1], b[1]) || fails(b[1]) - fails(a[1]) || b[1].length - a[1].length || a[0].localeCompare(b[0]));
    // A new grouping starts with every group collapsed; after that the
    // viewer's own choices stick.
    const seen = `${state.group}|${state.days}`;
    if (seen !== groupsSeen) { groupsSeen = seen; openGroups.clear(); }
    groupNames = ordered.map(([name]) => name);
    packets = [];
    return ordered.map(([name, members]) => {
      const open = openGroups.has(name);
      const errors = new Map();
      for (const [row] of members) errors.set(row.signal, (errors.get(row.signal) || 0) + 1);
      const summary = state.group === 'error' ? '' : ` · ${[...errors].sort((a, b) => b[1] - a[1]).slice(0, 3)
        .map(([error, count]) => `${count}× ${esc(error)}`).join(', ')}${errors.size > 3 ? ', …' : ''}`;
      const sticky = members.filter(([row]) => row.sticky).length;
      return `<tr class="grp" data-group="${esc(name)}" aria-expanded="${open}"><td colspan="7"><span class="caret">${open ? '▾' : '▸'}</span>${esc(name)}
        <span class="meta"> — ${members.length} test${members.length === 1 ? '' : 's'}, ${fails(members)} failing run${fails(members) === 1 ? '' : 's'}${sticky ? `, ${sticky} sticky` : ''}${summary}</span></td></tr>`
        + (open ? packetHtml(name, members) + members.map(([row, index]) => rowHtml(row, index)).join('') : '');
    }).join('');
  }

  // Work on a batch: one issue for a group's unclaimed failures that fail the
  // same way, a bucket per error type and message key (the message with its
  // run-specific parts taken out, read from the traces when the group opens).
  // A bucket of one is that row's own Work on it.
  let packets = [];
  const PACKET_LIMIT = 60;  // new_issue.PACKET_LIMIT
  const messageKeys = {};  // nodeid -> message key, '' when it could not be read
  const keysPending = new Set();  // nodeids being read
  async function loadKeys(rows) {
    for (const row of rows) keysPending.add(row.nodeid);
    try {
      const response = await fetch('/project-health/packets', {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({tests: rows.map(row => ({trace_id: row.trace_id, test_nodeid: row.nodeid}))})});
      const data = response.ok ? await response.json() : {};
      for (const row of rows) messageKeys[row.nodeid] = data.keys?.[row.nodeid] ?? '';
    } catch (_) {
      for (const row of rows) messageKeys[row.nodeid] = '';
    }
    for (const row of rows) keysPending.delete(row.nodeid);
    drawRows();
  }
  function packetHtml(name, members) {
    const eligible = members.map(([row]) => row)
      .filter(row => row.category !== 'slow' && row.trace_id && claimState(row) === 'free');
    const unknown = eligible.filter(row => !(row.nodeid in messageKeys));
    const ask = unknown.filter(row => !keysPending.has(row.nodeid));
    if (ask.length) loadKeys(ask);
    if (unknown.length && eligible.length > 1)
      return '<tr class="pk"><td colspan="7"><span class="lbl">Work on a batch:</span><span class="sub">reading the errors…</span></td></tr>';
    const buckets = new Map();
    for (const row of eligible) {
      const key = messageKeys[row.nodeid] || '';
      const id = `${row.signal}\n${key}`;
      if (!buckets.has(id)) buckets.set(id, {signal: row.signal, key, rows: []});
      buckets.get(id).rows.push(row);
    }
    const buttons = [...buckets.values()].filter(bucket => bucket.rows.length > 1)
      .sort((a, b) => b.rows.length - a.rows.length || a.signal.localeCompare(b.signal))
      .map(({signal, key, rows}) => {
        const tests = rows.slice(0, PACKET_LIMIT);
        const error = key ? `${signal}: ${key}` : signal;
        packets.push({group: name, signal: error, tests: tests.map(row => ({trace_id: row.trace_id, test_nodeid: row.nodeid}))});
        const more = rows.length > tests.length ? ` (the first ${tests.length} of ${rows.length})` : '';
        const short = key.length > 48 ? `${key.slice(0, 47)}…` : key;
        return `<button type="button" class="act pri" data-packet="${packets.length - 1}" title="One issue for these ${tests.length} unclaimed tests${more}, all failing with ${esc(error)}">${rows.length} × ${esc(signal)}${short ? `: ${esc(short)}` : ''}</button>`;
      });
    return buttons.length ? `<tr class="pk"><td colspan="7"><span class="lbl">Work on a batch:</span>${buttons.join('')}</td></tr>` : '';
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
      || `<tr><td colspan="7" class="state">${rows.length ? 'Nothing matches these filters.' : 'No failures on main in this window.'}</td></tr>`;
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
      fixIndexAt = data.fix_index_at || 0;
      // Right after an update the fix index is still building: look again soon.
      if (!fixIndexAt) setTimeout(() => loadRows(true), 10000);
      claims = data.claims || {};
      drawRows();
    } catch (error) {
      if (quiet) return;
      $('rows').innerHTML = `<tr><td colspan="7" class="state">Could not load failures (${esc(error.message)}).</td></tr>`;
    } finally {
      $('rows-progress').hidden = true;
    }
  }

  // ---- signal popup --------------------------------------------------------
  // The ? after a row's signal opens a centred popup: the traceback for a
  // failure, the daily timings for a slow test. It is centred on the part of
  // this page the viewer can see: inside Grafana the page is a tall iframe.
  const modal = $('modal'), box = $('modal-box'), frame = $('modal-frame');
  let modalRow = null;
  function visibleArea() {
    try {
      const rect = window.frameElement.getBoundingClientRect();
      const top = Math.max(0, -rect.top);
      return [top, Math.min(innerHeight, parent.innerHeight - rect.top) - top];
    } catch (_) {
      return [0, innerHeight];
    }
  }
  function openSignal(row) {
    const slow = row.category === 'slow';
    if (row !== modalRow) {
      modalRow = row;
      $('modal-title').textContent = `${row.model} · ${slow ? `slow: ${row.signal}` : row.signal}`;
      $('modal-sub').textContent = `${row.nodeid} · ${row.job} · ${slow ? 'longest run per day on main' : `last failure ${ago(row.last_seen)}`}`;
      $('modal-more').hidden = slow;
      frame.hidden = slow; $('modal-text').hidden = !slow;
      if (slow) {
        frame.removeAttribute('src');
        $('modal-text').textContent = row.durations.map((value, i) =>
          `${dayLabel(rowsFirstDay + i * DAY).padEnd(8)} ${value == null ? '—  (not among the day\'s slowest)' : `${minutes(value)}${row.days[i] ? '' : '  (outside the top 20 that day)'}`}`).join('\n')
          + `\n\nAmong the ${row.runner} top 20 on ${row.fails} of ${row.days.length} days.`;
      } else {
        $('modal-more').href = testUrl(row);
        frame.src = `/failure?${new URLSearchParams({trace_id: row.trace_id, test_nodeid: row.nodeid})}`;
      }
    }
    const [top, height] = visibleArea();
    const boxHeight = Math.max(240, Math.min(slow ? 420 : 640, height - 24));
    box.style.height = `${boxHeight}px`;
    box.style.top = `${top + Math.max(12, (height - boxHeight) / 2)}px`;
    modal.classList.add('show');
    $('modal-close').focus();
  }
  const closeSignal = () => modal.classList.remove('show');
  $('modal-close').addEventListener('click', closeSignal);
  modal.addEventListener('click', event => { if (event.target === modal) closeSignal(); });
  document.addEventListener('keydown', event => { if (event.key === 'Escape') closeSignal(); });

  // ---- actions -------------------------------------------------------------
  // Each row's buttons drive the Test page's own Serge Actions, unchanged: a
  // hidden /serge-actions frame per test, whose button we click. It handles
  // sign-in, the streamed steps and the dialogs (opened in the top-most page),
  // so the behaviour matches the Test page exactly. A frame is kept per test
  // because reloading it would abort a question still running in it.
  const FRAME_LIMIT = 8;
  const frames = new Map();
  function sergeFrame(row) {
    const context = row.category === 'slow'
      ? {kind: 'slow', test_nodeid: row.nodeid, job: row.job, runner: row.runner,
        first_day: new Date(rowsFirstDay * 1000).toISOString().slice(0, 10),
        durations: JSON.stringify(row.durations.map(value => value == null ? null : Math.round(value)))}
      : {trace_id: row.trace_id, test_nodeid: row.nodeid, job: row.job, status: 'ERROR',
        module: row.nodeid.split('::')[0].split('/').pop(), pr: 'main', run_id: row.run_id};
    return actionsFrame(`${row.category}\n${row.nodeid}\n${row.job}`,
      `/serge-actions?${new URLSearchParams({...context, work: '1'})}`);
  }
  // A packet travels in the fragment: up to 60 tests do not fit a query string.
  function packetFrame(packet) {
    return actionsFrame(`packet\n${packet.signal}\n${packet.tests.map(test => test.test_nodeid).join('\n')}`,
      `/serge-actions?work=1#${encodeURIComponent(JSON.stringify(packet))}`);
  }
  function actionsFrame(key, src) {
    let entry = frames.get(key);
    if (entry) { frames.delete(key); frames.set(key, entry); return entry; }
    const frame = document.createElement('iframe');
    frame.className = 'serge'; frame.title = 'Serge Actions'; frame.tabIndex = -1;
    frame.src = src;
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
    const packet = button.dataset.packet != null ? packets[Number(button.dataset.packet)] : null;
    const row = packet ? null : current[Number(button.dataset.i)];
    if ((!row && !packet) || button.getAttribute('aria-busy') === 'true') return;
    button.setAttribute('aria-busy', 'true');
    try {
      const {frame, loaded} = packet ? packetFrame(packet) : sergeFrame(row);
      await loaded;
      // The frame enables its buttons once it knows whether the viewer is signed in.
      for (let tries = 0; tries < 100; tries++) {
        const target = frame.contentDocument?.getElementById(packet ? 'work-on-it' : button.dataset.act);
        if (target && !target.disabled) { target.click(); return; }
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    } finally {
      button.removeAttribute('aria-busy');
    }
  }
  $('rows').addEventListener('click', event => {
    const why = event.target.closest('button[data-why]');
    if (why) { const row = current[Number(why.dataset.why)]; if (row) openSignal(row); return; }
    const button = event.target.closest('button[data-act],button[data-packet]');
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
    loadRows();
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
  $('fix-only').addEventListener('change', event => { state.fix = event.target.checked; drawRows(); });
  $('q').addEventListener('input', event => { state.q = event.target.value.trim().toLowerCase(); drawRows(); });

  press($('days'), state.days);
  loadRows();
  // Claims move while people work (Bug Days): refresh the rows quietly. The
  // server caches the Prometheus part, so this only re-reads the claims.
  setInterval(() => { if (!document.hidden) loadRows(true); }, 30000);
  // Show a claim as soon as its dialog closes (published, or a draft released),
  // not on the next tick. The Serge dialogs open in the top-most same-origin page.
  let dialogHost = document;
  try { for (let w = window; w.parent !== w;) { w = w.parent; if (w.document.body) dialogHost = w.document; } } catch (_) {}
  new MutationObserver(records => {
    if (records.some(record => [...record.removedNodes].some(node => node.id === 'tci-new-issue'))) loadRows(true);
  }).observe(dialogHost.body, {childList: true});
  fetch('/api/user', {credentials: 'same-origin', cache: 'no-store'})
    .then(response => response.ok ? response.json() : null)
    .then(user => { if (user?.login) { me = user.login; $('mine-toggle').hidden = false; if (rows.length) drawRows(); } })
    .catch(() => {});
})();
