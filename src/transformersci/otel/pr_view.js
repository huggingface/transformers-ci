/* PR dashboard view switch. Loaded by the Branch panel's same-origin iframe. */
(() => {
  const host = window.parent, doc = host.document;
  if (host.tciPrView) { host.tciPrView.sync(); return; }
  const style = doc.createElement('style');
  style.textContent = `
    .tci-pr-layout[data-tci-view="tests"] > .tci-pr-patch-panel,
    .tci-pr-layout[data-tci-view="patch"] > .tci-pr-tests-panel {display:none!important}
    .tci-pr-patch-panel iframe {display:block;width:100%;height:100%;min-height:640px;border:0}
    [data-pr-mode="tests"][data-running="true"]::after {content:"";display:inline-block;width:8px;height:8px;margin-left:7px;border:2px solid currentColor;border-right-color:transparent;border-radius:50%;vertical-align:-1px;animation:tci-tests-spin 1s linear infinite}
    @keyframes tci-tests-spin {to{transform:rotate(360deg)}}
    @media(prefers-reduced-motion:reduce){[data-pr-mode="tests"][data-running="true"]::after{animation:none}}
  `;
  doc.head.append(style);
  let pending = false, previous = null;
  const originals = new WeakMap();
  let activityKey = '', activityTime = 0, activityRequest = null, activityRunning = false;
  function markActivity(branch, running) {
    const button = branch.querySelector('[data-pr-mode="tests"]');
    if (!button) return;
    button.dataset.running = String(running);
    button.title = running ? 'Tests are running or queued' : 'Tests';
    button.setAttribute('aria-label', running ? 'Tests — running' : 'Tests');
  }
  async function checkActivity(branch = doc.querySelector('.tci-branch[data-pr-view]'), force = false) {
    if (!branch || doc.hidden) return;
    let pr, repository;
    try { pr = decodeURIComponent(branch.dataset.pr || ''); repository = decodeURIComponent(branch.dataset.repository || ''); } catch (_) { return; }
    if (!/^[1-9][0-9]*$/.test(pr) || !/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(repository)) { markActivity(branch, false); return; }
    const key = repository + '#' + pr;
    if (key !== activityKey) {
      if (activityRequest) activityRequest.abort();
      activityRequest = null; activityKey = key; activityTime = 0; activityRunning = false;
    }
    markActivity(branch, activityRunning);
    if (activityRequest || (!force && Date.now() - activityTime < 30000)) return;
    activityTime = Date.now();
    const controller = new host.AbortController(); activityRequest = controller;
    const timeout = host.setTimeout(() => controller.abort(), 10000);
    try {
      const response = await host.fetch('/api/ds/query', {method:'POST', credentials:'same-origin', signal:controller.signal, headers:{'Content-Type':'application/json'}, body:JSON.stringify({from:'now-5m', to:'now', queries:[{refId:'PRTestsActive', datasource:{type:'prometheus',uid:'prometheus'}, expr:'max(pytest_run_job_active{pr=' + JSON.stringify(pr) + ',repository=' + JSON.stringify(repository) + '}) or vector(0)', instant:true, range:false, maxDataPoints:1, intervalMs:30000}]})});
      if (!response.ok) throw Error('Live status unavailable');
      const result = (await response.json()).results?.PRTestsActive;
      if (!result || result.error) throw Error('Live status unavailable');
      const running = (result.frames || []).some(frame => frame.schema.fields.some((field, index) => field.type === 'number' && Number(frame.data.values[index]?.at(-1)) > 0));
      if (activityRequest === controller) { activityRunning = running; checkActivity(); }
    } catch (_) {
      if (activityRequest === controller) { activityRunning = false; checkActivity(); }
    } finally {
      host.clearTimeout(timeout);
      if (activityRequest === controller) activityRequest = null;
    }
  }
  function schedule() {
    if (pending) return;
    pending = true;
    host.requestAnimationFrame(() => { pending = false; sync(); });
  }
  function restore(layout) {
    if (!layout) return;
    layout.classList.remove('tci-pr-layout');
    layout.removeAttribute('data-tci-view');
    const old = originals.get(layout);
    if (old) {
      layout.style.height = old.height;
      old.patch.style.transform = old.transform;
    }
    layout.querySelectorAll('.tci-pr-tests-panel,.tci-pr-patch-panel').forEach(item => item.classList.remove('tci-pr-tests-panel','tci-pr-patch-panel'));
  }
  function sync() {
    const branch = doc.querySelector('.tci-branch[data-pr-view]');
    if (branch) checkActivity(branch);
    const branchItem = branch && branch.closest('.react-grid-item');
    const layout = branchItem && branchItem.parentElement;
    const patch = layout && layout.querySelector('[data-viz-panel-id="panel-15"]');
    const patchItem = patch && patch.closest('.react-grid-item');
    if (!layout || !patchItem) { if (!branch && previous) { restore(previous); previous = null; } return; }
    if (previous && previous !== layout) restore(previous);
    previous = layout;
    if (!originals.has(layout)) originals.set(layout, {height:layout.style.height, patch:patchItem, transform:patchItem.style.transform});
    const mode = new URL(host.location.href).searchParams.get('tci-view') === 'patch' ? 'patch' : 'tests';
    layout.classList.add('tci-pr-layout');
    if (layout.dataset.tciView !== mode) layout.dataset.tciView = mode;
    const items = [...layout.children].filter(item => item.classList.contains('react-grid-item'));
    for (const item of items) {
      item.classList.toggle('tci-pr-patch-panel', item === patchItem);
      item.classList.toggle('tci-pr-tests-panel', item !== branchItem && item !== patchItem);
    }
    branch.querySelectorAll('[data-pr-mode]').forEach(button => {
      const pressed = String(button.dataset.prMode === mode);
      if (button.getAttribute('aria-pressed') !== pressed) button.setAttribute('aria-pressed', pressed);
    });
    const origin = layout.getBoundingClientRect().top;
    let bottom = branchItem.getBoundingClientRect().bottom - origin;
    if (mode === 'patch') {
      const top = Math.round(bottom + 8);
      const transform = 'translate(0px, ' + top + 'px)';
      if (patchItem.style.transform !== transform) patchItem.style.setProperty('transform', transform, 'important');
      bottom = top + patchItem.getBoundingClientRect().height;
      const frame = patchItem.querySelector('iframe[data-patch-src]');
      if (frame && frame.getAttribute('src') !== frame.dataset.patchSrc) frame.setAttribute('src', frame.dataset.patchSrc);
    } else {
      for (const item of items) if (item !== patchItem) bottom = Math.max(bottom, item.getBoundingClientRect().bottom - origin);
    }
    const height = Math.ceil(bottom) + 'px';
    if (layout.style.height !== height) layout.style.setProperty('height', height, 'important');
  }
  host.tciSetPrView = mode => {
    if (mode !== 'patch' && mode !== 'tests') return;
    const url = new URL(host.location.href);
    if (mode === 'tests') url.searchParams.delete('tci-view'); else url.searchParams.set('tci-view', mode);
    host.history.replaceState(host.history.state, '', url);
    sync();
  };
  host.tciPrView = {sync:schedule, refreshActivity:() => checkActivity(undefined, true)};
  new host.MutationObserver(schedule).observe(doc.body, {childList:true, subtree:true});
  host.addEventListener('resize', schedule);
  host.addEventListener('popstate', schedule);
  host.setInterval(() => checkActivity(), 30000);
  sync();
})();
