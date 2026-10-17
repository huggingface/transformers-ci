// Shared by the exporter's own pages (PR CI view, Last 20 PRs, Project health,
// Branch Runs): theme, DOM helpers, the live spinner and lane icons, so the
// same thing looks and reads the same on every page. Exposes `window.tci`.
// Each page loads it from its own prefix (/pr-ci-view/common.js, …), which
// the ingress already routes; the exporter serves the one file for all.
(() => {
  'use strict';
  const style = document.createElement('style');
  style.textContent = [
    // Live: a run or job GitHub reports as queued or in progress.
    '.live{display:inline-block;width:12px;height:12px;border:2px solid var(--line);border-top-color:var(--amber,#f2cc60);border-radius:50%;animation:tci-spin .8s linear infinite;vertical-align:-2px}',
    '@keyframes tci-spin{to{transform:rotate(360deg)}}',
    '.lane-icon{display:inline-flex;gap:1px;vertical-align:middle;color:var(--muted)}',
    '.lane-icon svg{width:18px;height:18px}.lane-icon.nvidia{color:#76b900}.lane-icon.amd{color:#ed1c24}',
    '.lanes{display:inline-flex;gap:6px;vertical-align:middle}',
    '@media(prefers-reduced-motion:reduce){.live{animation:none}}',
  ].join('\n');
  // Prepended so a page's own rules (e.g. a pulsing queued spinner) still win.
  document.head.prepend(style);

  // Follow the Grafana page we are iframed into.
  function applyTheme() {
    try {
      const body = parent.document.body;
      if (/light/.test(body.className) || parent.grafanaBootData?.user?.lightTheme) document.body.classList.add('light');
    } catch (_) {}
  }
  const node = (tag, cls, value) => {
    const item = document.createElement(tag);
    if (cls) item.className = cls;
    if (value != null) item.textContent = String(value);
    return item;
  };
  const esc = value => String(value).replace(/[&<>"']/g, ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[ch]));
  function spinner(label) {
    const spin = node('span', 'live');
    if (label) { spin.title = label; spin.setAttribute('role', 'img'); spin.setAttribute('aria-label', label); }
    return spin;
  }

  // ---- lanes ---------------------------------------------------------------
  // single-gpu runs on aws-g5-4xlarge (1 A10G), multi-gpu on aws-g5-12xlarge
  // (4 A10G); AMD's runner names carry their GPU count (amd-1gpu, amd-2gpu).
  const LANES = {
    cpu: {vendor: 'cpu', name: 'CPU', cards: 1},
    gpu: {vendor: 'nvidia', name: 'Nvidia', cards: 1},
    'single-gpu': {vendor: 'nvidia', name: 'Nvidia', cards: 1},
    'multi-gpu': {vendor: 'nvidia', name: '4×Nvidia', cards: 2},
    'amd-1gpu': {vendor: 'amd', name: 'AMD', cards: 1},
    'amd-2gpu': {vendor: 'amd', name: '2×AMD', cards: 2},
  };
  // A lane id alone (a CPU lane's tests carry no hardware) picks its default.
  const DEFAULT_FOR_LANE = {cpu: 'cpu', amd: 'amd-1gpu', gpu: 'single-gpu', nvidia: 'single-gpu'};
  const PATHS = {
    cpu: 'M7 7h10v10H7zM10 10h4v4h-4zM10 3v4M14 3v4M10 17v4M14 17v4M3 10h4M3 14h4M17 10h4M17 14h4',
    card: 'M3 7h18v10H3zM8 12m-2.5 0a2.5 2.5 0 1 0 5 0a2.5 2.5 0 1 0-5 0M14 10h4M14 14h4M5 17v3M9 17v3',
  };
  const SVG = 'http://www.w3.org/2000/svg';
  const lane = (hardware, laneId) => LANES[hardware] || LANES[DEFAULT_FOR_LANE[laneId]] || {vendor: 'nvidia', name: hardware || 'GPU', cards: 1};
  function laneIcon(hardware, laneId) {
    const info = lane(hardware, laneId);
    const wrap = node('span', `lane-icon ${info.vendor}`);
    wrap.title = info.name; wrap.setAttribute('role', 'img'); wrap.setAttribute('aria-label', info.name);
    for (let index = 0; index < info.cards; index++) {
      const svg = document.createElementNS(SVG, 'svg');
      svg.setAttribute('viewBox', '0 0 24 24'); svg.setAttribute('fill', 'none');
      svg.setAttribute('stroke', 'currentColor'); svg.setAttribute('stroke-width', '1.7');
      svg.setAttribute('stroke-linecap', 'round'); svg.setAttribute('aria-hidden', 'true');
      const path = document.createElementNS(SVG, 'path');
      path.setAttribute('d', info.vendor === 'cpu' ? PATHS.cpu : PATHS.card);
      svg.append(path); wrap.append(svg);
    }
    return wrap;
  }
  // Several lanes side by side (a run on main can span CPU and both GPU pools).
  function laneIcons(hardwareList) {
    const order = Object.keys(LANES);
    const wrap = node('span', 'lanes');
    [...new Set(hardwareList)].sort((a, b) => order.indexOf(a) - order.indexOf(b)).forEach(hardware => wrap.append(laneIcon(hardware)));
    return wrap;
  }

  window.tci = {applyTheme, node, esc, spinner, laneName: (hardware, laneId) => lane(hardware, laneId).name, laneIcon, laneIcons};
})();
