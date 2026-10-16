/* Enrich the overview Category column from current GitHub labels. */
function installOverviewCategories() {
  const host = window, doc = host.document;
  if (host.tciOverviewCategories) { host.tciOverviewCategories(); return; }
  const cache = new Map(), pending = new Set();
  const colors = {'New model':['#4b3714','#f4ce73'],Bugfix:['#d85a6b','#fff'],Feature:['#5d9cec','#fff'],'CI/infra':['#9b7edb','#fff'],Tests:['#3f9cb5','#fff'],Docs:['#72b59a','#111217'],Performance:['#e2a050','#111217'],Refactor:['#8798b0','#111217'],Other:['#77808e','#fff']};
  let scheduled = false;
  function paint(cell, category) {
    if (!cell || !colors[category]) return;
    if (cell.dataset.tciCategory === category &&
        (category === 'New model' ? cell.firstElementChild?.textContent === 'NEW MODEL!' : cell.textContent === category)) return;
    cell.dataset.tciCategory = category;
    cell.style.setProperty('background', colors[category][0], 'important');
    cell.style.setProperty('color', colors[category][1], 'important');
    if (category === 'New model') {
      const stamp = doc.createElement('span');
      stamp.textContent = 'NEW MODEL!';
      stamp.style.cssText = 'display:inline-block;border:1px solid #d4a72c;border-radius:3px;padding:1px 5px;font:800 10px/1.2 system-ui,sans-serif;letter-spacing:.04em;transform:rotate(-2deg)';
      cell.replaceChildren(stamp);
    } else cell.textContent = category;
  }
  function sync() {
    const panel = doc.querySelector('[data-viz-panel-id="panel-10"]');
    const grid = panel && panel.querySelector('[role="grid"]');
    if (!grid || !grid.querySelector('[role="columnheader"][aria-colindex="4"] [title="Category"]')) return;
    for (const row of grid.querySelectorAll('[role="row"][aria-rowindex]:not([aria-rowindex="1"])')) {
      const pr = row.querySelector('[role="gridcell"][aria-colindex="2"] a')?.textContent?.trim();
      const cell = row.querySelector('[role="gridcell"][aria-colindex="4"]');
      if (!/^[1-9][0-9]*$/.test(pr || '') || !cell) continue;
      const entry = cache.get(pr);
      if (entry && entry.until > Date.now()) { paint(cell, entry.category); continue; }
      if (pending.has(pr)) continue;
      pending.add(pr);
      host.fetch('/pr-search/labels?' + new URLSearchParams({pr,repository:'huggingface/transformers'}), {credentials:'same-origin'})
        .then(response => response.ok ? response.json() : null)
        .then(data => {
          const category = data?.category || '';
          cache.set(pr, {category,until:Date.now()+60000});
          if (category) schedule();
        })
        .catch(() => cache.set(pr, {category:'',until:Date.now()+15000}))
        .finally(() => pending.delete(pr));
    }
  }
  function schedule() {
    if (scheduled) return;
    scheduled = true;
    host.requestAnimationFrame(() => { scheduled = false; sync(); });
  }
  host.tciOverviewCategories = schedule;
  new host.MutationObserver(schedule).observe(doc.body, {childList:true,subtree:true});
  host.setInterval(schedule, 30000);
  sync();
}
(() => {
  const host = window.parent;
  if (host.tciOverviewCategories) { host.tciOverviewCategories(); return; }
  const script = host.document.createElement('script');
  script.textContent = '(' + installOverviewCategories.toString() + ')();';
  host.document.head.append(script);
  script.remove();
})();
