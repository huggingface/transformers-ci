/* Shared unified diff renderer. Patch content is always inserted as text. */
(() => {
  function parse(patch) {
    const files = [];
    let file, oldLine = null, newLine = null;
    for (const line of patch.replace(/\r\n/g, "\n").replace(/\n$/, "").split("\n")) {
      if (line.startsWith("diff --git ")) {
        file = {name: line.slice(11), rows: [], additions: 0, deletions: 0};
        files.push(file);
        oldLine = newLine = null;
      }
      if (!file) {
        file = {name: "Patch", rows: [], additions: 0, deletions: 0};
        files.push(file);
      }
      const hunk = line.match(/^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/);
      let kind = "meta", old = null, next = null;
      if ((line.startsWith("+++ ") || line.startsWith("--- ")) && oldLine === null) {
        if (line.slice(4) !== "/dev/null") file.name = line.slice(4).replace(/^[ab]\//, "");
      }
      if (hunk) {
        oldLine = Number(hunk[1]); newLine = Number(hunk[2]); kind = "hunk";
      } else if (oldLine !== null) {
        if (line.startsWith("+")) {
          kind = "add"; next = newLine++; file.additions++;
        } else if (line.startsWith("-")) {
          kind = "del"; old = oldLine++; file.deletions++;
        } else if (line.startsWith(" ")) {
          kind = "context"; old = oldLine++; next = newLine++;
        }
      }
      file.rows.push({text: line, kind, old, next});
    }
    return files;
  }

  function el(tag, cls, value) {
    const node = document.createElement(tag);
    node.className = cls;
    if (value != null) node.textContent = value;
    return node;
  }

  function isTestFile(path) {
    return /(^|\/)(tests?|__tests__)(\/|$)|(^|\/)test_[^/]+$|(^|\/)[^/]+_test\.[^/]+$|\.(test|spec)\.[^/]+$/.test(path);
  }

  function isGeneratedFile(file, files) {
    if (file.rows.some(row => row.kind !== 'del' && /This file was automatically generated from/.test(row.text))) return true;
    const dir = file.name.slice(0, file.name.lastIndexOf('/') + 1);
    return (file.name.slice(dir.length) === '__init__.py' || /^(configuration|modeling|processing|image_processing(?:_pil|_fast)?|video_processing|tokenization(?:_fast)?|feature_extraction)_/.test(file.name.slice(dir.length))) && files.some(f => f.name.startsWith(dir + 'modular_') && !f.name.slice(dir.length).includes('/'));
  }

  function createRow(row) {
    const tr = el('tr', `patch-line ${row.kind}`);
    if (row.old != null) tr.dataset.oldLine = row.old;
    if (row.next != null) tr.dataset.newLine = row.next;
    tr.append(el('td', 'patch-number', row.old), el('td', 'patch-number', row.next), el('td', 'patch-code', row.text));
    return tr;
  }

  function isCommentLine(file, row) {
    return file.name.endsWith('.py') && ['add', 'del', 'context'].includes(row.kind) && /^\s*#/.test(row.text.slice(1));
  }

  function moduleHeaderRows(file) {
    const hidden = new Set();
    const markdown = /\.mdx?$/.test(file.name);
    if (!file.name.endsWith('.py') && !markdown) return hidden;
    for (const coordinate of ['old', 'next']) {
      const source = file.rows.filter(row => row[coordinate] != null);
      if (!source.length || source[0][coordinate] !== 1) continue;
      const header = [];
      if (markdown) {
        let started = false, closed = false;
        for (const row of source) {
          const content = row.text.slice(1).trim();
          if (!started && content && !content.startsWith('<!--')) break;
          header.push(row);
          if (content.startsWith('<!--')) started = true;
          if (started && content.includes('-->')) { closed = true; break; }
        }
        if (closed && header.some(row => /copyright/i.test(row.text))) header.forEach(row => hidden.add(row));
        continue;
      }
      for (const row of source) {
        const content = row.text.slice(1).trim();
        if (content && !content.startsWith('#')) break;
        header.push(row);
      }
      if (header.some(row => /copyright|licensed under|SPDX-License-Identifier|automatically generated/i.test(row.text))) header.forEach(row => hidden.add(row));
    }
    return hidden;
  }

  function render(container, patch, options = {}) {
    const allFiles = parse(patch);
    const files = allFiles.filter(file => (!options.hideTests || !isTestFile(file.name)) && (!options.onlyPython || file.name.endsWith('.py')) && (!options.onlyModels || file.name.startsWith('src/transformers/models/')) && (!(options.hideGenerated || options.onlyModels) || !isGeneratedFile(file, allFiles)));
    const wrapper = el("div", "patch-view");
    const toolbar = el("div", "patch-toolbar");
    toolbar.append(el("span", "", `${files.length === allFiles.length ? files.length : files.length + ' of ' + allFiles.length} file${files.length === 1 ? "" : "s"} · +${files.reduce((n, f) => n + f.additions, 0)} −${files.reduce((n, f) => n + f.deletions, 0)}`));
    const download = el("a", "secondary nav-link", "Download full patch");
    const url = URL.createObjectURL(new Blob([patch], {type: "text/plain"}));
    const previous = container.dataset.patchUrl;
    if (previous) URL.revokeObjectURL(previous);
    container.dataset.patchUrl = url;
    download.href = url; download.download = "pr.diff";
    toolbar.append(download);
    wrapper.append(toolbar);
    if (!files.length) wrapper.append(el('p', 'comments-status', 'No files match these filters.'));
    for (const file of files) {
      const details = el("details", "patch-file"); details.open = true;
      details.dataset.path = file.name;
      const summary = el("summary", "patch-file-header");
      summary.append(el("span", "patch-filename", file.name), el("span", "patch-counts", `+${file.additions} −${file.deletions}`));
      const scroll = el("div", "patch-scroll");
      const table = el("table", "patch-table");
      table.setAttribute("aria-label", `Unified diff for ${file.name}`);
      const body = document.createElement("tbody");
      const headerRows = options.hideModuleHeaders ? moduleHeaderRows(file) : new Set();
      for (const row of file.rows) {
        if (headerRows.has(row)) continue;
        if (options.hideComments && isCommentLine(file, row)) continue;
        body.append(createRow(row));
      }
      table.append(body); scroll.append(table); details.append(summary, scroll); wrapper.append(details);
    }
    const raw = el("details", "patch-raw");
    raw.append(el("summary", "", "Raw full patch"), el("pre", "", patch));
    wrapper.append(raw);
    container.replaceChildren(wrapper);
  }

  function showLines(container, comment) {
    container.querySelectorAll('.line-highlight').forEach(row => row.classList.remove('line-highlight'));
    container.querySelectorAll('.comment-context').forEach(context => context.remove());
    const file = [...container.querySelectorAll('.patch-file')].find(f => f.dataset.path === comment.path);
    const key = comment.side === 'LEFT' ? 'oldLine' : 'newLine';
    const end = Number(comment.outdated ? comment.original_line || comment.line : comment.line);
    const start = Number((comment.outdated ? comment.original_start_line : comment.start_line) || end);
    let rows = file && !comment.outdated ? [...file.querySelectorAll('.patch-line')].filter(row => row.dataset[key] != null && Number(row.dataset[key]) >= start && Number(row.dataset[key]) <= end) : [];
    if (!rows.length && comment.diff_hunk) {
      const parsed = parse(comment.diff_hunk)[0];
      const coordinate = key === 'oldLine' ? 'old' : 'next';
      const nearby = parsed.rows.filter(row => row.kind === 'hunk' || (row[coordinate] != null && row[coordinate] >= start - 4 && row[coordinate] <= end + 4));
      const context = el('section', 'comment-context');
      context.append(el('p', '', `${comment.path} · ${comment.outdated ? 'Comment context from an earlier revision' : 'Comment context'}`));
      const scroll = el('div', 'patch-scroll'), table = el('table', 'patch-table'), body = document.createElement('tbody');
      nearby.forEach(row => body.append(createRow(row))); table.append(body); scroll.append(table); context.append(scroll);
      if (file) file.append(context); else container.prepend(context);
      rows = [...context.querySelectorAll('.patch-line')].filter(row => row.dataset[key] != null && Number(row.dataset[key]) >= start && Number(row.dataset[key]) <= end);
    }
    if (!rows.length) return false;
    if (file) file.open = true;
    rows.forEach(row => row.classList.add('line-highlight'));
    rows[0].tabIndex = -1; rows[0].focus({preventScroll:true}); rows[0].scrollIntoView({behavior:'smooth', block:'center', inline:'nearest'});
    return true;
  }
  async function fetchText(url, progress, status, label) {
    const response = await fetch(url, {cache:'no-store'});
    if (!response.ok) throw Error(await response.text());
    if (!response.body) return response.text();
    // Fetch streams decoded bytes; compressed Content-Length is not comparable.
    const length = response.headers.get('Content-Encoding') ? 0 : Number(response.headers.get('Content-Length'));
    const total = Number.isFinite(length) && length > 0 ? length : 0;
    const reader = response.body.getReader(), decoder = new TextDecoder(), chunks = [];
    let loaded = 0, lastUpdate = 0;
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      loaded += value.byteLength;
      chunks.push(decoder.decode(value, {stream:true}));
      if (total) { progress.max = total; progress.value = Math.min(loaded, total); }
      if (performance.now() - lastUpdate > 200) {
        status.textContent = 'Loading ' + label + '… ' + (total ? Math.min(100, Math.floor(loaded / total * 100)) + '%' : Math.ceil(loaded / 1024).toLocaleString() + ' KB');
        lastUpdate = performance.now();
      }
    }
    chunks.push(decoder.decode());
    return chunks.join('');
  }

  globalThis.TCIPatch = {parse, render, isTestFile, isGeneratedFile, showLines, isCommentLine, moduleHeaderRows, fetchText};
})();
