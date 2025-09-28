// ---- Pfade (Server-Root ist Projekt-Root!) ----
const PATH_SAMPLES = "viewer_data/samples.csv";
const PATH_RUNS    = "viewer_data/runs.csv";

// ---- Zahlformat-Config (optional anpassen) ----
const FMT_CFG = {
  sci_threshold: 1e-3, // |x| < threshold → wissenschaftlich
  // Nachkommastellen für Bereiche:
  d_ge_1e4: 0,     // ≥ 10 000
  d_1e2_1e4: 1,    // 100 … 9 999.9
  d_1_100: 2,      // 1 … 99.99
  d_1e1_1: 3,      // 0.1 … <1
  d_1e2_1e1: 4,    // 0.01 … <0.1
  d_1e3_1e2: 5,    // 0.001 … <0.01
};

// ---- State ----
let SAMPLES = [];   // aus samples.csv
let RUNS = [];      // aus runs.csv
let FILTER = { run: "", phase: "", text: "" };
let ACTIVE_RANK = null;
let RUN_IDS = [];   // [{run_id, count}]

// --- MathJax (LaTeX) Helper: sicher und idempotent ---
function typesetMath() {
  // Keine Fehlermeldung, wenn MathJax noch nicht geladen ist
  try {
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise();
    }
  } catch (e) {
    // noop
  }
}

// --- LaTeX-Wissenschaftsformat (×10^k), OHNE Dollar ---
function fmtSciLatex(n, sig = 2) {
  const e = Number(n);
  if (!Number.isFinite(e) || e === 0) return "0";
  const [m, pow] = e.toExponential(sig).split("e");
  const mant = String(Number(m));          // trailing zeros weg
  const exp  = pow.replace("+", "");
  return `${mant}\\times 10^{${exp}}`;
}


// --- Smarter Formatter: normal oder LaTeX-Sci je Größe ---
function fmtSmart(x, { sci_threshold = 1e-3, sci_upper = 1e6, sigSci = 2 } = {}) {
  const n = Number(x);
  if (!Number.isFinite(n)) return { text: String(x), latex: null, raw: x };

  const trim = s =>
    s.replace(/(\.\d*?[1-9])0+$/, "$1")
     .replace(/\.0+$/, "")
     .replace(/^-0(?:\.0+)?$/, "0");

  const a = Math.abs(n);

  // sehr klein oder sehr groß → LaTeX
  if ((a > 0 && a < sci_threshold) || a >= sci_upper) {
    return { text: `$${fmtSciLatex(n, sigSci)}$`, latex: true, raw: n };
  }

  let digits = 2;
  if (a >= 1e4) digits = 0;
  else if (a >= 100) digits = 1;
  else if (a >= 1) digits = 2;
  else if (a >= 1e-1) digits = 3;
  else if (a >= 1e-2) digits = 4;
  else if (a >= 1e-3) digits = 5;

  return { text: trim(n.toFixed(digits)), latex: false, raw: n };
}

// Unicode-Superscript (ohne MathJax)
const SUP = { "-":"\u207B","0":"\u2070","1":"\u00B9","2":"\u00B2","3":"\u00B3","4":"\u2074","5":"\u2075","6":"\u2076","7":"\u2077","8":"\u2078","9":"\u2079" };
function expToSup(n) {
  const s = String(n);
  return [...s].map(ch => SUP[ch] ?? ch).join("");
}

// Nur Text, keine $…$:
// kleine/ große Zahlen als "1.23×10⁻⁴" (Unicode), sonst dezimal wie bisher.
function fmtSmartPlain(x, {sci_threshold=1e-3, sci_upper=1e6, sigSci=2} = {}) {
  const n = Number(x);
  if (!Number.isFinite(n)) return String(x);

  const trim = s => s.replace(/(\.\d*?[1-9])0+$/,'$1').replace(/\.0+$/,'').replace(/^-0(?:\.0+)?$/,'0');
  const a = Math.abs(n);

  if ((a > 0 && a < sci_threshold) || a >= sci_upper) {
    const [m, e] = n.toExponential(sigSci).split("e");
    const mant = trim(String(Number(m)));
    const pow  = e.replace("+","");
    return `${mant}×10${expToSup(pow)}`;
  }

  let digits = 2;
  if (a >= 1e4) digits = 0;
  else if (a >= 100) digits = 1;
  else if (a >= 1) digits = 2;
  else if (a >= 1e-1) digits = 3;
  else if (a >= 1e-2) digits = 4;
  else if (a >= 1e-3) digits = 5;
  return trim(n.toFixed(digits));
}


// ---- CSV Loading ----
async function loadCSV(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Fetch failed: ${path}`);
  const text = await res.text();
  return parseCSV(text);
}

// Minimal-CSV-Parser (kein Handling von Kommas in Anführungszeichen)
function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  const header = lines.shift().split(",");
  return lines.map(line => {
    const cols = line.split(",");
    const obj = {};
    header.forEach((h, i) => obj[h] = cols[i]);
    return obj;
  });
}

function fmt(x) {
  const n = Number(x);
  if (!Number.isFinite(n)) return String(x);

  const trim = s =>
    s.replace(/(\.\d*?[1-9])0+$/,'$1') // trailing zeros
     .replace(/\.0+$/,'')              // ".0"
     .replace(/^-0(?:\.0+)?$/,'0');    // "-0" → "0"

  const a = Math.abs(n);
  if (a >= 1e4)   return trim(n.toFixed(FMT_CFG.d_ge_1e4));
  if (a >= 100)   return trim(n.toFixed(FMT_CFG.d_1e2_1e4));
  if (a >= 1)     return trim(n.toFixed(FMT_CFG.d_1_100));
  if (a >= 1e-1)  return trim(n.toFixed(FMT_CFG.d_1e1_1));
  if (a >= 1e-2)  return trim(n.toFixed(FMT_CFG.d_1e2_1e1));
  if (a >= 1e-3)  return trim(n.toFixed(FMT_CFG.d_1e3_1e2));

  // sehr klein → wissenschaftlich mit 2 signifikanten Ziffern
  const exp = n.toExponential(2); // z.B. "-5.00e-4"
  const [m, e] = exp.split("e");
  const mant = trim(String(Number(m)));      // "-5", "3.1", …
  const pow  = e.startsWith("+") ? e.slice(1) : e; // "-4"
  return `${mant}e${pow}`;
}


function buildRunChips() {
  const cont = document.getElementById("run-chips");
  cont.innerHTML = "";
  // Übersicht Run-IDs mit Counts
  const counts = new Map();
  for (const s of SAMPLES) {
    const id = Number(s.run_id);
    counts.set(id, (counts.get(id) || 0) + 1);
  }
  RUN_IDS = Array.from(counts.entries()).map(([run_id, count]) => ({run_id, count})).sort((a,b)=>a.run_id-b.run_id);

  // Chip "Alle"
  const chipAll = document.createElement("div");
  chipAll.className = "chip" + (FILTER.run === "" ? " active" : "");
  chipAll.textContent = `Alle (${SAMPLES.length})`;
  chipAll.addEventListener("click", () => { FILTER.run = ""; renderList(); buildRunChips(); window.dispatchEvent(new CustomEvent('oedo:filter-changed'));});
  cont.appendChild(chipAll);

  // Chips je Run
  for (const {run_id, count} of RUN_IDS) {
    const chip = document.createElement("div");
    chip.className = "chip" + (FILTER.run !== "" && Number(FILTER.run) === run_id ? " active" : "");
    chip.textContent = `Run ${run_id} (${count})`;
    chip.title = `Zeige nur Samples aus Run ${run_id}`;
    chip.addEventListener("click", () => { FILTER.run = String(run_id); renderList(); buildRunChips(); window.dispatchEvent(new CustomEvent('oedo:filter-changed'));});
    cont.appendChild(chip);
  }
}

function renderList() {
  const list = document.getElementById("samples-list"); // ← ID passt?
  list.innerHTML = "";

  const q = (FILTER.text || "").toLowerCase();
  const runFilter = FILTER.run === "" ? null : Number(FILTER.run);
  const phaseFilter = FILTER.phase || null;

  const items = SAMPLES.filter(r => {
    if (runFilter !== null && Number(r.run_id) !== runFilter) return false;
    if (phaseFilter !== null && phaseFilter !== "" && r.phase !== phaseFilter) return false;
    if (q) {
      const hay = Object.values(r).join(" ").toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });

  const activeRun = getActiveRunId(); // ← z. B. Run des aktuell selektierten Samples

  items.forEach(r => {
    const runIdNum = Number(r.run_id);
    const isActive = (ACTIVE_RANK !== null && Number(r.rank) === ACTIVE_RANK);
    const isSameRun = !isActive && activeRun !== null && runIdNum === Number(activeRun);

    // NEU: list-item + States
    const div = document.createElement("div");
    div.className = "list-item" +
      (isActive ? " active" : "") +
      (isSameRun ? " same-run" : "");
    div.tabIndex = 0; // für :focus-visible
    div.dataset.globalIdx = r.global_idx;
    div.dataset.runId = r.run_id;
    div.dataset.rank = r.rank;
    div.setAttribute("aria-selected", String(isActive));

    // Werte (ohne LaTeX) für Sidebar
    const s0 = fmtSmartPlain(r.sigma_0);
    const et = fmtSmartPlain(r.eps_total);
    const de = fmtSmartPlain(r.eps_delta);
    const es = fmtSmartPlain(r.e_s);
    const ds = fmtSmartPlain(r.sigma_delta);

    div.innerHTML = `
      <div class="hdr">
        <div class="left">
          <b>rank=${r.rank}</b>
          <span class="pill">Run ${r.run_id}</span>
        </div>
        <div class="right"><span class="pill">${r.phase}</span></div>
      </div>
      <table class="mini-kv"><tbody>
        <tr>
          <th>σ₀ <span class="unit">[kPa]</span></th>
          <td title="sigma_0: ${r.sigma_0}">
            <span class="value"><span class="num">${s0}</span><span class="unit">kPa</span></span>
          </td>
        </tr>
        <tr>
          <th>ε_total <span class="unit">[-]</span></th>
          <td title="eps_total: ${r.eps_total}">
            <span class="value"><span class="num">${et}</span></span>
          </td>
        </tr>
        <tr>
          <th>Δε <span class="unit">[-]</span></th>
          <td title="eps_delta: ${r.eps_delta}">
            <span class="value"><span class="num">${de}</span></span>
          </td>
        </tr>
        <tr>
          <th>eₛ <span class="unit">[-]</span></th>
          <td title="e_s: ${r.e_s}">
            <span class="value"><span class="num">${es}</span></span>
          </td>
        </tr>
        <tr>
          <th>Δσ <span class="unit">[kPa]</span></th>
          <td title="sigma_delta: ${r.sigma_delta}">
            <span class="value"><span class="num">${ds}</span><span class="unit">kPa</span></span>
          </td>
        </tr>
      </tbody></table>
    `;

    // Click + Tastatur
    div.addEventListener("click", () => selectRank(Number(r.rank)));
    div.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        selectRank(Number(r.rank));
      }
    });

    list.appendChild(div);
  });

  // Auto-Select erstes Ergebnis
  if (items.length && (ACTIVE_RANK === null || !items.some(r => Number(r.rank) === ACTIVE_RANK))) {
    selectRank(Number(items[0].rank), false);
  }
  // Sidebar hat KEIN LaTeX → typesetMath() hier nicht nötig
}


function getActiveRunId() {
  if (ACTIVE_RANK === null) return null;
  const rec = SAMPLES.find(r => Number(r.rank) === ACTIVE_RANK);
  return rec ? Number(rec.run_id) : null;
}

function selectRank(rank, scrollIntoView = true) {
  ACTIVE_RANK = rank;
  // Liste neu rendern, damit Same-Run-Markierungen korrekt sind
  renderList();
  // Scroll zum aktiven Element
  if (scrollIntoView) {
    const list = document.getElementById("samples-list");
    const match = Array.from(list.children).find(div => div.textContent.includes(`rank=${rank}`));
    if (match) match.scrollIntoView({ block: "nearest" });
  }
  // Details & Plot
  const rec = SAMPLES.find(r => Number(r.rank) === rank);
  if (!rec) return;
  renderSample(rec);
  renderParams(rec);
  plotRun(Number(rec.run_id), Number(rec.sigma_0), Number(rec.eps_total), rank);
  window.dispatchEvent(new CustomEvent('oedo:rank-selected', { detail: { rank } }));
    window.dispatchEvent(new CustomEvent('samples:selection', {
    detail: { id: rank, run_id: Number(rec.run_id) }
  }));
}

function renderSample(rec) {
  const tbody = document.querySelector("#tbl-sample tbody");

  const entries = [
    ["rank", String(rec.rank), ""],
    ["global_idx", String(rec.global_idx), ""],
    ["run_id", String(rec.run_id), ""],
    ["step_idx", String(rec.step_idx), ""],
    ["phase", rec.phase, ""],
    ["$\\sigma_0$", fmtSmart(rec.sigma_0).text, "kPa", String(rec.sigma_0)],
    ["$\\varepsilon_{\\mathrm{total}}$", fmtSmart(rec.eps_total).text, "", String(rec.eps_total)],
    ["$\\Delta\\varepsilon$", fmtSmart(rec.eps_delta).text, "", String(rec.eps_delta)],
    ["$E_{s}$", fmtSmart(rec.e_s).text, "$\\frac{kPa}{1}$", String(rec.e_s)],
    ["$\\Delta\\sigma$", fmtSmart(rec.sigma_delta).text, "kPa", String(rec.sigma_delta)],
  ];

  tbody.innerHTML = entries.map(([label, val, unit, raw]) => `
    <tr title="${raw ? `${label.replace(/<[^>]*>/g,'')}: ${raw}` : ''}">
      <th>${label}${unit ? ` <span class="unit">[${unit}]</span>` : ""}</th>
      <td><span class="value"><span class="num">${val}</span>${unit ? `<span class="unit"> ${unit}</span>` : ""}</span></td>
    </tr>
  `).join("");

  typesetMath(); // LaTeX neu setzen
  latexTypeset(document.getElementById('main'));   // <— NEU: nur ein Call
}


function renderParams(rec) {
  const tbody = document.querySelector("#tbl-params tbody");
  const entries = Object.entries(rec)
    .filter(([k]) => k.startsWith("param."))
    .map(([k,v]) => [k.replace("param.",""), Number(v)])
    .sort(([a],[b]) => a.localeCompare(b));

  tbody.innerHTML = entries.map(([k,v]) => {
    const f = fmtSmart(v);           // → gibt ggf. $…$ zurück
    return `<tr title="${k}: ${v}">
      <th>${k}</th>
      <td><span class="value"><span class="num">${f.text}</span></span></td>
    </tr>`;
  }).join("");

  typesetMath();                     // ← WICHTIG: LaTeX rendern
}



function plotRun(run_id, sx, ex, rank) {
  const rows = RUNS.filter(r => Number(r.run_id) === run_id)
                   .sort((a,b) => Number(a.step_idx) - Number(b.step_idx));
  const sigma = rows.map(r => Number(r.sigma_0));
  const eps   = rows.map(r => Number(r.eps_total));

  // Punkte dieses Runs, die in der Auswahl sind:
  const sameRunPoints = SAMPLES.filter(s => Number(s.run_id) === run_id);
  const sX = sameRunPoints.map(s => Number(s.sigma_0));
  const eX = sameRunPoints.map(s => Number(s.eps_total));
  const labels = sameRunPoints.map(s => `rank=${s.rank}`);

  const tracePath = { x: sigma, y: eps, mode: "lines+markers", name: `Run ${run_id}` };
  const traceSame = { x: sX, y: eX, mode: "markers", name: "Samples (Run)",
                      marker: { size: 9, color: "#fbbf24" } };
  const tracePoint = { x: [sx], y: [ex], mode: "markers", name: `Aktiv (rank=${rank})`,
                       marker: { size: 12, color: "#f97316", line: { width: 2 } } };

  const layout = {
    margin: { l: 50, r: 20, t: 20, b: 40 },
    xaxis: { title: "Stress σ" },
    yaxis: { title: "Strain ε" },
    showlegend: true
  };

  Plotly.newPlot("plot", [tracePath, traceSame, tracePoint], layout, {displayModeBar: false});
}


// ===== Runs Table Module =====================================================
const RunsTable = (() => {
  let rows = [];           // gesamte RUNS (Array von Objekten)
  let sortKey = 'run_id';
  let sortDir = 'asc';     // 'asc'|'desc'
  let active = { run_id: null, step_idx: null }; // für Hervorhebung
  let paramCols = [];      // dynamische param.*-Spalten (stabile Reihenfolge)

  const $tbl   = document.getElementById('tbl-runs');
  const $thead = $tbl?.querySelector('thead');
  const $tbody = $tbl?.querySelector('tbody');
  const $exp   = document.getElementById('runs-export');

  function init(allRuns) {
    rows = Array.isArray(allRuns) ? allRuns.slice() : [];
    // param.* Spalten sammeln (rechts anfügen, stabil sortiert)
    const keys = new Set();
    for (const r of rows) {
      for (const k in r) if (k.startsWith('param.')) keys.add(k);
    }
    paramCols = Array.from(keys).sort((a,b)=>a.localeCompare(b,'de'));

    // Header um param.* erweitern (einmalig)
    const headRow = $thead.querySelector('tr');
    for (const c of paramCols) {
      const th = document.createElement('th');
      th.dataset.key = c;
      th.textContent = c.replace(/^param\./,'param.');
      headRow.appendChild(th);
    }

    // Sort per Header
    $thead.addEventListener('click', ev => {
      const th = ev.target.closest('th'); if (!th) return;
      const key = th.dataset.key; if (!key) return;
      if (sortKey === key) sortDir = (sortDir === 'asc') ? 'desc' : 'asc';
      else { sortKey = key; sortDir = th.dataset.type === 'number' ? 'asc' : 'asc'; }
      render();
    });

    // Export
    $exp.addEventListener('click', () => exportCSV(getVisibleRows(), getColumns()));

    // Reagiert auf externe Events (Run-Filter & Auswahl)
    window.addEventListener('oedo:rank-selected', ev => {
      const { rank } = ev.detail || {};
      const rec = SAMPLES.find(s => Number(s.rank) === Number(rank));
      if (rec) {
        active.run_id = Number(rec.run_id);
        active.step_idx = Number(rec.step_idx);
        render(); // markiert aktive Zeile
      }
    });
    window.addEventListener('oedo:filter-changed', () => render());

    // Click auf Tabellenzeile → Plot auf diesen Step
    $tbody.addEventListener('click', ev => {
      const tr = ev.target.closest('tr[data-run][data-step]'); if (!tr) return;
      const run_id  = Number(tr.dataset.run);
      const step_idx= Number(tr.dataset.step);
      active = { run_id, step_idx };
      // Plot neu (hole Werte aus RUNS für diesen Run/Step)
      const row = rows.find(r => r.run_id === run_id && r.step_idx === step_idx);
      if (row) plotRun(run_id, row.sigma_0, row.eps_total, ACTIVE_RANK ?? '–');
      render();
    });

    render();
  }

  function getColumns() {
    // Basis + param.*
    return ["run_id","step_idx","phase","sigma_0","eps_total","eps_delta","e_s","sigma_delta", ...paramCols];
  }

  function getVisibleRows() {
    const runFilter = FILTER.run === "" ? null : Number(FILTER.run);
    let data = rows;
    if (runFilter !== null) data = data.filter(r => Number(r.run_id) === runFilter);

    // Sort
    const type = ($thead.querySelector(`th[data-key="${sortKey}"]`)?.dataset.type) || 'string';
    data = data.slice().sort((a,b) => {
      const av = a[sortKey], bv = b[sortKey];
      let cmp = 0;
      if (type === 'number') cmp = (Number(av) || 0) - (Number(bv) || 0);
      else cmp = String(av ?? '').localeCompare(String(bv ?? ''), 'de', { numeric:true });
      return sortDir === 'asc' ? cmp : -cmp;
    });

    return data;
  }

  (function enhanceRunsTableUX(){
    const table  = document.getElementById('tbl-runs');
    const thead  = table?.querySelector('thead');
    const tbody  = table?.querySelector('tbody');
    const hintEl = document.getElementById('runs-col-hint');
    if (!table || !thead || !tbody || !hintEl) return;

    let currentCol = -1;
    let rafToken = null;

    function headerTextAt(colIdx) {
      const th = thead.querySelectorAll('th')[colIdx];
      if (!th) return '';
      // Text inkl. evtl. Subscript/HTML:
      const html = th.innerHTML.trim();
      // Fallback Plaintext:
      const txt  = th.textContent.trim();
      // Kurzer, klarer Titel:
      return txt || html.replace(/<[^>]+>/g,'');
    }

    function setColHover(colIdx) {
      if (colIdx === currentCol) return;
      currentCol = colIdx;

      // Klassen resetten
      thead.querySelectorAll('th').forEach(th => th.classList.remove('hover-col'));
      tbody.querySelectorAll('td').forEach(td => td.classList.remove('hover-col'));

      if (colIdx < 0) {
        hintEl.classList.remove('show');
        hintEl.textContent = '';
        return;
      }

      // Markiere Spalte
      const ths = thead.querySelectorAll('th');
      const tds = tbody.querySelectorAll(`td:nth-child(${colIdx + 1})`);
      if (ths[colIdx]) ths[colIdx].classList.add('hover-col');
      tds.forEach(td => td.classList.add('hover-col'));

      // Hint aktualisieren
      hintEl.textContent = headerTextAt(colIdx);
      hintEl.classList.add('show');
    }

    // Maus: Spaltenindex bestimmen
    tbody.addEventListener('mousemove', (ev) => {
      const td = ev.target.closest('td');
      if (!td) return;
      if (rafToken) cancelAnimationFrame(rafToken);
      rafToken = requestAnimationFrame(() => {
        const colIdx = Array.prototype.indexOf.call(td.parentElement.children, td);
        setColHover(colIdx);
      });
    });

    // Verlassen → Reset
    tbody.addEventListener('mouseleave', () => setColHover(-1));

    // Touch: Erste Berührung toggelt die Spaltenanzeige, zweite hebt auf
    tbody.addEventListener('touchstart', (ev) => {
      const td = ev.target.closest('td');
      if (!td) return;
      const colIdx = Array.prototype.indexOf.call(td.parentElement.children, td);
      setColHover(currentCol === colIdx ? -1 : colIdx);
    }, { passive: true });

    // Header-Hover zeigt auch Titel (nützlich beim Scrollen)
    thead.addEventListener('mousemove', (ev) => {
      const th = ev.target.closest('th');
      if (!th) return;
      const colIdx = Array.prototype.indexOf.call(th.parentElement.children, th);
      setColHover(colIdx);
    });
    thead.addEventListener('mouseleave', () => setColHover(-1));
  })();


  function render() {
    // Sortindikator
    $thead.querySelectorAll('th').forEach(th => th.classList.remove('sort-asc','sort-desc'));
    const activeTh = $thead.querySelector(`th[data-key="${sortKey}"]`);
    if (activeTh) activeTh.classList.add(sortDir === 'asc' ? 'sort-asc' : 'sort-desc');

    const cols = getColumns();
    const data = getVisibleRows();
    const frag = document.createDocumentFragment();

    for (const r of data) {
      const tr = document.createElement('tr');
      tr.dataset.run  = r.run_id;
      tr.dataset.step = r.step_idx;

      // Basis-Spalten in derselben Reihenfolge wie Header
      for (const c of cols) {
        const td = document.createElement('td');
        let v = r[c];
        if (v == null) v = '';
        else if (typeof v === 'number' && Number.isFinite(v)) v = fmtSmartPlain(v);
        td.textContent = v;
        tr.appendChild(td);
      }

      // Zeile hervorheben, wenn aktiv
      if (active.run_id != null && active.step_idx != null &&
          Number(r.run_id) === Number(active.run_id) &&
          Number(r.step_idx) === Number(active.step_idx)) {
        tr.classList.add('active');
      } else if (active.run_id != null && Number(r.run_id) === Number(active.run_id)) {
        tr.classList.add('same-run');
      }

      frag.appendChild(tr);
    }

    $tbody.replaceChildren(frag);
  }

  function exportCSV(rows, cols) {
    if (!rows.length) return;
    const lines = [
      cols.join(','),
      ...rows.map(r => cols.map(c => {
        const val = r[c];
        if (val == null) return '';
        const s = String(val).replace(/"/g,'""');
        return s.includes(',') ? `"${s}"` : s;
      }).join(','))
    ];
    const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = Object.assign(document.createElement('a'), { href: url, download: 'runs_view.csv' });
    document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
  }

  return { init, render };
})();


// ---- Filter-Events ----
function attachFilters() {
  document.getElementById("filter-phase").addEventListener("change", e => {
    FILTER.phase = e.target.value;
    renderList();
  });
  document.getElementById("filter-text").addEventListener("input", e => {
    FILTER.text = e.target.value.trim();
    renderList();
  });
}

// ersetzt die alte selectSample()
function selectSample(idOrRank) {
  // akzeptiere sowohl rank als auch (falls vorhanden) global_idx
  const sample = SAMPLES.find(r =>
    String(r.rank) === String(idOrRank) || String(r.global_idx) === String(idOrRank)
  );
  if (!sample) return;
  selectRank(Number(sample.rank));  // nutzt deine bestehende, funktionierende Pipeline
}



// ---- Boot ----
async function boot() {
  try {
    [SAMPLES, RUNS] = await Promise.all([loadCSV(PATH_SAMPLES), loadCSV(PATH_RUNS)]);
    // cast numerics
    SAMPLES = SAMPLES.map(o => {
      const out = {...o};
      ["rank","global_idx","run_id","step_idx"].forEach(k => out[k] = Number(out[k]));
      ["sigma_0","eps_total","eps_delta","e_s","sigma_delta"].forEach(k => out[k] = Number(out[k]));
      return out;
    });
    RUNS = RUNS.map(o => ({
      run_id: Number(o.run_id),
      step_idx: Number(o.step_idx),
      sigma_0: Number(o.sigma_0),
      eps_total: Number(o.eps_total),
      eps_delta: o.eps_delta !== undefined ? Number(o.eps_delta) : undefined,
      e_s:       o.e_s       !== undefined ? Number(o.e_s)       : undefined,
      sigma_delta: o.sigma_delta !== undefined ? Number(o.sigma_delta) : undefined,
      phase: o.phase ?? undefined,
      cut:   o.cut !== undefined && o.cut !== "" ? Number(o.cut) : undefined,
      ...Object.fromEntries(Object.entries(o).filter(([k]) => k.startsWith("param.")))
    }));


    attachFilters();
    buildRunChips();
    renderList(); // ruft typesetMath() selbst
    RunsTable.init(RUNS);

  } catch (err) {
    console.error(err);
    alert("Fehler beim Laden der CSVs. Prüfe die Pfade und den lokalen Server.");
  }
}

boot();