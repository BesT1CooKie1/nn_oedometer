// pages/overview/overview.js
const SAMPLES_CSV = "viewer_data/samples.csv";
const RUNS_CSV    = "viewer_data/runs.csv";

// Mappe af-* Klassen auf generische Klassen für Styling & JS-Hooks
document.querySelector(".af-segmented")?.classList.add("segmented");
document.querySelectorAll(".af-seg-btn").forEach(b => b.classList.add("seg-btn"));

const $  = sel => document.querySelector(sel);
const $$ = sel => Array.from(document.querySelectorAll(sel));

const uniq = arr => Array.from(new Set(arr));
const renderChips = (containerSel, items) => {
  const el = document.querySelector(containerSel);
  if (!el) return;
  el.innerHTML = (items && items.length)
    ? items.map(t => `<span class="pill">${t}</span>`).join(" ")
    : `<span class="muted">–</span>`;
};
const attachToggle = (btnSel, chipsSel) => {
  const btn = document.querySelector(btnSel);
  const chips = document.querySelector(chipsSel);
  if (!btn || !chips) return;
  const update = () => {
    const expanded = chips.classList.toggle("expanded");
    btn.setAttribute("aria-expanded", expanded ? "true" : "false");
    btn.textContent = expanded ? "weniger" : "mehr";
  };
  btn.onclick = update;
};

function setCount(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = String(value);
}

// -------- CSV ----------
function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  if (!lines.length) return { header: [], rows: [] };
  const header = lines[0].split(",").map(s => s.trim());
  const rows = lines.slice(1).map(line => {
    const parts = line.split(",").map(s => s.trim());
    const obj = {};
    header.forEach((h, i) => {
      const v = parts[i];
      const n = Number(v);
      obj[h] = Number.isFinite(n) ? n : (v === "" ? null : v);
    });
    return obj;
  });
  return { header, rows };
}
async function fetchCSV(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`Fehler beim Laden: ${url}`);
  return parseCSV(await res.text());
}

// -------- Schema-Persistenz ----------
const SCHEMA_LS_KEY = "oedo:finalSchema";
function loadSchema() {
  if (window.__oedoSchema) return window.__oedoSchema;
  try { return JSON.parse(localStorage.getItem(SCHEMA_LS_KEY) || "null"); }
  catch { return null; }
}
function saveSchema(schema) {
  try { localStorage.setItem(SCHEMA_LS_KEY, JSON.stringify(schema)); } catch {}
  window.__oedoSchema = schema;
}
function publishSchema(schema) {
  saveSchema({ ...schema, updatedAt: Date.now() });
  window.dispatchEvent(new CustomEvent("schema:updated", { detail: schema }));
}

// Optional: Map alias->Ausdruck (z. B. vom „Add Feature“-Modal) aus LS
function loadAliasExprMap() {
  try { return JSON.parse(localStorage.getItem("oedo:aliasExprMap") || "{}"); }
  catch { return {}; }
}

// -------- Ausdrucks-Evaluator (Whitelist) ----------
const SAFE_FUN = {
  abs: Math.abs, sign: Math.sign, sqrt: Math.sqrt, log: Math.log, exp: Math.exp,
  maximum: (a,b)=>Math.max(a,b), minimum: (a,b)=>Math.min(a,b), pow: Math.pow,
  clip: (x, lo, hi)=>Math.min(Math.max(x, lo), hi),
  where: (cond, a, b)=> (cond ? a : b),
};
function evalExprOnRow(expr, row) {
  const exprSafe = String(expr)
    .replace(/\bmaximum\s*\(/g, "SAFE_FUN.maximum(")
    .replace(/\bminimum\s*\(/g, "SAFE_FUN.minimum(")
    .replace(/\bpow\s*\(/g, "SAFE_FUN.pow(")
    .replace(/\bclip\s*\(/g, "SAFE_FUN.clip(")
    .replace(/\bwhere\s*\(/g, "SAFE_FUN.where(")
    .replace(/\babs\s*\(/g, "SAFE_FUN.abs(")
    .replace(/\bsign\s*\(/g, "SAFE_FUN.sign(")
    .replace(/\bsqrt\s*\(/g, "SAFE_FUN.sqrt(")
    .replace(/\blog\s*\(/g, "SAFE_FUN.log(")
    .replace(/\bexp\s*\(/g, "SAFE_FUN.exp(");

  if (/[;{}[\]]/.test(exprSafe)) throw new Error("Unerlaubtes Token im Ausdruck.");
  // eslint-disable-next-line no-new-func
  const f = new Function("row", "SAFE_FUN", `with(row){ return (${exprSafe}); }`);
  return f(row, SAFE_FUN);
}
function ensureDerivedColumns(rows, aliasToExpr) {
  if (!rows || !rows.length || !aliasToExpr) return rows || [];
  const aliases = Object.keys(aliasToExpr);
  if (!aliases.length) return rows;
  return rows.map(r => {
    const out = { ...r };
    for (const alias of aliases) {
      if (alias in out) continue;
      try {
        const val = evalExprOnRow(aliasToExpr[alias], out);
        out[alias] = Number.isFinite(val) ? val : null;
      } catch {
        out[alias] = null;
      }
    }
    return out;
  });
}

// -------- Z-Score-Normierung ----------
function zscoreNormalize(rows, cols) {
  if (!rows.length || !cols.length) return { rows: [], stats: {} };
  const stats = {};
  cols.forEach(c => {
    const arr = rows.map(r => Number(r[c]));
    const mean = arr.reduce((a, b) => a + (Number.isFinite(b) ? b : 0), 0) / arr.length;
    const varSum = arr.reduce((a, b) => {
      const x = Number.isFinite(b) ? b : mean;
      return a + (x - mean) * (x - mean);
    }, 0);
    const std = Math.sqrt(varSum / Math.max(1, arr.length - 1)) || 1;
    stats[c] = { mean, std };
  });
  const norm = rows.map(r => {
    const o = {};
    cols.forEach(c => { o[`${c}_norm`] = (Number(r[c]) - stats[c].mean) / stats[c].std; });
    return o;
  });
  return { rows: norm, stats };
}

// -------- Tabellen-Renderer ----------
function renderTable(container, data, columns, maxRows) {
  if (!container) return;
  const rows = data.slice(0, maxRows);
  let sort = { col: null, asc: true };

  const table = document.createElement("table");
  table.className = "table data-table";
  const thead = document.createElement("thead");
  const trh = document.createElement("tr");

  columns.forEach((c, idx) => {
    const th = document.createElement("th");
    th.textContent = c;
    th.dataset.colIndex = String(idx);
    th.tabIndex = 0;
    th.title = "Klicken zum Sortieren";
    th.addEventListener("click", () => {
      if (sort.col === idx) sort.asc = !sort.asc;
      else { sort.col = idx; sort.asc = true; }
      rows.sort((a, b) => {
        const va = a[c]; const vb = b[c];
        const na = typeof va === "number" ? va : Number(va);
        const nb = typeof vb === "number" ? vb : Number(vb);
        const A = Number.isFinite(na) ? na : String(va);
        const B = Number.isFinite(nb) ? nb : String(vb);
        return sort.asc ? (A > B ? 1 : A < B ? -1 : 0) : (A < B ? 1 : A > B ? -1 : 0);
      });
      renderBody(); highlightCol(idx);
    });
    trh.appendChild(th);
  });
  thead.appendChild(trh);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  table.appendChild(tbody);
  container.innerHTML = "";
  container.appendChild(table);

  function fmt(v) {
    if (typeof v === "number" && Number.isFinite(v)) {
      if (Math.abs(v) > 1e6 || (Math.abs(v) > 0 && Math.abs(v) < 1e-3)) return v.toExponential(3);
      return String(v);
    }
    return v ?? "";
  }
  function renderBody() {
    tbody.innerHTML = "";
    rows.forEach((r) => {
      const tr = document.createElement("tr");
      columns.forEach((c, idx) => {
        const td = document.createElement("td");
        td.textContent = fmt(r[c]);
        td.dataset.colIndex = String(idx);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
  }
  function highlightCol(colIdx) {
    $$(".col-highlight").forEach(el => el.classList.remove("col-highlight"));
    $$(`th[data-col-index="${colIdx}"]`).forEach(el => el.classList.add("col-highlight"));
    $$(`td[data-col-index="${colIdx}"]`).forEach(el => el.classList.add("col-highlight"));
  }
  table.addEventListener("mousemove", (ev) => {
    const cell = ev.target.closest("td,th");
    if (!cell) return;
    highlightCol(cell.dataset.colIndex);
  });

  renderBody();
}

// -------- View-Sichtbarkeit ----------
function activateView(name) {
  const show = {
    raw:  ["#rawTableWrap",  "#procTableWrap", "#normTableWrap"],
    proc: ["#procTableWrap", "#rawTableWrap",  "#normTableWrap"],
    norm: ["#normTableWrap", "#rawTableWrap",  "#procTableWrap"],
  }[name] || ["#rawTableWrap", "#procTableWrap", "#normTableWrap"];
  show.forEach((sel, i) => {
    const el = $(sel);
    if (el) el.hidden = i !== 0;
  });
}

// -------- Export ----------
function exportCSV(rows, cols, filename = "process_data.csv") {
  const head = cols.join(",");
  const body = rows.map(r => cols.map(c => r[c] ?? "").join(",")).join("\n");
  const blob = new Blob([head + "\n" + body], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

// -------- Heuristik: Schema aus RUNS ableiten ----------
function inferSchemaFromRuns(runHeader) {
  if (!runHeader || !runHeader.length) return null;
  const blacklist = new Set(["id","run_id","ts","timestamp"]);
  const guessTarget = runHeader.find(k => /^y$|^target$|^e_s$|^E_s$|^y_true$/i.test(k)) || null;
  const featureKeys = runHeader.filter(k => !blacklist.has(k) && k !== guessTarget);
  return { rawKeys: featureKeys.slice(), feature_keys: featureKeys, target_key: guessTarget, updatedAt: Date.now() };
}

// -------- Globaler State ----------
let SAMPLES = null;
let RUNS    = null;
let CURRENT_VIEW = "raw";
let CACHED_PROC_ROWS = [];  // zuletzt gerenderte Proc-Zeilen (für Export)

// -------- Main ----------
(async function main() {
  const [samples, runs] = await Promise.all([
    fetchCSV(SAMPLES_CSV),
    fetchCSV(RUNS_CSV).catch(() => ({ header: [], rows: [] })),
  ]);
  SAMPLES = samples.rows;
  RUNS    = runs.rows;

  setCount("samplesCount", SAMPLES.length);
  setCount("runsCount", RUNS.length);

  // Schema laden oder aus RUNS ableiten
  let schema = loadSchema();
  if (!schema) {
    schema = inferSchemaFromRuns(runs.header);
    if (schema) saveSchema(schema);
  }

  // Chips „alle X“ / „alle Y-Kandidaten“
  const allXKeys = schema?.feature_keys ? uniq(schema.feature_keys) : uniq(runs.header.filter(k => k !== schema?.target_key));
  const yCand = uniq(
    [schema?.target_key, "e_s", "E_s", "target", "y", "y_true"]
      .filter(Boolean)
      .filter(k => runs.header.includes(k) || samples.header.includes(k))
  );
  renderChips("#allXKeys", allXKeys);
  renderChips("#allYKeys", yCand);
  attachToggle("#toggleAllX", "#allXKeys");
  attachToggle("#toggleAllY", "#allYKeys");

  // --- Renderer: RAW / PROC / NORM ---
  const maxRowsInput = $("#maxRows");
  const getMax = () => Math.max(5, Math.min(500, Number(maxRowsInput?.value || 20)));

  function renderRAW() {
    renderTable($("#rawTableWrap"), SAMPLES, samples.header, getMax());
  }

  function renderPROC() {
    // Basis sind RUNS (Trainings-Kandidaten)
    const baseRows = RUNS || [];
    const aliasToExpr = loadAliasExprMap();
    const withDerived = ensureDerivedColumns(baseRows, aliasToExpr);

    // Spalten aus Schema (dynamisch) – plus optional Target hinten
    let cols = [];
    if (schema?.feature_keys?.length) cols = [...schema.feature_keys];
    else cols = inferSchemaFromRuns(runs.header)?.feature_keys ?? runs.header.slice();

    if (schema?.target_key && !cols.includes(schema.target_key)) cols.push(schema.target_key);

    CACHED_PROC_ROWS = withDerived;
    renderTable($("#procTableWrap"), withDerived, cols, getMax());
  }

  function renderNORM() {
    // Normiere die aktuell sichtbaren Feature-Spalten (aus Schema)
    const currentCols = (schema?.feature_keys?.length ? [...schema.feature_keys] : inferSchemaFromRuns(runs.header)?.feature_keys ?? []);
    if (schema?.target_key) currentCols.push(schema.target_key);

    if (!currentCols.length || !RUNS.length) {
      renderTable($("#normTableWrap"), [], [], getMax());
      return;
    }
    const aliasToExpr = loadAliasExprMap();
    const rowsForNorm = ensureDerivedColumns(RUNS, aliasToExpr);

    const { rows: normRows } = zscoreNormalize(rowsForNorm, currentCols);
    const normCols = currentCols.map(c => `${c}_norm`);
    renderTable($("#normTableWrap"), normRows, normCols, getMax());
  }

  function renderAll() {
    renderRAW();
    renderPROC();
    renderNORM();
  }

  // Buttons / Events
  $("#maxRows")?.addEventListener("change", renderAll);
  $("#exportProcCsv")?.addEventListener("click", () => {
    let cols = [];
    if (schema?.feature_keys?.length) cols = [...schema.feature_keys];
    else cols = inferSchemaFromRuns(runs.header)?.feature_keys ?? runs.header.slice();
    if (schema?.target_key && !cols.includes(schema.target_key)) cols.push(schema.target_key);
    exportCSV(CACHED_PROC_ROWS, cols, "processed_features.csv");
  });

  // View-Schalter
  $$(".seg-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const v = btn.dataset.view;
      document.querySelectorAll(".seg-btn").forEach(b => {
        b.classList.toggle("active", b === btn);
        b.setAttribute("aria-selected", b === btn ? "true" : "false");
      });
      CURRENT_VIEW = v;
      activateView(v);
      if (v === "proc") renderPROC();
      if (v === "norm") renderNORM();
      if (v === "raw")  renderRAW();
    });
  });

  // Schema-Updates (z. B. nach Speichern im Add-Feature-Modal)
  window.addEventListener("schema:updated", (ev) => {
    schema = loadSchema();
    // Chips aktualisieren
    const allX = schema?.feature_keys ? uniq(schema.feature_keys) : uniq(runs.header.filter(k => k !== schema?.target_key));
    renderChips("#allXKeys", allX);
    renderChips("#allYKeys", uniq([schema?.target_key].filter(Boolean)));
    if (CURRENT_VIEW === "proc") renderPROC();
    if (CURRENT_VIEW === "norm") renderNORM();
  });

  // Refresh-Button oben (falls vorhanden)
  $("#refreshBtn")?.addEventListener("click", renderAll);

  // Default-View: RAW
  activateView("raw");
  CURRENT_VIEW = "raw";
  renderAll();
})();
