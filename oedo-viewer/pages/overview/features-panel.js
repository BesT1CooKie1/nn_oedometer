/* pages/overview/features-panel.js
 * Rendert im "Features"-Tab eine Übersicht:
 * - Schema (Defaults) aus schema.json
 * - Additional aus viewer_data/additional_runs.csv (falls vorhanden)
 * - Roh-Variablen (runs.csv)
 * - Verwendete Variablen in Additional-Formeln
 */

const PATH_SCHEMA = "viewer_data/schema.json";
const PATH_RUNS = "viewer_data/runs.csv";
const PATH_ADD = "viewer_data/additional_runs.csv";

const ADD_PREVIEW_BTN_WRAP_ID = "additionalPreviewBtnWrap";

function ensureHiddenCSSOnce() {
    const id = "css-modal-hidden";
    if (document.getElementById(id)) return;
    const s = document.createElement("style");
    s.id = id;
    s.textContent = `.modal[hidden]{display:none!important}`;
    document.head.appendChild(s);
}

ensureHiddenCSSOnce();


document.addEventListener("DOMContentLoaded", () => {
    // Tab-Label vorsichtshalber umbenennen (falls im HTML nicht geändert)
    const procBtn = document.querySelector('.af-seg-btn[data-view="proc"]');
    if (procBtn && procBtn.textContent.trim().toLowerCase() === "process") {
        procBtn.textContent = "Features";
    }

    // Events
    const refreshBtn = document.getElementById("refreshBtn");
    refreshBtn?.addEventListener("click", loadAndRender);

    // Falls dein Add-Features-Modal später ein Event feuert, könnten wir hier auch lauschen:
    window.addEventListener("additional-runs-updated", loadAndRender);

    loadAndRender();
});

function ensureTopHost() {
  // host-Container direkt unter der schema-panel
  let host = document.getElementById("featuresPanelTop");
  if (host) return host;
  const schemaBar = document.querySelector(".schema-panel");
  host = document.createElement("div");
  host.id = "featuresPanelTop";
  host.className = "legend-sec"; // hübscher Rahmen
  // hinter die schema-panel einfügen
  if (schemaBar && schemaBar.parentNode) {
    schemaBar.parentNode.insertBefore(host, schemaBar.nextSibling);
  } else {
    // Fallback: an den Anfang der table-sec
    document.querySelector(".table-sec")?.prepend(host);
  }
  return host;
}


async function loadAndRender() {
    const cont = ensureTopHost();

    // Lade alles, was nötig ist
    const [schema, runsCsv, addCsv] = await Promise.all([
        fetchMaybeJson(PATH_SCHEMA),
        fetchText(PATH_RUNS),
        fetchMaybeText(PATH_ADD),
    ]);

    const runs = parseCsvQuick(runsCsv);
    const add = addCsv ? parseCsvQuick(addCsv) : null;

    // 1) Schema (Defaults)
    const schemaX = Array.isArray(schema?.feature_keys) ? schema.feature_keys : [];
    const schemaY = schema?.target_key ? [schema.target_key] : [];
    // 2) Additional – Feature-/Target-Spalten entzerren
    const additionalFeatures = [];
    const additionalTargets = [];
    if (add) {
        for (const h of add.header) {
            if (h.startsWith("F:")) additionalFeatures.push(h.slice(2));
            else if (h.startsWith("T:")) additionalTargets.push(h.slice(2));
        }
    }

    // 3) Roh-Variablen (alle Spalten aus runs.csv, ohne F:/T:)
    const rawVars = runs.header.filter(h => !(h.startsWith("F:") || h.startsWith("T:")));

    // 4) Verwendete Variablen in Additional-Formeln
    const usedVars = collectUsedVariables(additionalFeatures.concat(additionalTargets), rawVars);

    // Render
    const html = `
    <div class="card-title">Features & Targets</div>
    <div class="card">
       
      <div class="card-title">Aus Rohdaten (runs.csv)</div>
      <div class="card-body">
        <div class="schema-row"><strong>Features:</strong> ${chips(schemaX)}</div>
        <div class="schema-row"><strong>Targets:</strong> ${chips(schemaY)}</div>
        <div class="card-hint"><code>viewer_data/schema.json</code></div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Ergänzt (additional_runs.csv)</div>
      <div class="card-body">
        <div class="schema-row"><strong>Features:</strong> ${chips(additionalFeatures)}</div>
        <div class="schema-row"><strong>Target:</strong> ${chips(additionalTargets)}</div>
        <div class="card-hint"><code>${PATH_ADD}</code> ${add ? "" : "– nicht vorhanden"}</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Roh-Variablen (runs.csv)</div>
      <div class="card-body">
        <div class="schema-row">${chips(rawVars)}</div>
        <div class="card-hint"><code>${PATH_RUNS}</code></div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Verwendete Variablen in "Ergänzt"-Formeln</div>
      <div class="card-body">
        <div class="schema-row">${chips(usedVars)}</div>
        <div class="card-hint">Aus den Ausdrücken in <code>additional_runs.csv</code> extrahiert.</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Final „geladen“ (für Training)</div>
      <div class="card-body">
        <div class="schema-row">
          <strong>Features:</strong>
          ${chips(unique([...schemaX, ...additionalFeatures]))}
        </div>
        <div class="schema-row">
          <strong>Target:</strong>
          ${chips(additionalTargets.length ? additionalTargets : schemaY)}
        </div>
      </div>
    </div>
  `;

    cont.innerHTML = html;

// Inject Preview button when additional file is present
    if (add) {
        const wrap = document.getElementById(ADD_PREVIEW_BTN_WRAP_ID);
        if (wrap) {
            wrap.innerHTML = `<button id="openAddRunsPreview" class="btn" type="button">Preview Additional</button>`;
            document.getElementById("openAddRunsPreview")
                ?.addEventListener("click", () => openAddRunsPreview(add, PATH_RUNS));
        }
    }

}

async function openAddRunsPreview(addParsed, runsPath) {
    // Open modal
    const modal = document.getElementById("addRunsPreviewModal");
    const closeBtn = document.getElementById("addRunsPrevClose");
    const content = document.getElementById("addPrevContent");
    const limitInp = document.getElementById("addPrevLimit");
    const reloadBtn = document.getElementById("addPrevReload");
    if (!modal || !content) return;
    modal.hidden = false;

    const runsCsv = await fetchText(runsPath);
    const runs = parseCsvForPreview(runsCsv);
    const addCsv = await fetchText(PATH_ADD);
    const addFull = parseCsvForPreview(addCsv);

    // Build list of Additional feature/target columns
    const addFeatureCols = addParsed.header.filter(h => h.startsWith("F:"));
    const addTargetCols = addParsed.header.filter(h => h.startsWith("T:"));

    // Helper: find raw vars used in an expression (token-aware)
    const rawVars = runs.header.filter(h => !(h.startsWith("F:") || h.startsWith("T:")));
    const usedVarsOf = (expr) => {
        const used = [];
        const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        rawVars.forEach(name => {
            if (expr.indexOf(name) === -1) return;
            const pat = new RegExp(`(^|[^A-Za-z0-9_$])(${esc(name)})(?![A-Za-z0-9_$])`);
            if (pat.test(expr)) used.push(name);
        });
        return used;
    };
    console.log(rawVars)
    async function render(limit = 20) {
        content.innerHTML = "";
        const blocks = [];

        // Features
        for (const f of addFeatureCols) {
            const expr = f.slice(2);
            const used = usedVarsOf(expr);
            const cols = [...used, f];  // left raw vars, right computed col (from additional file)
            blocks.push(renderMiniTable(`Feature: ${expr}`, cols, runs, addFull, limit));
        }
        // Targets
        for (const t of addTargetCols) {
            const expr = t.slice(2);
            const used = usedVarsOf(expr);
            const cols = [...used, t];
            blocks.push(renderMiniTable(`Target: ${expr}`, cols, runs, addFull, limit));
        }

        if (!blocks.length) {
            content.innerHTML = `<div class="card-hint">Keine Additional-Spalten in <code>${PATH_ADD}</code> gefunden.</div>`;
        } else {
            blocks.forEach(b => content.appendChild(b));
        }
    }

    // first render
    render(Number(limitInp?.value || 20));

    // wiring
    closeBtn?.addEventListener("click", () => modal.hidden = true, {once: true});
    modal.addEventListener("click", (e) => {
        if (e.target === modal) modal.hidden = true;
    });
    reloadBtn?.addEventListener("click", () => render(Number(limitInp?.value || 20)));
}

function renderMiniTable(title, cols, runs, addFull, limit) {
    const card = document.createElement("div");
    card.className = "card";
    const head = document.createElement("div");
    head.className = "card-title";
    head.textContent = title;
    const body = document.createElement("div");
    body.className = "card-body";
    body.style.overflow = "auto";

    // Build header row
    const table = document.createElement("table");
    table.style.width = "100%";
    table.style.borderCollapse = "separate";
    table.style.borderSpacing = "0";
    const thead = document.createElement("thead");
    const trh = document.createElement("tr");
    cols.forEach(c => {
        const th = document.createElement("th");
        th.textContent = c;
        th.style.textAlign = "left";
        th.style.position = "sticky";
        th.style.top = "0";
        th.style.background = "var(--surface, #111)";
        th.style.padding = "6px 8px";
        th.style.borderBottom = "1px solid var(--border,#333)";
        trh.appendChild(th);
    });
    thead.appendChild(trh);
    table.appendChild(thead);

    // Body rows (from additional file, but fetch raw vars from runs)
    const tbody = document.createElement("tbody");
    const N = Math.min(limit, addFull.rows.length);
    for (let i = 0; i < N; i++) {
        const tr = document.createElement("tr");
        cols.forEach(c => {
            const td = document.createElement("td");
            const isComputed = c.startsWith("F:") || c.startsWith("T:");
            const src = isComputed ? addFull : runs;
            const v = src.rows[i]?.[c] ?? "";
            td.textContent = prettyNum(v);
            td.style.padding = "6px 8px";
            td.style.borderBottom = "1px dashed var(--border,#333)";
            if (isComputed) td.style.fontWeight = "600";
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    }
    table.appendChild(tbody);

    body.appendChild(table);
    const hint = document.createElement("div");
    hint.className = "card-hint";
    hint.textContent = `erste ${N} Zeilen`;
    body.appendChild(hint);

    card.appendChild(head);
    card.appendChild(body);
    return card;
}

function parseCsvForPreview(text) {
    const lines = text.replace(/\r/g, "").trim().split("\n");
    const header = splitCsvLine(lines[0]).map(s => s.trim());
    const rows = [];
    for (let i = 1; i < lines.length; i++) {
        const cols = splitCsvLine(lines[i]);
        const o = {};
        header.forEach((h, j) => o[h] = cols[j] ?? "");
        rows.push(o);
    }
    return {header, rows};
}

function prettyNum(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return String(v ?? "");
    // compact numeric look
    return Math.abs(n) >= 1e6 || (Math.abs(n) > 0 && Math.abs(n) < 1e-3)
        ? n.toExponential(3)
        : (+n.toFixed(6)).toString();
}


/* ------------------ helpers ------------------ */

async function fetchText(url) {
    const res = await fetch(url, {cache: "no-store"});
    if (!res.ok) throw new Error(`${url}: ${res.status}`);
    return await res.text();
}

async function fetchMaybeText(url) {
    try {
        const r = await fetch(url, {cache: "no-store"});
        if (r.ok) return await r.text();
    } catch {
    }
    return null;
}

async function fetchMaybeJson(url) {
    try {
        const r = await fetch(url, {cache: "no-store"});
        if (r.ok) return await r.json();
    } catch {
    }
    return null;
}

function parseCsvQuick(text) {
    const lines = text.replace(/\r/g, "").trim().split("\n");
    const header = splitCsvLine(lines[0]).map(s => s.trim());
    // Zeilen brauchen wir hier nicht – wir rendern nur Übersichten
    return {header};
}

function splitCsvLine(line) {
    const out = [];
    let cur = "", inQ = false;
    for (let i = 0; i < line.length; i++) {
        const ch = line[i];
        if (ch === '"') {
            if (inQ && line[i + 1] === '"') {
                cur += '"';
                i++;
            } else inQ = !inQ;
        } else if (ch === "," && !inQ) {
            out.push(cur);
            cur = "";
        } else cur += ch;
    }
    out.push(cur);
    return out;
}

function chips(arr) {
    if (!arr || !arr.length) return `<span class="badge muted">–</span>`;
    return arr.map(x => `<span class="badge">${escapeHtml(x)}</span>`).join(" ");
}

function unique(arr) {
    return [...new Set(arr)];
}

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }[c]));
}

/** Analysiert, welche Roh-Variablen in den Additional-Formeln vorkommen */
function collectUsedVariables(exprs, candidates) {
    if (!exprs || !exprs.length) return [];
    const used = new Set();
    // präzise Token-Matches (kein Teilstring)
    const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const pats = candidates.map(name =>
        new RegExp(`(^|[^A-Za-z0-9_$])(${esc(name)})(?![A-Za-z0-9_$])`)
    );
    for (const expr of exprs) {
        for (let i = 0; i < candidates.length; i++) {
            if (expr.indexOf(candidates[i]) === -1) continue;
            if (pats[i].test(expr)) used.add(candidates[i]);
        }
    }
    return [...used];
}

// ---- Globale Schema-State + Persistenz ----
const SCHEMA_LS_KEY = "oedo:finalSchema";

function loadSchemaFromLS() {
  try { return JSON.parse(localStorage.getItem(SCHEMA_LS_KEY) || "null"); }
  catch { return null; }
}
function saveSchemaToLS(schema) {
  try { localStorage.setItem(SCHEMA_LS_KEY, JSON.stringify(schema)); } catch {}
}

// Rufe das hier auf, sobald du deine finalen Keys kennst:
function publishSchema({ rawKeys, featureKeys, targetKey }) {
  const schema = {
    rawKeys: Array.from(new Set(rawKeys || [])),
    featureKeys: Array.from(new Set(featureKeys || [])),
    targetKey: targetKey || null,
    updatedAt: Date.now()
  };
  window.__oedoSchema = schema;
  saveSchemaToLS(schema);
  window.dispatchEvent(new CustomEvent("schema:updated", { detail: schema }));
}

// Beispiel: nachdem du im „Add Feature“-Modal auf „Speichern“ klickst
// und deine finale Featureliste + Target festlegst:
function onApplyExpressions(resolvedFeatureKeys, targetKey, availableRawKeys) {
  publishSchema({
    rawKeys: availableRawKeys,
    featureKeys: resolvedFeatureKeys,   // final aktiv (roh + abgeleitet)
    targetKey
  });
}
