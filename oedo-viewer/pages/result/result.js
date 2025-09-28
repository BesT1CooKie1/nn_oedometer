// result.js
// Adds split-aware views (All / Train / Val / Test) across chips, list, table, and plot.
// - Loads runs_final.csv and split_assignments.csv
// - Chips show only runs that have at least 1 sample in the chosen split
// - Samples list is filtered to the chosen split
// - Compare table shows the full selected run (so you see the whole run), but
//   highlights rows that belong to the split
// - Plot shows full run curves and overlays markers for points in the chosen split
//
// Expected IDs in HTML (adjust in SEL if yours differ):
//   #run-chips, #samples-list
//   #tbl-sample tbody, #tbl-params tbody, #tbl-preds tbody
//   #tbl-compare thead, #tbl-compare tbody
//   #toggle-delta (checkbox), #btn-export-csv (optional)
//   #plot-se  (plot container)
//   #split-filter (select with values: all/train/val/test)  <-- optional control

// ---------- Config ----------
const PATH_RUNS_FINAL = "../../viewer_data/runs_final.csv";
const PATH_RUNS_FINAL_PRED_PROP = "../../viewer_data/predictions_additional_processed_propagated.csv";
const PATH_RUNS_FINAL_PRED_PROP_REC = "../../viewer_data/predictions_additional_processed_propagated_recursive.csv";
const PATH_SPLITS = "../../viewer_data/split_assignments.csv";

// Colors (feel free to tweak)
const COLORS = {
    real: "#2196F3",     // blue
    pred: "#FF9800",     // orange
    activeReal: "#E91E63", // pink
    activePred: "#FFC107", // amber
    activePredProp: "#5e5e5e",
    activePredPropRec: "#E91E63",
    split: {
        train: "#4CAF50",  // green
        val: "#FFC107",  // amber
        test: "#9C27B0",  // purple
    }
};

// ---------- Selectors (change here to match your HTML if needed) ----------
const SEL = {
    plot: "#plot-se",
    runChips: "#run-chips",
    samplesList: "#samples-list",
    tblSampleBody: "#tbl-sample tbody",
    tblParamsBody: "#tbl-params tbody",
    tblPredsBody: "#tbl-preds tbody",
    cmpThead: "#tbl-compare thead",
    cmpTbody: "#tbl-compare tbody",
    splitFilter: "#split-filter", // <select> All/Train/Val/Test
};

// ---------- Tiny Utils ----------
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const byNum = (a, b) => a - b;

// --- Split assignments (global_idx -> split) ---
let SPLIT_OF = new Map();
let SPLITS_LOADED = false;

async function loadSplitsOnce() {
    if (SPLITS_LOADED) return;
    const res = await fetch(PATH_SPLITS, {cache: "no-store"});
    if (!res.ok) return;
    const text = await res.text();
    const {header, rows} = parseCSV(text);
    if (!header.length) return;

    const keyCol = header[0];              // first column is global_idx
    const splitCol = header.includes("split") ? "split" : header[header.length - 1];

    SPLIT_OF.clear();
    for (const r of rows) {
        const gid = Number(r[keyCol]);
        if (Number.isFinite(gid)) {
            const s = normalizeSplit(r[splitCol]);
            SPLIT_OF.set(gid, s);
        }
    }
    SPLITS_LOADED = true;

    // If ROWS already present, annotate with .split (by global_idx if available)
    if (ROWS && ROWS.length) {
        for (const r of ROWS) {
            const gid = Number(r.global_idx ?? r.global_index);
            r.split = Number.isFinite(gid) ? (SPLIT_OF.get(gid) ?? null) : (r.split ?? null);
        }
    }
}

// helper to ensure splits before plotting / listing
async function ensureSplits() {
    try {
        await loadSplitsOnce();
    } catch (_e) {
    }
}

function renderSplitLegendRow() {
    // Ensure a host element exists directly after the plot
    const plot = $(SEL.plot);
    if (!plot) return;

    let host = document.querySelector("#split-legend-row");
    if (!host) {
        host = document.createElement("div");
        host.id = "split-legend-row";
        // one-line row style; lightweight so it blends with your UI
        host.style.display = "flex";
        host.style.alignItems = "center";
        host.style.gap = "12px";
        host.style.margin = "6px 4px 0 4px";
        host.style.fontSize = "0.9rem";
        host.style.flexWrap = "nowrap";   // prevent items from wrapping to next line

        plot.insertAdjacentElement("afterend", host);
    }

    // Build the row: [Dataset] • [Train] [Val] [Test] • [Filter: X] • [□ Show split markers]
    const dot = (color) => `<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${color};margin-right:6px;"></span>`;
    const tag = (label, color) => `<span style="display:inline-flex;align-items:center;gap:6px;margin-right:10px;">${dot(color)}<span>${label}</span></span>`;

    const train = tag("Training", COLORS.split.train);
    const val = tag("Validation", COLORS.split.val);
    const test = tag("Test", COLORS.split.test);

    const filterTxt = `Filter: <b>${SPLIT_FILTER.toUpperCase()}</b>`;
    const chkId = "chk-show-split-markers";

    host.innerHTML = `
        <span style="opacity:.9;">Datensatz Versuchswerte</span>
        <span aria-hidden="true" style="opacity:.6;">•</span>
        ${train}${val}${test}
        <span aria-hidden="true" style="opacity:.6;">•</span>
        <span style="opacity:.9;">${filterTxt}</span>
        <span aria-hidden="true" style="opacity:.6;">•</span>
        <label for="${chkId}" style="display:inline-flex;align-items:center;gap:6px;cursor:pointer;opacity:.9;">
          <input id="${chkId}" type="checkbox" ${SHOW_SPLIT_MARKERS ? "checked" : ""} />
          <span>Marker</span>
        </label>
    `;

    // Wire the checkbox
    const chk = host.querySelector("#" + chkId);
    if (chk) {
        chk.onchange = () => {
            SHOW_SPLIT_MARKERS = !!chk.checked;
            // re-render current plot with the new setting
            renderPlot(ACTIVE.run_id, {step_idx: ACTIVE.step_idx});
        };
    }
}

// Simple human-readable formatter for numbers
function fmt(x) {
    const n = Number(x);
    if (!Number.isFinite(n)) return (x === null || x === undefined) ? "—" : String(x);
    const a = Math.abs(n);
    const trim = s => s
        .replace(/(\.\d*?[1-9])0+$/, "$1")  // strip trailing zeros
        .replace(/\.0+$/, "")               // strip ".0"
        .replace(/^-0(?:\.0+)?$/, "0");     // "-0" -> "0"
    let digits = 2;
    if (a >= 1e4) digits = 0; else if (a >= 100) digits = 1; else if (a >= 1) digits = 2; else if (a >= 1e-1) digits = 3; else if (a >= 1e-2) digits = 4; else if (a >= 1e-3) digits = 5;
    return trim(n.toFixed(digits));
}

// Minimal CSV parser (no quotes-with-commas handling)
function parseCSV(text) {
    const lines = text.trim().split(/\r?\n/);
    if (!lines.length) return {header: [], rows: []};
    const split = s => s.split(",").map(x => x.trim());
    const header = split(lines[0]);
    const rows = lines.slice(1).map(line => {
        const parts = split(line);
        const obj = {};
        for (let i = 0; i < header.length; i++) {
            const key = header[i];
            const raw = parts[i] ?? "";
            const num = Number(raw);
            obj[key] = (raw !== "" && Number.isFinite(num)) ? num : (raw === "" ? null : raw);
        }
        return obj;
    });
    return {header, rows};
}

// ---------- App State ----------
let ROWS = [];                        // all samples/steps from runs_final.csv (augmented with .split)
let ACTIVE = {run_id: null, step_idx: null}; // current selection
let PAIRS = [];                       // [ [realKey, predKey], ... ] for compare table
let TABLE_RUN_FILTER = null;
let SPLIT_FILTER = "test";             // 'all' | 'train' | 'val' | 'test'
const PROP_MAP = new Map();
const REC_MAP = new Map();   // NEW: recursive propagation

/** Which traces are currently visible in the plot (drives table columns) */
let TABLE_VIEW = {showTruth: true, showProp: true, showPred: false};
let SHOW_SPLIT_MARKERS = true;  // NEW: user toggle for split overlay markers

/** one-time wiring flag for plot events */
let PLOT_EVENTS_WIRED = false;


function safeStepIdx(r) {
    // Prefer numeric step_idx, else numeric rank; else null
    const s = Number(r.step_idx);
    if (Number.isFinite(s)) return s;
    const rk = Number(r.rank);
    return Number.isFinite(rk) ? rk : null;
}

// Keys that represent parameters: filter them out from the basic info table
function isParamKey(k) {
    const s = String(k);
    return /^param[._]/.test(s) || ["cut", "sigma_min", "sigma_max", "sigma_prime_p"].includes(s);
}

// Find (real, pred) value pairs based on keys like pred_<realKey>
function getComparePairs(header) {
    const hset = new Set(header);
    const out = [];
    header.forEach(k => {
        if (/^pred_/.test(k)) {
            const base = k.replace(/^pred_/, "");
            if (hset.has(base)) out.push([base, k]);
        }
    });
    return out;
}

// Split helpers
function normalizeSplit(s) {
    if (!s) return null;
    const t = String(s).trim().toLowerCase();
    if (t === "train" || t === "training") return "train";
    if (t === "val" || t === "validation" || t === "validation") return "val";
    if (t === "test" || t === "testing") return "test";
    return null;
}

function rowsForCurrentSplit() {
    if (SPLIT_FILTER === "all") return ROWS.slice();
    return ROWS.filter(r => r.split === SPLIT_FILTER);
}

// ---------- Rendering: Run chips (split-filtered) ----------
function renderRunChips() {
    const host = $(SEL.runChips);
    if (!host) return;

    // Compute counts from split-filtered rows
    const filtered = rowsForCurrentSplit();
    const counts = filtered.reduce((m, r) => {
        const rid = Number(r.run_id);
        m[rid] = (m[rid] || 0) + 1;
        return m;
    }, {});
    const runIds = Object.keys(counts).map(Number).sort(byNum);

    host.innerHTML = [`<span class="chip ${ACTIVE.run_id === null ? "active" : ""}" data-run="*">Alle (${filtered.length})</span>`, ...runIds.map(rid => `<span class="chip ${ACTIVE.run_id === rid ? "active" : ""}" data-run="${rid}">Run ${rid} (${counts[rid]})</span>`)].join(" ");

    // Delegated click listener is installed in installEvents()
}

// ---------- Rendering: Samples list (split-filtered) ----------
function renderSamplesList() {
    const host = $(SEL.samplesList);
    if (!host) return;

    const pool = rowsForCurrentSplit();
    const rows = (ACTIVE.run_id === null) ? pool : pool.filter(r => Number(r.run_id) === Number(ACTIVE.run_id));

    rows.sort((a, b) => (Number(a.run_id) - Number(b.run_id)) || ((a.step_idx ?? a.rank ?? 0) - (b.step_idx ?? b.rank ?? 0)));

    host.innerHTML = rows.map((r, i) => {
        const isActive = (Number(r.run_id) === Number(ACTIVE.run_id)) && ((r.step_idx ?? r.rank) === ACTIVE.step_idx);
        const step = safeStepIdx(r);
        const badge = r.split ? `<span class="pill split-${r.split}">${r.split}</span>` : "";
        return `
      <div class="list-item ${isActive ? "active" : ""}" data-run_id="${r.run_id}" data-step_idx="${step}">
        <div class="row spaced">
          <div><span class="pill">run ${fmt(r.run_id)}</span> <span class="pill">step=${fmt(step)}</span></div>
          <div class="muted">${r.phase ?? ""} ${badge}</div>
        </div>
        <div class="row spaced muted">
          <div>σ₀: ${fmt(r.sigma_0)}</div>
          <div>ε_total: ${fmt(r.eps_total)}</div>
        </div>
      </div>`;
    }).join("");
}

// ---------- Rendering: Current sample / params / preds ----------
function renderCurrentSample(row) {
    const tb = $(SEL.tblSampleBody);
    if (!tb) return;
    if (!row) {
        tb.innerHTML = `<tr><th class="muted">no sample selected</th><td class="muted" style="text-align:right;">—</td></tr>`;
        return;
    }

    const entries = Object.entries(row)
        .filter(([k]) => !String(k).startsWith("pred_") && !isParamKey(k))
        .sort(([a], [b]) => a.localeCompare(b))
        .slice(0, 24);

    tb.innerHTML = entries.map(([k, v]) => `<tr><th>${k}</th><td>${fmt(v)}</td></tr>`).join("");
}

function renderParams(row) {
    const tb = $(SEL.tblParamsBody);
    if (!tb) return;

    if (!row) {
        tb.innerHTML = `<tr><th class="muted">no params</th><td class="muted" style="text-align:right;">—</td></tr>`;
        return;
    }

    const entries = Object.entries(row)
        .filter(([k]) => isParamKey(k))
        .sort(([a], [b]) => a.localeCompare(b));

    tb.innerHTML = (entries.length ? entries.map(([k, v]) => `<tr><th>${k}</th><td>${fmt(v)}</td></tr>`).join("") : `<tr><th class="muted">no params</th><td class="muted" style="text-align:right;">—</td></tr>`);
}

function renderPreds(row) {
    const tb = $(SEL.tblPredsBody);
    if (!tb) return;

    if (!row) {
        tb.innerHTML = `<tr><th class="muted">no predictions</th><td class="muted" style="text-align:right;">—</td></tr>`;
        return;
    }

    const entries = Object.entries(row)
        .filter(([k]) => String(k).startsWith("pred_"))
        .sort(([a], [b]) => a.localeCompare(b));

    tb.innerHTML = (entries.length ? entries.map(([k, v]) => `<tr><th>${k.replace(/^pred_/, "")}</th><td>${fmt(v)}</td></tr>`).join("") : `<tr><th class="muted">no predictions</th><td class="muted" style="text-align:right;">—</td></tr>`);
}

function updateTableViewFromPlot(div) {
    const gd = div;
    if (!gd || !gd.data) return;

    let showTruth = false, showProp = false, showPred = false, showPropRec = false;

    for (const tr of gd.data) {
        const role = tr.meta && tr.meta.role;
        const vis = (tr.visible === undefined || tr.visible === true)
            ? true
            : (tr.visible === 'legendonly' ? false : !!tr.visible);
        if (role === 'truth') showTruth = showTruth || vis;
        if (role === 'prop') showProp = showProp || vis;
        if (role === 'pred') showPred = showPred || vis;
        if (role === 'prop_rec') showPropRec = showPropRec || vis;  // NEW
    }

    TABLE_VIEW = {showTruth, showProp, showPred, showPropRec};
}


// ---------- Plot (Plotly) ----------
async function renderPlot(run_id, opts = {}) {
    await ensureSplits();

    const {step_idx: hiStep = null} = opts;
    const div = $(SEL.plot);
    if (!div || !window.Plotly) return;

    // Truth rows (not split-filtered)
    const runRows = (run_id === null) ? ROWS.slice()
        : ROWS.filter(r => Number(r.run_id) === Number(run_id));
    const truthSorted = runRows.slice().sort((a, b) => (a.step_idx ?? a.rank ?? 0) - (b.step_idx ?? b.rank ?? 0));

    // Truth arrays
    const x_truth = [], y_truth = [], steps = [], splits = [], x_pred = [];
    let hasPred = false;

    for (const r of truthSorted) {
        const eps = (r.eps_total != null) ? Number(r.eps_total) : NaN;
        const s0 = (r.sigma_0 != null) ? Number(r.sigma_0) : NaN;
        const p0 = (r.pred_sigma_0 != null) ? Number(r.pred_sigma_0) : NaN;

        if (Number.isFinite(eps) && Number.isFinite(s0)) {
            y_truth.push(eps);
            x_truth.push(s0);
            x_pred.push(Number.isFinite(p0) ? (hasPred = true, p0) : null);
            steps.push(r.step_idx ?? r.rank ?? null);

            const gid = Number(r.global_idx ?? r.global_index);
            const sp = r.split ?? (Number.isFinite(gid) ? SPLIT_OF.get(gid) : null);
            splits.push(sp ?? null);
        }
    }

    // Direct propagated (for selected run)
    let x_prop = [], y_prop = [], steps_prop = [];
    if (run_id !== null && PROP_MAP.has(Number(run_id))) {
        const mp = PROP_MAP.get(Number(run_id));
        const propSorted = Array.from(mp.values()).sort((a, b) => (a.step_idx - b.step_idx));
        for (const pr of propSorted) {
            if (pr.sigma_0 != null && pr.eps_total != null) {
                const s0p = Number(pr.sigma_0);
                const ep = Number(pr.eps_total);
                if (Number.isFinite(s0p) && Number.isFinite(ep)) {
                    x_prop.push(s0p);
                    y_prop.push(ep);
                    steps_prop.push(pr.step_idx);
                }
            }
        }
    }

    // NEW: Recursive propagated (for selected run)
    let x_rec = [], y_rec = [], steps_rec = [];
    if (run_id !== null && REC_MAP.has(Number(run_id))) {
        const mp = REC_MAP.get(Number(run_id));
        const recSorted = Array.from(mp.values()).sort((a, b) => (a.step_idx - b.step_idx));
        for (const rr of recSorted) {
            if (rr.sigma_0_rec != null && rr.eps_total_rec != null) {
                const s0r = Number(rr.sigma_0_rec);
                const er = Number(rr.eps_total_rec);
                if (Number.isFinite(s0r) && Number.isFinite(er)) {
                    x_rec.push(s0r);
                    y_rec.push(er);
                    steps_rec.push(rr.step_idx);
                }
            }
        }
    }

    // --- vertical guides need a y-span across all series present ---
    const yAll = []
        .concat(y_truth)
        .concat(Array.isArray(y_prop) ? y_prop : [])
        .concat(Array.isArray(y_rec) ? y_rec : [])
        .filter(Number.isFinite);
    let ylo = yAll.length ? Math.min(...yAll) : 0;
    let yhi = yAll.length ? Math.max(...yAll) : 1;
    const ypad = (yhi - ylo) * 0.05 || 1;
    ylo -= ypad;
    yhi += ypad;

    // Pull σ bounds from run params (first finite encountered for this run)
    const sigmaMax = truthSorted.map(r => Number(r["param.sigma_max"] ?? r.sigma_max)).find(Number.isFinite);
    const sigmaMin = truthSorted.map(r => Number(r["param.sigma_min"] ?? r.sigma_min)).find(Number.isFinite);

    const traces = [];

    const guideStyle = "rgb(140,140,140)"; // subtle

    // 1) Hover-only traces (fully transparent line) so you still get a tooltip
    if (Number.isFinite(sigmaMax)) {
        traces.push({
            name: "σ_max",
            x: [sigmaMax, sigmaMax],
            y: [ylo, yhi],
            mode: "lines",
            line: {width: 6, color: "rgba(0,0,0,0)"}, // invisible but easy to hover
            hovertemplate: "σ_max = %{x}<extra></extra>",
            showlegend: false,
            meta: {role: "guide-hover"}
        });
    }
    if (Number.isFinite(sigmaMin)) {
        traces.push({
            name: "σ_min",
            x: [sigmaMin, sigmaMin],
            y: [ylo, yhi],
            mode: "lines",
            line: {width: 6, color: "rgba(0,0,0,0)"},
            hovertemplate: "σ_min = %{x}<extra></extra>",
            showlegend: false,
            meta: {role: "guide-hover"}
        });
    }

    // Truth legend suffix
    let sigmaPrimeRawTrue = truthSorted.map(r => Number(r["param.sigma_prime_p"])).find(Number.isFinite);
    const sigmaPrimeSuffixTrue = Number.isFinite(sigmaPrimeRawTrue) ? `, σ'=${sigmaPrimeRawTrue.toFixed(2)}` : "";
    const truthLegend = run_id === null ? `Truth (all runs)${sigmaPrimeSuffixTrue}` : `Truth (Run ${run_id}${sigmaPrimeSuffixTrue})`;

    traces.push({
        name: truthLegend,
        x: x_truth,
        y: y_truth,
        mode: "lines+markers",
        line: {width: 2, color: COLORS.real},
        marker: {size: 6, color: COLORS.real},
        hovertemplate: "σ₀=%{x}<br>ε=%{y}<extra></extra>",
        meta: {role: "truth"},
        legendgroup: "truth",
        legendgrouptitle: {text: "Truth"},
        visible: true
    });

    // Direct propagated legend suffix: first finite propagated σ'
    let seedSigmaProp = null;
    if (run_id !== null && PROP_MAP.has(Number(run_id))) {
        const propEntries = Array.from(PROP_MAP.get(Number(run_id)).values()).sort((a, b) => a.step_idx - b.step_idx);
        const firstFinite = propEntries.find(pr => pr.sigma_0 != null && Number.isFinite(pr.sigma_0));
        if (firstFinite) seedSigmaProp = Number(firstFinite.sigma_0);
    }
    const propSigmaSuffix = (seedSigmaProp !== null && Number.isFinite(seedSigmaProp)) ? `, σ'=${seedSigmaProp.toFixed(2)}` : "";

    if (run_id !== null && x_prop.length) {
        traces.push({
            name: `Propagated (direct${propSigmaSuffix})`,
            x: x_prop,
            y: y_prop,
            mode: "lines+markers",
            line: {width: 2, dash: "dash", color: COLORS.activePredProp},
            marker: {size: 6},
            hovertemplate: "σ₀(prop)=%{x}<br>ε(prop)=%{y}<extra></extra>",
            meta: {role: "prop"},
            legendgroup: "pred",
            legendgrouptitle: {text: "Predicted"},
            visible: "legendonly"
        });
    }

    // NEW: Recursive propagated legend suffix (first finite recursive σ')
    let seedSigmaRec = null;
    if (run_id !== null && REC_MAP.has(Number(run_id))) {
        const recEntries = Array.from(REC_MAP.get(Number(run_id)).values()).sort((a, b) => a.step_idx - b.step_idx);
        const firstFiniteRec = recEntries.find(rr => rr.sigma_0_rec != null && Number.isFinite(rr.sigma_0_rec));
        if (firstFiniteRec) seedSigmaRec = Number(firstFiniteRec.sigma_0_rec);
    }
    const recSigmaSuffix = (seedSigmaRec !== null && Number.isFinite(seedSigmaRec)) ? `, σ'=${seedSigmaRec.toFixed(2)}` : "";

    if (run_id !== null && x_rec.length) {
        traces.push({
            name: `Propagated (recursive${recSigmaSuffix})`,
            x: x_rec,
            y: y_rec,
            mode: "lines+markers",
            line: {width: 2, dash: "dashdot", color: COLORS.activePredPropRec},  // different style
            marker: {size: 6},
            hovertemplate: "σ₀(rec)=%{x}<br>ε(rec)=%{y}<extra></extra>",
            meta: {role: "prop_rec"},          // NEW role
            visible: true
        });
    }

    // Pred single-sample (legendonly)
    let hasPredX = x_pred.some(v => v != null);
    if (hasPredX) {
        const x_pred_clean = x_pred.map(v => (v == null ? NaN : v));
        traces.push({
            name: (run_id === null) ? "Pred single sample" : `Pred single sample (Run ${run_id})`,
            x: x_pred_clean,
            y: y_truth,
            mode: "markers",
            marker: {size: 6, color: COLORS.pred},
            hovertemplate: "σ₀(pred)=%{x}<br>ε=%{y}<extra></extra>",
            visible: "legendonly",
            meta: {role: "pred"},
        });
    }

    // Split overlay markers (honor user toggle)
    if (SHOW_SPLIT_MARKERS && SPLIT_FILTER !== "all") {
        const color = COLORS.split[SPLIT_FILTER] || "#000";
        const xs = [], ys = [];
        for (let i = 0; i < splits.length; i++) {
            if (splits[i] === SPLIT_FILTER && x_truth[i] != null && y_truth[i] != null) {
                xs.push(x_truth[i]);
                ys.push(y_truth[i]);
            }
        }
        if (xs.length) {
            traces.push({
                name: SPLIT_FILTER.toUpperCase(),
                x: xs, y: ys,
                mode: "markers",
                marker: {size: 10, color},
                hovertemplate: "σ₀=%{x}<br>ε=%{y}<extra></extra>",
                showlegend: false,
                meta: {role: "split"}
            });
        }
    }

    // Active point highlight
    if (hiStep != null) {
        const idxTruth = steps.findIndex(s => s === hiStep);
        if (idxTruth >= 0 && x_truth[idxTruth] != null && y_truth[idxTruth] != null) {
            traces.push({
                name: "Active (Truth)",
                x: [x_truth[idxTruth]], y: [y_truth[idxTruth]],
                mode: "markers",
                marker: {size: 12, color: COLORS.activeReal, line: {width: 2, color: "#fff"}},
                hoverinfo: "skip",
                showlegend: false
            });
        }
        if (run_id !== null && steps_prop.length) {
            const idxProp = steps_prop.findIndex(s => s === hiStep);
            if (idxProp >= 0 && x_prop[idxProp] != null && y_prop[idxProp] != null) {
                traces.push({
                    name: "Active (Prop)",
                    x: [x_prop[idxProp]], y: [y_prop[idxProp]],
                    mode: "markers",
                    marker: {size: 12, color: COLORS.activePred, line: {width: 2, color: "#fff"}},
                    hoverinfo: "skip",
                    showlegend: false
                });
            }
        }
    }

    // Legend moved to top-left, vertical (no overlap with axes labels)
    const layout = {
        margin: {l: 56, r: 16, t: 8, b: 48},
        xaxis: {title: "σ₀ [kPa]"},
        yaxis: {title: "ε_total [-]"},
        legend: {
            orientation: "v",
            x: 0.01, y: 0.99,
            xanchor: "left", yanchor: "top",
            bgcolor: "rgba(255,255,255,0.6)",
            groupclick: "toggleitem",
        },
        // Visual guide-lines that always span the plot vertically
        shapes: [
            ...(Number.isFinite(sigmaMax) ? [{
                type: "line",
                xref: "x", yref: "paper",
                x0: sigmaMax, x1: sigmaMax,
                y0: 0, y1: 1,
                line: {width: 1, dash: "dot", color: guideStyle}
            }] : []),
            ...(Number.isFinite(sigmaMin) ? [{
                type: "line",
                xref: "x", yref: "paper",
                x0: sigmaMin, x1: sigmaMin,
                y0: 0, y1: 1,
                line: {width: 1, dash: "dash", color: guideStyle}
            }] : []),
        ]
    };


    window.Plotly.react(div, traces, layout, {displayModeBar: true, responsive: true});
    renderSplitLegendRow();  // NEW: keep the row in sync with current SPLIT_FILTER

    updateTableViewFromPlot(div);
    renderCompareTable(TABLE_RUN_FILTER, PAIRS);

    if (!PLOT_EVENTS_WIRED) {
        div.on('plotly_legendclick', () => {
            setTimeout(() => {
                updateTableViewFromPlot(div);
                renderCompareTable(TABLE_RUN_FILTER, PAIRS);
            }, 0);
        });
        div.on('plotly_restyle', () => {
            updateTableViewFromPlot(div);
            renderCompareTable(TABLE_RUN_FILTER, PAIRS);
        });
        PLOT_EVENTS_WIRED = true;
    }
}


// ---------- Compare table (split-aware) ----------
function renderCompareTable(run_id, pairs) {
    const thead = $(SEL.cmpThead);
    const tbody = $(SEL.cmpTbody);
    if (!thead || !tbody) return;

    const rid = (TABLE_RUN_FILTER != null) ? Number(TABLE_RUN_FILTER)
        : (run_id != null ? Number(run_id)
            : (ACTIVE.run_id != null ? Number(ACTIVE.run_id) : null));

    // Dynamic columns based on visible traces
    const COLS = [
        {key: "run_id", label: "Run"},
        {key: "step_idx", label: "Step"},
        ...(TABLE_VIEW.showTruth ? [{key: "truth_sigma_0", label: "σ₀"}] : []),
        ...(TABLE_VIEW.showPred ? [{key: "pred_sigma_0", label: "σ₀ (pred)"}] : []),
        ...(TABLE_VIEW.showProp ? [{key: "prop_sigma_0", label: "σ₀ (prop)"}] : []),
        ...(TABLE_VIEW.showPropRec ? [{key: "rec_sigma_0", label: "σ₀ (rec)"}] : []),

        ...((TABLE_VIEW.showPred && TABLE_VIEW.showTruth) ? [{key: "d_sigma_pred", label: "σ₀ APE [%] (pred−truth)"}] : []),
        ...((TABLE_VIEW.showProp && TABLE_VIEW.showTruth) ? [{key: "d_sigma_prop", label: "σ₀ APE [%] (prop−truth)"}] : []),
        ...((TABLE_VIEW.showPropRec && TABLE_VIEW.showTruth) ? [{key: "d_sigma_rec", label: "σ₀ APE [%] (rec−truth)"}] : []),

        ...(TABLE_VIEW.showTruth ? [{key: "truth_e_s", label: "E_s"}] : []),
        ...(TABLE_VIEW.showPred ? [{key: "pred_e_s", label: "E_s (pred)"}] : []),
        ...(TABLE_VIEW.showProp ? [{key: "prop_e_s", label: "E_s (prop)"}] : []),
        ...(TABLE_VIEW.showPropRec ? [{key: "rec_e_s", label: "E_s (rec)"}] : []),

        ...((TABLE_VIEW.showPred && TABLE_VIEW.showTruth) ? [{key: "d_es_pred", label: "E_s APE [%] (pred−truth)"}] : []),
        ...((TABLE_VIEW.showProp && TABLE_VIEW.showTruth) ? [{key: "d_es_prop", label: "E_s APE [%] (prop−truth)"}] : []),
        ...((TABLE_VIEW.showPropRec && TABLE_VIEW.showTruth) ? [{key: "d_es_rec", label: "E_s APE [%] (rec−truth)"}] : []),
    ];

    // THEAD
    {
        const tr = document.createElement("tr");
        COLS.forEach(c => {
            const th = document.createElement("th");
            th.textContent = c.label;
            tr.appendChild(th);
        });
        thead.replaceChildren(tr);
    }

    if (!Number.isFinite(rid)) {
        tbody.replaceChildren();
        return;
    }

    const truthRows = ROWS
        .filter(r => Number(r.run_id) === rid)
        .slice()
        .sort((a, b) => (safeStepIdx(a) ?? 0) - (safeStepIdx(b) ?? 0));

    const truthByStep = new Map();
    for (const r of truthRows) {
        const s = safeStepIdx(r);
        if (s != null) truthByStep.set(s, r);
    }

    const propByStep = PROP_MAP.get(rid) || new Map();
    const recByStep = REC_MAP.get(rid) || new Map();   // NEW

    const steps = Array.from(new Set([...truthByStep.keys(), ...propByStep.keys(), ...recByStep.keys()])).sort(byNum);

    const frag = document.createDocumentFragment();

    for (const step of steps) {
        const t = truthByStep.get(step) || null;
        const p = propByStep.get(step) || null;
        const r = recByStep.get(step) || null;

        const truth_sigma_0 = (t && Number.isFinite(Number(t.sigma_0))) ? Number(t.sigma_0) : null;
        const pred_sigma_0 = (t && Number.isFinite(Number(t.pred_sigma_0))) ? Number(t.pred_sigma_0) : null;

        const prop_sigma_0 = (p && p.sigma_0 != null && Number.isFinite(Number(p.sigma_0))) ? Number(p.sigma_0) : null;
        const rec_sigma_0 = (r && r.sigma_0_rec != null && Number.isFinite(Number(r.sigma_0_rec))) ? Number(r.sigma_0_rec) : null;

        const truth_e_s = (t && Number.isFinite(Number(t.e_s))) ? Number(t.e_s) : null;
        const pred_e_s = (t && Number.isFinite(Number(t.pred_e_s))) ? Number(t.pred_e_s) : null;
        const prop_e_s = (p && p.e_s != null && Number.isFinite(Number(p.e_s))) ? Number(p.e_s) : null;
        const rec_e_s = (r && r.e_s_rec != null && Number.isFinite(Number(r.e_s_rec))) ? Number(r.e_s_rec) : null;

        const row = {
            run_id: rid,
            step_idx: step,

            truth_sigma_0,
            pred_sigma_0,
            prop_sigma_0,
            rec_sigma_0,

            d_sigma_pred: (Number.isFinite(pred_sigma_0) && Number.isFinite(truth_sigma_0)) ? (Math.abs(pred_sigma_0 - truth_sigma_0) / Math.abs(truth_sigma_0) * 100) : null,
            d_sigma_prop: (prop_sigma_0 != null && Number.isFinite(truth_sigma_0)) ? (Math.abs(prop_sigma_0 - truth_sigma_0)/ Math.abs(truth_sigma_0) * 100) : null,
            d_sigma_rec: (rec_sigma_0 != null && Number.isFinite(truth_sigma_0)) ? (Math.abs(rec_sigma_0 - truth_sigma_0)/ Math.abs(truth_sigma_0) * 100) : null,

            truth_e_s,
            pred_e_s,
            prop_e_s,
            rec_e_s,

            d_es_pred: (Number.isFinite(pred_e_s) && Number.isFinite(truth_e_s)) ? (Math.abs(pred_e_s - truth_e_s)/ Math.abs(truth_e_s) * 100) : null,
            d_es_prop: (prop_e_s != null && Number.isFinite(truth_e_s)) ? (Math.abs(prop_e_s - truth_e_s)/ Math.abs(truth_e_s) * 100) : null,
            d_es_rec: (rec_e_s != null && Number.isFinite(truth_e_s)) ? (Math.abs(rec_e_s - truth_e_s)/ Math.abs(truth_e_s) * 100) : null,
        };

        const tr = document.createElement("tr");
        tr.dataset.run_id = String(rid);
        tr.dataset.step_idx = String(step);

        const inSplit = (SPLIT_FILTER !== "all" && t?.split === SPLIT_FILTER);
        const isActive = (ACTIVE.run_id === rid && Number(ACTIVE.step_idx) === Number(step));
        if (isActive) tr.classList.add("active");
        if (inSplit) tr.classList.add("in-split");

        COLS.forEach(col => {
            const td = document.createElement("td");
            const v = row[col.key];
            td.textContent = fmt(v);
            if (v == null) td.classList.add("muted");
            tr.appendChild(td);
        });

        frag.appendChild(tr);
    }

    tbody.replaceChildren(frag);
}

// ---------- CSV Export for compare table ----------
function exportCompareCSV() {
    const table = $("#tbl-compare");
    if (!table) return;

    const headCells = Array.from(table.querySelectorAll("thead th")).map(th => th.textContent);
    const rows = Array.from(table.querySelectorAll("tbody tr")).map(tr => Array.from(tr.children).map(td => td.textContent));

    const csv = [headCells, ...rows].map(row => row.map(v => /[",;\n]/.test(v) ? `"${String(v).replace(/"/g, '""')}"` : String(v)).join(",")).join("\n");

    const blob = new Blob([csv], {type: "text/csv;charset=utf-8"});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `compare_${ACTIVE.run_id === null ? "all" : `run${ACTIVE.run_id}`}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
}

// ---------- Selection & Wiring ----------
function selectSample(run_id, step_idx) {
    ACTIVE.run_id = run_id;
    ACTIVE.step_idx = step_idx;

    // Update list highlight
    $$(SEL.samplesList + " .list-item").forEach(el => {
        const rid = Number(el.dataset.run_id);
        const sid = Number(el.dataset.step_idx);
        el.classList.toggle("active", rid === run_id && sid === step_idx);
    });

    // Current row (from all rows, not split-filtered)
    const row = ROWS.find(r => Number(r.run_id) === Number(run_id) && safeStepIdx(r) === step_idx);
    if (!row) return;

    renderCurrentSample(row);
    renderParams(row);
    renderPreds(row);
    renderPlot(run_id, {step_idx});

    // Keep compare table highlight in sync
    renderCompareTable(ACTIVE.run_id, PAIRS);

    // Optional: scroll active table row into view
    //const activeRow = $("#tbl-compare tbody tr.active");
    //if (activeRow) activeRow.scrollIntoView({block: "nearest"});
    renderCompareTable(TABLE_RUN_FILTER, PAIRS);
}

// ---------- CSV Export for compare table ----------
function exportCompareCSV() {
    const table = $("#tbl-compare");
    if (!table) return;

    const headCells = Array.from(table.querySelectorAll("thead th")).map(th => th.textContent);
    const rows = Array.from(table.querySelectorAll("tbody tr")).map(tr => Array.from(tr.children).map(td => td.textContent));

    const csv = [headCells, ...rows].map(row => row.map(v => /[",;\n]/.test(v) ? `"${String(v).replace(/"/g, '""')}"` : String(v)).join(",")).join("\n");

    const blob = new Blob([csv], {type: "text/csv;charset=utf-8"});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `compare_${ACTIVE.run_id === null ? "all" : `run${ACTIVE.run_id}`}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
}

// ---------- Selection & Wiring ----------
function selectSample(run_id, step_idx) {
    ACTIVE.run_id = run_id;
    ACTIVE.step_idx = step_idx;

    // Update list highlight
    $$(SEL.samplesList + " .list-item").forEach(el => {
        const rid = Number(el.dataset.run_id);
        const sid = Number(el.dataset.step_idx);
        el.classList.toggle("active", rid === run_id && sid === step_idx);
    });

    // Current row (from all rows, not split-filtered)
    const row = ROWS.find(r => Number(r.run_id) === Number(run_id) && safeStepIdx(r) === step_idx);
    if (!row) return;

    renderCurrentSample(row);
    renderParams(row);
    renderPreds(row);
    renderPlot(run_id, {step_idx});

    // Keep compare table highlight in sync
    renderCompareTable(ACTIVE.run_id, PAIRS);

    // Optional: scroll active table row into view
    //const activeRow = $("#tbl-compare tbody tr.active");
    //if (activeRow) activeRow.scrollIntoView({block: "nearest"});
    renderCompareTable(TABLE_RUN_FILTER, PAIRS);
}

function installEvents() {
    // Split selector changes the view universe
    $(SEL.splitFilter)?.addEventListener("change", (ev) => {
        const val = (ev.target.value || "all").toLowerCase();
        SPLIT_FILTER = (val === "train" || val === "val" || val === "test") ? val : "all";
        TABLE_RUN_FILTER = null;  // show all runs in this split by default

        // Reset run selection to first available run in this split (or all)
        const splitRows = rowsForCurrentSplit();
        const runIds = Array.from(new Set(splitRows.map(r => Number(r.run_id)))).sort(byNum);
        ACTIVE.run_id = runIds.length ? runIds[0] : null;
        ACTIVE.step_idx = null; // will get set below from first list item
        TABLE_RUN_FILTER = runIds.length ? runIds[0] : null;

        renderRunChips();
        renderSamplesList();

        const first = $(SEL.samplesList + " .list-item");
        if (first) {
            selectSample(Number(first.dataset.run_id), Number(first.dataset.step_idx));
        } else {
            // no items in this split: still update table/plot for null selection
            renderPlot(ACTIVE.run_id, {step_idx: null});
            renderCompareTable(ACTIVE.run_id, PAIRS);
        }
    });

    // Chips: filter by run within current split and refresh everything
    $(SEL.runChips)?.addEventListener("click", (ev) => {
        const chip = ev.target.closest(".chip");
        if (!chip) return;
        const val = chip.dataset.run;
        if (val === "*") {
            // Show ALL runs in the compare table for the current split
            TABLE_RUN_FILTER = null;
            ACTIVE.run_id = null;           // selection will be set from first list item below
        } else {
            const rid = Number(val);
            TABLE_RUN_FILTER = rid;         // table shows only this run
            ACTIVE.run_id = rid;            // selection starts in this run
        }

        renderRunChips();
        renderSamplesList();

        // Select first sample in this (split-filtered) list
        const first = $(SEL.samplesList + " .list-item");
        if (first) {
            selectSample(Number(first.dataset.run_id), Number(first.dataset.step_idx));
        } else {
            renderPlot(ACTIVE.run_id, {step_idx: null});
        }
        renderCompareTable(TABLE_RUN_FILTER, PAIRS);
    });

    // Clicking a sample in the list selects it
    $(SEL.samplesList)?.addEventListener("click", (ev) => {
        const li = ev.target.closest(".list-item");
        if (!li) return;
        selectSample(Number(li.dataset.run_id), Number(li.dataset.step_idx));
    });

    // Clicking a row in the compare table selects/plots that point
    $("#tbl-compare tbody")?.addEventListener("click", (ev) => {
        const tr = ev.target.closest("tr[data-run_id][data-step_idx]");
        if (!tr) return;
        const run_id = Number(tr.dataset.run_id);
        const step_idx = Number(tr.dataset.step_idx);
        selectSample(run_id, step_idx);
    });
}

// ---------- Initial load ----------
async function init() {
    // Load CSVs
    const [rfRes, propRes, recRes, spRes] = await Promise.all([
        fetch(PATH_RUNS_FINAL, {cache: "no-store"}).then(r => r.text()),
        fetch(PATH_RUNS_FINAL_PRED_PROP, {cache: "no-store"}).then(r => r.text()),
        fetch(PATH_RUNS_FINAL_PRED_PROP_REC, {cache: "no-store"}).then(r => r.text()).catch(() => ""), // NEW
        fetch(PATH_SPLITS, {cache: "no-store"}).then(r => r.text()).catch(() => null),
    ]);

    const {header, rows} = parseCSV(rfRes);

    // Parse direct propagated
    const {rows: propRowsRaw} = parseCSV(propRes);
    PROP_MAP.clear();
    for (const pr of propRowsRaw) {
        const run_id = Number(pr.run_id);
        const step_idx = Number(pr.step_idx);
        if (!Number.isFinite(run_id) || !Number.isFinite(step_idx)) continue;
        const rec = {
            run_id,
            step_idx,
            sigma_0: pr.sigma_0 != null ? Number(pr.sigma_0) : null,
            e_s: pr.e_s != null ? Number(pr.e_s) : null,
            eps_total: pr.eps_total != null ? Number(pr.eps_total) : null,
        };
        if (!PROP_MAP.has(run_id)) PROP_MAP.set(run_id, new Map());
        PROP_MAP.get(run_id).set(step_idx, rec);
    }

    // NEW: Parse recursive propagated
    REC_MAP.clear();
    if (recRes && recRes.trim().length) {
        const {rows: recRowsRaw} = parseCSV(recRes);
        for (const rr of recRowsRaw) {
            const run_id = Number(rr.run_id);
            const step_idx = Number(rr.step_idx);
            if (!Number.isFinite(run_id) || !Number.isFinite(step_idx)) continue;
            const rec = {
                run_id,
                step_idx,
                sigma_0_rec: rr.sigma_0_rec != null ? Number(rr.sigma_0_rec) : null,
                e_s_rec: rr.e_s_rec != null ? Number(rr.e_s_rec) : null,
                eps_total_rec: rr.eps_total_rec != null ? Number(rr.eps_total_rec) : null,
            };
            if (!REC_MAP.has(run_id)) REC_MAP.set(run_id, new Map());
            REC_MAP.get(run_id).set(step_idx, rec);
        }
    }
    let splitMap = new Map();
    if (spRes) {
        const {rows: splitRows} = parseCSV(spRes);
        // Build global_idx -> split map
        for (const r of splitRows) {
            const gi = Number(r.global_idx);
            const s = normalizeSplit(r.split);
            if (Number.isFinite(gi) && s) splitMap.set(gi, s);
        }
    }

    // Normalize numerics, add .split via global_idx join
    ROWS = rows.map(r => {
        const o = {...r};
        o.run_id = Number(o.run_id);
        if (o.step_idx == null && o.rank != null) o.step_idx = Number(o.rank); else o.step_idx = Number(o.step_idx);
        ["sigma_0", "eps_total", "pred_sigma_0", "eps_delta", "e_s", "sigma_delta", "global_idx"]
            .forEach(k => {
                if (o[k] != null) o[k] = Number(o[k]);
            });
        const gi = Number(o.global_idx);
        o.split = splitMap.size && Number.isFinite(gi) ? (splitMap.get(gi) || null) : null;
        return o;
    });

    // Derive compare pairs once
    PAIRS = getComparePairs(header);

    // Initial split = 'all'. Choose first run that exists in the current split universe
    const runIds = Array.from(new Set(rowsForCurrentSplit().map(r => Number(r.run_id)))).filter(Number.isFinite).sort(byNum);
    ACTIVE.run_id = runIds.length ? runIds[0] : null;

    if (ACTIVE.run_id != null) {
        const inRun = rowsForCurrentSplit().filter(r => Number(r.run_id) === Number(ACTIVE.run_id))
            .sort((a, b) => (a.step_idx ?? a.rank ?? 0) - (b.step_idx ?? b.rank ?? 0));
        ACTIVE.step_idx = inRun.length ? (inRun[0].step_idx ?? inRun[0].rank ?? 0) : null;
    }
    // First render + events
    renderRunChips();
    renderSamplesList();
    installEvents();

    // Force table to show the first run’s data by default
    TABLE_RUN_FILTER = ACTIVE.run_id;

    // Select first item if exists, else still draw plot/table
    const first = $(SEL.samplesList + " .list-item");
    if (first) {
        selectSample(Number(first.dataset.run_id), Number(first.dataset.step_idx));
    } else {
        renderPlot(ACTIVE.run_id, {step_idx: null});
        renderCompareTable(TABLE_RUN_FILTER, PAIRS);
    }
}

window.addEventListener("DOMContentLoaded", init);

window.addEventListener("DOMContentLoaded", () => {
    // compute header height and store in CSS var
    function setHeaderH() {
        const h = Math.max(
            0,
            (document.querySelector('#app-header')?.offsetHeight) || 0
        );
        document.documentElement.style.setProperty('--header-h', `${h}px`);
    }

    setHeaderH();
    // navbar may render async; re-measure shortly and on resize
    setTimeout(setHeaderH, 100);
    setTimeout(setHeaderH, 400);
    window.addEventListener('resize', setHeaderH);

    // then your normal init
    init();
});

