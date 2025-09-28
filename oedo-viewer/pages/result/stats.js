// File: pages/result/stats.js
// Purpose: Load training/validation/test statistics CSVs and render plots + table on the stats cards.

(function () {
  // --- Config: paths & element IDs -----------------------------------------
  const PATHS = {
    history: "../../viewer_data/training_history.csv",
    splits:  "../../viewer_data/split_assignments.csv",
    test:    "../../viewer_data/test_metrics.csv",
  };

  const EL = {
    plotMSE:   document.getElementById("plot-train-mse"),
    plotR2:    document.getElementById("plot-train-r2"),
    plotSplit: document.getElementById("plot-splits"),
    tblStats:  document.getElementById("tbl-stats"),
  };

  // --- Utilities ------------------------------------------------------------
  function csvFetch(path) {
    return fetch(path, { cache: "no-store" }).then(async (res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status} for ${path}`);
      const text = await res.text();
      return new Promise((resolve, reject) => {
        // Papa is global (loaded via JSDelivr in HTML)
        Papa.parse(text, {
          header: true,
          dynamicTyping: false, // we’ll coerce manually
          skipEmptyLines: true,
          complete: (res) => resolve(res.data || []),
          error: (err) => reject(err),
        });
      });
    });
  }

  function toNum(x) {
    if (x === null || x === undefined) return NaN;
    if (typeof x === "number") return x;
    const s = String(x).trim().replace(",", "."); // tolerate comma decimals
    const v = parseFloat(s);
    return Number.isFinite(v) ? v : NaN;
  }

  function hasAnyFinite(arr) {
    return arr.some((v) => Number.isFinite(v));
  }

  function fmtVal(x) {
    if (!Number.isFinite(x)) return "—";
    const ax = Math.abs(x);
    if (ax > 0 && (ax < 1e-3 || ax >= 1e4)) return x.toExponential(3);
    return x.toFixed(4);
  }

  function byEpochAsc(a, b) {
    const ea = toNum(a.epoch), eb = toNum(b.epoch);
    if (Number.isFinite(ea) && Number.isFinite(eb)) return ea - eb;
    return 0;
  }

  function summarizeSplits(rows) {
    const counts = { train: 0, val: 0, test: 0, other: 0 };
    for (const r of rows) {
      const s = (r.split || "").toString().trim().toLowerCase();
      if (s === "train" || s === "training") counts.train++;
      else if (s === "val" || s === "valid" || s === "validation") counts.val++;
      else if (s === "test" || s === "testing") counts.test++;
      else counts.other++;
    }
    return counts;
  }

  function ensureTBody(tbl) {
    let tb = tbl.querySelector("tbody");
    if (!tb) {
      tb = document.createElement("tbody");
      tbl.appendChild(tb);
    }
    tb.innerHTML = "";
    return tb;
  }

  // --- Rendering: Plots -----------------------------------------------------
  async function renderTrainingPlots(historyRows) {
    if (!EL.plotMSE || !EL.plotR2) return;

    // Sort by epoch
    historyRows.sort(byEpochAsc);

    const epoch = historyRows.map((r) => toNum(r.epoch));
    const tr_mse = historyRows.map((r) => toNum(r.train_mse));
    const va_mse = historyRows.map((r) => toNum(r.val_mse));
    const tr_r2  = historyRows.map((r) => toNum(r.train_r2));
    const va_r2  = historyRows.map((r) => toNum(r.val_r2));

    // MSE plot
    const tracesMSE = [];
    if (hasAnyFinite(tr_mse)) {
      tracesMSE.push({
        x: epoch, y: tr_mse, name: "Train MSE", mode: "lines", type: "scatter",
      });
    }
    if (hasAnyFinite(va_mse)) {
      tracesMSE.push({
        x: epoch, y: va_mse, name: "Val MSE", mode: "lines", type: "scatter", marker: {color: "#ffffff"},
      });
    }

    const layoutMSE = {
      margin: { l: 50, r: 10, t: 20, b: 40 },
      xaxis: { title: "Epoch" },
      yaxis: { title: "MSE" },
      showlegend: true,
    };

    await Plotly.react(EL.plotMSE, tracesMSE, layoutMSE, { responsive: true });

    // R² plot
    const tracesR2 = [];
    if (hasAnyFinite(tr_r2)) {
      tracesR2.push({
        x: epoch, y: tr_r2, name: "Train R²", mode: "lines", type: "scatter",
      });
    }
    if (hasAnyFinite(va_r2)) {
      tracesR2.push({
        x: epoch, y: va_r2, name: "Val R²", mode: "lines", type: "scatter", marker: {color: "#ffffff"},
      });
    }

    const layoutR2 = {
      margin: { l: 50, r: 10, t: 20, b: 40 },
      xaxis: { title: "Epoch" },
      yaxis: { title: "R²", range: [-0.05, 1.02] },
      showlegend: true,
    };

    await Plotly.react(EL.plotR2, tracesR2, layoutR2, { responsive: true });
  }

  async function renderSplitPlot(splitRows) {
    if (!EL.plotSplit) return;
    const c = summarizeSplits(splitRows);
    const labels = ["train", "val", "test"];
    const values = [c.train, c.val, c.test];

    const traces = [{
      x: labels, y: values, type: "bar", text: values.map(String), textposition: "auto",
      hovertemplate: "%{x}: %{y}<extra></extra>",
    }];

    const layout = {
      margin: { l: 50, r: 10, t: 20, b: 40 },
      xaxis: { title: "Split" },
      yaxis: { title: "Anzahl" },
      showlegend: false,
    };

    await Plotly.react(EL.plotSplit, traces, layout, { responsive: true });
  }

  // --- Rendering: Table ------------------------------------------------------
  function renderStatsTable(historyRows, testRows) {
    if (!EL.tblStats) return;
    const tb = ensureTBody(EL.tblStats);

    if (!historyRows.length && !testRows.length) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 4;
      td.textContent = "Keine Statistikdaten gefunden.";
      tr.appendChild(td);
      tb.appendChild(tr);
      return;
    }

    // last epoch row
    historyRows.sort(byEpochAsc);
    const last = historyRows.length ? historyRows[historyRows.length - 1] : {};

    const train = {
      mse: toNum(last.train_mse), rmse: toNum(last.train_rmse),
      mae: toNum(last.train_mae), r2: toNum(last.train_r2),
    };
    const val = {
      mse: toNum(last.val_mse), rmse: toNum(last.val_rmse),
      mae: toNum(last.val_mae), r2: toNum(last.val_r2),
    };

    // test (assume single-row CSV with columns mse, rmse, mae, r2)
    const t = (testRows && testRows[0]) || {};
    const test = {
      mse: toNum(t.mse), rmse: toNum(t.rmse), mae: toNum(t.mae), r2: toNum(t.r2),
    };

    const metrics = [
      ["MSE",  train.mse,  val.mse,  test.mse],
      ["RMSE", train.rmse, val.rmse, test.rmse],
      ["MAE",  train.mae,  val.mae,  test.mae],
      ["R²",   train.r2,   val.r2,   test.r2],
    ];

    for (const [name, a, b, c] of metrics) {
      const tr = document.createElement("tr");
      const td0 = document.createElement("td"); td0.textContent = name; tr.appendChild(td0);
      const td1 = document.createElement("td"); td1.textContent = fmtVal(a); tr.appendChild(td1);
      const td2 = document.createElement("td"); td2.textContent = fmtVal(b); tr.appendChild(td2);
      const td3 = document.createElement("td"); td3.textContent = fmtVal(c); tr.appendChild(td3);
      tb.appendChild(tr);
    }
  }

  // --- Init ------------------------------------------------------------------
  async function init() {
    try {
      const [historyRows, splitRows, testRows] = await Promise.all([
        csvFetch(PATHS.history).catch((e) => { console.warn("history CSV:", e); return []; }),
        csvFetch(PATHS.splits).catch((e) => { console.warn("splits CSV:", e); return []; }),
        csvFetch(PATHS.test).catch((e) => { console.warn("test CSV:", e); return []; }),
      ]);

      await renderTrainingPlots(historyRows);
      await renderSplitPlot(splitRows);
      renderStatsTable(historyRows, testRows);
    } catch (err) {
      console.error("[stats.js] init error:", err);
      // Optional: user-facing fallback messages
      if (EL.plotMSE)   EL.plotMSE.innerHTML = "<div class='muted'>Fehler beim Laden der Trainingsdaten.</div>";
      if (EL.plotR2)    EL.plotR2.innerHTML = "<div class='muted'>Fehler beim Laden der Trainingsdaten.</div>";
      if (EL.plotSplit) EL.plotSplit.innerHTML = "<div class='muted'>Fehler beim Laden der Split-Daten.</div>";
      if (EL.tblStats) {
        const tb = ensureTBody(EL.tblStats);
        const tr = document.createElement("tr");
        const td = document.createElement("td");
        td.colSpan = 4;
        td.textContent = "Fehler beim Laden der Statistik.";
        tr.appendChild(td);
        tb.appendChild(tr);
      }
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
