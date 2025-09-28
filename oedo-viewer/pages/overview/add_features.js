/* pages/overview/add-features.js
 * Add-Features-Modal für runs.csv (Download-only)
 * - nutzt viewer_data/runs.csv
 * - liest optional viewer_data/additional_runs.csv (falls manuell vorhanden)
 * - "Bestätigen" => erzeugt Download "additional_runs.csv" (Originaldateien bleiben unberührt)
 */

(function AF () {
  // ----------------------------
  // Pfade
  // ----------------------------
  const RUNS_CSV_PATH = window.RUNS_CSV ?? "viewer_data/runs.csv";
  const ADD_CSV_PATH  = "viewer_data/additional_runs.csv"; // nur zum Einlesen (falls vorhanden)

  // ----------------------------
  // Whitelist-Funktionen (skalare Varianten)
  // ----------------------------
  const FN = {
    abs: Math.abs,
    sign: Math.sign,
    sqrt: Math.sqrt,
    log: Math.log,
    exp: Math.exp,
    clip: (x, lo, hi) => Math.min(Math.max(x, lo), hi),
    where: (cond, a, b) => (cond ? a : b),
    maximum: (a, b) => (a > b ? a : b),
    minimum: (a, b) => (a < b ? a : b),
    pow: Math.pow,
  };
  const ALLOWED_FUNCS = Object.keys(FN);

  // Sicherheits-Regex gegen gefährliche Tokens
  const ILLEGAL = /(^|[^A-Za-z0-9_])(window|document|globalThis|Function|=>|while|for|class|constructor|import|export|new\s+Function)/;

  // ----------------------------
  // UI-Refs
  // ----------------------------
  const $modal       = document.getElementById("featureModal");
  const $open        = document.getElementById("addFeatureBtn");
  const $close       = document.getElementById("featClose");
  const $varBadges   = document.getElementById("varBadges");
  const $exprInput   = document.getElementById("exprInput");
  const $aliasInput  = document.getElementById("aliasInput");
  const $funcSel     = document.getElementById("funcSel");
  const $exprList    = document.getElementById("exprList");
  const $targetInput = document.getElementById("targetInput");
  const $preview     = document.getElementById("previewBox");

  const $addExpr      = document.getElementById("addExpr");
  const $clearExprs   = document.getElementById("clearExprs");
  const $previewExprs = document.getElementById("previewExprs");
  const $applyExprs   = document.getElementById("applyExprs"); // Button-Label "Bestätigen"

  if (!$modal || !$open) return;

  ensureHiddenCSS();

  // ----------------------------
  // State
  // ----------------------------
  let runsData = null;     // { header, rows, columns }
  let exprBag = [];        // [{expr, alias}]
  let targetExpr = "";     // optionaler Target-Ausdruck
  let initialized = false;

  // ----------------------------
  // Modal öffnen/schließen
  // ----------------------------
  function openModal() {
    $modal.hidden = false;
    initIfNeeded();
  }
  function closeModal(){ $modal.hidden = true; }

  $open.addEventListener("click", openModal);
  if ($close) $close.addEventListener("click", closeModal);
  $modal.addEventListener("click", (e) => { if (e.target === $modal) closeModal(); });

  // ----------------------------
  // Init: runs.csv laden, Variablen, vorhandenes additional_runs.csv einlesen
  // ----------------------------
  async function initIfNeeded() {
    if (initialized) return;
    try {
      const txt = await fetchText(RUNS_CSV_PATH);
      runsData = parseCSV(txt);
      renderVarBadges(runsData.header);

      // vorhandenes additional_runs.csv (nur anzeigen, wenn vorhanden)
      try {
        const addTxt = await fetchText(ADD_CSV_PATH);
        const parsed = parseCSV(addTxt);
        const feats = [];
        let targ = "";
        for (const col of parsed.header) {
          if (col.startsWith("F:")) feats.push({ expr: col.slice(2), alias: "" });
          else if (col.startsWith("T:")) targ = col.slice(2);
        }
        if (feats.length) { exprBag = feats; renderExprList(); }
        if (targ) { targetExpr = targ; if ($targetInput) $targetInput.value = targ; }
        showInfo(`Geladen: ${ADD_CSV_PATH} (${parsed.rows.length} Zeilen).`);
      } catch {
        // nicht vorhanden → ok
        showInfo("Kein additional_runs.csv gefunden (noch nicht erzeugt).");
      }
    } catch (e) {
      showError("Initialisierung: " + e.message);
    }
    attachOperatorsUI();
    initialized = true;
  }

  // ----------------------------
  // Variablen-Badges
  // ----------------------------
  function renderVarBadges(header) {
    $varBadges.innerHTML = "";
    header.forEach(h => {
      const b = document.createElement("span");
      b.className = "badge";
      b.textContent = h;
      b.title = "In Ausdruck einfügen";
      b.addEventListener("click", () => insertAtCursor($exprInput, h));
      $varBadges.appendChild(b);
    });
  }

  // ----------------------------
  // Ausdrucksliste
  // ----------------------------
  function renderExprList() {
    $exprList.innerHTML = "";
    exprBag.forEach((x, i) => {
      const chip = document.createElement("span");
      chip.className = "badge";
      chip.textContent = (x.alias ? `${x.alias}=` : "") + x.expr;
      chip.title = "Entfernen";
      chip.addEventListener("click", () => { exprBag.splice(i, 1); renderExprList(); });
      $exprList.appendChild(chip);
    });
  }

  $addExpr?.addEventListener("click", () => {
    const expr = ($exprInput?.value ?? "").trim();
    if (!expr) return;
    const alias = ($aliasInput?.value ?? "").trim();
    exprBag.push({ expr, alias });
    if ($exprInput)  $exprInput.value  = "";
    if ($aliasInput) $aliasInput.value = "";
    renderExprList();
  });

  $clearExprs?.addEventListener("click", () => { exprBag = []; renderExprList(); });

  // Operator/Funktions-Buttons
  function attachOperatorsUI() {
    document.querySelectorAll("#featureModal .ops button[data-op]").forEach(btn => {
      btn.addEventListener("click", () => insertAtCursor($exprInput, ` ${btn.dataset.op} `));
    });
    if ($funcSel) {
      $funcSel.addEventListener("change", () => {
        const v = $funcSel.value;
        if (!v) return;
        insertAtCursor($exprInput, v.replace("( )", "(") + ")");
        if ($exprInput && $exprInput.selectionStart != null) {
          $exprInput.selectionStart = $exprInput.selectionEnd = $exprInput.selectionStart - 1;
        }
        $funcSel.value = "";
      });
    }
  }

  // ----------------------------
  // Preview
  // ----------------------------
  $previewExprs?.addEventListener("click", () => {
    if (!runsData) { showInfo("Keine Daten geladen."); return; }
    targetExpr = ($targetInput?.value ?? "").trim();

    try {
      const env = runsData.columns;
      let out = "";
      exprBag.forEach(({ expr, alias }) => {
        const col = evaluateExpressionVector(expr, env);
        out += formatPreview(alias ? `${alias}=${expr}` : expr, col);
      });
      if (targetExpr) {
        const tcol = evaluateExpressionVector(targetExpr, env);
        out += formatPreview(`Target: ${targetExpr}`, tcol);
      }
      $preview.textContent = out || "—";
    } catch (e) {
      showError(e.message);
    }
  });

  // ----------------------------
  // Bestätigen = Download additional_runs.csv erzeugen
  // ----------------------------
  $applyExprs?.addEventListener("click", () => {
    if (!runsData) { showInfo("Keine Daten geladen."); return; }
    targetExpr = ($targetInput?.value ?? "").trim();

    try {
      const env = runsData.columns;
      const N = runsData.rows.length;

      // Start: Kopie von runs.csv, vorherige F:/T:-Spalten NICHT übernehmen
      const rows = runsData.rows.map(r => {
        const o = {};
        for (const k of runsData.header) if (!(k.startsWith("F:") || k.startsWith("T:"))) o[k] = r[k];
        return o;
      });
      const headerBase = runsData.header.filter(h => !(h.startsWith("F:") || h.startsWith("T:")));

      // Features berechnen
      const headerOrder = [...headerBase];
      for (const { expr, alias } of exprBag) {
        const col = evaluateExpressionVector(expr, env);
        const name = (alias || `F:${expr}`).replace(/\s+/g, "");
        if (!headerOrder.includes(name)) headerOrder.push(name);
        for (let i=0;i<N;i++) rows[i][name] = col[i];
      }

      // Optional Target
      if (targetExpr) {
        const tname = `T:${targetExpr}`.replace(/\s+/g, "");
        const tcol = evaluateExpressionVector(targetExpr, env);
        if (!headerOrder.includes(tname)) headerOrder.push(tname);
        for (let i=0;i<N;i++) rows[i][tname] = tcol[i];
      }

      const csv = toCSV(rows, headerOrder);
      downloadCSV("additional_runs.csv", csv);
      showSuccess(`Download erstellt: additional_runs.csv (${N} Zeilen).`);
    } catch (e) {
      showError(e.message);
    }
  });

  // ----------------------------
  // Evaluator: sichere Variablen-Ersetzung & nur benutzte Spalten
  // ----------------------------
  function makeSafeIdent(name) {
    let s = name.replace(/[^A-Za-z0-9_$]/g, "_");
    if (/^[0-9]/.test(s)) s = "_" + s;
    return s;
  }
  function escRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }
  function replaceVarTokens(expr, original, replacement) {
    const pat = new RegExp(`(^|[^A-Za-z0-9_$])(${escRe(original)})(?![A-Za-z0-9_$])`, "g");
    return expr.replace(pat, (_, pre) => pre + replacement);
  }
  function findUsedColumns(expr, columns) {
    const used = [];
    for (const colName of Object.keys(columns)) {
      if (expr.indexOf(colName) === -1) continue;
      const pat = new RegExp(`(^|[^A-Za-z0-9_$])(${escRe(colName)})(?![A-Za-z0-9_$])`);
      if (pat.test(expr)) used.push(colName);
    }
    return used;
  }

  function evaluateExpressionVector(expr, envColumns) {
    if (!expr) throw new Error("Leerer Ausdruck.");
    if (ILLEGAL.test(expr)) throw new Error("Ausdruck enthält unzulässige Tokens.");

    const allKeys = Object.keys(envColumns);
    if (!allKeys.length) throw new Error("Keine Daten geladen.");

    // Nur verwendete Spalten
    const usedCols = findUsedColumns(expr, envColumns);

    // Mapping & Ausdruck transformieren
    let transformed = expr;
    const usedSafe = [];
    for (const col of usedCols) {
      const safe = makeSafeIdent(col);
      transformed = replaceVarTokens(transformed, col, safe);
      usedSafe.push(safe);
    }

    const argNames = [...ALLOWED_FUNCS, ...usedSafe, "NaN", "Infinity"];
    const argValuesBase = [...ALLOWED_FUNCS.map(n => FN[n]), NaN, Infinity];

    let scalarFn;
    try {
      scalarFn = new Function(...argNames, `"use strict"; return (${transformed});`);
    } catch (e) {
      throw new Error("Parse-Fehler im Ausdruck: " + e.message);
    }

    const N = envColumns[allKeys[0]]?.length ?? 0;
    const out = new Float64Array(N);
    const usedArrays = usedCols.map(name => envColumns[name]);

    for (let i=0;i<N;i++) {
      const argValues = argValuesBase.slice(0, ALLOWED_FUNCS.length); // Funktionen
      // NaN/Infinity am Ende ergänzen
      argValues.push(NaN, Infinity);
      // Variablen einfügen (an korrekter Position direkt nach den Funktionsargs)
      for (let k=0;k<usedArrays.length;k++) {
        const col = usedArrays[k];
        const v = (col instanceof Float64Array) ? col[i]
              : (Array.isArray(col) ? col[i] : undefined);
        argValues.splice(ALLOWED_FUNCS.length + k, 0, v);
      }
      let val;
      try {
        val = scalarFn(...argValues);
      } catch (e) {
        throw new Error(`Laufzeitfehler (Zeile ${i}): ${e.message}`);
      }
      out[i] = Number.isFinite(val) ? val : NaN;
    }
    return out;
  }

  // ----------------------------
  // CSV & Helpers
  // ----------------------------
  async function fetchText(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`Fetch fehlgeschlagen (${res.status}) für ${url}`);
    return await res.text();
  }

  function parseCSV(text) {
    const lines = text.replace(/\r/g, "").trim().split("\n");
    if (!lines.length) return { header: [], rows: [], columns: {} };
    const header = splitCsvLine(lines[0]).map(s => s.trim());
    const rows = [];
    for (let i=1;i<lines.length;i++) {
      const cols = splitCsvLine(lines[i]);
      const obj = {};
      header.forEach((h, j) => { obj[h] = cols[j] ?? ""; });
      rows.push(obj);
    }
    const columns = {};
    header.forEach(h => {
      const arr = rows.map(r => r[h]);
      const nums = new Float64Array(arr.length);
      let numeric = true;
      for (let i=0;i<arr.length;i++){
        const v = String(arr[i]).trim();
        if (v === "" || v.toLowerCase() === "nan") { nums[i] = NaN; continue; }
        const n = Number(v);
        if (!Number.isFinite(n)) { numeric = false; break; }
        nums[i] = n;
      }
      columns[h] = numeric ? nums : arr;
    });
    return { header, rows, columns };
  }

  function splitCsvLine(line) {
    const out = [];
    let cur = "", inQ = false;
    for (let i=0;i<line.length;i++){
      const ch = line[i];
      if (ch === '"') {
        if (inQ && line[i+1] === '"') { cur += '"'; i++; }
        else inQ = !inQ;
      } else if (ch === "," && !inQ) {
        out.push(cur); cur = "";
      } else cur += ch;
    }
    out.push(cur);
    return out;
  }

  function toCSV(rows, headerOrder = null) {
    if (!rows.length) return "";
    const headers = headerOrder ?? Object.keys(rows[0]);
    const escape = v => {
      const s = String(v ?? "");
      return (/[",\n]/.test(s)) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const lines = [];
    lines.push(headers.join(","));
    for (const r of rows) lines.push(headers.map(h => escape(r[h])).join(","));
    return lines.join("\n");
  }

  function downloadCSV(filename, csvText) {
    const blob = new Blob([csvText], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  }

  function insertAtCursor(input, text) {
    input.focus();
    const s = input.selectionStart ?? input.value.length;
    const e = input.selectionEnd ?? input.value.length;
    const before = input.value.slice(0, s);
    const after  = input.value.slice(e);
    input.value = before + text + after;
    const newPos = before.length + text.length;
    input.selectionStart = input.selectionEnd = newPos;
  }

  function formatPreview(title, vec) {
    const head = Array.from(vec.slice(0,5)).map(v => Number.isFinite(v) ? +v.toPrecision(6) : "NaN");
    const stats = { min: numMin(vec), max: numMax(vec), mean: numMean(vec), nan: numNan(vec) };
    return `${title}\n  head: ${JSON.stringify(head)}\n  stats: ${JSON.stringify(stats)}\n\n`;
  }
  function numNan(a){ let c=0; for (let i=0;i<a.length;i++) if(!Number.isFinite(a[i])) c++; return c; }
  function numMin(a){ let m=Infinity; for (let i=0;i<a.length;i++){ const v=a[i]; if(Number.isFinite(v)&&v<m) m=v; } return (m===Infinity?NaN:m); }
  function numMax(a){ let m=-Infinity; for (let i=0;i<a.length;i++){ const v=a[i]; if(Number.isFinite(v)&&v>m) m=v; } return (m===-Infinity?NaN:m); }
  function numMean(a){ let s=0,c=0; for (let i=0;i<a.length;i++){ const v=a[i]; if(Number.isFinite(v)){ s+=v; c++; } } return c? s/c : NaN; }

  function showInfo(msg){ if($preview) $preview.textContent = msg; }
  function showError(msg){ if($preview) $preview.textContent = "Fehler: " + msg; }
  function showSuccess(msg){ if($preview) $preview.textContent = msg; }

  function ensureHiddenCSS(){
    const style = document.createElement("style");
    style.textContent = `.modal[hidden]{display:none!important}`;
    document.head.appendChild(style);
  }
})();