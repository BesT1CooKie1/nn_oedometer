# === Traceable Trainingsdatensatz für Oedometer ===============================
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

def _load_oedometer_class(oedo_model=0):
    if oedo_model == 0:
        from classes.classOedometerSimple import Oedometer
    else:
        from classes.classOedometerImproved import Oedometer
    return Oedometer

def _is_range(x):
    return isinstance(x, (list, tuple)) and len(x) == 2 and all(isinstance(v, (int, float)) for v in x)

def _sample_value(spec, rng):
    if _is_range(spec):
        lo, hi = float(spec[0]), float(spec[1])
        if lo > hi:
            lo, hi = hi, lo
        return rng.uniform(lo, hi)
    return float(spec)

def _sample_params(param_spec, rng):
    params = {}
    for k, v in param_spec.items():
        params[k] = _sample_value(v, rng)
    return params

# --- Helfer: farthest-point (diverse) auf Submenge ----------------------------
def _farthest_point_indices(X_sub, k, rng):
    """Greedy farthest-point auf X_sub (np.ndarray mit shape (n,d))."""
    n = X_sub.shape[0]
    if k >= n:
        return np.arange(n, dtype=int)
    idxs = []
    first = int(rng.integers(0, n))
    idxs.append(first)
    d2 = np.sum((X_sub - X_sub[first])**2, axis=1)
    for _ in range(1, k):
        nxt = int(np.argmax(d2))
        idxs.append(nxt)
        d2 = np.minimum(d2, np.sum((X_sub - X_sub[nxt])**2, axis=1))
    return np.array(idxs, dtype=int)

# --- Helfer: stratifiziertes Subsampling nach Phase ---------------------------
def _stratified_subsample(X, phases, final_samples, ratio_ext=0.5,
                          method="diverse", rng=None):
    """
    phases: np.array(["compression"|"extension"]) Länge N
    ratio_ext: Zielanteil für 'extension' in [0,1], Rest = compression
    Gibt globale Indizes zurück (np.array).
    """
    N = X.shape[0]
    mask_ext  = (phases == "extension")
    mask_comp = ~mask_ext

    idx_all = np.arange(N, dtype=int)
    idx_ext_all  = idx_all[mask_ext]
    idx_comp_all = idx_all[mask_comp]

    n_ext_target  = int(round(final_samples * ratio_ext))
    n_comp_target = final_samples - n_ext_target

    # Verfügbarkeit prüfen und Reste umschichten
    n_ext_avail, n_comp_avail = idx_ext_all.size, idx_comp_all.size
    take_ext  = min(n_ext_target,  n_ext_avail)
    take_comp = min(n_comp_target, n_comp_avail)

    # Rest auffüllen, falls eine Seite knapp ist
    rem = final_samples - (take_ext + take_comp)
    if rem > 0:
        if n_ext_avail - take_ext >= n_comp_avail - take_comp:
            add = min(rem, n_ext_avail - take_ext)
            take_ext += add
            rem -= add
        if rem > 0:
            add = min(rem, n_comp_avail - take_comp)
            take_comp += add
            rem -= add

    # Sicherheit: mind. 1 pro Phase, wenn vorhanden
    if take_ext == 0 and n_ext_avail > 0:
        take_ext = 1
        if take_comp > 1:
            take_comp -= 1
    if take_comp == 0 and n_comp_avail > 0:
        take_comp = 1
        if take_ext > 1:
            take_ext -= 1

    # Auswahl je Schicht
    chosen = []

    if take_comp > 0 and idx_comp_all.size > 0:
        Xc = X[idx_comp_all]
        if method == "random":
            sel = rng.choice(idx_comp_all, size=take_comp, replace=False)
        else:
            loc = _farthest_point_indices(Xc, take_comp, rng)
            sel = idx_comp_all[loc]
        chosen.append(sel)

    if take_ext > 0 and idx_ext_all.size > 0:
        Xe = X[idx_ext_all]
        if method == "random":
            sel = rng.choice(idx_ext_all, size=take_ext, replace=False)
        else:
            loc = _farthest_point_indices(Xe, take_ext, rng)
            sel = idx_ext_all[loc]
        chosen.append(sel)

    if not chosen:
        # Fallback (sollte praktisch nie passieren)
        return rng.choice(N, size=final_samples, replace=False)

    idx = np.concatenate(chosen)
    # Falls durch Rundung minimal zu viel/zu wenig: hart trimmen/auffüllen
    if idx.size > final_samples:
        idx = rng.choice(idx, size=final_samples, replace=False)
    elif idx.size < final_samples:
        rest = np.setdiff1d(np.arange(N), idx, assume_unique=False)
        add  = rng.choice(rest, size=(final_samples - idx.size), replace=False)
        idx  = np.concatenate([idx, add])

    return idx


def generate_oedometer_dataset(
    param_spec,
    n_runs=50,
    final_samples=100,
    oedo_model=0,
    subsample_method="diverse",
    seed=None,
    device=None,
    debug=False,
    feature_keys=("sigma_0","eps_delta"),   # NEU: frei wählbare Feature-Spalten
    target_key="e_s"                        # NEU: frei wählbare Ziel-Spalte
):
    rng = np.random.default_rng(seed)
    OedometerClass = _load_oedometer_class(oedo_model)

    # Sammelcontainer für rohe Felder (damit wir flexibel auswählen können)
    _buf = { "sigma_0": [], "eps_delta": [], "e_s": [], "sigma_delta": [], "eps_total": [] }
    records, runs = [], []
    total_before = 0

    for run_id in range(n_runs):
        params = _sample_params(param_spec, rng)
        oed = OedometerClass(**params)
        df = oed.run(debug=debug)

        run_meta = {
            "run_id": run_id,
            "params": dict(params),
            "sigma_path": (oed.sigma_0_list_compress + oed.sigma_0_list_extension),
            "eps_path":   (oed.eps_list_compress + oed.eps_list_extension),
            "sigma_delta_path": (oed.sigma_delta_list_compress + oed.sigma_delta_list_extension),
            "cut": len(oed.sigma_delta_list_compress) - 1
        }
        runs.append(run_meta)

        for step_idx, row in df.reset_index(drop=True).iterrows():
            s0 = float(row["sigma_0"])
            ed = float(row["eps_delta"])
            es = float(row["e_s"])
            ds = float(row["sigma_delta"])
            et = float(row["eps_total"])

            _buf["sigma_0"].append(s0)
            _buf["eps_delta"].append(ed)
            _buf["e_s"].append(es)
            _buf["sigma_delta"].append(ds)
            _buf["eps_total"].append(et)

            records.append({
                "run_id": run_id,
                "step_idx": int(step_idx),
                "sigma_0": s0,
                "eps_delta": ed,
                "e_s": es,
                "sigma_delta": ds,
                "eps_total": et,
                "phase": "compression" if ed < 0 else "extension",
                "params": dict(params),
            })
            total_before += 1

    # In NumPy umwandeln
    data_np = {k: np.asarray(v, dtype=np.float32) for k, v in _buf.items()}
    N_total = len(records)

    # Feature-/Target-Matrizen gemäß Auswahl bauen
    try:
        X = np.stack([data_np[k] for k in feature_keys], axis=1)      # (N, d)
        Y = data_np[target_key].reshape(-1, 1)                         # (N, 1)
    except KeyError as e:
        raise KeyError(f"Unbekannter Spaltenname {e}. Erlaubt sind: {list(data_np.keys())}")

    if X.shape[0] != Y.shape[0]:
        raise ValueError("Formfehler: Feature- und Target-Länge passen nicht zusammen.")

    phases = np.array([rec["phase"] for rec in records], dtype=object)
    final_samples = min(final_samples, X.shape[0])
    ratio_ext = 0.5

    # Subsampling (stratifiziert)
    run_ids = np.array([rec["run_id"] for rec in records], dtype=int)

    # --- Sonderfall: keine Extension vorhanden -> run-stratifizierte Auswahl ---
    has_ext = np.any(phases == "extension")
    if not has_ext:
        # Round-Robin Zielverteilung über Runs (mind. 1 pro Run, solange Budget reicht)
        unique_runs, counts = np.unique(run_ids, return_counts=True)
        R = unique_runs.size
        take = np.zeros(R, dtype=int)

        # 1) einmal jedem Run 1 Sample geben (solange Budget)
        budget = int(final_samples)
        for i in range(R):
            if budget == 0: break
            if counts[i] > 0:
                take[i] += 1
                budget -= 1

        # 2) Rest proportional zu Verfügbarkeit auffüllen (sehr simpel & fair)
        if budget > 0:
            # verfügbare Restslots pro Run
            rest_cap = counts - take
            # solange Budget existiert, verteile im Round-Robin über Runs mit Restkapazität
            i = 0
            while budget > 0 and np.any(rest_cap > 0):
                if rest_cap[i] > 0:
                    take[i] += 1
                    rest_cap[i] -= 1
                    budget -= 1
                i = (i + 1) % R

        # 3) innerhalb jedes Runs per 'diverse' oder 'random' auswählen
        chosen = []
        for r_idx, r in enumerate(unique_runs):
            need = int(take[r_idx])
            if need <= 0:
                continue
            mask = (run_ids == r)
            idx_r_all = np.flatnonzero(mask)
            if need >= idx_r_all.size:
                chosen.append(idx_r_all)
            else:
                if subsample_method == "random":
                    sel = rng.choice(idx_r_all, size=need, replace=False)
                else:
                    loc = _farthest_point_indices(X[idx_r_all], need, rng)
                    sel = idx_r_all[loc]
                chosen.append(sel)

        idx = np.concatenate(chosen) if chosen else rng.choice(X.shape[0], size=final_samples, replace=False)

    else:
        # --- wie bisher: phasen-stratifiziert ---
        idx = _stratified_subsample(
            X, phases, final_samples,
            ratio_ext=0.5,
            method=subsample_method,
            rng=rng,
        )

    tX = torch.tensor(X[idx], dtype=torch.float32, device=device)
    tY = torch.tensor(Y[idx], dtype=torch.float32, device=device)

    info = {
        "indices": idx,
        "total_before_subsample": N_total,
        "records": records,
        "runs": runs,
        "feature_names": list(feature_keys),  # NEU: reflektiert deine Auswahl
        "target_name": str(target_key),       # NEU
        "oedo_model": oedo_model,
        "param_spec": param_spec,
        "seed": seed,
        "selected_counts": {
            "compression": int(np.sum(phases[idx] == "compression")),
            "extension":   int(np.sum(phases[idx] == "extension")),
        },
        "selected_ratio_extension": float(np.mean(phases[idx] == "extension")),
    }
    return tX, tY, info


def export_dataset_for_html(info, out_dir="../oedo-viewer/viewer_data/", samples_csv="samples.csv", runs_csv="runs.csv"):
    """
    Erzeugt zwei CSVs:
      - samples.csv: ausgewählte Samples (entspricht info["indices"])
      - runs.csv:    vollständige Pfade je run_id, pro Step, mit allen Metadaten
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    # --- 1) SAMPLES unverändert (wie bei dir) -------------------------------
    rows = []
    for rank, global_idx in enumerate(info["indices"]):
        rec = info["records"][int(global_idx)]
        row = {
            "rank": int(rank),
            "global_idx": int(global_idx),
            "run_id": rec["run_id"],
            "step_idx": rec["step_idx"],
            "phase": rec["phase"],
            "sigma_0": rec["sigma_0"],
            "eps_delta": rec["eps_delta"],
            "e_s": rec["e_s"],
            "sigma_delta": rec["sigma_delta"],
            "eps_total": rec["eps_total"],
        }
        for k, v in rec["params"].items():
            row[f"param.{k}"] = v
        rows.append(row)

    df_samples = pd.DataFrame(rows).sort_values(["rank"]).reset_index(drop=True)
    df_samples.to_csv(os.path.join(out_dir, samples_csv), index=False)

    # --- 2) RUNS: aus info["records"] statt aus run_meta.*_path -------------
    # Ziel: volle Nachvollziehbarkeit pro Step mit ALLEN Feldern
    recs = info["records"]
    # Parameternamen stabilisieren (falls runs heterogen sind)
    param_keys = sorted({k for r in recs for k in r["params"].keys()})
    # run-spezifische "cut"-Marke (optional, für Split comp/ext)
    cut_by_run = {rm["run_id"]: rm.get("cut") for rm in info.get("runs", [])}

    run_rows = []
    for r in sorted(recs, key=lambda x: (x["run_id"], x["step_idx"])):
        base = {
            "run_id": int(r["run_id"]),
            "step_idx": int(r["step_idx"]),
            # Mindest-Set (abwärtskompatibel zum alten runs.csv):
            "sigma_0": float(r["sigma_0"]),
            "eps_total": float(r["eps_total"]),
            # Zusätzliche, von dir geforderte Felder:
            "eps_delta": float(r["eps_delta"]),
            "e_s": float(r["e_s"]),
            "sigma_delta": float(r["sigma_delta"]),
            "phase": r["phase"],
            "cut": cut_by_run.get(r["run_id"], None),
        }
        # Param-Säulen anflachen (pro Zeile identisch für den Run – bewusst redundant, aber tracebar)
        for pk in param_keys:
            base[f"param.{pk}"] = r["params"].get(pk)
        run_rows.append(base)

    # Spaltenreihenfolge festzurren (alte zuerst, dann neue, dann params):
    base_cols = ["run_id", "step_idx", "sigma_0", "eps_total"]
    extra_cols = ["eps_delta", "e_s", "sigma_delta", "phase", "cut"]
    param_cols = [c for c in (f"param.{pk}" for pk in param_keys)]
    cols = base_cols + extra_cols + param_cols
    df_runs = pd.DataFrame(run_rows)[cols].sort_values(["run_id", "step_idx"]).reset_index(drop=True)
    df_runs.to_csv(os.path.join(out_dir, runs_csv), index=False)
    df_runs.to_csv(os.path.join(out_dir, "additional_runs.csv"), index=False)
    # Check: step_idx fortlaufend je Run
    grp = df_runs.groupby("run_id")["step_idx"]
    bad = [rid for rid, s in grp if (s != range(s.iloc[0], s.iloc[0]+len(s))).any()]
    if bad:
        print(f"[WARN] Nicht-fortlaufende step_idx in Runs: {bad}")

    # Check: phase-Grenze vs. cut
    if "cut" in df_runs.columns:
        g = df_runs.groupby("run_id")
        for rid, sub in g:
            c = sub["cut"].iloc[0]
            if pd.notna(c):
                if not ((sub.loc[sub["step_idx"]<=c, "phase"]=="compression").all() and
                        (sub.loc[sub["step_idx"]> c, "phase"]=="extension").all()):
                    print(f"[WARN] phase stimmt nicht zur cut-Grenze in run {rid}")

    print(f"[OK] exportiert: {os.path.join(out_dir, samples_csv)}  /  {os.path.join(out_dir, runs_csv)}")


import json
from pathlib import Path

def export_oedometer_schema(feature_keys, target_key, origin, n_runs, final_samples, seed,
                            path="../oedo-viewer/viewer_data/schema.json"):
    """
    Exportiert die Oedometer-Schema-Definition als JSON.

    Args:
        feature_keys (list[str] | tuple[str]): Eingangsgrößen (X).
        target_key (str): Zielgröße (Y).
        origin (str): Name der Ursprungfunktion (z.B. "generate_oedometer_dataset").
        n_runs (int): Anzahl der Runs.
        final_samples (int): Anzahl der finalen Samples.
        seed (int): Zufallssaat.
        path (str): Speicherpfad der JSON-Datei.
    """

    schema = {
        "feature_keys": list(feature_keys),
        "target_key": target_key,
        "info": {
            "origin": origin,
            "n_runs": n_runs,
            "final_samples": final_samples,
            "seed": seed
        }
    }

    # Ordner erzeugen, falls nicht vorhanden
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    return schema

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import joblib

# ===== vorhandene Scaler-Helper (wie bei dir) =====
def fit_normalizers(X: np.ndarray, y: np.ndarray,
                    save_path: Optional[Union[str, Path]] = None
                   ) -> Tuple[StandardScaler, StandardScaler]:
    xsc = StandardScaler()
    ysc = StandardScaler()
    xsc.fit(X)
    ysc.fit(y)
    if save_path is not None:
        joblib.dump((xsc, ysc), save_path)
    return xsc, ysc

def load_normalizers(load_path: Union[str, Path]) -> Tuple[StandardScaler, StandardScaler]:
    return joblib.load(load_path)

# ===== neue Hauptfunktion mit Zeilenindex-Join =====
def compose_dataset_from_files(
    samples_csv: Union[str, Path],
    additional_runs_csv: Union[str, Path],
    schema_json: Union[str, Path],
    *,
    # Auswahl / Erkennung
    feature_keys: Optional[Sequence[str]] = None,   # None => aus schema.json
    target_keys: Optional[Sequence[str]] = None,    # None => aus schema.json ('target_key' oder 'target_keys')
    include_additional_features: Optional[Sequence[str]] = None,  # None => alle F:-Spalten
    include_additional_targets: Optional[Sequence[str]] = None,   # None => alle T:-Spalten
    # Join-Steuerung
    join_key: str = "global_index",
    join_how: str = "left",         # "left" (Samples als Master) oder "inner"
    # NEU: falls Key in CSV fehlt -> aus Zeilenindex erzeugen
    additional_index_from_row: bool = True,
    additional_index_start: int = 0,      # 0 => nullbasiert; stell auf 1, wenn deine Samples 1-basiert zählen
    samples_index_from_row: bool = False, # optional, meist False weil samples.csv i.d.R. die Spalte schon hat
    samples_index_start: int = 0,
    # Cleaning
    dropna: bool = True,
    # Normalisierung & Torch
    normalize: bool = False,
    scaler_path: Optional[Union[str, Path]] = None,
    refit_scaler: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    # Ausgabe
    additional_samples_out: Optional[Union[str, Path]] = "oedo-viewer/viewer_data/additional_samples.csv",
    # NEW: ensure run id is carried through and saved
    run_col: str = "run_id",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Liest samples.csv, additional_runs.csv, schema.json; merged über 'join_key'.
    Schreibt 'additional_samples.csv' mit [join_key, run_id, X..., Y...] (falls run_id vorhanden)
    und gibt (tX, tY, meta) zurück. Unterstützt mehrere Targets (Y-Shape: (N, T)).

    - F:-Spalten in additional_runs => zusätzliche Features
    - T:-Spalten in additional_runs => zusätzliche Targets
    """

    samples_csv = Path(samples_csv)
    additional_runs_csv = Path(additional_runs_csv)
    schema_json = Path(schema_json)

    # --- Laden ---
    df_s = pd.read_csv(samples_csv)
    df_a = pd.read_csv(additional_runs_csv)
    with open(schema_json, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # --- Join-Key ggf. aus Zeilenindex erzeugen ---
    if join_key not in df_a.columns:
        if not additional_index_from_row:
            raise KeyError(
                f"Join-Key '{join_key}' fehlt in additional_runs.csv "
                f"und additional_index_from_row=False."
            )
        df_a = df_a.copy()
        df_a[join_key] = np.arange(additional_index_start,
                                   additional_index_start + len(df_a),
                                   dtype=int)

    if join_key not in df_s.columns:
        if not samples_index_from_row:
            raise KeyError(
                f"Join-Key '{join_key}' fehlt in samples.csv "
                f"und samples_index_from_row=False."
            )
        df_s = df_s.copy()
        df_s[join_key] = np.arange(samples_index_start,
                                   samples_index_start + len(df_s),
                                   dtype=int)

    # --- Basis-Features/Targets aus Schema ---
    if feature_keys is None:
        if "feature_keys" in schema and isinstance(schema["feature_keys"], list):
            feature_keys = list(schema["feature_keys"])
        else:
            raise ValueError("schema.json enthält keine 'feature_keys' und feature_keys wurde nicht übergeben.")

    if target_keys is None:
        if "target_keys" in schema and isinstance(schema["target_keys"], list):
            target_keys = list(schema["target_keys"])
        elif "target_key" in schema and isinstance(schema["target_key"], str):
            target_keys = [schema["target_key"]]
        else:
            target_keys = []  # erlaubt: nur T:-Targets nutzen

    # --- Zusätzliche Spalten (F:/T:) erkennen & filtern ---
    addl_feature_cols = [c for c in df_a.columns if isinstance(c, str) and c.startswith("F:")]
    addl_target_cols  = [c for c in df_a.columns if isinstance(c, str) and c.startswith("T:")]

    if include_additional_features is not None:
        missing = set(include_additional_features) - set(addl_feature_cols)
        if missing:
            raise KeyError(f"Gewünschte zusätzliche Feature-Spalten fehlen: {sorted(missing)}")
        addl_feature_cols = list(include_additional_features)

    if include_additional_targets is not None:
        missing = set(include_additional_targets) - set(addl_target_cols)
        if missing:
            raise KeyError(f"Gewünschte zusätzliche Target-Spalten fehlen: {sorted(missing)}")
        addl_target_cols = list(include_additional_targets)

    # --- Merge (Samples als Master per Default) ---
    # Ensure run_col is retained if present in either input
    keep_cols_from_s = [join_key]
    if run_col in df_s.columns:
        keep_cols_from_s.append(run_col)
    keep_cols_from_a = [join_key] + addl_feature_cols + addl_target_cols
    if run_col in df_a.columns and run_col not in keep_cols_from_s:
        keep_cols_from_a.append(run_col)

    df_m = pd.merge(
        df_s[keep_cols_from_s + [c for c in df_s.columns if c in (feature_keys or []) or c in (target_keys or [])]],
        df_a[keep_cols_from_a],
        on=join_key,
        how=join_how
    )

    # If run_col exists in both sides and pandas added suffixes, prefer df_s version
    if f"{run_col}_x" in df_m.columns or f"{run_col}_y" in df_m.columns:
        if f"{run_col}_x" in df_m.columns:
            df_m[run_col] = df_m[f"{run_col}_x"]
        elif f"{run_col}_y" in df_m.columns:
            df_m[run_col] = df_m[f"{run_col}_y"]
        df_m = df_m.drop(columns=[c for c in [f"{run_col}_x", f"{run_col}_y"] if c in df_m.columns])

    # --- X/Y Spaltenlisten ---
    X_cols = list(feature_keys) + addl_feature_cols
    Y_cols = list(target_keys)  + addl_target_cols

    for c in X_cols + Y_cols:
        if c not in df_m.columns:
            raise KeyError(f"Spalte '{c}' fehlt nach Merge. Prüfe schema/additional/samples.")

    # --- 'additional_samples.csv' (join_key + optional run_id + X + Y) ---
    df_out_cols = [join_key]
    if run_col in df_m.columns:
        df_out_cols.append(run_col)
    df_out_cols += X_cols + Y_cols

    df_out = df_m[df_out_cols].copy()

    # --- Cleaning ---
    dropped = 0
    if dropna:
        before = len(df_out)
        df_out = df_out.dropna(subset=[c for c in (X_cols + (Y_cols if Y_cols else []))])
        dropped = before - len(df_out)

    # --- Arrays ---
    X = df_out[X_cols].to_numpy(dtype=np.float32)
    if Y_cols:
        Y = df_out[Y_cols].to_numpy(dtype=np.float32)
    else:
        Y = np.empty((len(df_out), 0), dtype=np.float32)

    # --- Normalisierung (optional) ---
    xsc = ysc = None
    if normalize:
        if scaler_path is not None and Path(scaler_path).exists() and not refit_scaler:
            xsc, ysc = load_normalizers(scaler_path)
            X = xsc.transform(X).astype(np.float32)
            Y = ysc.transform(Y).astype(np.float32) if Y.shape[1] > 0 else Y
        else:
            # y braucht beim Fit min. 1 Spalte
            y_fit = Y if Y.shape[1] > 0 else np.zeros((X.shape[0], 1), dtype=np.float32)
            xsc, ysc = fit_normalizers(X, y_fit, save_path=scaler_path)
            X = xsc.transform(X).astype(np.float32)
            Y = ysc.transform(Y).astype(np.float32) if Y.shape[1] > 0 else Y

    # --- Persistenz: additional_samples.csv ---
    if additional_samples_out is not None:
        Path(additional_samples_out).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(additional_samples_out, index=False)

    # --- Torch ---
    tX = torch.tensor(X, dtype=torch.float32, device=device)
    tY = torch.tensor(Y, dtype=torch.float32, device=device)

    # --- Meta ---
    meta: Dict[str, Any] = {
        "join_key": join_key,
        "join_how": join_how,
        "dropped_rows": int(dropped),
        "feature_names": X_cols,
        "target_names": Y_cols,
        "additional_feature_cols": addl_feature_cols,
        "additional_target_cols": addl_target_cols,
        "x_scaler": xsc,
        "y_scaler": ysc,
        "additional_samples_path": str(additional_samples_out) if additional_samples_out else None,
        "schema_features": list(feature_keys),
        "schema_targets": list(target_keys),
        "additional_index_from_row": additional_index_from_row,
        "additional_index_start": int(additional_index_start),
        "samples_index_from_row": samples_index_from_row,
        "samples_index_start": int(samples_index_start),
        # NEW:
        "run_col": run_col,
        "run_col_present": bool(run_col in df_out.columns),
    }
    return tX, tY, meta


