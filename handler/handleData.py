from __future__ import annotations
from pathlib import Path
from typing import Union, List, Dict
from numpy import sign
from classes.classOedometerImproved import Oedometer
from pathlib import Path
from typing import Dict, Optional, Tuple, Sequence, Any, List
import numpy as np
import pandas as pd
import torch


def load_predictions_csv(path: Union[str, Path]) -> List[Dict]:
    """
    Lädt predictions_additional.csv (oder eine beliebige CSV mit Vorhersagen)
    und gibt eine Liste von Dicts zurück.
    """
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def load_runs_csv(path: Union[str, Path]) -> List[Dict]:
    """
    Lädt runs.csv (oder eine Variante wie runs_SimpleModel.csv)
    und gibt eine Liste von Dicts zurück.
    """
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


# === process_pred_3stage.py ===================================================


# Expect these to be available in your project:
# from your_module import Oedometer, load_predictions_csv, load_runs_csv, sign


# ---------------------------------------------------------------------
# 1) SINGLE-SAMPLE PREDICTIONS (compute pred_sigma_0 for each row)
# ---------------------------------------------------------------------
def compute_single_predictions(
        *,
        oedo_model: int = 1,
        root_dir: str = "oedo-viewer/viewer_data/",
        preds_path: str = "predictions_additional.csv",
        runs_path: str = "runs.csv",
        merged_path: str = "predictions_additional_processed.csv",
) -> pd.DataFrame:
    """
    For every row i:
      - read pred_e_s from predictions_additional.csv
      - invert E_s -> pred_sigma_0 using the chosen oedometer model and row params
    Writes merged CSV (same length as runs) that includes 'pred_sigma_0'.
    Returns the merged DataFrame as a convenience.
    """
    preds = load_predictions_csv(root_dir + preds_path)  # list-like of dict
    runs = load_runs_csv(root_dir + runs_path)  # list-like of dict

    for i in range(len(runs)):
        r = runs[i];
        p = preds[i]
        e0 = r["param.e0"];
        c_s = r["param.c_s"];
        c_c = r["param.c_c"]
        sgn = sign(r["eps_delta"])
        e_s = p["pred_e_s"]

        if oedo_model == 0:
            p["pred_sigma_0"] = - e_s / ((1.0 + e0) / c_c)
        else:
            c1 = - (1.0 + e0) / 2.0 * (c_c + c_s) / (c_s * c_c)
            c2 = - (1.0 + e0) / 2.0 * (c_c - c_s) / (c_s * c_c)
            p["pred_sigma_0"] = e_s / (c1 + sgn * c2)

    df_merged = pd.DataFrame(preds)
    df_merged.to_csv(root_dir + merged_path, index=False)
    return df_merged


# ---------------------------------------------------------------------
# Helpers shared by (2) and (3)
# ---------------------------------------------------------------------
def _load_sorted_test_indices(split_csv: str) -> Tuple[str, set[int]]:
    """Load split_assignments.csv, sort by its first column, and return (key_name, set_of_TEST_indices)."""
    df_split = pd.read_csv(split_csv)
    first_col = df_split.columns[0]
    df_split = df_split.sort_values(first_col, kind="mergesort")

    split_col = "split" if "split" in df_split.columns else df_split.columns[-1]
    test_gidx = set(
        df_split.loc[
            df_split[split_col].astype(str).str.lower().str.strip().isin(["test", "testing"]),
            first_col
        ].astype(int).tolist()
    )
    return first_col, test_gidx


def _group_runs_with_indices(runs: List[dict]) -> tuple[
    Dict[int, list[tuple[int, int, int]]], Dict[tuple[int, int], int]]:
    """
    Returns:
      - by_run_rows: run_id -> list of (step_idx, global_idx, i_row) sorted by step_idx
      - idx_by_run_step: (run_id, step_idx) -> i_row
    Also ensures each run row has a global_idx field (row number if absent).
    """
    from collections import defaultdict
    by_run_rows = defaultdict(list)
    idx_by_run_step = {}

    for j, r in enumerate(runs):
        if "global_idx" not in r:
            r["global_idx"] = j
        t = (int(r["step_idx"]), int(r["global_idx"]), j)
        by_run_rows[r["run_id"]].append(t)
        idx_by_run_step[(r["run_id"], int(r["step_idx"]))] = j

    for rid in by_run_rows:
        by_run_rows[rid].sort(key=lambda t: t[0])

    return by_run_rows, idx_by_run_step


# ---------------------------------------------------------------------
# 2) DIRECT PHYSICS PROPAGATION from the first TEST row per run
# ---------------------------------------------------------------------
def propagate_direct_from_first_test(
        *,
        root_dir: str = "oedo-viewer/viewer_data/",
        runs_path: str = "runs.csv",
        merged_path: str = "predictions_additional_processed.csv",
        split_path: str = "split_assignments.csv",
        out_path: str = "predictions_additional_processed_propagated.csv",
) -> pd.DataFrame:
    """
    For each run:
      - find the FIRST TEST row (via split_assignments.csv first column == global_idx)
      - seed Oedometer with pred_sigma_0 at that row (from merged CSV)
      - run full Oedometer; write rows *aligned to the original run*: None before seed, then filled.
    Writes 'out_path' and returns the DataFrame.
    """
    runs = load_runs_csv(root_dir + runs_path)
    dfm = pd.read_csv(root_dir + merged_path)  # must contain 'pred_sigma_0'
    gkey, test_gidx = _load_sorted_test_indices(root_dir + split_path)

    # sanity: align by plain row order
    for j, r in enumerate(runs):
        r["global_idx"] = j
        # also copy pred_sigma_0 into a list of dicts so indexing matches runs
        runs[j]["pred_sigma_0"] = float(dfm.loc[j, "pred_sigma_0"]) if "pred_sigma_0" in dfm.columns else None

    by_run_rows, _ = _group_runs_with_indices(runs)

    out_rows: list[dict] = []
    # prefill skeleton
    for rid, steps in by_run_rows.items():
        for step_idx, gidx, _ in steps:
            out_rows.append({
                "run_id": rid, "step_idx": step_idx, "global_idx": gidx,
                "sigma_0": None, "e_s": None, "eps_total": None
            })

    # quick index
    index_out = {(row["run_id"], row["step_idx"]): k for k, row in enumerate(out_rows)}

    # propagate
    for rid, steps in by_run_rows.items():
        # find first TEST step
        seed = None
        for step_idx, gidx, i_row in steps:
            if gidx in test_gidx:
                seed = (step_idx, gidx, i_row);
                break
        if seed is None:
            continue

        seed_step, _, seed_i = seed
        r0 = runs[seed_i]
        sigma0_seed = r0.get("pred_sigma_0", None)
        if sigma0_seed is None:
            continue

        prop = Oedometer(
            r0["param.e0"], r0["param.c_c"], r0["param.c_s"], float(sigma0_seed),
            r0["param.sigma_max"], r0["param.sigma_min"], r0["param.eps_delta"], r0["eps_total"]
        )
        prop.run()

        max_step = steps[-1][0]
        for k in range(len(prop.sigma_0_list)):
            abs_step = seed_step + k
            if abs_step > max_step: break
            j = index_out.get((rid, abs_step))
            if j is not None:
                out_rows[j]["sigma_0"] = prop.sigma_0_list[k]
                out_rows[j]["e_s"] = prop.e_s_list[k]
                out_rows[j]["eps_total"] = prop.eps_s_list[k]

    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(root_dir + out_path, index=False)
    return df_out


# ---------------------------------------------------------------------
# 3) RECURSIVE PROPAGATION (Oedometer step -> model Es -> invert -> feed back)
# ---------------------------------------------------------------------
def propagate_recursive_from_first_test(
        *,
        oedo_model: int = 1,
        root_dir: str = "oedo-viewer/viewer_data/",
        runs_path: str = "runs.csv",
        merged_path: str = "predictions_additional_processed.csv",
        split_path: str = "split_assignments.csv",
        out_path: str = "predictions_additional_processed_propagated_recursive.csv",
        # model + (optional) scalers
        model: Optional[torch.nn.Module] = None,
        x_scaler=None,
        y_scaler=None,
        device: Optional[str | torch.device] = None,
) -> Optional[pd.DataFrame]:
    """
    For each run, starting at its FIRST TEST row:
      σ_cur = σ0_pred(seed)
      loop j = 1..:
        (σ_phys_next, eps_total_next) = Oedometer one-step advance from (σ_cur, eps_state)
        Ês = model([σ_phys_next, Δε])  (with optional x_scaler/y_scaler)
        σ_ML_next = invert(Ês, params)
        write row at abs_step = seed_step + j:
           sigma_0_rec = σ_ML_next, e_s_rec = Ês, eps_total_rec = eps_total_next
        feedback: σ_cur = σ_ML_next, eps_state = eps_total_next
      stop at end of run.
    Writes 'out_path'. Returns DataFrame, or None if model is None.
    """
    if model is None:
        return None

    # --- load and align ---
    runs = load_runs_csv(root_dir + runs_path)
    dfm = pd.read_csv(root_dir + merged_path)
    gkey, test_gidx = _load_sorted_test_indices(root_dir + split_path)

    for j, r in enumerate(runs):
        r["global_idx"] = j
        runs[j]["pred_sigma_0"] = float(dfm.loc[j, "pred_sigma_0"]) if "pred_sigma_0" in dfm.columns else None

    by_run_rows, idx_by_run_step = _group_runs_with_indices(runs)

    # --- model wrappers ---
    model_device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = model.to(model_device).eval()

    @torch.no_grad()
    def model_predict_es(sigma0_val: float, eps_delta_val: float) -> float:
        # X = [σ0, Δε]
        x = np.array([[sigma0_val, eps_delta_val]], dtype=np.float32)
        if x_scaler is not None:
            x = x_scaler.transform(x).astype(np.float32)
        xt = torch.from_numpy(x).to(model_device)
        yhat = model(xt)
        y = yhat.detach().cpu().numpy().reshape(-1, 1)
        if y_scaler is not None:
            y = y_scaler.inverse_transform(y)
        return float(y[0, 0])

    def invert_es_to_sigma(e_s_val: float, e0: float, c_s: float, c_c: float, sgn: float) -> float:
        if oedo_model == 0:
            return - e_s_val / ((1.0 + e0) / c_c)
        else:
            c1 = - (1.0 + e0) / 2.0 * (c_c + c_s) / (c_s * c_c)
            c2 = - (1.0 + e0) / 2.0 * (c_c - c_s) / (c_s * c_c)
            return e_s_val / (c1 + sgn * c2)

    # helper: one Oedometer *step* from (σ_cur, eps_state) -> (σ_next, eps_total_next)
    def oedo_one_step(e0, c_c, c_s, sigma_cur, sigma_max, sigma_min, eps_delta, eps_state):
        """
        Execute exactly ONE increment using the Oedometer model.

        Parameters
        ----------
        e0 : float
            Initial void ratio
        c_c : float
            Compression index
        c_s : float
            Swelling/recompression index
        sigma_cur : float
            Current effective stress [kPa]
        sigma_max : float
            Lower stress bound for compression
        sigma_min : float
            Upper stress bound for extension
        eps_delta : float
            Increment step size (negative = compression, positive = extension)
        eps_state : float
            Current total strain

        Returns
        -------
        (sigma_next, eps_next) : tuple[float, float]
            Next stress–strain state after ONE increment.
            If the step would violate the stress bounds, returns (None, None).
        """
        oed = Oedometer(
            e0=e0,
            c_c=c_c,
            c_s=c_s,
            sigma_prime_p=sigma_cur,  # starting stress
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            eps_delta=eps_delta,
            eps_0=eps_state,
            max_iter=1,  # just a safety; we only need one call
        )

        sigma_next, eps_next, e_s, sigma_delta = oed.step(eps_delta, sigma_cur, eps_state)

        return sigma_next, eps_next

    # --- build output skeleton ---
    out_rows: list[dict] = []
    for rid, steps in by_run_rows.items():
        for step_idx, gidx, _ in steps:
            out_rows.append({
                "run_id": rid, "step_idx": step_idx, "global_idx": gidx,
                "sigma_0_rec": None, "e_s_rec": None, "eps_total_rec": None
            })
    index_out = {(row["run_id"], row["step_idx"]): k for k, row in enumerate(out_rows)}

    def _copy_prev_into_current_row(rid, prev_step, cur_step):
        """
        Copy the previously recorded row (prev_step) into the current row (cur_step),
        duplicating sigma_0_rec, e_s_rec, and eps_total_rec to create a 'turning point'.
        """
        jprev = index_out.get((rid, prev_step))
        jcur  = index_out.get((rid, cur_step))
        if jprev is None or jcur is None:
            return
        out_rows[jcur]["sigma_0_rec"]   = out_rows[jprev]["sigma_0_rec"]
        out_rows[jcur]["e_s_rec"]       = out_rows[jprev]["e_s_rec"]
        out_rows[jcur]["eps_total_rec"] = out_rows[jprev]["eps_total_rec"]

    for rid, steps in by_run_rows.items():
        # find first TEST seed
        seed_step = None;
        seed_i = None
        for step_idx, gidx, i_row in steps:
            if gidx in test_gidx:
                seed_step, seed_i = step_idx, i_row
                break
        if seed_step is None or seed_i is None:
            continue

        r0 = runs[seed_i]
        sigma_cur = r0.get("pred_sigma_0", None)
        if sigma_cur is None:
            continue

        # params
        e0 = r0["param.e0"]
        c_s = r0["param.c_s"]
        c_c = r0["param.c_c"]
        eps_delta = r0["eps_delta"]
        sgn = sign(eps_delta)
        sigma_max = r0["param.sigma_max"]
        sigma_min = r0["param.sigma_min"]

        # eps state should start at the seed's *true* eps_total (for alignment/physics consistency)
        eps_state = r0["eps_total"]

        # --- NEW: write the seed σ₀ into the recursive dataset at seed_step ---
        # Prefer predicted σ₀ if present; fall back to true σ₀ otherwise.
        sigma_seed = r0.get("pred_sigma_0", None)
        if sigma_seed is None:
            sigma_seed = r0.get("sigma_0", None)

        jrow0 = index_out.get((rid, seed_step))
        if jrow0 is not None and sigma_seed is not None:
            # Store the seed state as the first recursive entry (no model step yet)
            out_rows[jrow0]["sigma_0_rec"] = float(sigma_seed)
            out_rows[jrow0]["e_s_rec"] = None  # No Es prediction at the seed
            out_rows[jrow0]["eps_total_rec"] = float(eps_state)

        # From here on we advance recursively from the seed state
        max_step = steps[-1][0]
        j = 1
        while True:
            abs_step = seed_step + j
            if abs_step > max_step:
                break

            # (A) Oedometer one step from current state
            sigma_phys_next, eps_total_next = oedo_one_step(
                e0, c_c, c_s, float(sigma_cur), sigma_max, sigma_min, eps_delta, float(eps_state)
            )

            # --- Direction control + duplication at bounds (one-sided) ---

            # CASE 1: Compressing and dipped below compression bound -> create a turning point.
            # We want to: (i) duplicate the *last valid* row into the current abs_step,
            # (ii) flip eps_delta to extension, (iii) recompute the physical step for the *next* row.
            if eps_delta < 0 and sigma_phys_next <= sigma_max:
                # Duplicate previous step (abs_step-1) into the current row (abs_step)
                _copy_prev_into_current_row(rid, abs_step - 1, abs_step)

                # Move to the next table row (so we now write the *first* extension point there)
                j += 1
                abs_step = seed_step + j

                # Flip direction to extension and recompute from the *same* state
                eps_delta = abs(eps_delta)  # switch to extension
                sgn = 1  # keep invert_es_to_sigma consistent with the new direction
                sigma_phys_next, eps_total_next = oedo_one_step(
                    e0, c_c, c_s, float(sigma_cur), sigma_max, sigma_min, eps_delta, float(eps_state)
                )

            # CASE 2: Extending and exceeded extension bound -> write the last valid entry once more and stop.
            # No flip back to compression (prevents ping-pong). Ensures the end-of-extension is recorded.
            if eps_delta > 0 and sigma_phys_next >= sigma_min:
                _copy_prev_into_current_row(rid, abs_step - 1, abs_step)
                break

            # --- END Direction control + duplication ---

            # (B) Model predicts Es at the *physical* next stress
            es_hat = model_predict_es(float(sigma_phys_next), float(eps_delta))

            # (C) Invert Es -> ML-next σ0 and record row
            sigma_ml_next = invert_es_to_sigma(es_hat, e0, c_s, c_c, sgn)

            key = (rid, abs_step)
            jrow = index_out.get(key, None)
            if jrow is not None:
                out_rows[jrow]["sigma_0_rec"] = sigma_ml_next
                out_rows[jrow]["e_s_rec"] = es_hat
                out_rows[jrow]["eps_total_rec"] = eps_total_next

            # (D) feedback for next iteration
            sigma_cur = sigma_ml_next
            eps_state = eps_total_next
            j += 1

    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(root_dir + out_path, index=False)
    return df_out


# ---------------------------------------------------------------------
# 4) Orchestrator (optional): run any combination of the three stages
# ---------------------------------------------------------------------
def process_pred_3stage(
        *,
        # shared
        oedo_model: int = 1,
        root_dir: str = "oedo-viewer/viewer_data/",
        runs_path: str = "runs.csv",
        split_path: str = "split_assignments.csv",
        # single predictions
        preds_path: str = "predictions_additional.csv",
        merged_path: str = "predictions_additional_processed.csv",
        # direct prop
        propagated_path: str = "predictions_additional_processed_propagated.csv",
        # recursive prop
        propagated_recursive_path: str = "predictions_additional_processed_propagated_recursive.csv",
        model: Optional[torch.nn.Module] = None,
        x_scaler=None,
        y_scaler=None,
        device: Optional[str | torch.device] = None,
        # toggles
        do_single: bool = True,
        do_direct: bool = True,
        do_recursive: bool = True,
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Convenience wrapper to run the three stages. Use the toggles to enable/disable parts.
    Returns a dict with DataFrames (or None if a stage is skipped).
    """
    out: Dict[str, Optional[pd.DataFrame]] = {"single": None, "direct": None, "recursive": None}

    if do_single:
        out["single"] = compute_single_predictions(
            oedo_model=oedo_model,
            root_dir=root_dir,
            preds_path=preds_path,
            runs_path=runs_path,
            merged_path=merged_path,
        )

    if do_direct:
        out["direct"] = propagate_direct_from_first_test(
            root_dir=root_dir,
            runs_path=runs_path,
            merged_path=merged_path,
            split_path=split_path,
            out_path=propagated_path,
        )

    if do_recursive and (model is not None):
        out["recursive"] = propagate_recursive_from_first_test(
            oedo_model=oedo_model,
            root_dir=root_dir,
            runs_path=runs_path,
            merged_path=merged_path,
            split_path=split_path,
            out_path=propagated_recursive_path,
            model=model,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            device=device,
        )

    return out


from pathlib import Path
from typing import Union, Sequence, Optional
import pandas as pd


def merge_csv_simple(
        left_csv: Union[str, Path],
        right_csv: Union[str, Path],
        *,
        right_start_col: int = 0,
        keys: Optional[Sequence[str]] = None,  # NEW: join keys; auto-detect if None
        how: str = "left",  # keep all left rows; NaN where right is missing
) -> pd.DataFrame:
    """
    Simple CSV merge that ALIGNs by keys (not by row order) and yields NaN for non-matching rows.

    - Reads both CSVs.
    - From right CSV, takes columns from `right_start_col` onward (but *always* keeps join keys).
    - Auto-detects join keys if not provided:
        1) ['run_id','step_idx'] if present in both
        2) ['global_idx'] if present in both
        3) else falls back to RangeIndex alignment (outer concat) — last resort
    - Renames overlapping non-key columns from the right with a '_r' suffix.
    - Returns merged DataFrame (does not write to disk).
    """
    left = pd.read_csv(left_csv)
    right_full = pd.read_csv(right_csv)

    if right_start_col < 0:
        raise ValueError("right_start_col must be >= 0")

    # --- Choose join keys -----------------------------------------------------
    if keys is None:
        cand1 = ["run_id", "step_idx"]
        cand2 = ["global_idx"]
        if all(c in left.columns for c in cand1) and all(c in right_full.columns for c in cand1):
            keys = cand1
        elif all(c in left.columns for c in cand2) and all(c in right_full.columns for c in cand2):
            keys = cand2
        else:
            keys = []  # no shared keys -> fallback to index-based outer concat

    if keys:
        # Ensure keys are retained even if they lie before right_start_col
        right_keep_cols = list(keys)
        # take the right-side data columns from the requested offset
        right_tail_cols = list(right_full.columns[right_start_col:])
        # avoid duplicating keys
        right_tail_cols = [c for c in right_tail_cols if c not in right_keep_cols]
        right_use = right_full[right_keep_cols + right_tail_cols].copy()

        # Disambiguate overlapping non-key columns
        overlap = [c for c in right_tail_cols if c in left.columns and c not in keys]
        if overlap:
            right_use = right_use.rename(columns={c: f"{c}_r" for c in overlap})

        merged = pd.merge(left, right_use, on=keys, how=how)

    else:
        # No common keys: last-resort outer concat on the RangeIndex
        # -> produces NaN where one side is shorter; WARNING: not semantics-safe
        left_ri = left.copy()
        right_ri = right_full.iloc[:, right_start_col:].copy()
        overlap = [c for c in right_ri.columns if c in left_ri.columns]
        if overlap:
            right_ri = right_ri.rename(columns={c: f"{c}_r" for c in overlap})
        # Outer concat to allow non-equal lengths; keep NaN where missing
        merged = pd.concat([left_ri, right_ri], axis=1, join="outer").reset_index(drop=True)

    return merged
