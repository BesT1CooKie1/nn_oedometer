# === oedo_train_module.py =====================================================
from __future__ import annotations
import math, json
from pathlib import Path
from typing import Dict, Optional, Tuple, Sequence, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# MODELS (examples – keep your own or extend the registry)
# ---------------------------------------------------------------------
class PiecewiseFromRaw(nn.Module):
    """
    Expects X with columns: [σ0, Δε] (order matters).
    Internally builds Δε⁺ = ReLU(Δε), Δε⁻ = ReLU(-Δε),
    then feeds [σ0, Δε⁺, Δε⁻] to an MLP to predict Es.
    """
    def __init__(self, width: int = 32, depth: int = 2, activation: str = "relu"):
        super().__init__()
        in_dim = 2
        acts = {
            "relu": nn.ReLU,
            "softplus": nn.Softplus,
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
        }
        Act = acts.get(activation.lower(), nn.ReLU)
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), Act()]
            d = width
        layers += [nn.Linear(d, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mlp(X)

MODEL_REGISTRY = {
    "piecewise_relu":  lambda **kw: PiecewiseFromRaw(activation="relu", **kw),
    "piecewise_tanh":  lambda **kw: PiecewiseFromRaw(activation="tanh", **kw),
    "piecewise_silu":  lambda **kw: PiecewiseFromRaw(activation="silu", **kw),
    "piecewise_softplus":  lambda **kw: PiecewiseFromRaw(activation="softplus", **kw),
}

# ---------------------------------------------------------------------
# SPLITTING (run-aware Test; sample-level Train/Val)
# ---------------------------------------------------------------------
def _normalize_ratios(r):
    r = np.array(r, dtype=float)
    s = r.sum()
    if s <= 0:
        raise ValueError("Ratios must sum to a positive number.")
    return r / s

def split_indices_by_ratio(
    N: int,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    *,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Legacy: pure sample-level split of indices [0..N-1] into Train/Val/Test.
    """
    r = _normalize_ratios(ratios)
    n_train = int(round(N * r[0]))
    n_val   = int(round(N * r[1]))
    n_test  = N - n_train - n_val

    idx = np.arange(N, dtype=int)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    i_train = idx[:n_train]
    i_val   = idx[n_train:n_train+n_val]
    i_test  = idx[n_train+n_val:]
    return {"train": i_train, "val": i_val, "test": i_test}

def split_by_run_for_test(
    additional_samples_csv: str | Path,
    *,
    join_key: str = "global_idx",
    run_col: str = "run_id",
    ratios_runs: Tuple[float, float, float] = (0.7, 0.15, 0.15),  # on RUNS
    ratios_trainval_samples: Tuple[float, float] = (0.8, 0.2),    # within Train+Val samples
    seed: Optional[int] = 42,
    shuffle_runs: bool = True,
    shuffle_samples: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Run-aware Test split:
      1) Split UNIQUE runs into Train/Val/Test by ratios_runs.
      2) All rows from Test runs -> Test (run-level unseen).
      3) Rows from remaining (Train+Val runs) are split by ratios_trainval_samples (sample-level) into Train/Val.

    Returns dict of index arrays (positions in additional_samples_csv), suitable to slice X/y (which match that CSV).
    """
    df = pd.read_csv(additional_samples_csv)
    if run_col not in df.columns:
        raise KeyError(f"Column '{run_col}' not found in {additional_samples_csv}. Needed for run-aware splitting.")

    N = len(df)
    if join_key not in df.columns:
        # Ensure we can still emit split_assignments aligned with file order
        df = df.copy()
        df.insert(0, join_key, np.arange(N, dtype=int))

    # 1) run-level split for Test
    runs = df[run_col].to_numpy()
    unique_runs = np.unique(runs)
    rng = np.random.default_rng(seed)
    if shuffle_runs:
        rng.shuffle(unique_runs)

    r_run = _normalize_ratios(ratios_runs)
    n_train_runs = int(round(len(unique_runs) * r_run[0]))
    n_val_runs   = int(round(len(unique_runs) * r_run[1]))
    n_test_runs  = len(unique_runs) - n_train_runs - n_val_runs

    train_runs = set(unique_runs[:n_train_runs])
    val_runs   = set(unique_runs[n_train_runs:n_train_runs + n_val_runs])
    test_runs  = set(unique_runs[n_train_runs + n_val_runs:])

    # Indices by membership
    idx_all = np.arange(N, dtype=int)
    idx_test_all = idx_all[np.isin(runs, list(test_runs))]
    idx_trainval_all = idx_all[~np.isin(runs, list(test_runs))]  # all rows from Train+Val runs

    # 2) Within Train+Val rows, do simple sample-level split into Train/Val
    idx_tv = idx_trainval_all.copy()
    if shuffle_samples:
        rng.shuffle(idx_tv)

    r_tv = _normalize_ratios(ratios_trainval_samples)
    n_tr = int(round(len(idx_tv) * r_tv[0]))
    idx_train = idx_tv[:n_tr]
    idx_val   = idx_tv[n_tr:]

    return {"train": idx_train, "val": idx_val, "test": idx_test_all}

def save_split_assignments(
    df_additional_samples_csv: str | Path,
    join_key: str,
    splits: Dict[str, np.ndarray],
    out_csv: str | Path = "split_assignments.csv"
) -> pd.DataFrame:
    """
    Writes mapping join_key -> split. Assumes df_additional_samples_csv order == X/y order.
    """
    df = pd.read_csv(df_additional_samples_csv)
    N = len(df)
    if join_key not in df.columns:
        # create explicit join_key from row order if missing
        df = df.copy()
        df.insert(0, join_key, np.arange(N, dtype=int))

    mapping = np.full(N, "", dtype=object)
    for name, idx in splits.items():
        mapping[idx] = name

    out = df[[join_key]].copy()
    out["split"] = mapping
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


# ---------------------------------------------------------------------
# METRICS / EVAL
# ---------------------------------------------------------------------
def _mse(yhat, y):  return torch.mean((yhat - y) ** 2)
def _rmse(yhat, y): return torch.sqrt(_mse(yhat, y) + 1e-12)
def _mae(yhat, y):  return torch.mean(torch.abs(yhat - y))
def _r2(yhat, y):
    y_mean = torch.mean(y)
    ss_res = torch.sum((yhat - y) ** 2)
    ss_tot = torch.sum((y - y_mean) ** 2) + 1e-12
    return 1.0 - (ss_res / ss_tot)

@torch.no_grad()
def eval_on_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    ms, rs, ma, r2, n = 0.0, 0.0, 0.0, 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        yhat = model(xb)
        b = xb.size(0)
        ms += _mse(yhat, yb).item() * b
        rs += _rmse(yhat, yb).item() * b
        ma += _mae(yhat, yb).item() * b
        r2 += _r2(yhat, yb).item() * b
        n  += b
    if n == 0:
        return {"mse": math.nan, "rmse": math.nan, "mae": math.nan, "r2": math.nan}
    return {"mse": ms/n, "rmse": rs/n, "mae": ma/n, "r2": r2/n}

# ---------------------------------------------------------------------
# TRAINING (records per-epoch stats + CSV + optional plots)
# ---------------------------------------------------------------------
def train_with_xy_stats(
    model: nn.Module,
    X_train: torch.Tensor, y_train: torch.Tensor,
    X_val: Optional[torch.Tensor] = None, y_val: Optional[torch.Tensor] = None,
    *,
    epochs: int = 300,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[str | torch.device] = None,
    verbose_every: int = 25,
    history_csv: Optional[str | Path] = "training_history.csv",
    make_plot: bool = True,
):
    """
    Logs per-epoch: train/val MSE, RMSE, MAE, R². Saves history to CSV if path given.
    Restores best weights by validation MSE (if val given).
    """
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = model.to(device)

    # shapes to [N,D] and [N,1]
    if X_train.ndim == 1: X_train = X_train.view(-1, 1)
    if y_train.ndim == 1: y_train = y_train.view(-1, 1)

    train_ds = TensorDataset(X_train.float(), y_train.float())
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_dl = None
    if X_val is not None and y_val is not None:
        if X_val.ndim == 1: X_val = X_val.view(-1, 1)
        if y_val.ndim == 1: y_val = y_val.view(-1, 1)
        val_ds = TensorDataset(X_val.float(), y_val.float())
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    history = []
    best_state, best_val = None, float('inf')

    for ep in range(1, epochs+1):
        model.train()
        tr_loss_sum, n_sum = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss_sum += loss.item() * xb.size(0); n_sum += xb.size(0)
        # epoch end: compute full metrics on loaders
        tr_stats = eval_on_loader(model, DataLoader(train_ds, batch_size=batch_size), device)
        if val_dl is not None:
            va_stats = eval_on_loader(model, val_dl, device)
            if va_stats["mse"] < best_val:
                best_val = va_stats["mse"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            va_stats = {"mse": math.nan, "rmse": math.nan, "mae": math.nan, "r2": math.nan}

        row = {
            "epoch": ep,
            "train_mse": tr_stats["mse"], "train_rmse": tr_stats["rmse"],
            "train_mae": tr_stats["mae"], "train_r2": tr_stats["r2"],
            "val_mse": va_stats["mse"],   "val_rmse": va_stats["rmse"],
            "val_mae": va_stats["mae"],   "val_r2": va_stats["r2"],
        }
        history.append(row)

        if verbose_every and ep % verbose_every == 0:
            print(f"ep {ep:4d} | "
                  f"train MSE {row['train_mse']:.4e}  RMSE {row['train_rmse']:.4e}  MAE {row['train_mae']:.4e}  R² {row['train_r2']:.4f} | "
                  f"val MSE {row['val_mse']:.4e}  RMSE {row['val_rmse']:.4e}  MAE {row['val_mae']:.4e}  R² {row['val_r2']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    history_df = pd.DataFrame(history)
    if history_csv:
        Path(history_csv).parent.mkdir(parents=True, exist_ok=True)
        history_df.to_csv(history_csv, index=False)

    if make_plot:
        fig, ax = plt.subplots(1, 2, figsize=(10,4))
        ax[0].plot(history_df["epoch"], history_df["train_mse"], label="train MSE")
        if not history_df["val_mse"].isna().all():
            ax[0].plot(history_df["epoch"], history_df["val_mse"], label="val MSE")
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("MSE"); ax[0].set_title("Loss"); ax[0].legend()

        ax[1].plot(history_df["epoch"], history_df["train_r2"], label="train R²")
        if not history_df["val_r2"].isna().all():
            ax[1].plot(history_df["epoch"], history_df["val_r2"], label="val R²")
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel("R²"); ax[1].set_ylim(-0.1,1.02); ax[1].set_title("Goodness of fit"); ax[1].legend()
        plt.tight_layout(); plt.show()

    return model, history_df

# ---------------------------------------------------------------------
# HIGH-LEVEL ORCHESTRATION
# ---------------------------------------------------------------------
def train_eval_save(
    *,
    X: torch.Tensor, y: torch.Tensor,
    feature_names: Sequence[str],
    additional_samples_csv: str | Path,  # produced by compose_dataset_from_files (must contain run_id)
    join_key: str = "global_idx",
    model_name: str = "piecewise_relu",
    model_kwargs: Optional[Dict[str, Any]] = None,
    # splitting
    split_mode: str = "run_test",                     # "run_test" (default) or "sample"
    split_ratios: Tuple[float,float,float] = (0.7, 0.15, 0.15),   # used as runs-ratios for run_test, or sample-level for sample
    split_seed: Optional[int] = 42,
    shuffle: bool = True,
    # training
    epochs: int = 300,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[str | torch.device] = None,
    out_dir: str | Path = "oedo_outputs",
):
    """
    End-to-end: split -> train -> save history CSV -> save split CSV -> test metrics CSV.

    split_mode:
      - "run_test": Test is composed of whole unseen runs; Train/Val drawn from remaining runs (sample-level).
      - "sample":   Legacy pure sample-level split by split_ratios.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # --- split
    if split_mode == "run_test":
        splits = split_by_run_for_test(
            additional_samples_csv,
            join_key=join_key,
            run_col="run_id",
            ratios_runs=split_ratios,
            ratios_trainval_samples=(0.8, 0.2),  # can make this a parameter if you want
            seed=split_seed,
            shuffle_runs=shuffle,
            shuffle_samples=shuffle,
        )
    elif split_mode == "sample":
        N = len(X)
        splits = split_indices_by_ratio(N, ratios=split_ratios, shuffle=shuffle, seed=split_seed)
    else:
        raise ValueError("split_mode must be 'run_test' or 'sample'.")

    save_split_assignments(
        additional_samples_csv,
        join_key,
        splits,
        out_csv=Path(out_dir) / "split_assignments.csv"
    )

    # --- tensors per split
    Xtr, ytr = X[splits["train"]], y[splits["train"]]
    Xva, yva = X[splits["val"]],   y[splits["val"]]
    Xte, yte = X[splits["test"]],  y[splits["test"]]

    # --- model
    ModelCtor = MODEL_REGISTRY[model_name]
    model = ModelCtor(**(model_kwargs or {}))

    # --- train + history CSV
    model, history_df = train_with_xy_stats(
        model,
        Xtr, ytr,
        X_val=Xva, y_val=yva,
        epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        device=device,
        history_csv=Path(out_dir) / "training_history.csv",
        make_plot=True,
    )

    # --- final test metrics CSV
    te_loader = DataLoader(TensorDataset(Xte.float(), yte.float()), batch_size=batch_size, shuffle=False)
    metrics = eval_on_loader(model, te_loader, torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu')))
    test_metrics_df = pd.DataFrame([metrics])
    test_metrics_df.to_csv(Path(out_dir) / "test_metrics.csv", index=False)

    return model, {"train": (Xtr,ytr), "val": (Xva,yva), "test": (Xte,yte)}, history_df, test_metrics_df