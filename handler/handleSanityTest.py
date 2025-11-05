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

def sanity_test_gen_data(
    oedo_model=1,
    device=None
):
    OedometerClass = _load_oedometer_class(oedo_model)
    runs_list = []
    param_spec = {
        "e0": 1.0,
        "c_c": 0.005,
        "c_s": 0.002,
        "sigma_max": -1000,
        "sigma_min": -100,
        "eps_delta": -0.0005,
        "eps_0": 0,
    }
    runs = {
        "global_idx" : 0,
        "run_id": 0,
        "step_idx": 0,
        "sigma_0": None,
        "eps_total": 0,
        "eps_delta": 0,
        "e_s": None,
        "sigma_delta": None,
        "phase": None,
        # "cut",
        "param.c_c": param_spec["c_c"],
        "param.c_s": param_spec["c_s"],
        "param.e0": param_spec["e0"],
        "param.eps_0": 0,
        "param.eps_delta": None,
        "param.sigma_max": param_spec["sigma_max"],
        "param.sigma_min": param_spec["sigma_min"],
        "param.sigma_prime_p": None
    }
    e_s_list = []
    sigma_0_list = []
    eps_delta_list = []
    for _ in range(2):
        param_spec["eps_delta"] = param_spec["eps_delta"] * -1
        runs["phase"] = "extension" if param_spec["eps_delta"] > 0 else "compression"
        runs["eps_delta"] = float(param_spec["eps_delta"])
        for i in np.arange(1, 1200.1, 0.1):
            param_spec["sigma_prime_p"] = i
            oedo = OedometerClass(**param_spec)
            sigma_1, eps_1, e_s, sigma_delta = oedo.step(eps_delta=param_spec["eps_delta"],sigma_0=i, eps_0=0)
            runs["e_s"] = e_s
            runs["sigma_0"] = i
            runs["param.sigma_prime_p"] = i
            runs["sigma_delta"] = sigma_delta
            runs_list.append(dict(runs))
            runs["global_idx"] += 1
            e_s_list.append(e_s)
            sigma_0_list.append(i)
            eps_delta_list.append(param_spec["eps_delta"])
    pd.DataFrame(runs_list).to_csv('../oedo-viewer/viewer_data/runs.csv', index=False)
    pd.DataFrame(runs_list).to_csv('../oedo-viewer/viewer_data/samples.csv', index=False)
    pd.DataFrame(runs_list).to_csv('../oedo-viewer/viewer_data/runs_final.csv', index=False)
    pd.DataFrame(runs_list).to_csv('../oedo-viewer/viewer_data/additional_runs.csv', index=False)
    pd.DataFrame(runs_list).to_csv('../oedo-viewer/viewer_data/additional_samples.csv', index=False)
    split = []
    for idx in range(len(runs_list)):
        split.append({"global_idx" : idx, "split" : "train"})
        split.append({"global_idx": idx, "split": "test"})
    pd.DataFrame(split).to_csv('../oedo-viewer/viewer_data/split_assignments.csv', index=False)
    X = pd.DataFrame({"sigma_0": sigma_0_list, "eps_delta": eps_delta_list}).values
    Y = pd.DataFrame({"e_s" : e_s_list}).values
    tX = torch.from_numpy(X)
    tY = torch.from_numpy(Y)
    return tX, tY

from pathlib import Path
import torch
import pandas as pd

# Reuse your model + trainer utilities
from nn_model.model import PiecewiseFromRaw, train_with_xy_stats, eval_on_loader  # uses PiecewiseFromRaw + helpers
from torch.utils.data import TensorDataset, DataLoader

def epochs_for_steps(n_samples: int, batch_size: int, target_steps: int) -> int:
    """
    Utility to compute the number of epochs required to reach
    a given number of optimizer steps.

    Parameters
    ----------
    n_samples : int
        Total number of training samples.
    batch_size : int
        Number of samples per training batch.
    target_steps : int
        Desired total number of optimizer steps.

    Returns
    -------
    int
        Number of epochs required to reach (or exceed) the target steps.
    """
    import math
    steps_per_epoch = math.ceil(n_samples / batch_size)
    return math.ceil(target_steps / max(1, steps_per_epoch))


def run_sanity_train(
    *,
    oedo_model: int = 1,
    out_dir: str = "../oedo-viewer/viewer_data/",
    model_width: int = 32,
    model_depth: int = 2,
    activation: str = "relu",
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    target_steps: int = 35000,   # target optimizer steps for training
    device: str | torch.device | None = None,
):
    """
    Run a sanity test training on the full oedometer dataset.

    - Generates deterministic tensor data from FunctionSanityTestGenData.
    - Normalizes X and y (mean/std) before training.
    - Trains PiecewiseFromRaw on 100 % of the samples (no split).
    - Evaluates on the same data.
    - Saves training history, test metrics, predictions, and normalization stats.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 1) Generate raw tensors
    tX_raw, tY_raw = sanity_test_gen_data(oedo_model=oedo_model, device=device)
    tX_raw = tX_raw.float().cpu().numpy()
    tY_raw = tY_raw.float().cpu().numpy().reshape(-1, 1)

    # 2) Compute normalization statistics
    X_mean, X_std = tX_raw.mean(axis=0), tX_raw.std(axis=0)
    y_mean, y_std = tY_raw.mean(axis=0), tY_raw.std(axis=0)
    norm_stats = pd.DataFrame({
        "variable": ["sigma_0", "eps_delta", "Es"],
        "mean": [X_mean[0], X_mean[1], y_mean[0]],
        "std": [X_std[0], X_std[1], y_std[0]],
    })
    norm_stats.to_csv(Path(out_dir) / "normalization_stats.csv", index=False)

    # 3) Normalize data
    tX = torch.tensor((tX_raw - X_mean) / X_std, dtype=torch.float32, device=device)
    tY = torch.tensor((tY_raw - y_mean) / y_std, dtype=torch.float32, device=device)

    # 4) Determine epochs automatically from step budget
    epochs = epochs_for_steps(n_samples=len(tX), batch_size=batch_size, target_steps=target_steps)
    print(f"Training for approximately {epochs} epochs to reach ~{target_steps} steps.")

    # 5) Build model
    model = PiecewiseFromRaw(width=model_width, depth=model_depth, activation=activation).to(device)

    # 6) Train on all data (no validation split)
    history_csv = Path(out_dir) / "training_history.csv"
    model, history_df = train_with_xy_stats(
        model,
        X_train=tX, y_train=tY,
        X_val=None, y_val=None,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        history_csv=history_csv,
        make_plot=False,
    )

    # 7) Evaluate on same dataset
    full_loader = DataLoader(TensorDataset(tX.cpu(), tY.cpu()), batch_size=batch_size, shuffle=False)
    metrics = eval_on_loader(model, full_loader, device)
    pd.DataFrame([metrics]).to_csv(Path(out_dir) / "test_metrics.csv", index=False)

    # 8) Generate and inverse-transform predictions
    with torch.no_grad():
        y_hat_norm = model(tX).detach().cpu().numpy().reshape(-1)
    y_hat = y_hat_norm * y_std + y_mean  # inverse transform
    Y_true = tY.cpu().numpy().reshape(-1) * y_std + y_mean

    # 9) Save per-sample predictions
    df_pred = pd.DataFrame({
        "global_idx": range(len(Y_true)),
        "sigma_0": tX_raw[:, 0],
        "eps_delta": tX_raw[:, 1],
        "true_e_s": Y_true,
        "pred_e_s": y_hat,
    })
    df_pred.to_csv(Path(out_dir) / "predictions_additional.csv", index=False)

    return model, history_df, pd.DataFrame([metrics]), df_pred


if __name__ == "__main__":
    run_sanity_train(
        oedo_model=1,
        out_dir="../oedo-viewer/viewer_data/",
        model_width=64,
        model_depth=2,
        activation="silu",
        batch_size=512,
        lr=1e-3,
        weight_decay=1e-5,
        target_steps=35000,
    )
