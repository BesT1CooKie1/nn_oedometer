import numpy as np
import pandas as pd


def _to_np(a):
    import torch
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def make_bin_edges(x, strategy="quantile-20", data_range=None):
    """
    strategy:
      - 'quantile-K'  -> K gleich besetzte Bins (K int)
      - 'fd' | 'scott'-> Freedman–Diaconis / Scott
      - 'n-K'         -> K gleich breite Bins über data_range oder x-[min,max]
      - 'width-W'     -> Breite W (float)
      - array_like    -> direkte Kanten
    """
    x = _to_np(x).ravel()
    x = x[np.isfinite(x)]
    if isinstance(strategy, (list, tuple, np.ndarray)):
        edges = np.asarray(strategy, dtype=float)
        return np.unique(edges)

    if data_range is None:
        xmin, xmax = np.nanmin(x), np.nanmax(x)
    else:
        xmin, xmax = data_range

    if isinstance(strategy, str) and strategy.startswith("quantile-"):
        K = int(strategy.split("-")[1])
        edges = np.quantile(x, np.linspace(0, 1, K + 1))
        return np.unique(edges)

    if strategy == "fd":
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        bw = 2 * iqr / (len(x) ** (1 / 3)) if iqr > 0 else (x.std() * 2 / (len(x) ** (1 / 3)))
        bw = bw if bw > 0 else (xmax - xmin) / max(1, int(np.sqrt(len(x))))
        return np.arange(xmin, xmax + bw, bw)

    if strategy == "scott":
        bw = 3.5 * x.std() / (len(x) ** (1 / 3))
        bw = bw if bw > 0 else (xmax - xmin) / max(1, int(np.sqrt(len(x))))
        return np.arange(xmin, xmax + bw, bw)

    if strategy.startswith("n-"):
        K = int(strategy.split("-")[1])
        return np.linspace(xmin, xmax, K + 1)

    if strategy.startswith("width-"):
        W = float(strategy.split("-")[1])
        return np.arange(xmin, xmax + W, W)

    raise ValueError(f"Unbekannte strategy: {strategy}")


def bin_stats_1d(x, y, bins="quantile-20", min_count=5, data_range=None, agg=("mean", "median", "std")):
    """
    Liefert DataFrame je X-Bin mit robusten Kennzahlen von Y.
    Spalten: bin_left, bin_right, x_mean, x_median, count, y_mean, y_median, y_std, y_sem, ci95
    """
    x = _to_np(x).ravel()
    y = _to_np(y).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    edges = make_bin_edges(x, bins, data_range=data_range)
    idx = np.digitize(x, edges, right=False) - 1  # 0..n-1
    nbin = len(edges) - 1

    rows = []
    for b in range(nbin):
        in_bin = idx == b
        if not np.any(in_bin):
            continue
        xb, yb = x[in_bin], y[in_bin]
        if xb.size < min_count:
            continue
        row = {
            "bin_left": edges[b],
            "bin_right": edges[b + 1],
            "count": xb.size,
            "x_mean": float(np.mean(xb)),
            "x_median": float(np.median(xb)),
        }
        if "mean" in agg:
            row["y_mean"] = float(np.mean(yb))
        if "median" in agg:
            row["y_median"] = float(np.median(yb))
        if "std" in agg:
            row["y_std"] = float(np.std(yb, ddof=1)) if xb.size > 1 else 0.0
        # immer praktisch:
        sem = (row.get("y_std", np.std(yb, ddof=1)) / np.sqrt(xb.size)) if xb.size > 1 else 0.0
        row["y_sem"] = float(sem)
        row["ci95"] = float(1.96 * sem)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values("x_mean").reset_index(drop=True)


def bin_stats_2d(x1, x2, y, bins1="quantile-15", bins2="quantile-15", min_count=5):
    """
    2D-Binning: gibt Pivot-geeigneten DataFrame mit y_mean, y_std, count zurück.
    Nutze für Heatmaps von Mittelwert/Varianz.
    """
    x1 = _to_np(x1).ravel();
    x2 = _to_np(x2).ravel();
    y = _to_np(y).ravel()
    mask = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y)
    x1, x2, y = x1[mask], x2[mask], y[mask]

    e1 = make_bin_edges(x1, bins1);
    e2 = make_bin_edges(x2, bins2)
    i1 = np.digitize(x1, e1) - 1
    i2 = np.digitize(x2, e2) - 1

    recs = []
    for a in range(len(e1) - 1):
        for b in range(len(e2) - 1):
            m = (i1 == a) & (i2 == b)
            if not np.any(m): continue
            if m.sum() < min_count: continue
            yb = y[m]
            recs.append(dict(
                x1_left=e1[a], x1_right=e1[a + 1],
                x2_left=e2[b], x2_right=e2[b + 1],
                count=int(m.sum()),
                y_mean=float(np.mean(yb)),
                y_std=float(np.std(yb, ddof=1)) if m.sum() > 1 else 0.0
            ))
    return pd.DataFrame(recs)


df = pd.read_csv("oedo-viewer/viewer_data/runs.csv")
res = bin_stats_1d(
    x=df["sigma_0"],
    y=df["eps_delta"],

    bins="quantile-20",  # gut bei nichtlinear + ungleichmäßiger Dichte
)

import matplotlib.pyplot as plt

plt.figure()
plt.errorbar(res["x_mean"], res["y_mean"], yerr=res["ci95"], fmt="o-")
plt.xlabel("sigma_0 (gebinnt, x_mean je Bin)")
plt.ylabel("eps_delta (y_mean ± 95% CI)")
plt.title("Binned Trend & Unsicherheit")
plt.grid(True, alpha=0.3)
plt.show()
