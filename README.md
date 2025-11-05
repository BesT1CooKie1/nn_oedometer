# Vorhersage des Ödometer-Versuchs mit einem neuronalen Netzwerk

**Ziel:**
Entwicklung eines neuronalen Netzes, das auf Basis gegebener Input-Parameter den Elastizitätsmodul $E_s$ im Ödometer-Versuch vorhersagt.

---

## 1. Problemformulierung

Es wird folgende Beziehung zugrunde gelegt:

$$
\dot{\sigma} = C_1\,\sigma_t\,\dot{\varepsilon} + C_2\,\sigma_t\,\left|\dot{\varepsilon}\right|
$$

Diese Gleichung beschreibt die Änderung der Spannung $\dot{\sigma}$ in Abhängigkeit von der aktuellen Spannung $\sigma_t$, der Dehnungsrate $\dot{\varepsilon}$ und den Koeffizienten $C_1, C_2$, die aus dem gewählten Modell abgeleitet sind.

---

## 2. Annahmen / Startwerte

Die Berechnung basiert auf folgenden festen Parametern:

- **Startspannung:** $\sigma_0 = -1{,}00\,\text{kPa}$
- **Porenverhältnis:** $e_0 = 1{,}00$
- **Koeffizienten:**
  - $C_c = 0{,}005$
  - $C_s = 0{,}002$
- **Dehnungsraten:**
  - Stauchungsphase: $\dot{\varepsilon}_c = -0{,}0005$
  - Dehnungsphase: $\dot{\varepsilon}_e = +0{,}0005$

---

## 3. Trainingssetup

- **Input:**
  $\sigma_t$, $\dot{\varepsilon}$
- **Output:**
  Elastizitätsmodul $E_s$

Das neuronale Netz soll aus den aktuellen Zustandsgrößen ($\sigma_t$, $\dot{\varepsilon}$) lernen, in welcher Phase (Kompression vs. Entlastung) sich der Versuch befindet, und darauf basierend $E_s$ schätzen.

---

## 4. Variablendeklaration

| Symbol              | Variable im Code     | Bedeutung |
|---------------------|----------------------|-----------|
| $\sigma_t$          | `sigma_t`            | Aktuelle Spannung zum Zeitpunkt $t$ |
| $\dot{\varepsilon}$ | `delta_epsilon`      | Dehnungsrate; negative Werte: Kompression, positive: Entlastung |
| $\dot\sigma_t$      | `delta_sigma`        | Inkrementelle Änderung der Spannung |
| $E_s$               | `e_s`                | Elastizitätsmodul (Zielgröße) |
| $e_0$               | `e_0`                | Porenverhältniszahl |

---

## 5. Hinweise zur Phase

Die Phase (Stauchung vs. Dehnung) lässt sich über das Vorzeichen von $\dot{\varepsilon}$ ablesen. Alternativ kann explizit ein Zustandsindikator (z. B. one-hot oder diskrete Labels für Belastung/Entlastung) zusätzlich als Feature mitgegeben werden, um dem Modell das Unterscheiden zu erleichtern.


# Data generation


```python
from handler.handleMetaData import *

oedo_model = 1

param_spec = {
    "e0": 1.0,
    "c_c": 0.005,
    "c_s": 0.002,
    "sigma_prime_p": [-.8, -1.5],
    "sigma_max": [-900, -1100],
    "sigma_min": [-90, -110],
    "eps_delta": -0.0005,
    "eps_0": 0,
}

para_data = {
    "n_runs": 30,
    "final_samples": 500,
    "features_keys": ("sigma_0","eps_delta"),
    "target_keys": ("e_s"),
    "seed": 8,
}

tX_raw, tY_raw, info = generate_oedometer_dataset(param_spec,
                                                  n_runs=para_data["n_runs"],
                                                  final_samples=para_data["final_samples"],
                                                  oedo_model=oedo_model,
                                                  seed=para_data["seed"],
                                                  feature_keys=para_data["features_keys"],
                                                  target_key=para_data["target_keys"]
                                                  )

export_dataset_for_html(info, out_dir="oedo-viewer/viewer_data/")

schema = export_oedometer_schema(
    feature_keys=para_data["features_keys"],
    target_key=para_data["target_keys"],
    origin="generate_oedometer_dataset",
    n_runs=para_data["n_runs"],
    final_samples=para_data["final_samples"],
    seed=para_data["seed"],
    path="oedo-viewer/viewer_data/schema.json",
)
```

    [OK] exportiert: oedo-viewer/viewer_data/samples.csv  /  oedo-viewer/viewer_data/runs.csv



```python
tX, tY, meta = compose_dataset_from_files(
    samples_csv="oedo-viewer/viewer_data/samples.csv",
    additional_runs_csv="oedo-viewer/viewer_data/additional_runs.csv",  # oder runs.csv
    schema_json="oedo-viewer/viewer_data/schema.json",
    join_key="global_idx",
    join_how="left",
    # WICHTIG: runs/additional_runs haben keinen Key -> aus Zeilennummer bauen
    additional_index_from_row=True,
    additional_index_start=0,  # stell auf 1, wenn deine Samples 1-basiert zählen
    # samples hat i.d.R. die Spalte; falls nicht:
    samples_index_from_row=False,
    samples_index_start=0,
    # Auswahl (optional)
    include_additional_features=None,  # None => alle F:
    include_additional_targets=None,  # None => alle T:
    # Datenqualität
    dropna=True,
    # Normalisierung
    normalize=True,
    scaler_path="scalers.joblib",
    refit_scaler=True,
    # Export
    additional_samples_out="oedo-viewer/viewer_data/additional_samples.csv",
)
print(tX.shape)
print(tY.shape)
```

    torch.Size([500, 2])
    torch.Size([500, 1])


# Training


```python
from nn_model.model import train_eval_save
model, splits, history_df, test_metrics_df = train_eval_save(
    X=tX, y=tY,
    feature_names=meta["feature_names"],
    additional_samples_csv=meta["additional_samples_path"],
    join_key=meta["join_key"],
    model_name="piecewise_relu",                  # swap to "piecewise_tanh", etc.
    model_kwargs=dict(width=32, depth=2),
    split_ratios=(0.70, 0.15, 0.15),              # set your own ratios
    split_seed=123,
    epochs=3000,
    out_dir="oedo-viewer/viewer_data/"                        # writes 3 CSVs here
)

```

    ep   25 | train MSE 2.1753e-01  RMSE 4.6136e-01  MAE 2.9196e-01  R² 0.7685 | val MSE 2.2995e-01  RMSE 4.7953e-01  MAE 3.0238e-01  R² 0.7733
    ep   50 | train MSE 9.1799e-03  RMSE 9.5649e-02  MAE 7.1440e-02  R² 0.9898 | val MSE 9.6867e-03  RMSE 9.8421e-02  MAE 6.7070e-02  R² 0.9905
    ep   75 | train MSE 1.3972e-03  RMSE 3.7035e-02  MAE 2.4730e-02  R² 0.9985 | val MSE 1.7814e-03  RMSE 4.2207e-02  MAE 2.6370e-02  R² 0.9982
    ep  100 | train MSE 5.0854e-04  RMSE 2.2294e-02  MAE 1.2851e-02  R² 0.9995 | val MSE 6.7895e-04  RMSE 2.6057e-02  MAE 1.3504e-02  R² 0.9993
    ep  125 | train MSE 2.5708e-04  RMSE 1.5867e-02  MAE 9.6899e-03  R² 0.9997 | val MSE 3.5778e-04  RMSE 1.8915e-02  MAE 9.9314e-03  R² 0.9996
    ep  150 | train MSE 1.0581e-04  RMSE 1.0242e-02  MAE 6.4814e-03  R² 0.9999 | val MSE 1.6090e-04  RMSE 1.2685e-02  MAE 6.3288e-03  R² 0.9998
    ep  175 | train MSE 5.9364e-05  RMSE 7.6787e-03  MAE 5.3077e-03  R² 0.9999 | val MSE 1.1543e-04  RMSE 1.0744e-02  MAE 5.1165e-03  R² 0.9999
    ep  200 | train MSE 3.9410e-05  RMSE 6.2552e-03  MAE 4.3876e-03  R² 1.0000 | val MSE 8.8657e-05  RMSE 9.4158e-03  MAE 4.3150e-03  R² 0.9999
    ep  225 | train MSE 2.8705e-05  RMSE 5.3418e-03  MAE 3.8337e-03  R² 1.0000 | val MSE 7.6088e-05  RMSE 8.7229e-03  MAE 3.8981e-03  R² 0.9999
    ep  250 | train MSE 2.0432e-05  RMSE 4.5080e-03  MAE 3.2992e-03  R² 1.0000 | val MSE 6.6968e-05  RMSE 8.1834e-03  MAE 3.4737e-03  R² 0.9999
    ep  275 | train MSE 1.6837e-05  RMSE 4.0910e-03  MAE 3.1072e-03  R² 1.0000 | val MSE 6.1383e-05  RMSE 7.8347e-03  MAE 3.4057e-03  R² 0.9999
    ep  300 | train MSE 1.5462e-05  RMSE 3.9257e-03  MAE 2.9390e-03  R² 1.0000 | val MSE 6.1029e-05  RMSE 7.8121e-03  MAE 3.3874e-03  R² 0.9999
    ep  325 | train MSE 1.2788e-05  RMSE 3.5710e-03  MAE 2.5323e-03  R² 1.0000 | val MSE 5.1415e-05  RMSE 7.1704e-03  MAE 2.8501e-03  R² 0.9999
    ep  350 | train MSE 1.1768e-05  RMSE 3.4243e-03  MAE 2.4591e-03  R² 1.0000 | val MSE 5.2806e-05  RMSE 7.2668e-03  MAE 2.8650e-03  R² 0.9999
    ep  375 | train MSE 1.1107e-05  RMSE 3.3289e-03  MAE 2.2680e-03  R² 1.0000 | val MSE 5.1999e-05  RMSE 7.2110e-03  MAE 2.6015e-03  R² 0.9999
    ep  400 | train MSE 1.0439e-05  RMSE 3.2263e-03  MAE 2.2794e-03  R² 1.0000 | val MSE 5.2744e-05  RMSE 7.2625e-03  MAE 2.7406e-03  R² 0.9999
    ep  425 | train MSE 9.9672e-06  RMSE 3.1521e-03  MAE 2.1991e-03  R² 1.0000 | val MSE 4.7642e-05  RMSE 6.9023e-03  MAE 2.5886e-03  R² 1.0000
    ep  450 | train MSE 9.7480e-06  RMSE 3.1175e-03  MAE 2.2224e-03  R² 1.0000 | val MSE 4.7873e-05  RMSE 6.9190e-03  MAE 2.6618e-03  R² 1.0000
    ep  475 | train MSE 9.1628e-06  RMSE 3.0198e-03  MAE 2.1497e-03  R² 1.0000 | val MSE 4.9476e-05  RMSE 7.0339e-03  MAE 2.5526e-03  R² 1.0000
    ep  500 | train MSE 8.7344e-06  RMSE 2.9515e-03  MAE 1.9959e-03  R² 1.0000 | val MSE 4.9296e-05  RMSE 7.0211e-03  MAE 2.3668e-03  R² 1.0000
    ep  525 | train MSE 9.4776e-06  RMSE 3.0700e-03  MAE 2.2392e-03  R² 1.0000 | val MSE 5.0142e-05  RMSE 7.0811e-03  MAE 2.7031e-03  R² 1.0000
    ep  550 | train MSE 9.4862e-06  RMSE 3.0783e-03  MAE 2.1937e-03  R² 1.0000 | val MSE 4.9940e-05  RMSE 7.0668e-03  MAE 2.6324e-03  R² 1.0000
    ep  575 | train MSE 8.2218e-06  RMSE 2.8546e-03  MAE 1.8983e-03  R² 1.0000 | val MSE 4.9437e-05  RMSE 7.0312e-03  MAE 2.2526e-03  R² 1.0000
    ep  600 | train MSE 7.8377e-06  RMSE 2.7928e-03  MAE 1.9931e-03  R² 1.0000 | val MSE 4.9479e-05  RMSE 7.0341e-03  MAE 2.4065e-03  R² 1.0000
    ep  625 | train MSE 8.0459e-06  RMSE 2.8221e-03  MAE 1.9642e-03  R² 1.0000 | val MSE 4.9785e-05  RMSE 7.0559e-03  MAE 2.3741e-03  R² 1.0000
    ep  650 | train MSE 7.3841e-06  RMSE 2.7066e-03  MAE 1.8452e-03  R² 1.0000 | val MSE 4.6471e-05  RMSE 6.8169e-03  MAE 2.2357e-03  R² 1.0000
    ep  675 | train MSE 7.6063e-06  RMSE 2.7432e-03  MAE 1.8562e-03  R² 1.0000 | val MSE 4.8635e-05  RMSE 6.9739e-03  MAE 2.2784e-03  R² 1.0000
    ep  700 | train MSE 6.8652e-06  RMSE 2.6182e-03  MAE 1.7976e-03  R² 1.0000 | val MSE 4.1670e-05  RMSE 6.4552e-03  MAE 2.0942e-03  R² 1.0000
    ep  725 | train MSE 6.7495e-06  RMSE 2.5904e-03  MAE 1.7891e-03  R² 1.0000 | val MSE 4.5834e-05  RMSE 6.7701e-03  MAE 2.2248e-03  R² 1.0000
    ep  750 | train MSE 6.5773e-06  RMSE 2.5532e-03  MAE 1.7379e-03  R² 1.0000 | val MSE 5.1016e-05  RMSE 7.1425e-03  MAE 2.2214e-03  R² 0.9999
    ep  775 | train MSE 6.5909e-06  RMSE 2.5659e-03  MAE 1.7939e-03  R² 1.0000 | val MSE 4.8123e-05  RMSE 6.9371e-03  MAE 2.1501e-03  R² 1.0000
    ep  800 | train MSE 6.1896e-06  RMSE 2.4857e-03  MAE 1.7560e-03  R² 1.0000 | val MSE 4.8831e-05  RMSE 6.9879e-03  MAE 2.1466e-03  R² 1.0000
    ep  825 | train MSE 5.7319e-06  RMSE 2.3875e-03  MAE 1.6394e-03  R² 1.0000 | val MSE 4.7505e-05  RMSE 6.8924e-03  MAE 2.0794e-03  R² 1.0000
    ep  850 | train MSE 5.7025e-06  RMSE 2.3855e-03  MAE 1.6915e-03  R² 1.0000 | val MSE 4.6450e-05  RMSE 6.8154e-03  MAE 2.1657e-03  R² 1.0000
    ep  875 | train MSE 5.5807e-06  RMSE 2.3602e-03  MAE 1.6423e-03  R² 1.0000 | val MSE 4.8167e-05  RMSE 6.9403e-03  MAE 2.0458e-03  R² 1.0000
    ep  900 | train MSE 6.0193e-06  RMSE 2.4534e-03  MAE 1.6935e-03  R² 1.0000 | val MSE 4.2846e-05  RMSE 6.5457e-03  MAE 2.0831e-03  R² 1.0000
    ep  925 | train MSE 5.5294e-06  RMSE 2.3501e-03  MAE 1.6082e-03  R² 1.0000 | val MSE 4.8577e-05  RMSE 6.9697e-03  MAE 2.0326e-03  R² 1.0000
    ep  950 | train MSE 5.5806e-06  RMSE 2.3622e-03  MAE 1.6067e-03  R² 1.0000 | val MSE 4.6601e-05  RMSE 6.8265e-03  MAE 1.9820e-03  R² 1.0000
    ep  975 | train MSE 5.4796e-06  RMSE 2.3403e-03  MAE 1.6186e-03  R² 1.0000 | val MSE 4.4159e-05  RMSE 6.6452e-03  MAE 2.0563e-03  R² 1.0000
    ep 1000 | train MSE 4.9793e-06  RMSE 2.2312e-03  MAE 1.4939e-03  R² 1.0000 | val MSE 4.6147e-05  RMSE 6.7931e-03  MAE 1.9442e-03  R² 1.0000
    ep 1025 | train MSE 4.7552e-06  RMSE 2.1767e-03  MAE 1.5244e-03  R² 1.0000 | val MSE 4.9630e-05  RMSE 7.0449e-03  MAE 2.0287e-03  R² 1.0000
    ep 1050 | train MSE 4.9113e-06  RMSE 2.2051e-03  MAE 1.4894e-03  R² 1.0000 | val MSE 5.3690e-05  RMSE 7.3273e-03  MAE 1.9687e-03  R² 0.9999
    ep 1075 | train MSE 5.0119e-06  RMSE 2.2386e-03  MAE 1.5131e-03  R² 1.0000 | val MSE 4.8300e-05  RMSE 6.9498e-03  MAE 1.9040e-03  R² 1.0000
    ep 1100 | train MSE 4.4702e-06  RMSE 2.1117e-03  MAE 1.4050e-03  R² 1.0000 | val MSE 4.7359e-05  RMSE 6.8818e-03  MAE 1.8514e-03  R² 1.0000
    ep 1125 | train MSE 5.9653e-06  RMSE 2.4410e-03  MAE 1.7438e-03  R² 1.0000 | val MSE 5.1253e-05  RMSE 7.1591e-03  MAE 2.1531e-03  R² 0.9999
    ep 1150 | train MSE 4.4099e-06  RMSE 2.0970e-03  MAE 1.4074e-03  R² 1.0000 | val MSE 5.0811e-05  RMSE 7.1282e-03  MAE 1.8535e-03  R² 0.9999
    ep 1175 | train MSE 4.1021e-06  RMSE 2.0206e-03  MAE 1.4050e-03  R² 1.0000 | val MSE 5.0865e-05  RMSE 7.1320e-03  MAE 1.8928e-03  R² 0.9999
    ep 1200 | train MSE 3.9772e-06  RMSE 1.9878e-03  MAE 1.3648e-03  R² 1.0000 | val MSE 4.9114e-05  RMSE 7.0081e-03  MAE 1.8436e-03  R² 1.0000
    ep 1225 | train MSE 4.7922e-06  RMSE 2.1857e-03  MAE 1.5024e-03  R² 1.0000 | val MSE 5.0067e-05  RMSE 7.0758e-03  MAE 1.8604e-03  R² 1.0000
    ep 1250 | train MSE 4.1712e-06  RMSE 2.0359e-03  MAE 1.3776e-03  R² 1.0000 | val MSE 5.3581e-05  RMSE 7.3199e-03  MAE 1.8603e-03  R² 0.9999
    ep 1275 | train MSE 4.5406e-06  RMSE 2.1204e-03  MAE 1.5346e-03  R² 1.0000 | val MSE 5.5363e-05  RMSE 7.4406e-03  MAE 2.1086e-03  R² 0.9999
    ep 1300 | train MSE 3.7769e-06  RMSE 1.9378e-03  MAE 1.3100e-03  R² 1.0000 | val MSE 5.2049e-05  RMSE 7.2145e-03  MAE 1.7694e-03  R² 0.9999
    ep 1325 | train MSE 3.7924e-06  RMSE 1.9431e-03  MAE 1.3281e-03  R² 1.0000 | val MSE 4.9068e-05  RMSE 7.0048e-03  MAE 1.7348e-03  R² 1.0000
    ep 1350 | train MSE 3.6687e-06  RMSE 1.9087e-03  MAE 1.3071e-03  R² 1.0000 | val MSE 5.3831e-05  RMSE 7.3369e-03  MAE 1.7591e-03  R² 0.9999
    ep 1375 | train MSE 3.6460e-06  RMSE 1.9010e-03  MAE 1.2952e-03  R² 1.0000 | val MSE 5.3607e-05  RMSE 7.3217e-03  MAE 1.7653e-03  R² 0.9999
    ep 1400 | train MSE 4.0336e-06  RMSE 2.0021e-03  MAE 1.3991e-03  R² 1.0000 | val MSE 5.1888e-05  RMSE 7.2034e-03  MAE 1.8681e-03  R² 0.9999
    ep 1425 | train MSE 3.7024e-06  RMSE 1.9192e-03  MAE 1.3474e-03  R² 1.0000 | val MSE 5.1273e-05  RMSE 7.1605e-03  MAE 1.8115e-03  R² 0.9999
    ep 1450 | train MSE 3.7199e-06  RMSE 1.9164e-03  MAE 1.3382e-03  R² 1.0000 | val MSE 5.7380e-05  RMSE 7.5749e-03  MAE 1.9044e-03  R² 0.9999
    ep 1475 | train MSE 3.6336e-06  RMSE 1.8950e-03  MAE 1.2743e-03  R² 1.0000 | val MSE 5.9474e-05  RMSE 7.7119e-03  MAE 1.8108e-03  R² 0.9999
    ep 1500 | train MSE 3.6061e-06  RMSE 1.8930e-03  MAE 1.2791e-03  R² 1.0000 | val MSE 5.4132e-05  RMSE 7.3574e-03  MAE 1.8496e-03  R² 0.9999
    ep 1525 | train MSE 3.2097e-06  RMSE 1.7852e-03  MAE 1.2258e-03  R² 1.0000 | val MSE 5.0735e-05  RMSE 7.1228e-03  MAE 1.7771e-03  R² 0.9999
    ep 1550 | train MSE 4.5050e-06  RMSE 2.1193e-03  MAE 1.4769e-03  R² 1.0000 | val MSE 4.6487e-05  RMSE 6.8181e-03  MAE 1.9905e-03  R² 1.0000
    ep 1575 | train MSE 3.2514e-06  RMSE 1.7886e-03  MAE 1.1449e-03  R² 1.0000 | val MSE 5.8614e-05  RMSE 7.6560e-03  MAE 1.7377e-03  R² 0.9999
    ep 1600 | train MSE 3.1434e-06  RMSE 1.7579e-03  MAE 1.1772e-03  R² 1.0000 | val MSE 5.6989e-05  RMSE 7.5491e-03  MAE 1.7887e-03  R² 0.9999
    ep 1625 | train MSE 3.0181e-06  RMSE 1.7211e-03  MAE 1.1972e-03  R² 1.0000 | val MSE 5.4843e-05  RMSE 7.4056e-03  MAE 1.7747e-03  R² 0.9999
    ep 1650 | train MSE 3.0494e-06  RMSE 1.7361e-03  MAE 1.1587e-03  R² 1.0000 | val MSE 5.0726e-05  RMSE 7.1222e-03  MAE 1.6699e-03  R² 0.9999
    ep 1675 | train MSE 4.1801e-06  RMSE 2.0387e-03  MAE 1.4047e-03  R² 1.0000 | val MSE 5.1149e-05  RMSE 7.1519e-03  MAE 2.0231e-03  R² 0.9999
    ep 1700 | train MSE 3.8441e-06  RMSE 1.9506e-03  MAE 1.3467e-03  R² 1.0000 | val MSE 4.8938e-05  RMSE 6.9956e-03  MAE 1.8541e-03  R² 1.0000
    ep 1725 | train MSE 3.0281e-06  RMSE 1.7211e-03  MAE 1.1407e-03  R² 1.0000 | val MSE 5.2610e-05  RMSE 7.2533e-03  MAE 1.7889e-03  R² 0.9999
    ep 1750 | train MSE 3.0653e-06  RMSE 1.7289e-03  MAE 1.1344e-03  R² 1.0000 | val MSE 5.6108e-05  RMSE 7.4905e-03  MAE 1.7740e-03  R² 0.9999
    ep 1775 | train MSE 6.8962e-06  RMSE 2.6162e-03  MAE 2.1032e-03  R² 1.0000 | val MSE 6.3272e-05  RMSE 7.9544e-03  MAE 2.7842e-03  R² 0.9999
    ep 1800 | train MSE 2.9943e-06  RMSE 1.7139e-03  MAE 1.0937e-03  R² 1.0000 | val MSE 5.5803e-05  RMSE 7.4701e-03  MAE 1.7272e-03  R² 0.9999
    ep 1825 | train MSE 4.3731e-06  RMSE 2.0854e-03  MAE 1.5245e-03  R² 1.0000 | val MSE 5.3149e-05  RMSE 7.2903e-03  MAE 2.1920e-03  R² 0.9999
    ep 1850 | train MSE 3.0409e-06  RMSE 1.7253e-03  MAE 1.1783e-03  R² 1.0000 | val MSE 5.2088e-05  RMSE 7.2172e-03  MAE 1.7347e-03  R² 0.9999
    ep 1875 | train MSE 2.7778e-06  RMSE 1.6551e-03  MAE 1.1001e-03  R² 1.0000 | val MSE 5.2090e-05  RMSE 7.2173e-03  MAE 1.6248e-03  R² 0.9999
    ep 1900 | train MSE 3.7261e-06  RMSE 1.9182e-03  MAE 1.3675e-03  R² 1.0000 | val MSE 5.3847e-05  RMSE 7.3381e-03  MAE 1.8933e-03  R² 0.9999
    ep 1925 | train MSE 3.1337e-06  RMSE 1.7509e-03  MAE 1.1739e-03  R² 1.0000 | val MSE 5.1845e-05  RMSE 7.2004e-03  MAE 1.7538e-03  R² 0.9999
    ep 1950 | train MSE 3.0196e-06  RMSE 1.7271e-03  MAE 1.1551e-03  R² 1.0000 | val MSE 4.6508e-05  RMSE 6.8197e-03  MAE 1.6932e-03  R² 1.0000
    ep 1975 | train MSE 2.8168e-06  RMSE 1.6650e-03  MAE 1.1058e-03  R² 1.0000 | val MSE 5.1658e-05  RMSE 7.1873e-03  MAE 1.7687e-03  R² 0.9999
    ep 2000 | train MSE 3.6074e-06  RMSE 1.8891e-03  MAE 1.3174e-03  R² 1.0000 | val MSE 5.7347e-05  RMSE 7.5728e-03  MAE 1.9291e-03  R² 0.9999
    ep 2025 | train MSE 5.3565e-06  RMSE 2.3061e-03  MAE 1.6604e-03  R² 1.0000 | val MSE 6.1612e-05  RMSE 7.8493e-03  MAE 2.2776e-03  R² 0.9999
    ep 2050 | train MSE 2.8053e-06  RMSE 1.6623e-03  MAE 1.0436e-03  R² 1.0000 | val MSE 5.0979e-05  RMSE 7.1400e-03  MAE 1.6042e-03  R² 0.9999
    ep 2075 | train MSE 2.8773e-06  RMSE 1.6873e-03  MAE 1.1893e-03  R² 1.0000 | val MSE 5.0213e-05  RMSE 7.0861e-03  MAE 1.7691e-03  R² 1.0000
    ep 2100 | train MSE 3.7007e-06  RMSE 1.9047e-03  MAE 1.1937e-03  R² 1.0000 | val MSE 5.0854e-05  RMSE 7.1312e-03  MAE 1.8007e-03  R² 0.9999
    ep 2125 | train MSE 2.7355e-06  RMSE 1.6361e-03  MAE 1.0095e-03  R² 1.0000 | val MSE 5.3488e-05  RMSE 7.3136e-03  MAE 1.5605e-03  R² 0.9999
    ep 2150 | train MSE 3.6863e-06  RMSE 1.9107e-03  MAE 1.4310e-03  R² 1.0000 | val MSE 5.9060e-05  RMSE 7.6851e-03  MAE 1.9505e-03  R² 0.9999
    ep 2175 | train MSE 2.5351e-06  RMSE 1.5857e-03  MAE 1.0799e-03  R² 1.0000 | val MSE 4.9403e-05  RMSE 7.0287e-03  MAE 1.5458e-03  R² 1.0000
    ep 2200 | train MSE 3.7871e-06  RMSE 1.9426e-03  MAE 1.4283e-03  R² 1.0000 | val MSE 4.7762e-05  RMSE 6.9110e-03  MAE 1.9746e-03  R² 1.0000
    ep 2225 | train MSE 4.9599e-06  RMSE 2.2176e-03  MAE 1.4843e-03  R² 1.0000 | val MSE 4.5866e-05  RMSE 6.7725e-03  MAE 2.0095e-03  R² 1.0000
    ep 2250 | train MSE 2.4311e-06  RMSE 1.5488e-03  MAE 9.4783e-04  R² 1.0000 | val MSE 5.2155e-05  RMSE 7.2218e-03  MAE 1.4837e-03  R² 0.9999
    ep 2275 | train MSE 2.5794e-06  RMSE 1.5844e-03  MAE 9.6414e-04  R² 1.0000 | val MSE 5.3705e-05  RMSE 7.3284e-03  MAE 1.6271e-03  R² 0.9999
    ep 2300 | train MSE 4.4224e-06  RMSE 2.0887e-03  MAE 1.4965e-03  R² 1.0000 | val MSE 6.2379e-05  RMSE 7.8981e-03  MAE 1.9976e-03  R² 0.9999
    ep 2325 | train MSE 2.5826e-06  RMSE 1.5924e-03  MAE 1.0696e-03  R² 1.0000 | val MSE 5.5297e-05  RMSE 7.4362e-03  MAE 1.7428e-03  R² 0.9999
    ep 2350 | train MSE 4.3501e-06  RMSE 2.0751e-03  MAE 1.5396e-03  R² 1.0000 | val MSE 4.7037e-05  RMSE 6.8584e-03  MAE 2.1983e-03  R² 1.0000
    ep 2375 | train MSE 3.4372e-06  RMSE 1.8498e-03  MAE 1.2574e-03  R² 1.0000 | val MSE 4.7250e-05  RMSE 6.8738e-03  MAE 1.8026e-03  R² 1.0000
    ep 2400 | train MSE 2.2114e-06  RMSE 1.4704e-03  MAE 9.1073e-04  R² 1.0000 | val MSE 5.4908e-05  RMSE 7.4100e-03  MAE 1.5394e-03  R² 0.9999
    ep 2425 | train MSE 7.3604e-06  RMSE 2.7044e-03  MAE 1.9865e-03  R² 1.0000 | val MSE 6.5463e-05  RMSE 8.0909e-03  MAE 2.6101e-03  R² 0.9999
    ep 2450 | train MSE 4.7050e-06  RMSE 2.1611e-03  MAE 1.5024e-03  R² 1.0000 | val MSE 4.8042e-05  RMSE 6.9312e-03  MAE 2.0337e-03  R² 1.0000
    ep 2475 | train MSE 1.8065e-06  RMSE 1.3291e-03  MAE 7.8984e-04  R² 1.0000 | val MSE 5.0170e-05  RMSE 7.0831e-03  MAE 1.3868e-03  R² 1.0000
    ep 2500 | train MSE 2.3121e-06  RMSE 1.5030e-03  MAE 9.3080e-04  R² 1.0000 | val MSE 5.0381e-05  RMSE 7.0980e-03  MAE 1.4984e-03  R² 1.0000
    ep 2525 | train MSE 5.6593e-06  RMSE 2.3722e-03  MAE 1.6965e-03  R² 1.0000 | val MSE 4.9531e-05  RMSE 7.0378e-03  MAE 2.2124e-03  R² 1.0000
    ep 2550 | train MSE 3.2049e-06  RMSE 1.7796e-03  MAE 1.2764e-03  R² 1.0000 | val MSE 5.4878e-05  RMSE 7.4080e-03  MAE 1.9441e-03  R² 0.9999
    ep 2575 | train MSE 1.8095e-06  RMSE 1.3284e-03  MAE 7.6594e-04  R² 1.0000 | val MSE 4.9297e-05  RMSE 7.0212e-03  MAE 1.3719e-03  R² 1.0000
    ep 2600 | train MSE 4.7512e-06  RMSE 2.1656e-03  MAE 1.5156e-03  R² 1.0000 | val MSE 6.3443e-05  RMSE 7.9651e-03  MAE 2.0772e-03  R² 0.9999
    ep 2625 | train MSE 3.2090e-06  RMSE 1.7778e-03  MAE 1.3279e-03  R² 1.0000 | val MSE 5.5158e-05  RMSE 7.4269e-03  MAE 1.8510e-03  R² 0.9999
    ep 2650 | train MSE 3.0483e-06  RMSE 1.7346e-03  MAE 1.1780e-03  R² 1.0000 | val MSE 5.2488e-05  RMSE 7.2449e-03  MAE 1.8596e-03  R² 0.9999
    ep 2675 | train MSE 6.0710e-06  RMSE 2.4395e-03  MAE 1.4928e-03  R² 1.0000 | val MSE 5.6925e-05  RMSE 7.5449e-03  MAE 1.9213e-03  R² 0.9999
    ep 2700 | train MSE 1.9881e-06  RMSE 1.3889e-03  MAE 7.7459e-04  R² 1.0000 | val MSE 5.0652e-05  RMSE 7.1170e-03  MAE 1.4162e-03  R² 1.0000
    ep 2725 | train MSE 2.6821e-06  RMSE 1.6336e-03  MAE 1.2125e-03  R² 1.0000 | val MSE 4.1437e-05  RMSE 6.4371e-03  MAE 1.8944e-03  R² 1.0000
    ep 2750 | train MSE 5.0149e-06  RMSE 2.2314e-03  MAE 1.4065e-03  R² 1.0000 | val MSE 4.5583e-05  RMSE 6.7516e-03  MAE 1.9441e-03  R² 1.0000
    ep 2775 | train MSE 7.3309e-06  RMSE 2.7041e-03  MAE 1.9844e-03  R² 1.0000 | val MSE 4.7389e-05  RMSE 6.8839e-03  MAE 2.5179e-03  R² 1.0000
    ep 2800 | train MSE 3.0574e-06  RMSE 1.7454e-03  MAE 1.1901e-03  R² 1.0000 | val MSE 3.9547e-05  RMSE 6.2886e-03  MAE 1.6706e-03  R² 1.0000
    ep 2825 | train MSE 4.1832e-06  RMSE 2.0414e-03  MAE 1.4845e-03  R² 1.0000 | val MSE 4.9831e-05  RMSE 7.0591e-03  MAE 2.1345e-03  R² 1.0000
    ep 2850 | train MSE 2.4310e-06  RMSE 1.5435e-03  MAE 1.0672e-03  R² 1.0000 | val MSE 5.1773e-05  RMSE 7.1953e-03  MAE 1.6922e-03  R² 0.9999
    ep 2875 | train MSE 1.9419e-06  RMSE 1.3721e-03  MAE 8.1893e-04  R² 1.0000 | val MSE 5.3276e-05  RMSE 7.2990e-03  MAE 1.4029e-03  R² 0.9999
    ep 2900 | train MSE 2.0455e-06  RMSE 1.4177e-03  MAE 8.1760e-04  R² 1.0000 | val MSE 5.0934e-05  RMSE 7.1368e-03  MAE 1.3821e-03  R² 0.9999
    ep 2925 | train MSE 3.7598e-06  RMSE 1.9225e-03  MAE 1.2354e-03  R² 1.0000 | val MSE 5.5388e-05  RMSE 7.4423e-03  MAE 1.8143e-03  R² 0.9999
    ep 2950 | train MSE 2.0083e-06  RMSE 1.3972e-03  MAE 8.5915e-04  R² 1.0000 | val MSE 5.0189e-05  RMSE 7.0844e-03  MAE 1.4094e-03  R² 1.0000
    ep 2975 | train MSE 1.7676e-06  RMSE 1.3147e-03  MAE 7.3313e-04  R² 1.0000 | val MSE 5.0980e-05  RMSE 7.1400e-03  MAE 1.3401e-03  R² 0.9999
    ep 3000 | train MSE 1.7459e-06  RMSE 1.3036e-03  MAE 7.3337e-04  R² 1.0000 | val MSE 5.0852e-05  RMSE 7.1311e-03  MAE 1.3621e-03  R² 0.9999



    
![png](README_files/README_5_1.png)
    


# Metrics


```python
import json
from pathlib import Path
from typing import Optional, Sequence, Union, Dict, Any, List

import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def predict_additional_runs_to_csv(
        model: torch.nn.Module,
        samples_csv: Union[str, Path],
        additional_runs_csv: Union[str, Path],
        schema_json: Union[str, Path],
        meta: Dict[str, Any],
        *,
        out_csv: Union[str, Path] = "predictions_additional.csv",
        join_key: str = "global_idx",
        additional_index_start: int = 0,  # 0 => nullbasiert; ggf. auf 1 stellen
        include_additional_features: Optional[Sequence[str]] = None,  # None => alle F:
        include_true_T_columns: bool = True,  # echte T:-Spalten mit in die CSV schreiben (falls vorhanden)
        device: Optional[Union[str, torch.device]] = None,
) -> pd.DataFrame:
    # --- Laden ---
    samples_csv = Path(samples_csv)
    additional_runs_csv = Path(additional_runs_csv)
    schema_json = Path(schema_json)
    out_csv = Path(out_csv)

    df_s = pd.read_csv(samples_csv)
    df_a = pd.read_csv(additional_runs_csv)
    with open(schema_json, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # --- join_key in additional erzeugen (autoritative Indexierung für OUT) ---
    if join_key not in df_a.columns:
        df_a = df_a.copy()
        df_a[join_key] = np.arange(additional_index_start,
                                   additional_index_start + len(df_a),
                                   dtype=int)

    # --- Trainings-Featureliste rekonstruieren ---
    X_cols_needed: List[str] = list(meta.get("feature_names", []))
    if not X_cols_needed:
        raise ValueError(
            "meta['feature_names'] fehlt – ohne diese Information können die Features nicht gebaut werden.")

    # --- Verfügbare F:/T: Spalten in additional_runs ---
    f_cols_available = [c for c in df_a.columns if isinstance(c, str) and c.startswith("F:")]
    t_cols_available = [c for c in df_a.columns if isinstance(c, str) and c.startswith("T:")]
    if include_additional_features is not None:
        missing = set(include_additional_features) - set(f_cols_available)
        if missing:
            raise KeyError(f"Gewünschte zusätzliche Feature-Spalten fehlen in additional_runs.csv: {sorted(missing)}")
        f_cols_available = list(include_additional_features)

    # --- Basis: additional_runs ist die Quelle. Nur wenn dort Features fehlen, selektiv aus samples mergen. ---
    df_m = df_a.copy()

    # Welche benötigten Features fehlen noch?
    missing_in_a = [c for c in X_cols_needed if c not in df_m.columns]
    if missing_in_a:
        # nur dann mergen, wenn samples den join_key + diese Spalten wirklich hat
        if join_key not in df_s.columns:
            raise KeyError(
                f"Folgende Modell-Features fehlen in additional_runs und können nicht aus samples gemappt werden, "
                f"weil '{join_key}' in samples.csv fehlt: {missing_in_a}"
            )
        missing_in_s = [c for c in missing_in_a if c not in df_s.columns]
        if missing_in_s:
            raise KeyError(
                f"Diese Modell-Features fehlen sowohl in additional_runs.csv als auch in samples.csv: {missing_in_s}"
            )
        df_m = pd.merge(
            df_m,
            df_s[[join_key] + missing_in_a],
            on=join_key,
            how="left"
        )

    # Final prüfen: Alle X-Spalten müssen jetzt vorhanden sein
    really_missing = [c for c in X_cols_needed if c not in df_m.columns]
    if really_missing:
        raise KeyError(
            f"Folgende für das Modell benötigte Feature-Spalten fehlen nach Zusammenführung: {really_missing}")

    # --- Numerik erzwingen (robuster gegen Strings/Whitespace) ---
    for c in X_cols_needed:
        df_m[c] = pd.to_numeric(df_m[c], errors="coerce")

    # --- Validitätsmaske: nur vollständige Zeilen in die Inferenz geben ---
    X_all = df_m[X_cols_needed].to_numpy(dtype=np.float32)
    valid_mask = ~np.any(np.isnan(X_all), axis=1)
    n, d = X_all.shape

    # Ergebniscontainer (NaN vorbelegen)
    # Zieldimension später nach erstem Forward definiert
    preds = None

    # --- Optional Normalisierung ---
    x_scaler = meta.get("x_scaler", None)
    if x_scaler is not None:
        X_all_scaled = np.empty_like(X_all)
        # nur valide Zeilen transformieren; invalide bleiben NaN
        X_all_scaled[:] = np.nan
        if valid_mask.any():
            X_all_scaled[valid_mask] = x_scaler.transform(X_all[valid_mask]).astype(np.float32)
        X_use = X_all_scaled
    else:
        X_use = X_all

    # --- Inferenz (nur für valide Zeilen) ---
    if valid_mask.any():
        X_valid = X_use[valid_mask]
        tX = torch.tensor(X_valid, dtype=torch.float32)
        model_device = next(model.parameters()).device
        if device is not None:
            tX = tX.to(device)
        else:
            tX = tX.to(model_device)

        model.eval()
        y_pred = model(tX)
        if isinstance(y_pred, torch.Tensor):
            y_pred_np = y_pred.detach().cpu().numpy()
        else:
            y_pred_np = np.asarray(y_pred, dtype=np.float32)

        # 2D-Form erzwingen
        if y_pred_np.ndim == 1:
            y_pred_np = y_pred_np.reshape(-1, 1)

        # --- ggf. Denormalisierung ---
        y_scaler = meta.get("y_scaler", None)
        if y_scaler is not None and y_pred_np.size > 0:
            y_pred_np = y_scaler.inverse_transform(y_pred_np)

        # In Gesamtcontainer eintragen
        T = y_pred_np.shape[1]
        preds = np.full((n, T), np.nan, dtype=np.float32)
        preds[valid_mask, :] = y_pred_np
    else:
        # keine validen Zeilen
        preds = np.full((n, 0), np.float32)

    # --- Zielspaltennamen bestimmen ---
    target_names = meta.get("target_names", None)
    if not target_names:
        # aus schema ableiten oder generisch
        if "target_keys" in schema and isinstance(schema["target_keys"], list) and preds.shape[1] == len(
                schema["target_keys"]):
            target_names = list(schema["target_keys"])
        elif "target_key" in schema and isinstance(schema["target_key"], str) and preds.shape[1] in (0, 1):
            target_names = [schema["target_key"]] if preds.shape[1] == 1 else []
        else:
            target_names = [f"target_{i}" for i in range(preds.shape[1])]

    # --- Output-Frame bauen ---
    out = pd.DataFrame({join_key: df_m[join_key].to_numpy()})

    if include_true_T_columns and t_cols_available:
        for col in t_cols_available:
            if col in df_m.columns:
                out[col] = df_m[col].to_numpy()

    for j, tname in enumerate(target_names):
        colname = f"pred_{tname}"
        if preds.shape[1] == 0:
            out[colname] = np.nan
        else:
            out[colname] = preds[:, j]

    out.sort_values(by=join_key, inplace=True, kind="mergesort")
    out.to_csv(out_csv, index=False)
    return out
```


```python
# Nach dem Training:
pred_df = predict_additional_runs_to_csv(
    model=model,
    samples_csv="oedo-viewer/viewer_data/samples.csv",
    additional_runs_csv="oedo-viewer/viewer_data/additional_runs.csv",  # oder runs.csv
    schema_json="oedo-viewer/viewer_data/schema.json",
    meta=meta,  # von compose_dataset_from_files beim Training
    out_csv="oedo-viewer/viewer_data/predictions_additional.csv",
    join_key="global_idx",
    additional_index_start=0,  # 0-basiert; auf 1 setzen, wenn du 1-basiert zählen willst
    include_additional_features=None,  # optional subset der F:-Spalten
    include_true_T_columns=True,  # echte T:-Spalten zum Vergleichen mitspeichern
    device=None  # None => nimmt automatisch das Model-Device
)

print("Saved:", "oedo-viewer/viewer_data/predictions_additional.csv")
```

    Saved: oedo-viewer/viewer_data/predictions_additional.csv



```python
from handler.handleData import *

# Run mit deinen Dateien
root_dir="oedo-viewer/viewer_data/"
original_pred_path = "predictions_additional.csv"
left_path = "oedo-viewer/viewer_data/runs.csv"
right_path = "predictions_additional_processed.csv"
process_pred_3stage(
    oedo_model= 1,
    root_dir= "oedo-viewer/viewer_data/",
    preds_path= "predictions_additional.csv",
    runs_path="oedo-viewer/viewer_data/runs.csv",
    merged_path= "predictions_additional_processed.csv",
    propagated_path = "predictions_additional_processed_propagated.csv",
    split_path= "split_assignments.csv",   # in same folder
    model = model,   # PyTorch model that predicts Es from [σ0, Δε]
    x_scaler = meta.get("x_scaler", None),           # sklearn-like scaler for X (optional)
    y_scaler = meta.get("y_scaler", None),           # sklearn-like scaler for y (optional, inverse_transform)
    device = None,
    propagated_recursive_path = "predictions_additional_processed_propagated_recursive.csv",
)
merged_df = merge_csv_simple(root_dir + left_path,root_dir + right_path, right_start_col=0)
merged_csv_path = Path(root_dir + "runs_final.csv")

merged_df.to_csv(merged_csv_path, index=False)
```
