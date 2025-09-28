import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import sys

eps_0 = 0
eps_d = -0.0005
sigma_max = -1000  # kPa (untere Grenze für Druck)
sigma_min = -100   # kPa (obere Grenze für Zug)
sigma_prime_p = -1
sigma_0 = sigma_prime_p
e_0 = 1
c_c = 0.005
c_s = 0.002

c1 = - (1 + e_0) / 2 * (c_c + c_s) / (c_s * c_c)
c2 = - (1 + e_0) / 2 * (c_c - c_s) / (c_s * c_c)
print("C_1", c1, "C_2", c2)

# Ergebnis-Container wie bei dir benannt
sigma_0_list_compress = []
eps_list_compress = []
sigma_0_list_extension = []
eps_list_extension = []
e_s_list_compress = []
e_s_list_extension = []
sigma_delta_list_compress = []
sigma_delta_list_extension = []

# --- 1) EIN Schritt: nur Mathematik, kein Modus/Grenzen ---
def strain_stress_step(eps_delta, sigma_0, eps_0, c1, c2):
    """
    Führt genau EIN Inkrement aus.
    - sigma_0, eps_0 werden um eps_delta fortgeschrieben
    - e_s wird über das Vorzeichen von eps_delta bestimmt:
        Druck (eps_delta < 0):  e_s = (c1 - c2) * sigma_neu
        Zug   (eps_delta > 0):  e_s = (c1 + c2) * sigma_neu
    Rückgabe:
        sigma_1, eps_1, e_s, sigma_delta
    """
    sigma_delta = c1 * sigma_0 * eps_delta + c2 * sigma_0 * abs(eps_delta)
    sigma_1 = sigma_0 + sigma_delta
    eps_1 = eps_0 + eps_delta

    if eps_delta < 0:
        e_s = (c1 - c2) * sigma_1
    else:
        e_s = (c1 + c2) * sigma_1

    return sigma_1, eps_1, e_s, sigma_delta

# --- 2) Handler: läuft Schritte, bis Grenze erreicht/überschritten ---
def strain_stress_path(eps_delta, sigma_0, eps_0, sigma_bound, c1, c2, debug=False):
    """
    Läuft wiederholt strain_stress_step, bis die passende Grenze erreicht ist.
    Konvention:
      - eps_delta < 0  (Druck):   laufe, bis sigma_0 < sigma_bound   (z.B. sigma_max)
      - eps_delta > 0  (Zug):     laufe, bis sigma_0 > sigma_bound   (z.B. sigma_min)
    Gibt Listen (sigma, eps, e_s, sigma_delta) zurück – Startzustand inklusive.
    Optional: debug=True -> pro Schritt Δσ etc. printen.
    """
    sigma_list = [sigma_0]
    eps_list = [eps_0]

    if eps_delta < 0:
        e_s0 = (c1 - c2) * sigma_0
        modus = "Druck"
    else:
        e_s0 = (c1 + c2) * sigma_0
        modus = "Zug"

    e_s_list = [e_s0]
    sigma_delta_list = [0]  # Startzustand: kein Inkrement gemacht

    if eps_delta == 0:
        return sigma_list, eps_list, e_s_list, sigma_delta_list

    laufen = True
    step_idx = 0
    while laufen:
        s_prev = sigma_list[-1]
        e_prev = eps_list[-1]
        s_new, e_new, e_s, s_delta = strain_stress_step(eps_delta, s_prev, e_prev, c1, c2)

        # Abbruchkriterium abhängig vom Vorzeichen (Druck/Zug)
        if eps_delta < 0:
            # Druckpfad „nach unten“ – stoppe, wenn Grenze unterschritten
            if s_new < sigma_bound:
                laufen = False
            else:
                sigma_list.append(s_new)
                eps_list.append(e_new)
                e_s_list.append(e_s)
                sigma_delta_list.append(s_delta)
                if debug:
                    print(f"[{modus}] step={step_idx:4d}  sigma_prev={s_prev: .6f}  "
                          f"delta_sigma={s_delta: .6f}  sigma_new={s_new: .6f}")
        else:
            # Zugpfad „nach oben“ – stoppe, wenn Grenze überschritten
            if s_new > sigma_bound:
                laufen = False
            else:
                sigma_list.append(s_new)
                eps_list.append(e_new)
                e_s_list.append(e_s)
                sigma_delta_list.append(s_delta)
                if debug:
                    print(f"[{modus}] step={step_idx:4d}  sigma_prev={s_prev: .6f}  "
                          f"delta_sigma={s_delta: .6f}  sigma_new={s_new: .6f}")

        step_idx += 1

    return sigma_list, eps_list, e_s_list, sigma_delta_list

# --- DataFrame-Helfer (einfach aufrufen) ---
def build_dataframe(eps_list_compress, eps_list_extension,
                    e_s_list_compress, e_s_list_extension,
                    sigma_0_list_compress, sigma_0_list_extension,
                    sigma_delta_list_compress, sigma_delta_list_extension,
                    eps_d):
    data = {
        "eps_total":   eps_list_compress + eps_list_extension,
        "eps_delta":   len(e_s_list_compress) * [eps_d] + len(e_s_list_extension) * [abs(eps_d)],
        "sigma_0":     sigma_0_list_compress + sigma_0_list_extension,
        "e_s":         e_s_list_compress + e_s_list_extension,
        "sigma_delta": sigma_delta_list_compress + sigma_delta_list_extension,
    }
    df = pd.DataFrame(
        data=data,
        columns=["eps_total", "eps_delta", "sigma_0", "e_s", "sigma_delta"]
    )
    # Print für Debug-Zwecke
    print(tabulate(df, headers='keys'))
    return df

# --- Plot-Helfer (eine Figure, zwei Subplots) ---
def plot_results(sigma_0_list_compress, eps_list_compress,
                 sigma_0_list_extension, eps_list_extension,
                 sigma_max, sigma_min,
                 sigma_delta_list_compress, sigma_delta_list_extension,
                 show_gif=False, gif_path="strain_debug.gif"):
    """
    Eine Figure mit zwei Achsen:
      Achse 1: σ–ε (Belastung/Entlastung)
      Achse 2: Δσ je Schritt (Index auf x)
    Optional: show_gif=True erzeugt eine einfache GIF-Animation des σ–ε-Pfads.
    """
    import numpy as np
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), constrained_layout=True)

    # --- Plot oben: σ–ε ---
    ax1.plot(sigma_0_list_compress, eps_list_compress, label="Belastung")
    ax1.plot([sigma_0_list_compress[-1]] + sigma_0_list_extension,
             [eps_list_compress[-1]] + eps_list_extension,
             label="Entlastung")
    ax1.grid(True, linestyle='-', linewidth=.75)
    ax1.axvline(sigma_max, linestyle='--', color='grey')
    ax1.text(
        sigma_max + 10,
        ax1.get_ylim()[1] * -.1,
        r"$\sigma_{max}$ = " + str(sigma_max),
        color='grey',
        rotation=90,
        va='top',
        ha='left'
    )
    ax1.axvline(sigma_min, linestyle='--', color='grey')
    ax1.text(
        sigma_min - 30,
        ax1.get_ylim()[1] * -.1,
        r"$\sigma_{min}$ = " + str(sigma_min),
        color='grey',
        rotation=90,
        va='top',
        ha='left'
    )
    ax1.set_xlabel(r"Stress $\sigma$ [kPa]")
    ax1.set_ylabel(r"Strain $\epsilon$ [-]")
    ax1.set_title("Einaxialer Kompressionsversuch (Einfachstes Modell)")
    ax1.legend()

    # --- Plot unten: Δσ ---
    sigma_delta_all = sigma_delta_list_compress + sigma_delta_list_extension
    idx = list(range(len(sigma_delta_all)))
    ax2.plot(idx, sigma_delta_all, marker="o", label=r"$\Delta\sigma$")
    # optische Trennlinie zwischen Druck- und Zugphase
    cut = len(sigma_delta_list_compress) - 1  # -1, weil Startwert 0 enthält
    if cut >= 0 and cut < len(idx):
        ax2.axvline(cut, linestyle='--', color='grey')
        ax2.text(cut + 0.3, np.nanmax(sigma_delta_all)*0.9 if len(sigma_delta_all) else 0,
                 "Wechsel", rotation=90, color='grey', va='top')
    ax2.grid(True, linestyle='-', linewidth=.75)
    ax2.set_xlabel("Schritt-Index")
    ax2.set_ylabel(r"$\Delta\sigma$ [kPa]")
    ax2.set_title(r"Inkremente $\Delta\sigma$")

    plt.show()

    # --- Optional: GIF-Animation (abschaltbar per Boolean) ---
    if show_gif:
        from matplotlib import animation
        from matplotlib.animation import PillowWriter

        # Pfad als Gesamtkurve (Kompression + Entlastung mit Übergangspunkt)
        sigma_all = sigma_0_list_compress + sigma_0_list_extension
        eps_all   = eps_list_compress + eps_list_extension

        fig_gif, ax_gif = plt.subplots(figsize=(6, 5))
        ax_gif.grid(True, linestyle='-', linewidth=.75)
        ax_gif.set_xlabel(r"Stress $\sigma$ [kPa]")
        ax_gif.set_ylabel(r"Strain $\epsilon$ [-]")
        ax_gif.set_title("σ–ε Pfad (Animation)")

        line, = ax_gif.plot([], [], lw=2)
        ax_gif.set_xlim(min(sigma_all), max(sigma_all))
        ax_gif.set_ylim(min(eps_all), max(eps_all))

        def init():
            line.set_data([], [])
            return (line,)

        def update(frame):
            line.set_data(sigma_all[:frame+1], eps_all[:frame+1])
            return (line,)

        ani = animation.FuncAnimation(fig_gif, update, frames=len(sigma_all),
                                      init_func=init, blit=True, interval=30)
        ani.save(gif_path, writer=PillowWriter(fps=30))
        plt.close(fig_gif)
        print(f"[INFO] GIF gespeichert unter: {gif_path}")

# --- Nutzung wie bei dir: erst Druck bis sigma_max, dann Zug bis sigma_min ---
# Druckpfad (mit Debug-Prints der Δσ)
sigma_0_list_compress, eps_list_compress, e_s_list_compress, sigma_delta_list_compress = strain_stress_path(
    eps_d, sigma_0, eps_0, sigma_max, c1, c2, debug=False
)

# Zugpfad (vom Endzustand des Druckpfads weiter, mit positivem Schritt)
sigma_start = sigma_0_list_compress[-1]
eps_start = eps_list_compress[-1]
eps_step_tension = abs(eps_d)

sigma_0_list_extension, eps_list_extension, e_s_list_extension, sigma_delta_list_extension = strain_stress_path(
    eps_step_tension, sigma_start, eps_start, sigma_min, c1, c2, debug=False
)

# DataFrame erzeugen und ausgeben (inkl. sigma_delta)
df = build_dataframe(
    eps_list_compress, eps_list_extension,
    e_s_list_compress, e_s_list_extension,
    sigma_0_list_compress, sigma_0_list_extension,
    sigma_delta_list_compress, sigma_delta_list_extension,
    eps_d
)

# Plots zeichnen (eine Figure mit 2 Subplots) und GIF optional abschalten/aktivieren
plot_results(
    sigma_0_list_compress, eps_list_compress,
    sigma_0_list_extension, eps_list_extension,
    sigma_max, sigma_min,
    sigma_delta_list_compress, sigma_delta_list_extension,
    show_gif=False,  # <- hier per Boolean GIF ein/aus
    gif_path="strain_debug.gif"
)
