import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

class Oedometer:
    """Einfaches 1D-Modell für den einaxialen Kompressions-/Entlastungspfad.

    Args:
        e0 (float): Anfangs-Porosität (dimensionslos).
        c_c (float): Kompressionskoeffizient.
        c_s (float): Rekompressionskoeffizient.
        sigma_prime_p (float): Vorbelastungsspannung [kPa], negative Werte = Druck.
        sigma_max (float): Untere Druckgrenze [kPa].
        sigma_min (float): Obere Zuggrenze [kPa].
        eps_delta (float): Dehnungsinkrement pro Schritt; Vorzeichen steuert Modus.
        eps_0 (float): Startdehnung.
        max_iter (int): Sicherheitslimit gegen Endlosschleifen.

    Notes:
        Abgeleitete Koeffizienten:

        $$C_1 = -\\frac{1+e_0}{2}\\cdot\\frac{C_c+C_s}{C_s\\cdot C_c},\\quad
          C_2 = -\\frac{1+e_0}{2}\\cdot\\frac{C_c-C_s}{C_s\\cdot C_c}$$

        Spannungsinkrement:

        $$\\Delta\\sigma = C_1\\cdot\\sigma_0\\cdot\\Delta\\varepsilon
                           + C_2\\cdot\\sigma_0\\cdot|\\Delta\\varepsilon|$$

        Update und Modul:

        $$\\sigma_1=\\sigma_0+\\Delta\\sigma,\\qquad
          E_s = (C_1 + \\mathrm{sgn}(\\Delta\\varepsilon)\\cdot C_2)\\cdot\\sigma_1$$
    """
    def __init__(
        self,
        e0=1,
        c_c=0.005,
        c_s=0.002,
        sigma_prime_p=-1,
        sigma_max=-1000,  # kPa
        sigma_min=-100,   # kPa
        eps_delta=-0.0005,
        eps_0=0,
        max_iter=1_000_000_000,
    ):
        # fixe Parameter
        self.e0 = e0
        self.c_c = c_c
        self.c_s = c_s
        self.sigma_prime_p = sigma_prime_p
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.eps_delta = eps_delta
        self.eps_0_init = eps_0
        self.max_iter = max_iter


        # abgeleitete Koeffizienten
        self.c1 = - (1 + self.e0) / 2 * (self.c_c + self.c_s) / (self.c_s * self.c_c)
        self.c2 = - (1 + self.e0) / 2 * (self.c_c - self.c_s) / (self.c_s * self.c_c)

        # Ergebniscontainer
        self.sigma_0_list_compress = []
        self.eps_list_compress = []
        self.e_s_list_compress = []
        self.sigma_delta_list_compress = []

        self.sigma_0_list_extension = []
        self.eps_list_extension = []
        self.e_s_list_extension = []
        self.sigma_delta_list_extension = []

        # Gesamtausgaben (für Bequemlichkeit)
        self.e_s_list = []
        self.sigma_0_list = []
        self.eps_s_list = []
        self.eps_delta_list = []
        self.df = None  # wird nach run() gesetzt

    # --- EIN Schritt: nur Mathematik, kein Modus/Grenzen ---
    def step(self, eps_delta, sigma_0, eps_0):
        """Führt ein Inkrement aus (rein mathematisch, kein Grenzcheck).

        Args:
            eps_delta (float): Dehnungsinkrement (Vorzeichen definiert Druck/Zug).
            sigma_0 (float): Aktuelle effektive Spannung [kPa].
            eps_0 (float): Aktuelle Dehnung [-].

        Returns:
            tuple[float, float, float, float]:
                ``(sigma_1, eps_1, E_s, sigma_delta)``.

        Notes:
            $$\\Delta\\sigma = C_1\\cdot\\sigma_0\\cdot\\Delta\\varepsilon
                               + C_2\\cdot\\sigma_0\\cdot|\\Delta\\varepsilon|.$$
        """

        sigma_delta = self.c1 * sigma_0 * eps_delta + self.c2 * sigma_0 * abs(eps_delta)
        sigma_1 = sigma_0 + sigma_delta
        eps_1 = eps_0 + eps_delta

        if eps_delta < 0:
            e_s = (self.c1 - self.c2) * sigma_1
        else:
            e_s = (self.c1 + self.c2) * sigma_1

        return sigma_1, eps_1, e_s, sigma_delta

    # --- Handler: läuft Schritte, bis Grenze erreicht/überschritten ---
    def run_phase(self, eps_delta, sigma_0, eps_0, compression=True, debug=False):
        """Läuft den Pfad iterativ, bis die jeweilige Grenze erreicht wird.

        Args:
            eps_delta (float): Dehnungsinkrement pro Schritt. Vorzeichen steuert den Modus
                (eps_delta < 0 ⇒ Druck, eps_delta > 0 ⇒ Zug).
            sigma_0 (float): Start-Spannung [kPa].
            eps_0 (float): Start-Dehnung [-].
            compression (bool): Wenn True, Kompressionsphase (Abbruch an ``sigma_max``),
                sonst Extension (Abbruch an ``sigma_min``).
            debug (bool): Wenn True, pro Schritt Debug-Ausgabe inkl. ``delta_sigma``.

        Returns:
            tuple[list[float], list[float], list[float], list[float], float, float]:
                ``(sigma_list, eps_list, E_s_list, sigma_delta_list, sigma_last, eps_last)``.
                Der **Startzustand** ist enthalten (``sigma_delta_list[0] == 0``);
                ``E_s_list[0]`` wird mit dem Modus konsistent berechnet.

        Raises:
            RuntimeError: Wenn ``max_iter`` erreicht ist (Schutz gegen Endlosschleifen).

        Notes:
            Schritt-Update (intern via ``step``):

            $$\\Delta\\sigma = C_1\\cdot\\sigma_0\\cdot\\Delta\\varepsilon
               \\cdot + \\cdot C_2\\cdot\\sigma_0\\cdot|\\Delta\\varepsilon|,$$

            $$\\sigma_1 = \\sigma_0 + \\Delta\\sigma, \\qquad
              E_s = (C_1 + \\mathrm{sgn}(\\Delta\\varepsilon)\\cdot C_2)\\cdot\\sigma_1.$$

            **Abbruchkriterien**:
            - Kompression: stoppe, wenn ``sigma_new < sigma_max`` (Grenze unterschritten).
            - Extension:  stoppe, wenn ``sigma_new > sigma_min`` (Grenze überschritten).

            Der Grenz-verletzende Schritt wird **nicht** angehängt (letzter gültiger Zustand).
        """
        sigma_list = [sigma_0]
        eps_list = [eps_0]
        if compression:
            e_s0 = (self.c1 - self.c2) * sigma_0
            modus = "Druck"
        else:
            e_s0 = (self.c1 + self.c2) * sigma_0
            modus = "Zug"

        e_s_list = [e_s0]
        sigma_delta_list = [0]  # Startzustand: kein Inkrement gemacht

        if eps_delta == 0:
            return sigma_list, eps_list, e_s_list, sigma_delta_list, sigma_0, eps_0

        iter_count = 0
        step_idx = 0
        while True:
            if iter_count >= self.max_iter:
                raise RuntimeError("Maximale Iterationen erreicht, Abbruch (möglicher Endlosschleifenfall).")
            iter_count += 1

            s_prev = sigma_list[-1]
            e_prev = eps_list[-1]
            s_new, e_new, e_s, s_delta = self.step(eps_delta, s_prev, e_prev)

            # Abbruchkriterium abhängig vom Vorzeichen (Druck/Zug)
            if compression:
                # Druckpfad „nach unten“ – stoppe, wenn Grenze unterschritten
                if s_new < self.sigma_max:
                    break
            else:
                # Zugpfad „nach oben“ – stoppe, wenn Grenze überschritten
                if s_new > self.sigma_min:
                    break

            sigma_list.append(s_new)
            eps_list.append(e_new)
            e_s_list.append(e_s)
            sigma_delta_list.append(s_delta)

            if debug:
                print(f"[{modus}] step={step_idx:4d}  sigma_prev={s_prev: .6f}  "
                      f"delta_sigma={s_delta: .6f}  sigma_new={s_new: .6f}")
            step_idx += 1

        # Rückgabe inkl. letztem gültigen Zustand (ohne Grenzüberschreitung)
        return sigma_list, eps_list, e_s_list, sigma_delta_list, sigma_list[-1], eps_list[-1]

    def run(self, debug=False):
        """Führt Kompression und anschließend Extension aus und baut den Result-DataFrame.

        Args:
            debug (bool): Durchgereicht an ``run_phase``. Wenn True, Debug-Prints je Schritt.

        Returns:
            pandas.DataFrame: Spalten
                ``["eps_total", "eps_delta", "sigma_0", "E_s", "sigma_delta"]``.
                Der Aufbau ist **identisch** zum Playground:
                - ``eps_total``: Verkettung von Kompressions- und Extensions-Dehnung.
                - ``eps_delta``: konstantes Inkrement je Phase
                  (Kompression = ``eps_delta``, Extension = ``abs(eps_delta)``).
                - ``sigma_0``: Spannungsverlauf (Startpunkt enthalten).
                - ``E_s``: Modul je Schritt, konsistent mit dem Vorzeichen von ``eps_delta``.
                - ``sigma_delta``: Spannungsinkremente (Start = 0).

        Notes:
            Interner Ablauf:
            1. Kompressionsphase: ``run_phase(self.eps_delta, sigma_prime_p, eps_0_init, compression=True)``
            2. Extensionsphase:  ``run_phase(abs(self.eps_delta), ..., compression=False)`` ab Endzustand der Kompression.

            Nebenprodukte (für Plots/Tabellen) werden auf ``self.*`` abgelegt
            (z. B. ``self.sigma_0_list_compress``, ``self.sigma_delta_list_extension``).
        """

        # Startwerte
        sigma_0 = self.sigma_prime_p
        eps_0 = self.eps_0_init

        # Kompressionsphase (eps_delta negativ)
        (
            self.sigma_0_list_compress,
            self.eps_list_compress,
            self.e_s_list_compress,
            self.sigma_delta_list_compress,
            sigma_0_after_compression,
            eps_0_after_compression,
        ) = self.run_phase(self.eps_delta, sigma_0, eps_0, compression=True, debug=debug)

        # Extensionsphase (positives eps_delta)
        (
            self.sigma_0_list_extension,
            self.eps_list_extension,
            self.e_s_list_extension,
            self.sigma_delta_list_extension,
            _,  # finaler sigma_0 nach extension
            _,
        ) = self.run_phase(abs(self.eps_delta), sigma_0_after_compression, eps_0_after_compression,
                            compression=False, debug=debug)

        # DataFrame zusammenbauen – IDENTISCH zum Playground (inkl. sigma_delta)
        eps_total = self.eps_list_compress + self.eps_list_extension
        sigma_0_total = self.sigma_0_list_compress + self.sigma_0_list_extension
        e_s_total = self.e_s_list_compress + self.e_s_list_extension
        sigma_delta_total = self.sigma_delta_list_compress + self.sigma_delta_list_extension

        eps_delta_list = (
            len(self.e_s_list_compress) * [self.eps_delta] +
            len(self.e_s_list_extension) * [abs(self.eps_delta)]
        )

        self.e_s_list = e_s_total
        self.sigma_0_list = sigma_0_total
        self.eps_s_list = eps_total
        self.eps_delta_list = eps_delta_list

        data = {
            "eps_total":   eps_total,
            "eps_delta":   eps_delta_list,
            "sigma_0":     sigma_0_total,
            "e_s":         e_s_total,
            "sigma_delta": sigma_delta_total,
        }

        self.df = pd.DataFrame(data=data, columns=["eps_total", "eps_delta", "sigma_0", "e_s", "sigma_delta"])
        return self.df

    def plot(self, show=True, show_gif=False, gif_path="strain_debug.gif"):
        """Zeichnet eine Figure mit zwei Subplots (oben σ–ε, unten Δσ). GIF optional.

        Args:
            show (bool): Wenn True, zeigt die Figure direkt mit ``plt.show()``.
            show_gif (bool): Wenn True, speichert eine GIF-Animation des σ–ε-Pfads.
            gif_path (str): Ausgabepfad für das GIF (nur wirksam, wenn ``show_gif=True``).

        Returns:
            None

        Notes:
            **Subplot 1 (oben):** σ–ε-Pfad
            - Kompression: ``(self.sigma_0_list_compress, self.eps_list_compress)`` (Label „Belastung“).
            - Extension:  wird ab dem Kompressions-Endpunkt angeschlossen (Label „Entlastung“).
            - Vertikale Hilfslinien: ``sigma_max`` und ``sigma_min``.

            **Subplot 2 (unten):** Δσ je Schritt
            - Kurve: ``self.sigma_delta_list_compress + self.sigma_delta_list_extension``.
            - Trennmarke zwischen den Phasen (basierend auf der Länge der Kompressionsliste).
            - Startwert ist 0 (entspricht dem initialen Zustand ohne Inkrement).

            Das optionale GIF zeichnet den σ–ε-Pfad schrittweise nach (Fixpunkt-Animation).
        """
        if self.df is None:
            raise RuntimeError("Run() muss zuerst aufgerufen werden, bevor geplottet wird.")

        # Eine Figure mit 2 Subplots: oben σ–ε, unten Δσ
        import numpy as np
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), constrained_layout=True)

        # --- Plot oben: σ–ε ---
        ax1.plot(self.sigma_0_list_compress, self.eps_list_compress, label="Belastung")
        if self.sigma_0_list_compress and self.sigma_0_list_extension:
            ax1.plot(
                [self.sigma_0_list_compress[-1]] + self.sigma_0_list_extension,
                [self.eps_list_compress[-1]] + self.eps_list_extension,
                label="Entlastung"
            )
        ax1.grid(True, linestyle='-', linewidth=.75)
        ax1.axvline(self.sigma_max, linestyle='--', color='grey')
        ax1.text(
            self.sigma_max + 10,
            ax1.get_ylim()[1] * -0.1,
            r"$\sigma_{max}$ = " + str(self.sigma_max),
            color='grey',
            rotation=90,
            va='top',
            ha='left',
        )
        ax1.axvline(self.sigma_min, linestyle='--', color='grey')
        ax1.text(
            self.sigma_min - 30,
            ax1.get_ylim()[1] * -0.1,
            r"$\sigma_{min}$ = " + str(self.sigma_min),
            color='grey',
            rotation=90,
            va='top',
            ha='left',
        )
        ax1.set_xlabel(r"Stress $\sigma$ [kPa]")
        ax1.set_ylabel(r"Strain $\epsilon$ [-]")
        ax1.set_title("Einaxialer Kompressionsversuch (Einfachstes Modell)")
        ax1.legend()

        # --- Plot unten: Δσ ---
        sigma_delta_all = self.sigma_delta_list_compress + self.sigma_delta_list_extension
        idx = list(range(len(sigma_delta_all)))
        ax2.plot(idx, sigma_delta_all, marker="o", label=r"$\Delta\sigma$")
        # Trennlinie zwischen Druck- und Zugphase
        cut = len(self.sigma_delta_list_compress) - 1  # -1 wegen Startwert 0
        if 0 <= cut < len(idx):
            ax2.axvline(cut, linestyle='--', color='grey')
            ymax = np.nanmax(sigma_delta_all) if len(sigma_delta_all) else 0
            ax2.text(cut + 0.3, ymax*0.9 if ymax != 0 else 0, "Wechsel", rotation=90, color='grey', va='top')
        ax2.grid(True, linestyle='-', linewidth=.75)
        ax2.set_xlabel("Schritt-Index")
        ax2.set_ylabel(r"$\Delta\sigma$ [kPa]")
        ax2.set_title(r"Inkremente $\Delta\sigma$")
        ax2.legend()

        if show:
            plt.show()

        # Optional: GIF (σ–ε Pfad)
        if show_gif:
            from matplotlib import animation
            from matplotlib.animation import PillowWriter

            sigma_all = self.sigma_0_list_compress + self.sigma_0_list_extension
            eps_all = self.eps_list_compress + self.eps_list_extension

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

    def summary_table(self):
        """Gibt die Ergebnistabelle als formatierten Text zurück (``tabulate``).

        Returns:
            str: Formatierter Tabellen-String der DataFrame-Inhalte mit den Spalten
                ``["eps_total", "eps_delta", "sigma_0", "E_s", "sigma_delta"]``.

        Raises:
            RuntimeError: Wenn ``run()`` noch nicht aufgerufen wurde (``self.df is None``).

        Notes:
            Die Tabelle ist ein **reines Leseformat** (Debug/Reporting).
            Für Auswertung/Export nutze den DataFrame ``self.df`` direkt.
        """
        if self.df is None:
            raise RuntimeError("Run() muss zuerst aufgerufen werden, bevor die Tabelle erstellt wird.")
        return tabulate(self.df, headers="keys")  # gleich wie im Playground
