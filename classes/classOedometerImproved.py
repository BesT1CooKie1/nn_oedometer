import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

class Oedometer:
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
        max_iter=1_000_000,
    ):
        # fixe Parameter
        self.e0 = e0
        self.c_c = c_c
        self.c_s = c_s
        self.sigma_prime_p = sigma_prime_p
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.eps_delta = eps_delta  # negativ für Kompression
        self.eps_0_init = eps_0
        self.max_iter = max_iter

        # abgeleitete Koeffizienten
        self.c1 = - (1 + self.e0) / 2 * (self.c_c + self.c_s) / (self.c_s * self.c_c)
        self.c2 = - (1 + self.e0) / 2 * (self.c_c - self.c_s) / (self.c_s * self.c_c)

        # Ergebniscontainer
        self.sigma_0_list_compress = []
        self.eps_list_compress = []
        self.e_s_list_compress = []

        self.sigma_0_list_extension = []
        self.eps_list_extension = []
        self.e_s_list_extension = []

        self.e_s_list = []
        self.sigma_0_list = []
        self.eps_s_list = []
        self.eps_delta_list = []

        self.df = None  # wird nach run() gesetzt

    def _run_phase(self, eps_delta, sigma_0, eps_0, compression=True):
        """
        Führt eine Phase (Kompression oder Extension) durch.
        compression=True: erste Phase, benutzt e_s = (c1 - c2)*sigma_0 und bricht ab, wenn sigma_0 < sigma_max.
        compression=False: Extension, benutzt e_s = (c1 + c2)*sigma_0 und bricht ab, wenn sigma_0 > sigma_min.
        """
        sigma_list = []
        e_s_list = []
        eps_list = []
        if compression:
            sigma_list.append(sigma_0)
            e_s_list.append((self.c1 - self.c2) * sigma_0)
            eps_list.append(0)

        iter_count = 0
        while True:
            if iter_count >= self.max_iter:
                raise RuntimeError("Maximale Iterationen erreicht, Abbruch (möglicher Endlosschleifenfall).")
            iter_count += 1

            sigma_delta = self.c1 * sigma_0 * eps_delta + self.c2 * sigma_0 * abs(eps_delta)
            new_sigma_0 = sigma_0 + sigma_delta
            new_eps_0 = eps_0 + eps_delta

            if compression:
                e_s = (self.c1 - self.c2) * new_sigma_0
                # Abbruchbedingung für Kompression: wenn sigma_0 < sigma_max (zu stark komprimiert), dann revertieren
                if new_sigma_0 < self.sigma_max:
                    # vorheriger Zustand ist Ende der Kompression
                    break
            else:
                e_s = (self.c1 + self.c2) * new_sigma_0
                # Abbruchbedingung für Extension: wenn sigma_0 > sigma_min, dann beenden (ohne einzufügen)
                if new_sigma_0 > self.sigma_min:
                    break

            # Zustand übernehmen
            sigma_0 = new_sigma_0
            eps_0 = new_eps_0
            sigma_list.append(sigma_0)
            eps_list.append(eps_0)
            e_s_list.append(e_s)

        return sigma_list, eps_list, e_s_list, sigma_0, eps_0

    def run(self):
        # Startwerte
        sigma_0 = self.sigma_prime_p
        eps_0 = self.eps_0_init

        # Kompressionsphase (eps_delta negativ)
        (
            self.sigma_0_list_compress,
            self.eps_list_compress,
            self.e_s_list_compress,
            sigma_0_after_compression,
            eps_0_after_compression,
        ) = self._run_phase(self.eps_delta, sigma_0, eps_0, compression=True)

        # Extensionsphase (positives eps_delta)
        (
            self.sigma_0_list_extension,
            self.eps_list_extension,
            self.e_s_list_extension,
            _,  # finaler sigma_0 nach extension
            _,
        ) = self._run_phase(abs(self.eps_delta), sigma_0_after_compression, eps_0_after_compression, compression=False)

        # DataFrame zusammenbauen
        eps_total = self.eps_list_compress + self.eps_list_extension
        sigma_0_total = self.sigma_0_list_compress + self.sigma_0_list_extension
        e_s_total = self.e_s_list_compress + self.e_s_list_extension
        eps_delta_list = (
                (len(self.e_s_list_compress)-1) * [self.eps_delta] + [abs(self.eps_delta)]
            + (len(self.e_s_list_extension)) * [abs(self.eps_delta)]
        )
        # 0 Ausgangszustand, -1 Belastungszustand, 1 Entlastungszustand
        state_list = [0] + (len(self.e_s_list_compress)-1) * [-1] + len(self.e_s_list_extension) * [1]

        self.e_s_list = e_s_total
        self.sigma_0_list = sigma_0_total
        self.eps_s_list = eps_total
        self.eps_delta_list = eps_delta_list

        data = {
            "eps_total": eps_total,
            "eps_delta": eps_delta_list,
            "sigma_0": sigma_0_total,
            "e_s": e_s_total,
            "state": state_list,
        }

        self.df = pd.DataFrame(data=data, columns=["eps_total", "eps_delta", "sigma_0", "e_s", "state"])
        return self.df

    def plot(self, show=True):
        if self.df is None:
            raise RuntimeError("Run() muss zuerst aufgerufen werden, bevor geplottet wird.")

        # Belastung (Kompression)
        plt.plot(
            self.sigma_0_list_compress,
            self.eps_list_compress,
            label="Belastung",
            color="red",
        )

        # Entlastung: Übergangspunkt + Extension
        if self.sigma_0_list_compress and self.sigma_0_list_extension:
            plt.plot(
                [self.sigma_0_list_compress[-1]] + self.sigma_0_list_extension,
                [self.eps_list_compress[-1]] + self.eps_list_extension,
                label="Entlastung",
                color="blue",
            )

        plt.grid(color="grey", linestyle="-", linewidth=0.75)
        plt.axvline(self.sigma_max, linestyle="--", color="grey")
        plt.text(
            self.sigma_max + 10,
            plt.ylim()[1] * -0.1,
            r"$\sigma_{max}$ = " + str(self.sigma_max),
            color="grey",
            rotation=90,
            va="top",
            ha="left",
        )
        plt.axvline(self.sigma_min, linestyle="--", color="grey")
        plt.text(
            self.sigma_min - 30,
            plt.ylim()[1] * -0.1,
            r"$\sigma_{min}$ = " + str(self.sigma_min),
            color="grey",
            rotation=90,
            va="top",
            ha="left",
        )
        plt.xlabel(r"Stress $\sigma$ [kPa]")
        plt.ylabel(r"Strain $\epsilon$ [-]")
        plt.title("Einaxialer Kompressionsversuch (Einfachstes Modell)")
        plt.legend()
        plt.tight_layout()
        if show:
            plt.show()

    def summary_table(self):
        if self.df is None:
            raise RuntimeError("Run() muss zuerst aufgerufen werden, bevor die Tabelle erstellt wird.")
        return tabulate(self.df, headers="keys", tablefmt="psql", showindex=False)