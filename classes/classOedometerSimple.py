from classes.classOedometer import OedometerParent
import pandas as pd

class Oedometer(OedometerParent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Berechnungen durchführen
        self._calc_sigma_t_p1()

        # Listenlängen anpassen
        self._calc_total_epsilon()


    def _calc_total_epsilon(self):
        for i in range(len(self.delta_epsilon) - 1):
            self.total_epsilon.append(self.total_epsilon[i] + self.delta_epsilon[i])

    def _calc_e_s(self, sigma_t):
        """Berechnet `e_s` aus `sigma_t`."""
        e_s = -(1 + self.e_0[0]) / self.C_c[0] * sigma_t
        self.e_s.append(e_s)
        return e_s

    def _calc_sigma_t_p1(self):
        """Berechnet `sigma_t` und `delta_sigma` für die nächsten Schritte."""
        for i in range(self.max_n):  # -1, weil sigma_t bereits gesetzt ist
            e_s = self._calc_e_s(self.sigma_t[i])
            delta_sigma = e_s * self.delta_epsilon[0]
            self.delta_epsilon.append(self.delta_epsilon[0])
            sigma = self.sigma_t[i] + delta_sigma
            self.sigma_t.append(sigma)
            self.delta_sigma.append(delta_sigma)

    import pandas as pd

    # --- NEU: minimaler Output-Builder im Simple ---
    def _build_output_df(self):
        """
        Baut den kanonischen Output wie in 'Improved':
        - self.sigma_0_list_compress, self.eps_list_compress, self.e_s_list_compress, self.sigma_delta_list_compress
        - leere *_extension-Listen
        - self.df mit Spalten ['eps_total','eps_delta','sigma_0','e_s','sigma_delta'] (in dieser Reihenfolge)
        """
        # 1) Defensive Kopien der Basisreihen (so wie sie Simple bereits füllt)
        sigma = list(self.sigma_t) if hasattr(self, "sigma_t") else []
        d_sigma = list(self.delta_sigma) if hasattr(self, "delta_sigma") else []
        eps_tot = list(self.total_epsilon) if hasattr(self, "total_epsilon") else []
        e_s_calc = list(self.e_s) if hasattr(self, "e_s") else []
        d_eps0 = (self.delta_epsilon[0] if getattr(self, "delta_epsilon", None) else 0.0)

        # 2) Längen robust ausrichten (Startzustand muss enthalten sein; erste delta_sigma = 0)
        n = len(sigma)
        if n == 0:
            # Nichts gerechnet → minimaler Dummy (ein Startzustand)
            sigma = [getattr(self, "sigma_0", 0.0)]
            eps_tot = [getattr(self, "eps_0", 0.0)]
            e_s_calc = []  # wird unten ergänzt
            d_sigma = []

        # delta_sigma-Liste: erste = 0, danach vorhandene Werte, ggf. kürzen/padden
        d_sigma_full = [0.0] + d_sigma
        if len(d_sigma_full) > n:
            d_sigma_full = d_sigma_full[:n]
        elif len(d_sigma_full) < n:
            d_sigma_full += [0.0] * (n - len(d_sigma_full))

        # e_s-Liste: wenn ein Eintrag fehlt, aus Simple-Formel nachziehen
        # Simple nutzt: e_s = -(1 + e_0)/C_c * sigma
        def calc_es(s):
            return -(1.0 + self.e_0[0]) / self.C_c[0] * s

        if len(e_s_calc) >= n:
            e_s_full = e_s_calc[:n]
        else:
            # vorhandene übernehmen, Rest nachrechnen (inkl. Start falls nötig)
            e_s_full = e_s_calc + [calc_es(s) for s in sigma[len(e_s_calc):]]
            if len(e_s_full) > n:
                e_s_full = e_s_full[:n]

        # eps_total: falls zu kurz/lang, sanft richten (zur Not aus Start + d_eps0 hochzählen)
        if len(eps_tot) != n:
            eps0 = eps_tot[0] if eps_tot else getattr(self, "eps_0", 0.0)
            eps_tot = [eps0 + i * d_eps0 for i in range(n)]

        # 3) In Improved-Form bringen: alles als "Kompression"; Extension leer lassen
        self.sigma_0_list_compress = sigma
        self.eps_list_compress = eps_tot
        self.e_s_list_compress = e_s_full
        self.sigma_delta_list_compress = d_sigma_full

        self.sigma_0_list_extension = []
        self.eps_list_extension = []
        self.e_s_list_extension = []
        self.sigma_delta_list_extension = []

        # 4) DataFrame im exakt gleichen Format/der gleichen Reihenfolge
        eps_delta_list = [d_eps0] * n  # pro Phase konstant; Startzustand inklusive
        data = {
            "eps_total": self.eps_list_compress,
            "eps_delta": eps_delta_list,
            "sigma_0": self.sigma_0_list_compress,
            "e_s": self.e_s_list_compress,
            "sigma_delta": self.sigma_delta_list_compress,
        }
        self.df = pd.DataFrame(data, columns=["eps_total", "eps_delta", "sigma_0", "e_s", "sigma_delta"])

        # 5) (Optional, Matches Improved) Bequemlichkeits-Gesamtliste
        self.e_s_list = list(self.e_s_list_compress)  # Extension leer, also identisch
        self.sigma_0_list = list(self.sigma_0_list_compress)
        self.eps_s_list = list(self.eps_list_compress)
        self.eps_delta_list = list(eps_delta_list)

    # --- NEU: öffentliche API wie bei Improved ---
    def run(self, debug: bool = False):
        """
        Baut nur den Output-Frame und die Listen im Improved-Format.
        Keine zusätzliche Logik – nutzt die bereits berechneten Reihen der Simple-Variante.
        """
        self._build_output_df()
        return self.df

