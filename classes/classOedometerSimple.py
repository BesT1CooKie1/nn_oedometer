from classes.classOedometer import OedometerParent

class Oedometer(OedometerParent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Berechnungen durchführen
        self._calc_sigma_t_p1()

        # Listenlängen anpassen
        self._calc_total_epsilon()

        for i in range(len(self.sigma_t)):
            print(self.sigma_t[i], self.total_epsilon[i])


    def _calc_total_epsilon(self):
        for i in range(len(self.delta_epsilon) - 1):
            self.total_epsilon.append(self.total_epsilon[i] + self.delta_epsilon[i])
        print(self.total_epsilon)

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
