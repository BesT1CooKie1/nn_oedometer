from classes.classOedometer import OedometerParent

class Oedometer(OedometerParent):
    def _calc_e_s(self, sigma_t):
        """Berechnet `e_s` aus `sigma_t`."""
        e_s = (1 + self.e_0[0]) / self.C_c[0] * sigma_t
        self.e_s.append(e_s)
        return e_s

    def _calc_sigma_t_p1(self):
        """Berechnet `sigma_t` und `delta_sigma` für die nächsten Schritte."""
        for i in range(self.max_n):
            e_s = self._calc_e_s(self.sigma_t[i])
            self.delta_epsilon.append(self.delta_epsilon[i-1])
            C_1 = - (1+self.e_0[0]) / 2 * (self.C_s[0] + self.C_c[0]) / (self.C_s[0] * self.C_c[0])
            C_2 = - (1+self.e_0[0]) / 2 * (self.C_c[0] - self.C_s[0]) / (self.C_s[0] * self.C_c[0])

            print(C_1, C_2, (self.C_s[0] + self.C_c[0]) / (self.C_s[0] * self.C_c[0]))

            term_1 = C_1 * self.sigma_t[i] * self.delta_epsilon[i]
            term_2 = (C_2 * self.sigma_t[i] * self.total_epsilon[i]) if self.e_s[i-1] * self.delta_epsilon[i-1] >= 1000 else 0
            delta_sigma = term_1 + term_2
            print(term_1, term_2, delta_sigma)

            sigma = self.sigma_t[i] + delta_sigma
            self.total_epsilon.append(self.total_epsilon[i] + self.delta_epsilon[i] * (-1 if e_s * self.delta_epsilon[i] >= 1000 else 1))
            self.sigma_t.append(sigma)
            self.delta_sigma.append(delta_sigma)