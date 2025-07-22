from abc import ABC, abstractmethod

class OedometerParent(ABC):
    def __init__(
        self,
        e_0: float = 1.00,
        C_c: float = 0.005,
        C_s: float = 0.001,
        delta_epsilon: float = -0.0005,
        sigma_t: float = -1.00,
        max_n: int = 50,
        rand_epsilon: bool = False,
        **kwargs
    ):
        self.max_n = max_n

        # Standardwerte als Listen setzen
        self.e_0 = [e_0]
        self.C_c = [C_c]
        self.C_s = [C_s]
        self.sigma_t = [sigma_t]
        self.delta_epsilon = [delta_epsilon]
        self.total_epsilon = [0]

        # Initiale Listen für Berechnungen
        self.delta_sigma = []
        self.e_s = []

        # Dynamische Zuweisung von kwargs, falls vorhanden
        for key, value in kwargs.items():
            if hasattr(self, key):  # Nur vorhandene Attribute setzen
                setattr(self, key, [value])


    @abstractmethod
    def _calc_e_s(self, sigma_t):
        """Berechnet `e_s` für die nächsten Schritte."""
        pass

    @abstractmethod
    def _calc_sigma_t_p1(self):
        """Berechnet `sigma_t` und `delta_sigma` für die nächsten Schritte."""
        pass