# Template
## Load modules & Check PyTorch


```python
# Import modules
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd

# data processing
import random as r
from sys import exit
import numpy as np
import matplotlib.pyplot as plt
```

## Parameters


```python
# Debugger: Aktiviert
debug_mode = True
normalize = False
```

## Preloaded Modules


```python
def dict_to_markdown_table(data: dict, title: str = "Datenübersicht", include_index: bool = True, round_digits: int = 4):
    """
    Wandelt ein Dictionary mit Listenwerten in eine Markdown-Tabelle für Jupyter Notebooks um.
    
    - Schlüssel werden als Header genutzt
    - Erste Spalte ist ein Index, falls `include_index=True`
    - Einzelwerte werden als separate Tabelle unterhalb dargestellt
    - Zahlenwerte werden auf eine einstellbare Anzahl an Nachkommastellen gerundet

    :param data: Dictionary mit Key-Value-Paaren
    :param title: Überschrift für die Tabelle
    :param include_index: Falls True, wird eine Index-Spalte erstellt
    :param round_digits: Anzahl der Nachkommastellen, auf die Werte gerundet werden sollen
    :return: Markdown-String zur Anzeige in Jupyter
    """
    
    # Hilfsfunktion zum Runden von Zahlen
    def round_value(val):
        if isinstance(val, (int, float)):
            return round(val, round_digits)
        return val

    # Listen und einzelne Werte trennen
    list_data = {k: v for k, v in data.items() if isinstance(v, list)}
    single_values = {k: v for k, v in data.items() if not isinstance(v, list)}

    # Falls es Listen gibt, erstelle eine Tabelle mit Index
    if list_data:
        max_len = max(len(v) for v in list_data.values())  # Längste Liste bestimmen

        # Tabellenkopf
        md_table = f"### {title}\n\n"
        md_table += "| " + ("Index | " if include_index else "") + " | ".join(list_data.keys()) + " |\n"
        md_table += "|-" + ("-|" if include_index else "") + "-|".join(["-" * len(k) for k in list_data.keys()]) + "-|\n"

        # Datenzeilen
        for i in range(max_len):
            row = [str(i)] if include_index else []  # Index hinzufügen (optional)
            for key in list_data:
                if i < len(list_data[key]):
                    row.append(str(round_value(list_data[key][i])))
                else:
                    row.append("")  # Leere Werte für ungleich lange Listen
            md_table += "| " + " | ".join(row) + " |\n"
    
    else:
        md_table = ""

    # Einzelwerte als extra Tabelle darstellen
    if single_values:
        md_table += "\n\n#### Einzelwerte\n\n"
        md_table += "| " + " | ".join(single_values.keys()) + " |\n"
        md_table += "|-" + "-|".join(["-" * len(k) for k in single_values.keys()]) + "-|\n"
        md_table += "| " + " | ".join(map(lambda v: str(round_value(v)), single_values.values())) + " |\n"

    return Markdown(md_table)


def display_data_loss_table(data_dict, delta_sigma_pred, max_i):
    """
    Erstellt eine Markdown-Tabelle zur übersichtlichen Darstellung von Datenverlust.
    
    Unterstützt sowohl Python-Listen als auch NumPy-Arrays.
    
    :param data_dict: Dictionary mit `sigma_t` und `delta_sigma` (Listen oder np.arrays)
    :param delta_sigma_pred: Vorhergesagte Werte für `delta_sigma` (Liste oder np.array)
    :param max_i: Anzahl der Werte, die in der Tabelle angezeigt werden sollen
    """
    
    # Sicherstellen, dass `sigma_t` und `delta_sigma` existieren
    if "sigma_t" not in data_dict or "delta_sigma" not in data_dict or delta_sigma_pred is None:
        print("Fehler: `data_dict` oder `delta_sigma_pred` ist nicht korrekt definiert!")
        return

    # Konvertiere alle Werte zu Listen (falls sie NumPy-Arrays sind)
    def to_list(arr):
        return arr.tolist() if isinstance(arr, np.ndarray) else arr

    total_epsilon = to_list(data_dict["total_epsilon"])
    delta_epsilon = to_list(data_dict["delta_epsilon"])
    sigma_t = to_list(data_dict["sigma_t"])
    delta_sigma_true = to_list(data_dict["delta_sigma"])
    delta_sigma_pred = to_list(delta_sigma_pred)  # Falls `delta_sigma_pred` ein 2D-Array ist
    
    # Überprüfen, ob die Längen konsistent sind
    min_len = min(len(total_epsilon), len(sigma_t), len(delta_epsilon), len(delta_sigma_true), len(delta_sigma_pred), max_i)

    # Erstelle eine Tabelle für die übersichtliche Darstellung
    data_loss_table = {
        "total_epsilon" : list(total_epsilon[:min_len]), 
        "delta_epsilon" : list(delta_epsilon[:min_len]), 
        "sigma_t" : list(sigma_t[:min_len]), 
        "True delta_sigma": list(delta_sigma_true[:min_len]),
        "Predicted delta_sigma": list(delta_sigma_pred[:min_len]),
        "Test-Loss (True - Predicted)": list(np.round(np.array(delta_sigma_true[:min_len]) - np.array(delta_sigma_pred[:min_len]), 5))
    }

    # Markdown-Tabelle für bessere Darstellung in Jupyter
    display(dict_to_markdown_table(data_loss_table, title=f"Data-Loss bis sigma_{min_len-1}", include_index=True))
```

## Check for use of CONDA if available


```python
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

if device.type == 'cpu':
    device_num = 0
    print('No GPU available.')
else:
    device_num = torch.cuda.device_count()
    print('Device:', device, '-- Number of devices:', device_num)
```

    No GPU available.
    

# Recurrent Neural Networks

Just as people do not have to think again each time about the things they have already learned, it is also possible to teach neural networks to recall knowledge they were being taught. This is done in so-called Recurrent Neural Networks (RNNs) with loops inside, which allow information to be retained. Currently the most used architectures of RNNs are Long short-term memory (LSTM) networks. LSTMs are RNNs that overcome the problem of long-term dependencies and thus have achieved the most state-of-the-art results in this area. In this exercise we will look at how to use LSTMs to predict future values using time series data sets.

## Data processing


```python
import random

class Oedometer:
    def __init__(self, e_0: float = 1.00, C_c: float = 0.005, delta_epsilon: float = 0.0005, 
                 sigma_t: float = 1.00, max_n: int = 50, rand_epsilon:bool=False, **kwargs):
        self.max_n = max_n

        # Standardwerte als Listen setzen
        self.e_0 = [e_0]
        self.C_c = [C_c]
        self.sigma_t = [sigma_t]
        self.delta_epsilon = []
        self.total_epsilon = [0]

        # Initiale Listen für Berechnungen
        self.sigma_t = [sigma_t]
        self.delta_sigma = []
        self.e_s = []
        self.delta_epsilon = [delta_epsilon]
        
        # Dynamische Zuweisung von kwargs, falls vorhanden
        for key, value in kwargs.items():
            if hasattr(self, key):  # Nur vorhandene Attribute setzen
                setattr(self, key, [value])
        
        # Berechnungen durchführen
        self.__calc_sigma_t_p1()

        # Listenlängen anpassen
        self.__adjust_list_lengths()
        self.__calc_total_epsilon()

    def __adjust_list_lengths(self):
        """ Passt ALLE Listen-Attribute an `max_n` an. """
        attributes = ['e_0', 'C_c', 'delta_epsilon', 'sigma_t', 'sigma_t', 'delta_sigma', 'e_s']
        for attr in attributes:
            value_list = getattr(self, attr, [])
            current_length = len(value_list)

            if current_length > self.max_n:
                setattr(self, attr, value_list[:self.max_n])  # Kürzen
            elif current_length < self.max_n:
                setattr(self, attr, value_list + [value_list[-1] if value_list else 0] * (self.max_n - current_length))  # Auffüllen
    
    def __calc_total_epsilon(self):
        for i in range(len(self.delta_epsilon)-1):
            self.total_epsilon.append(self.total_epsilon[i] + self.delta_epsilon[i])            
    
    def __calc_e_s(self, sigma_t):
        """ Berechnet `e_s` aus `sigma_t`. """
        e_s = (1 + self.e_0[0]) / self.C_c[0] * sigma_t
        self.e_s.append(e_s)
        return e_s

    def __calc_sigma_t_p1(self):
        """ Berechnet `sigma_t` und `delta_sigma` für die nächsten Schritte. """
        for i in range(self.max_n):  # -1, weil sigma_t bereits gesetzt ist
            e_s = self.__calc_e_s(self.sigma_t[i])
            delta_sigma = e_s * self.delta_epsilon[0]
            sigma = self.sigma_t[i] + delta_sigma
            self.sigma_t.append(sigma)
            self.delta_sigma.append(delta_sigma)

def plot_input():
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(data_dict_raw['sigma_t'], data_dict_raw['delta_sigma'], marker='o', linestyle='-', label='Sigma_0 = 1')
    
    # Formatting
    plt.xlabel('sigma_t')
    plt.ylabel('delta_simga')
    plt.title('Sigma_0 in relation to Sigma_1')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
```


```python
i = 100

oedo_para = {
    'max_n': 1, 
    'e_0': 1.0, 
    'C_c': 0.005,   
    'total_epsilon': 0,
    'e_s': 400.0
}

# Vorbereitung Tensoren
sigma_t = np.random.choice(range(1, 10000), size=i, replace=False)
delta_sigma = []
delta_epsilon = np.repeat(np.array(np.float64(0.0005)), i)
# delta_epsilon = np.random.uniform(0.0001, 0.001, size=i)

for i in range(i):
    oedo_para['sigma_t'] = sigma_t[i]
    oedo_para['delta_epsilon'] = delta_epsilon[i]
    oedo = Oedometer(**oedo_para)
    delta_sigma.append(round(oedo.delta_sigma[0], 2))
    

print('Anzahl Elemente delta_epsilon: ' + str(len(delta_epsilon)))
print('Anzahl Elemente sigma_t: ' + str(len(sigma_t)))
print('Anzahl Elemente delta_sigma: ' + str(len(delta_sigma)))
```

    Anzahl Elemente delta_epsilon: 100
    Anzahl Elemente sigma_t: 100
    Anzahl Elemente delta_sigma: 100
    


```python
data = [['sigma_t'] + sigma_t.tolist(), ['delta_sigma'] + delta_sigma, ['delta_epsilon'] + delta_epsilon.tolist()]
import tabulate
table = tabulate.tabulate(data, tablefmt='html')
table
```




<table>
<tbody>
<tr><td>sigma_t      </td><td style="text-align: right;">114     </td><td style="text-align: right;">7885     </td><td style="text-align: right;">1779     </td><td style="text-align: right;">1716     </td><td style="text-align: right;">1137     </td><td style="text-align: right;">2943     </td><td style="text-align: right;">5590     </td><td style="text-align: right;">2610     </td><td style="text-align: right;">3318     </td><td style="text-align: right;">3570     </td><td style="text-align: right;">9981     </td><td style="text-align: right;">9921     </td><td style="text-align: right;">6895     </td><td style="text-align: right;">1598     </td><td style="text-align: right;">3732     </td><td style="text-align: right;">891     </td><td style="text-align: right;">5188     </td><td style="text-align: right;">8573     </td><td style="text-align: right;">151     </td><td style="text-align: right;">4371     </td><td style="text-align: right;">5049     </td><td style="text-align: right;">1260     </td><td style="text-align: right;">7843     </td><td style="text-align: right;">5258     </td><td style="text-align: right;">7447     </td><td style="text-align: right;">650     </td><td style="text-align: right;">4020     </td><td style="text-align: right;">7478     </td><td style="text-align: right;">9785     </td><td style="text-align: right;">7286     </td><td style="text-align: right;">296     </td><td style="text-align: right;">7396     </td><td style="text-align: right;">836     </td><td style="text-align: right;">478     </td><td style="text-align: right;">9821     </td><td style="text-align: right;">4850     </td><td style="text-align: right;">2393     </td><td style="text-align: right;">6188     </td><td style="text-align: right;">2229     </td><td style="text-align: right;">2099     </td><td style="text-align: right;">2198     </td><td style="text-align: right;">4037     </td><td style="text-align: right;">5689     </td><td style="text-align: right;">1642     </td><td style="text-align: right;">4427     </td><td style="text-align: right;">5341     </td><td style="text-align: right;">9452     </td><td style="text-align: right;">3155     </td><td style="text-align: right;">7942     </td><td style="text-align: right;">7613     </td><td style="text-align: right;">4055     </td><td style="text-align: right;">3660     </td><td style="text-align: right;">3714     </td><td style="text-align: right;">4824     </td><td style="text-align: right;">7765     </td><td style="text-align: right;">170     </td><td style="text-align: right;">1267     </td><td style="text-align: right;">8340     </td><td style="text-align: right;">8867     </td><td style="text-align: right;">1741     </td><td style="text-align: right;">8077     </td><td style="text-align: right;">2136     </td><td style="text-align: right;">8519     </td><td style="text-align: right;">1886     </td><td style="text-align: right;">4147     </td><td style="text-align: right;">7831     </td><td style="text-align: right;">8691     </td><td style="text-align: right;">2673     </td><td style="text-align: right;">4930     </td><td style="text-align: right;">9376     </td><td style="text-align: right;">7990     </td><td style="text-align: right;">1030     </td><td style="text-align: right;">7403     </td><td style="text-align: right;">5991     </td><td style="text-align: right;">8356     </td><td style="text-align: right;">6676     </td><td style="text-align: right;">5075     </td><td style="text-align: right;">2776     </td><td style="text-align: right;">471     </td><td style="text-align: right;">496     </td><td style="text-align: right;">6444     </td><td style="text-align: right;">6358     </td><td style="text-align: right;">6339     </td><td style="text-align: right;">6265     </td><td style="text-align: right;">6781     </td><td style="text-align: right;">7770     </td><td style="text-align: right;">227     </td><td style="text-align: right;">4388     </td><td style="text-align: right;">1866     </td><td style="text-align: right;">3485     </td><td style="text-align: right;">6476     </td><td style="text-align: right;">4001     </td><td style="text-align: right;">5041     </td><td style="text-align: right;">778     </td><td style="text-align: right;">8087     </td><td style="text-align: right;">8108     </td><td style="text-align: right;">5038     </td><td style="text-align: right;">1371     </td><td style="text-align: right;">7805     </td><td style="text-align: right;">6141     </td></tr>
<tr><td>delta_sigma  </td><td style="text-align: right;"> 22.8   </td><td style="text-align: right;">1577     </td><td style="text-align: right;"> 355.8   </td><td style="text-align: right;"> 343.2   </td><td style="text-align: right;"> 227.4   </td><td style="text-align: right;"> 588.6   </td><td style="text-align: right;">1118     </td><td style="text-align: right;"> 522     </td><td style="text-align: right;"> 663.6   </td><td style="text-align: right;"> 714     </td><td style="text-align: right;">1996.2   </td><td style="text-align: right;">1984.2   </td><td style="text-align: right;">1379     </td><td style="text-align: right;"> 319.6   </td><td style="text-align: right;"> 746.4   </td><td style="text-align: right;">178.2   </td><td style="text-align: right;">1037.6   </td><td style="text-align: right;">1714.6   </td><td style="text-align: right;"> 30.2   </td><td style="text-align: right;"> 874.2   </td><td style="text-align: right;">1009.8   </td><td style="text-align: right;"> 252     </td><td style="text-align: right;">1568.6   </td><td style="text-align: right;">1051.6   </td><td style="text-align: right;">1489.4   </td><td style="text-align: right;">130     </td><td style="text-align: right;"> 804     </td><td style="text-align: right;">1495.6   </td><td style="text-align: right;">1957     </td><td style="text-align: right;">1457.2   </td><td style="text-align: right;"> 59.2   </td><td style="text-align: right;">1479.2   </td><td style="text-align: right;">167.2   </td><td style="text-align: right;"> 95.6   </td><td style="text-align: right;">1964.2   </td><td style="text-align: right;"> 970     </td><td style="text-align: right;"> 478.6   </td><td style="text-align: right;">1237.6   </td><td style="text-align: right;"> 445.8   </td><td style="text-align: right;"> 419.8   </td><td style="text-align: right;"> 439.6   </td><td style="text-align: right;"> 807.4   </td><td style="text-align: right;">1137.8   </td><td style="text-align: right;"> 328.4   </td><td style="text-align: right;"> 885.4   </td><td style="text-align: right;">1068.2   </td><td style="text-align: right;">1890.4   </td><td style="text-align: right;"> 631     </td><td style="text-align: right;">1588.4   </td><td style="text-align: right;">1522.6   </td><td style="text-align: right;"> 811     </td><td style="text-align: right;"> 732     </td><td style="text-align: right;"> 742.8   </td><td style="text-align: right;"> 964.8   </td><td style="text-align: right;">1553     </td><td style="text-align: right;"> 34     </td><td style="text-align: right;"> 253.4   </td><td style="text-align: right;">1668     </td><td style="text-align: right;">1773.4   </td><td style="text-align: right;"> 348.2   </td><td style="text-align: right;">1615.4   </td><td style="text-align: right;"> 427.2   </td><td style="text-align: right;">1703.8   </td><td style="text-align: right;"> 377.2   </td><td style="text-align: right;"> 829.4   </td><td style="text-align: right;">1566.2   </td><td style="text-align: right;">1738.2   </td><td style="text-align: right;"> 534.6   </td><td style="text-align: right;"> 986     </td><td style="text-align: right;">1875.2   </td><td style="text-align: right;">1598     </td><td style="text-align: right;"> 206     </td><td style="text-align: right;">1480.6   </td><td style="text-align: right;">1198.2   </td><td style="text-align: right;">1671.2   </td><td style="text-align: right;">1335.2   </td><td style="text-align: right;">1015     </td><td style="text-align: right;"> 555.2   </td><td style="text-align: right;"> 94.2   </td><td style="text-align: right;"> 99.2   </td><td style="text-align: right;">1288.8   </td><td style="text-align: right;">1271.6   </td><td style="text-align: right;">1267.8   </td><td style="text-align: right;">1253     </td><td style="text-align: right;">1356.2   </td><td style="text-align: right;">1554     </td><td style="text-align: right;"> 45.4   </td><td style="text-align: right;"> 877.6   </td><td style="text-align: right;"> 373.2   </td><td style="text-align: right;"> 697     </td><td style="text-align: right;">1295.2   </td><td style="text-align: right;"> 800.2   </td><td style="text-align: right;">1008.2   </td><td style="text-align: right;">155.6   </td><td style="text-align: right;">1617.4   </td><td style="text-align: right;">1621.6   </td><td style="text-align: right;">1007.6   </td><td style="text-align: right;"> 274.2   </td><td style="text-align: right;">1561     </td><td style="text-align: right;">1228.2   </td></tr>
<tr><td>delta_epsilon</td><td style="text-align: right;">  0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">  0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">  0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">  0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">  0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">  0.0005</td><td style="text-align: right;">  0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">  0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">  0.0005</td><td style="text-align: right;">  0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">  0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">  0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td><td style="text-align: right;">   0.0005</td></tr>
</tbody>
</table>




```python
from sklearn.preprocessing import MinMaxScaler

data = {
    'input1': sigma_t,
    'input2': delta_epsilon,
    'output': delta_sigma
}

df = pd.DataFrame(data)

# Normalisierung
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(df)

# Zurück in DataFrame
df_scaled = pd.DataFrame(scaled_data, columns=df.columns)
df_scaled
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>input1</th>
      <th>input2</th>
      <th>output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.787575</td>
      <td>0.0</td>
      <td>0.787575</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.168744</td>
      <td>0.0</td>
      <td>0.168744</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.162359</td>
      <td>0.0</td>
      <td>0.162359</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.103679</td>
      <td>0.0</td>
      <td>0.103679</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>0.810175</td>
      <td>0.0</td>
      <td>0.810175</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.499037</td>
      <td>0.0</td>
      <td>0.499037</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.127394</td>
      <td>0.0</td>
      <td>0.127394</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.779467</td>
      <td>0.0</td>
      <td>0.779467</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.610824</td>
      <td>0.0</td>
      <td>0.610824</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 3 columns</p>
</div>



## Auszug aus Tutorial 6

    # Create sequences
    # Function to create sequences of input (passenger of train_window months) and ...
    # ... output passenger of train_window+1 month
    
    def create_inout_sequences(input_data, tw):
        inout_seq = []
        L = len(input_data)
        for i in range(L-tw):
            train_seq = input_data[i:i+tw]
            train_label = input_data[i+tw:i+tw+1]
            inout_seq.append((train_seq ,train_<label))
        return inout_seq
    
    # Set an appropiate Train window
    # TODO start**
    train_window = 24
    
    # Create sequences 
    trva_inout_seq = create_inout_sequences(trva_data_normalized, train_window)
    
    # Print the first 3 sequences
    print(trva_inout_seq[:3])


```python
import numpy as np
import torch

lookback=1

def create_dataset(df, lookback=1):
    """
    Erzeugt die Eingabe- und Ziel-Datensätze für ein LSTM.
    
    lookback definiert hier, wie viele vergangene Zeitschritte
    (Train Window / Sequences im Tutorial) das Modell als Input
    für jede Vorhersage erhält. Dadurch entsteht die nötige
    3D-Form (batch_size, seq_length, input_size), die ein LSTM erwartet,
    um zeitliche Abhängigkeiten zu lernen.
    """
    X, y = [], []
    for i in range(len(df) - lookback):
        # Wir nehmen hier 'lookback' Zeilen als eine Sequenz,
        # damit das LSTM-Modell weiß, wie weit es in die Vergangenheit
        # zurückschauen muss. Ohne diese Sequenz-Dimension (seq_length)
        # wäre die Eingabe nur eine 2D-Matrix und das LSTM könnte
        # keine zeitlichen Muster erfassen.
        seq = df.iloc[i:i+lookback, :-1].values
        X.append(seq)
        
        # Zielwert: der Wert direkt nach dieser Sequenz
        y.append(df.iloc[i+lookback, -1])
    
    # Rückgabe als Tensor mit Form (batch_size, seq_length, input_size) und (batch_size,)
    return torch.tensor(np.array(X), dtype=torch.float32), \
           torch.tensor(np.array(y), dtype=torch.float32)

X, y = create_dataset(df_scaled, lookback)
X_input_all = df_scaled.iloc[lookback:, :-1].to_numpy(dtype=np.float32)

print(np.shape(X))  # z.B. (4997, 3, 2)

# Falls lookback=1, wäre X anfangs (5000, 1, 2).
# Mit squeeze() könnte man die mittlere Dimension entfernen,
# aber für LSTM brauchen wir sie genau so:
# X = X.squeeze()
print(np.shape(X))
```

    torch.Size([99, 1, 2])
    torch.Size([99, 1, 2])
    


```python
# --- Aufteilen in Training (2/3) und Test (1/3) ---
train_size = int(X.shape[0] * 2/3)
X_train = X[:train_size]
y_train = y[:train_size].view(-1, 1)  # als (batch_size, 1)
X_test  = X[train_size:]
y_test  = y[train_size:].view(-1, 1)

X_input_train = X_input_all[:train_size]
X_input_test  = X_input_all[train_size:]

print("Shapes:")
print("  X_train:", X_train.shape, "y_train:", y_train.shape, "X_input_train:", X_input_train.shape)
print("  X_test: ", X_test.shape,  "y_test: ", y_test.shape,  "X_input_test: ",  X_input_test.shape)
```

    Shapes:
      X_train: torch.Size([66, 1, 2]) y_train: torch.Size([66, 1]) X_input_train: (66, 2)
      X_test:  torch.Size([33, 1, 2]) y_test:  torch.Size([33, 1]) X_input_test:  (33, 2)
    


```python
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

# Modell initialisieren
model = LSTMModel(input_size=2, hidden_size=50, num_layers=2, output_size=1)

```


```python
# Hyperparameter
input_size = X_train.shape[2]  # Anzahl der Input-Features
hidden_size = 50  # Die Anzahl der Neuronen im LSTM
num_layers = 3  # Anzahl der LSTM-Schichten
output_size = 1  # Output ist eine einzelne Zahl (z. B. Vorhersage)

# Modell, Loss und Optimizer
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = torch.nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
```


```python
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Frühzeitige Beendigung: Überwache den Verlust auf den Validierungsdaten
best_val_loss = float('inf')  # Initialer hoher Verlust
patience = 20  # Anzahl der Epochen ohne Verbesserung, bevor das Training gestoppt wird
patience_counter = 0  # Zähler für die Anzahl der Epochen ohne Verbesserung

# Liste zum Speichern der Loss-Werte
losses = []

# Batches erstellen
batch_size = 10

# Erstelle TensorDataset und DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_test, y_test)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Improvement Block Aktivierung (True/False)
improvement_block = False

# Training
epochs = 100
for epoch in range(epochs):
    model.train()  # Setze das Modell in den Trainingsmodus
    running_loss = 0.0
    
    # Schleife über Batches
    for i, (x_batch, y_batch) in enumerate(train_dataloader):
        optimizer.zero_grad()

        # Modellvorhersage
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)  # Verlust auf den Trainingsdaten
        loss.backward()
        optimizer.step()

        # Verlust sammeln
        running_loss += loss.item()

    # Durchschnittlichen Trainingsverlust berechnen
    avg_train_loss = running_loss / len(train_dataloader)
    losses.append(avg_train_loss)

    # Frühzeitige Beendigung (improvement block) aktivieren
    if improvement_block:
        # Validierungsverlust berechnen
        model.eval()  # Modell in den Evaluierungsmodus setzen
        val_loss = 0.0
        with torch.no_grad():
            for x_batch_val, y_batch_val in val_dataloader:
                val_outputs = model(x_batch_val)
                loss = criterion(val_outputs, y_batch_val)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)

        # Frühzeitige Beendigung überwachen
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  # Reset der Geduld
        else:
            patience_counter += 1

        # Wenn keine Verbesserung erfolgt, abbrechen
        if patience_counter >= patience:
            print(f"Frühzeitige Beendigung nach {epoch + 1} Epochen wegen fehlender Verbesserung.")
            break

        # Ausgabe der Verlustinformationen
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}')
    else:
        # Ausgabe der Verlustinformationen ohne Validierung
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}')
```

    Epoch 10/100, Train Loss: 0.090912
    Epoch 20/100, Train Loss: 0.092178
    Epoch 30/100, Train Loss: 0.095494
    Epoch 40/100, Train Loss: 0.091790
    Epoch 50/100, Train Loss: 0.089686
    Epoch 60/100, Train Loss: 0.090200
    Epoch 70/100, Train Loss: 0.086385
    Epoch 80/100, Train Loss: 0.087436
    Epoch 90/100, Train Loss: 0.089451
    Epoch 100/100, Train Loss: 0.087567
    


```python
import matplotlib.pyplot as plt
# Plot der Losskurve
plt.plot(range(epochs), losses, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()
```


    
![png](README_files/README_20_0.png)
    



```python
def evaluate_model(model, X_eval, y_eval, X_input_eval, scaler, max_print=5):
    """
    Führt das trainierte Modell auf X_eval aus,
    berechnet den MSE auf dem normierten Datensatz und
    zeigt für die ersten `max_print` Samples:
      - die unskalierten Lookback-Input-Sequenzen
      - den unskalierten aktuellen Input (zum Real-Wert)
      - die unskalierten Vorhersagen
      - die unskalierten Realwerte
    """

    model.eval()
    with torch.no_grad():
        # Vorhersagen im skalierten Raum
        preds_scaled = model(X_eval).cpu().numpy().ravel()
        y_scaled     = y_eval.cpu().numpy().ravel()
        mse_scaled   = np.mean((preds_scaled - y_scaled)**2)
        print(f"Evaluierung auf {X_eval.shape[0]} Proben → MSE (skaliert): {mse_scaled:.6f}\n")

        # Min/Max für jeden Kanal (input1, input2, output)
        mins = scaler.data_min_    # [min_input1, min_input2, min_output]
        maxs = scaler.data_max_    # [max_input1, max_input2, max_output]

        for i in range(min(max_print, len(preds_scaled))):
            # --- Unskalierte Lookback-Sequenz ---
            seq_scaled = X_eval[i].cpu().numpy()           # shape (lookback, 2)
            dummy_out  = np.zeros((seq_scaled.shape[0],1))
            stacked    = np.hstack([seq_scaled, dummy_out])  # (lookback, 3)
            seq_unscaled = scaler.inverse_transform(stacked)[:, :2]  # nur Input-Spalten

            # --- Unskalierter aktueller Input ---
            curr_scaled = X_input_eval[i]                  # shape (2,)
            curr_unscaled = [
                curr_scaled[j] * (maxs[j] - mins[j]) + mins[j]
                for j in range(2)
            ]

            # --- Unskalierte Vorhersage und Realwert ---
            pred_unscaled = preds_scaled[i] * (maxs[2] - mins[2]) + mins[2]
            real_unscaled = y_scaled[i]      * (maxs[2] - mins[2]) + mins[2]

            # --- Ausgabe ---
            print(f"Sample {i}:")
            print(f"  Lookback-Sequenz (unskaliert):")
            for t, vals in enumerate(seq_unscaled[::-1], 1):
                print(f"    t-{t}: {vals.tolist()}")
            print(f"  Aktueller Input (unskaliert): {curr_unscaled}")
            print(f"  Vorhersage = {pred_unscaled:.4f}, Real = {real_unscaled:.4f}\n")

    return mse_scaled

mse_test = evaluate_model(model, X_test, y_test, X_input_test, scaler)

```

    Evaluierung auf 33 Proben → MSE (skaliert): 0.073774
    
    Sample 0:
      Lookback-Sequenz (unskaliert):
        t-1: [8690.999723553658, 0.0005]
      Aktueller Input (unskaliert): [np.float64(2672.9999985694885), np.float64(0.0005)]
      Vorhersage = 907.3828, Real = 534.6000
    
    Sample 1:
      Lookback-Sequenz (unskaliert):
        t-1: [2672.9999985694885, 0.0005]
      Aktueller Input (unskaliert): [np.float64(4929.999993056059), np.float64(0.0005)]
      Vorhersage = 964.3512, Real = 986.0000
    
    Sample 2:
      Lookback-Sequenz (unskaliert):
        t-1: [4929.999993056059, 0.0005]
      Aktueller Input (unskaliert): [np.float64(9376.000189483166), np.float64(0.0005)]
      Vorhersage = 942.3276, Real = 1875.2000
    
    Sample 3:
      Lookback-Sequenz (unskaliert):
        t-1: [9376.000189483166, 0.0005]
      Aktueller Input (unskaliert): [np.float64(7990.0000675320625), np.float64(0.0005)]
      Vorhersage = 901.2668, Real = 1598.0000
    
    Sample 4:
      Lookback-Sequenz (unskaliert):
        t-1: [7990.000067532063, 0.0005]
      Aktueller Input (unskaliert): [np.float64(1030.00002066046), np.float64(0.0005)]
      Vorhersage = 913.7226, Real = 206.0000
    
    


```python
import numpy as np
import matplotlib.pyplot as plt

def plot_residuals(y_true, y_pred, max_points=None):
    """
    Plots the residuals (y_pred - y_true) for each sample.
    If max_points is set, only the first max_points residuals are shown.
    Converts lists to numpy arrays for subtraction.
    """
    # Ensure numpy arrays
    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)
    residuals = y_pred_arr - y_true_arr

    if max_points is not None:
        residuals = residuals[:max_points]
    plt.figure()
    plt.scatter(range(len(residuals)), residuals)
    plt.xlabel('Sample Index')
    plt.ylabel('Residual (Prediction - Real)')
    plt.title('Residual Plot')
    plt.show()

def plot_pred_vs_real(y_true, y_pred, max_points=None):
    """
    Plots predicted vs. real values.
    If max_points is set, only the first max_points points are plotted.
    Converts lists to numpy arrays for plotting.
    """
    # Ensure numpy arrays
    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)

    if max_points is not None:
        y_true_arr = y_true_arr[:max_points]
        y_pred_arr = y_pred_arr[:max_points]
    plt.figure()
    plt.plot(y_true_arr, label='Real')
    plt.plot(y_pred_arr, label='Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Prediction vs. Real')
    plt.legend()
    plt.show()

# Usage example:
# y_true = [...]  # list of real values
# y_pred = [...]  # list of predicted values
# plot_residuals(y_true, y_pred)
# plot_pred_vs_real(y_true, y_pred)


y_true = []
y_pred = []

model.eval()
with torch.no_grad():
    preds_scaled = model(X_test).cpu().numpy().ravel()
    y_scaled     = y_test.cpu().numpy().ravel()

    # Min/Max für jeden Kanal (input1, input2, output)
    mins = scaler.data_min_    # [min_input1, min_input2, min_output]
    maxs = scaler.data_max_    # [max_input1, max_input2, max_output]
for i in range(len(preds_scaled)):
    pred_unscaled = preds_scaled[i] * (maxs[2] - mins[2]) + mins[2]
    real_unscaled = y_scaled[i]      * (maxs[2] - mins[2]) + mins[2]
    y_true.append(real_unscaled)
    y_pred.append(pred_unscaled)

# print(y_true[0:10])
# print(y_pred[0:10])
```


```python
plot_residuals(y_true, y_pred)
```


    
![png](README_files/README_23_0.png)
    



```python
plot_pred_vs_real(y_true, y_pred)
```


    
![png](README_files/README_24_0.png)
    



```python
def print_table_results():
    model.eval()
    sigma_t_print = []
    delta_epsilon_print = []
    delta_sigma_true_print = []
    delta_sigma_pred_print = []
    with torch.no_grad():
        # Vorhersagen im skalierten Raum
        preds_scaled = model(X_test).cpu().numpy().ravel()
        y_scaled     = y_test.cpu().numpy().ravel()
    
        # Min/Max für jeden Kanal (input1, input2, output)
        mins = scaler.data_min_    # [min_input1, min_input2, min_output]
        maxs = scaler.data_max_    # [max_input1, max_input2, max_output]
    
        for i in range(len(preds_scaled)):
            # --- Unskalierte Lookback-Sequenz ---
            seq_scaled = X_test[i].cpu().numpy()           # shape (lookback, 2)
            dummy_out  = np.zeros((seq_scaled.shape[0],1))
            stacked    = np.hstack([seq_scaled, dummy_out])  # (lookback, 3)
            seq_unscaled = scaler.inverse_transform(stacked)[:, :2]  # nur Input-Spalten
    
            # --- Unskalierter aktueller Input ---
            curr_scaled = X_input_test[i]                  # shape (2,)
            curr_unscaled = [
                curr_scaled[j] * (maxs[j] - mins[j]) + mins[j]
                for j in range(2)
            ]
    
            # --- Unskalierte Vorhersage und Realwert ---
            pred_unscaled = preds_scaled[i] * (maxs[2] - mins[2]) + mins[2]
            real_unscaled = y_scaled[i]      * (maxs[2] - mins[2]) + mins[2]
            
            sigma_t_print.append(seq_unscaled[0][0])
            delta_epsilon_print.append(seq_unscaled[0][1])
            delta_sigma_true_print.append(pred_unscaled)
            delta_sigma_pred_print.append(real_unscaled)

        df = {
            'sigma_t' : sigma_t_print,
            'delta_epsilon' : delta_epsilon_print,
            'delta_sigma_true' : delta_sigma_true_print,
            'delta_sigma_pred' : delta_sigma_pred_print,
            'Pred - True' : [a_i - b_i for a_i, b_i in zip(delta_sigma_pred_print, delta_sigma_true_print)]
        }
        return pd.DataFrame(df)
print_table_results()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sigma_t</th>
      <th>delta_epsilon</th>
      <th>delta_sigma_true</th>
      <th>delta_sigma_pred</th>
      <th>Pred - True</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8690.999724</td>
      <td>0.0005</td>
      <td>907.382834</td>
      <td>534.600000</td>
      <td>-372.782835</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2672.999999</td>
      <td>0.0005</td>
      <td>964.351161</td>
      <td>985.999999</td>
      <td>21.648838</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4929.999993</td>
      <td>0.0005</td>
      <td>942.327632</td>
      <td>1875.200038</td>
      <td>932.872406</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9376.000189</td>
      <td>0.0005</td>
      <td>901.266808</td>
      <td>1598.000014</td>
      <td>696.733205</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7990.000068</td>
      <td>0.0005</td>
      <td>913.722581</td>
      <td>206.000004</td>
      <td>-707.722577</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1030.000021</td>
      <td>0.0005</td>
      <td>980.844547</td>
      <td>1480.600046</td>
      <td>499.755498</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7403.000228</td>
      <td>0.0005</td>
      <td>919.093578</td>
      <td>1198.199990</td>
      <td>279.106412</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5990.999952</td>
      <td>0.0005</td>
      <td>932.241156</td>
      <td>1671.200013</td>
      <td>738.958858</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8356.000067</td>
      <td>0.0005</td>
      <td>910.402414</td>
      <td>1335.199966</td>
      <td>424.797552</td>
    </tr>
    <tr>
      <th>9</th>
      <td>6675.999830</td>
      <td>0.0005</td>
      <td>925.823071</td>
      <td>1014.999972</td>
      <td>89.176900</td>
    </tr>
    <tr>
      <th>10</th>
      <td>5074.999858</td>
      <td>0.0005</td>
      <td>940.938907</td>
      <td>555.199986</td>
      <td>-385.738920</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2775.999931</td>
      <td>0.0005</td>
      <td>963.329892</td>
      <td>94.200003</td>
      <td>-869.129889</td>
    </tr>
    <tr>
      <th>12</th>
      <td>471.000014</td>
      <td>0.0005</td>
      <td>986.540010</td>
      <td>99.200000</td>
      <td>-887.340010</td>
    </tr>
    <tr>
      <th>13</th>
      <td>495.999998</td>
      <td>0.0005</td>
      <td>986.284413</td>
      <td>1288.799962</td>
      <td>302.515549</td>
    </tr>
    <tr>
      <th>14</th>
      <td>6443.999811</td>
      <td>0.0005</td>
      <td>927.988467</td>
      <td>1271.599951</td>
      <td>343.611484</td>
    </tr>
    <tr>
      <th>15</th>
      <td>6357.999753</td>
      <td>0.0005</td>
      <td>928.793308</td>
      <td>1267.799996</td>
      <td>339.006688</td>
    </tr>
    <tr>
      <th>16</th>
      <td>6338.999980</td>
      <td>0.0005</td>
      <td>928.971273</td>
      <td>1252.999981</td>
      <td>324.028708</td>
    </tr>
    <tr>
      <th>17</th>
      <td>6264.999903</td>
      <td>0.0005</td>
      <td>929.665018</td>
      <td>1356.200050</td>
      <td>426.535032</td>
    </tr>
    <tr>
      <th>18</th>
      <td>6781.000249</td>
      <td>0.0005</td>
      <td>924.845853</td>
      <td>1554.000006</td>
      <td>629.154153</td>
    </tr>
    <tr>
      <th>19</th>
      <td>7770.000030</td>
      <td>0.0005</td>
      <td>915.728949</td>
      <td>45.399999</td>
      <td>-870.328950</td>
    </tr>
    <tr>
      <th>20</th>
      <td>226.999996</td>
      <td>0.0005</td>
      <td>989.039045</td>
      <td>877.600016</td>
      <td>-111.439029</td>
    </tr>
    <tr>
      <th>21</th>
      <td>4388.000081</td>
      <td>0.0005</td>
      <td>947.547130</td>
      <td>373.199995</td>
      <td>-574.347135</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1865.999974</td>
      <td>0.0005</td>
      <td>972.405392</td>
      <td>697.000013</td>
      <td>-275.405379</td>
    </tr>
    <tr>
      <th>23</th>
      <td>3485.000064</td>
      <td>0.0005</td>
      <td>956.341038</td>
      <td>1295.199991</td>
      <td>338.858953</td>
    </tr>
    <tr>
      <th>24</th>
      <td>6475.999956</td>
      <td>0.0005</td>
      <td>927.689291</td>
      <td>800.200023</td>
      <td>-127.489267</td>
    </tr>
    <tr>
      <th>25</th>
      <td>4001.000116</td>
      <td>0.0005</td>
      <td>951.301093</td>
      <td>1008.200022</td>
      <td>56.898928</td>
    </tr>
    <tr>
      <th>26</th>
      <td>5041.000108</td>
      <td>0.0005</td>
      <td>941.264254</td>
      <td>155.599997</td>
      <td>-785.664257</td>
    </tr>
    <tr>
      <th>27</th>
      <td>777.999985</td>
      <td>0.0005</td>
      <td>983.406864</td>
      <td>1617.399943</td>
      <td>633.993079</td>
    </tr>
    <tr>
      <th>28</th>
      <td>8086.999715</td>
      <td>0.0005</td>
      <td>912.840520</td>
      <td>1621.600054</td>
      <td>708.759534</td>
    </tr>
    <tr>
      <th>29</th>
      <td>8108.000270</td>
      <td>0.0005</td>
      <td>912.649734</td>
      <td>1007.600023</td>
      <td>94.950288</td>
    </tr>
    <tr>
      <th>30</th>
      <td>5038.000113</td>
      <td>0.0005</td>
      <td>941.292954</td>
      <td>274.200007</td>
      <td>-667.092947</td>
    </tr>
    <tr>
      <th>31</th>
      <td>1371.000035</td>
      <td>0.0005</td>
      <td>977.390701</td>
      <td>1561.000034</td>
      <td>583.609333</td>
    </tr>
    <tr>
      <th>32</th>
      <td>7805.000169</td>
      <td>0.0005</td>
      <td>915.409248</td>
      <td>1228.199942</td>
      <td>312.790694</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
