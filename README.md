# Vorhersage des Ödometerversuches implementiert mit PINA
Ziel war die Implementierung eines neuronalen Netzwerks zur Modellierung des Ödometerversuchs. Dabei wurden gegebene Input-Parameter verarbeitet, um Output-Parameter vorherzusagen. Die physikalischen Rahmenbedingungen wurden zunächst auf Null gesetzt, sodass das Modell ausschließlich auf der KI-basierten Struktur arbeitet, ohne physikalische Optimierungen durch Physical Informed Neural Networks (PINNs).
<br>
Diese grundlegende Umsetzung bildet die Basis für weiterführende Optimierungen, wie die Integration physikalischer Gesetzmäßigkeiten, die jedoch nicht Teil des initialen Arbeitsauftrags waren.

### Was ist PINA?
PINA ist eine Open-Source-Python-Bibliothek, die eine intuitive Schnittstelle zur Lösung von Differentialgleichungen bietet, indem sie Physik-informierte Neuronale Netze (PINNs), Neuronale Operatoren (NOs) oder eine Kombination aus beiden verwendet. Basierend auf PyTorch und PyTorch Lightning ermöglicht PINA die formale Darstellung spezifischer (differentieller) Probleme und deren Lösung mittels neuronaler Netze.<br><br>
<strong>Hauptmerkmale von PINA:</strong>

- <span style="color:gray;"><i>Problemformulierung: Ermöglicht die Übersetzung mathematischer Gleichungen in Python-Code, um das Differentialproblem zu definieren.</i></span>
    - <small><i>→ In diesem Arbeitsauftrag nicht notwendig, da das neuronale Netzwerk ohne physikalische Gesetzmäßigkeiten trainiert wurde.</i></small>
- Modelltraining: Bietet Werkzeuge zum Training neuronaler Netze zur Lösung des definierten Problems.
- Lösungsauswertung: Erlaubt die Visualisierung und Analyse der approximierten Lösungen.

<small><i>Hinweis: Die physikalische Modellierung und die Einbindung von Differentialgleichungen zur Optimierung des Netzwerks (z. B. mittels PINNs) war nicht Teil dieses Arbeitsauftrags, könnte aber in einem späteren Schritt ergänzt werden.</i></small>
## Grundlagen
In diesem Notebook wird der Ödometerversuch <strong>ohne</strong> Randbedingungen betrachtet. Es werden vorberechnetet Daten aus der Exceltabelle `files/oedometer/oedo_trainingsdata.xlsx` verwendet.<br>
#### Das Problem ist wie folgt definiert:
$$
\begin{array}{rcl}
    \sigma_{t+1} & = & \sigma_{t}+\Delta\sigma \\ \\
    \Delta\sigma & = & E_s\cdot \Delta\epsilon \\ 
    E_s & = & \frac{1+e_0}{C_c} \cdot \sigma_t
\end{array}
\hspace{2cm}
\begin{array}{l}
    \textbf{Annahmen:} \\ \\
    \text{Startwert d. Iteration: } \sigma_t = 1,00 \\ 
    e_0 = 1,00 \\ 
    C_c = 0,005 \\
    \Delta\epsilon = 0,0005
\end{array}
$$
<div = style="text-align: center;">
    <img alt="Problem Oedometer Preview" src="./graph/problem_preview.png" width="50%" height=auto>
</div>

<br> 

Um das PINA-Model zu testen werden wir folgende vorberechnete Werte verwenden: `Input` { $\sigma_t$ ; $\Delta\epsilon$ }, `Output` { $\sigma_{t+1}$ }.
<br>
### Variablendeklaration
- $\sigma_t$ = `sigma_t`
- $\Delta\epsilon$ = `delta_epsilon`
- $\sigma_{t+1}$ = `delta_sigma`
## Einstellungen und Utilities


```python
from IPython.display import display, Markdown
import matplotlib
matplotlib.use('Agg')  # Keine doppelte Darstellung des Plots
import matplotlib.pyplot as plt
import torch
import numpy as np

# Debugger: Aktiviert
debug_mode = True
# Normalisierung der Daten: Deaktiviert
normalize_data = False
use_excel = False
max_input_pts = 20

# Trainingsdaten e_0:float=1.00, C_c:float=0.005, delta_epsilon:float=0.0005, sigma_t:float=1.00, max_n:int=50
oedo_parameter = {'e_0':1.00, 'C_c':0.005, 'delta_epsilon':0.0005, 'sigma_t':1.00, 'max_n':100, 'rand_epsilon':False}

new_test = False

graph_folder = 'graph'
img_extensions = '.png'

img_visual_loss = 'visual_loss'
img_nn_result_error = 'img_nn_result_error'
img_visual_prediction_vs_truesolution_comp = 'visual_prediction-vs-truesolution_comp'
img_visual_prediction_vs_truesolution_comp0 = 'visual_prediction-vs-truesolution_comp0'
img_visual_prediction_vs_truesolution_comp1 = 'visual_prediction-vs-truesolution_comp1'
img_visual_prediction_vs_truesolution_comp2 = 'visual_prediction-vs-truesolution_comp2'
img_visual_prediction_vs_truesolution = 'visual_prediction-vs-truesolution'
img_visual_sampling = 'visual_sampling'

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
    delta_sigma_true = to_list(data_dict["delta_sigma"])
    delta_sigma_pred = to_list(delta_sigma_pred.flatten())  # Falls `delta_sigma_pred` ein 2D-Array ist
    
    # Überprüfen, ob die Längen konsistent sind
    min_len = min(len(total_epsilon), len(delta_epsilon), len(delta_sigma_true), len(delta_sigma_pred), max_i)

    # Erstelle eine Tabelle für die übersichtliche Darstellung
    data_loss_table = {
        "total_epsilon" : list(total_epsilon[:min_len]), 
        "delta_epsilon" : list(delta_epsilon[:min_len]), 
        "True sigma_t+1": list(delta_sigma_true[:min_len]),
        "Predicted sigma_t+1": list(delta_sigma_pred[:min_len]),
        "Loss (True - Predicted)": list(np.round(np.array(delta_sigma_true[:min_len]) - np.array(delta_sigma_pred[:min_len]), 5))
    }

    # Markdown-Tabelle für bessere Darstellung in Jupyter
    display(dict_to_markdown_table(data_loss_table, title=f"Data-Loss bis sigma_{min_len-1}", include_index=True))

def plot_prediction_vs_true_solution(pinn, data_dict, graph_folder, img_visual_prediction_vs_truesolution, 
                                     img_extensions, y_axis='delta_sigma', max_i=20, plot_type="line"):
    """
    Erstellt und speichert eine Vorhersage- vs. True-Solution-Grafik für ein gegebenes PINN-Modell.

    :param pinn: Das trainierte PINN-Modell zur Vorhersage von delta_sigma
    :param data_dict: Dictionary mit den Eingabe- und wahren Ausgabe-Daten
    :param graph_folder: Ordner, in dem das Bild gespeichert wird
    :param img_visual_prediction_vs_truesolution: Dateiname der gespeicherten Grafik (ohne Erweiterung)
    :param img_extensions: Dateiformat der gespeicherten Grafik (z.B. '.png' oder '.jpg')
    :param max_i: Anzahl der Datenpunkte, die im Plot gezeigt werden sollen (Default: 20)
    :param delta_epsilon: Wert für delta_epsilon, um ihn im Titel anzuzeigen (optional)
    :param plot_type: Art der Darstellung - "line" für Linienplot, "scatter" für Punktplot (Default: "line")
    """

    # Überprüfen, ob die notwendigen Keys vorhanden sind
    if "sigma_t" not in data_dict or y_axis not in data_dict:
        print(f"Fehler: sigma_t oder y_axis fehlen im data_dict!")
        return

    # Eingabedaten für das Modell vorbereiten
    input_data = LabelTensor(torch.tensor(
        np.column_stack((data_dict['sigma_t'], data_dict['delta_epsilon'])), 
        dtype=torch.float), ['sigma_t', 'delta_epsilon'])

    # Vorhersage berechnen
    sigma_t_pred = pinn(input_data).detach().numpy()

    # Plot erstellen
    plt.figure(figsize=(10, 5))

    y_vals = data_dict[y_axis][0:max_i]
    x_true = data_dict['delta_sigma'][0:max_i]
    x_pred = sigma_t_pred[0:max_i]

    if plot_type == "line":
        plt.plot(x_true, y_vals, label="True Solution (delta_sigma)", linestyle='dashed', color='blue')
        plt.plot(x_pred, y_vals, label="NN Prediction (delta_sigma)", linestyle='solid', color='red')
    elif plot_type == "scatter":
        plt.scatter(x_true, y_vals, label="True Solution (delta_sigma)", color='blue', marker='o')
        plt.scatter(x_pred, y_vals, label="NN Prediction (delta_sigma)", color='red', marker='x')

    plt.xlabel("delta_sigma")
    plt.ylabel(y_axis)
    plt.title(f"Prediction vs. True Solution (max_i={max_i-1})")
    plt.legend()
    plt.grid()
    plt.gca().invert_yaxis()
    
    # Bild speichern
    img_path = f'./{graph_folder}/{img_visual_prediction_vs_truesolution}{img_extensions}'
    plt.savefig(img_path)
    plt.close()  # Verhindert doppelte Darstellung

    # Markdown-Ausgabe in Jupyter Notebook
    display(Markdown(f'![Prediction vs True Solution]({img_path})<br>**Hinweis:** Datenpunkte liegen sehr nahe beieinander.'))

```

## Laden der Daten aus `oedo_trainingsdata.xlsx`


```python
import pandas as pd
import numpy as np
from sympy.integrals.heurisch import components

def extract_excel(file_path, sheet_name, selected_columns, row_start_range):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # Dynamische Ermittlung der letzten Zeile mit Daten
    row_start_range = 0  # Startet bei Zeile 6 (0-basiert)
    row_end_range = df.dropna(how="all").last_valid_index() + 1  # Letzte Zeile mit Daten
        
    # Daten extrahieren
    data_subset = df.iloc[row_start_range:row_end_range, selected_columns]
    data_dict = {col: np.array(data_subset[col]) for col in data_subset.columns}
    
    if debug_mode:
        print(data_dict)
        dict_to_markdown_table(data=data_dict,title=file_path)
    
    # Daten als dict speichern
    return data_dict
    
if use_excel:
    data_dict = extract_excel(file_path, sheet_name, selected_columns, row_start_range)
```

## Laden der Daten aus `Python`


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

if not use_excel:
    data_dict = dict(vars(Oedometer(**oedo_parameter)))
    display(dict_to_markdown_table(data_dict, 'Ödometerdaten'))
```


### Ödometerdaten

| Index | e_0 | C_c | sigma_t | delta_epsilon | total_epsilon | delta_sigma | e_s |
|--|----|----|--------|--------------|--------------|------------|----|
| 0 | 1.0 | 0.005 | 1.0 | 0.0005 | 0 | 0.2 | 400.0 |
| 1 | 1.0 | 0.005 | 1.2 | 0.0005 | 0.0005 | 0.24 | 480.0 |
| 2 | 1.0 | 0.005 | 1.44 | 0.0005 | 0.001 | 0.288 | 576.0 |
| 3 | 1.0 | 0.005 | 1.728 | 0.0005 | 0.0015 | 0.3456 | 691.2 |
| 4 | 1.0 | 0.005 | 2.0736 | 0.0005 | 0.002 | 0.4147 | 829.44 |
| 5 | 1.0 | 0.005 | 2.4883 | 0.0005 | 0.0025 | 0.4977 | 995.328 |
| 6 | 1.0 | 0.005 | 2.986 | 0.0005 | 0.003 | 0.5972 | 1194.3936 |
| 7 | 1.0 | 0.005 | 3.5832 | 0.0005 | 0.0035 | 0.7166 | 1433.2723 |
| 8 | 1.0 | 0.005 | 4.2998 | 0.0005 | 0.004 | 0.86 | 1719.9268 |
| 9 | 1.0 | 0.005 | 5.1598 | 0.0005 | 0.0045 | 1.032 | 2063.9121 |
| 10 | 1.0 | 0.005 | 6.1917 | 0.0005 | 0.005 | 1.2383 | 2476.6946 |
| 11 | 1.0 | 0.005 | 7.4301 | 0.0005 | 0.0055 | 1.486 | 2972.0335 |
| 12 | 1.0 | 0.005 | 8.9161 | 0.0005 | 0.006 | 1.7832 | 3566.4402 |
| 13 | 1.0 | 0.005 | 10.6993 | 0.0005 | 0.0065 | 2.1399 | 4279.7282 |
| 14 | 1.0 | 0.005 | 12.8392 | 0.0005 | 0.007 | 2.5678 | 5135.6739 |
| 15 | 1.0 | 0.005 | 15.407 | 0.0005 | 0.0075 | 3.0814 | 6162.8086 |
| 16 | 1.0 | 0.005 | 18.4884 | 0.0005 | 0.008 | 3.6977 | 7395.3704 |
| 17 | 1.0 | 0.005 | 22.1861 | 0.0005 | 0.0085 | 4.4372 | 8874.4444 |
| 18 | 1.0 | 0.005 | 26.6233 | 0.0005 | 0.009 | 5.3247 | 10649.3333 |
| 19 | 1.0 | 0.005 | 31.948 | 0.0005 | 0.0095 | 6.3896 | 12779.2 |
| 20 | 1.0 | 0.005 | 38.3376 | 0.0005 | 0.01 | 7.6675 | 15335.04 |
| 21 | 1.0 | 0.005 | 46.0051 | 0.0005 | 0.0105 | 9.201 | 18402.048 |
| 22 | 1.0 | 0.005 | 55.2061 | 0.0005 | 0.011 | 11.0412 | 22082.4576 |
| 23 | 1.0 | 0.005 | 66.2474 | 0.0005 | 0.0115 | 13.2495 | 26498.9491 |
| 24 | 1.0 | 0.005 | 79.4968 | 0.0005 | 0.012 | 15.8994 | 31798.7389 |
| 25 | 1.0 | 0.005 | 95.3962 | 0.0005 | 0.0125 | 19.0792 | 38158.4867 |
| 26 | 1.0 | 0.005 | 114.4755 | 0.0005 | 0.013 | 22.8951 | 45790.184 |
| 27 | 1.0 | 0.005 | 137.3706 | 0.0005 | 0.0135 | 27.4741 | 54948.2208 |
| 28 | 1.0 | 0.005 | 164.8447 | 0.0005 | 0.014 | 32.9689 | 65937.8649 |
| 29 | 1.0 | 0.005 | 197.8136 | 0.0005 | 0.0145 | 39.5627 | 79125.4379 |
| 30 | 1.0 | 0.005 | 237.3763 | 0.0005 | 0.015 | 47.4753 | 94950.5255 |
| 31 | 1.0 | 0.005 | 284.8516 | 0.0005 | 0.0155 | 56.9703 | 113940.6306 |
| 32 | 1.0 | 0.005 | 341.8219 | 0.0005 | 0.016 | 68.3644 | 136728.7567 |
| 33 | 1.0 | 0.005 | 410.1863 | 0.0005 | 0.0165 | 82.0373 | 164074.5081 |
| 34 | 1.0 | 0.005 | 492.2235 | 0.0005 | 0.017 | 98.4447 | 196889.4097 |
| 35 | 1.0 | 0.005 | 590.6682 | 0.0005 | 0.0175 | 118.1336 | 236267.2917 |
| 36 | 1.0 | 0.005 | 708.8019 | 0.0005 | 0.018 | 141.7604 | 283520.75 |
| 37 | 1.0 | 0.005 | 850.5622 | 0.0005 | 0.0185 | 170.1124 | 340224.9 |
| 38 | 1.0 | 0.005 | 1020.6747 | 0.0005 | 0.019 | 204.1349 | 408269.88 |
| 39 | 1.0 | 0.005 | 1224.8096 | 0.0005 | 0.0195 | 244.9619 | 489923.856 |
| 40 | 1.0 | 0.005 | 1469.7716 | 0.0005 | 0.02 | 293.9543 | 587908.6272 |
| 41 | 1.0 | 0.005 | 1763.7259 | 0.0005 | 0.0205 | 352.7452 | 705490.3526 |
| 42 | 1.0 | 0.005 | 2116.4711 | 0.0005 | 0.021 | 423.2942 | 846588.4232 |
| 43 | 1.0 | 0.005 | 2539.7653 | 0.0005 | 0.0215 | 507.9531 | 1015906.1078 |
| 44 | 1.0 | 0.005 | 3047.7183 | 0.0005 | 0.022 | 609.5437 | 1219087.3293 |
| 45 | 1.0 | 0.005 | 3657.262 | 0.0005 | 0.0225 | 731.4524 | 1462904.7952 |
| 46 | 1.0 | 0.005 | 4388.7144 | 0.0005 | 0.023 | 877.7429 | 1755485.7542 |
| 47 | 1.0 | 0.005 | 5266.4573 | 0.0005 | 0.0235 | 1053.2915 | 2106582.9051 |
| 48 | 1.0 | 0.005 | 6319.7487 | 0.0005 | 0.024 | 1263.9497 | 2527899.4861 |
| 49 | 1.0 | 0.005 | 7583.6985 | 0.0005 | 0.0245 | 1516.7397 | 3033479.3833 |
| 50 | 1.0 | 0.005 | 9100.4382 | 0.0005 | 0.025 | 1820.0876 | 3640175.26 |
| 51 | 1.0 | 0.005 | 10920.5258 | 0.0005 | 0.0255 | 2184.1052 | 4368210.312 |
| 52 | 1.0 | 0.005 | 13104.6309 | 0.0005 | 0.026 | 2620.9262 | 5241852.3744 |
| 53 | 1.0 | 0.005 | 15725.5571 | 0.0005 | 0.0265 | 3145.1114 | 6290222.8493 |
| 54 | 1.0 | 0.005 | 18870.6685 | 0.0005 | 0.027 | 3774.1337 | 7548267.4191 |
| 55 | 1.0 | 0.005 | 22644.8023 | 0.0005 | 0.0275 | 4528.9605 | 9057920.903 |
| 56 | 1.0 | 0.005 | 27173.7627 | 0.0005 | 0.028 | 5434.7525 | 10869505.0836 |
| 57 | 1.0 | 0.005 | 32608.5153 | 0.0005 | 0.0285 | 6521.7031 | 13043406.1003 |
| 58 | 1.0 | 0.005 | 39130.2183 | 0.0005 | 0.029 | 7826.0437 | 15652087.3203 |
| 59 | 1.0 | 0.005 | 46956.262 | 0.0005 | 0.0295 | 9391.2524 | 18782504.7844 |
| 60 | 1.0 | 0.005 | 56347.5144 | 0.0005 | 0.03 | 11269.5029 | 22539005.7413 |
| 61 | 1.0 | 0.005 | 67617.0172 | 0.0005 | 0.0305 | 13523.4034 | 27046806.8895 |
| 62 | 1.0 | 0.005 | 81140.4207 | 0.0005 | 0.031 | 16228.0841 | 32456168.2674 |
| 63 | 1.0 | 0.005 | 97368.5048 | 0.0005 | 0.0315 | 19473.701 | 38947401.9209 |
| 64 | 1.0 | 0.005 | 116842.2058 | 0.0005 | 0.032 | 23368.4412 | 46736882.3051 |
| 65 | 1.0 | 0.005 | 140210.6469 | 0.0005 | 0.0325 | 28042.1294 | 56084258.7661 |
| 66 | 1.0 | 0.005 | 168252.7763 | 0.0005 | 0.033 | 33650.5553 | 67301110.5193 |
| 67 | 1.0 | 0.005 | 201903.3316 | 0.0005 | 0.0335 | 40380.6663 | 80761332.6232 |
| 68 | 1.0 | 0.005 | 242283.9979 | 0.0005 | 0.034 | 48456.7996 | 96913599.1478 |
| 69 | 1.0 | 0.005 | 290740.7974 | 0.0005 | 0.0345 | 58148.1595 | 116296318.9774 |
| 70 | 1.0 | 0.005 | 348888.9569 | 0.0005 | 0.035 | 69777.7914 | 139555582.7729 |
| 71 | 1.0 | 0.005 | 418666.7483 | 0.0005 | 0.0355 | 83733.3497 | 167466699.3275 |
| 72 | 1.0 | 0.005 | 502400.098 | 0.0005 | 0.036 | 100480.0196 | 200960039.193 |
| 73 | 1.0 | 0.005 | 602880.1176 | 0.0005 | 0.0365 | 120576.0235 | 241152047.0315 |
| 74 | 1.0 | 0.005 | 723456.1411 | 0.0005 | 0.037 | 144691.2282 | 289382456.4379 |
| 75 | 1.0 | 0.005 | 868147.3693 | 0.0005 | 0.0375 | 173629.4739 | 347258947.7254 |
| 76 | 1.0 | 0.005 | 1041776.8432 | 0.0005 | 0.038 | 208355.3686 | 416710737.2705 |
| 77 | 1.0 | 0.005 | 1250132.2118 | 0.0005 | 0.0385 | 250026.4424 | 500052884.7246 |
| 78 | 1.0 | 0.005 | 1500158.6542 | 0.0005 | 0.039 | 300031.7308 | 600063461.6695 |
| 79 | 1.0 | 0.005 | 1800190.385 | 0.0005 | 0.0395 | 360038.077 | 720076154.0034 |
| 80 | 1.0 | 0.005 | 2160228.462 | 0.0005 | 0.04 | 432045.6924 | 864091384.8041 |
| 81 | 1.0 | 0.005 | 2592274.1544 | 0.0005 | 0.0405 | 518454.8309 | 1036909661.7649 |
| 82 | 1.0 | 0.005 | 3110728.9853 | 0.0005 | 0.041 | 622145.7971 | 1244291594.1179 |
| 83 | 1.0 | 0.005 | 3732874.7824 | 0.0005 | 0.0415 | 746574.9565 | 1493149912.9415 |
| 84 | 1.0 | 0.005 | 4479449.7388 | 0.0005 | 0.042 | 895889.9478 | 1791779895.5298 |
| 85 | 1.0 | 0.005 | 5375339.6866 | 0.0005 | 0.0425 | 1075067.9373 | 2150135874.6358 |
| 86 | 1.0 | 0.005 | 6450407.6239 | 0.0005 | 0.043 | 1290081.5248 | 2580163049.563 |
| 87 | 1.0 | 0.005 | 7740489.1487 | 0.0005 | 0.0435 | 1548097.8297 | 3096195659.4755 |
| 88 | 1.0 | 0.005 | 9288586.9784 | 0.0005 | 0.044 | 1857717.3957 | 3715434791.3707 |
| 89 | 1.0 | 0.005 | 11146304.3741 | 0.0005 | 0.0445 | 2229260.8748 | 4458521749.6448 |
| 90 | 1.0 | 0.005 | 13375565.2489 | 0.0005 | 0.045 | 2675113.0498 | 5350226099.5737 |
| 91 | 1.0 | 0.005 | 16050678.2987 | 0.0005 | 0.0455 | 3210135.6597 | 6420271319.4885 |
| 92 | 1.0 | 0.005 | 19260813.9585 | 0.0005 | 0.046 | 3852162.7917 | 7704325583.3862 |
| 93 | 1.0 | 0.005 | 23112976.7502 | 0.0005 | 0.0465 | 4622595.35 | 9245190700.0634 |
| 94 | 1.0 | 0.005 | 27735572.1002 | 0.0005 | 0.047 | 5547114.42 | 11094228840.0761 |
| 95 | 1.0 | 0.005 | 33282686.5202 | 0.0005 | 0.0475 | 6656537.304 | 13313074608.0913 |
| 96 | 1.0 | 0.005 | 39939223.8243 | 0.0005 | 0.048 | 7987844.7649 | 15975689529.7096 |
| 97 | 1.0 | 0.005 | 47927068.5891 | 0.0005 | 0.0485 | 9585413.7178 | 19170827435.6515 |
| 98 | 1.0 | 0.005 | 57512482.307 | 0.0005 | 0.049 | 11502496.4614 | 23004992922.7818 |
| 99 | 1.0 | 0.005 | 69014978.7683 | 0.0005 | 0.0495 | 13802995.7537 | 27605991507.3382 |


#### Einzelwerte

| max_n |
|-------|
| 100 |



<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">

## Daten normalisieren
Die Normalisierung von Daten für neuronale Netze bedeutet, dass Eingabedaten auf eine vergleichbare Skala gebracht werden, um das Training stabiler und effizienter zu machen. Hier verwendete Methode:
- Min-Max-Skalierung: Werte auf einen Bereich (0 bis 1) bringen.  <i class="fa fa-info"> [Wiki](https://en.wikipedia.org/wiki/Feature_scaling#Methods)</i>



```python
if normalize_data:
    data_dict.update({'sigma_t_raw': data_dict.pop('sigma_t')})
    data_dict.update({'delta_sigma_raw': data_dict.pop('delta_sigma')})
    
    sigma_t_min, sigma_t_max = data_dict['sigma_t_raw'].min(), data_dict['sigma_t_raw'].max()
    delta_sigma_min, delta_sigma_max = data_dict['delta_sigma_raw'].min(), data_dict['delta_sigma_raw'].max()
    
    # Min-Max-Normalisierung
    data_dict['sigma_t'] = (data_dict['sigma_t_raw'] - sigma_t_min) / (sigma_t_max - sigma_t_min)
    data_dict['delta_sigma'] = (data_dict['delta_sigma_raw'] - delta_sigma_min) / (delta_sigma_max - delta_sigma_min)
    print('‼️Tabellenwerte des Oedometerversuches normalisiert.')
else:
    print('‼️ Es wurde keine Normalisierung der Werte vorgenommen.')
```

    ‼️ Es wurde keine Normalisierung der Werte vorgenommen.
    

## **Datenvorbereitung für PINA mit LabelTensor**
In diesem Code werden die Eingabedaten aus `data_dict` als **LabelTensor** gespeichert, um sie strukturiert und mit benannten Dimensionen für das neuronale Netz in PINA bereitzustellen.  

- `sigma_t_train`, `delta_epsilon_train` und `delta_sigma_train` werden als **einzelne beschriftete Tensoren** erstellt.  
- `input_points_combined` kombiniert `sigma_t` und `delta_epsilon` in einem **2D-Tensor** für das Training.  
- `LabelTensor` erleichtert die Nutzung der Daten in PINA, indem es Variablen klar zuordnet und mit physischen Größen verknüpft.

**Mehr zu `LabelTensor`:**  
[PINA Documentation – LabelTensor](https://mathlab.github.io/PINA/_rst/label_tensor.html)



```python
import torch
from pina.utils import LabelTensor
from torch import tensor

# Beispiel-Daten
sigma_t_train = LabelTensor(tensor(data_dict['sigma_t'], dtype=torch.float).unsqueeze(-1),['sigma_t'])
delta_epsilon_train = LabelTensor(tensor(data_dict['delta_epsilon'], dtype=torch.float).unsqueeze(-1), ['delta_epsilon'])
delta_sigma_train = LabelTensor(tensor(data_dict['delta_sigma'], dtype=torch.float).unsqueeze(-1), ['delta_sigma'])

# Kombinieren der Trainingsdaten (Verwendung von 'np.column_stack' für bessere Performance)
input_points_combined = LabelTensor(torch.tensor(np.column_stack([data_dict['sigma_t'], data_dict['delta_epsilon']]), dtype=torch.float), ['sigma_t', 'delta_epsilon'])

if debug_mode:
    print('‼️Data Loaded')
    print(f' sigma_t: {sigma_t_train.size()}')
    print(f' delta_epsilon: {delta_epsilon_train.shape}')
    print(f' sigma_t und delta_epsilon combined: {input_points_combined.size()}')
    print(f' delta_sigma: {delta_sigma_train.shape}')
```

    ‼️Data Loaded
     sigma_t: torch.Size([100, 1])
     delta_epsilon: torch.Size([100, 1])
     sigma_t und delta_epsilon combined: torch.Size([100, 2])
     delta_sigma: torch.Size([100, 1])
    

### **Definition eines einfachen PINN-Problems in PINA**  
Dieser Code definiert ein **Physics-Informed Neural Network (PINN)**-Problem mithilfe der PINA-Bibliothek.  
 
- **Klassenstruktur (`SimpleODE`)**: Erbt von `AbstractProblem` und spezifiziert die Eingabe- und Ausgabevariablen basierend auf `LabelTensor`.
    - [PINA-Dokumentation - AbstractProblem](https://mathlab.github.io/PINA/_rst/problem/abstractproblem.html) 
- **Definitionsbereich (`domain`)**: Der Wertebereich der Eingabevariablen (`sigma_t`, `delta_epsilon`) wird als `CartesianDomain` festgelegt.
    - **Hinweis:** `domain` muss immer definiert sein, selbst wenn sie nicht direkt zur Datengenerierung verwendet wird.  
    - [PINA-Dokumentation - CartesianDomain](https://mathlab.github.io/PINA/_rst/geometry/cartesian.html) 
- **Randbedingungen (`conditions`)**: Die echten Messwerte (`in sigma_t, delta_epsilon` `out delta_sigma_train`) werden als Randbedingung (`Condition`) für das Modell definiert.
    - [PINA-Dokumentation - Condition](https://mathlab.github.io/PINA/_rst/condition.html) 
- **"Wahre Lösung" (`truth_solution`)**: Falls erforderlich, kann eine analytische Lösung (hier `torch.exp(...)`) zur Validierung genutzt werden.
    - **Hinweis:** Funktioniert in unserem Fall nicht, da die Implementierung nicht für reine Input und Outpunkt Punkte implementiert ist.
    - [PINA-Tutorial - Physics Informed Neural Networks on PINA](https://mathlab.github.io/PINA/_rst/tutorials/tutorial1/tutorial.html) 
- **Probleminstanz (`problem = SimpleODE()`)**: Erstellt das Problem, das für das Training eines PINN verwendet wird.  



```python
from pina.problem import AbstractProblem
from pina.geometry import CartesianDomain
from pina import Condition

# Datengenerierung, falls Randbedingungen definiert
# problem.discretise_domain(n=993, mode='random', variables='all', locations='all') # Notwendig, wenn "input_pts" und "output_pts" nicht vorgegeben sind
if new_test:
    # Trainingsdaten e_0:float=1.00, C_c:float=0.005, delta_epsilon:float=0.0005, sigma_t:float=1.00, max_n:int=50
    oedo_parameter = {'e_0':1.00, 'C_c':0.005, 'delta_epsilon':0.0005, 'sigma_t':1.00, 'max_n':100, 'rand_epsilon':False}
    input_conditions = {}
    for i in range(max_input_pts):
        oedo_parameter['delta_epsilon'] = oedo_parameter['delta_epsilon'] + .0001
        data_dict = vars(Oedometer(**oedo_parameter))

        sigma_t_train = LabelTensor(tensor(data_dict['sigma_t'][0], dtype=torch.float).unsqueeze(-1),['sigma_t'])
        delta_epsilon_train = LabelTensor(tensor(data_dict['delta_epsilon'][0], dtype=torch.float).unsqueeze(-1), ['delta_epsilon'])
        delta_sigma_train = LabelTensor(tensor(data_dict['delta_sigma'][0], dtype=torch.float).unsqueeze(-1), ['delta_sigma'])
        # Kombinieren der Trainingsdaten (Verwendung von 'np.column_stack' für bessere Performance)
        input_points_combined = LabelTensor(torch.tensor(np.column_stack([data_dict['sigma_t'][0], data_dict['delta_epsilon'][0]]), dtype=torch.float), ['sigma_t', 'delta_epsilon'])

        input_conditions['condition' + str(i)] = Condition(input_points=input_points_combined, output_points=delta_sigma_train)
else:
    input_conditions = {'data': Condition(input_points=input_points_combined, output_points=delta_sigma_train),}


class SimpleODE(AbstractProblem):

    # Definition der Eingabe- und Ausgabevariablen basierend auf LabelTensor
    input_variables = input_points_combined.labels
    output_variables = delta_sigma_train.labels

    # Wertebereich
    domain = CartesianDomain({'sigma_t': [0, 1], 'delta_epsilon': [0, 1]})  # Wertebereich immer definieren!

    # Definition der Randbedingungen und (hier: nur) vorberechnetet Punkte
    conditions = input_conditions

    output_pts=delta_sigma_train

    # Methode zur Definition der "wahren Lösung" des Problems
    def truth_solution(self, pts):
        return torch.exp(pts.extract(['sigma_t']))

# Problem-Instanz erzeugen
problem = SimpleODE()



if debug_mode:
    # Debugging-Ausgaben
    print("‼️Geladene Input Variablen: ", problem.input_variables)
    print("‼️Geladene Output Variablen: ", problem.output_variables)
    print('‼️Input points:', problem.input_pts)
```

    ‼️Geladene Input Variablen:  ['sigma_t', 'delta_epsilon']
    ‼️Geladene Output Variablen:  ['delta_sigma']
    ‼️Input points: {'data': LabelTensor([[[1.0000e+00, 5.0000e-04]],
                 [[1.2000e+00, 5.0000e-04]],
                 [[1.4400e+00, 5.0000e-04]],
                 [[1.7280e+00, 5.0000e-04]],
                 [[2.0736e+00, 5.0000e-04]],
                 [[2.4883e+00, 5.0000e-04]],
                 [[2.9860e+00, 5.0000e-04]],
                 [[3.5832e+00, 5.0000e-04]],
                 [[4.2998e+00, 5.0000e-04]],
                 [[5.1598e+00, 5.0000e-04]],
                 [[6.1917e+00, 5.0000e-04]],
                 [[7.4301e+00, 5.0000e-04]],
                 [[8.9161e+00, 5.0000e-04]],
                 [[1.0699e+01, 5.0000e-04]],
                 [[1.2839e+01, 5.0000e-04]],
                 [[1.5407e+01, 5.0000e-04]],
                 [[1.8488e+01, 5.0000e-04]],
                 [[2.2186e+01, 5.0000e-04]],
                 [[2.6623e+01, 5.0000e-04]],
                 [[3.1948e+01, 5.0000e-04]],
                 [[3.8338e+01, 5.0000e-04]],
                 [[4.6005e+01, 5.0000e-04]],
                 [[5.5206e+01, 5.0000e-04]],
                 [[6.6247e+01, 5.0000e-04]],
                 [[7.9497e+01, 5.0000e-04]],
                 [[9.5396e+01, 5.0000e-04]],
                 [[1.1448e+02, 5.0000e-04]],
                 [[1.3737e+02, 5.0000e-04]],
                 [[1.6484e+02, 5.0000e-04]],
                 [[1.9781e+02, 5.0000e-04]],
                 [[2.3738e+02, 5.0000e-04]],
                 [[2.8485e+02, 5.0000e-04]],
                 [[3.4182e+02, 5.0000e-04]],
                 [[4.1019e+02, 5.0000e-04]],
                 [[4.9222e+02, 5.0000e-04]],
                 [[5.9067e+02, 5.0000e-04]],
                 [[7.0880e+02, 5.0000e-04]],
                 [[8.5056e+02, 5.0000e-04]],
                 [[1.0207e+03, 5.0000e-04]],
                 [[1.2248e+03, 5.0000e-04]],
                 [[1.4698e+03, 5.0000e-04]],
                 [[1.7637e+03, 5.0000e-04]],
                 [[2.1165e+03, 5.0000e-04]],
                 [[2.5398e+03, 5.0000e-04]],
                 [[3.0477e+03, 5.0000e-04]],
                 [[3.6573e+03, 5.0000e-04]],
                 [[4.3887e+03, 5.0000e-04]],
                 [[5.2665e+03, 5.0000e-04]],
                 [[6.3197e+03, 5.0000e-04]],
                 [[7.5837e+03, 5.0000e-04]],
                 [[9.1004e+03, 5.0000e-04]],
                 [[1.0921e+04, 5.0000e-04]],
                 [[1.3105e+04, 5.0000e-04]],
                 [[1.5726e+04, 5.0000e-04]],
                 [[1.8871e+04, 5.0000e-04]],
                 [[2.2645e+04, 5.0000e-04]],
                 [[2.7174e+04, 5.0000e-04]],
                 [[3.2609e+04, 5.0000e-04]],
                 [[3.9130e+04, 5.0000e-04]],
                 [[4.6956e+04, 5.0000e-04]],
                 [[5.6348e+04, 5.0000e-04]],
                 [[6.7617e+04, 5.0000e-04]],
                 [[8.1140e+04, 5.0000e-04]],
                 [[9.7369e+04, 5.0000e-04]],
                 [[1.1684e+05, 5.0000e-04]],
                 [[1.4021e+05, 5.0000e-04]],
                 [[1.6825e+05, 5.0000e-04]],
                 [[2.0190e+05, 5.0000e-04]],
                 [[2.4228e+05, 5.0000e-04]],
                 [[2.9074e+05, 5.0000e-04]],
                 [[3.4889e+05, 5.0000e-04]],
                 [[4.1867e+05, 5.0000e-04]],
                 [[5.0240e+05, 5.0000e-04]],
                 [[6.0288e+05, 5.0000e-04]],
                 [[7.2346e+05, 5.0000e-04]],
                 [[8.6815e+05, 5.0000e-04]],
                 [[1.0418e+06, 5.0000e-04]],
                 [[1.2501e+06, 5.0000e-04]],
                 [[1.5002e+06, 5.0000e-04]],
                 [[1.8002e+06, 5.0000e-04]],
                 [[2.1602e+06, 5.0000e-04]],
                 [[2.5923e+06, 5.0000e-04]],
                 [[3.1107e+06, 5.0000e-04]],
                 [[3.7329e+06, 5.0000e-04]],
                 [[4.4794e+06, 5.0000e-04]],
                 [[5.3753e+06, 5.0000e-04]],
                 [[6.4504e+06, 5.0000e-04]],
                 [[7.7405e+06, 5.0000e-04]],
                 [[9.2886e+06, 5.0000e-04]],
                 [[1.1146e+07, 5.0000e-04]],
                 [[1.3376e+07, 5.0000e-04]],
                 [[1.6051e+07, 5.0000e-04]],
                 [[1.9261e+07, 5.0000e-04]],
                 [[2.3113e+07, 5.0000e-04]],
                 [[2.7736e+07, 5.0000e-04]],
                 [[3.3283e+07, 5.0000e-04]],
                 [[3.9939e+07, 5.0000e-04]],
                 [[4.7927e+07, 5.0000e-04]],
                 [[5.7512e+07, 5.0000e-04]],
                 [[6.9015e+07, 5.0000e-04]]])}
    

## Visualisierung Sampling
Darstellung Input: `sigma_t` und `delta_epsilon`


```python
from pina import Plotter

pl = Plotter()
pl.plot_samples(problem=problem, filename=f'./{graph_folder}/{img_visual_sampling}{img_extensions}', variables=['delta_epsilon','sigma_t'])
display(Markdown('![Result of sampling](' + f'./{graph_folder}/{img_visual_sampling}{img_extensions}' + ')'))
```


![Result of sampling](./graph/visual_sampling.png)


# Training eines Physics-Informed Neural Networks (PINN) mit PINA

Dieser Code definiert und trainiert ein **Physics-Informed Neural Network (PINN)** zur Lösung des Problems in PINA.

- **Modell (`FeedForward`)**: Ein neuronales Netz mit drei versteckten Schichten (`[50, 50, 50]`), das mit der ReLU-Aktivierungsfunktion arbeitet.
- **PINN-Objekt (`PINN`)**: Erstellt das PINN-Modell, das die physikalischen Randbedingungen des Problems berücksichtigt.
- **TensorBoard-Logger (`TensorBoardLogger`)**: Speichert Trainingsmetriken zur Visualisierung.
- **Trainer (`Trainer`)**: Führt das Training für 1500 Epochen mit Batch-Größe 10 durch.
- **Training starten (`trainer.train()`)**: Startet den Optimierungsprozess und protokolliert die Metriken.

Am Ende wird die **finale Loss-Funktion** ausgegeben, um die Trainingsqualität zu bewerten.

**Mehr zu `Trainer`:**  
[PINA-Dokumentation – Trainer](https://mathlab.github.io/PINA/_rst/trainer.html)



```python
from pina import Trainer
from pina.solvers import PINN
from pina.model import FeedForward
from pina.callbacks import MetricTracker
import torch
from pytorch_lightning.loggers import TensorBoardLogger  # Import TensorBoard Logger

if debug_mode:
    print('Debugging Info:')
    # Überprüfen der Größe der Eingabepunkte und Ausgabepunkte
    print("‼️Länge der Eingabepunkte (input_pts):", len(problem.input_pts))
    print("‼️Länge der Ausgabepunkte (output_pts):", len(problem.output_pts))

# Model erstellen
model = FeedForward(
    layers=[50, 50, 50],
    func=torch.nn.ReLU,
    output_dimensions=len(problem.output_variables),
    input_dimensions=len(problem.input_variables)
)

# PINN-Objekt erstellen
pinn = PINN(problem, model)

# TensorBoard-Logger
logger = TensorBoardLogger("tensorboard_logs", name="pina_experiment")

# Trainer erstellen mit TensorBoard-Logger
trainer = Trainer(
    solver=pinn,
    max_epochs=1000,
    callbacks=[MetricTracker()],
    batch_size=10,
    accelerator='cpu',
    logger=logger,
    enable_model_summary=False,
)


# Training starten
trainer.train()

print('\nFinale Loss Werte')
# Inspect final loss
trainer.logged_metrics
```

    Debugging Info:
    ‼️Länge der Eingabepunkte (input_pts): 1
    ‼️Länge der Ausgabepunkte (output_pts): 100
    

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs
    C:\Users\hab185\Documents\00_Tim\01_Implementierung\pina_oedometer\venv\Lib\site-packages\pytorch_lightning\loops\fit_loop.py:310: The number of training batches (10) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.
    


    Training: |                                                                                      | 0/? [00:00<…


    `Trainer.fit` stopped: `max_epochs=1000` reached.
    

    
    Finale Loss Werte
    




    {'data_loss': tensor(0.0171), 'mean_loss': tensor(0.0171)}



## **Visualisierung der Modellvorhersage für delta_sigma**

Dieser Code erstellt einen **Plot der wahren Werte (`delta_sigma`)** im Vergleich zur **Vorhersage des neuronalen Netzwerks**.

- **Datenvorbereitung (`input_data`)**: Die Eingabedaten (`sigma_t` und `delta_epsilon`) werden als `LabelTensor` für das trainierte Modell erstellt.
- **Modellvorhersage (`pinn(input_data)`)**: Das trainierte PINN-Modell gibt eine Prognose für `delta_sigma` aus.
- **Plot-Erstellung mit `matplotlib`**:  
  - Die wahre Lösung (`delta_sigma`) wird als **blaue gestrichelte Linie** dargestellt.  
  - Die Vorhersage des neuronalen Netzwerks wird als **rote durchgezogene Linie** geplottet.  

**Zusätzlicher Schritt:**  
Die Nutzung von `matplotlib` war notwendig, da die interne Plot-Funktion von PINA `pl.plot()` das Diagramm nicht wie in den Tutorials erwartungsgemäß generierte, selbst wenn `delta_epsilon` auf einen fixen Wert gesetzt wurde. Dies könnte auf eine fehlerhafte Nutzung der Funktion oder auf eine Inkompatibilität in der Darstellung zurückzuführen sein.


```python
display_data_loss_table(data_dict=data_dict, delta_sigma_pred=pinn(input_points_combined).detach().numpy(), max_i=20)
plot_prediction_vs_true_solution(pinn=pinn, data_dict=data_dict, graph_folder=graph_folder, img_visual_prediction_vs_truesolution=img_visual_prediction_vs_truesolution, 
                                     img_extensions=img_extensions, y_axis='total_epsilon', max_i=20)
```


### Data-Loss bis sigma_19

| Index | total_epsilon | delta_epsilon | True sigma_t+1 | Predicted sigma_t+1 | Loss (True - Predicted) |
|--|--------------|--------------|---------------|--------------------|------------------------|
| 0 | 0 | 0.0005 | 0.2 | 0.2 | -0.0 |
| 1 | 0.0005 | 0.0005 | 0.24 | 0.24 | -0.0 |
| 2 | 0.001 | 0.0005 | 0.288 | 0.288 | -0.0 |
| 3 | 0.0015 | 0.0005 | 0.3456 | 0.3452 | 0.0004 |
| 4 | 0.002 | 0.0005 | 0.4147 | 0.415 | -0.0002 |
| 5 | 0.0025 | 0.0005 | 0.4977 | 0.4993 | -0.0016 |
| 6 | 0.003 | 0.0005 | 0.5972 | 0.5987 | -0.0015 |
| 7 | 0.0035 | 0.0005 | 0.7166 | 0.7156 | 0.001 |
| 8 | 0.004 | 0.0005 | 0.86 | 0.8566 | 0.0033 |
| 9 | 0.0045 | 0.0005 | 1.032 | 1.0211 | 0.0108 |
| 10 | 0.005 | 0.0005 | 1.2383 | 1.2223 | 0.0161 |
| 11 | 0.0055 | 0.0005 | 1.486 | 1.4768 | 0.0092 |
| 12 | 0.006 | 0.0005 | 1.7832 | 1.7816 | 0.0016 |
| 13 | 0.0065 | 0.0005 | 2.1399 | 2.1474 | -0.0076 |
| 14 | 0.007 | 0.0005 | 2.5678 | 2.5864 | -0.0185 |
| 15 | 0.0075 | 0.0005 | 3.0814 | 3.1131 | -0.0317 |
| 16 | 0.008 | 0.0005 | 3.6977 | 3.7477 | -0.05 |
| 17 | 0.0085 | 0.0005 | 4.4372 | 4.51 | -0.0728 |
| 18 | 0.009 | 0.0005 | 5.3247 | 5.4129 | -0.0882 |
| 19 | 0.0095 | 0.0005 | 6.3896 | 6.4934 | -0.1038 |




![Prediction vs True Solution](./graph/visual_prediction-vs-truesolution.png)<br>**Hinweis:** Datenpunkte liegen sehr nahe beieinander.


## Visualisierung Error-Result


```python
pl.plot(solver=pinn, filename=f'./{graph_folder}/{img_nn_result_error}{img_extensions}')
display(Markdown('![NN Error result](' + f'./{graph_folder}/{img_nn_result_error}{img_extensions}' + ')'))
```


![NN Error result](./graph/img_nn_result_error.png)


## Visualisierung Loss-Kurve



```python
# plotting the solution
pl.plot_loss(trainer, label='mean_loss', logy=True, filename=f'./{graph_folder}/{img_visual_loss}{img_extensions}')
display(Markdown('![Loss Kurve](' + f'./{graph_folder}/{img_visual_loss}{img_extensions}' + ')'))
```


![Loss Kurve](./graph/visual_loss.png)


# Testdaten (1 Input-Wert) $\Delta\epsilon=0,0005$


```python
new_data = extract_excel(file_path="files/oedometer/oedo_trainingsdata_compare.xlsx", sheet_name="Res", selected_columns=[1, 2, 3, 5], row_start_range=0)

# Erstelle die Eingabedaten als LabelTensor für das trainierte Modell
input_data = LabelTensor(torch.tensor(
    np.column_stack((new_data['sigma_t'], new_data['delta_epsilon'])), dtype=torch.float
), ['sigma_t', 'delta_epsilon'])

display_data_loss_table(data_dict=new_data, delta_sigma_pred=pinn(input_data).detach().numpy(), max_i=20)
plot_prediction_vs_true_solution(pinn=pinn, data_dict=new_data, graph_folder=graph_folder, img_visual_prediction_vs_truesolution=img_visual_prediction_vs_truesolution_comp0, 
                                     img_extensions=img_extensions, y_axis='total_epsilon', max_i=20, plot_type="scatter")
```

    {'sigma_t': array([1500,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0], dtype=int64), 'total_epsilon': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          dtype=int64), 'delta_epsilon': array([0.0005, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
           0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
           0.    , 0.    , 0.    , 0.    ]), 'delta_sigma': array([300,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0], dtype=int64)}
    


### Data-Loss bis sigma_19

| Index | total_epsilon | delta_epsilon | True sigma_t+1 | Predicted sigma_t+1 | Loss (True - Predicted) |
|--|--------------|--------------|---------------|--------------------|------------------------|
| 0 | 0 | 0.0005 | 300 | 300.1018 | -0.1018 |
| 1 | 0 | 0.0 | 0 | 0.1316 | -0.1316 |
| 2 | 0 | 0.0 | 0 | 0.1316 | -0.1316 |
| 3 | 0 | 0.0 | 0 | 0.1316 | -0.1316 |
| 4 | 0 | 0.0 | 0 | 0.1316 | -0.1316 |
| 5 | 0 | 0.0 | 0 | 0.1316 | -0.1316 |
| 6 | 0 | 0.0 | 0 | 0.1316 | -0.1316 |
| 7 | 0 | 0.0 | 0 | 0.1316 | -0.1316 |
| 8 | 0 | 0.0 | 0 | 0.1316 | -0.1316 |
| 9 | 0 | 0.0 | 0 | 0.1316 | -0.1316 |
| 10 | 0 | 0.0 | 0 | 0.1316 | -0.1316 |
| 11 | 0 | 0.0 | 0 | 0.1316 | -0.1316 |
| 12 | 0 | 0.0 | 0 | 0.1316 | -0.1316 |
| 13 | 0 | 0.0 | 0 | 0.1316 | -0.1316 |
| 14 | 0 | 0.0 | 0 | 0.1316 | -0.1316 |
| 15 | 0 | 0.0 | 0 | 0.1316 | -0.1316 |
| 16 | 0 | 0.0 | 0 | 0.1316 | -0.1316 |
| 17 | 0 | 0.0 | 0 | 0.1316 | -0.1316 |
| 18 | 0 | 0.0 | 0 | 0.1316 | -0.1316 |
| 19 | 0 | 0.0 | 0 | 0.1316 | -0.1316 |




![Prediction vs True Solution](./graph/visual_prediction-vs-truesolution_comp0.png)<br>**Hinweis:** Datenpunkte liegen sehr nahe beieinander.


# Testwerte (2 Input-Wert) $\Delta\epsilon=0,0005$


```python
new_data = extract_excel(file_path="files/oedometer/oedo_trainingsdata_compare2.xlsx", sheet_name="Res", selected_columns=[1, 2, 3, 5], row_start_range=0)

# Erstelle die Eingabedaten als LabelTensor für das trainierte Modell
input_data = LabelTensor(torch.tensor(
    np.column_stack((new_data['sigma_t'], new_data['delta_epsilon'])), dtype=torch.float
), ['sigma_t', 'delta_epsilon'])

display_data_loss_table(data_dict=new_data, delta_sigma_pred=pinn(input_data).detach().numpy(), max_i=20)
plot_prediction_vs_true_solution(pinn=pinn, data_dict=new_data, graph_folder=graph_folder, img_visual_prediction_vs_truesolution=img_visual_prediction_vs_truesolution_comp1, 
                                     img_extensions=img_extensions, y_axis='total_epsilon', max_i=20, plot_type="scatter")
```

    {'sigma_t': array([1500,    0,    0,    0,    0,    0,    0,    0,    0,  854,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0], dtype=int64), 'total_epsilon': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          dtype=int64), 'delta_epsilon': array([0.0005, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
           0.    , 0.0005, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
           0.    , 0.    , 0.    , 0.    ]), 'delta_sigma': array([300. ,   0. ,   0. ,   0. ,   0. ,   0. ,   0. ,   0. ,   0. ,
           170.8,   0. ,   0. ,   0. ,   0. ,   0. ,   0. ,   0. ,   0. ,
             0. ,   0. ])}
    


### Data-Loss bis sigma_19

| Index | total_epsilon | delta_epsilon | True sigma_t+1 | Predicted sigma_t+1 | Loss (True - Predicted) |
|--|--------------|--------------|---------------|--------------------|------------------------|
| 0 | 0 | 0.0005 | 300.0 | 300.1018 | -0.1018 |
| 1 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 2 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 3 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 4 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 5 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 6 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 7 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 8 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 9 | 0 | 0.0005 | 170.8 | 170.9017 | -0.1017 |
| 10 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 11 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 12 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 13 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 14 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 15 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 16 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 17 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 18 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 19 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |




![Prediction vs True Solution](./graph/visual_prediction-vs-truesolution_comp1.png)<br>**Hinweis:** Datenpunkte liegen sehr nahe beieinander.


# Testwerte (2 Input-Wert) $\Delta\epsilon=0,001$


```python
new_data = extract_excel(file_path="files/oedometer/oedo_trainingsdata_compare3.xlsx", sheet_name="Res", selected_columns=[1, 2, 3, 5], row_start_range=0)

# Erstelle die Eingabedaten als LabelTensor für das trainierte Modell
input_data = LabelTensor(torch.tensor(
    np.column_stack((new_data['sigma_t'], new_data['delta_epsilon'])), dtype=torch.float
), ['sigma_t', 'delta_epsilon'])

display_data_loss_table(data_dict=new_data, delta_sigma_pred=pinn(input_data).detach().numpy(), max_i=20)
plot_prediction_vs_true_solution(pinn=pinn, data_dict=new_data, graph_folder=graph_folder, img_visual_prediction_vs_truesolution=img_visual_prediction_vs_truesolution_comp2, 
                                     img_extensions=img_extensions, y_axis='total_epsilon', max_i=20, plot_type="scatter")
```

    {'sigma_t': array([1500,    0,    0,    0,    0,    0,    0,    0,    0,  854,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0], dtype=int64), 'total_epsilon': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          dtype=int64), 'delta_epsilon': array([0.001 , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
           0.    , 0.0005, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
           0.    , 0.    , 0.    , 0.    ]), 'delta_sigma': array([600. ,   0. ,   0. ,   0. ,   0. ,   0. ,   0. ,   0. ,   0. ,
           341.6,   0. ,   0. ,   0. ,   0. ,   0. ,   0. ,   0. ,   0. ,
             0. ,   0. ])}
    


### Data-Loss bis sigma_19

| Index | total_epsilon | delta_epsilon | True sigma_t+1 | Predicted sigma_t+1 | Loss (True - Predicted) |
|--|--------------|--------------|---------------|--------------------|------------------------|
| 0 | 0 | 0.001 | 600.0 | 300.1018 | 299.8982 |
| 1 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 2 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 3 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 4 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 5 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 6 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 7 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 8 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 9 | 0 | 0.0005 | 341.6 | 170.9017 | 170.6982 |
| 10 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 11 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 12 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 13 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 14 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 15 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 16 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 17 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 18 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |
| 19 | 0 | 0.0 | 0.0 | 0.1316 | -0.1316 |




![Prediction vs True Solution](./graph/visual_prediction-vs-truesolution_comp2.png)<br>**Hinweis:** Datenpunkte liegen sehr nahe beieinander.


Gemäß statischem Trainingswert für $\Delta\epsilon$ wurde keine korrekte Prognose vorgenommen.
