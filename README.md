# MODUL PINA (Python 3.12)

# Vorhersage des Ödometerversuches implementiert
Ziel war die Implementierung eines neuronalen Netzwerks zur Modellierung des Ödometerversuchs. Dabei wurden gegebene Input-Parameter verarbeitet, um Output-Parameter vorherzusagen. Die physikalischen Rahmenbedingungen wurden zunächst auf Null gesetzt, sodass das Modell ausschließlich auf der KI-basierten Struktur arbeitet, ohne physikalische Optimierungen durch Physical Informed Neural Networks (PINNs).
<br>
Diese grundlegende Umsetzung bildet die Basis für weiterführende Optimierungen, wie die Integration physikalischer Gesetzmäßigkeiten, die jedoch nicht Teil des initialen Arbeitsauftrags waren.

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

<br> 

Um das PINA-Model zu testen werden wir folgende vorberechnete Werte verwenden: `Input` { $\sigma_t$ }, `Output` { $E_s$ }.
<br>
### Variablendeklaration
- $\sigma_t$ = `sigma_t`
- $\Delta\epsilon$ = `delta_epsilon`
- $\sigma_{t+1}$ = `delta_sigma
- $E_s$ = `e_s`

# Generating random trainings data


```python
from random import randint

# Define input and output parameters
input_str = "sigma_t"
output_str = "e_s"

# Defining problem parameters
delta_epsilon=0.0005
C_c = 0.005
e_0 = 1.0
amount_trainings_data = 100

# Data preparation for 
oedo_para = {
    'max_n': 1,
    'e_0': e_0,
    'C_c': C_c,
    'delta_epsilon' : delta_epsilon,
}
```

# Load problem and generate  data from 00_problem_settings_functions.ipynb

Available classes: `Oedometer` <br>
Returns `list_input` and `list_output` as type `list` <br>
Returns `tensor_input` and `tensor_output` as type `tensor`


```python
%run 00_problem_settings_functions.ipynb

# Loads:
# Oedometer class

# Returns
# list_input: list
# list_output: list

# tensor_input: tensor
# tensor_output: tensor
```

# Show trainingsdata (List) as DataFrame
Type `list`: `list_input` and `list_output`


```python
import pandas as pd
from pandas import DataFrame

pd.DataFrame([[input_str] + list_input, [output_str] + list_output])
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>91</th>
      <th>92</th>
      <th>93</th>
      <th>94</th>
      <th>95</th>
      <th>96</th>
      <th>97</th>
      <th>98</th>
      <th>99</th>
      <th>100</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sigma_t</td>
      <td>20.0</td>
      <td>30.0</td>
      <td>15.0</td>
      <td>20.0</td>
      <td>18.0</td>
      <td>4.0</td>
      <td>33.0</td>
      <td>25.0</td>
      <td>6.0</td>
      <td>...</td>
      <td>44.0</td>
      <td>39.0</td>
      <td>50.0</td>
      <td>14.0</td>
      <td>5.0</td>
      <td>26.0</td>
      <td>16.0</td>
      <td>4.0</td>
      <td>24.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>e_s</td>
      <td>8000.0</td>
      <td>12000.0</td>
      <td>6000.0</td>
      <td>8000.0</td>
      <td>7200.0</td>
      <td>1600.0</td>
      <td>13200.0</td>
      <td>10000.0</td>
      <td>2400.0</td>
      <td>...</td>
      <td>17600.0</td>
      <td>15600.0</td>
      <td>20000.0</td>
      <td>5600.0</td>
      <td>2000.0</td>
      <td>10400.0</td>
      <td>6400.0</td>
      <td>1600.0</td>
      <td>9600.0</td>
      <td>8000.0</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 101 columns</p>
</div>



# Show trainingsdata (Tensor) as DataFrame
Type `tensor`: `tensor_input` and `tensor_output`


```python
tensor_data_df = pd.DataFrame(torch.cat((tensor_input, tensor_output), dim=1), columns = [input_str, output_str])
tensor_data_df
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
      <th>e_s</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20.0</td>
      <td>8000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30.0</td>
      <td>12000.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15.0</td>
      <td>6000.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20.0</td>
      <td>8000.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18.0</td>
      <td>7200.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>26.0</td>
      <td>10400.0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>16.0</td>
      <td>6400.0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>4.0</td>
      <td>1600.0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>24.0</td>
      <td>9600.0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>20.0</td>
      <td>8000.0</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 2 columns</p>
</div>



## Tensor to LabelTensor for PINA


```python
from pina.utils import LabelTensor

label_tensor_input = LabelTensor(tensor_input,[input_str])
label_tensor_output = LabelTensor(tensor_output, [output_str])
```

# Show trainingsdata (LabelTensor) as DataFrame
Type `LabelTensor`: `label_tensor_input` and `label_tensor_output`


```python
tensor_input_df = pd.DataFrame(torch.cat((label_tensor_input, label_tensor_output), dim=1), columns = [input_str, output_str])

print('Input Size: ', label_tensor_input.size())
print('Output Size: ', label_tensor_output.size(), '\n')
tensor_input_df
```

    Input Size:  torch.Size([100, 1])
    Output Size:  torch.Size([100, 1]) 
    





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
      <th>e_s</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20.0</td>
      <td>8000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30.0</td>
      <td>12000.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15.0</td>
      <td>6000.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20.0</td>
      <td>8000.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18.0</td>
      <td>7200.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>26.0</td>
      <td>10400.0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>16.0</td>
      <td>6400.0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>4.0</td>
      <td>1600.0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>24.0</td>
      <td>9600.0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>20.0</td>
      <td>8000.0</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 2 columns</p>
</div>



### **Definition eines PINN-Problems in PINA**  


```python
from pina.problem import AbstractProblem
from pina.domain import CartesianDomain
from pina import Condition

input_conditions = {'data': Condition(input=label_tensor_input, target=label_tensor_output),}

class SimpleODE(AbstractProblem):

    # Definition der Eingabe- und Ausgabevariablen basierend auf LabelTensor
    input_variables = label_tensor_input.labels
    output_variables = label_tensor_output.labels

    # Wertebereich
    domain = CartesianDomain({label_tensor_input: [0, 1]})#, 'delta_epsilon': [0, 1]})  # Wertebereich immer definieren!

    # Definition der Randbedingungen und (hier: nur) vorberechnetet Punkte
    conditions = input_conditions

    label_tensor_output=label_tensor_output

    # Methode zur Definition der "wahren Lösung" des Problems
    def truth_solution(self, pts):
        return torch.exp(pts.extract(label_tensor_input))

# Problem-Instanz erzeugen
problem = SimpleODE()

print('Input: ', problem.input_variables)
print('Output: ', problem.output_variables)
```

    Input:  ['sigma_t']
    Output:  ['e_s']


# Training eines Physics-Informed Neural Networks (PINN) mit PINA


```python
from pina import Trainer
from pina.solver import PINN
from pina.model import FeedForward
from pina.callback import MetricTracker
import torch.nn as nn
# Model erstellen
model = FeedForward(
    layers=[50,50,50],
    func=nn.ReLU,
    output_dimensions=len(problem.output_variables),
    input_dimensions=len(problem.input_variables)
)

# PINN-Objekt erstellen
pinn = PINN(problem, model)

# Trainer erstellen mit TensorBoard-Logger
trainer = Trainer(
    solver=pinn,
    max_epochs=1000,
    callbacks=[MetricTracker()],
    batch_size=16,
    accelerator='cpu',
    enable_model_summary=False,
)


# Training starten
trainer.train()

print('\nFinale Loss Werte')
# Inspect final loss
trainer.logged_metrics
```

    💡 Tip: For seamless cloud uploads and versioning, try installing [litmodels](https://pypi.org/project/litmodels/) to enable LitModelCheckpoint, which syncs automatically with the Lightning model registry.
    GPU available: True (cuda), used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs
    /home/mrschiller/Dokumente/git_projects/nn_oedometer_lstm/venv/lib/python3.12/site-packages/lightning/pytorch/trainer/setup.py: PossibleUserWarning: GPU available but not used. You can set it by doing `Trainer(accelerator='gpu')`.
    /home/mrschiller/Dokumente/git_projects/nn_oedometer_lstm/venv/lib/python3.12/site-packages/lightning/pytorch/trainer/connectors/logger_connector/logger_connector.py: UserWarning: Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the `lightning.pytorch` package, due to potential conflicts with other packages in the ML ecosystem. For this reason, `logger=True` will use `CSVLogger` as the default logger, unless the `tensorboard` or `tensorboardX` packages are found. Please `pip install lightning[extra]` or one of them to enable TensorBoard support by default
    /home/mrschiller/Dokumente/git_projects/nn_oedometer_lstm/venv/lib/python3.12/site-packages/lightning/pytorch/trainer/configuration_validator.py: PossibleUserWarning: You defined a `validation_step` but have no `val_dataloader`. Skipping val loop.
    /home/mrschiller/Dokumente/git_projects/nn_oedometer_lstm/venv/lib/python3.12/site-packages/lightning/pytorch/loops/fit_loop.py: PossibleUserWarning: The number of training batches (7) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.


    Epoch 999: 100%|█| 7/7 [00:00<00:00, 126.66it/s, v_num=0, data_loss_step=0.446, 

    `Trainer.fit` stopped: `max_epochs=1000` reached.


    Epoch 999: 100%|█| 7/7 [00:00<00:00, 113.17it/s, v_num=0, data_loss_step=0.446, 
    
    Finale Loss Werte





    {'data_loss_step': tensor(0.4465),
     'train_loss_step': tensor(0.4465),
     'data_loss_epoch': tensor(0.3878),
     'train_loss_epoch': tensor(0.3878)}




```python
import matplotlib.pyplot as plt

data_loss = trainer.callbacks[0].metrics["train_loss_epoch"].tolist()

plt.plot(data_loss, label="Loss")
plt.xlabel('Epochs')
plt.ylabel('Train Loss')
plt.show()
```


    
![png](output_18_0.png)
    


# Plot of stress–strain curve


```python
def plot_result(iterations=20, start_sigma=1, delta_epsilon=0.0005):
    oedo_para = {
    'max_n': iterations,
    'e_0': 1.0,
    'C_c': 0.005,
    'delta_epsilon' : delta_epsilon,
    'sigma_t' : start_sigma,
    }

    oedo = Oedometer(**oedo_para)
    sigma_true = oedo.sigma_t
    e_s_true = oedo.e_s
    
    # print(sigma_true)
    # print(e_s_true)
    model.eval()
    e_s_pred = []
    e_s_true_plot = []
    sigma_t = start_sigma
    sigma_pred = []
    with torch.no_grad():
        for i in range(iterations):
            sigma_true_tensor = torch.tensor(sigma_true[i], dtype=torch.float).unsqueeze(-1) 
            pred = model(sigma_true_tensor)
            e_s_pred.append(pred * sigma_true[i])
            e_s_true_plot.append(e_s_true[i] * sigma_true[i])
            sigma_t = sigma_t + pred * delta_epsilon
            sigma_pred.append(sigma_t)
            
    # Plot der Losskurve
    plt.scatter(sigma_true, e_s_pred, label='$E_{s,pred}$, $\sigma_{true}$').set_color("red")
    plt.scatter(sigma_pred, e_s_pred, label='$E_{s,pred}$, $\sigma_{pred}$')
    plt.plot(sigma_true, e_s_true_plot, label='$E_{s,true}$, $\sigma_{true}$')

    plt.gca().invert_yaxis()
    plt.xlabel('Sigma_t')
    plt.ylabel('Epsilon')
    plt.title(f'Spannungsdehnungs Verlauf mit $\sigma_0={start_sigma}$ und $\Delta\epsilon=0.0005$')
    plt.legend()
    plt.show()
plot_result()
```

    <>: SyntaxWarning: invalid escape sequence '\s'
    <>: SyntaxWarning: invalid escape sequence '\s'
    <>: SyntaxWarning: invalid escape sequence '\s'
    <>: SyntaxWarning: invalid escape sequence '\s'
    <>: SyntaxWarning: invalid escape sequence '\D'
    <>: SyntaxWarning: invalid escape sequence '\s'
    <>: SyntaxWarning: invalid escape sequence '\s'
    <>: SyntaxWarning: invalid escape sequence '\s'
    <>: SyntaxWarning: invalid escape sequence '\s'
    <>: SyntaxWarning: invalid escape sequence '\D'
    /tmp/ipykernel_6718/3899886523.py: SyntaxWarning: invalid escape sequence '\s'
    /tmp/ipykernel_6718/3899886523.py: SyntaxWarning: invalid escape sequence '\s'
    /tmp/ipykernel_6718/3899886523.py: SyntaxWarning: invalid escape sequence '\s'
    /tmp/ipykernel_6718/3899886523.py: SyntaxWarning: invalid escape sequence '\s'
    /tmp/ipykernel_6718/3899886523.py: SyntaxWarning: invalid escape sequence '\D'



    
![png](output_20_1.png)
    

