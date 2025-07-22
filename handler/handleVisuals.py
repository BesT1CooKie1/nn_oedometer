from matplotlib import pyplot as plt
import torch

def plot_result_graph(model, oedo, iterations=48):
    sigma_true = oedo.sigma_t
    delta_sigma = oedo.delta_sigma
    total_epsilon = oedo.total_epsilon
    e_s_true = oedo.e_s

    # print(sigma_true)
    # print(e_s_true)
    model.eval()
    delta_sigma_pred = []
    e_s_true_plot = []

    sigma_pred = []
    e_s_pred_list = []
    total_strain = []
    delta_sigma_pred_prop = []
    with torch.no_grad():
        for i in range(iterations):
            # Konvertierung sigma zu Tensor
            sigma_true_tensor = torch.tensor(
                sigma_true[i], dtype=torch.float
            ).unsqueeze(-1)

            # Schätzung des E_s Wertes mit wahrem sigma Wert
            e_s_pred = model(sigma_true_tensor)

            # # Ermittlung delta_sigma mit geschätztem Wert
            # delta_sigma_pred_val = e_s_pred * oedo.delta_epsilon[i]

            # Ermittlung sigma mit geschätztem Wert
            sigma_pred = -(e_s_pred / ((1 + oedo.e_0[0]) / oedo.C_c[0]))

            # Ermittlung neues E_s,i+1 mit geschätztem Wert
            e_s_pred_prop = model(sigma_pred)

            # Ermittlung Delta_Sigma mit geschätztem Wert
            delta_sigma_pred_val = e_s_pred_prop * oedo.delta_epsilon[i]

            # Zusammenführung in Listen
            delta_sigma_pred.append(delta_sigma_pred_val)
            delta_sigma_pred_prop.append(delta_sigma_pred_val)
            total_strain.append(total_epsilon[i])
            e_s_pred_list.append(e_s_pred)

    # Plot der Losskurve
    plt.scatter(
        delta_sigma_pred, total_strain, label=r"$\Delta\sigma_{pred}$"
    ).set_color("red")
    plt.scatter(
        delta_sigma_pred_prop, total_strain, marker="x", label=r"$\Delta\sigma_{pred,propagated}$"
    ).set_color("green")
    plt.plot(delta_sigma, total_strain, linewidth=1, label=r"$\Delta\sigma_{true}$")

    plt.grid(color='grey', linestyle='-', linewidth=.5)
    plt.xlabel(r"$\Delta\sigma$ [kPa]")
    plt.ylabel(r"$\epsilon$ [-]")
    plt.title(
        rf"Spannungsdehnungs Verlauf mit $\sigma_0={sigma_true[0]}$ und $\Delta\epsilon={oedo.delta_epsilon[0]}$"
    )

    plt.legend()
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.show()

    return [e_s_true, e_s_pred_list], [delta_sigma, delta_sigma_pred]


def plot_result_dataframe(pd, e_s_list, delta_sigma_list):
    key_1 = r"$E_{s,i,true}$"
    key_2 = r"$E_{s,i,pred}$"
    key_3 = r"$\Delta\sigma_{true}$"
    key_4 = r"$\Delta\sigma_{pred}$"
    key_34 = r"$\Delta \sigma$"
    key_12 = r"$\Delta E_s$"

    dict_diff = {
        key_1: [],
        key_2: [],
        key_12: [],
        "|": ["|"] * len(e_s_list[0]),
        key_3: [],
        key_4: [],
        key_34: [],
    }

    for e_s_true, e_s_pred, d_sigma_true, d_sigma_pred in zip(e_s_list[0], e_s_list[0], delta_sigma_list[0],
                                                              delta_sigma_list[1]):
        delta_sigma = d_sigma_true - d_sigma_pred.item()
        dict_diff[key_1].append(e_s_true)
        dict_diff[key_2].append(e_s_pred)
        dict_diff[key_12].append(e_s_true - e_s_pred)
        dict_diff[key_3].append(d_sigma_true)
        dict_diff[key_4].append(d_sigma_pred.item())
        dict_diff[key_34].append(delta_sigma)
    print(sum(dict_diff[key_34])/len(dict_diff[key_34]))
    return pd.DataFrame(dict_diff)