import matplotlib.pyplot as plt

def show_data():
    oedo_para = {
        "max_n": 48,
        "e_0": 1,
        "C_c": 0.005,
        "delta_epsilon": -0.0005,
        "sigma_t": -1.00,
    }

    oedo_model = 0

    if oedo_model == 0:
        from classes.classOedometerSimple import Oedometer
    elif oedo_model == 1:
        from classes.classOedometerImproved import Oedometer

    oedo = Oedometer(**oedo_para)

    dict_df = {
        "|epsilon|" : [],
        "e_s" : [],
        "delta_sigma" : [],
        "sigma" : []
    }

    for i in range(len(oedo.delta_sigma)):
        dict_df["|epsilon|"].append(oedo.total_epsilon[i])
        dict_df["e_s"].append(oedo.e_s[i])
        dict_df["delta_sigma"].append(oedo.delta_sigma[i])
        dict_df["sigma"].append(oedo.sigma_t[i])

    import pandas as pd
    df = pd.DataFrame(dict_df)

    print(oedo.sigma_t)
    print(oedo.total_epsilon)

    # Plot setup
    fig, axs = plt.subplots(1, 2, figsize=(15, 7), sharey=True)

    # X-Werte vorbereiten
    x_vals = oedo.sigma_t[:39]
    y_vals = oedo.total_epsilon[:39]

    # Linearer Plot (links)
    axs[0].plot(x_vals, y_vals, linewidth=1, label="$\Delta\sigma_{true}$")
    axs[0].grid(color='grey', linestyle='-', linewidth=.75)
    axs[0].set_xlabel(r"stress $\sigma$ [kPa]")
    axs[0].set_ylabel(r"strain $\epsilon$ [-]")
    axs[0].set_title("Linear")
    axs[0].legend()

    # Logarithmischer Plot (rechts)
    axs[1].plot([abs(x) for x in x_vals], y_vals, linewidth=1, label="$\Delta\sigma_{true}$")
    axs[1].set_xscale("log")
    axs[1].grid(color='grey', linestyle='-', linewidth=0.75)
    axs[1].set_xlabel(r"stress $\sigma$ [kPa]")
    axs[1].set_title("Log")
    axs[1].legend()

    fig.suptitle(
        "Spannungsdehnungs Verlauf" + "\n" + rf"mit $\sigma_0$={oedo_para['sigma_t']} und $\Delta\epsilon$={oedo_para["delta_epsilon"]}",
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    return df

show_data()
