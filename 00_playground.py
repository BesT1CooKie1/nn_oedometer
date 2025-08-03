import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import sys
eps_0 = 0
eps_d = -0.0005
sigma_max = -1000 # kPa
sigma_min = -100 # kPa
sigma_prime_p = -1
sigma_0 = sigma_prime_p
e_0 = 1
c_c = 0.005
c_s = 0.002

c1 = - (1 + e_0) / 2 * (c_c + c_s) / (c_s * c_c)
c2 = - (1 + e_0) / 2 * (c_c - c_s) / (c_s * c_c)
print("C_1", c1, "C_2", c2)

sigma_0_list_compress = []
eps_list_compress = []
sigma_0_list_extension = []
eps_list_extension = []
e_s_list_compress = []
e_s_list_extension = []

def strain_stress_modul(eps_delta, sigma_0, eps_0):
    sigma_0_list = [0]
    eps_list = [0]
    e_s_list = [0]
    global sigma_0_list_compress, eps_list_compress, sigma_0_list_extension, eps_list_extension, e_s_list_compress, e_s_list_extension
    pressure = True
    while pressure:
        sigma_delta = c1 * sigma_0 * eps_delta + c2 * sigma_0 * abs(eps_delta)
        sigma_0 += sigma_delta
        eps_0 += eps_delta
        e_s = (c1 - c2) * sigma_0
        if sigma_0 < sigma_max:
            sigma_0 -= sigma_delta
            eps_0 -= eps_delta
            sigma_0_list_compress = sigma_0_list
            eps_list_compress = eps_list
            e_s_list_compress = e_s_list
            e_s_list = []
            sigma_0_list = []
            eps_list = []
            pressure = False
        else:
            sigma_0_list.append(sigma_0)
            eps_list.append(eps_0)
            e_s_list.append(e_s)
    tension = True
    eps_delta = abs(eps_delta)
    while tension:
        sigma_delta = c1 * sigma_0 * eps_delta + c2 * sigma_0 * abs(eps_delta)
        sigma_0 += sigma_delta
        eps_0 += eps_delta
        e_s = (c1 + c2) * sigma_0
        if sigma_0 > sigma_min:
            eps_list_extension = eps_list
            sigma_0_list_extension = sigma_0_list
            e_s_list_extension = e_s_list
            tension = False
        else:
            sigma_0_list.append(sigma_0)
            eps_list.append(eps_0)
            e_s_list.append(e_s)

strain_stress_modul(eps_d, sigma_0, eps_0)

data = {
        "eps_total": eps_list_compress + eps_list_extension,
        "eps_delta": len(e_s_list_compress) * [eps_d] + len(e_s_list_extension) * [abs(eps_d)],
        "sigma_0": sigma_0_list_compress + sigma_0_list_extension,

        "e_s": e_s_list_compress + e_s_list_extension,
        "state" : len(e_s_list_compress) * [1] + len(e_s_list_extension) * [0]
    }

df = pd.DataFrame(
    data=data,
    columns=["eps_total", "eps_delta", "sigma_0", "e_s", "state"] )
print(tabulate(df, headers='keys'))


plt.plot(sigma_0_list_compress, eps_list_compress, label="Belastung", color="red")
plt.plot([sigma_0_list_compress[-1]] + sigma_0_list_extension, [eps_list_compress[-1]] + eps_list_extension, label="Entlastung", color="blue")
plt.grid(color='grey', linestyle='-', linewidth=.75)
# plt.axvline(sigma_prime_p, linestyle='--', color='grey', label=r"$\sigma'_p$ (Vorbelastung)")
plt.axvline(sigma_max, linestyle='--', color='grey')
plt.text(
    sigma_max + 10,   # etwas rechts von der Linie
    plt.ylim()[1] * -.1,  # knapp unterhalb des oberen y-Werts
    r"$\sigma_{max}$ = " + str(sigma_max),
    color='grey',
    rotation=90,
    va='top',
    ha='left'
)
plt.axvline(sigma_min, linestyle='--', color='grey')
plt.text(
    sigma_min - 30,   # etwas rechts von der Linie
    plt.ylim()[1] * -.1,  # knapp unterhalb des oberen y-Werts
    r"$\sigma_{min}$ = " + str(sigma_min),
    color='grey',
    rotation=90,
    va='top',
    ha='left'
)
plt.xlabel(r"Stress $\sigma$ [kPa]")
plt.ylabel(r"Strain $\epsilon$ [-]")
plt.title("Einaxialer Kompressionsversuch (Einfachstes Modell)")
#plt.xscale("log")
plt.legend()
plt.tight_layout()
plt.show()