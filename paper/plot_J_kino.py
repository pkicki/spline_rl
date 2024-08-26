import os
import numpy as np
import matplotlib.pyplot as plt

from plot_J_airhokey import read_file

def read_stats(path):
    J_m, J_s = read_file(path)
    steps = np.arange(0, len(J_m)) * 256 * 100
    return J_m, J_s, steps


dir = "kino_fixed"

pcpo_J_path = os.path.join(os.path.dirname(__file__), f"results/{dir}/pcpo_J_sto.csv")
pcpo_J_m, pcpo_J_s, pcpo_steps = read_stats(pcpo_J_path)

trpolag_J_path = os.path.join(os.path.dirname(__file__), f"results/{dir}/trpolag_J_sto.csv")
trpolag_J_m, trpolag_J_s, trpolag_steps = read_stats(trpolag_J_path)

ppolag_J_path = os.path.join(os.path.dirname(__file__), f"results/{dir}/ppolag_J_sto.csv")
ppolag_J_m, ppolag_J_s, ppolag_steps = read_stats(ppolag_J_path)

ours_uns_J_path = os.path.join(os.path.dirname(__file__), f"results/{dir}/ours_unstructured_J_sto.csv")
ours_uns_J_m, ours_uns_J_s, ours_uns_steps = read_stats(ours_uns_J_path)

ours_J_path = os.path.join(os.path.dirname(__file__), f"results/{dir}/ours_structured_J_sto.csv")
ours_J_m, ours_J_s, ours_steps = read_stats(ours_J_path)

promp_J_path = os.path.join(os.path.dirname(__file__), f"results/{dir}/promp_structured_J_sto.csv")
promp_J_m, promp_J_s, promp_steps = read_stats(promp_J_path)

promp_uns_J_path = os.path.join(os.path.dirname(__file__), f"results/{dir}/promp_unstructured_J_sto.csv")
promp_uns_J_m, promp_uns_J_s, promp_uns_steps = read_stats(promp_uns_J_path)

prodmp_J_path = os.path.join(os.path.dirname(__file__), f"results/{dir}/prodmp_structured_J_sto.csv")
prodmp_J_m, prodmp_J_s, prodmp_steps = read_stats(prodmp_J_path)

prodmp_uns_J_path = os.path.join(os.path.dirname(__file__), f"results/{dir}/prodmp_unstructured_J_sto.csv")
prodmp_uns_J_m, prodmp_uns_J_s, prodmp_uns_steps = read_stats(prodmp_uns_J_path)

#pk_linestyle = "--"
pk_linestyle = "-"
plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(12, 6))

plt.plot(ours_uns_steps, ours_uns_J_m, label='CNP3O', color='tab:orange')
plt.fill_between(ours_uns_steps, ours_uns_J_m - ours_uns_J_s, ours_uns_J_m + ours_uns_J_s, alpha=0.2, color='tab:orange')
plt.plot(promp_uns_steps, promp_uns_J_m, label='CNP3O-ProMP', color='tab:green')
plt.fill_between(promp_uns_steps, promp_uns_J_m - promp_uns_J_s, promp_uns_J_m + promp_uns_J_s, alpha=0.2, color='tab:green')
plt.plot(prodmp_uns_steps, prodmp_uns_J_m, label='CNP3O-ProDMP', color='tab:red')
plt.fill_between(prodmp_uns_steps, prodmp_uns_J_m - prodmp_uns_J_s, prodmp_uns_J_m + prodmp_uns_J_s, alpha=0.2, color='tab:red')
plt.plot(trpolag_steps, trpolag_J_m, label='TRPOLag', color='tab:brown')
plt.fill_between(trpolag_steps, trpolag_J_m - trpolag_J_s, trpolag_J_m + trpolag_J_s, alpha=0.2, color='tab:brown')
plt.plot(ppolag_steps, ppolag_J_m, label='PPOLag', color='tab:pink')
plt.fill_between(ppolag_steps, ppolag_J_m - ppolag_J_s, ppolag_J_m + ppolag_J_s, alpha=0.2, color='tab:pink')
plt.plot(pcpo_steps, pcpo_J_m, label='PCPO', color='tab:gray')
plt.fill_between(pcpo_steps, pcpo_J_m - pcpo_J_s, pcpo_J_m + pcpo_J_s, alpha=0.2, color='tab:gray')
plt.ylim(2.5, 25.)

#plt.plot(ours_steps, ours_J_m, label='CNP3O-PK')
#plt.fill_between(ours_steps, ours_J_m - ours_J_s, ours_J_m + ours_J_s, alpha=0.2)
#plt.plot(promp_steps, promp_J_m, label='CNP3O-PK-ProMP', color='tab:green', linestyle=pk_linestyle)
#plt.fill_between(promp_steps, promp_J_m - promp_J_s, promp_J_m + promp_J_s, alpha=0.2, color='tab:green')
#plt.plot(prodmp_steps, prodmp_J_m, label='CNP3O-PK-ProDMP', color='tab:red', linestyle=pk_linestyle)
#plt.fill_between(prodmp_steps, prodmp_J_m - prodmp_J_s, prodmp_J_m + prodmp_J_s, alpha=0.2, color='tab:red')

plt.xlim(0, 0.9e7)
#plt.xlim(0, 0.7e7)
#plt.ylim(0, 110.)
plt.xlabel('Steps')
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
#           fancybox=False, shadow=False, frameon=False, ncol=6)
plt.tight_layout(pad=2)
plt.show()
