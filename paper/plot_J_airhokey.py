import os
import numpy as np
import matplotlib.pyplot as plt

def read_file(path):
    means = []
    stds = []
    samples = {}
    save = {}
    name = "_".join(os.path.basename(path)[:-4].split("_")[1:])
    with open(path, 'r') as fh:
        lines = fh.readlines()
        for i, line in enumerate(lines):
            data = line.strip().split(',')
            if i > 0:
                for k, d in enumerate(data[1:]):
                    if save[k]:
                        if len(d):
                            if d[0] == '"':
                                d = d[1:-1]
                            if d != '':
                                samples[k].append(float(d))
            else:
                for k, d in enumerate(data[1:]):
                    save[k] = True
                    if "__MIN" in d or "__MAX" in d:
                        save[k] = False
                    samples[k] = []


                #data_line = []
                #for d in data[1:]:
                #    d_ = d[1:-1]
                #    if d_ != '':
                #        data_line.append(float(d_))
                #means.append(np.mean(data_line))
                ##means.append(np.median(data_line))
                ##stds.append(np.std(data_line) / len(data_line) ** 0.5)
                #stds.append(np.std(data_line))
    # smoothing
    s = 0.1
    samples_filtered = {}
    for k, v in samples.items():
        if len(v) == 0:
            continue
        a = [v[0]]
        for i in range(1, len(v)):
            a.append(a[-1] * (1 - s) + v[i] * s)
        samples_filtered[k] = np.array(a)

    max_length = max(*[len(v) for v in samples_filtered.values()])
    # aggregation
    for i in range(max_length):
        data = [samples_filtered[k][i] for k in samples_filtered.keys() if i < len(samples_filtered[k])]
        means.append(np.mean(data))
        stds.append(np.std(data))
        #stds.append(np.std(data) / len(data) ** 0.5)
    return np.array(means), np.array(stds)

def read_steps_file(path):
    m, s = read_file(path)
    return m[1:], s[1:]

def read_eplen_file(path):
    m, s = read_file(path)
    m = np.cumsum(m)
    m *= 256
    s *= 256
    return m, s#m[1:], s[1:]

def read_stats(path):
    J_m, J_s = read_file(path)
    steps_m, steps_s = read_eplen_file(path.replace("J_sto", "ep_len"))
    l_J = len(J_m)
    l_steps = len(steps_m)
    l = min(l_J, l_steps)
    J_m = J_m[:l]
    J_s = J_s[:l]
    J_m_f = [J_m[0]]
    J_s_f = [J_s[0]]
    s = 0.1
    for i in range(1, len(J_m)):
        J_m_f.append(J_m_f[-1] * (1 - s) + J_m[i] * s)
        J_s_f.append(J_s_f[-1] * (1 - s) + J_s[i] * s)
    #J_m = np.array(J_m_f)
    #J_s_ = np.array(J_s_f)
    #plt.plot(J_s)
    #plt.plot(J_s_)
    #plt.show()
    steps_m = steps_m[:l]
    steps_s = steps_s[:l]
    return J_m, J_s, steps_m, steps_s


if __name__ == '__main__':
    results_dir = "results/air_hockey_fixed/"
    ppolag_J_path = os.path.join(os.path.dirname(__file__), results_dir,  "ppolag_J_sto.csv")
    ppolag_J_m, ppolag_J_s, ppolag_steps_m, ppolag_steps_s = read_stats(ppolag_J_path)

    trpolag_J_path = os.path.join(os.path.dirname(__file__), results_dir,  "trpolag_J_sto.csv")
    trpolag_J_m, trpolag_J_s, trpolag_steps_m, trpolag_steps_s = read_stats(trpolag_J_path)

    pcpo_J_path = os.path.join(os.path.dirname(__file__), results_dir,  "pcpo_J_sto.csv")
    pcpo_J_m, pcpo_J_s, pcpo_steps_m, pcpo_steps_s = read_stats(pcpo_J_path)

    ours_uns_J_path = os.path.join(os.path.dirname(__file__), results_dir,  "ours_unstructured_J_sto.csv")
    ours_uns_J_m, ours_uns_J_s, ours_uns_steps_m, ours_uns_steps_s = read_stats(ours_uns_J_path)

    ours_J_path = os.path.join(os.path.dirname(__file__), results_dir,  "ours_J_sto.csv")
    ours_J_m, ours_J_s, ours_steps_m, ours_steps_s = read_stats(ours_J_path)

    promp_J_path = os.path.join(os.path.dirname(__file__), results_dir,  "promp_J_sto.csv")
    promp_J_m, promp_J_s, promp_steps_m, promp_steps_s = read_stats(promp_J_path)

    prodmp_J_path = os.path.join(os.path.dirname(__file__), results_dir,  "prodmp_J_sto.csv")
    prodmp_J_m, prodmp_J_s, prodmp_steps_m, prodmp_steps_s = read_stats(prodmp_J_path)

    atacom_sac_J_path = os.path.join(os.path.dirname(__file__), results_dir,  "atacom_sac_J_sto.csv")
    atacom_sac_J_m, atacom_sac_J_s, atacom_sac_steps_m, atacom_sac_steps_s = read_stats(atacom_sac_J_path)

    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(12, 6))
    plt.plot(ours_steps_m, ours_J_m, label='CNP3O-PK')
    plt.fill_between(ours_steps_m, ours_J_m - ours_J_s, ours_J_m + ours_J_s, alpha=0.2)
    plt.plot(ours_uns_steps_m, ours_uns_J_m, label='CNP3O')
    plt.fill_between(ours_uns_steps_m, ours_uns_J_m - ours_uns_J_s, ours_uns_J_m + ours_uns_J_s, alpha=0.2)
    plt.plot(promp_steps_m, promp_J_m, label='CNP3O-ProMP')
    plt.fill_between(promp_steps_m, promp_J_m - promp_J_s, promp_J_m + promp_J_s, alpha=0.2)
    plt.plot(prodmp_steps_m, prodmp_J_m, label='CNP3O-ProDMP')
    plt.fill_between(prodmp_steps_m, prodmp_J_m - prodmp_J_s, prodmp_J_m + prodmp_J_s, alpha=0.2)
    plt.plot(atacom_sac_steps_m, atacom_sac_J_m, label='ATACOM-SAC')
    plt.fill_between(atacom_sac_steps_m, atacom_sac_J_m - atacom_sac_J_s, atacom_sac_J_m + atacom_sac_J_s, alpha=0.2)
    plt.plot(trpolag_steps_m, trpolag_J_m, label='TRPOLag')
    plt.fill_between(trpolag_steps_m, trpolag_J_m - trpolag_J_s, trpolag_J_m + trpolag_J_s, alpha=0.2)
    plt.plot(ppolag_steps_m, ppolag_J_m, label='PPOLag')
    plt.fill_between(ppolag_steps_m, ppolag_J_m - ppolag_J_s, ppolag_J_m + ppolag_J_s, alpha=0.2)
    plt.plot(pcpo_steps_m, pcpo_J_m, label='PCPO')
    plt.fill_between(pcpo_steps_m, pcpo_J_m - pcpo_J_s, pcpo_J_m + pcpo_J_s, alpha=0.2)
    plt.xlim(0, 2.35e7)
    plt.ylim(0, 110.)
    plt.xlabel('Steps')
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
    #           fancybox=False, shadow=False, frameon=False, ncol=6)
    plt.tight_layout(pad=2)
    plt.show()
