import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def compute_J(dataset, gamma=1.):
    """
    Compute the cumulative discounted reward of each episode in the dataset.

    Args:
        dataset (list): the dataset to consider;
        gamma (float, 1.): discount factor.

    Returns:
        The cumulative discounted reward of each episode in the dataset.

    """
    js = list()

    j = 0.
    episode_steps = 0
    for i in range(len(dataset)):
        j += gamma ** episode_steps * dataset[i][2]
        episode_steps += 1
        if dataset[i][-1] or i == len(dataset) - 1:
            js.append(j)
            j = 0.
            episode_steps = 0

    if len(js) == 0:
        return [0.]
    return js

def compute_episodes_length(dataset):
    """
    Compute the length of each episode in the dataset.

    Args:
        dataset (list): the dataset to consider.

    Returns:
        A list of length of each episode in the dataset.

    """
    lengths = list()
    l = 0
    for sample in dataset:
        l += 1
        if sample[-1] == 1:
            lengths.append(l)
            l = 0

    return lengths

def compute_stats(dataset, dataset_info):
    J = compute_J(dataset, 0.99)
    R = compute_J(dataset)

    states = np.array([sample[0] for sample in dataset])

    eps_length = compute_episodes_length(dataset)
    current_idx = 0
    success = []
    time_to_hit = []
    max_puck_vel = []
    scored = []
    joint_pos = []
    joint_vel = []
    ee_xlb = []
    ee_ylb = []
    ee_yub = []
    ee_zlb = []
    ee_zub = []
    ee_zeb = []
    for episode_len in eps_length:
        ep_states = states[current_idx:current_idx + episode_len]
        puck_pos = ep_states[:, :2]
        x_crit = puck_pos[:, 0] > 2.38
        y_crit = np.logical_and(puck_pos[:, 1] > -0.125, puck_pos[:, 1] < 0.125)
        crit = np.logical_and(x_crit, y_crit)
        #print(np.any(crit))
        #plt.plot(puck_pos[:, 0], puck_pos[:, 1])
        #plt.show()
        success.append(np.any(crit))
        #success.append(dataset_info["success"][current_idx + episode_len - 1])
        hit_time = dataset_info["hit_time"][current_idx + episode_len - 1]
        scored.append(dataset_info["success"][current_idx + episode_len - 1])
        if hit_time > 0:
            time_to_hit.append(hit_time)
        max_puck_vel.append(np.max(dataset_info["puck_velocity"][current_idx:current_idx + episode_len]))
        joint_pos.append(np.mean(dataset_info['joint_pos_constraint'][current_idx:current_idx+episode_len]))
        joint_vel.append(np.mean(dataset_info['joint_vel_constraint'][current_idx:current_idx+episode_len]))
        ee_xlb.append(np.mean(dataset_info['ee_xlb_constraint'][current_idx:current_idx+episode_len]))
        ee_ylb.append(np.mean(dataset_info['ee_ylb_constraint'][current_idx:current_idx+episode_len]))
        ee_yub.append(np.mean(dataset_info['ee_yub_constraint'][current_idx:current_idx+episode_len]))
        ee_zlb.append(np.mean(dataset_info['ee_zlb_constraint'][current_idx:current_idx+episode_len]))
        ee_zub.append(np.mean(dataset_info['ee_zub_constraint'][current_idx:current_idx+episode_len]))
        ee_zeb.append(np.mean(dataset_info['ee_zeb_constraint'][current_idx:current_idx+episode_len]))
        current_idx += episode_len
    return np.array(J), np.array(R), np.array(success), np.array(time_to_hit), np.array(max_puck_vel), np.array(eps_length), \
           np.array(joint_pos), np.array(joint_vel), np.array(ee_xlb), np.array(ee_ylb), np.array(ee_yub), \
           np.array(ee_zlb), np.array(ee_zub), np.array(ee_zeb)



path = os.path.join(os.path.dirname(__file__), "results/real/")
with open(os.path.join(path, "dataset_ATACOM.pkl"), 'rb') as f:
    dataset_atacom = pickle.load(f)
with open(os.path.join(path, "dataset_info_ATACOM.pkl"), 'rb') as f:
    dataset_info_atacom = pickle.load(f)
with open(os.path.join(path, "dataset_SplineRL.pkl"), 'rb') as f:
    dataset_ours = pickle.load(f)
with open(os.path.join(path, "dataset_info_SplineRL.pkl"), 'rb') as f:
    dataset_info_ours = pickle.load(f)

J_atacom, R_atacom, success_atacom, time_to_hit_atacom, max_puck_vel_atacom, episode_length_atacom, \
joint_pos_atacom, joint_vel_atacom, ee_xlb_atacom, ee_ylb_atacom, ee_yub_atacom, \
ee_zlb_atacom, ee_zub_atacom, ee_zeb_atacom \
= compute_stats(dataset_atacom, dataset_info_atacom)

J_ours, R_ours, success_ours, time_to_hit_ours, max_puck_vel_ours, episode_length_ours, \
joint_pos_ours, joint_vel_ours, ee_xlb_ours, ee_ylb_ours, ee_yub_ours, \
ee_zlb_ours, ee_zub_ours, ee_zeb_ours \
= compute_stats(dataset_ours, dataset_info_ours)
a = 0

metrics_list = ["success", "J", "R", "max_puck_vel", "joint_vel", "ee_zeb"]
metrics_names = {
    "success": "Score ratio [%]",
    "J": "Discounted reward",
    "R": "Reward",
    "max_puck_vel": "Maximal puck\n velocity [m/s]",
    "ee_zeb": "Table constraint\n violation [mm]",
    "joint_vel": "Maximal joint velocity\n violation [rad/s]"
}
scales = {
    "success": 1.,
    "J": 1.,
    "R": 1.,
    "max_puck_vel": 1.,
    "ee_zeb": 1e3,
    "joint_vel": 1.
}
atacom_list = [eval(m + "_atacom") for m in metrics_list]
ours_list = [eval(m + "_ours") for m in metrics_list]
color_ours = "tab:blue"
color_atacom = "tab:purple"

plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(20, 4))
w = 0.1
positions = [0.35, 0.45]
for i, (atacom, ours) in enumerate(zip(atacom_list, ours_list)):
    metric = metrics_list[i]
    ax = plt.subplot(1, len(metrics_list), i + 1)
    ax.set_title(metrics_names[metric])
    if metric == "success":
        ax.bar([positions[0]], [np.mean(ours) * 100.], color=color_ours, width=w)
        ax.bar([positions[1]], [np.mean(atacom) * 100.], color=color_atacom, width=w)
    else:
        bp = ax.boxplot([ours * scales[metric], atacom * scales[metric]], positions=[0.3, 0.5], labels=["ATACOM", "SplineRL"], showfliers=False, widths=w)
        bp["boxes"][0].set_color(color_ours)
        bp["whiskers"][0].set_color(color_ours)
        bp["whiskers"][1].set_color(color_ours)
        bp["caps"][0].set_color(color_ours)
        bp["caps"][1].set_color(color_ours)
        bp["boxes"][1].set_color(color_atacom)
        bp["whiskers"][2].set_color(color_atacom)
        bp["whiskers"][3].set_color(color_atacom)
        bp["caps"][2].set_color(color_atacom)
        bp["caps"][3].set_color(color_atacom)
        ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.set_xlim(0.2, 0.6)
#plt.subplots_adjust(hspace=0.3)
labels = ["CNP3O-PK", "ATACOM"]
plt.gcf().legend([x for x in bp["boxes"]], labels, ncol=len(labels), bbox_to_anchor=(0.6, 0.1),
                    frameon=False)
plt.tight_layout()
plt.show()