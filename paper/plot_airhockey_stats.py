from collections import OrderedDict
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt

model_dict = dict(
    ours="CNP3O-PK",
    ours_unstructured="CNP3O",
    promp="CNP3O-ProMP",
    prodmp="CNP3O-ProDMP",
    atacom_sac="ATACOM-SAC",
    trpolag="TRPOLag",
    ppolag="PPOLag",
    pcpo="PCPO",
)

colors = OrderedDict(ours='tab:blue', ours_unstructured='tab:orange', promp='tab:green', prodmp='tab:red',
                     atacom_sac='tab:purple', trpolag='tab:brown', ppolag='tab:pink', pcpo='tab:gray')
positions = np.linspace(0.1, 0.9, len(colors))

results = {}
for model_type in colors.keys():
    results[model_type] = dict(
        J_det=[],
        R=[],
        success=[],
        max_puck_vel=[],
        episode_length=[],
        joint_pos=[],
        joint_vel=[],
        ee_xlb=[],
        ee_ylb=[],
        ee_yub=[],
        ee_zlb=[],
        ee_zub=[],
        ee_zeb=[]
    )
    for res_path in glob(os.path.join(os.path.dirname(__file__), f"results/air_hockey_fixed/{model_type}/*.npz")):
        data = np.load(res_path, allow_pickle=True)
        results[model_type]["J_det"].append(data["J_det"])
        results[model_type]["R"].append(data["R"])
        results[model_type]["success"].append(data["success"].astype(np.float32))
        results[model_type]["max_puck_vel"].append(data["max_puck_vel"])
        results[model_type]["episode_length"].append(data["episode_length"])
        results[model_type]["joint_pos"].append(data["joint_pos_constraint"])
        results[model_type]["joint_vel"].append(data["joint_vel_constraint"])
        results[model_type]["ee_xlb"].append(data["ee_xlb_constraint"])
        results[model_type]["ee_ylb"].append(data["ee_ylb_constraint"])
        results[model_type]["ee_yub"].append(data["ee_yub_constraint"])
        results[model_type]["ee_zlb"].append(data["ee_zlb_constraint"])
        results[model_type]["ee_zub"].append(data["ee_zub_constraint"])
        results[model_type]["ee_zeb"].append(data["ee_zeb_constraint"])
        
    for key in results[model_type].keys():
        #print(results[model_type][key])
        data = results[model_type][key]
        if len(data) == 0:
            continue
        if hasattr(data[0], "shape") and data[0].shape == ():
            results[model_type][key] = np.array(data)
        else:
            results[model_type][key] = np.concatenate(data)

#plot_crits = ["J_det", "R", "success", "max_puck_vel", "joint_pos", "joint_vel", "ee_xlb", "ee_ylb", "ee_yub", "ee_zlb", "ee_zub", "ee_zeb"]
titles = ["Scoring ratio [%]", "Discounted reward", "Maximal puck\n velocity [m/s]", "Maximal joint velocity\n violation [rad/s]", "Table constraint\n violation [mm]"]
plot_crits = ["success", "J_det", "max_puck_vel", "joint_vel", "ee_zeb"]
scale = dict(
    success=100.,
    J_det=1.,
    max_puck_vel=1.,
    joint_vel=1.,
    ee_zeb=1000.,
)
w = 0.10
plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(16, 4))
for i, crit in enumerate(plot_crits):
    ax = plt.subplot(1, 5, 1 + i)
    ax.set_title(titles[i])
    ax.set_xlim(0., 1.)
    for k, model_type in enumerate(colors.keys()):
        print(model_type)
        model_name = model_dict[model_type]
        c = colors[model_type]
        if crit == "success":
            ax.bar([positions[k]], [np.mean(results[model_type][crit]) * scale[crit]], color=c, label=model_name, width=w)
        else:
            bp = ax.boxplot(results[model_type][crit] * scale[crit], positions=[positions[k]], labels=[model_name], showfliers=False, widths=w)
            bp["boxes"][0].set_color(c)
            bp["whiskers"][0].set_color(c)
            bp["whiskers"][1].set_color(c)
            bp["caps"][0].set_color(c)
            bp["caps"][1].set_color(c)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    #if crit == "ee_zeb":
    #    ax.set_yscale("log")
plt.subplots_adjust(wspace=0.3)
labels = [model_dict[model_type] for model_type in colors.keys()]
plt.gcf().legend(labels, ncol=len(labels), bbox_to_anchor=(0.85, 0.1),
                 frameon=False)
#[plt.gca().get_legend().legend_handles[i].set_color(v) for i, (k, v) in enumerate(colors.items()) if k in model_dict.keys()]
#plt.tight_layout(pad=1)
plt.gcf().tight_layout(rect=[0., 0., 1., 0.95])
plt.show()