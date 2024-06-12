from collections import OrderedDict
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt

model_dict = OrderedDict(
    ours_structured="CNP3O-PK",
    promp_structured="CNP3O-PK-ProMP",
    prodmp_structured="CNP3O-PK-ProDMP",
    ours_unstructured="CNP3O",
    promp_unstructured="CNP3O-ProMP",
    prodmp_unstructured="CNP3O-ProDMP",
    #ppolag="PPOLag-PK",
)

colors = dict(ours_structured='tab:blue', promp_structured='tab:green', prodmp_structured='tab:red',
              ours_unstructured='tab:orange', promp_unstructured='tab:olive', prodmp_unstructured='tab:cyan',
              ppolag='tab:brown')
positions = np.linspace(0.1, 0.9, len(colors))

results = {}
for model_type in colors.keys():
    results[model_type] = dict(
        J_det=[],
        R=[],
        success=[],
        joint_pos=[],
        joint_vel=[],
        orientation=[],
        collision=[],
    )
    for res_path in glob(os.path.join(os.path.dirname(__file__), f"results/kino/{model_type}/*.npz")):
        data = np.load(res_path, allow_pickle=True)
        results[model_type]["J_det"].append(data["J_det"])
        results[model_type]["R"].append(data["R"])
        results[model_type]["joint_pos"].append(data["joint_pos_constraint"])
        results[model_type]["joint_vel"].append(data["joint_vel_constraint"])
        results[model_type]["orientation"].append(data["orientation_constraint"])
        results[model_type]["collision"].append(data["collision_constraint"])
        
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
#plot_crits = ["success", "J_det", "max_puck_vel", "joint_vel", "ee_zeb"]
titles = ["Discounted reward", "Vertical orientation\n violation", "Maximal joint velocity\n violation [rad/s]", "Collision"]
plot_crits = ["J_det", "orientation", "joint_vel", "collision"]
#plot_crits = ["J_det"]
w = 0.10
plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(16, 3.5))
plots = []
for i, crit in enumerate(plot_crits):
    ax = plt.subplot(1, len(plot_crits), 1 + i)
    ax.set_title(titles[i])
    ax.set_xlim(0., 1.)
    for k, model_type in enumerate(model_dict.keys()):
        model_name = model_dict[model_type]
        c = colors[model_type]
        if crit == "success":
            plots.append(ax.bar([positions[k]], [np.mean(results[model_type][crit])], color=c, label=model_name, width=w))
        else:
            bp = ax.boxplot(results[model_type][crit], positions=[positions[k]], labels=[model_name], showfliers=False, widths=w)
            plots.append(bp)
            bp["boxes"][0].set_color(c)
            bp["whiskers"][0].set_color(c)
            bp["whiskers"][1].set_color(c)
            bp["caps"][0].set_color(c)
            bp["caps"][1].set_color(c)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
plt.subplots_adjust(hspace=0.3)
labels = [model_dict[model_type] for model_type in model_dict.keys()]
#plt.gcf().legend(labels)
#plt.gcf().legend([x["boxes"][0] for x in plots[-len(model_dict.keys()):]], labels, ncol=len(labels), bbox_to_anchor=(0.85, 0.1),
#                 frameon=False)
plt.gcf().tight_layout()
plt.show()