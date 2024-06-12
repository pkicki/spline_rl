import os
import numpy as np

import omnisafe
from spline_rl.envs.omnisafe_wrapper import OmnisafeWrapper


# Just fill your experiment's log directory in here.
# Such as: ~/omnisafe/examples/runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2023-03-07-20-25-48
if __name__ == '__main__':
    evaluator = omnisafe.Evaluator(render_mode='rgb_array')

    models_dir = os.path.join(os.path.dirname(__file__), 'trained_models/air_hockey/ppolag')
    for item in os.scandir(os.path.join(models_dir, "torch_save")):
        if item.is_file() and item.name.split('.')[-1] == 'pt':
            evaluator.load_saved(
                save_dir=models_dir,
                model_name=item.name,
                camera_name='track',
                width=256,
                height=256,
            )
            #evaluator.render(num_episodes=1)
            R, J, C, episode_lengths, info = evaluator.evaluate(num_episodes=100)
            success = []
            joint_pos = []
            joint_vel = []
            ee_xlb = []
            ee_ylb = []
            ee_yub = []
            ee_zlb = []
            ee_zub = []
            ee_zeb = []
            max_puck_vel = []
            cur_idx = 0
            for ep_len in episode_lengths:
                ep_len = int(ep_len)
                max_puck_vel.append(np.max([x['puck_velocity'] for x in info[cur_idx:cur_idx+ep_len]]))
                success.append(info[cur_idx+ep_len-1]["success"])
                joint_pos.append(np.mean([x['joint_pos_constraint'] for x in info[cur_idx:cur_idx+ep_len]]))
                joint_vel.append(np.mean([x['joint_vel_constraint'] for x in info[cur_idx:cur_idx+ep_len]]))
                ee_xlb.append(np.mean([x['ee_xlb_constraint'] for x in info[cur_idx:cur_idx+ep_len]]))
                ee_ylb.append(np.mean([x['ee_ylb_constraint'] for x in info[cur_idx:cur_idx+ep_len]]))
                ee_yub.append(np.mean([x['ee_yub_constraint'] for x in info[cur_idx:cur_idx+ep_len]]))
                ee_zlb.append(np.mean([x['ee_zlb_constraint'] for x in info[cur_idx:cur_idx+ep_len]]))
                ee_zub.append(np.mean([x['ee_zub_constraint'] for x in info[cur_idx:cur_idx+ep_len]]))
                ee_zeb.append(np.mean([x['ee_zeb_constraint'] for x in info[cur_idx:cur_idx+ep_len]]))
                cur_idx += ep_len

            results = dict(
                J_det=J,
                R=R,
                success=success,
                max_puck_vel=max_puck_vel,
                episode_length=episode_lengths,
                joint_pos_constraint=joint_pos,
                joint_vel_constraint=joint_vel,
                ee_xlb_constraint=ee_xlb,
                ee_ylb_constraint=ee_ylb,
                ee_yub_constraint=ee_yub,
                ee_zlb_constraint=ee_zlb,
                ee_zub_constraint=ee_zub,
                ee_zeb_constraint=ee_zeb
            )

            model_type = "ppolag"
            model_id = item.name.split('.')[0]
            save_path = os.path.join(os.path.dirname(__file__), f"../paper/results/air_hockey/{model_type}")
            os.makedirs(save_path, exist_ok=True)
            np.savez(os.path.join(save_path, f"{model_id}.npz"), **results)