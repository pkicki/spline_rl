import os
import numpy as np

import omnisafe
from spline_rl.envs.omnisafe_wrapper import OmnisafeWrapper


# Just fill your experiment's log directory in here.
# Such as: ~/omnisafe/examples/runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2023-03-07-20-25-48
if __name__ == '__main__':
    evaluator = omnisafe.Evaluator(render_mode='rgb_array')

    models_dir = os.path.join(os.path.dirname(__file__), 'trained_models/kinodynamic/ppolag')
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
            R, J, C, episode_lengths, info = evaluator.evaluate(num_episodes=10)
            success = []
            joint_pos = []
            joint_vel = []
            orientation = []
            collision = []
            cur_idx = 0
            for ep_len in episode_lengths:
                ep_len = int(ep_len)
                success.append(info[cur_idx+ep_len-1]["success"])
                joint_pos.append(np.mean([x['joint_pos_constraint'] for x in info[cur_idx:cur_idx+ep_len]]))
                joint_vel.append(np.mean([x['joint_vel_constraint'] for x in info[cur_idx:cur_idx+ep_len]]))
                orientation.append(np.mean([x['orientation_constraint'] for x in info[cur_idx:cur_idx+ep_len]]))
                collision.append(np.mean([x['collision_constriant'] for x in info[cur_idx:cur_idx+ep_len]]))

                cur_idx += ep_len

            results = dict(
                    J_det=J,
                    R=R,
                    success=success,
                    joint_pos_constraint=joint_pos,
                    joint_vel_constraint=joint_vel,
                    orientation_constraint=orientation,
                    collision_constraint=collision,
                )

            model_type = "ppolag"
            model_id = item.name.split('.')[0]
            save_path = os.path.join(os.path.dirname(__file__), f"../paper/results/kino/{model_type}")
            os.makedirs(save_path, exist_ok=True)
            np.savez(os.path.join(save_path, f"{model_id}.npz"), **results)