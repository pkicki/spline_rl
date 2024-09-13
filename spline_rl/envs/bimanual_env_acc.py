import numpy as np
from dm_control import mujoco
from mushroom_rl.rl_utils.spaces import Box

from spline_rl.envs.bimanual_env import BimanualEnv


class AbsorbType:
    NONE = -1
    SUCCESS = 0
    DROP = 1

class BimanualAccEnv(BimanualEnv):
    def __init__(self, gamma=0.99, horizon=100, interpolation_order=-1,
                 success_scale=2., return_cost=True, **kwargs):
        self.return_cost = return_cost
        super(BimanualAccEnv, self).__init__(gamma=gamma, horizon=horizon,
                                             interpolation_order=interpolation_order,
                                             success_scale=success_scale, **kwargs)

    def _compute_action(self, obs, action):
        q, dq = self.get_current_robot_state()
        acc_high = np.minimum(self.env_info['robot']['joint_acc_limit'][1],
                              5 * (self.env_info['robot']['joint_vel_limit'][1] - dq))
        acc_low = np.maximum(self.env_info['robot']['joint_acc_limit'][0],
                             5 * (self.env_info['robot']['joint_vel_limit'][0] - dq))
        acc = np.clip(action, acc_low, acc_high)
        self.env_info['data'].qpos[self.robot_joint_ids] = q
        self.env_info['data'].qvel[self.robot_joint_ids] = dq
        self.env_info['data'].qacc[self.robot_joint_ids] = acc
        torque = np.zeros(self._model.nv)
        mujoco.mj_fwdPosition(self.env_info['model'], self.env_info['data'])
        mujoco.mj_rne(self.env_info['model'], self.env_info['data'], 1, torque)
        #qfrc = self._data.qfrc_bias[self.robot_joint_ids]
        torque = torque[self.robot_joint_ids]
        #mujoco.mj_rne(self._model, self._data, 1, torque)
        self._data.ctrl[self.left_gripper_id] = self.gripper_force
        self._data.ctrl[self.right_gripper_id] = self.gripper_force
        return torque

    def _modify_mdp_info(self, mdp_info):
        super(BimanualAccEnv, self)._modify_mdp_info(mdp_info)
        mdp_info.action_space = Box(low=-np.ones(13), high=np.ones(13))
        return mdp_info

    def step(self, action):
        obs, reward, done, info = super(BimanualAccEnv, self).step(action)

        constraint_violations = np.max(info["cost"])

        if self.return_cost:
            return obs, reward, constraint_violations, done, info
        return obs, reward, done, info


if __name__ == "__main__":
    env = BimanualAccEnv(interpolation_order=5)
    env.reset()
    robot_pos = env.get_current_robot_state()[0]
    for i in range(10000):
        #robot_pos[0] += 0.001
        #robot_pos[1] += 0.0001
        action = np.zeros_like(robot_pos)
        state, reward, absorbing, info = env.step(action)
        env.render()

        #plate_pos = np.array([0.01, 0.6, 0.04])
        #plate_angle = np.arctan2(plate_pos[1], plate_pos[0])
        #plate_rot = np.random.uniform(low=[0, 0, 0], high=[0., 0., 0.]) + np.array([np.pi - plate_angle, 0., 0.])
        #plate_quat = euler2quat(plate_rot)
        #success = env.move_plate_to_pose(plate_pos, plate_quat)
        #robot_pos = env.get_current_robot_state()[0]

        #print(env._data.site("EE_ur5left").xmat.reshape(3, 3))
        #ee_left = env._data.site("EE_ur5left").xpos
        #ee_right = env._data.site("EE_ur5right").xpos
        #print(np.linalg.norm(ee_left - ee_right))
        #env.reward(None, None, None, None)
    print("Done.")