import mujoco
from mushroom_rl.rl_utils.spaces import Box
import numpy as np
from enum import Enum
from air_hockey_challenge.environments.iiwas import AirHockeySingle
from air_hockey_challenge.environments.position_control_wrapper import PositionControlIIWA
from air_hockey_challenge.constraints import *
from spline_rl.utils.constraints import AirHockeyConstraints


class AbsorbType(Enum):
    NONE = 0
    GOAL = 1
    UP = 2
    RIGHT = 3
    LEFT = 4
    BOTTOM = 5

class AirHockeyEnv(PositionControlIIWA, AirHockeySingle):
    def __init__(self, gamma=0.99, horizon=150, moving_init=True, interpolation_order=-1, reward_type="new", return_cost=False):
        super().__init__(gamma=gamma, horizon=horizon, interpolation_order=interpolation_order)

        assert reward_type in ["new", "mixed", "puze"]
        if reward_type == "new":
            self.reward = self.new_reward
        elif reward_type == "mixed":
            self.reward = self.mixed_reward
        elif reward_type == "puze":
            self.reward = self.puze_reward
        else:
            raise ValueError("Unknown reward type")

        if return_cost:
            self.step = self._step_with_cost

        self.ee_z_eb = 0.16#self.env_info['robot']['ee_desired_height']
        self.ee_x_lb = - self.env_info['robot']['base_frame'][0][0, 3] \
                       - self.env_info['table']['length'] / 2 + self.env_info['mallet']['radius']
        self.ee_y_lb = - self.env_info['table']['width'] / 2 + self.env_info['mallet']['radius']
        self.ee_y_ub = self.env_info['table']['width'] / 2 - self.env_info['mallet']['radius']
        self.constraints = AirHockeyConstraints(
            self.env_info['robot']['joint_vel_limit'][1],
            self.env_info['robot']['joint_acc_limit'][1],
            self.ee_x_lb,
            self.ee_y_lb,
            self.ee_y_ub,
            self.ee_z_eb
        )
        self.env_info['rl_info'].constraints = self.constraints

        self.hit_range = np.array([[-0.7, -0.2], [-0.35, 0.35]], dtype=np.float32)
        self.moving_init = moving_init
        self.init_velocity_range = (0, 0.3)
        self.init_state = np.array([-7.16000830e-06, 6.97494070e-01, 7.26955352e-06, -5.04898567e-01, 6.60813111e-07, 1.92857916e+00, 0.], dtype=np.float32)
        self.ee_puck_dist = np.inf
        self.hit_time = -1


        low = np.stack([self.env_info['robot']['joint_pos_limit'][0],
                        self.env_info['robot']['joint_vel_limit'][0],
                        self.env_info['robot']['joint_acc_limit'][0]], dtype=np.float32)
        high = np.stack([self.env_info['robot']['joint_pos_limit'][1],
                         self.env_info['robot']['joint_vel_limit'][1],
                         self.env_info['robot']['joint_acc_limit'][1]], dtype=np.float32)
        if interpolation_order in [1, 2]:
            low = low[:1]
            high = high[:1]
        elif interpolation_order in [-1, 3, 4]:
            low = low[:2]
            high = high[:2]
        self.env_info['rl_info'].action_space = Box(low, high)
        self.env_info['rl_info'].interpolation_order = interpolation_order



    @property
    def constraints_num(self):
        return self.constraints.constraints_num

    def setup(self, obs):
        # Initial position of the puck
        puck_pos = np.random.rand(2) * (self.hit_range[:, 1] - self.hit_range[:, 0]) + self.hit_range[:, 0]
        puck_yaw = np.random.uniform(-np.pi, np.pi)
        puck_vel = np.zeros(3)

        if self.moving_init:
            init_vel_mag = np.random.uniform(self.init_velocity_range[0], self.init_velocity_range[1])
            init_vel_angle = np.random.uniform(-np.pi, np.pi / 2)
            init_vel = np.array([np.cos(init_vel_angle), np.sin(init_vel_angle)]) * init_vel_mag

            t_reach_x = (np.array([self.hit_range[0, 0], self.hit_range[0, 1]]) - puck_pos[0]) / init_vel[0]
            t_reach_y = (np.array([self.hit_range[1, 0], self.hit_range[1, 1]]) - puck_pos[1]) / init_vel[1]
            t_reach = np.concatenate([t_reach_x, t_reach_y])
            t_reach = t_reach[t_reach > 0]
            if np.any(t_reach < 1.5):
                init_vel *= np.min(t_reach) / 1.5

            puck_vel[0] = init_vel[0]
            puck_vel[1] = init_vel[1]
            puck_vel[2] = np.random.uniform(-5, 5)

        self._write_data("puck_x_pos", puck_pos[0])
        self._write_data("puck_y_pos", puck_pos[1])
        self._write_data("puck_yaw_pos", puck_yaw)
        self._write_data("puck_x_vel", puck_vel[0])
        self._write_data("puck_y_vel", puck_vel[1])
        self._write_data("puck_yaw_vel", puck_vel[2])
        self.absorb_type = AbsorbType.NONE
        self.ee_puck_dist = np.inf
        self.hit_time = -1

        for i in range(7):
            self._data.joint("iiwa_1/joint_" + str(i + 1)).qpos = self.init_state[i]

            self.q_pos_prev[i] = self.init_state[i]
            self.q_vel_prev[i] = self._data.joint("iiwa_1/joint_" + str(i + 1)).qvel

        self.universal_joint_plugin.reset()

        super(AirHockeySingle, self).setup(obs)
        # Update body positions, needed for _compute_universal_joint
        mujoco.mj_fwdPosition(self._model, self._data)

    #def _controller(self, desired_pos, desired_vel, desired_acc, current_pos, current_vel):
    #    torque = super()._controller(desired_pos, desired_vel, desired_acc, current_pos, current_vel)
    #    low = self.robot_model.actuator_ctrlrange[:, 0]
    #    high = self.robot_model.actuator_ctrlrange[:, 1]
    #    torque = np.clip(torque, low, high)
    #    return torque

    def mixed_reward(self, state, action, next_state, absorbing):
        r = 0
        # Get puck's position and velocity (The position is in the world frame, i.e., center of the table)
        puck_pos, puck_vel = self.get_puck(next_state)
        self.puck_velocity = np.linalg.norm(puck_vel[:2])

        # Define goal position
        goal = np.array([0.98, 0])

        goal_dist = np.linalg.norm(goal - puck_pos[:2])
        if self.hit_time > 0:
            r = np.exp(-2. * goal_dist**2)

        if absorbing:
            t = self._data.time
            it = int(t / self.info.dt)
            horizon = self.info.horizon
            gamma = self.info.gamma 
            factor = (1 - gamma ** (horizon - it + 1)) / (1 - gamma)
            return r * factor

        ee_pos = self.get_ee()[0][:2]
        ee_puck_dist = np.linalg.norm(ee_pos - puck_pos[:2])
        if self.ee_puck_dist == np.inf:
            self.ee_puck_dist = ee_puck_dist
        elif ee_puck_dist < self.ee_puck_dist:
            r += (self.ee_puck_dist - ee_puck_dist) * 1#0
            self.ee_puck_dist = ee_puck_dist
        return r

    def new_reward(self, state, action, next_state, absorbing):
        r = 0
        # Get puck's position and velocity (The position is in the world frame, i.e., center of the table)
        puck_pos, puck_vel = self.get_puck(next_state)
        self.puck_velocity = np.linalg.norm(puck_vel[:2])

        # Define goal position
        goal = np.array([0.98, 0])

        goal_dist = np.linalg.norm(goal - puck_pos[:2])
        r = np.exp(-2. * goal_dist**2)

        if absorbing:
            t = self._data.time
            it = int(t / self.info.dt)
            horizon = self.info.horizon
            gamma = self.info.gamma 
            factor = (1 - gamma ** (horizon - it + 1)) / (1 - gamma)
            return r * factor
        return r

    def puze_reward(self, state, action, next_state, absorbing):
        puck_pos = next_state[:2].copy()
        puck_pos[0] += 1.51
        puck_vel = next_state[3:5]

        if absorbing:
            r = 0
            factor = (1 - self.info.gamma ** self.info.horizon) / (1 - self.info.gamma)
            if self.absorb_type == AbsorbType.GOAL:
                r = 1.5 - (np.clip(abs(puck_pos[1]), 0, 0.1) * 5)
            elif self.absorb_type == AbsorbType.UP:
                r = (1 - np.clip(abs(puck_pos[1]) - 0.1, 0, 0.35) * 2)
            elif self.absorb_type == AbsorbType.LEFT:
                r = (0.3 - np.clip(2.43 - puck_pos[0], 0, 1) * 0.3)
            elif self.absorb_type == AbsorbType.RIGHT:
                r = (0.3 - np.clip(2.43 - puck_pos[0], 0, 1) * 0.3)
            r *= factor
        else:
            if puck_pos[0] > 1.51:
                r = 1.5 * np.clip(puck_vel[0], 0, 3)
            else:
                r = 0

        ee_pos = self.get_ee()[0][:2] + np.array([1.51, 0.])
        ee_puck_dist = np.linalg.norm(ee_pos - puck_pos)
        if self.ee_puck_dist == np.inf:
            self.ee_puck_dist = ee_puck_dist
        elif ee_puck_dist < self.ee_puck_dist:
            r += (self.ee_puck_dist - ee_puck_dist) * 10
            self.ee_puck_dist = ee_puck_dist

        return r
    
    def is_absorbing(self, obs):
        puck_pos = obs[:2].copy()
        puck_pos[0] += 1.51
        puck_vel = obs[3:5].copy()

        if puck_pos[0] < 0.58 or (puck_pos[0] < 0.63 and puck_vel[0] > 0.):
            self.absorb_type = AbsorbType.BOTTOM
            return True

        if puck_pos[0] > 2.46 or (puck_pos[0] > 2.39 and puck_vel[0] < 0):
            self.absorb_type = AbsorbType.UP
            if abs(puck_pos[1]) < 0.1:
                self.absorb_type = AbsorbType.GOAL
            return True

        #if puck_vel[0] > 0. and puck_pos[0] > 1.51:
        if (puck_pos[1] > 0.45 and puck_vel[1] > 0.) or (puck_pos[1] > 0.42 and puck_vel[1] < 0.):
            self.absorb_type = AbsorbType.LEFT
            return True
        if (puck_pos[1] < -0.45 and puck_vel[0] < 0.) or (puck_pos[1] < -0.42 and puck_vel[1] > 0.):
            self.absorb_type = AbsorbType.RIGHT
            return True
        return False

    def _create_info_dictionary(self, state):
        puck_pos, puck_vel = self.get_puck(state)
        ee_pos, ee_vel = self.get_ee()
        j_pos, j_vel = self.get_joints(state)

        task_info = {}

        task_info['joint_pos_constraint'] = np.sum(np.maximum(np.abs(j_pos) - self.env_info['robot']['joint_pos_limit'][-1], 0))
        task_info['joint_vel_constraint'] = np.sum(np.maximum(np.abs(j_vel) - self.env_info['robot']['joint_vel_limit'][-1], 0))
        task_info['ee_xlb_constraint'] = np.maximum(-self.env_info['table']['length'] / 2 + self.env_info['mallet']['radius'] - ee_pos[0], 0)
        task_info['ee_ylb_constraint'] = np.maximum(-self.env_info['table']['width'] / 2 + self.env_info['mallet']['radius'] - ee_pos[1], 0)
        task_info['ee_yub_constraint'] = np.maximum(ee_pos[1] - self.env_info['table']['width'] / 2 + self.env_info['mallet']['radius'], 0)
        desired_z = 0.06
        task_info['ee_zeb_constraint'] = np.abs(ee_pos[2] - desired_z)
        task_info['ee_zlb_constraint'] = np.maximum(desired_z - 0.02 - ee_pos[2], 0)
        task_info['ee_zub_constraint'] = np.maximum(ee_pos[2] - 0.02 - desired_z, 0)

        task_info["success"] = puck_pos[0] - (self.env_info['table']['length'] / 2 - self.env_info['puck']['radius']) > 0 and \
                               np.abs(puck_pos[1]) - self.env_info['table']['goal_width'] / 2 < 0

        puck_mallet_dist = self.env_info['puck']['radius'] + self.env_info['mallet']['radius'] + 5e-3
        if self.hit_time < 0 and np.linalg.norm(puck_pos[:2] - ee_pos[:2]) < puck_mallet_dist and np.abs(ee_pos[2] - 0.065) < 0.02:
            self.hit_time = self._data.time
        task_info["hit_time"] = self.hit_time
        task_info["puck_velocity"] = np.linalg.norm(puck_vel[:2])

        return task_info

    def _step_with_cost(self, action):
        obs, reward, absorbing, info = super().step(action)
        cost = 0.
        cost = np.max(np.stack([info['joint_pos_constraint'], info['joint_vel_constraint'], info['ee_xlb_constraint'], info['ee_ylb_constraint'],
                                info['ee_yub_constraint'], info['ee_zlb_constraint'], info['ee_zub_constraint']]))
        return obs, reward, cost, absorbing, info


if __name__ == '__main__':
    env = AirHockeyEnv()
    env.reset()
    env.render()
    while True:
        action = np.array([0, -3, 0, 5, 0, -3, 0])
        s, r, c, done, info = env.step(action)
        env.render()
