import mujoco
from mushroom_rl.rl_utils.spaces import Box
import numpy as np
from enum import Enum
from air_hockey_challenge.environments.iiwas import AirHockeySingle
from air_hockey_challenge.environments.position_control_wrapper import PositionControlIIWA
from air_hockey_challenge.constraints import *
from utils.constraints import AirHockeyConstraints


class AbsorbType(Enum):
    NONE = 0
    GOAL = 1
    UP = 2
    RIGHT = 3
    LEFT = 4
    BOTTOM = 5


#class AccelerationControl:
#    # TODO: use PID + FF control or something already used by Puze
#    def _compute_action(self, obs, action):
#        q, dq = self.get_joints(obs)
#        acc_high = np.minimum(self.env_info['robot']['joint_acc_limit'][1],
#                              5 * (self.env_info['robot']['joint_vel_limit'][1] - dq))
#        acc_low = np.maximum(self.env_info['robot']['joint_acc_limit'][0],
#                             5 * (self.env_info['robot']['joint_vel_limit'][0] - dq))
#        acc = np.clip(action, acc_low, acc_high)
#        self.env_info['robot']['robot_data'].qpos[:] = q
#        self.env_info['robot']['robot_data'].qvel[:] = dq
#        self.env_info['robot']['robot_data'].qacc[:] = acc
#        torque = np.zeros(7)
#        mujoco.mj_rne(self.env_info['robot']['robot_model'], self.env_info['robot']['robot_data'], 1, torque)
#        return torque


class AirHockeyEnv(PositionControlIIWA, AirHockeySingle):
    def __init__(self, gamma, horizon, moving_init):
        super().__init__(gamma=gamma, horizon=horizon, interpolation_order=5)

        self.ee_z_eb = self.env_info['robot']['ee_desired_height']
        self.ee_x_lb = - self.env_info['robot']['base_frame'][0][0, 3] \
                       - self.env_info['table']['length'] / 2 + self.env_info['mallet']['radius']
        self.ee_y_lb = - self.env_info['table']['width'] / 2 + self.env_info['mallet']['radius']
        self.ee_y_ub = self.env_info['table']['width'] / 2 - self.env_info['mallet']['radius']
        self.constraints = AirHockeyConstraints(self.env_info['robot']['joint_vel_limit'][1],
                                                    self.env_info['robot']['joint_acc_limit'][1],
                                                    self.ee_x_lb, self.ee_y_lb, self.ee_y_ub, self.ee_z_eb)
        self.env_info['rl_info'].constraints = self.constraints

        self.hit_range = np.array([[-0.7, -0.2], [-0.35, 0.35]])
        self.moving_init = moving_init
        self.init_velocity_range = (0, 0.3)
        self.init_state = np.array([-7.16000830e-06, 6.97494070e-01, 7.26955352e-06, -5.04898567e-01, 6.60813111e-07, 1.92857916e+00, 0.])


        low = np.stack([self.env_info['robot']['joint_pos_limit'][0],
                        self.env_info['robot']['joint_vel_limit'][0],
                        self.env_info['robot']['joint_acc_limit'][0]])
        high = np.stack([self.env_info['robot']['joint_pos_limit'][1],
                         self.env_info['robot']['joint_vel_limit'][1],
                         self.env_info['robot']['joint_acc_limit'][1]])
        self.env_info['rl_info'].action_space = Box(low, high)



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

        for i in range(7):
            self._data.joint("iiwa_1/joint_" + str(i + 1)).qpos = self.init_state[i]

            self.q_pos_prev[i] = self.init_state[i]
            self.q_vel_prev[i] = self._data.joint("iiwa_1/joint_" + str(i + 1)).qvel

        self.universal_joint_plugin.reset()

        super(AirHockeySingle, self).setup(obs)
        # Update body positions, needed for _compute_universal_joint
        mujoco.mj_fwdPosition(self._model, self._data)


    def reward(self, state, action, next_state, absorbing):
        r = 0
        # Get puck's position and velocity (The position is in the world frame, i.e., center of the table)
        puck_pos, puck_vel = self.get_puck(next_state)
        self.puck_velocity = np.linalg.norm(puck_vel[:2])

        # Define goal position
        goal = np.array([0.98, 0])

        goal_dist = np.linalg.norm(goal - puck_pos[:2])
        r = np.exp(-2. * goal_dist**2)

        factor = 1.
        if absorbing:
            t = self._data.time
            it = int(t / self.info.dt)
            horizon = self.info.horizon
            gamma = self.info.gamma 
            factor = (1 - gamma ** (horizon - it + 1)) / (1 - gamma)
        return r * factor

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
        success = False
        if self.absorb_type == AbsorbType.GOAL:
            success = True

        puck_pos, puck_vel = self.get_puck(state)
        ee_pos, ee_vel = self.get_ee()

        hit_time = -1.
        puck_mallet_dist = self.env_info['puck']['radius'] + self.env_info['mallet']['radius'] + 5e-3
        if np.linalg.norm(puck_pos[:2] - ee_pos[:2]) < puck_mallet_dist and np.abs(ee_pos[2] - 0.065) < 0.02:
            hit_time = self._data.time

        return {'success': success, "hit_time": hit_time, "puck_velocity": np.linalg.norm(puck_vel[:2])}


if __name__ == '__main__':
    env = AirHockeyEnv()
    env.reset()
    env.render()
    while True:
        action = np.array([0, -3, 0, 5, 0, -3, 0])
        s, r, c, done, info = env.step(action)
        env.render()
