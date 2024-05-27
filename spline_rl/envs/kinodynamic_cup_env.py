import os
from time import perf_counter
from mushroom_rl.environments.mujoco import MuJoCo
from mushroom_rl.utils.mujoco.observation_helper import ObservationType
import mujoco
from mushroom_rl.rl_utils.spaces import Box
import numpy as np
from enum import Enum
from air_hockey_challenge.environments.iiwas import AirHockeySingle
from air_hockey_challenge.environments.position_control_wrapper import PositionControlIIWA
from air_hockey_challenge.constraints import *
from spline_rl.utils.collisions_numpy import np_two_tables_object_collision
from spline_rl.utils.constraints import AirHockeyConstraints, KinodynamicCupConstraints

from scipy.spatial.transform import Rotation as R
from scipy import optimize as spo
import torch
import matplotlib.pyplot as plt

class AbsorbType(Enum):
    NONE = 0
    GOAL = 1
    COLLISION = 2
    SPILL = 3

class KinodynamicCupEnv(PositionControlIIWA, MuJoCo):
    def __init__(self, gamma=0.99, horizon=150, interpolation_order=5, timestep=1 / 1000., n_intermediate_steps=20, n_substeps=1,
                 viewer_params={}):

        self.n_agents = 1
        scene = os.path.join(os.path.dirname(__file__), "data", "kinodynamic_cup.xml")

        collision_spec = [("robot_body", ["iiwa_cup/link_1", "iiwa_cup/link_2", "iiwa_cup/link_3", "iiwa_cup/link_4",
                                          "iiwa_cup/link_5", "iiwa_cup/link_6", "iiwa_cup/link_7", "iiwa_cup/cup"]),
                           ("environment", ["box_1", "box_2"]),
                          ]
        action_spec = ["iiwa_cup/joint_1", "iiwa_cup/joint_2", "iiwa_cup/joint_3", "iiwa_cup/joint_4", "iiwa_cup/joint_5",
                        "iiwa_cup/joint_6", "iiwa_cup/joint_7"]
        observation_spec = [("robot/joint_1_pos", "iiwa_cup/joint_1", ObservationType.JOINT_POS),
                            ("robot/joint_2_pos", "iiwa_cup/joint_2", ObservationType.JOINT_POS),
                            ("robot/joint_3_pos", "iiwa_cup/joint_3", ObservationType.JOINT_POS),
                            ("robot/joint_4_pos", "iiwa_cup/joint_4", ObservationType.JOINT_POS),
                            ("robot/joint_5_pos", "iiwa_cup/joint_5", ObservationType.JOINT_POS),
                            ("robot/joint_6_pos", "iiwa_cup/joint_6", ObservationType.JOINT_POS),
                            ("robot/joint_7_pos", "iiwa_cup/joint_7", ObservationType.JOINT_POS),
                            ("robot/joint_1_vel", "iiwa_cup/joint_1", ObservationType.JOINT_VEL),
                            ("robot/joint_2_vel", "iiwa_cup/joint_2", ObservationType.JOINT_VEL),
                            ("robot/joint_3_vel", "iiwa_cup/joint_3", ObservationType.JOINT_VEL),
                            ("robot/joint_4_vel", "iiwa_cup/joint_4", ObservationType.JOINT_VEL),
                            ("robot/joint_5_vel", "iiwa_cup/joint_5", ObservationType.JOINT_VEL),
                            ("robot/joint_6_vel", "iiwa_cup/joint_6", ObservationType.JOINT_VEL),
                            ("robot/joint_7_vel", "iiwa_cup/joint_7", ObservationType.JOINT_VEL)]

        additional_data_spec = [("cup_pos", "iiwa_cup/cup", ObservationType.BODY_POS),
                                ("cup_rot", "iiwa_cup/cup", ObservationType.BODY_ROT),
                                ("link_base", "iiwa_cup/base", ObservationType.BODY_POS),
                                ("link_1", "iiwa_cup/link_1", ObservationType.BODY_POS),
                                ("link_2", "iiwa_cup/link_2", ObservationType.BODY_POS),
                                ("link_3", "iiwa_cup/link_3", ObservationType.BODY_POS),
                                ("link_4", "iiwa_cup/link_4", ObservationType.BODY_POS),
                                ("link_5", "iiwa_cup/link_5", ObservationType.BODY_POS),
                                ("link_6", "iiwa_cup/link_6", ObservationType.BODY_POS),
                                ("link_7", "iiwa_cup/link_7", ObservationType.BODY_POS),
                                ("link_ee", "iiwa_cup/link_ee", ObservationType.BODY_POS),
                               ]

        self.env_info = dict()
        self.env_info['box_1'] = {"size": [0.4, 0.4, 0.4], "pos": [0.5, 0.5, 0.2]}
        self.env_info['box_2'] = {"size": [0.4, 0.4, 0.4], "pos": [-0.5, 0.5, 0.2]}
        self.env_info['cup'] = {"size": [0.2, 0.2, 0.3]}
        self.env_info['robot'] = {
            "n_joints": 7,
            "joint_pos_limit": np.array([[-2.967, -2.094, -2.967, -2.094, -2.967, -2.094, -3.054],
                                         [ 2.967,  2.094,  2.967,  2.094,  2.967,  2.094,  3.054]]),
            "joint_vel_limit": np.array([[-85, -85, -100, -75, -130, -135, -135],
                                         [85, 85, 100, 75, 130, 135, 135]]) / 180. * np.pi,
            "joint_acc_limit": np.array([[-85, -85, -100, -75, -130, -135, -135],
                                         [85, 85, 100, 75, 130, 135, 135]]) / 180. * np.pi * 10,
            "joint_torque_limit": np.array([[-320, -320, -176, -176, -110, -40, -40],
                                            [320, 320, 176, 176, 110, 40, 40]]),
            "control_frequency": 50,
        }

        self.env_info['joint_pos_ids'] = [0, 1, 2, 3, 4, 5, 6]
        self.env_info['joint_vel_ids'] = [7, 8, 9, 10, 11, 12, 13]

        max_joint_vel = list(self.env_info["robot"]["joint_vel_limit"][1, :7])

        # Construct the mujoco model at origin
        robot_model = mujoco.MjModel.from_xml_path(
            os.path.join(os.path.dirname(__file__), "data", "iiwa_cup.xml"))
        robot_data = mujoco.MjData(robot_model)
        self.env_info["robot"]["robot_model"] = robot_model
        self.env_info["robot"]["robot_data"] = robot_data
        self.env_info["robot"]["ee_desired_height"] = -1. # TODO: needed just for backward compatibility

        # Ids of the joint, which are controller by the action space
        self.actuator_joint_ids = [robot_model.joint(name).id for name in action_spec]

        #self.init_range = np.array([[-0.7, -0.3], [0.3, 0.5], [0.55, 0.55]])
        #self.end_range = np.array([[0.7, 0.3], [0.3, 0.5], [0.55, 0.55]])
        #self.init_range = np.array([[-0.65, -0.35], [0.35, 0.65], [0.55, 0.55]])
        #self.end_range = np.array([[0.65, 0.35], [0.35, 0.65], [0.55, 0.55]])
        self.init_range = np.array([[-0.5, -0.5], [0.4, 0.4], [0.55, 0.55]])
        self.end_range = np.array([[0.5, 0.5], [0.4, 0.4], [0.55, 0.55]])

        super().__init__(xml_file=scene, actuation_spec=action_spec, observation_spec=observation_spec,
                         gamma=gamma, horizon=horizon, timestep=timestep, n_substeps=n_substeps,
                         n_intermediate_steps=n_intermediate_steps, additional_data_spec=additional_data_spec,
                         collision_groups=collision_spec, max_joint_vel=max_joint_vel,
                         interpolation_order=interpolation_order, **viewer_params)

        # Add env_info that requires mujoco models
        self.env_info['dt'] = self.dt
        self.env_info["robot"]["joint_pos_limit"] = np.array(
            [self._model.joint(f"iiwa_cup/joint_{i + 1}").range for i in range(7)]).T
        self.env_info['robot']['radius'] = 0.14
        self.env_info["rl_info"] = self.info

        self.constraints = KinodynamicCupConstraints(
            q_max=self.env_info['robot']['joint_pos_limit'][1],
            q_dot_max=self.env_info['robot']['joint_vel_limit'][1],
            q_ddot_max=self.env_info['robot']['joint_acc_limit'][1],
            torque_max=self.env_info['robot']['joint_torque_limit'][1],
            cup_width=self.env_info['cup']["size"][0],
            cup_height=self.env_info['cup']["size"][2],
            robot_radius=self.env_info['robot']['radius'],
            box1_xl=self.env_info['box_1']['pos'][0] - self.env_info['box_1']['size'][0] / 2,
            box1_xh=self.env_info['box_1']['pos'][0] + self.env_info['box_1']['size'][0] / 2,
            box1_yl=self.env_info['box_1']['pos'][1] - self.env_info['box_1']['size'][1] / 2,
            box1_yh=self.env_info['box_1']['pos'][1] + self.env_info['box_1']['size'][1] / 2,
            box1_height=self.env_info['box_1']['pos'][2] + self.env_info['box_1']['size'][2] / 2,
            box2_xl=self.env_info['box_2']['pos'][0] - self.env_info['box_2']['size'][0] / 2,
            box2_xh=self.env_info['box_2']['pos'][0] + self.env_info['box_2']['size'][0] / 2,
            box2_yl=self.env_info['box_2']['pos'][1] - self.env_info['box_2']['size'][1] / 2,
            box2_yh=self.env_info['box_2']['pos'][1] + self.env_info['box_2']['size'][1] / 2,
            box2_height=self.env_info['box_2']['pos'][2] + self.env_info['box_2']['size'][2] / 2,
        )
        self.env_info['rl_info'].constraints = self.constraints
        self.env_info['rl_info'].interpolation_order = interpolation_order

        
        low = np.stack([self.env_info['robot']['joint_pos_limit'][0],
                        self.env_info['robot']['joint_vel_limit'][0],
                        self.env_info['robot']['joint_acc_limit'][0]])
        high = np.stack([self.env_info['robot']['joint_pos_limit'][1],
                         self.env_info['robot']['joint_vel_limit'][1],
                         self.env_info['robot']['joint_acc_limit'][1]])
        self.env_info['rl_info'].action_space = Box(low, high)
        self.dists = []
        self.qs = []
        self.qds = []
        self.torque_s = []
        self.torques = []
        self.velds = []
        self.vels = []
        self.absorb_type = AbsorbType.NONE
        self.last_torque = np.zeros(7)


    def _modify_mdp_info(self, mdp_info):
        obs_low = np.concatenate([self.env_info['robot']['joint_pos_limit'][0],
                                  self.env_info['robot']['joint_vel_limit'][0],
                                  self.env_info['robot']['joint_pos_limit'][0],])
        obs_high = np.concatenate([self.env_info['robot']['joint_pos_limit'][1],
                                   self.env_info['robot']['joint_vel_limit'][1],
                                   self.env_info['robot']['joint_pos_limit'][1],])
        mdp_info.observation_space = Box(obs_low, obs_high)
        return mdp_info

    def _modify_observation(self, observation):
        observation = np.concatenate([observation, self.qd], axis=-1)
        return observation

    def _controller(self, desired_pos, desired_vel, desired_acc, current_pos, current_vel):
        self.last_torque = super()._controller(desired_pos, desired_vel, desired_acc, current_pos, current_vel)
        return self.last_torque
    
    def get_joints(self, obs):
        """
        Get joint position and velocity of the robot
        """
        q_pos = np.zeros(7)
        q_vel = np.zeros(7)
        for i in range(7):
            q_pos[i] = self.obs_helper.get_from_obs(obs, "robot/joint_" + str(i + 1) + "_pos")[0]
            q_vel[i] = self.obs_helper.get_from_obs(obs, "robot/joint_" + str(i + 1) + "_vel")[0]

        return q_pos, q_vel

    def cup_pose(self):
        pos = self._read_data("cup_pos")
        rot_quat = self._read_data("cup_rot")
        rot_mat = R.from_quat(rot_quat).as_matrix()
        return pos, rot_mat

    def ik_obj(self, q, desired_pos):
        for i in range(7):
            self._data.joint("iiwa_cup/joint_" + str(i + 1)).qpos = q[i]
        mujoco.mj_fwdPosition(self._model, self._data)
        pos, rot = self.cup_pose()
        diff_x = desired_pos - pos
        #diff_dir = 1.0 - rot[2, 2]
        diff_dir = 1.0 - rot[0, 0] # crazy mujoco thing that first axis seems to be z
        return np.linalg.norm(diff_x) + np.linalg.norm(diff_dir)

    def collision(self):
        def interpolate_links(xyzs):
            dists = np.linalg.norm(np.diff(xyzs, axis=-2), axis=-1)
            xyzs_ = [xyzs[..., 0, :]]
            for i, n in enumerate([1, 2, 2, 2, 1, 2, 0, 0]):
                s = np.linspace(0., 1., n + 2)[1:]
                for x in s:
                    xyzs_.append(x * xyzs[..., i + 1, :] + (1. - x) * xyzs[..., i, :])
            xyzs_interp = np.stack(xyzs_, axis=-2)
            dists_ = np.linalg.norm(np.diff(xyzs_interp, axis=-2), axis=-1)
            return xyzs_interp
        poses = np.stack([self._read_data(f"link_{x}") for x in ["base", "1", "2", "3", "4", "5", "6", "7", "ee"]])
        robot_xyzs = interpolate_links(poses)
        _, R = self.cup_pose()
        robot_xyzs = robot_xyzs[np.newaxis, np.newaxis]
        R = R[np.newaxis, np.newaxis]
        coll = np_two_tables_object_collision(robot_xyzs, R, self.dt, self.constraints.robot_radius, self.constraints.box1_xl,
                                              self.constraints.box1_xh, self.constraints.box1_yl, self.constraints.box1_yh,
                                              self.constraints.box1_height, self.constraints.box2_xl, self.constraints.box2_xh,
                                              self.constraints.box2_yl, self.constraints.box2_yh, self.constraints.box2_height,
                                              self.constraints.cup_width, self.constraints.cup_height)
        return coll

    def inverse_kinematics(self, pos, q0):
        self.bounds = spo.Bounds(*self.env_info['robot']['joint_pos_limit'])
        options = {'maxiter': 300, 'ftol': 1e-06, 'iprint': 1, 'disp': False,
                'eps': 1.4901161193847656e-08, 'finite_diff_rel_step': None}
        t0 = perf_counter()
        r = spo.minimize(lambda x: self.ik_obj(x, pos), q0, method='SLSQP',
                        bounds=self.bounds, options=options)
        t1 = perf_counter()
        #print(r)
        #print("TIME:", t1 - t0)
        return r.x
    
    
    def validate_optimization(self, q, expected_xyz):
        for i in range(7):
            self._data.joint("iiwa_cup/joint_" + str(i + 1)).qpos = q[i]
        mujoco.mj_fwdPosition(self._model, self._data)
        pos, rot = self.cup_pose()
        xyz_error = np.linalg.norm(pos - expected_xyz)
        return xyz_error < 1e-2


    def validate_torques(self, q):
        q = torch.tensor(q) if type(q) is np.ndarray else q
        q = q[None] if len(q.shape) == 1 else q
        tau = self.constraints.robot.compute_inverse_dynamics(q, torch.zeros_like(q), torch.zeros_like(q))
        error = np.abs(tau.numpy()) - self.env_info['robot']['joint_torque_limit'][1]
        valid = np.all(error < 0)
        return valid

    @property
    def constraints_num(self):
        return self.constraints.constraints_num

    def setup(self, obs):
        # Initial position of the cup
        while True:
            cup_init_pos = np.random.rand(3) * (self.init_range[:, 1] - self.init_range[:, 0]) + self.init_range[:, 0]
            cup_end_pos = np.random.rand(3) * (self.end_range[:, 1] - self.end_range[:, 0]) + self.end_range[:, 0]

            init_q = self.inverse_kinematics(cup_init_pos, np.zeros(7))
            end_q = self.inverse_kinematics(cup_end_pos, init_q)

            a = self.validate_optimization(end_q, cup_end_pos)
            b = self.validate_torques(end_q)
            if a and b:
                break

        self.qd = end_q
        self.xyzd = cup_end_pos

        #init_q = np.zeros(7)

        for i in range(7):
            self._data.joint("iiwa_cup/joint_" + str(i + 1)).qpos = init_q[i]
            self._data.joint("iiwa_cup/joint_" + str(i + 1)).qvel = 0.
        # Update body positions
        mujoco.mj_fwdPosition(self._model, self._data)
        self.absorb_type = AbsorbType.NONE
        xyz, rot = self.cup_pose()
        xyz_, rot_ = self.constraints.compute_forward_kinematics(torch.tensor(init_q)[None], torch.zeros((1, 7)))
        a = 0
        #if len(self.dists) > 0:
        #    plt.subplot(331)
        #    plt.plot(self.dists)
        #    self.qs = np.array(self.qs)
        #    self.qds = np.array(self.qds)
        #    for i in range(7):
        #        plt.subplot(332 + i)
        #        plt.plot(self.qs[:, i], label="q")
        #        plt.plot(self.qds[:, i], label="qd")
        #        plt.legend()
        #    plt.figure()
        #    self.torques = np.array(self.torques)
        #    self.torque_s = np.array(self.torque_s)
        #    for i in range(7):
        #        plt.subplot(331 + i)
        #        plt.plot(self.torques[:, i], label="applied")
        #        plt.plot(self.torque_s[:, i], label="desired")
        #        plt.legend()
        #    plt.figure()
        #    self.velds = np.array(self.velds)
        #    self.vels = np.array(self.vels)
        #    for i in range(7):
        #        plt.subplot(331 + i)
        #        plt.plot(self.vels[:, i], label="applied")
        #        plt.plot(self.velds[:, i], label="desired")
        #        plt.legend()
        #    plt.show()
        self.dists = []
        self.qs = []
        self.qds = []
        self.torque_s = []
        self.torques = []
        self.velds = []
        self.vels = []

    def reward(self, state, action, next_state, absorbing):
        r = 0
        xyz, R = self.cup_pose()
        j_pos, j_vel = self.get_joints(state)
        goal_dist = np.linalg.norm(self.xyzd - xyz)
        #r = np.exp(-10. * goal_dist**2)
        r = 1. / (10. * goal_dist + 1.)
        if goal_dist < 1e-2:
            r += 1e-2 / (np.linalg.norm(j_vel) + 1e-2)

        #torque = self._controller(action[0], action[1], action[2], j_pos, j_vel)
        torque = self.last_torque
        torque_sq_norm = np.sum(torque**2)
        r -= 1e-6 * torque_sq_norm

        #self.qs.append(state[:7])
        #self.qds.append(action[0])

        if absorbing:
            t = self._data.time
            it = int(t / self.info.dt)
            horizon = self.info.horizon
            gamma = self.info.gamma 
            factor = (1 - gamma ** (horizon - it + 1)) / (1 - gamma)
            #if self.absorb_type == AbsorbType.COLLISION:
            #    return -np.linalg.norm(j_vel) * factor
            #if self.absorb_type == AbsorbType.SPILL:
            #    return goal_dist * factor
            return r * factor
        return r

    def is_absorbing(self, obs):
        #j_pos, j_vel = self.get_joints(obs)
        #xyz, R = self.cup_pose()
        #collision = self._check_collision("robot_body", "environment")
        #if collision and np.linalg.norm(j_vel) > 1e-1:
        #    self.absorb_type = AbsorbType.COLLISION
        #    return True
        #if np.linalg.norm(R[0, 0] - 1.0) > 1e-3:
        #    self.absorb_type = AbsorbType.SPILL
        #    return True
        ##a = np.linalg.norm(self.xyzd - xyz) < 1e-2
        ##b = np.linalg.norm(R[0, 0] - 1.0) < 1e-2
        ##c = np.linalg.norm(j_vel)# < 1e-2
        ##print(a, b, c)
        #if np.linalg.norm(self.xyzd - xyz) < 1e-2 and \
        #   np.linalg.norm(j_vel) < 1e-2:
        #    self.absorb_type = AbsorbType.GOAL
        #    return True
        return False

    def _create_info_dictionary(self, state):
        j_pos, j_vel = self.get_joints(state)
        cup_pos, cup_rot = self.cup_pose()

        task_info = {}

        task_info['joint_pos_constraint'] = np.sum(np.maximum(np.abs(j_pos) - self.env_info['robot']['joint_pos_limit'][-1], 0))
        task_info['joint_vel_constraint'] = np.sum(np.maximum(np.abs(j_vel) - self.env_info['robot']['joint_vel_limit'][-1], 0))
        task_info['orientation_constraint'] = 1.0 - cup_rot[0, 0]
        task_info['collision_constriant'] = np.sum(self.collision())

        task_info["success_position"] = np.linalg.norm(self.xyzd - cup_pos) < 1e-2
        task_info["success_orientation"] = np.linalg.norm(cup_rot[0, 0] - 1.0) < 1e-3 
        task_info["success_velocity"] = np.linalg.norm(j_vel) < 1e-2
        task_info["success"] = np.all([task_info["success_position"], task_info["success_orientation"], task_info["success_velocity"]])
        #task_info["success"] = self.absorb_type == AbsorbType.GOAL
        #self.dists.append(np.linalg.norm(cup_pos - self.xyzd))
        return task_info


if __name__ == '__main__':
    env = KinodynamicCupEnv()
    env.reset()
    env.render()
    while True:
        action = np.array([0, -3, 0, 5, 0, -3, 0])
        s, r, c, done, info = env.step(action)
        env.render()
