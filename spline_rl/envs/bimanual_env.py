import os
import pickle
from time import sleep
import numpy as np
from dm_control import mujoco
from dm_control.utils import inverse_kinematics as ik
import torch

from mushroom_rl.rl_utils.spaces import Box
from mushroom_rl.environments.mujoco import MuJoCo
from mushroom_rl.environments.mujoco import ObservationType

import numpy as np

from spline_rl.utils.constants import BIMANUAL_EE_DIST
from spline_rl.utils.constraints import BimanualConstraints
from spline_rl.utils.geometry import error_mat, euler2mat, euler2quat, mat2euler, mat2quat, mjquat2quat, quat2euler, quat2mat, quat_mul, rotation_distance


class AbsorbType:
    NONE = -1
    SUCCESS = 0
    DROP = 1

class BimanualEnv(MuJoCo):
    def __init__(self, gamma=0.99, horizon=100, interpolation_order=-1, success_scale=2., **kwargs):
        self.xml_file =  os.path.join(os.path.dirname(__file__), "data", "quad_insert.xml")
        observation_spec = []
        observation_spec += [("left_EE_pos", "EE_ur5left", ObservationType.SITE_POS),]
        observation_spec += [("right_EE_pos", "EE_ur5right", ObservationType.SITE_POS),]
        observation_spec += [("plate_pos", "grommet_11mm", ObservationType.BODY_POS),]
        observation_spec += [("plate_rot", "grommet_11mm", ObservationType.BODY_ROT),]
        observation_spec += [("plate_vel", "grommet_11mm", ObservationType.BODY_VEL),]
        observation_spec += [("pegs_pos", "quad_peg", ObservationType.BODY_POS),]
        observation_spec += [("pegs_rot", "quad_peg", ObservationType.BODY_ROT),]
        observation_spec += [("stand_joint_pos", "ur_stand_joint", ObservationType.JOINT_POS),]
        observation_spec += [(f"left_joint{i}_pos", f"joint{i}_ur5left", ObservationType.JOINT_POS) for i in range(6)]
        observation_spec += [(f"right_joint{i}_pos", f"joint{i}_ur5right", ObservationType.JOINT_POS) for i in range(6)]
        observation_spec += [("stand_joint_vel", "ur_stand_joint", ObservationType.JOINT_VEL),]
        observation_spec += [(f"left_joint{i}_vel", f"joint{i}_ur5left", ObservationType.JOINT_VEL) for i in range(6)]
        observation_spec += [(f"right_joint{i}_vel", f"joint{i}_ur5right", ObservationType.JOINT_VEL) for i in range(6)]
        actuation_spec = ["ur_stand_joint_motor"]
        actuation_spec += [f"joint{i}_motor_ur5left" for i in range(6)]
        actuation_spec += [f"joint{i}_motor_ur5right" for i in range(6)]
        additional_data_spec = []
        additional_data_spec += [
            ("plate_pos", "grommet_11mm", ObservationType.BODY_POS),
            ("plate_rot", "grommet_11mm", ObservationType.BODY_ROT),
        ]
        additional_data_spec += [(f"left_joint{i}", f"joint{i}_motor_ur5left", ObservationType.JOINT_POS) for i in range(6)]
        additional_data_spec += [(f"right_joint{i}", f"joint{i}_motor_ur5right", ObservationType.JOINT_POS) for i in range(6)]

        camera_params = dict(camera_params=dict(
                static=dict(distance=2.0, elevation=-20.0, azimuth=-90.0, lookat=np.array([0.0, 0.0, 0.0])),
                follow=dict(distance=3.5, elevation=0.0, azimuth=90.0),
                top_static=dict(distance=5.0, elevation=-90.0, azimuth=90.0, lookat=np.array([0.0, 0.0, 0.0]))
            )
        )

        self.env_info = {}
        self.interpolation_order = interpolation_order
        super().__init__(xml_file=self.xml_file, actuation_spec=actuation_spec,
                         observation_spec=observation_spec, additional_data_spec=additional_data_spec,
                         gamma=gamma, horizon=horizon, **camera_params)

        self.constraints = BimanualConstraints(
            self.env_info['robot']['joint_vel_limit'][1],
            self.env_info['robot']['joint_acc_limit'][1],
            BIMANUAL_EE_DIST,
        )
        self.env_info['rl_info'] = self.info
        self.env_info['rl_info'].constraints = self.constraints
        self.env_info['dt'] = self.info.dt
        self.env_info['episode_duration'] = self.info.horizon * self.info.dt
        self.env_info['rl_info'].interpolation_order = self.interpolation_order

        self.dm_physics = mujoco.Physics.from_xml_path(self.xml_file)
        p_gain = [1500., 1500., 1200., 1200., 1000., 1000.]
        d_gain = [80, 80, 30, 30, 10, 1]
        i_gain = [0, 0, 0, 0, 0, 0]
        self.p_gain = [2000.] + p_gain + p_gain
        self.d_gain = [100.] + d_gain + d_gain
        self.i_gain = [0.] + i_gain + i_gain
        self.i_error = np.zeros_like(self.i_gain)
        self.gripper_force = 0.2 #0.2

        self.actuator_joint_ids = [self._data.actuator(joint_name).id for joint_name in actuation_spec]
        self.robot_joint_ids = [self._model.actuator(joint_name).trnid[0] for joint_name in actuation_spec]
        self.left_gripper_id = self._model.actuator("gripper_ur5left").id
        self.right_gripper_id = self._model.actuator("gripper_ur5right").id

        self.target_pos = None
        self.target_mjquat = None
        self.target_quat = None
        self.absorbing_type = AbsorbType.NONE

        self.desired_gripper_joints = {
            'left_inner_finger_joint_ur5left': -0.73191728,
            'right_inner_finger_joint_ur5left': -0.78984115,
            'left_inner_knuckle_joint_ur5left': 0.7341096,
            'right_inner_knuckle_joint_ur5left': 0.7538488,
            'left_outer_knuckle_joint_ur5left': 0.73120889,
            'right_outer_knuckle_joint_ur5left': 0.75101009,
            'left_inner_finger_joint_ur5right': -0.74807509,
            'right_inner_finger_joint_ur5right': -0.7812078,
            'left_inner_knuckle_joint_ur5right': 0.73657761,
            'right_inner_knuckle_joint_ur5right': 0.75597062,
            'left_outer_knuckle_joint_ur5right': 0.7337423,
            'right_outer_knuckle_joint_ur5right': 0.75317061,
        }

        ## create normalization file
        #observations = np.array([self.setup(None) for _ in range(1000)])
        #self.observation_stats = dict(mean=observations.mean(axis=0),
        #                              std=observations.std(axis=0))
        ##with open(os.path.join(os.path.dirname(__file__), 'data', 'easy_xyzpm10.pickle'), 'wb') as fh:
        #with open(os.path.join(os.path.dirname(__file__), 'data', 'middle.pickle'), 'wb') as fh:
        #    pickle.dump(self.observation_stats, fh, protocol=pickle.HIGHEST_PROTOCOL)

        # read normalization file
        #with open(os.path.join(os.path.dirname(__file__), 'data', 'easy_xyzpm10.pickle'), 'rb') as fh:
        with open(os.path.join(os.path.dirname(__file__), 'data', 'middle.pickle'), 'rb') as fh:
        #with open(os.path.join(os.path.dirname(__file__), 'data', 'hard.pickle'), 'rb') as fh:
            self.observation_stats = pickle.load(fh)

        self.env_info['observation_stats'] = self.observation_stats
        #self.env_info['observation_stats'] = None
        self.success_scale = success_scale


    def _modify_mdp_info(self, mdp_info):
        self.joint_pos_limits = np.concatenate([
            np.array([self._model.joint("ur_stand_joint").range]).T,
            np.array([self._model.joint(f"joint{i}_ur5left").range for i in range(6)]).T,
            np.array([self._model.joint(f"joint{i}_ur5right").range for i in range(6)]).T,
        ], axis=-1)
        self.max_joint_vel = np.pi * np.ones(self.joint_pos_limits.shape[1])
        self.max_joint_acc = 14. * np.ones(self.joint_pos_limits.shape[1])
        self.env_info['robot'] = {
            "joint_pos_limit": self.joint_pos_limits,
            "joint_vel_limit": np.stack([-self.max_joint_vel, self.max_joint_vel], axis=0),
            "joint_acc_limit": np.stack([-self.max_joint_acc, self.max_joint_acc], axis=0),
        }
        low = np.stack([self.env_info['robot']['joint_pos_limit'][0],
                        self.env_info['robot']['joint_vel_limit'][0],
                        self.env_info['robot']['joint_acc_limit'][0]]).astype(np.float32)
        high = np.stack([self.env_info['robot']['joint_pos_limit'][1],
                         self.env_info['robot']['joint_vel_limit'][1],
                         self.env_info['robot']['joint_acc_limit'][1]]).astype(np.float32)
        if self.interpolation_order in [1, 2]:
            low = low[:1]
            high = high[:1]
        elif self.interpolation_order in [-1, 3, 4]:
            low = low[:2]
            high = high[:2]
        action_space = Box(low, high)
        mdp_info.action_space = action_space

        #observation_spec += [("left_EE_pos", "EE_ur5left", ObservationType.SITE_POS),]
        #observation_spec += [("right_EE_pos", "EE_ur5right", ObservationType.SITE_POS),]
        #observation_spec += [("plate_pos", "grommet_11mm", ObservationType.BODY_POS),]
        #observation_spec += [("plate_rot", "grommet_11mm", ObservationType.BODY_ROT),]
        #observation_spec += [("plate_vel", "grommet_11mm", ObservationType.BODY_VEL),]
        #observation_spec += [("pegs_pos", "quad_peg", ObservationType.BODY_POS),]
        #observation_spec += [("pegs_rot", "quad_peg", ObservationType.BODY_ROT),]
        #observation_spec += [("stand_joint_pos", "ur_stand_joint", ObservationType.JOINT_POS),]
        #observation_spec += [(f"left_joint{i}_pos", f"joint{i}_ur5left", ObservationType.JOINT_POS) for i in range(6)]
        #observation_spec += [(f"right_joint{i}_pos", f"joint{i}_ur5right", ObservationType.JOINT_POS) for i in range(6)]
        #observation_spec += [("stand_joint_vel", "ur_stand_joint", ObservationType.JOINT_VEL),]
        #observation_spec += [(f"left_joint{i}_vel", f"joint{i}_ur5left", ObservationType.JOINT_VEL) for i in range(6)]
        #observation_spec += [(f"right_joint{i}_vel", f"joint{i}_ur5right", ObservationType.JOINT_VEL) for i in range(6)]
        #low_o = np.concatenate([
        #    -np.ones(3), # left_EE_pos
        #    -np.ones(3), # right_EE_pos
        #    -np.ones(3), # plate_pos
        #    -np.ones(4), # plate_rot
        #    -3. * np.ones(3), # plate_vel

        #    
        #observation_space = Box(low_o, high_o)
        return mdp_info

    def inverse_kinematics(self, robot, pos, rot):
        #return True
        assert robot in ["left", "right"]
        #result = ik.qpos_from_site_pose(dm_physics, f"ft_frame_ur5{robot}", pos, rot,
        self.dm_physics.data.qpos[:] = self._data.qpos[:]
        result = ik.qpos_from_site_pose(self.dm_physics, f"EE_ur5{robot}", pos, rot,
                            joint_names=[f"joint{i}_ur5{robot}" for i in range(6)])
        if result.success:
            qpos = result.qpos
            id0 = self._data.joint(f"joint0_ur5{robot}").id
            self._data.qpos[id0:id0 + 6] = qpos[id0:id0 + 6]
            mujoco.mj_fwdPosition(self._model, self._data)
            return True
        return False 

    def get_current_robot_state(self):
        return self._data.qpos[self.robot_joint_ids], self._data.qvel[self.robot_joint_ids]

    def get_plate(self, obs):
        plate_pos = self.obs_helper.get_from_obs(obs, "plate_pos")
        plate_rot = self.obs_helper.get_from_obs(obs, "plate_rot")
        plate_vel = self.obs_helper.get_from_obs(obs, "plate_vel")
        return plate_pos, plate_rot, plate_vel

    def get_pegs(self, obs):
        pegs_pos = self.obs_helper.get_from_obs(obs, "pegs_pos")
        pegs_rot = self.obs_helper.get_from_obs(obs, "pegs_rot")
        return pegs_pos, pegs_rot

    def get_robots_state(self, obs):
        base_joint_pos = [self.obs_helper.get_from_obs(obs, "stand_joint_pos")]
        left_joint_pos = [self.obs_helper.get_from_obs(obs, f"left_joint{i}_pos") for i in range(6)]
        right_joint_pos = [self.obs_helper.get_from_obs(obs, f"right_joint{i}_pos") for i in range(6)]
        base_joint_vel = [self.obs_helper.get_from_obs(obs, "stand_joint_vel")]
        left_joint_vel = [self.obs_helper.get_from_obs(obs, f"left_joint{i}_vel") for i in range(6)]
        right_joint_vel = [self.obs_helper.get_from_obs(obs, f"right_joint{i}_vel") for i in range(6)]
        return np.array(base_joint_pos + left_joint_pos + right_joint_pos), \
               np.array(base_joint_vel + left_joint_vel + right_joint_vel)

    def get_ee_poses(self, obs):
        left_EE_pos = self.obs_helper.get_from_obs(obs, "left_EE_pos")
        right_EE_pos = self.obs_helper.get_from_obs(obs, "right_EE_pos")
        return left_EE_pos, right_EE_pos


    def _compute_action(self, obs, action):
        controls = self._controller(action[0], action[1], action[2])
        self._data.ctrl[self.left_gripper_id] = self.gripper_force
        self._data.ctrl[self.right_gripper_id] = self.gripper_force
        return controls

        
    def _controller(self, desired_pos, desired_vel, desired_acc):
        clipped_pos, clipped_vel = desired_pos, desired_vel

        error = (clipped_pos - self._data.qpos[self.robot_joint_ids])

        self.i_error += self.i_gain * error * self.dt
        torque = self.p_gain * error + self.d_gain * (clipped_vel - self._data.qvel[self.robot_joint_ids]) + self.i_error

        mujoco.mj_forward(self._model, self._data)

        tau_ff = np.zeros(self._model.nv)
        acc_ff = np.zeros(self._model.nv)
        acc_ff[self.robot_joint_ids] = desired_acc
        mujoco.mj_mulM(self._model, self._data, tau_ff, acc_ff)
        torque += tau_ff[self.robot_joint_ids]

        # Gravity Compensation and Coriolis and Centrifugal force
        torque += self._data.qfrc_bias[self.robot_joint_ids]

        return torque
        

    def get_grip_pose(self):
        grip_height = 0.015
        handle_orientation_mat = euler2mat([np.pi/2., 0., 0.])
        left_handle_pos = self._data.geom("quad_handle_b").xpos
        left_handle_world_mat = self._data.geom("quad_handle_b").xmat.reshape(3, 3)
        left_handle_mat = left_handle_world_mat @ handle_orientation_mat
        left_handle_mjquat = np.zeros(4)
        mujoco.mju_mat2Quat(left_handle_mjquat, left_handle_mat.reshape(-1))
        left_grip_pos = left_handle_pos + np.matmul(left_handle_mat, np.array([0.0, 0.0, -grip_height]))
        right_handle_world_mat = self._data.geom("quad_handle_a").xmat.reshape(3, 3)
        right_handle_pos = self._data.geom("quad_handle_a").xpos
        right_handle_mat = right_handle_world_mat @ handle_orientation_mat
        right_handle_mjquat = np.zeros(4)
        mujoco.mju_mat2Quat(right_handle_mjquat, right_handle_mat.reshape(-1))
        right_grip_pos = right_handle_pos + np.matmul(right_handle_mat, np.array([0.0, 0.0, -grip_height]))
        return left_grip_pos, left_handle_mjquat, right_grip_pos, right_handle_mjquat
        

    def move_plate_to_pose(self, plate_pos, plate_quat):
        plate_angle = np.arctan2(plate_pos[1], plate_pos[0])
        stand_joint_rot = plate_angle - np.pi / 2

        # sanity check
        #plate_pos = self._model.body("quad_peg").pos + np.array([0., 0., 0.03])
        #plate_quat = self._model.body("quad_peg").quat

        #self._data.joint("ur_stand_joint").qpos = -np.pi / 2
        #self._data.joint("ur_stand_joint").qpos = -np.pi / 20.
        self._data.joint("ur_stand_joint").qpos = stand_joint_rot 
        self._data.joint("free_joint_quad_grommet").qpos[:3] = plate_pos
        self._data.joint("free_joint_quad_grommet").qpos[3:] = plate_quat
        #self._data.joint("free_joint_quad_peg").qpos[:3] = plate_pos
        #self._data.joint("free_joint_quad_peg").qpos[3:] = quad_quat
        mujoco.mj_fwdPosition(self._model, self._data)

        left_grip_pos, left_handle_mjquat, right_grip_pos, right_handle_mjquat = self.get_grip_pose()            

        left_ik_success = self.inverse_kinematics("left", left_grip_pos, left_handle_mjquat)
        right_ik_success = self.inverse_kinematics("right", right_grip_pos, right_handle_mjquat)
        mujoco.mj_fwdPosition(self._model, self._data)
        return left_ik_success and right_ik_success

    def setup(self, obs):
        self.absorbing_type = AbsorbType.NONE
        # place plate in random position and move robots to its handles
        while True:
            #plate_pos = np.random.uniform(low=[-0.1, -0.1, 0.0], high=[0.1, 0.1, 0.0]) + np.array([0.0, 0.6, 0.1])
            #plate_pos = np.array([0.0, 0.6, 0.1])
            #plate_pos = np.array([0.5, 0.0, 0.2])
            #plate_pos = np.array([0.0, 0.6, 0.025])
            #plate_pos = np.array([0.0, 0.6, 0.2])
            #plate_pos = np.array([0.0, 0.6, 0.4])
            # easy
            #plate_pos = np.random.uniform(low=[-0.1, -0.1, -0.1], high=[0.1, 0.1, 0.1]) + np.array([0.0, 0.6, 0.4])
            # middle, hard
            plate_pos = np.random.uniform(low=[-0.5, -0.4, -0.2], high=[0.5, 0.2, 0.2]) + np.array([0.0, 0.6, 0.4])

            plate_angle = np.arctan2(plate_pos[1], plate_pos[0])
            # easy
            #plate_rot = np.random.uniform(low=[0, 0, 0], high=[0., 0., 0.]) + np.array([np.pi - plate_angle, 0., 0.])
            # middle
            max_angle = np.pi / 10.
            plate_rot = np.random.uniform(low=[-max_angle, -max_angle, -max_angle], high=[max_angle, max_angle, max_angle]) + np.array([np.pi - plate_angle, 0., 0.])
            # hard
            #max_angle = np.pi / 6
            #plate_rot = np.random.uniform(low=[-3*max_angle, -max_angle, -max_angle], high=[3*max_angle, max_angle, max_angle]) + np.array([np.pi - plate_angle, 0., 0.])
            plate_quat = euler2quat(plate_rot)

            success = self.move_plate_to_pose(plate_pos, plate_quat)
            if success:
                break
        
        ## close grippers
        #for i in range(80):
        #    desired_pos = self._data.qpos.copy()[self.robot_joint_ids]
        #    desired_vel = np.zeros_like(desired_pos)
        #    desired_acc = np.zeros_like(desired_pos)
        #    torques = self._controller(desired_pos, desired_vel, desired_acc)
        #    self._data.ctrl[self.actuator_joint_ids] = torques
        #    self._data.ctrl[self.left_gripper_id] = self.gripper_force
        #    self._data.ctrl[self.right_gripper_id] = self.gripper_force
        #    mujoco.mj_step(self._model, self._data)
        #    # maintain plate position
        #    self._data.joint("free_joint_quad_grommet").qpos[:3] = plate_pos
        #    self._data.joint("free_joint_quad_grommet").qpos[3:] = plate_quat
        #    mujoco.mj_fwdPosition(self._model, self._data)
        #    #self.render()
        #    #print(i)
        #    #sleep(0.03)
        #gj = {}
        #for r in ["left", "right"]:
        #    for j in ["left_inner_finger_joint", "right_inner_finger_joint",
        #              "left_inner_knuckle_joint", "right_inner_knuckle_joint",
        #              "left_outer_knuckle_joint", "right_outer_knuckle_joint"]:
        #        joint_name = f"{j}_ur5{r}"
        #        self._data.qpos[self._data.joint(joint_name).id] = self._data.joint(joint_name).qpos
        #        gj[joint_name] = self._data.joint(joint_name).qpos
        #        self._data.qvel[self._data.joint(joint_name).id] = 0.
        #        self._data.qacc[self._data.joint(joint_name).id] = 0.

        for k, v in self.desired_gripper_joints.items():
            self._data.qpos[self._data.joint(k).id] = v
        
        mujoco.mj_fwdPosition(self._model, self._data)

        super().setup(obs)
        mujoco.mj_fwdPosition(self._model, self._data)
        obs = self._create_observation(self.obs_helper._build_obs(self._data))
        self.min_weighted_dist = self.weighted_dist(obs)
        return obs

    def weighted_dist(self, state):
        goal_pos_dist, goal_rot_dist = self.goal_dists(state)
        return 2. * goal_pos_dist +  1. * goal_rot_dist

    def is_absorbing(self, state):
        # check for goal reaching
        goal_pos_dist, goal_rot_dist = self.goal_dists(state)
        #if goal_pos_dist < 0.015 and goal_rot_dist < 0.015:
        # NOTE only position is considered as it is hard to define the orientation limits
        if goal_pos_dist < 0.015:
            self.absorbing_type = AbsorbType.SUCCESS
            return True

        # check for dropping the plate
        left_ee_pos, right_ee_pos = self.get_ee_poses(state)
        left_grip_pos, left_handle_mjquat, right_grip_pos, right_handle_mjquat = self.get_grip_pose()            
        left_grip_dist = np.linalg.norm(left_grip_pos - left_ee_pos)
        right_grip_dist = np.linalg.norm(right_grip_pos - right_ee_pos)
        if left_grip_dist > 0.015 or right_grip_dist > 0.015:
            self.absorbing_type = AbsorbType.DROP
            return True

        # check for bad grasps
        gripper_joint_errors = [np.abs(self._data.qpos[self._data.joint(k).id] - v) for k, v in self.desired_gripper_joints.items()]
        #for k, v in self.desired_gripper_joints.items():
        #    print(k, np.abs(self._data.qpos[self._data.joint(k).id] - v))
        #    if np.abs(self._data.qpos[self._data.joint(k).id] - v) > 0.2:
        #        self.absorbing_type = AbsorbType.DROP
        #        return True
        #print("NORM:", np.linalg.norm(gripper_joint_errors))
        #print("MEAN:", np.mean(gripper_joint_errors))
        #print("MAX:", np.max(gripper_joint_errors))
        if np.mean(gripper_joint_errors) > 0.2:
            self.absorbing_type = AbsorbType.DROP
            return True
        return False

    def goal_dists(self, state):
        plate_pos, plate_mjquat, _ = self.get_plate(state)
        pegs_pos, pegs_mjquat = self.get_pegs(state)

        target_pos = pegs_pos + np.array([0., 0., 0.025])
        target_mjquat = pegs_mjquat

        goal_pos_dist = np.linalg.norm(plate_pos - target_pos)
        goal_rot_dist = rotation_distance(plate_mjquat, target_mjquat)
        return goal_pos_dist, goal_rot_dist

    def reward(self, state, action, next_state, absorbing):
        weighted_dist = self.weighted_dist(state)

        #reward = -0.003 # default penalty for spending time
        #if weighted_dist < self.min_weighted_dist:
        #    reward += self.min_weighted_dist - weighted_dist
        #    self.min_weighted_dist = weighted_dist

        #reward = -weighted_dist
        #reward = np.exp(-2. * weighted_dist**2)

        #reward = max(self.min_weighted_dist - weighted_dist, 0.) / self.min_weighted_dist
        #reward = reward ** 2
        reward = 1. / ((weighted_dist / self.min_weighted_dist) + 0.01) - 1.
        reward *= 0.01
        if absorbing:
            t = self._data.time
            it = int(t / self.info.dt)
            horizon = self.info.horizon
            gamma = self.info.gamma 
            factor = (1 - gamma ** (horizon - it + 1)) / (1 - gamma)
            if self.absorbing_type == AbsorbType.SUCCESS:
                #reward += 10.
                #reward += 100.
                reward *= factor * self.success_scale
                print("Success")
            elif self.absorbing_type == AbsorbType.DROP:
                #reward -= 10.
                #reward -= 100.
                #reward *= factor / mul
                reward = 0.
                print("DROP")
        return reward

    def _create_info_dictionary(self, state):
        left_ee_pos, right_ee_pos = self.get_ee_poses(state)
        ee_dist = np.linalg.norm(left_ee_pos - right_ee_pos)
        ee_dist_constraint = np.abs(ee_dist - self.constraints.ee_dist)
        j_pos, j_vel = self.get_robots_state(state)

        #left_ee_vel = self._data.body("EE_ur5left").cvel[3:]
        #right_ee_vel = self._data.body("EE_ur5right").cvel[3:]
        ##_, _, plate_vel = self.get_plate(state)
        #plate_vel = self._data.body("grommet_11mm").cvel
        #plate_vel_rot = plate_vel[:3]
        #plate_vel_lin = plate_vel[3:]
        #rl = left_ee_pos - self._data.body("grommet_11mm").subtree_com
        #left_ee_vel_ = plate_vel_lin + np.cross(plate_vel_rot, rl)
        #left_ee_vel__ = plate_vel_lin + np.cross(plate_vel_rot, -rl)
        #left_handle_vel = self._data.body("handle_11mm_flap_b").cvel[3:]
        left_grip_pos, left_handle_mjquat, right_grip_pos, right_handle_mjquat = self.get_grip_pose()            

        ## sanity check of the FK computation
        ## right robot first in xml
        #j_pos_ = np.concatenate([j_pos[:1], j_pos[7:13], j_pos[1:7]])
        #left_ee_pos_ = self.constraints.compute_forward_kinematics(torch.tensor(j_pos.T)[None],
        #                                                           torch.tensor(j_vel.T)[None])[0]

        left_ee_mat = self._data.site("EE_ur5left").xmat.reshape(3, 3)
        right_ee_mat = self._data.site("EE_ur5right").xmat.reshape(3, 3)
        left_right_vector = left_ee_pos - right_ee_pos
        error_left = np.abs(left_ee_mat @ left_right_vector)
        error_right = np.abs(right_ee_mat @ left_right_vector)
        #z_error_left = np.abs(left_ee_mat @ left_right_vector)[-1]
        #z_error_right = np.abs(right_ee_mat @ left_right_vector)[-1]

        goal_pos_dist, goal_rot_dist = self.goal_dists(state)

        task_info = {}

        task_info['joint_vel_constraint'] = np.sum(np.maximum(np.abs(j_vel) - self.env_info['robot']['joint_vel_limit'][-1], 0))
        task_info['ee_dist_constraint'] = ee_dist_constraint
        task_info['left_distance_constraint'] = np.linalg.norm(error_left - np.array([self.constraints.ee_dist, 0., 0.]))
        task_info['right_distance_constraint'] = np.linalg.norm(error_right - np.array([self.constraints.ee_dist, 0., 0.]))
        task_info['orientation_constraint'] = error_mat(left_ee_mat, right_ee_mat)
        task_info['goal_pos_dist'] = goal_pos_dist
        task_info['goal_rot_dist'] = goal_rot_dist
        task_info['left_handle_dist'] = np.linalg.norm(left_grip_pos - left_ee_pos)
        task_info['right_handle_dist'] = np.linalg.norm(right_grip_pos - right_ee_pos)

        task_info['left_ee_orientation'] = mat2euler(self._data.site("EE_ur5left").xmat.reshape(3, 3))
        task_info['right_ee_orientation'] = mat2euler(self._data.site("EE_ur5right").xmat.reshape(3, 3))

        #task_info["success"] = goal_pos_dist < 0.015#and goal_rot_dist < 0.015
        task_info["success"] = (self.absorbing_type == AbsorbType.SUCCESS)
        return task_info


if __name__ == "__main__":
    env = BimanualEnv(interpolation_order=5)
    env.reset()
    robot_pos = env.get_current_robot_state()[0]
    for i in range(10000):
        #robot_pos[0] += 0.001
        #robot_pos[1] += 0.0001
        action = np.stack([robot_pos, np.zeros_like(robot_pos), np.zeros_like(robot_pos)], axis=0)
        state, reward, absorbing, info = env.step(action)
        env.render()

        plate_pos = np.array([0.01, 0.6, 0.04])
        plate_angle = np.arctan2(plate_pos[1], plate_pos[0])
        plate_rot = np.random.uniform(low=[0, 0, 0], high=[0., 0., 0.]) + np.array([np.pi - plate_angle, 0., 0.])
        plate_quat = euler2quat(plate_rot)
        success = env.move_plate_to_pose(plate_pos, plate_quat)
        robot_pos = env.get_current_robot_state()[0]

        #print(env._data.site("EE_ur5left").xmat.reshape(3, 3))
        #ee_left = env._data.site("EE_ur5left").xpos
        #ee_right = env._data.site("EE_ur5right").xpos
        #print(np.linalg.norm(ee_left - ee_right))
        #env.reward(None, None, None, None)
    print("Done.")