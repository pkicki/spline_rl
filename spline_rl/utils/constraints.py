import os
from spline_rl.utils.collisions import collision_with_box, simple_collision_with_box
import torch
import numpy as np
from time import perf_counter

from storm_kit.differentiable_robot_model.differentiable_robot_model import DifferentiableRobotModel

from spline_rl.utils.utils import equality_loss, huber, limit_loss


class Constraint:
    def __init__(self) -> None:
        pass

    def evaluate(self, q, q_dot, q_ddot, dt):
        raise NotImplementedError

class AirHockeyConstraints(Constraint):
    def __init__(self, q_dot_max, q_ddot_max, ee_x_lb, ee_y_lb, ee_y_ub, ee_z_eb) -> None:
        self.q_dot_max = q_dot_max
        self.q_ddot_max = q_ddot_max
        self.ee_x_lb = ee_x_lb
        self.ee_y_lb = ee_y_lb
        self.ee_y_ub = ee_y_ub
        self.ee_z_eb = ee_z_eb
        self.violation_limits = np.array([1e-4] * 7 + [1e-5] * 7 + [5e-6] * 4)
        self.constraints_num = 18
        self.urdf_path = os.path.join(os.path.dirname(__file__), "../urdf/iiwa_striker.urdf")
        self.robot = DifferentiableRobotModel(urdf_path=self.urdf_path, name="iiwa")

    def compute_forward_kinematics(self, q, q_dot):
        q_ = q.reshape((-1, q.shape[-1]))
        q_ = torch.cat([q_, torch.zeros((q_.shape[0], 9 - q_.shape[1]))], dim=-1)
        q_dot_ = q_dot.reshape((-1, q_dot.shape[-1]))
        q_dot_ = torch.cat([q_dot_, torch.zeros((q_dot_.shape[0], 9 - q_dot_.shape[1]))], dim=-1)
        ee_pos, ee_rot = self.robot.compute_forward_kinematics(q_, q_dot_, "F_striker_tip")
        ee_pos = ee_pos.reshape((q.shape[0], q.shape[1], 3))
        ee_rot = ee_rot.reshape((q.shape[0], q.shape[1], 3, 3))
        return ee_pos, ee_rot

    def evaluate(self, q, q_dot, q_ddot, dt):
        # TODO: make this shape adaptation more general and not hardcoded
        dt_ = dt[..., None]
        # Prepare the constraint limits tensors
        q_dot_limits = torch.Tensor(self.q_dot_max)[None, None]
        q_ddot_limits = torch.Tensor(self.q_ddot_max)[None, None]

        q_dot_loss = limit_loss(torch.abs(q_dot), dt_, q_dot_limits)
        q_ddot_loss = limit_loss(torch.abs(q_ddot), dt_, q_ddot_limits)

        ee_pos, ee_rot = self.compute_forward_kinematics(q, q_dot)

        x_ee_loss_low = limit_loss(self.ee_x_lb, dt, ee_pos[..., 0])[..., None]
        y_ee_loss_low = limit_loss(self.ee_y_lb, dt, ee_pos[..., 1])[..., None]
        y_ee_loss_high = limit_loss(ee_pos[..., 1], dt, self.ee_y_ub)[..., None]
        z_ee_loss = equality_loss(ee_pos[..., 2], dt, self.ee_z_eb)[..., None]

        constraint_losses = torch.cat([q_dot_loss, q_ddot_loss, x_ee_loss_low, y_ee_loss_low,
                                        y_ee_loss_high, z_ee_loss], dim=-1)
        return constraint_losses


class KinodynamicCupConstraints(Constraint):
    def __init__(self, q_max, q_dot_max, q_ddot_max, torque_max, cup_width, cup_height, robot_radius,
                 box1_xl, box1_xh, box1_yl, box1_yh, box1_height, box2_xl, box2_xh, box2_yl, box2_yh, box2_height) -> None:
        self.q_max = q_max
        self.q_dot_max = q_dot_max
        self.q_ddot_max = q_ddot_max
        self.q_torque_max = torque_max
        self.cup_height = cup_height 
        self.cup_width = cup_width
        self.robot_radius = robot_radius
        self.box1_xl = box1_xl
        self.box1_xh = box1_xh
        self.box1_yl = box1_yl
        self.box1_yh = box1_yh
        self.box1_height = box1_height
        self.box2_xl = box2_xl
        self.box2_xh = box2_xh
        self.box2_yl = box2_yl
        self.box2_yh = box2_yh
        self.box2_height = box2_height
        self.violation_limits = np.array([1e-4] * 7 + [1e-4] * 7 + [1e-5] * 7 + [1e-5] * 7 + [1e-8] + [1e-6] * 4)
        self.constraints_num = len(self.violation_limits)
        self.urdf_path = os.path.join(os.path.dirname(__file__), "../urdf/iiwa_cup.urdf")
        self.robot = DifferentiableRobotModel(urdf_path=self.urdf_path, name="iiwa_cup")
        h = self.cup_height / 2.
        w = self.cup_width / 2.
        xyz_cuboid = np.array([
                                # corners
                            [w, w, h], [w, w, -h], [w, -w, h], [w, -w, -h],
                            [-w, w, h], [-w, w, -h], [-w, -w, h], [-w, -w, -h],
                            # middle points on the edges
                            [w, w, 0], [w, -w, 0], [-w, w, 0], [-w, -w, 0],
                            [w, 0, h], [w, 0, -h], [-w, 0, h], [-w, 0, -h],
                            [0, w, h], [0, w, -h], [0, -w, h], [0, -w, -h],
                            # middle points on the faces
                            [w, 0, 0], [-w, 0, 0],
                            [0, w, 0], [0, -w, 0],
                            [0, 0, h], [0, 0, -h],
                            ])[np.newaxis, np.newaxis]
        self.cuboid = torch.tensor(xyz_cuboid, dtype=torch.float32)[..., None]

    def compute_inverse_dynamics(self, q, q_dot, q_ddot):
        q_ = q.reshape((-1, q.shape[-1]))
        q_dot_ = q_dot.reshape((-1, q_dot.shape[-1]))
        q_ddot_ = q_ddot.reshape((-1, q_ddot.shape[-1]))
        torques = self.robot.compute_inverse_dynamics(q_, q_dot_, q_ddot_)
        torques = torques.reshape(q.shape)
        return torques

    def compute_forward_kinematics(self, q, q_dot):
        q_ = q.reshape((-1, q.shape[-1]))
        q_dot_ = q_dot.reshape((-1, q_dot.shape[-1]))
        ee_pos, ee_rot = self.robot.compute_forward_kinematics(q_, q_dot_, "F_link_cup")
        ee_pos = ee_pos.reshape((*q.shape[:-1], 3))
        ee_rot = ee_rot.reshape((*q.shape[:-1], 3, 3))
        return ee_pos, ee_rot

    def interpolate_links(self, xyzs):
        #dists = torch.linalg.norm(torch.diff(xyzs, axis=-2), dim=-1)
        xyzs_ = [xyzs[..., 0, :]]
        for i, n in enumerate([1, 2, 2, 2, 1, 2, 1]):
            s = torch.linspace(0., 1., n + 2)[1:]
            for x in s:
                xyzs_.append(x * xyzs[..., i + 1, :] + (1. - x) * xyzs[..., i, :])
        xyzs_interp = torch.stack(xyzs_, axis=-2)
        #dists_ = torch.linalg.norm(torch.diff(xyzs_interp, axis=-2), dim=-1)
        return xyzs_interp

    def two_tables_object_collision(self, xyz, R, dt):
        huber_along_path = lambda x: torch.sum(dt * huber(x), axis=-1)

        robot_collision_table_1 = collision_with_box(xyz, self.robot_radius, self.box1_xl, self.box1_xh,
                                                    self.box1_yl, self.box1_yh, -1e10,
                                                    self.box1_height)
        robot_collision_table_2 = collision_with_box(xyz, self.robot_radius, self.box2_xl, self.box2_xh,
                                                    self.box2_yl, self.box2_yh, -1e10,
                                                    self.box2_height)

        robot_collision_table_1_loss = huber_along_path(torch.sum(robot_collision_table_1, axis=-1))
        robot_collision_table_2_loss = huber_along_path(torch.sum(robot_collision_table_2, axis=-1))

        xyz_end = xyz[:, :, -1:]
        xyz_object = xyz_end + (R[:, :, None] @ self.cuboid)[..., 0]

        object_collision_table_1 = simple_collision_with_box(xyz_object, self.box1_xl, self.box1_xh,
                                                            self.box1_yl, self.box1_yh, -1e10,
                                                            self.box1_height)
        object_collision_table_2 = simple_collision_with_box(xyz_object, self.box2_xl, self.box2_xh,
                                                            self.box2_yl, self.box2_yh, -1e10,
                                                            self.box2_height)
        object_collision_table_1_loss = huber_along_path(torch.sum(object_collision_table_1, axis=-1))
        object_collision_table_2_loss = huber_along_path(torch.sum(object_collision_table_2, axis=-1))
        constraint_losses = torch.stack([robot_collision_table_1_loss, robot_collision_table_2_loss,
                                         object_collision_table_1_loss, object_collision_table_2_loss,
                                        ], axis=-1)
        return constraint_losses

    def evaluate(self, q, q_dot, q_ddot, dt):
        # TODO: make this shape adaptation more general and not hardcoded
        dt_ = dt[..., None]
        # Prepare the constraint limits tensors
        q_limits = torch.Tensor(self.q_max)[None, None]
        q_dot_limits = torch.Tensor(self.q_dot_max)[None, None]
        q_ddot_limits = torch.Tensor(self.q_ddot_max)[None, None]
        q_torque_limits = torch.Tensor(self.q_torque_max)[None, None]

        q_loss = limit_loss(torch.abs(q), dt_, q_limits)
        q_dot_loss = limit_loss(torch.abs(q_dot), dt_, q_dot_limits)
        q_ddot_loss = limit_loss(torch.abs(q_ddot), dt_, q_ddot_limits)

        torque = self.compute_inverse_dynamics(q, q_dot, q_ddot)
        q_torque_loss = limit_loss(torch.abs(torque), dt_, q_torque_limits)


        ee_pos, ee_rot = self.compute_forward_kinematics(q, q_dot)

        xyzs = torch.stack([self.robot.get_link_pose(f'F_link_{i}')[0] for i in list(range(7)) + ['ee']], dim=1)
        xyzs = xyzs.reshape(q.shape[:-1] + xyzs.shape[1:])
        xyzs_interpolated = self.interpolate_links(xyzs)
        collision_loss = self.two_tables_object_collision(xyzs_interpolated, ee_rot, dt_[..., 0])

        orientation_loss = equality_loss(ee_rot[..., 2, 2], dt, 1.)[..., None]

        constraint_losses = torch.cat([q_loss, q_dot_loss, q_ddot_loss,
                                       q_torque_loss, orientation_loss, collision_loss], dim=-1)
        return constraint_losses
