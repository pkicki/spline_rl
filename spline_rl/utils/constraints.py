import os
import torch
import numpy as np

from storm_kit.differentiable_robot_model.differentiable_robot_model import DifferentiableRobotModel

from utils.utils import equality_loss, limit_loss


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
