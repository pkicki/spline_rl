import os
import torch
import numpy as np

from storm_kit.differentiable_robot_model.differentiable_robot_model import DifferentiableRobotModel

from utils.utils import equality_loss, limit_loss



#import pinocchio as pino

#class FKPin(torch.autograd.Function):
#    urdf_path = os.path.join(os.path.dirname(__file__), "../urdf/iiwa_striker.urdf")
#    pino_model = pino.buildModelFromUrdf(urdf_path)
#    pino_data = pino_model.createData()
#    @staticmethod
#    def forward(ctx, q):
#        q = q.detach().numpy()
#        batch_dims = q.shape[:-1]
#        q_last_dim = q.shape[-1]
#        q = np.concatenate([q, np.zeros(batch_dims + (FKPin.pino_model.nv - q_last_dim,))], axis=-1)
#        q_flat = q.reshape((-1, q.shape[-1]))
#        ctx.q_flat = q_flat
#        ctx.q_last_dim = q_last_dim
#        xyz = np.zeros((q_flat.shape[0], 3))
#        for i in range(q_flat.shape[0]):
#            pino.forwardKinematics(FKPin.pino_model, FKPin.pino_data, q_flat[i])
#            xyz[i] = FKPin.pino_data.oMf[-1].translation
#            #pino.computeJointJacobians(FKPin.pino_model, FKPin.pino_data, q_flat[i])
#        xyz = xyz.reshape(batch_dims + (3,))
#        return torch.tensor(xyz)
#
#    @staticmethod
#    def backward(ctx, grad_output):
#        q_flat = ctx.q_flat
#        q_last_dim = ctx.q_last_dim
#        batch_dims = grad_output.shape[:-1]
#        Js = np.zeros((q_flat.shape[0], 3, FKPin.pino_model.nv))
#        for i in range(q_flat.shape[0]):
#            pino.computeJointJacobians(FKPin.pino_model, FKPin.pino_data, q_flat[i])
#            Js[i] = FKPin.pino_data.J[:3]
#        J = Js.reshape(batch_dims + (3, FKPin.pino_model.nv))
#        J = torch.tensor(J)[..., :q_last_dim]
#        grad_output = (grad_output[..., None, :] @ J)[..., 0, :]
#        return grad_output


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
        self.constraints_num = 17
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
        #ee_pos = FKPin.apply(q)

        x_ee_loss_low = limit_loss(self.ee_x_lb, dt, ee_pos[..., 0])[..., None]
        y_ee_loss_low = limit_loss(self.ee_y_lb, dt, ee_pos[..., 1])[..., None]
        y_ee_loss_high = limit_loss(ee_pos[..., 1], dt, self.ee_y_ub)[..., None]
        z_ee_loss = equality_loss(ee_pos[..., 2], dt, self.ee_z_eb)[..., None]

        constraint_losses = torch.cat([q_dot_loss, q_ddot_loss, x_ee_loss_low, y_ee_loss_low,
                                        y_ee_loss_high, z_ee_loss], dim=-1)
        return constraint_losses
