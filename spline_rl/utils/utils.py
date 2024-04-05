import torch
import numpy as np

def huber_loss(x, dt):
    x = torch.nn.HuberLoss(reduction='none')(x, torch.zeros_like(x))
    x = torch.sum(x * dt, dim=1)
    return x

def limit_loss(x, dt, limit):
    loss = torch.relu(x - limit)
    loss = huber_loss(loss, dt)
    return loss

def equality_loss(x, dt, value):
    loss = x - value
    loss = huber_loss(loss, dt)
    return loss

def unpack_data_airhockey(x):
    n = 7
    puck = x[..., :3]
    puck_dot = x[..., 3:6]
    q0 = x[..., 6:n+6]
    dq0 = x[..., n+6:2*n+6]
    opponent_mallet = x[..., 2*n+6:2*n+9]
    z = torch.zeros_like(dq0) if isinstance(dq0, torch.Tensor) else np.zeros_like(dq0)
    ddq0 = z
    qk = z
    dqk = z
    ddqk = z
    return puck, puck_dot, q0, qk, dq0, dqk, ddq0, ddqk, opponent_mallet

def unpack_data_ndof(x, n=7):
    q0 = x[..., :n]
    dq0 = x[..., n:2*n]
    ddq0 = x[..., 2*n:3*n]
    qk = x[..., 3*n:4*n]
    dqk = x[..., 4*n:5*n]
    ddqk = x[..., 5*n:6*n]
    return q0, qk, dq0, dqk, ddq0, ddqk

def unpack_data_obstacles2D(x):
    xy0 = x[..., :2]
    dxy0 = x[..., 2:4]
    xyk = x[..., 4:6]
    dxyk = x[..., 6:8]
    obstacles = x[..., 8:38]
    return xy0, xyk, dxy0, dxyk, obstacles
    
def project_entropy(chol, e_lb):
    a_dim = chol.size()[-1]
    def entropy(chol):
        return a_dim / 2 * np.log(2 * np.pi * np.e) + torch.diagonal(chol, dim1=-2, dim2=-1).log().sum(-1)
    ent = entropy(chol)[:, None, None]
    chol = torch.where(ent < e_lb, chol * torch.exp((e_lb - ent) / a_dim), chol)
    return chol

def project_entropy_independently(chol, e_lb):
    a_dim = chol.size()[-1]
    c = a_dim / 2 * np.log(2 * np.pi * np.e)
    avg_log_diag = (e_lb - c) / a_dim
    chol_diag = torch.maximum(chol.diagonal(dim1=-2, dim2=-1).log(), torch.tensor(avg_log_diag)).exp()
    chol_ = torch.diag_embed(chol_diag, dim1=-2, dim2=-1)
    return chol_