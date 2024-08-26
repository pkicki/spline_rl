import numpy as np
from scipy.spatial.transform import Rotation as R

def euler2quat(xyz):
    return R.from_euler("xyz", xyz).as_quat()

def quat2euler(quat):
    return R.from_quat(quat).as_euler("xyz")

def quat2mat(quat):
    return R.from_quat(quat).as_matrix()

def mat2quat(mat):
    return R.from_matrix(mat).as_quat()

def mat2euler(mat):
    return R.from_matrix(mat).as_euler("xyz")

def euler2mat(xyz):
    return R.from_euler("xyz", xyz).as_matrix()

def mjquat2quat(mjquat):
    return np.concatenate([mjquat[1:], mjquat[:1]])

def quat2mjquat(quat):
    return np.concatenate([quat[3:], quat[:3]])

def error_mat(a, b):
    return (R.from_matrix(a).inv() * R.from_matrix(b)).magnitude()

def quat_mul(p, q):
    """
    Multiplies two quaternions
    param p: quaternion
    param q: quaternion
    r: quaternion
    """
    assert p.shape == q.shape, "p and q should be quaternion"
    r = np.zeros(p.shape)
    r[0] = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3]
    r[1] = p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2]
    r[2] = p[0]*q[2] - p[1]*q[3] + p[2]*q[0] + p[3]*q[1]
    r[3] = p[0]*q[3] + p[1]*q[2] - p[2]*q[1] + p[3]*q[0]
    r_ = R.from_quat(p) * R.from_quat(q)
    return r

def rotation_distance(p: np.array, q: np.array):
    """
    Calculates the rotation angular between two quaternions
    param p: quaternion
    param q: quaternion
    theta: rotation angle between p and q (rad)
    """
    assert p.shape == q.shape, "p and q should be quaternion"
    theta = 2 * np.arccos(abs(p @ q))
    return theta