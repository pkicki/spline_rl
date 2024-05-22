from time import perf_counter
from spline_rl.utils.utils import np_huber
import torch
import numpy as np


def np_inside_box(xyz, xl, xh, yl, yh, zl, zh):
    pxl = xyz[..., 0] > xl
    pxh = xyz[..., 0] < xh
    pyl = xyz[..., 1] > yl
    pyh = xyz[..., 1] < yh
    pzl = xyz[..., 2] > zl
    pzh = xyz[..., 2] < zh
    return np.all(np.stack([pxl, pxh, pyl, pyh, pzl, pzh], axis=-1), axis=-1)


def np_inside_rectangle(xyz, xl, xh, yl, yh):
    pxl = xyz[..., 0] > xl
    pxh = xyz[..., 0] < xh
    pyl = xyz[..., 1] > yl
    pyh = xyz[..., 1] < yh
    return np.all(np.stack([pxl, pxh, pyl, pyh], axis=-1), axis=-1)


def np_dist_point_2_box(xyz, xl, xh, yl, yh, zl, zh):
    l = np.stack([xl, yl, zl], axis=-1)
    h = np.stack([xh, yh, zh], axis=-1)
    xyz_dist = np.max(np.stack([l - xyz, np.zeros_like(xyz), xyz - h], axis=-1), axis=-1)
    dist = np.sqrt(np.sum(np.square(xyz_dist), axis=-1) + 1e-8)
    return dist


def np_dist_point_2_box_inside(xyz, xl, xh, yl, yh, zl, zh):
    dist = np.min(np.abs(np.stack([xyz[..., 0] - xl, xyz[..., 0] - xh,
                                   xyz[..., 1] - yl, xyz[..., 1] - yh,
                                   xyz[..., 2] - zl, xyz[..., 2] - zh,
                                   ], axis=-1)), axis=-1)
    return dist


def np_collision_with_box(xyz, r, xl, xh, yl, yh, zl, zh):
    o = np.ones_like(xyz[..., 0])
    xl, xh, yl, yh, zl, zh = (x * o for x in [xl, xh, yl, yh, zl, zh])
    inside = np_inside_box(xyz, xl, xh, yl, yh, zl, zh)
    dist2box = np_dist_point_2_box(xyz, xl, xh, yl, yh, zl, zh)
    dist2box = np.maximum(r - dist2box, 0.)
    collision = np.where(inside, np.zeros_like(dist2box), dist2box)
    return collision


def np_simple_collision_with_box(xyz, xl, xh, yl, yh, zl, zh):
    o = np.ones_like(xyz[..., 0])
    xl, xh, yl, yh, zl, zh = (x * o for x in [xl, xh, yl, yh, zl, zh])
    inside = np_inside_box(xyz, xl, xh, yl, yh, zl, zh)
    dist2box_inside = np_dist_point_2_box_inside(xyz, xl, xh, yl, yh, zl, zh)
    collision = np.where(inside, dist2box_inside, np.zeros_like(dist2box_inside))
    return collision

def np_two_tables_object_collision(xyz, R, dt, robot_radius, box1_xl, box1_xh, box1_yl, box1_yh, box1_height,
                                   box2_xl, box2_xh, box2_yl, box2_yh, box2_height, cup_width, cup_height):
    huber_along_path = lambda x: np.sum(dt * np_huber(x), axis=-1)

    robot_collision_table_1 = np_collision_with_box(xyz, robot_radius, box1_xl, box1_xh,
                                                box1_yl, box1_yh, -1e10,
                                                box1_height)
    robot_collision_table_2 = np_collision_with_box(xyz, robot_radius, box2_xl, box2_xh,
                                                box2_yl, box2_yh, -1e10,
                                                box2_height)

    robot_collision_table_1_loss = huber_along_path(np.sum(robot_collision_table_1, axis=-1))
    robot_collision_table_2_loss = huber_along_path(np.sum(robot_collision_table_2, axis=-1))

    h = cup_height / 2.
    w = cup_width / 2.
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
                        ], dtype=np.float32)[np.newaxis, np.newaxis, ..., np.newaxis]
    xyz_end = xyz[:, :, -1:]
    xyz_object = xyz_end + (R[:, :, None] @ xyz_cuboid)[..., 0]

    object_collision_table_1 = np_simple_collision_with_box(xyz_object, box1_xl, box1_xh,
                                                        box1_yl, box1_yh, -1e10,
                                                        box1_height)
    object_collision_table_2 = np_simple_collision_with_box(xyz_object, box2_xl, box2_xh,
                                                        box2_yl, box2_yh, -1e10,
                                                        box2_height)
    object_collision_table_1_loss = huber_along_path(np.sum(object_collision_table_1, axis=-1))
    object_collision_table_2_loss = huber_along_path(np.sum(object_collision_table_2, axis=-1))
    constraint_losses = np.stack([robot_collision_table_1_loss, robot_collision_table_2_loss,
                                        object_collision_table_1_loss, object_collision_table_2_loss,
                                    ], axis=-1)
    return constraint_losses