from time import perf_counter
import torch
import numpy as np


def inside_box(xyz, xl, xh, yl, yh, zl, zh):
    pxl = xyz[..., 0] > xl
    pxh = xyz[..., 0] < xh
    pyl = xyz[..., 1] > yl
    pyh = xyz[..., 1] < yh
    pzl = xyz[..., 2] > zl
    pzh = xyz[..., 2] < zh
    return torch.all(torch.stack([pxl, pxh, pyl, pyh, pzl, pzh], axis=-1), axis=-1)


def inside_rectangle(xyz, xl, xh, yl, yh):
    pxl = xyz[..., 0] > xl
    pxh = xyz[..., 0] < xh
    pyl = xyz[..., 1] > yl
    pyh = xyz[..., 1] < yh
    return torch.all(torch.stack([pxl, pxh, pyl, pyh], axis=-1), axis=-1)


def dist_point_2_box(xyz, xl, xh, yl, yh, zl, zh):
    l = torch.stack([xl, yl, zl], axis=-1)
    h = torch.stack([xh, yh, zh], axis=-1)
    xyz_dist = torch.max(torch.stack([l - xyz, torch.zeros_like(xyz), xyz - h], axis=-1), axis=-1)[0]
    dist = torch.sqrt(torch.sum(torch.square(xyz_dist), axis=-1) + 1e-8)
    return dist


def dist_point_2_box_inside(xyz, xl, xh, yl, yh, zl, zh):
    dist = torch.min(torch.abs(torch.stack([xyz[..., 0] - xl, xyz[..., 0] - xh,
                                          xyz[..., 1] - yl, xyz[..., 1] - yh,
                                          xyz[..., 2] - zl, xyz[..., 2] - zh,
                                          ], axis=-1)), axis=-1)[0]
    return dist


def collision_with_box(xyz, r, xl, xh, yl, yh, zl, zh):
    o = torch.ones_like(xyz[..., 0])
    xl, xh, yl, yh, zl, zh = (x * o for x in [xl, xh, yl, yh, zl, zh])
    inside = inside_box(xyz, xl, xh, yl, yh, zl, zh)
    dist2box = dist_point_2_box(xyz, xl, xh, yl, yh, zl, zh)
    dist2box = torch.nn.functional.relu(r - dist2box)
    collision = torch.where(inside, torch.zeros_like(dist2box), dist2box)
    return collision


def simple_collision_with_box(xyz, xl, xh, yl, yh, zl, zh):
    o = torch.ones_like(xyz[..., 0])
    xl, xh, yl, yh, zl, zh = (x * o for x in [xl, xh, yl, yh, zl, zh])
    inside = inside_box(xyz, xl, xh, yl, yh, zl, zh)
    dist2box_inside = dist_point_2_box_inside(xyz, xl, xh, yl, yh, zl, zh)
    collision = torch.where(inside, dist2box_inside, torch.zeros_like(dist2box_inside))
    return collision