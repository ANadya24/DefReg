import numpy as np
import scipy as sp
from scipy import interpolate
#TODO add description


def ffremap(def_prev, def_cur):
    h, w, _ = def_prev.shape
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    x_grid = x + def_prev[:, :, 0]
    y_grid = y + def_prev[:, :, 1]
    new_def = np.empty(def_prev.shape)
    indices = np.column_stack([np.reshape(x, (-1, 1)), np.reshape(y, (-1, 1))])
    new_def[:, :, 0] = sp.interpolate.griddata(indices, def_cur[:, :, 0].reshape(-1, 1),
                                               (x_grid, y_grid), method='linear', fill_value=0).squeeze()
    new_def[:, :, 1] = sp.interpolate.griddata(indices, def_cur[:, :, 1].reshape(-1, 1),
                                               (x_grid, y_grid), method='linear', fill_value=0).squeeze()
    return new_def


def ffremap2(def_prev, def_cur):
    h, w, _ = def_prev.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x_grid = x + def_prev[:, :, 0]
    y_grid = y + def_prev[:, :, 1]
    x_grid = np.clip(x_grid, 0, w - 2)
    y_grid = np.clip(y_grid, 0, h - 2)
    cy = y_grid - np.floor(y_grid)
    cx = x_grid - np.floor(x_grid)
    ix = np.floor(x_grid).astype(np.int32)
    iy = np.floor(y_grid).astype(np.int32)

    vy00 = def_cur[iy, ix, 1]
    vy01 = def_cur[iy+1, ix, 1]
    vy10 = def_cur[iy, ix+1, 1]
    vy11 = def_cur[iy + 1, ix + 1, 1]

    vx00 = def_cur[iy, ix, 0]
    vx01 = def_cur[iy+1, ix, 0]
    vx10 = def_cur[iy, ix+1, 0]
    vx11 = def_cur[iy + 1, ix + 1, 0]

    ys = (vy11 * cx * cy + vy10 * cy * (1 - cx) + vy01 * cx * (1 - cy) + vy00 * (1 - cx) * (
            1 - cy))
    xs = (vx11 * cx * cy + vx10 * cy * (1 - cx) + vx01 * cx * (1 - cy) + vx00 * (1 - cx) * (
            1 - cy))

    new_def = np.zeros(def_prev.shape)

    new_def[..., 0] = xs
    new_def[..., 1] = ys

    return new_def


def dots_remap_bcw(dots, deformation, num=4):
    h, w = deformation.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x + deformation[:, :, 0]
    y = y + deformation[:, :, 1]

    out_dots = dots.copy()
    for d in range(len(dots)):
        k = ((x - dots[d, 0]) ** 2 + (y - dots[d, 1]) ** 2) ** 0.5
        indexes = []
        dist = []
        fx = 0
        fy = 0
        for _ in range(num):
            id = np.unravel_index(np.argmin(k, axis=None), k.shape)
            dist.append(k[id])
            indexes.append(id)
            k[id] = k.max()
        indexes = np.array(indexes)
        indexesi = indexes[:, 0]
        indexesj = indexes[:, 1]

        for i, (idi, idj) in enumerate(zip(indexesi, indexesj)):
            fx += -1*deformation[idi, idj, 0]
            fy += -1*deformation[idi, idj, 1]

        fx = fx / num
        fy = fy / num

        out_dots[d, 0] += fx
        out_dots[d, 1] += fy

    out_dots[:, 0] = np.clip(out_dots[:, 0], 0, w)
    out_dots[:, 1] = np.clip(out_dots[:, 1], 0, h)
    return out_dots


def dots_remap_bcw_mt(dots, deformation, num=4):
    h, w = deformation.shape[:2]
    x, y = np.meshgrid(np.arange(1, w + 1), np.arange(1, h + 1))

    def_x_grid = x + deformation[:, :, 0]
    def_y_grid = y + deformation[:, :, 1]

    def_x = deformation[..., 0]
    def_y = deformation[..., 1]

    valid_points = (dots[:, 0] > 0) * (dots[:, 1] > 0) * \
                   (dots[:, 0] < w) * (dots[:, 1] < h) * ~np.isnan(dots[:, 0]) * ~np.isnan(dots[:, 1])
    out_dots = dots.copy()
    if valid_points.sum() != 0:
        ind = np.where(valid_points > 0)[0]
        for d in ind:
            p_dist = (def_x_grid - dots[d, 0]) ** 2 + (def_y_grid - dots[d, 1]) ** 2
            p_dist_ind = np.argsort(p_dist.reshape(-1))
            fx = -np.mean(def_x[np.unravel_index(p_dist_ind[:num], p_dist.shape)])
            fy = -np.mean(def_y[np.unravel_index(p_dist_ind[:num], p_dist.shape)])

            out_dots[d, 0] += fx
            out_dots[d, 1] += fy

    return out_dots


def ff_1_to_k(ff_1_to_k_minus_1, ff_k_minus_1_to_k):
    new_def = ff_1_to_k_minus_1 + ffremap2(ff_1_to_k_minus_1, ff_k_minus_1_to_k)
    return new_def


def forward_warp(image, deformation):
    h, w, _ = deformation.shape
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    x_grid = x + deformation[:, :, 0]
    y_grid = y + deformation[:, :, 1]
    indices = np.column_stack([np.reshape(x, (-1, 1)), np.reshape(y, (-1, 1))])
    im_new = sp.interpolate.griddata(indices, image.reshape(-1, 1), (x_grid, y_grid),
                                     method='linear', fill_value=0)
    return np.clip(im_new.squeeze(), 0, 255).astype(np.uint8)


def backward_warp(image, deformation):
    h, w, _ = deformation.shape
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    x_grid = x + deformation[:, :, 0]
    y_grid = y + deformation[:, :, 1]
    indices = np.column_stack([np.reshape(x_grid, (-1, 1)), np.reshape(y_grid, (-1, 1))])
    im_new = sp.interpolate.griddata(indices, image.reshape(-1, 1), (x, y), method='linear', fill_value=0)
    return np.clip(im_new.squeeze(), 0, 255).astype(np.uint8)
