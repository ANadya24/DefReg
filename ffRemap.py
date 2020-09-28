import numpy as np
import scipy as sp
from scipy import interpolate
from scipy.ndimage import interpolation


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
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    x_grid = x + def_prev[:, :, 0]
    y_grid = y + def_prev[:, :, 1]
    x_grid = np.clip(x_grid, 0, w-2)
    y_grid = np.clip(y_grid, 0, h-2)
    # y_grid = y_grid.reshape(-1, 1)

    cy = y_grid - np.floor(y_grid)
    cx = x_grid - np.floor(x_grid)
    ix = np.floor(x_grid).astype('int')
    iy = np.floor(y_grid).astype('int')

    vy00 = def_cur[iy, ix, 1]
    vy01 = def_cur[iy, ix + 1, 1]
    vy10 = def_cur[iy + 1, ix, 1]
    vy11 = def_cur[iy + 1, ix + 1, 1]

    vx00 = def_cur[iy, ix, 0]
    vx01 = def_cur[iy, ix + 1, 0]
    vx10 = def_cur[iy + 1, ix, 0]
    vx11 = def_cur[iy + 1, ix + 1, 0]

    ys = (vy11 * cx * cy + vy10 * cy * (1 - cx) + vy01 * cx * (1 - cy) + vy00 * (1 - cx) * (
            1 - cy))
    xs = (vx11 * cx * cy + vx10 * cy * (1 - cx) + vx01 * cx * (1 - cy) + vx00 * (1 - cx) * (
            1 - cy))

    # xs = xs.reshape((h, w))
    # ys = ys.reshape((h, w))

    new_def = np.zeros(def_prev.shape)

    new_def[:, :, 0] = xs
    new_def[:, :, 1] = ys

    return new_def


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
    return  np.clip(im_new.squeeze(), 0, 255).astype('uint8')


def backward_warp(image, deformation):
    h, w, _ = deformation.shape
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    x_grid = x + deformation[:, :, 0]
    y_grid = y + deformation[:, :, 1]
    indices = np.column_stack([np.reshape(x_grid, (-1, 1)), np.reshape(y_grid, (-1, 1))])
    im_new = sp.interpolate.griddata(indices, image.reshape(-1, 1), (x, y), method='linear', fill_value=0)
    return np.clip(im_new.squeeze(), 0, 255).astype('uint8')