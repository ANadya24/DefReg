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


def ffremap2(def_prev, def_cur, num=4):
    h, w, _ = def_prev.shape
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    x_grid = x + def_prev[:, :, 0]
    y_grid = y + def_prev[:, :, 1]
    x_grid = np.clip(x_grid, 0, w - 2)
    y_grid = np.clip(y_grid, 0, h - 2)
    # y_grid = y_grid.reshape(-1, 1)
    # def_cur = np.pad(def_cur, ((1, 1), (1, 1), (0, 0)))
    # x_grid = x_grid + 1
    # y_grid = y_grid + 1
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


def dots_remap_bcw(dots, deformation, num=5):
    def get_prod(ii, seq, val_d):
        prod = 1
        for j in range(len(seq)):
            if ii == j:
                continue
            prod *= (val_d - seq[j]) / (seq[i] - seq[j])
        return prod

    h, w = deformation.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x + deformation[:, :, 0]
    y = y + deformation[:, :, 1]
    deformation *= -1.

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
        indexes = np.array(indexes).reshape(-1, 2)
        indexesi = indexes[:,0]
        indexesj = indexes[:,1]
        # print(list(zip(x[indexesi, indexesj],y[indexesi, indexesj])), dots[d])
        # input()
        # indices = np.column_stack([np.reshape(x[indexesi, indexesj],
        #                                       (-1, 1)), np.reshape(y[indexesi, indexesj], (-1, 1))])
        # fx = sp.interpolate.griddata(indices, deformation[indexesi, indexesj, 0].reshape(-1, 1),
        #                                        dots[d], method='linear', fill_value=0).squeeze()
        # fy = sp.interpolate.griddata(indices, deformation[indexesi, indexesj, 1].reshape(-1, 1),
        #                                        dots[d], method='linear', fill_value=0).squeeze()
        dist_sum = sum([1/di for di in dist])
        for i, (idi, idj) in enumerate(zip(indexesi, indexesj)):
            fx += deformation[idi, idj, 0] / dist[i]
            fy += deformation[idi, idj, 1] / dist[i]
            # fx += deformation[idi, idj, 0] * get_prod(i, x[indexesi, indexesj], dots[d, 0]) * \
            #       get_prod(i, y[indexesi, indexesj], dots[d, 1])
            # fy += deformation[idi, idj, 1] * get_prod(i, x[indexesi, indexesj], dots[d, 0]) * \
            #       get_prod(i, y[indexesi, indexesj], dots[d, 1])
            # print(fx, fy)
            # input()
        fx = fx/dist_sum
        fy = fy/dist_sum
        # print(deformation[indexesi, indexesj], [fx, fy])
        # input()
        dots[d] += [fx, fy]
    # dots[:, 0] -= fx
    # dots[:, 1] -= fy
    dots[:, 0] = np.clip(dots[:, 0], 0, w)
    dots[:, 1] = np.clip(dots[:, 1], 0, h)
    return dots


def ff_1_to_k(ff_1_to_k_minus_1, ff_k_minus_1_to_k):
    # new_def = ffremap2(ff_1_to_k_minus_1, ff_k_minus_1_to_k)
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
    return np.clip(im_new.squeeze(), 0, 255).astype('uint8')


def backward_warp(image, deformation):
    h, w, _ = deformation.shape
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    x_grid = x + deformation[:, :, 0]
    y_grid = y + deformation[:, :, 1]
    indices = np.column_stack([np.reshape(x_grid, (-1, 1)), np.reshape(y_grid, (-1, 1))])
    im_new = sp.interpolate.griddata(indices, image.reshape(-1, 1), (x, y), method='linear', fill_value=0)
    return np.clip(im_new.squeeze(), 0, 255).astype('uint8')
