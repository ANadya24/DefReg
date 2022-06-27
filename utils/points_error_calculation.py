import math
import numpy as np
import cv2
from scipy import interpolate
from matplotlib import pyplot as plt


def WarpCoords2(poi, V, out_size):
    h = out_size[1]
    w = out_size[2]
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    indices = np.column_stack([np.reshape(x, (-1, 1)), np.reshape(y, (-1, 1))])
    vec_x = interpolate.griddata(indices, V[0, :, :, 0].reshape(-1), poi[0], method='linear')
    vec_y = interpolate.griddata(indices, V[0, :, :, 1].reshape(-1), poi[0], method='linear')
    p = np.array(poi)
    p[0, :, 0] += vec_x.squeeze()
    p[0, :, 1] += vec_y.squeeze()
    return p


def WarpCoords(poi, V, out_size):
    num_batch = out_size[0]
    out_height = out_size[1]
    out_width = out_size[2]

    V = np.transpose(V, [0, 3, 1, 2])  # [n, 2, h, w]
    cy = poi[:, :, 1] - np.floor(poi[:, :, 1])
    cx = poi[:, :, 0] - np.floor(poi[:, :, 0])

    idx = np.floor(poi[:, :, 0]).astype('int')
    idy = np.floor(poi[:, :, 1]).astype('int')
    vy00 = np.zeros((num_batch, poi.shape[1]))
    vx00 = np.zeros((num_batch, poi.shape[1]))
    vy01 = np.zeros((num_batch, poi.shape[1]))
    vx01 = np.zeros((num_batch, poi.shape[1]))
    vy10 = np.zeros((num_batch, poi.shape[1]))
    vx10 = np.zeros((num_batch, poi.shape[1]))
    vy11 = np.zeros((num_batch, poi.shape[1]))
    vx11 = np.zeros((num_batch, poi.shape[1]))

    for b in range(num_batch):
        iy = idy[b]
        ix = idx[b]
        vy00[b] = V[b, 1, iy, ix]
        vy01[b] = V[b, 1, iy, ix + 1]
        vy10[b] = V[b, 1, iy + 1, ix]
        vy11[b] = V[b, 1, iy + 1, ix + 1]

        vx00[b] = V[b, 0, iy, ix]
        vx01[b] = V[b, 0, iy, ix + 1]
        vx10[b] = V[b, 0, iy + 1, ix]
        vx11[b] = V[b, 0, iy + 1, ix + 1]

    ys = (vy11 * cx * cy + vy10 * cy * (1 - cx) + vy01 * cx * (1 - cy) + vy00 * (1 - cx) * (
            1 - cy))
    xs = (vx11 * cx * cy + vx10 * cy * (1 - cx) + vx01 * cx * (1 - cy) + vx00 * (1 - cx) * (
            1 - cy))
    p = np.array(poi)

    p[:, :, 0] += xs
    p[:, :, 1] += ys
    p[:, :, 0] = np.clip(p[:, :, 0], 0, out_width - 1)
    p[:, :, 1] = np.clip(p[:, :, 1], 0, out_height - 1)

    return p


def area(line1, line2):
    return abs(np.trapz(line1[:, 1], line1[:, 0]) - np.trapz(line2[:, 1], line2[:, 0]))


def euc_dist(pt1, pt2):
    return math.sqrt((pt2[0] - pt1[0]) * (pt2[0] - pt1[0]) + (pt2[1] - pt1[1]) * (pt2[1] - pt1[1]))


def _c(ca, i, j, P, Q):
    if ca[i, j] > -1:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = euc_dist(P[0], Q[0])
    elif i > 0 and j == 0:
        ca[i, j] = max(_c(ca, i - 1, 0, P, Q), euc_dist(P[i], Q[0]))
    elif i == 0 and j > 0:
        ca[i, j] = max(_c(ca, 0, j - 1, P, Q), euc_dist(P[0], Q[j]))
    elif i > 0 and j > 0:
        ca[i, j] = max(min(_c(ca, i - 1, j, P, Q), _c(ca, i - 1, j - 1, P, Q), _c(ca, i, j - 1, P, Q)),
                       euc_dist(P[i], Q[j]))
    else:
        ca[i, j] = float("inf")
    return ca[i, j]


def frechetDist(P, Q):
    """ Computes the discrete frechet distance between two polygonal lines
    Algorithm: http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf
    P and Q are arrays of 2-element arrays (points)
    """
    ca = np.ones((len(P), len(Q)))
    ca = np.multiply(ca, -1)
    return _c(ca, len(P) - 1, len(Q) - 1, P, Q)


def plot_bef_aft(init_err, err, base_err, title='test', x_label='', y_label='', save=''):
    err[0] = 0.
    plt.plot(np.arange(len(init_err)), init_err, 'kx-', linewidth=1, label='Unregistered')
    plt.plot(np.arange(len(err)), err, 'rx-', linewidth=1, label='Proposed method')
    plt.plot(np.arange(len(base_err)), base_err, 'bx-', linewidth=1, label='Elast. dynamic method')
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xlim(0, len(init_err) - 1)
    plt.ylim(0)
    plt.grid()
    plt.legend(loc='upper left')
    # plt.show()
    if save != '':
        plt.savefig(save, bbox_inches="tight")
        plt.close()


def draw(img, points, lens, color):
    im = img.copy()
    bound, inner, lines = points
    len1, len2,len3, len4 = lens
    for p in bound:
        cv2.circle(im, tuple(p.astype('int')), 3, color, -1)
    for p in inner:
        cv2.circle(im, tuple(p.astype('int')), 3, color, -1)
    line = lines[:len1]
    for p1, p2 in zip(line[:-1], line[1:]):
        cv2.line(im, tuple(p1.astype('int')), tuple(p2.astype('int')), color, 2)
    line = lines[len1:len1+len2]
    for p1, p2 in zip(line[:-1], line[1:]):
        cv2.line(im, tuple(p1.astype('int')), tuple(p2.astype('int')), color, 2)
    line = lines[len1+len2:len1+len2+len3]
    for p1, p2 in zip(line[:-1], line[1:]):
        cv2.line(im, tuple(p1.astype('int')), tuple(p2.astype('int')), color, 2)
    line = lines[len3+len2+len1:]
    for p1, p2 in zip(line[:-1], line[1:]):
        cv2.line(im, tuple(p1.astype('int')), tuple(p2.astype('int')), color, 2)
    return im