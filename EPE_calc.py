# this script calculates initial and elastic method errors
import numpy as np
import math
from skimage import transform as trf
from matplotlib import pyplot as plt
from scipy import io as spio
from skimage import io
from glob import glob
from ffRemap import  *
from scipy.ndimage import interpolation
import pickle


def adjust01(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


def WarpCoords2(poi, V, out_size):
    h = out_size[1]
    w = out_size[2]
    indices = poi[0, :, 0].reshape(-1, 1), poi[0, :, 1].reshape(-1, 1)
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    indices = np.column_stack([np.reshape(x, (-1, 1)), np.reshape(y, (-1, 1))])
    vec_x = interpolate.griddata(indices, V[0, :, :, 0].reshape(-1), poi[0], method='linear')
    vec_y = interpolate.griddata(indices, V[0, :, :, 1].reshape(-1), poi[0], method='linear')
    # vec_x = interpolation.map_coordinates(indices, V[0, :, :, 0], order=1, mode='reflect')
    # vec_y = interpolation.map_coordinates(indices, V[0, :, :, 1], order=1, mode='reflect')
    # print(vec_x.min(), vec_x.max(), V.min(), V.max())
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
    elif i > 0 and j >  0:
        ca[i, j] = max(min(_c(ca, i - 1, j, P, Q), _c(ca, i - 1, j - 1, P, Q), _c(ca, i, j - 1, P, Q)),
                       euc_dist(P[i], Q[j]))
    else:
        ca[i, j] = float("inf")
    return ca[i, j]


""" Computes the discrete frechet distance between two polygonal lines
Algorithm: http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf
P and Q are arrays of 2-element arrays (points)
"""


def frechetDist(P, Q):
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


if __name__ == "__main__":
    prefix = 'bcw'
    sequences = glob('/home/nadya/Projects/VoxelMorph/data/*.tif')
    for seq_name in filter(lambda name: name.find('Seq') != -1, sequences):
        print(seq_name)
        point_name = seq_name.split('.tif')[0] + '.mat'
        subf = ('/').join(seq_name.split('/')[:-1]) + '/deformations/numpy/'
        baseline_def_name = subf + seq_name.split('/')[-1].split('.tif')[0] + f'_{prefix}.npy'
        images = io.imread(seq_name).astype('float')
        c, h, w = images.shape
        poi = spio.loadmat(point_name)
        base_deformation = np.load(baseline_def_name)

        bound = np.stack(poi['spotsB'][0].squeeze())
        inner = np.stack(poi['spotsI'][0].squeeze())

        bound = bound[:, :, :2]
        inner = inner[:, :, :2]

        line1 = np.stack(poi['lines'][:, 0])
        line2 = np.stack(poi['lines'][:, 1])
        line3 = np.stack(poi['lines'][:, 2])
        line4 = np.stack(poi['lines'][:, 3])

        len1 = len(line1[0])
        len2 = len(line2[0])
        len3 = len(line3[0])
        len4 = len(line4[0])

        lines = np.concatenate((line1, line2, line3, line4), axis=1)

        in_sh = images.shape

        arr = np.zeros(in_sh, dtype='float32')

        fbound = bound[0][None]
        finner = inner[0][None]
        fline = lines[0]
        fline = np.stack([fline] * in_sh[0])

        base_bound_err = np.zeros(len(images))
        base_inner_err = np.zeros(len(images))
        base_line_err = np.zeros(len(images))

        b1 = np.zeros(len(images))
        b2 = np.zeros(len(images))
        b3 = np.zeros(len(images))
        b4 = np.zeros(len(images))

        for f in range(len(images)):
            b1[f] = frechetDist(lines[f, :len1], fline[f, :len1])
            b2[f] = frechetDist(lines[f, len1:len1 + len2], fline[f, len1:len1 + len2])
            b3[f] = frechetDist(lines[f, len1 + len2:len1 + len2 + len3],
                                fline[f, len1 + len2:len1 + len2 + len3])
            b4[f] = frechetDist(lines[f, len1 + len2 + len3:], fline[f, len1 + len2 + len3:])
        line_init_err = (b1 + b2 + b3 + b4) / 4.  # float(fline.shape[1])
        init_err = ((((bound - fbound) ** 2).sum(axis=2)) ** 0.5).sum(axis=1) / float(fbound.shape[1])
        initin_err = ((((inner - finner) ** 2).sum(axis=2)) ** 0.5).sum(axis=1) / float(finner.shape[1])

        init_def = None

        initial_data = {}
        initial_data['bound'] = init_err.tolist()
        initial_data['inner'] = initin_err.tolist()
        initial_data['lines'] = line_init_err.tolist()

        with open('/home/nadya/Projects/VoxelMorph/data/graphics/initial_data/unregistered_' +
                  seq_name.split('/')[-1].split('.')[0], 'wb') as wrt_f:
            pickle.dump(initial_data, wrt_f)

        for i in range(1, len(images)):
            x = images[i]
            b_p = bound[i]
            in_p = inner[i]
            line_p = lines[i]

            base_v = base_deformation[i]
            if i != 1:
                base_v = ff_1_to_k(init_def, base_v)
            init_def = base_v.copy()

            if prefix == 'fwd':
                base_v *= -1.

            base_b_p = WarpCoords(b_p[None], base_v[None], (1, in_sh[1], in_sh[2]))[0]
            base_in_p = WarpCoords(in_p[None], base_v[None], (1, in_sh[1], in_sh[2]))[0]
            base_line_p = WarpCoords(line_p[None], base_v[None], (1, in_sh[1], in_sh[2]))[0]

            base_bound_err[i] = ((((base_b_p - fbound.squeeze()) ** 2).sum(axis=1)) ** 0.5).sum(axis=0) / \
                           float(fbound.shape[1])

            base_inner_err[i] = ((((base_in_p - finner.squeeze()) ** 2).sum(axis=1)) ** 0.5).sum(axis=0) / \
                           float(finner.shape[1])
            b1 = frechetDist(base_line_p[:len1], fline[0, :len1])
            b2 = frechetDist(base_line_p[len1:len1 + len2], fline[0, len1:len1 + len2])
            b3 = frechetDist(base_line_p[len1 + len2:len1 + len2 + len3], fline[0, len1 + len2:len1 + len2 + len3])
            b4 = frechetDist(base_line_p[len1 + len2 + len3:], fline[0, len1 + len2 + len3:])
            base_line_err[i] = (b1 + b2 + b3 + b4) / 4.

        elast_data = {}
        elast_data['bound'] = base_bound_err.tolist()
        elast_data['inner'] = base_inner_err.tolist()
        elast_data['lines'] = base_line_err.tolist()
        print(np.mean(init_err), np.mean(base_bound_err))
        print(np.mean(initin_err), np.mean(base_inner_err))
        print(np.mean(line_init_err), np.mean(base_line_err))

        with open(f'/home/nadya/Projects/VoxelMorph/data/graphics/initial_data/elastic_method_{prefix}_' +
                  seq_name.split('/')[-1].split('.')[0], 'wb') as wrt_f:
            pickle.dump(elast_data, wrt_f)