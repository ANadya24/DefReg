import numpy as np
import math
from skimage import transform as trf
from matplotlib import pyplot as plt
from scipy import io as spio
from skimage import io
from glob import glob
from ffRemap import *
from scipy.ndimage import interpolation
import pickle
import cv2


def adjust01(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


def WarpCoords2(poi, V, out_size):
    h = out_size[1]
    w = out_size[2]
    # indices = poi[0, :, 0].reshape(-1, 1), poi[0, :, 1].reshape(-1, 1)
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    indices = np.column_stack([np.reshape(x, (-1, 1)), np.reshape(y, (-1, 1))])
    vec_x = interpolate.griddata(indices, V[0, :, :, 0].reshape(-1, 1), poi[0], method='linear')
    vec_y = interpolate.griddata(indices, V[0, :, :, 1].reshape(-1, 1), poi[0], method='linear')
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
    # V = np.pad(V, ((0,0), (1,1), (1,1), (0,0)))
    # out_height = V.shape[1]
    # out_width = V.shape[2]
    # poi[:, :, 0] = poi[:, :, 0] + 1
    # poi[:, :, 1] = poi[:, :, 1] + 1

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


def draw(img, points, lens, color):
    im = img.copy()
    bound, inner, lines = points
    len1, len2,len3, len4 = lens
    for p in bound:
        cv2.circle(im, tuple(p.astype('int')), 3, color)
    for p in inner:
        cv2.circle(im, tuple(p.astype('int')), 3, color)
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


DRAW_POINTS = True

if __name__ == "__main__":
    prefix = 'fwd'
    model_name = 'ssim1'
    sequences = glob('/data/sim/Notebooks/VM/data/*1.tif')
    for seq_name in filter(lambda name: name.find('Seq') != -1, sequences):
        print(seq_name)
        # seq_name = '/home/nadya/Projects/VoxelMorph/data/SeqB1.tif'
        point_name = seq_name.split('.tif')[0] + '.mat'
        subf = ('/').join(seq_name.split('/')[:-1]) + f'/registered/result/{model_name}/deformations/'
        def_name = subf + seq_name.split('/')[-1].split('.tif')[0] + '.npy'
        images = io.imread(seq_name).astype('float')
        c, h, w = images.shape
        poi = spio.loadmat(point_name)
        deformation = np.load(def_name)
        baseline_def_name = ('/').join(seq_name.split('/')[:-1]) + f'/deformations/numpy/' \
                            + seq_name.split('/')[-1].split('.tif')[0] +f'_{prefix}.npy'
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

        if DRAW_POINTS:
            seq_name_elast = '/data/sim/Notebooks/VM/data/viz/fwd/check/elastic/' + seq_name.split('/')[-1]
            seq_name_prop = '/data/sim/Notebooks/VM/data/viz/fwd/check/proposed/init_' + seq_name.split('/')[-1]
            seq_init = io.imread(seq_name)
            seq_init = np.stack([seq_init, seq_init, seq_init], -1)
            seq_init[0] = draw(seq_init[0], (bound[0], inner[0], lines[0]), (len1, len2, len3, len4), (255, 255, 255))

            # seq = io.imread(f'/home/nadya/Projects/VoxelMorph/data/registered/result/{prefix}/' + seq_name.split('/')[-1])
            seq = io.imread(seq_name_prop)
            seq = np.stack([seq, seq, seq], -1)
            our_seq = [draw(seq[0], (bound[0], inner[0], lines[0]), (len1, len2, len3, len4), (255, 0, 0))]

            # bseq = io.imread(f'/home/nadya/Projects/VoxelMorph/data/registered/gt/{prefix}/' + seq_name.split('/')[-1])
            bseq = io.imread(seq_name_elast)
            # print('bseq shape', bseq.shape)
            bseq = np.stack([bseq, bseq, bseq], -1)
            base_seq = [draw(bseq[0], (bound[0], inner[0], lines[0]), (len1, len2, len3, len4), (0, 0, 255))]
        in_sh = images.shape

        init_def = None
        init_def_v = None
        for i in range(1, len(images)):
            b_p = bound[i]
            in_p = inner[i]
            line_p = lines[i]

            v = deformation[i]
            base_v = base_deformation[i]

            if i != 1:
                base_v = ff_1_to_k(init_def, base_v)
                v = ff_1_to_k(init_def_v, v)
            init_def = base_v.copy()
            init_def_v = v.copy()
            print(v.min(), v.max(), base_v.min(), base_v.max())

            if prefix == 'fwd':
                base_v *= -1.
                v *= -1.

            def_b_p = WarpCoords(b_p[None], v[None], (1, in_sh[1], in_sh[2]))[0]
            def_in_p = WarpCoords(in_p[None], v[None], (1, in_sh[1], in_sh[2]))[0]
            def_line_p = WarpCoords(line_p[None], v[None], (1, in_sh[1], in_sh[2]))[0]

            base_b_p = WarpCoords(b_p[None], base_v[None], (1, in_sh[1], in_sh[2]))[0]
            base_in_p = WarpCoords(in_p[None], base_v[None], (1, in_sh[1], in_sh[2]))[0]
            base_line_p = WarpCoords(line_p[None], base_v[None], (1, in_sh[1], in_sh[2]))[0]

            if DRAW_POINTS:
                seq_draw = draw(seq[i], (def_b_p, def_in_p, def_line_p), (len1, len2, len3, len4), (255, 0, 0))
                # seq_draw = draw(seq_draw, (bound[0], inner[0], lines[0]), (len1, len2, len3, len4), (0, 255, 255))
                our_seq.append(seq_draw)
                seq_draw = draw(bseq[i], (base_b_p, base_in_p, base_line_p), (len1, len2, len3, len4), (0, 0, 255))
                # seq_draw = draw(seq_draw, (bound[0], inner[0], lines[0]), (len1, len2, len3, len4), (0, 255, 255))
                base_seq.append(seq_draw)
                seq_init[i] = draw(seq_init[i], (bound[i], inner[i], lines[i]), (len1, len2, len3, len4), (255, 255, 255))



        if DRAW_POINTS:
            io.imsave(f'/data/sim/Notebooks/VM/data/viz/{prefix}/init_' + seq_name.split('/')[-1], seq_init)
            io.imsave(f'/data/sim/Notebooks/VM/data/viz/{prefix}/prop_' + seq_name.split('/')[-1],
                      np.array(our_seq))
            io.imsave(f'/data/sim/Notebooks/VM/data/viz/{prefix}/elastic_' + seq_name.split('/')[-1],
                      np.array(base_seq))