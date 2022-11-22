# this script visualize predicted deformations by calculationg errors from point position
import numpy as np
from scipy import io as spio
from skimage import io
from glob import glob
import argparse
import os

from utils.ffRemap import dots_remap_bcw, ff_1_to_k
from utils.points_error_calculation import draw


def parse_args():
    parser = argparse.ArgumentParser(description='EPE script for drawing')
    parser.add_argument('--prefix', type=str, default='fwd',
                        help='prefix of deformation type')
    parser.add_argument('--sequence_path', type=str,
                        help='whole sequence path or path, where tif image sequneces are stored')
    parser.add_argument('--seq_name_pattern', type=str, default='Seq',
                        help='path, where tif image sequneces are stored')
    parser.add_argument('--save_path', type=str,
                        help='path where to save drawings')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.save_drawing_path, exist_ok=True)

    if args.sequence_path[-3:] == 'tif':
        sequences = [args.sequence_path]
    else:
        sequences = glob(args.sequence_path + '/*.tif')
    for seq_name in filter(lambda name: name.find(args.seq_name_pattern) != -1, sequences):
        print(seq_name)
        images = io.imread(seq_name).astype(np.float32)

        # load annotated points from initial unregistered sequence
        point_name = seq_name.split('.tif')[0] + '.mat'
        poi = spio.loadmat(point_name)

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

        # initial unregistered sequence
        seq_init = io.imread(seq_name)
        seq_init = np.stack([seq_init, seq_init, seq_init], -1)
        seq_init[0] = draw(seq_init[0], (bound[0], inner[0], lines[0]),
                           (len1, len2, len3, len4), (255, 255, 255))

        for i in range(1, len(images)):
            bound_points = bound[i].copy()
            inner_points = inner[i].copy()
            line_points = lines[i].copy()

            # draw initial_points on initial sequence
            seq_init[i] = draw(seq_init[i], (bound[i], inner[i], lines[i]), (len1, len2, len3, len4),
                               (255, 255, 255))

        os.makedirs(args.save_drawing_path, exist_ok=True)
        io.imsave(args.save_drawing_path + f'/init_' + seq_name.split('/')[-1],
                  seq_init)
