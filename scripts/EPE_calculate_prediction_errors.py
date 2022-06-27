# this script calculates initial and elastic method errors
import numpy as np
from scipy import io as spio
from skimage import io
from glob import glob
import pickle
import argparse

from utils.ffRemap import dots_remap_bcw, ff_1_to_k
from utils.points_error_calculation import frechetDist


def parse_args():
    parser = argparse.ArgumentParser(description='EPE script for drawing')
    parser.add_argument('--prefix', type=str, default='fwd',
                        help='prefix of deformation type')
    parser.add_argument('--sequence_path', type=str, default='fwd',
                        help='path, where tif image sequneces are stored')
    parser.add_argument('--seq_name_pattern', type=str, default='Seq',
                        help='path, where tif image sequneces are stored')
    parser.add_argument('--save_pickle_path', type=str,
                        help='path to save calculated errors')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    sequences = glob(args.sequence_path + '/*.tif')
    for seq_name in filter(lambda name: name.find(args.seq_name_pattern) != -1, sequences):
        print(seq_name)
        images = io.imread(seq_name).astype(np.float32)

        # load annotated points from initial unregistered sequence
        point_name = seq_name.split('.tif')[0] + '.mat'
        poi = spio.loadmat(point_name)

        # load deformations from proposed nn method
        # subf = ('/').join(seq_name.split('/')[:-1]) + f'/registered/result/{model_name}/deformations/'
        def_name = args.prediction_path + '/deformations/' + \
                   seq_name.split('/')[-1].split('.tif')[0] + '.npy'
        proposed_deformations = np.load(def_name)

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

        fbound = bound[0][None]
        finner = inner[0][None]
        fline = lines[0]
        fline = np.stack([fline] * in_sh[0])

        bound_err = np.zeros(len(images))
        inner_err = np.zeros(len(images))
        line_err = np.zeros(len(images))

        b1 = np.zeros(len(images))
        b2 = np.zeros(len(images))
        b3 = np.zeros(len(images))
        b4 = np.zeros(len(images))

        proposed_init_def = None
        for i in range(1, len(images)):
            bound_points = bound[i]
            inner_points = inner[i]
            line_points = lines[i]

            proposed_deformation = proposed_deformations[i]
            if i != 1:
                proposed_deformation = ff_1_to_k(proposed_init_def, proposed_deformation)
            proposed_init_def = proposed_deformation.copy()
            proposed_def_bound_points = dots_remap_bcw(bound_points.copy(), proposed_deformation.copy())
            proposed_def_inner_points = dots_remap_bcw(inner_points.copy(), proposed_deformation.copy())
            proposed_def_lines = dots_remap_bcw(line_points.copy(), proposed_deformation.copy())

            bound_err[i] = ((((proposed_def_bound_points - fbound.squeeze()) ** 2).sum(axis=1)
                             ) ** 0.5).sum(axis=0) / float(fbound.shape[1])

            inner_err[i] = ((((proposed_def_inner_points - finner.squeeze()) ** 2).sum(axis=1)
                             ) ** 0.5).sum(axis=0) / float(finner.shape[1])

            b1 = frechetDist(proposed_def_lines[:len1], fline[0, :len1])
            b2 = frechetDist(proposed_def_lines[len1:len1 + len2], fline[0, len1:len1 + len2])
            b3 = frechetDist(proposed_def_lines[len1 + len2:len1 + len2 + len3],
                             fline[0, len1 + len2:len1 + len2 + len3])
            b4 = frechetDist(proposed_def_lines[len1 + len2 + len3:], fline[0, len1 + len2 + len3:])
            line_err[i] = (b1 + b2 + b3 + b4) / 4.

        # save calculated results
        errors_data = {'bound': bound_err.tolist(), 'inner': inner_err.tolist(),
                       'lines': line_err.tolist()}

        with open(args.save_pickle_path +
                  seq_name.split('/')[-1].split('.')[0], 'wb') as wrt_f:
            pickle.dump(errors_data, wrt_f)

        print(np.mean(bound_err))
        print(np.mean(inner_err))
        print(np.mean(line_err))
