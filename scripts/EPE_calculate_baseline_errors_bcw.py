# this script calculates initial and elastic method errors
import numpy as np
from scipy import io as spio
from skimage import io
from glob import glob
import pickle
import sys
import os

from utils.dots_remap_backward import remap_centroids_all_backward
from utils.points_error_calculation import (compute_l2_error_sequence,
                                            compute_frechet_error_sequence)

if __name__ == "__main__":
    if sys.argv[1][-3:] == 'tif':
        sequences = [sys.argv[1]]
    else:
        sequences = glob(sys.argv[1] + '/*.tif')
        
    os.makedirs(sys.argv[2], exist_ok=True)
    # going through sequences
    for seq_name in filter(lambda name: name.find('Seq') != -1, sequences):
        print(seq_name)
        images = io.imread(seq_name).astype(np.float32)

        # load annotated points from initial unregistered sequence
        point_name = seq_name.split('.tif')[0] + '.mat'
        poi = spio.loadmat(point_name)

        # load deformations from base elastic method
        subf = ('/').join(seq_name.split('/')[:-2]) + '/elastic_deformations/numpy/'
        print(subf)
        baseline_def_name = subf + seq_name.split('/')[-1].split('.tif')[0] + f'_bcw.npy'
        base_deformation = np.load(baseline_def_name)

        in_sh = images.shape

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

        reg_bound = remap_centroids_all_backward(bound, base_deformation)
        reg_inner = remap_centroids_all_backward(inner, base_deformation)
        reg_lines = remap_centroids_all_backward(lines, base_deformation)

        base_bound_err = compute_l2_error_sequence(reg_bound)
        base_inner_err = compute_l2_error_sequence(reg_inner)
        base_line_err = compute_frechet_error_sequence(reg_lines, [len1, len2, len3, len4])

        # save calculated results
        elast_data = {'bound': base_bound_err.tolist(), 'inner': base_inner_err.tolist(),
                      'lines': base_line_err.tolist()}

        with open(sys.argv[2] + f'/elastic_method_bcw_' +
                  seq_name.split('/')[-1].split('.')[0], 'wb') as wrt_f:
            pickle.dump(elast_data, wrt_f)

        print(np.mean(base_bound_err))
        print(np.mean(base_inner_err))
        print(np.mean(base_line_err))