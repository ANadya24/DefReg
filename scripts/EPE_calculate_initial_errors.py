# this script calculates initial and elastic method errors
import numpy as np
from scipy import io as spio
from skimage import io
from glob import glob
import pickle
import sys
import os

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

        init_err = compute_l2_error_sequence(bound)
        initin_err = compute_l2_error_sequence(inner)
        line_init_err = compute_frechet_error_sequence(lines, [len1, len2, len3, len4])

        # save calculated errors for unregistered sequence
        initial_data = {'bound': init_err.tolist(), 'inner': initin_err.tolist(),
                        'lines': line_init_err.tolist()}

        with open(sys.argv[2] + '/unregistered_' +
                  seq_name.split('/')[-1].split('.')[0], 'wb') as wrt_f:
            pickle.dump(initial_data, wrt_f)

        print(np.mean(init_err))
        print(np.mean(initin_err))
        print(np.mean(line_init_err))