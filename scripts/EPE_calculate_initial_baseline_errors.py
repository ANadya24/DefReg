# this script calculates initial and elastic method errors
import numpy as np
from scipy import io as spio
from skimage import io
from glob import glob
import pickle
import sys
import os

from utils.ffRemap import dots_remap_bcw, ff_1_to_k
from utils.points_error_calculation import frechetDist

if __name__ == "__main__":
    prefix = 'fwd'
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
        baseline_def_name = subf + seq_name.split('/')[-1].split('.tif')[0] + f'_{prefix}.npy'
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

        # annotated points and lines from first sequence frame
        fbound = bound[0][None]
        finner = inner[0][None]
        fline = lines[0]
        fline = np.stack([fline] * in_sh[0])

        # calculate freshet distance for each of 4 lines for the whole  unregistered sequence
        b1 = np.zeros(len(images))
        b2 = np.zeros(len(images))
        b3 = np.zeros(len(images))
        b4 = np.zeros(len(images))

        for frame in range(len(images)):
            b1[frame] = frechetDist(lines[frame, :len1], fline[frame, :len1])
            b2[frame] = frechetDist(lines[frame, len1:len1 + len2], fline[frame, len1:len1 + len2])
            b3[frame] = frechetDist(lines[frame, len1 + len2:len1 + len2 + len3],
                                    fline[frame, len1 + len2:len1 + len2 + len3])
            b4[frame] = frechetDist(lines[frame, len1 + len2 + len3:], fline[frame, len1 + len2 + len3:])

        line_init_err = (b1 + b2 + b3 + b4) / 4.  # float(fline.shape[1])

        # calculate l2 distance for bound points between i_th and first frame
        # for the whole unregistered sequence
        init_err = ((((bound - fbound) ** 2).sum(axis=2)) ** 0.5).sum(axis=1) / float(fbound.shape[1])
        # calculate l2 distance for inner points between i_th and first frame
        # for the whole unregistered sequence
        initin_err = ((((inner - finner) ** 2).sum(axis=2)) ** 0.5).sum(axis=1) / float(finner.shape[1])

        # save calculated errors for unregistered sequence
        initial_data = {'bound': init_err.tolist(), 'inner': initin_err.tolist(),
                        'lines': line_init_err.tolist()}

        with open(sys.argv[2] + '/unregistered_' +
                  seq_name.split('/')[-1].split('.')[0], 'wb') as wrt_f:
            pickle.dump(initial_data, wrt_f)

        # now let's calculate errors for the elastic method
        init_def = None

        base_bound_err = np.zeros(len(images))
        base_inner_err = np.zeros(len(images))
        base_line_err = np.zeros(len(images))

        for i in range(1, len(images)):
            x = images[i]
            b_p = bound[i]
            in_p = inner[i]
            line_p = lines[i]

            base_v = base_deformation[i]
            if i != 1:
                base_v = ff_1_to_k(init_def, base_v)
            init_def = base_v.copy()

            # calculate points position using deformation
            base_b_p = dots_remap_bcw(b_p.copy(), base_v.copy())
            base_in_p = dots_remap_bcw(in_p.copy(), base_v.copy())
            base_line_p = dots_remap_bcw(line_p.copy(), base_v.copy())

            # calculate corresponding errors
            base_bound_err[i] = ((((base_b_p - fbound.squeeze()) ** 2).sum(axis=1)
                                  ) ** 0.5).sum(axis=0) / float(fbound.shape[1])

            base_inner_err[i] = ((((base_in_p - finner.squeeze()) ** 2).sum(axis=1)
                                  ) ** 0.5).sum(axis=0) / float(finner.shape[1])

            b1 = frechetDist(base_line_p[:len1], fline[0, :len1])
            b2 = frechetDist(base_line_p[len1:len1 + len2], fline[0, len1:len1 + len2])
            b3 = frechetDist(base_line_p[len1 + len2:len1 + len2 + len3],
                             fline[0, len1 + len2:len1 + len2 + len3])
            b4 = frechetDist(base_line_p[len1 + len2 + len3:], fline[0, len1 + len2 + len3:])
            base_line_err[i] = (b1 + b2 + b3 + b4) / 4.

        # save calculated results
        elast_data = {'bound': base_bound_err.tolist(), 'inner': base_inner_err.tolist(),
                      'lines': base_line_err.tolist()}

        with open(sys.argv[2] + f'/elastic_method_{prefix}_' +
                  seq_name.split('/')[-1].split('.')[0], 'wb') as wrt_f:
            pickle.dump(elast_data, wrt_f)

        print(np.mean(init_err), np.mean(base_bound_err))
        print(np.mean(initin_err), np.mean(base_inner_err))
        print(np.mean(line_init_err), np.mean(base_line_err))