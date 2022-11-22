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
    parser.add_argument('--sequence_path', type=str, default='fwd',
                        help='whole sequence path or path, where tif image sequneces are stored')
    parser.add_argument('--seq_name_pattern', type=str, default='Seq',
                        help='path, where tif image sequneces are stored')
    parser.add_argument('--prediction_path', type=str,
                        help='path to predicted deformations and sequences')
    parser.add_argument('--base_prediction_path', type=str,
                        help='path to predicted deformations and sequences')
    parser.add_argument('--save_drawing_path', type=str,
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

        # load deformations from proposed nn method
        # subf = ('/').join(seq_name.split('/')[:-1]) + f'/registered/result/{model_name}/deformations/'
        def_name = args.prediction_path + '/deformations/' + \
                   seq_name.split('/')[-1].split('.tif')[0] + '.npy'
        proposed_deformations = np.load(def_name)

        # load deformations from base elastic method
        subf = ('/').join(seq_name.split('/')[:-2]) + '/elastic_deformations/numpy/'
        baseline_def_name = subf \
                            + seq_name.split('/')[-1].split('.tif')[0] + f'_{args.prefix}.npy'
        base_deformations = np.load(baseline_def_name)

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

        seq_name_elast = args.base_prediction_path + seq_name.split('/')[-1]
        seq_name_prop = args.prediction_path + seq_name.split('/')[-1]
        # initial unregistered sequence
        seq_init = io.imread(seq_name)
        seq_init = np.stack([seq_init, seq_init, seq_init], -1)
        seq_init[0] = draw(seq_init[0], (bound[0], inner[0], lines[0]),
                           (len1, len2, len3, len4), (255, 255, 255))

        # registered sequence by proposed method
        proposed_method_seq = io.imread(seq_name_prop)
        proposed_method_seq = np.stack([proposed_method_seq, proposed_method_seq, proposed_method_seq], -1)
        proposed_seq = [draw(proposed_method_seq[0], (bound[0], inner[0], lines[0]),
                             (len1, len2, len3, len4), (255, 0, 0))]

        # registered sequence by elastic baseline method
        base_method_seq = io.imread(seq_name_elast)
        base_method_seq = np.stack([base_method_seq, base_method_seq, base_method_seq], -1)
        base_seq = [draw(base_method_seq[0], (bound[0], inner[0], lines[0]), (len1, len2, len3, len4), (0, 0, 255))]

        in_sh = images.shape

        base_init_def = None
        proposed_init_def = None
        for i in range(1, len(images)):
            bound_points = bound[i].copy()
            inner_points = inner[i].copy()
            line_points = lines[i].copy()

            proposed_deformation = proposed_deformations[i].copy()
            base_deformation = base_deformations[i].copy()

            if i != 1:
                base_deformation = ff_1_to_k(base_init_def, base_deformation)
                proposed_deformation = ff_1_to_k(proposed_init_def, proposed_deformation)
            base_init_def = base_deformation.copy()
            proposed_init_def = proposed_deformation.copy()

            print(proposed_deformation.min(), proposed_deformation.max(),
                  base_deformation.min(), base_deformation.max())

            proposed_def_bound_points = dots_remap_bcw(bound_points.copy(), proposed_deformation.copy())
            proposed_def_inner_points = dots_remap_bcw(inner_points.copy(), proposed_deformation.copy())
            proposed_def_lines = dots_remap_bcw(line_points.copy(), proposed_deformation.copy())
            base_def_bound_points = dots_remap_bcw(bound_points.copy(), base_deformation.copy())
            base_def_inner_points = dots_remap_bcw(inner_points.copy(), base_deformation.copy())
            base_def_lines = dots_remap_bcw(line_points.copy(), base_deformation.copy())

            # assert proposed_method_seq is not None
            # assert proposed_seq is not None
            # assert seq_init is not None
            # assert base_method_seq is not None
            # assert base_seq is not None

            # draw deformed points on our predicted sequence
            seq_draw = draw(proposed_method_seq[i],
                            (proposed_def_bound_points,
                             proposed_def_inner_points, proposed_def_lines),
                            (len1, len2, len3, len4), (255, 0, 0))
            # draw initial points on our predicted sequence
            seq_draw = draw(seq_draw, (bound[0], inner[0], lines[0]),
                            (len1, len2, len3, len4), (0, 255, 255))
            proposed_seq.append(seq_draw)

            # draw deformed by base method points on predicted by base method sequence
            seq_draw = draw(base_method_seq[i],
                            (base_def_bound_points,
                             base_def_inner_points, base_def_lines),
                            (len1, len2, len3, len4), (0, 0, 255))
            # draw initial points on predicted by base method sequence
            seq_draw = draw(seq_draw, (bound[0], inner[0], lines[0]),
                            (len1, len2, len3, len4), (0, 255, 255))
            base_seq.append(seq_draw)

            # draw initial_points on initial sequence
            seq_init[i] = draw(seq_init[i], (bound[i], inner[i], lines[i]), (len1, len2, len3, len4),
                               (255, 255, 255))

        os.makedirs(args.save_drawing_path + f'/{args.prefix}', exist_ok=True)
        io.imsave(args.save_drawing_path + f'/{args.prefix}/init_' + seq_name.split('/')[-1],
                  seq_init)
        io.imsave(args.save_drawing_path + f'/{args.prefix}/prop_' + seq_name.split('/')[-1],
                  np.array(proposed_seq))
        io.imsave(args.save_drawing_path + f'/{args.prefix}/elastic_' + seq_name.split('/')[-1],
                  np.array(base_seq))
