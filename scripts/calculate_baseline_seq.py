# this script calculates initial and elastic method errors
import numpy as np
from scipy import io as spio
from skimage import io, color
from glob import glob
import pickle
import sys
import os
import argparse

from utils.ffRemap import forward_warp, backward_warp, ff_1_to_k

def parse_args():
    parser = argparse.ArgumentParser(description='EPE script for drawing')
    parser.add_argument('--prefix', type=str, default='fwd',
                        help='prefix of deformation type')
    parser.add_argument('--sequence_path', type=str,
                        help='whole sequence path or path, where tif image sequneces are stored')
    parser.add_argument('--seq_name_pattern', type=str, default='Seq',
                        help='path, where tif image sequneces are stored')
    parser.add_argument('--base_prediction_path', type=str, help='predicted by model deformations')
    parser.add_argument('--save_path', type=str,
                        help='path to save deformed sequences')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if args.sequence_path[-3:] == 'tif':
        sequences = [args.sequence_path]
    else:
        sequences = glob(args.sequence_path + '*.tif')
    for seq_name in filter(lambda name: name.find(args.seq_name_pattern) != -1, sequences):
        print(seq_name)
        images = io.imread(seq_name)
        if images.shape[-1] == 3:
            images = (color.rgb2gray(images) * 255).astype('uint8')

        # load deformations from base elastic method
        subf = args.base_prediction_path + '/numpy/'

        baseline_def_name = subf + seq_name.split('/')[-1].split('.tif')[0] + f'_{args.prefix}.npy'
        if not os.path.exists(baseline_def_name):
            baseline_def_name = baseline_def_name.replace('init_', '')
            if not os.path.exists(baseline_def_name):
                continue
        base_deformation = np.load(baseline_def_name)
        
        deformed_images = images.copy()
        init_def = None
        for i in range(1, len(images)):
            if args.prefix == 'fwd':
                if i != 1:
                    deform = ff_1_to_k(init_def, base_deformation[i])
                else:
                    deform = base_deformation[i]
                init_def = deform
                deformed_images[i] = forward_warp(images[i], deform)
            else:
                if i != 1:
                    deform = ff_1_to_k(init_def, base_deformation[i])
                else:
                    deform = base_deformation[i]
                init_def = deform
                deformed_images[i] = backward_warp(images[i], deform)
        io.imsave(args.save_path + '/' + seq_name.split('/')[-1].replace('init_', ''), 
                  deformed_images)