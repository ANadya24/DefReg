# this script calculates initial and elastic method errors
import numpy as np
from scipy import io as spio
from skimage import io
from glob import glob
import pickle
import argparse
import os
import shutil

from inference import save


def parse_args():
    parser = argparse.ArgumentParser(description='EPE script for drawing')
    parser.add_argument('--prefix', type=str, default='fwd',
                        help='prefix of deformation type')
    parser.add_argument('--sequence_path', type=str, default='fwd',
                        help='whole sequence path or path, where tif image sequneces are stored')
    parser.add_argument('--seq_name_pattern', type=str, default='Seq',
                        help='path, where tif image sequneces are stored')
    parser.add_argument('--prediction_path', type=str, help='predicted by model deformations')
    parser.add_argument('--use_thetas', type=int, default=1, help='if positive than use affine matrix '
                                                                  'transform before deformation application')
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

        # load deformations from proposed nn method
        def_name = args.prediction_path + '/deformations/' + \
                   seq_name.split('/')[-1].split('.tif')[0] + '.npy'
        if not os.path.exists(def_name):
            def_name = def_name.replace('init_', '')
            if not os.path.exists(def_name):
                continue
        proposed_deformations = np.load(def_name)

        if args.use_thetas:
            theta_name = args.prediction_path + '/thetas/' + \
                         seq_name.split('/')[-1].split('.tif')[0] + '.npy'
            if not os.path.exists(theta_name):
                theta_name = theta_name.replace('init_', '')
                if not os.path.exists(theta_name):
                    continue
            proposed_thetas = np.load(theta_name)
        else:
            proposed_thetas = None
        
        save(images, proposed_thetas, proposed_deformations,
             args.save_path, seq_name.split('/')[-1].replace('init_', ''),
             use_theta=bool(args.use_thetas))

        shutil.rmtree(args.save_path + '/deformations')
        shutil.rmtree(args.save_path + '/thetas')