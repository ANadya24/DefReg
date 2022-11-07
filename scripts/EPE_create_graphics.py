# this script creates graphics for calculated errors
import numpy as np
from glob import glob
import pickle
import argparse
import os

from utils.points_error_calculation import plot_bef_aft

def parse_args():
    parser = argparse.ArgumentParser(description='EPE script for drawing')
    parser.add_argument('--baseline_pickle_path', type=str,
                        help='path to calculated errors for baseline registered sequence')
    parser.add_argument('--initial_error_pickle_path', type=str,
                        help='path to calculated errors for unregistered sequence')
    parser.add_argument('--proposed_pickle_path', type=str,
                        help='path to calculated errors for proposed method registered sequence')
    parser.add_argument('--sequence_path', type=str, default='fwd',
                        help='whole sequence path or path, where tif image sequneces are stored')
    parser.add_argument('--seq_name_pattern', type=str, default='Seq',
                        help='path, where tif image sequneces are stored')
    parser.add_argument('--save_graphics_path', type=str,
                        help='path to calculated errors for proposed method registered sequence')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    if args.sequence_path[-3:] == 'tif':
        sequences = [args.sequence_path]
    else:
        sequences = glob(args.sequence_path + '/*.tif')
        
    for seq_name in filter(lambda name: name.find(args.seq_name_pattern) != -1, sequences):
        print(seq_name)
        with open(args.baseline_pickle_path + seq_name.split('/')[-1].split('.')[0], 'rb') as rd_f:
            elast_data = pickle.load(rd_f)
        with open(args.initial_error_pickle_path + seq_name.split('/')[-1].split('.')[0], 'rb') as rd_f:
            initial_data = pickle.load(rd_f)
        with open(args.proposed_pickle_path + seq_name.split('/')[-1].split('.')[0], 'rb') as rd_f:
            proposed_data = pickle.load(rd_f)

        print('Bound points error: init | baseline | proposed: ',
              [np.mean(data['bound']) for data in [initial_data, elast_data, proposed_data]])
        print('Inner points error: init | baseline | proposed: ',
              [np.mean(data['inner']) for data in [initial_data, elast_data, proposed_data]])
        print('Line points error: init | baseline | proposed: ',
              [np.mean(data['lines']) for data in [initial_data, elast_data, proposed_data]])

        os.makedirs(args.save_graphics_path, exist_ok=True)
        plot_bef_aft(initial_data['bound'], proposed_data['bound'],
                     elast_data['bound'], 'Bound points', 'Time', 'Error',
                     args.save_graphics_path + seq_name.split('/')[-1].split('.tif')[0] +
                     '_bound_points_error.jpg')
        plot_bef_aft(initial_data['inner'], proposed_data['inner'],
                     elast_data['inner'], 'Inner points', 'Time', 'Error',
                     args.save_graphics_path + seq_name.split('/')[-1].split('.tif')[0] +
                     '_inner_points_error.jpg')
        plot_bef_aft(initial_data['lines'], proposed_data['lines'],
                     elast_data['lines'], 'Lines', 'Time', 'Error',
                     args.save_graphics_path + seq_name.split('/')[-1].split('.tif')[0] +
                     '_lines_error.jpg')
