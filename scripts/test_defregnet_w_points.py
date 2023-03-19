import sys

sys.path.append('/srv/fast1/n.anoshina/DefReg/')
from typing import cast
import torch
import pickle
import numpy as np
import pandas as pd
import cv2

from models.defregnet_model import DefRegNet

from inference_config import InferenceConfig
from src.config import load_yaml
from inference_sequence import iterative_neigbours_predict, \
    collect_deformations_frame, apply_2nd_step_defs
from utils.points_error_calculation import (
    compute_l2_error_sequence,
    compute_frechet_error_sequence,
    load_points
)

if __name__ == '__main__':
    config = cast(InferenceConfig, load_yaml(InferenceConfig,
                                             './inference_config.yaml'))
    elast_config = cast(InferenceConfig, load_yaml(InferenceConfig,
                                                   './inference_config_elastic.yaml'))
    model = DefRegNet(2, image_size=config.im_size[1], device=config.device)  # , use_theta=False)
    model_dict = torch.load(config.model_path, map_location=config.device)
    model.load_state_dict(model_dict['model_state_dict'])
    model.to(config.device)
    
    exp_name = config.model_path.split('/')[-2]

    print("Model loaded successfully!")

    df = pd.DataFrame(index=['Unregistered', 'Contour', 'Denoisereg', 'Denoisereg+VM', 'Proposed'],
                      columns=['Bound_SeqB1', 'Bound_SeqB2', 'Bound_SeqB3', 'Bound_SeqB4',
                               'Inner_SeqB1', 'Inner_SeqB2', 'Inner_SeqB3', 'Inner_SeqB4',
                               'Lines_SeqB1', 'Lines_SeqB2', 'Lines_SeqB3', 'Lines_SeqB4'])

    for iterator_file, filename in enumerate(config.image_sequences):
        seq_name = filename.split("/")[-1].split(".")[0]
        print('#' * 20)
        print(seq_name)
        with open(f'../data/point_dicts/elast_point_dict_{seq_name}.pkl', 'rb') as file:
            elast_data = pickle.load(file)

        with open(f'../data/point_dicts/init_point_dict_{seq_name}.pkl', 'rb') as file:
            init_data = pickle.load(file)
        elast_data['line_length'] = init_data['line_length']

        print('initial_error')
        errB = np.mean(compute_l2_error_sequence(init_data['bound']))
        errI = np.mean(compute_l2_error_sequence(init_data['inner']))
        errL = np.mean(compute_frechet_error_sequence(init_data['lines'], init_data['line_length']))
        print(errB)
        print(errI)
        print(errL)
        df.at['Unregistered', f'Bound_{seq_name}'] = errB
        df.at['Unregistered', f'Inner_{seq_name}'] = errI
        df.at['Unregistered', f'Lines_{seq_name}'] = errL

        print('elastic_error')
        errB = np.mean(compute_l2_error_sequence(elast_data['bound']))
        errI = np.mean(compute_l2_error_sequence(elast_data['inner']))
        errL = np.mean(compute_frechet_error_sequence(elast_data['lines'], elast_data['line_length']))
        print(errB)
        print(errI)
        print(errL)
        df.at['Contour', f'Bound_{seq_name}'] = errB
        df.at['Contour', f'Inner_{seq_name}'] = errI
        df.at['Contour', f'Lines_{seq_name}'] = errL

#         print('iterative_neigbours_predict')
#         model.use_theta = True
#         prev_use_elastic = False
#         errB, errI, errL = iterative_neigbours_predict(iterator_file=iterator_file,
#                                     file=filename, model=model,
#                                     config=config, prev_use_elastic=prev_use_elastic,
#                                     elast_data=elast_data, save_folder='initial_image_deformed')
        
#         print('elastic_iterative_neigbours_predict')
#         model.use_theta = True
#         prev_use_elastic = True
#         errB, errI, errL = iterative_neigbours_predict(iterator_file=iterator_file,
#                                     file=elast_config.image_sequences[iterator_file], model=model,
#                                     config=elast_config, prev_use_elastic=prev_use_elastic,
#                                     elast_data=elast_data, save_folder='elastic_image_deformed')

        print('elastic+frame0 deformations')
        model.use_theta = False
        defs = collect_deformations_frame(iterator_file=iterator_file,
                                          file=elast_config.image_sequences[iterator_file], model=model,
                                          config=elast_config, num_frame=0)
        errB, errI, errL = apply_2nd_step_defs(elast_data, defs, summarize=False)
        df.at['Proposed', f'Bound_{seq_name}'] = errB
        df.at['Proposed', f'Inner_{seq_name}'] = errI
        df.at['Proposed', f'Lines_{seq_name}'] = errL

        # print('elastic+frame_i deformations')
        # model.use_theta = False
        # defs = collect_deformations_frame(iterator_file=iterator_file,
        #                                   file=elast_config.image_sequences[iterator_file], model=model,
        #                                   config=elast_config, num_frame=1)
        # errB, errI, errL = apply_2nd_step_defs(elast_data, defs, summarize=True)

    with pd.ExcelWriter(f'DefRegResults_{exp_name}.xlsx') as writer:
        df.to_excel(writer, sheet_name=f'Sheet0', float_format="%.2f")
