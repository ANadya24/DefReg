import sys

sys.path.append('/srv/fast1/n.anoshina/DefReg/')
from typing import cast
import torch
import pickle
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path, PurePath

from models.defregnet_model import DefRegNet

from inference_config import InferenceConfig
from src.config import load_yaml
from inference_sequence import iterative_neigbours_predict, \
    collect_deformations_frame, apply_2nd_step_defs
from utils.points_error_calculation import (
    compute_l2_error_sequence,
    compute_frechet_error_sequence
)

if __name__ == '__main__':
    config = cast(InferenceConfig, load_yaml(InferenceConfig,
                                             './inference_config.yaml'))
    elast_config = cast(InferenceConfig, load_yaml(InferenceConfig,
                                                   './inference_config_elastic.yaml'))

    exps = glob('../logs/cont_based_exp*')
    exp_names = [exp.split('/')[-1] + '_p' for exp in exps]
    exp_names = [[item + '_p', item + '_l'] for item in exp_names]
    exp_names = [[subitem, subitem + '_after_cont'] for item in exp_names for subitem in item]
    exp_names = np.array(exp_names).reshape(-1).tolist()

    df = pd.DataFrame(index=['Unregistered', 'Contour'] + exp_names,
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

    model = DefRegNet(2, image_size=config.im_size[1], device=config.device)  # , use_theta=False)
    for exp in exps:
        exp_name = PurePath(exp)
        if exp_name.name.find('gauss') > 0:
            if exp_name.name.find('_7') > 0 or exp_name.name.find('_8') > 0:
                config.gauss_sigma = 2.1
            else:
                config.gauss_sigma = 1.3
        else:
            config.gauss_sigma = -1.
        for model_name, prefix in zip(['defregnet_loss_best', 'defregnet_points_error_l2'], ['_p', '_l']):
            model_dict = torch.load(Path(exp_name / model_name), map_location=config.device)
            model.load_state_dict(model_dict['model_state_dict'])
            model.to(config.device)
            table_name = exp_name.name + prefix

            print(f"Model from {PurePath(exp_name) / model_name} loaded successfully!")

            for iterator_file, filename in enumerate(config.image_sequences):
                seq_name = filename.split("/")[-1].split(".")[0]
                print('#' * 20)
                print(seq_name)
                with open(f'../data/point_dicts/elast_point_dict_{seq_name}.pkl', 'rb') as file:
                    elast_data = pickle.load(file)

                with open(f'../data/point_dicts/init_point_dict_{seq_name}.pkl', 'rb') as file:
                    init_data = pickle.load(file)
                elast_data['line_length'] = init_data['line_length']

                print('iterative_neigbours_predict')
                model.use_theta = True
                prev_use_elastic = False
                errB, errI, errL = iterative_neigbours_predict(iterator_file=iterator_file,
                                            file=filename, model=model,
                                            config=config, prev_use_elastic=prev_use_elastic,
                                            elast_data=elast_data, save_folder='initial_image_deformed')
                df.at[table_name, f'Bound_{seq_name}'] = errB
                df.at[table_name, f'Inner_{seq_name}'] = errI
                df.at[table_name, f'Lines_{seq_name}'] = errL

                print('elastic+frame0 deformations')
                model.use_theta = False
                defs = collect_deformations_frame(iterator_file=iterator_file,
                                                  file=elast_config.image_sequences[iterator_file], model=model,
                                                  config=elast_config, num_frame=0)
                errB, errI, errL = apply_2nd_step_defs(elast_data, defs, summarize=False)
                df.at[table_name + '_after_cont', f'Bound_{seq_name}'] = errB
                df.at[table_name + '_after_cont', f'Inner_{seq_name}'] = errI
                df.at[table_name + '_after_cont', f'Lines_{seq_name}'] = errL

    with pd.ExcelWriter(f'DefRegResults.xlsx') as writer:
        df.to_excel(writer, sheet_name=f'Sheet0', float_format="%.2f")
