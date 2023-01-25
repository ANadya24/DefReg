from typing import cast
import torch
import pickle

from models.defregnet_model import DefRegNet

from inference_config import InferenceConfig
from src.config import load_yaml
from inference_sequence import iterative_neigbours_predict, \
    collect_deformations_frame, apply_2nd_step_defs


if __name__ == '__main__':
    config = cast(InferenceConfig, load_yaml(InferenceConfig,
                                             './DefReg/scripts/inference_config.yaml'))
    elast_config = cast(InferenceConfig, load_yaml(InferenceConfig,
                                                   './DefReg/scripts/inference_config_elastic.yaml'))
    model = DefRegNet(2, image_size=config.im_size[1], device=config.device)  # , use_theta=False)
    model_dict = torch.load(config.model_path, map_location=config.device)
    model.load_state_dict(model_dict['model_state_dict'])
    model.to(config.device)

    print("Model loaded successfully!")

    with open('elast_point_dict.pkl', 'rb') as file:
        elast_data = pickle.load(file)

    with open('init_point_dict.pkl', 'rb') as file:
        init_data = pickle.load(file)

    for iterator_file, file in enumerate(config.image_sequences):
        print('iterative_neigbours_predict')
        model.use_theta = True
        prev_use_elastic = False
        iterative_neigbours_predict(iterator_file=iterator_file,
                                    file=file, model=model,
                                    config=config, prev_use_elastic=prev_use_elastic,
                                    elast_data=elast_data, save_folder='initial_image_deformed')

        print('elastic_iterative_neigbours_predict')
        model.use_theta = True
        prev_use_elastic = True
        iterative_neigbours_predict(iterator_file=iterator_file,
                                    file=elast_config.image_sequences[iterator_file], model=model,
                                    config=elast_config, prev_use_elastic=prev_use_elastic,
                                    elast_data=elast_data, save_folder='elastic_image_deformed')

        print('elastic+frame0 deformations')
        model.use_theta = False
        defs = collect_deformations_frame(iterator_file=iterator_file,
                                          file=elast_config.image_sequences[iterator_file], model=model,
                                          config=elast_config, num_frame=0)
        apply_2nd_step_defs(elast_data, defs, summarize=False)

        print('elastic+frame_i deformations')
        model.use_theta = False
        defs = collect_deformations_frame(iterator_file=iterator_file,
                                          file=elast_config.image_sequences[iterator_file], model=model,
                                          config=elast_config, num_frame=1)
        apply_2nd_step_defs(elast_data, defs, summarize=True)




