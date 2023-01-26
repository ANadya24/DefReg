import numpy as np
import skimage.io as io
from tqdm import tqdm
import torch

from functions import (
    preprocess_image_pair,
    apply_theta_and_deformation2image,
    resize_deformation,
    save_asis
)
from basic_nets.spatial_transform import SpatialTransformation
from inference_config import InferenceConfig
from src.config import load_yaml
from utils.dots_remap_backward import remap_centroids_all_backward
from utils.ffRemap import dots_remap_bcw, ff_1_to_k
from utils.points_error_calculation import (
    compute_l2_error_sequence,
    compute_frechet_error_sequence,
    load_points,
    draw
)
from utils.affine_transform import dots_affine_transform


def iterative_neigbours_predict(iterator_file, file, model, config, prev_use_elastic, elast_data={},
                                save_folder='image_deformed'):
    seq = io.imread(file)

    point_name = file.replace('elastic_sequences', 'SeqB').split('.tif')[0] + '.mat'
    data = load_points(point_name)
    if prev_use_elastic:
        bound = elast_data['bound']
        inner = elast_data['inner']
        lines = elast_data['lines']
    else:
        bound = data['bound']
        inner = data['inner']
        lines = data['lines']
    line_lengths = data['line_length']

    if config.use_masks:
        assert config.mask_sequences is not None
        mask_seq = io.imread(config.mask_sequences[iterator_file])
        if mask_seq.shape[-1] == 3:
            mask_seq = mask_seq.sum(-1)
        mask_seq = 1. - np.clip(np.array(mask_seq, dtype=np.float32), 0., 1.)

    reg_bound = bound.copy()
    reg_inner = inner.copy()
    reg_lines = lines.copy()

    new_seq = seq.copy()
    new_mask_seq = mask_seq.copy()

    # for i in range(len(new_seq)):
    #     new_seq[i] = draw(new_seq[i],
    #                        [reg_bound[i], reg_inner[i]*0, reg_lines[i]*0], line_lengths, (255, 255, 255))
    #     seq[i] = draw(seq[i],
    #                        [reg_bound[i], reg_inner[i]*0, reg_lines[i]*0], line_lengths, (255, 255, 255))

    for i in tqdm(range(len(seq) - 2, -1, -1)):
        for j in range(len(seq) - 1, i, -1):
            if config.use_masks:
                fixed, moving = preprocess_image_pair(seq[i], new_seq[j], config,
                                                      mask_seq[i], new_mask_seq[j])
            else:
                fixed, moving = preprocess_image_pair(seq[i], new_seq[j], config)

            fixed = fixed.to(config.device)
            moving = moving.to(config.device)
            model_output = model(moving, fixed)

            # registered = model_output['batch_registered'].detach().cpu().numpy()
            # fixed = fixed.detach().cpu().numpy()
            deformation = model_output['batch_deformation'].permute(0, 2, 3, 1).detach().cpu().numpy()

            # moving = moving.detach().cpu().numpy().squeeze()

            # show(moving, fixed.squeeze(),
            #      registered.squeeze())
            if 'theta' in model_output:
                theta = model_output['theta'].detach().cpu()
            else:
                theta = torch.zeros((1, 2, 3))
            new_seq[j] = apply_theta_and_deformation2image(new_seq[j], theta,
                                                           deformation[0])[0]
            h, w = new_seq[j].shape
            numpy_deformation = resize_deformation(deformation[0], h, w)
            theta = theta.numpy().squeeze()
            if model.use_theta:
                reg_bound[j] = dots_affine_transform(reg_bound[j], theta, h=h,
                                                     w=w)

                reg_inner[j] = dots_affine_transform(reg_inner[j], theta, h=h,
                                                     w=w)

                reg_lines[j] = dots_affine_transform(reg_lines[j], theta, h=h,
                                                 w=w)

            reg_bound[j] = dots_remap_bcw(reg_bound[j], numpy_deformation)

            reg_inner[j] = dots_remap_bcw(reg_inner[j], numpy_deformation)

            reg_lines[j] = dots_remap_bcw(reg_lines[j], numpy_deformation)

            if config.use_masks:
                new_mask_seq[j] = apply_theta_and_deformation2image(new_mask_seq[j], theta,
                                                                    deformation[0])[0]

    errB = np.mean(compute_l2_error_sequence(reg_bound))
    errI = np.mean(compute_l2_error_sequence(reg_inner))
    errL = np.mean(compute_frechet_error_sequence(reg_lines, line_lengths))
    print(errB)
    print(errI)
    print(errL)

    new_seq2 = new_seq.copy()
    for i in range(len(new_seq)):
        new_seq2[i] = draw(new_seq2[i],
                           [reg_bound[i], reg_inner[i] * 0, reg_lines[i] * 0], line_lengths, (255, 0, 0))

    save_asis(new_seq, config.save_path + f'/{save_folder}/', file.split('/')[-1])
    save_asis(new_seq2, config.save_path + f'/{save_folder}/', 'viz_' + file.split('/')[-1])
    return errB, errI, errL


def collect_deformations_frame(iterator_file, file, model, config, num_frame=0):
    model.use_theta = False

    seq = io.imread(file)
    defs = np.zeros((1,) + seq[0].shape + (2,))

    h, w = seq[0].shape

    if config.use_masks:
        assert config.mask_sequences is not None
        mask_seq = io.imread(config.mask_sequences[iterator_file])
        if mask_seq.shape[-1] == 3:
            mask_seq = mask_seq.sum(-1)
        mask_seq = 1. - np.clip(np.array(mask_seq, dtype=np.float32), 0., 1.)

    for i in tqdm(range(1, len(seq))):
        if num_frame != 0:
            frame = i-1
        else:
            frame = 0

        if config.use_masks:
            fixed, moving = preprocess_image_pair(seq[frame], seq[i], config,
                                                  mask_seq[frame], mask_seq[i])
        else:
            fixed, moving = preprocess_image_pair(seq[frame], seq[i], config)

        fixed = fixed.to(config.device)
        moving = moving.to(config.device)
        model_output = model(moving, fixed)
        deformation = model_output['batch_deformation'].permute(0, 2, 3, 1).detach().cpu().numpy()
        deformation = resize_deformation(deformation[0], h, w)[None]
        defs = np.concatenate([defs, deformation], axis=0)

    return defs


def apply_2nd_step_defs(data, defs, summarize=False):
    reg_bound = data['bound']
    reg_inner = data['inner']
    reg_lines = data['lines']

    deform = None
    reg2_bound = reg_bound.copy()
    reg2_inner = reg_inner.copy()
    reg2_lines = reg_lines.copy()

    for i in range(1, len(defs)):
        cur_def = defs[i]
        if summarize:
            if deform is not None:
                deform = ff_1_to_k(cur_def, deform)
            else:
                deform = cur_def
        else:
            deform = cur_def

        reg2_bound[i] = dots_remap_bcw(reg_bound[i].copy(), deform.copy())
        reg2_inner[i] = dots_remap_bcw(reg_inner[i].copy(), deform.copy())
        reg2_lines[i] = dots_remap_bcw(reg_lines[i].copy(), deform.copy())

    errB = np.mean(compute_l2_error_sequence(reg2_bound))
    errI = np.mean(compute_l2_error_sequence(reg2_inner))
    errL = np.mean(compute_frechet_error_sequence(reg2_lines, data['line_length']))
    print(errB)
    print(errI)
    print(errL)
    return errB, errI, errL
