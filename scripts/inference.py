from typing import cast
import os
import numpy as np
import argparse
import skimage.io as io
from skimage import color
from skimage.transform import resize
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import albumentations as A

from scripts.inference_config import InferenceConfig
from src.config import load_yaml
from models.defregnet_model import DefRegNet
from basic_nets.spatial_transform import SpatialTransformation
from utils.data_process import (
    pad_image, match_histograms,
    normalize_min_max,
    normalize_mean_std
)
from utils.ffRemap import ff_1_to_k


def preprocess_image_pair(image1, image2, config: InferenceConfig, mask1=None, mask2=None):
    to_tensor = ToTensor()
    resizer = A.Resize(*config.im_size[1:])

    if image1.shape[-1] == 3:
        image1 = color.rgb2gray(image1)

    if image2.shape[-1] == 3:
        image1 = color.rgb2gray(image1)

    if config.normalization == 'min_max':
        image1 = normalize_min_max(image1)
        image2 = normalize_min_max(image2)
    else:
        image1 = normalize_mean_std(image1)
        image2 = normalize_mean_std(image2)

    image1, image2, _ = match_histograms(image1, image2, random_switch=False)
    h, w = image1.shape

    if h != w:
        if h < w:
            pad_params = (0, w - h, 0, 0)
        else:
            pad_params = (0, 0, 0, h - w)
        image1 = pad_image(image1, pad_params)
        image2 = pad_image(image2, pad_params)
        if config.use_masks:
            mask1 = pad_image(mask1, pad_params)
            mask2 = pad_image(mask2, pad_params)

    data1 = {'image': image1}
    if config.use_masks:
        data1['mask'] = mask1
    data1 = resizer(**data1)
    image1 = data1['image']
    if config.use_masks:
        mask1 = data1['mask']

    data2 = {'image': image2}
    if config.use_masks:
        data2['mask'] = mask2
    data2 = resizer(**data2)
    image2 = data2['image']
    if config.use_masks:
        mask2 = data2['mask']

    image1 = to_tensor(image1).float()
    image2 = to_tensor(image2).float()

    if config.use_masks:
        if config.multiply_mask:
            image1 = image1 * torch.Tensor(mask1).float()[None]
            image2 = image2 * torch.Tensor(mask2).float()[None]
        else:
            image1 = torch.cat([image1, torch.Tensor(mask1).float()[None]], 0)
            image2 = torch.cat([image2, torch.Tensor(mask2).float()[None]], 0)
    return image1[None], image2[None]


def save256(new_seq, path, name):
    os.makedirs(path, exist_ok=True)
    io.imsave(path + name, new_seq)


def apply_theta_and_deformation2image(image, theta, deformation):
    SP = SpatialTransformation()

    if image.shape[-1] == 3:
        image = color.rgb2gray(image) * 255
        image = image.astype(np.uint8)

    h, w = image.shape[:2]
    dh, dw = deformation.shape[:2]
    if dh != h or dw != w:
        if h / w != 1.:
            if h < w:
                d = w - h
                tmp = resize(deformation, (w, w))
                deformation = tmp[:-d]
            else:
                d = h - w
                tmp = resize(deformation, (h, h))
                deformation = tmp[:, :-d]
        else:
            deformation = resize(deformation, (h, w))

    tensor_image = torch.tensor(image[None, None, :, :], dtype=torch.float)
    tensor_deformation = torch.tensor(deformation.transpose((2, 0, 1))[None],
                                      dtype=torch.float)

    if isinstance(theta, np.ndarray):
        theta = torch.tensor(theta[None], dtype=torch.float)

    grid = F.affine_grid(theta, tensor_image.size())
    tensor_image = F.grid_sample(tensor_image, grid)

    deformed_image = SP(tensor_image, tensor_deformation).squeeze()
    deformed_image = deformed_image.numpy().astype(image.dtype)
    return deformed_image, deformation


def save(seq, thetas, deformations, path, name):
    os.makedirs(path, exist_ok=True)
    os.makedirs(path + '/deformations/', exist_ok=True)
    os.makedirs(path + '/thetas/', exist_ok=True)
    new_seq = []
    new_def = []
    SP = SpatialTransformation()

    for i, im in enumerate(seq):
        if i == 0:
            if im.shape[-1] == 3:
                im = color.rgb2gray(im) * 255
                im = im.astype(np.uint8)
            new_seq.append(im)
            new_def.append(np.zeros(tuple(im.shape[:2]) + (2,)))
        else:
            deformed_image, deformation = apply_theta_and_deformation2image(im, thetas[i], deformations[i])
            new_def.append(deformation)
            new_seq.append(deformed_image)

    io.imsave(path + name, np.array(new_seq, dtype=np.uint8))
    np.save(path + '/deformations/' + name.split('.tif')[0], new_def)
    np.save(path + '/thetas/' + name.split('.tif')[0], thetas)


def show(moving, fixed, reg, name=''):
    from matplotlib import pyplot as plt
    mov = moving * 255
    fix = fixed * 255
    reg = reg * 255
    h, w = moving.shape
    im1 = np.zeros((h, w, 3), dtype=np.uint8)
    im2 = np.zeros_like(im1)
    im1[:, :, 0] = np.uint8(fix)
    im2[:, :, 0] = np.uint8(fix)
    im1[:, :, 1] = np.uint8(mov)
    im2[:, :, 1] = np.uint8(reg)
    im = np.concatenate([im1, im2], axis=1)
    print(im.shape)
    plt.figure()
    plt.imshow(im)
    plt.title(name)

    # cv2.imwrite('../old/test.jpg', im)
    input()


def predict(config: InferenceConfig):
    """
    Совмещаем соседние и так итеративно приводим все к первой картинке с конца.
    """

    model_dict = torch.load(config.model_path, map_location=config.device)
    if 'model' not in model_dict:
        model = DefRegNet(2, image_size=config.im_size[1],
                          device=config.device)
    else:
        model = model_dict['model']
    model.load_state_dict(model_dict['model_state_dict'])

    model.to(config.device)
    model.eval()

    print("Model loaded successfully!")

    for iterator_file, file in enumerate(config.image_sequences):
        seq = io.imread(file)
        defs = np.zeros((len(seq),) + config.im_size[1:] + (2,))
        thetas = np.stack([np.eye(2, 3)] * len(seq), axis=0)
        if config.use_masks:
            assert config.mask_sequences is not None
            mask_seq = io.imread(config.mask_sequences[iterator_file])
            if mask_seq.shape[-1] == 3:
                mask_seq = mask_seq.sum(-1)
            mask_seq = 1. - np.clip(np.array(mask_seq, dtype=np.float32), 0., 1.)

        SP = SpatialTransformation()

        new_seq = seq.copy()
        new_mask_seq = mask_seq.copy()
        # small_seq = np.zeros((len(seq),) + (256, 256))
        # small_mask_seq = np.zeros((len(seq),) + (256, 256))

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

                registered = model_output['batch_registered'].detach().cpu().numpy()
                fixed = fixed.detach().cpu().numpy()
                deformation = model_output['batch_deformation'].permute(0, 2, 3, 1).detach().cpu().numpy()

                moving = moving.detach().cpu().numpy().squeeze()

                # show(moving, fixed.squeeze(),
                #      registered.squeeze())
                new_seq[j] = apply_theta_and_deformation2image(new_seq[j], model_output['theta'].detach().cpu(),
                                                               deformation[0])[0]
                # prev_mask = new_mask_seq[j].copy()
                if config.use_masks:
                    new_mask_seq[j] = \
                    apply_theta_and_deformation2image(new_mask_seq[j], model_output['theta'].detach().cpu(),
                                                      deformation[0])[0]

                # show(prev_mask, mask_seq[i], new_mask_seq[j], '2registered')

                defs[j] = ff_1_to_k(defs[j], deformation[0])
                new_theta = np.concatenate([model_output['theta'].detach().cpu().numpy()[0], np.array([[0, 0, 1]])], 0)
                new_theta = np.concatenate([thetas[j],
                                            np.array([[0, 0, 1]])], 0) @ new_theta
                thetas[j] = new_theta[:2][None]

        save(seq, thetas, defs, config.save_path, file.split('/')[-1])
        save256(new_seq, config.save_path + '/image_deformed/', file.split('/')[-1])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = cast(InferenceConfig, load_yaml(InferenceConfig, args.config_file))
    predict(config)
