from matplotlib import pyplot as plt
import os
import numpy as np
import skimage.io as io
from skimage import color, filters
from skimage.transform import resize
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import albumentations as A

from scripts.inference_config import InferenceConfig
from basic_nets.spatial_transform import SpatialTransformation
from utils.data_process import (
    pad_image, match_histograms,
    normalize_min_max,
    normalize_mean_std
)


def preprocess_image_pair(image1, image2, config: InferenceConfig, mask1=None, mask2=None):
    to_tensor = ToTensor()
    resizer = A.Resize(*config.im_size[1:])

    if image1.shape[-1] == 3:
        image1 = color.rgb2gray(image1)

    if image2.shape[-1] == 3:
        image2 = color.rgb2gray(image2)
    # image1 = filters.gaussian(image1, 5.5)
    # image2 = filters.gaussian(image2, 5.5)

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
        
    if config.gauss_sigma > 0.:
        image1 = filters.gaussian(image1, config.gauss_sigma)
        image2 = filters.gaussian(image2, config.gauss_sigma)


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


def save_asis(new_seq, path, name):
    os.makedirs(path, exist_ok=True)
    io.imsave(path + name, new_seq)


def to_numpy(im):
    return im.detach().cpu().numpy().squeeze()


def resize_deformation(deformation, h, w):
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
    return deformation


def apply_theta_and_deformation2image(image, theta, deformation):
    SP = SpatialTransformation()
    
    if image.shape[-1] == 3:
        image = color.rgb2gray(image) * 255
        image = image.astype(np.uint8)
    
    h, w = image.shape[:2]
    if deformation.sum() != 0:
        deformation = resize_deformation(deformation, h, w)

        tensor_deformation = torch.tensor(deformation.transpose((2, 0, 1))[None],
                                          dtype=torch.float)
    tensor_image = torch.tensor(image[None, None, :, :], dtype=torch.float)
    
    if theta.sum() != 0:
        if isinstance(theta, np.ndarray):
            theta = torch.tensor(theta[None], dtype=torch.float)

        grid = F.affine_grid(theta, tensor_image.size())
        tensor_image = F.grid_sample(tensor_image, grid)

    if deformation.sum() != 0:
        deformed_image = SP(tensor_image, tensor_deformation).squeeze()
        deformed_image = to_numpy(deformed_image).astype(image.dtype)
        
        return deformed_image, deformation
    else:
        return to_numpy(tensor_image).astype(image.dtype), deformation


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
            new_def.append(np.zeros(tuple(im.shape[:2]) + (2, )))
        else:
            deformed_image, deformation = apply_theta_and_deformation2image(im, thetas[i], deformations[i])
            new_def.append(deformation)
            new_seq.append(deformed_image)

    io.imsave(path + name, np.array(new_seq, dtype=np.uint8))
    np.save(path + '/deformations/' + name.split('.tif')[0], new_def)
    np.save(path + '/thetas/' + name.split('.tif')[0], thetas)


def show(moving, fixed, reg, name='', pause_by_input=True):
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
    if pause_by_input:
        input()
        
        
def add_row(matrix, row):
    if isinstance(row, list):
        row = np.array(row).reshape(1, -1)
    
    return np.concatenate([matrix, row], axis=0)
