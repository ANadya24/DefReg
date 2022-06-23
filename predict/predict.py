import voxelmorph2d as vm2d
import voxelmorph3d as vm3d
import torch
import torchvision
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import skimage.io as io
import os
from skimage.transform import resize
import multiprocessing as mp
from tqdm import tqdm
import gc
import time
from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D
use_gpu = False#torch.cuda.is_available()
# devices = ['cpu', 'cuda']
# device = devices[use_gpu]


def GetBatch(chunk, imsize):
    batch_size = len(chunk)
    im_sizes = []
    batch_fixed = torch.empty((batch_size,) + imsize)
    batch_moving = torch.empty((batch_size,) + imsize)
    to_tensor = torchvision.transforms.ToTensor()
    for i, it in enumerate(chunk):
        fixed = io.imread(it + '_1.jpg')
        im_sizes.append(fixed.shape)
        # print(to_tensor(resize(fixed, imsize[1:])).shape)
        batch_fixed[i] = to_tensor(resize(fixed, imsize[1:]))
        moving = io.imread(it + '_2.jpg')
        batch_moving[i] = to_tensor(resize(moving, imsize[1:]))
    return batch_fixed, batch_moving, im_sizes


def save_images(batch_fixed, registered_image, im_sizes, chunk, path):
    if not os.path.exists(path):
        os.mkdir(path)
    for i in tqdm(range(len(batch_fixed))):
        image_fixed = batch_fixed[i].permute(1, 2, 0).detach().cpu().numpy().squeeze()
        image_fixed = resize(image_fixed, im_sizes[i])
        name = chunk[i].split('/')[-1]
        io.imsave(path + f'{name}_1.jpg', np.uint8(image_fixed*255.))
        image_moving = registered_image[i].permute(1, 2, 0).detach().cpu().numpy().squeeze()
        image_moving = resize(image_moving, im_sizes[i])
        io.imsave(path + f'{name}_2.jpg', np.uint8(image_moving*255.))


def predict(model_path, image_path, im_size, batch_size, save_path, is_2d=True):

    if is_2d:
        vm = vm2d
        voxelmorph = vm2d.VoxelMorph2d(im_size[0] * 2)
    else:
        vm = vm3d
        voxelmorph = vm3d.VoxelMorph3d(im_size[0] * 2)

    voxelmorph.load_state_dict(torch.load(model_path))
    voxelmorph.eval()

    print("Voxelmorph loaded successfully!")

    filename = list(set([x.split('_')[0]
                         for x in glob(image_path + '/*.jpg')]))
    print(len(filename))

    chunks = [filename[i:i + batch_size] for i in range(len(filename) // batch_size)]
    print(len(chunks))
    for chunk in tqdm(chunks):
        batch_fixed, batch_moving, imsizes = GetBatch(chunk, im_size)
        # print(batch_fixed.shape)
        if use_gpu:
            batch_fixed = batch_fixed.cuda()
            batch_moving = batch_moving.cuda()
        registered_image = voxelmorph(batch_moving, batch_fixed)
        # print(registered_image.max(), registered_image.min())
        save_images(batch_fixed, registered_image, imsizes, chunk, save_path)


if __name__ == "__main__":
    predict('/home/nadya/VoxelMorph/saved_models_test/vm_200', '/home/nadya/VoxelMorph/dataset/imgs/', (1, 256, 320),
            5, '/home/nadya/VoxelMorph/dataset/registered/')
