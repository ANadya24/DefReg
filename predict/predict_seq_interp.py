import voxelmorph2d as vm2d
import voxelmorph3d as vm3d
import torch
import torchvision
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import skimage.io as io
from skimage import color
import os
from skimage.transform import resize
from tqdm import tqdm
from ffRemap import *

use_gpu = True#torch.cuda.is_available()
devices = ['cpu', 'cuda']
device = devices[use_gpu]


def GetBatch(chunk, imsize):
    batch_size = len(chunk)-1
    im_sizes = []
    batch_fixed = torch.empty((batch_size,) + imsize)
    batch_moving = torch.empty((batch_size,) + imsize)
    to_tensor = torchvision.transforms.ToTensor()
    for i in range(batch_size):
        fixed = chunk[i]
        if fixed.shape[-1] == 3:
            fixed = color.rgb2gray(fixed)
        im_sizes.append(fixed.shape)
        h, w = fixed.shape[:2]
        if h / w != 1.:
            if h < w:
                d = w - h
                fixed = np.pad(fixed, ((d // 2, d // 2 + d % 2), (0, 0)), 'constant')
            else:
                d = h - w
                fixed = np.pad(fixed, ((0, 0), (d // 2, d // 2 + d % 2)), 'constant')
        batch_fixed[i] = to_tensor(resize(fixed, imsize[1:]))
        moving = chunk[i+1]
        if moving.shape[-1] == 3:
            moving = color.rgb2gray(moving)
        if h / w != 1.:
            if h < w:
                d = w - h
                moving = np.pad(moving, ((d // 2, d // 2 + d % 2), (0, 0)), 'constant')
            else:
                d = h - w
                moving = np.pad(moving, ((0, 0), (d // 2, d // 2 + d % 2)), 'constant')
        batch_moving[i] = to_tensor(resize(moving, imsize[1:]))
    return batch_fixed, batch_moving, im_sizes


def save256(new_seq, path, name):
    new_seq.astype('uint8')
    os.makedirs(path, exist_ok=True)
    io.imsave(path + name, new_seq)


def save(seq, deformations, path, name):
    os.makedirs(path, exist_ok=True)
    os.makedirs(path + '/deformations/', exist_ok=True)
    new_seq = []
    new_def = []

    for i, im in enumerate(seq):
        if im.shape[-1] == 3:
            im = color.rgb2gray(im)*255
            im = im.astype('uint8')
        if i == 0:
            new_seq.append(im)
            new_def.append(deformations[i])
        else:
            h, w = im.shape[:2]
            defxy = deformations[i]
            if h / w != 1.:
                if h < w:
                    d = w - h
                    tmp = resize(defxy, (w, w))
                    defxy = tmp[d // 2: -(d // 2 + d % 2)]
                else:
                    d = h - w
                    tmp = resize(defxy, (h, h))
                    defxy = tmp[:, d // 2: -(d // 2 + d % 2)]
            else:
                defxy = resize(defxy, (h, w))

            print(defxy.min(), defxy.max())
            new_def.append(defxy)
            from voxelmorph2d import SpatialTransformation
            SP = SpatialTransformation()
            # print(im.max(), im[None, :, :, None].shape, defxy.shape)
            im_new = SP(torch.tensor(im[None, :, :, None]/255), torch.tensor(defxy[None])).squeeze()
            im_new = np.uint8(im_new.numpy() * 255)
            # print(im_new.shape)
            # plt.imshow(im_new)
            # plt.pause(2)
            # im_new = forward_warp(im, defxy)
            # print(im_new.shape)
            new_seq.append(im_new)

    io.imsave(path + name, np.array(new_seq, dtype='uint8'))
    np.save(path + '/deformations/' + name.split('.tif')[0], new_def)


def predict(model_path, image_path, im_size, batch_size, save_path, is_2d=True):

    if is_2d:
        vm = vm2d
        voxelmorph = vm.VoxelMorph2d(im_size[0] * 2, use_gpu=use_gpu)
    else:
        vm = vm3d
        voxelmorph = vm.VoxelMorph3d(im_size[0] * 2, use_gpu=use_gpu)

    voxelmorph = torch.nn.DataParallel(voxelmorph)
    voxelmorph.load_state_dict(torch.load(model_path))
    voxelmorph.eval()

    print("Voxelmorph loaded successfully!")

    filenames = glob(image_path + '/Seq*1.tif')
    for file in filenames:
        defs = np.zeros((1, 256, 256, 2))
        seq = io.imread(file)[:15]
        # print(seq.shape)
        chunks = [seq[i:min(i + batch_size + 1, len(seq))] for i in range(0, len(seq)-1, batch_size)]
        print(len(chunks), len(chunks[0]), len(seq), file)
        first = True
        new_seq = []
        for chunk in tqdm(chunks):
            if first:
                batch_fixed, batch_moving, imsizes = GetBatch(chunk, im_size)
            else:
                # batch_fixed = registered
                _, batch_moving, imsizes = GetBatch(chunk, im_size)

            if use_gpu:
                batch_fixed = batch_fixed.cuda()
                batch_moving = batch_moving.cuda()

            registered, deformations = voxelmorph(batch_moving, batch_fixed)
            fixed = np.uint8(batch_fixed.detach().cpu().numpy().squeeze() * 255)
            moving = np.uint8(batch_moving.detach().cpu().numpy().squeeze() * 255)
            reg = np.uint8(registered.detach().cpu().numpy().squeeze() * 255)
            # plt.subplot(1, 3,1)
            # plt.imshow(np.stack([fixed, moving, np.zeros(fixed.shape, dtype='int')], axis=-1), cmap='gray')
            # plt.subplot(1, 3, 2)
            # plt.imshow(np.stack([fixed, reg, np.zeros(fixed.shape, dtype='int')], axis=-1), cmap='gray')
            # plt.subplot(1, 3, 3)
            # plt.imshow(np.stack([moving, reg, np.zeros(fixed.shape, dtype='int')], axis=-1), cmap='gray')
            # plt.pause(3)
            # plt.close()
            if first:
                new_seq = np.concatenate([batch_fixed[0].detach().cpu().numpy()[None],
                                          abs(batch_moving-registered).detach().cpu().numpy()], axis=0)
                first = False
            else:
                new_seq = np.concatenate([new_seq, abs(batch_moving-registered).detach().cpu().numpy()], axis=0)
            defs = np.concatenate([defs, deformations.detach().cpu().numpy()], axis=0)
        # save(seq, defs, save_path, file.split('/')[-1])
        save256(new_seq, save_path + '/256/', file.split('/')[-1])


if __name__ == "__main__":
    predict('./VoxelMorph/saved_models_def/vm_900', './VoxelMorph/data/', (1, 256, 256),
            6, './VoxelMorph/data/registered/result/interp/')
