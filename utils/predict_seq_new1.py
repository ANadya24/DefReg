import src.voxelmorph2d as vm2d
import src.voxelmorph3d as vm3d
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
from src.model_vm import DefNet
from src.voxelmorph2d import SpatialTransformation

use_gpu = torch.cuda.is_available()
devices = ['cpu', 'cuda']
device = devices[use_gpu]


def pad(img, padwidth):
    h0, h1, w0, w1 = padwidth[:4]
    if len(padwidth) < 5:
        img = np.pad(img, ((h0, h1), (w0, w1)))
    else:
        img = np.pad(img, ((h0, h1), (w0, w1), (0, 0)))

    return img


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
                fixed = pad(fixed, (0, w - h, 0, 0))
            else:
                fixed = pad(fixed, (0, 0, 0, h - w))
            # ch, cw = h // 2, w // 2
            # cr = min(h, w)
            # if h < w:
            #     d = w - h
            #     fixed = np.pad(fixed, ((d // 2, d // 2 + d % 2), (0, 0)), 'constant')
            # else:
            #     d = h - w
            #     fixed = np.pad(fixed, ((0, 0), (d // 2, d // 2 + d % 2)), 'constant')
            # fixed = fixed[ch - cr // 2:ch - cr // 2 + cr, cw - cr // 2: cw - cr//2 + cr]
        # print(to_tensor(resize(fixed, imsize[1:])).shape)
        batch_fixed[i] = to_tensor(resize(fixed, imsize[1:]))
        moving = chunk[i+1]
        if moving.shape[-1] == 3:
            moving = color.rgb2gray(moving)
        if h / w != 1.:
            if h < w:
                moving = pad(moving, (0, w - h, 0, 0))
            else:
                moving = pad(moving, (0, 0, 0, h - w))

            # if h < w:
            #     d = w - h
            #     moving = np.pad(moving, ((d // 2, d // 2 + d % 2), (0, 0)), 'constant')
            # else:
            #     d = h - w
            #     moving = np.pad(moving, ((0, 0), (d // 2, d // 2 + d % 2)), 'constant')
            # ch, cw = h // 2, w // 2
            # cr = min(h, w)
            # moving = moving[ch - cr // 2:ch - cr // 2 + cr, cw - cr // 2: cw - cr//2 + cr]
        batch_moving[i] = to_tensor(resize(moving, imsize[1:]))
    return batch_fixed, batch_moving, im_sizes


def save256(new_seq, path, name):
    # new_seq = new_seq * 255
    # print(np.array(new_seq.shape))
    new_seq.astype('uint8')
    os.makedirs(path, exist_ok=True)
    io.imsave(path + name, new_seq)


def save(seq, deformations, path, name):
    os.makedirs(path, exist_ok=True)
    os.makedirs(path + '/deformations/', exist_ok=True)
    new_seq = []
    new_def = []
    SP = SpatialTransformation(False)

    for i, im in enumerate(seq):
        if im.shape[-1] == 3:
            im = color.rgb2gray(im)*255
            im = im.astype('uint8')
        if i == 0:
            new_seq.append(im)
            new_def.append(np.zeros(tuple(im.shape[:2]) + (2, )))
        else:
            h, w = im.shape[:2]
            defxy = deformations[i]
            # val = np.load('./data/pairs/' + name.split('.tif')[0] + f'_{i-1}.npy').reshape(4, h,w)
            # _, _, defx, defy = val
            if h / w != 1.:
                if h < w:
                    d = w - h
                    tmp = resize(defxy, (w, w))
                    # defxy = tmp[d // 2: -(d // 2 + d % 2)]
                    defxy = tmp[:-d]
                else:
                    d = h - w
                    tmp = resize(defxy, (h, h))
                    # defxy = tmp[:, d // 2: -(d // 2 + d % 2)]
                    defxy = tmp[:, :-d]
                # ch, cw = h // 2, w // 2
                # cr = min(h, w)
                # tmp = resize(defxy, (cr, cr))
                # defxy = np.zeros((h, w, 2))
                # defxy[ch - cr // 2:ch - cr // 2 + cr, cw - cr // 2: cw - cr//2 + cr] = tmp
            else:
                defxy = resize(defxy, (h, w))
            print('Before ', defxy.min(), defxy.max())
            # defxy = np.stack([defx, defy], axis=-1)
            # print(new_def[-1].shape, defxy.shape)
            if i != 1:
                defxy = ff_1_to_k(new_def[i-1], defxy)

            print('After ', defxy.min(), defxy.max())
            # print(np.array(new_def).shape)
            new_def.append(defxy)
            # print(im[None, None].shape, defxy[None].transpose((0,3,1,2)).shape)
            # print(im.max(), im[None, None, :, :].shape, defxy[None].transpose((0,3,1,2)).shape)

            im_new = SP(torch.tensor(im[None, None, :, :], dtype=torch.float),
                        torch.tensor(defxy[None].transpose((0,3,1,2)), dtype=torch.float)).squeeze()

            im_new = np.uint8(im_new.numpy())
            # print(im_new.shape)
            # im_new = forward_warp(im, defxy)
            new_seq.append(im_new)
            # print(np.array(new_seq, dtype='uint8'))
    io.imsave(path + name, np.array(new_seq, dtype='uint8'))
    np.save(path + '/deformations/' + name.split('.tif')[0], new_def)


def predict(model_path, image_path, im_size, batch_size, save_path):
    voxelmorph = DefNet(im_size[1:])

    # voxelmorph = torch.nn.DataParallel(voxelmorph)
    # voxelmorph.to('cuda')
    voxelmorph.load_state_dict(torch.load(model_path)['model_state_dict'])

    voxelmorph.cuda()
    voxelmorph.eval()

    print("Voxelmorph loaded successfully!")

    filenames = glob(image_path + '*eq*.tif')
    for file in filenames:
        defs = np.zeros((tuple(im_size) + (2,)))
        seq = io.imread(file)
        chunks = [seq[i:min(i + batch_size + 1, len(seq))] for i in range(0, len(seq)-1, batch_size)]
        print(len(chunks), len(chunks[0]), len(seq), file)
        first = True
        new_seq = []
        for chunk in tqdm(chunks):
            batch_fixed, batch_moving, imsizes = GetBatch(chunk, im_size)
            if use_gpu:
                batch_fixed = batch_fixed.cuda()
                batch_moving = batch_moving.cuda()
            registered, deformations = voxelmorph(batch_moving, batch_fixed)
            registered = registered.squeeze(1)
            # print(registered.shape, deformations.shape)
            if first:
                new_seq = np.concatenate([batch_fixed[0].detach().cpu().numpy().squeeze()[None],
                                          registered.detach().cpu().numpy()], axis=0)
                first = False
            else:
                new_seq = np.concatenate([new_seq, registered.squeeze(1).detach().cpu().numpy()], axis=0)
            defs = np.concatenate([defs, deformations.detach().cpu().numpy().transpose((0,2,3,1))], axis=0)
        # print(new_seq.shape)
        # input()
        save(seq, defs, save_path, file.split('/')[-1])
        save256(new_seq, save_path + '/256/', file.split('/')[-1])

            # # print(registered_image.max(), registered_image.min())
            # save_images(batch_fixed, registered_images, deformations, imsizes, chunk, save_path)


if __name__ == "__main__":
    # predict('./snapshots/ssim1/vm_1000', '/data/sim/Notebooks/VM/data/viz/fwd/', (1, 256, 256),
    #         1, '/data/sim/Notebooks/VM/data/viz/fwd/check/proposed/')
    predict('/data/sim/DefReg/snapshots/ssim1/vm_1000',
            '/data/sim/Notebooks/VM/data/', (1, 256, 256),
            1, '/data/sim/Notebooks/VM/data/registered/result/ssim1/')
