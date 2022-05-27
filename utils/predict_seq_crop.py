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
# from model_vm import DefNet
from src.voxelmorph2d import SpatialTransformation, DefNet

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


def crop2full(registered, deformations, batch_fixed, batch_moving, im_sizes):
    h, w = im_sizes[0][:2]
    fixed = np.zeros((h, w))
    moving = np.zeros((h, w))
    reg = np.zeros((h, w))
    deform = np.zeros((h, w, 2))
    dh, dw = batch_fixed.shape[-2:]
    m = h // dh + 1
    n = w // dw + 1
    k = 0
    for ii in range(m):
        for jj in range(n):
            h0, h1 = ii*dh, (ii+1)*dh
            w0, w1 = jj*dw, (jj+1)*dw
            if ii == m-1:
                h0, h1 = h-dh, h
            if jj == n-1:
                w0, w1 = w-dw, w
            fixed[h0: h1, w0: w1] = batch_fixed[k, 0]
            moving[h0: h1, w0: w1] = batch_moving[k, 0]
            reg[h0: h1, w0: w1] = registered[k, 0]
            deform[h0: h1, w0: w1] = deformations[k]
            k += 1

    return reg[None], deform[None], fixed[None], moving[None]


def GetBatch(chunk, imsize, mask_chunk):
    if mask_chunk is None:
        batch_size = len(chunk)-1
        im_sizes = []
        batch_fixed = torch.empty((0,) + imsize)
        batch_moving = torch.empty((0,) + imsize)

        for i in range(batch_size):
            fixed = chunk[i]
            if fixed.shape[-1] == 3:
                fixed = color.rgb2gray(fixed)
            im_sizes.append(fixed.shape)
            crops_fixed = []
            h, w = fixed.shape[:2]
            dh, dw = imsize[1:]
            m = h // dh + 1
            n = w // dw + 1
            for ii in range(m):
                for jj in range(n):
                    h0, h1 = ii*dh, (ii+1)*dh
                    w0, w1 = jj*dw, (jj+1)*dw
                    if ii == m-1:
                        h0, h1 = h-dh, h
                    if jj == n-1:
                        w0, w1 = w-dw, w
                    crops_fixed.append(fixed[h0: h1, w0: w1])
            crops_fixed = torch.Tensor(np.array(crops_fixed)[:, None]).float()
            batch_fixed = torch.cat([batch_fixed, crops_fixed / 255.], dim=0)
            moving = chunk[i+1]
            if moving.shape[-1] == 3:
                moving = color.rgb2gray(moving)
            crops_moving = []
            for ii in range(m):
                for jj in range(n):
                    h0, h1 = ii * dh, (ii + 1) * dh
                    w0, w1 = jj * dw, (jj + 1) * dw
                    if ii == m - 1:
                        h0, h1 = h - dh, h
                    if jj == n - 1:
                        w0, w1 = w - dw, w
                    crops_moving.append(moving[h0: h1, w0: w1])
            crops_moving = torch.Tensor(np.array(crops_moving)[:, None]).float()
            batch_moving = torch.cat([batch_moving, crops_moving / 255.], dim=0)
        return batch_fixed, batch_moving, im_sizes
    else:
        batch_size = len(chunk) - 1
        im_sizes = []
        batch_fixed = torch.empty((0,2) + imsize[1:])
        batch_moving = torch.empty((0,2) + imsize[1:])

        for i in range(batch_size):
            fixed = chunk[i]
            mask_fixed = mask_chunk[i]
            if fixed.shape[-1] == 3:
                fixed = color.rgb2gray(fixed)
            im_sizes.append(fixed.shape)
            crops_fixed = []
            crops_mask_f = []
            h, w = fixed.shape[:2]
            dh, dw = imsize[1:]
            m = h // dh + 1
            n = w // dw + 1
            for ii in range(m):
                for jj in range(n):
                    h0, h1 = ii * dh, (ii + 1) * dh
                    w0, w1 = jj * dw, (jj + 1) * dw
                    if ii == m - 1:
                        h0, h1 = h - dh, h
                    if jj == n - 1:
                        w0, w1 = w - dw, w
                    crops_fixed.append(fixed[h0: h1, w0: w1])
                    crops_mask_f.append(mask_fixed[h0: h1, w0: w1])
            crops_fixed = torch.Tensor(np.array(crops_fixed)[:, None]).float()/ 255.
            crops_mask = torch.Tensor(np.array(crops_mask_f)[:, None]).float()
            crops_fixed = torch.cat([crops_fixed, crops_mask], dim=1)
            batch_fixed = torch.cat([batch_fixed, crops_fixed], dim=0)
            moving = chunk[i + 1]
            mask_moving = mask_chunk[i+1]
            if moving.shape[-1] == 3:
                moving = color.rgb2gray(moving)
            crops_moving = []
            crops_mask_m = []
            for ii in range(m):
                for jj in range(n):
                    h0, h1 = ii * dh, (ii + 1) * dh
                    w0, w1 = jj * dw, (jj + 1) * dw
                    if ii == m - 1:
                        h0, h1 = h - dh, h
                    if jj == n - 1:
                        w0, w1 = w - dw, w
                    crops_moving.append(moving[h0: h1, w0: w1])
                    crops_mask_m.append(mask_moving[h0: h1, w0: w1])
            crops_moving = torch.Tensor(np.array(crops_moving)[:, None]).float() / 255.
            crops_mask = torch.Tensor(np.array(crops_mask_m)[:, None]).float()
            # print(crops_mask.max())
            crops_moving = torch.cat([crops_moving, crops_mask], dim=1)
            batch_moving = torch.cat([batch_moving, crops_moving], dim=0)
        return batch_fixed, batch_moving, im_sizes


def save256(new_seq, path, name):
    new_seq = new_seq * 255
    # print(np.array(new_seq.shape))
    new_seq = new_seq.astype('uint8')
    os.makedirs(path, exist_ok=True)
    io.imsave(path + name, new_seq)


def save(seq, deformations, path, name):
    os.makedirs(path, exist_ok=True)
    os.makedirs(path + '/deformations/', exist_ok=True)
    new_seq = []
    new_def = []
    SP = SpatialTransformation(False)
    #deformations *= 40.
    for i, im in enumerate(seq):
        if im.shape[-1] == 3:
            im = color.rgb2gray(im)*255
            im = im.astype('uint8')
        if i == 0:
            new_seq.append(im.copy())
            # new_def.append(np.zeros(tuple(im.shape[:2]) + (2, )))
        else:
            defxy = deformations[i].copy()

            if i != 1:
                defxy1 = ff_1_to_k(defxy1.copy(), defxy)
            else:
                defxy1 = defxy.copy()

            # new_def.append(defxy)

            im_new = SP(torch.tensor(im[None, None, :, :], dtype=torch.float),
                        torch.tensor(defxy1[None].transpose((0,3,1,2)), dtype=torch.float)).squeeze()
            im_new = np.uint8(im_new.numpy())
            new_seq.append(im_new.copy())
    io.imsave(path + name, np.array(new_seq, dtype='uint8'))
    np.save(path + '/deformations/' + name.split('.tif')[0], deformations)


def predict(model_path, image_path, im_size, batch_size, save_path):
    voxelmorph = DefNet(im_size[1:])

    # voxelmorph = torch.nn.DataParallel(voxelmorph)
    # voxelmorph.to('cuda')
    voxelmorph.load_state_dict(torch.load(model_path)['model_state_dict'])
    if use_gpu:
        voxelmorph.cuda()
    voxelmorph.eval()

    print("Voxelmorph loaded successfully!")

    filenames = glob(image_path + '*eq*1.tif')
    mask_filenames = [image_path + '/masks/'+ f.split('/')[-1].split('.tif')[0] + '_body.tif' for f in filenames]
    for file, mask_file in zip(filenames, mask_filenames):
        seq = io.imread(file)
        defs = np.zeros(((1,) + seq[0].shape + (2,)))
        masks = 1. - io.imread(mask_file)
        chunks = [seq[i:min(i + batch_size + 1, len(seq))] for i in range(0, len(seq)-1, batch_size)]
        mask_chunks = [masks[i:min(i + batch_size + 1, len(masks))] for i in range(0, len(masks)-1, batch_size)]
        print(len(chunks), len(chunks[0]), len(seq), file)
        first = True
        new_seq = []
        for i, (chunk, mask_chunk) in enumerate(zip(chunks, mask_chunks)):
            batch_fixed, batch_moving, imsizes = GetBatch(chunk, im_size, None)#mask_chunk)
            #print(batch_fixed.shape, len(chunk))
            if use_gpu:
                batch_fixed = batch_fixed.cuda()
                batch_moving = batch_moving.cuda()
            registered, deformations = voxelmorph(batch_moving, batch_fixed)
            registered = registered.detach().cpu().numpy()
            batch_fixed = batch_fixed.detach().cpu().numpy()
            batch_moving = batch_moving.detach().cpu().numpy()
            deformations = deformations.detach().cpu().numpy().transpose((0,2,3,1))

            
            registered, deformations, batch_fixed, batch_moving = crop2full(registered, deformations,
                                                                            batch_fixed, batch_moving, imsizes)
            #print(registered.shape, deformations.shape)
            if first:
                new_seq = np.concatenate([batch_fixed, registered], axis=0)
                first = False
            else:
                for d in defs[1::-1]:
                    if use_gpu:
                        registered = voxelmorph.spatial_transform(torch.tensor(registered[None]).float().cuda(), 
                                                                  torch.tensor(d.transpose((2, 0, 1))[None]).float().cuda()).squeeze(0)
                        registered = registered.cpu().numpy()
                    else:
                        registered = voxelmorph.spatial_transform(torch.tensor(registered[None]).float(), 
                                                                  torch.tensor(d.transpose((2, 0, 1))[None]).float()).squeeze(0)
                        registered = registered.numpy()
                new_seq = np.concatenate([new_seq, registered], axis=0)
            defs = np.concatenate([defs, deformations], axis=0)
            #test draw#################################################
            im1 = (batch_fixed*255).astype('uint8')
            im2 = (batch_moving * 255).astype('uint8')
            im3 = (registered * 255).astype('uint8')
            im_zero = np.zeros_like(im3)
            im = np.concatenate([im1, im2, im_zero], axis=0).transpose((1, 2, 0))
            imp = np.concatenate([im1, im3, im_zero], axis=0).transpose((1, 2, 0))
            im = np.concatenate([im, imp], axis=0)
            ds = np.concatenate([deformations[:, :,:, 0], deformations[:, :,:, 1], im_zero], axis=0).transpose((1, 2, 0)).astype('float')
            io.imsave(f'./tmp_{name}/def_{i}.png', ds)
            io.imsave(f'./tmp_{name}/{i}.jpg', im)
        # print(new_seq.shape)
        # input()
        save(seq, defs, save_path, file.split('/')[-1])
        # print(new_seq.shape, new_seq.max(), seq.max(), seq.shape)
        save256(new_seq, save_path + '/256/', file.split('/')[-1])

            # # print(registered_image.max(), registered_image.min())
            # save_images(batch_fixed, registered_images, deformations, imsizes, chunk, save_path)


if __name__ == "__main__":
    name = 'cross-corr_1311'
    os.makedirs(f'./tmp_{name}', exist_ok=True)
    # predict('./snapshots/ssim1/vm_1000', '/data/sim/Notebooks/VM/data/viz/fwd/ini', (1, 256, 256),
    #         1, '/data/sim/Notebooks/VM/data/viz/fwd/check/proposed/')
    predict(f'/data/sim/DefReg/snapshots/{name}/vm_800',
            '/data/sim/Notebooks/VM/data/', (1, 256, 256),
            1, f'/data/sim/Notebooks/VM/data/registered/result/{name}/')
