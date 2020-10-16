from model_vm import DefNet
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
from voxelmorph2d import SpatialTransformation

use_gpu = torch.cuda.is_available()
devices = ['cpu', 'cuda']
device = devices[use_gpu]


def preprocess_im(img, imsize):
    to_tensor = torchvision.transforms.ToTensor()
    if img.shape[-1] == 3:
        img = color.rgb2gray(img)
    h, w = img.shape[:2]
    if h / w != 1.:
        # ch, cw = h // 2, w // 2
        # cr = min(h, w)
        if h < w:
            img = pad(img, (0, w - h, 0, 0))
        else:
            img = pad(img, (0, 0, 0, h - w))
        # if h < w:
        #     d = w - h
        #     img = np.pad(img, ((d // 2, d // 2 + d % 2), (0, 0)), 'constant')
        # else:
        #     d = h - w
        #     img = np.pad(img, ((0, 0), (d // 2, d // 2 + d % 2)), 'constant')
    return to_tensor(resize(img, imsize[1:]))[None].float()


def pad(img, padwidth):
    h0, h1, w0, w1 = padwidth[:4]
    if len(padwidth) < 5:
        img = np.pad(img, ((h0, h1), (w0, w1)))
    else:
        img = np.pad(img, ((h0, h1), (w0, w1), (0, 0)))

    return img


def GetBatch(chunk, imsize, first):
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
            # ch, cw = h // 2, w // 2
            # cr = min(h, w)
            if h < w:
                fixed = pad(fixed, (0, w - h, 0, 0))
            else:
                fixed = pad(fixed, (0, 0, 0, h - w))
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
            # if h < w:
            #     d = w - h
            #     moving = np.pad(moving, ((d // 2, d // 2 + d % 2), (0, 0)), 'constant')
            # else:
            #     d = h - w
            #     moving = np.pad(moving, ((0, 0), (d // 2, d // 2 + d % 2)), 'constant')
            if h < w:
                moving = pad(moving, (0, w - h, 0, 0))
            else:
                moving = pad(moving, (0, 0, 0, h - w))
            # ch, cw = h // 2, w // 2
            # cr = min(h, w)
            # moving = moving[ch - cr // 2:ch - cr // 2 + cr, cw - cr // 2: cw - cr//2 + cr]
        batch_moving[i] = to_tensor(resize(moving, imsize[1:]))
    return batch_fixed, batch_moving, im_sizes


def save256(new_seq, path, name):
    # new_seq = new_seq * 255
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
            # print(defxy.shape)
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
            # print('Before ', defxy.min(), defxy.max())
            # defxy = np.stack([defx, defy], axis=-1)
            # prev_def_xy = defxy
            # defxy = ff_1_to_k(new_def[i-1], defxy)

            new_def.append(defxy)
            # print(defxy.shape, im.shape)
            im_new = SP(torch.tensor(im[None, None, :, :], dtype=torch.float),
                        torch.tensor(defxy.transpose((2, 0, 1))[None], dtype=torch.float)).squeeze()
            im_new = np.uint8(im_new.numpy())
            # print(im_new.shape)
            # im_new = forward_warp(im, defxy)
            new_seq.append(im_new)


    io.imsave(path + name, np.array(new_seq, dtype='uint8'))
    np.save(path + '/deformations/' + name.split('.tif')[0], new_def)


def show(moving, fixed, reg):
    import cv2
    mov = moving[0]*255#.squeeze().detach().cpu().numpy()*255
    fix = fixed[0]*255#.squeeze().detach().cpu().numpy()*255
    reg = reg[0]*255#.squeeze().detach().cpu().numpy()*255
    im1 = np.zeros((256, 256, 3), dtype='uint8')
    im2 = np.zeros_like(im1)
    im1[:,:,0] = np.uint8(fix)
    im2[:, :, 0] = np.uint8(fix)
    im1[:, :, 1] = np.uint8(mov)
    im2[:, :, 1] = np.uint8(reg)
    im = np.concatenate([im1, im2], axis=1)
    print(im.shape)
    cv2.imwrite('test.jpg', im)
    input()


def predict(model_path, image_path, im_size, batch_size, save_path, is_2d=True):

    voxelmorph = DefNet(im_size[1:])

    # voxelmorph = torch.nn.DataParallel(voxelmorph)
    # voxelmorph.to('cuda')
    voxelmorph.load_state_dict(torch.load(model_path)['model_state_dict'])

    voxelmorph.cuda()
    voxelmorph.eval()

    print("Voxelmorph loaded successfully!")

    filenames = glob(image_path + '*eq*.tif')

    for file in filenames:
        defs = np.zeros(im_size + (2, ))
        seq = io.imread(file)
        # chunks = [seq[i:min(i + batch_size + 1, len(seq))] for i in range(0, len(seq)-1, batch_size)]
        first = True
        new_seq = []
        fixed = preprocess_im(seq[0], im_size)
        for i in tqdm(range(1, len(seq))):
            moving = preprocess_im(seq[i], im_size)
            if use_gpu:
                fixed = fixed.cuda()
                moving = moving.cuda()
            registered, deformations = voxelmorph(moving, fixed)
            # print(registered.shape, deformations.shape)
            registered = registered.detach().cpu().numpy()
            fixed1 = fixed.detach().cpu().numpy()
            deformations = deformations.detach().cpu().numpy()
            # show(moving.detach().cpu().numpy(), fixed, registered)
            if first:
                new_seq = np.concatenate([fixed1[0], registered.squeeze(1)], axis=0)
                first = False
            else:
                new_seq = np.concatenate([new_seq, registered.squeeze(1)], axis=0)
            # del fixed
            del moving
            # fixed = torch.Tensor(registered)
            defs = np.concatenate([defs, (deformations).transpose((0, 2, 3, 1))], axis=0)
        save(seq, defs, save_path, file.split('/')[-1])
        save256(new_seq, save_path + '/256/', file.split('/')[-1])

            # # print(registered_image.max(), registered_image.min())
            # save_images(batch_fixed, registered_images, deformations, imsizes, chunk, save_path)


if __name__ == "__main__":
    # predict('/data/sim/DefReg/snapshots/ssim1/vm_1000',
    #         '/data/sim/Notebooks/VM/data/', (1, 256, 256),
    #         1, '/data/sim/Notebooks/VM/data/registered/result/ssim1/')
    predict('./snapshots/ssim1/vm_840', '/data/sim/Notebooks/VM/data/viz/fwd/init*', (1, 256, 256),
            1, '/data/sim/Notebooks/VM/data/viz/fwd/check/proposed/')

    # predict('/data/sim/Notebooks/VM/snapshots/saved_models_def0.3/vm_180',
    #         '/data/sim/Notebooks/VM/data/', (1, 128, 128),
    #         1, '/data/sim/Notebooks/VM/data/registered/result/saved_models_def0.3/')
