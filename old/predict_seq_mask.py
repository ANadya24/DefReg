import torch
import torchvision
from glob import glob
import skimage.io as io
from skimage import color
import os
from skimage.transform import resize
from utils.ffRemap import *
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


def GetBatch(chunk, imsize, mask_chunk):
    if mask_chunk is None:
        batch_size = len(chunk)-1
        im_sizes = []
        batch_fixed = torch.empty((0,) + imsize)
        batch_moving = torch.empty((0,) + imsize)
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
            batch_fixed = torch.cat([batch_fixed, to_tensor(resize(fixed, imsize[1:]))][None], dim=0)
            moving = chunk[i+1]
            if moving.shape[-1] == 3:
                moving = color.rgb2gray(moving)
            if h / w != 1.:
                if h < w:
                    moving = pad(moving, (0, w - h, 0, 0))
                else:
                    moving = pad(moving, (0, 0, 0, h - w))

            batch_moving = torch.cat([batch_moving, to_tensor(resize(moving, imsize[1:]))][None], dim=0)
    else:
        batch_size = len(chunk) - 1
        im_sizes = []
        batch_fixed = torch.empty((0,2) + imsize[1:])
        batch_moving = torch.empty((0,2) + imsize[1:])
        to_tensor = torchvision.transforms.ToTensor()
        for i in range(batch_size):
            fixed = chunk[i]
            mask_fixed = mask_chunk[i]
            if fixed.shape[-1] == 3:
                fixed = color.rgb2gray(fixed)
            im_sizes.append(fixed.shape)
            h, w = fixed.shape[:2]
            if h / w != 1.:
                if h < w:
                    fixed = pad(fixed, (0, w - h, 0, 0))
                    mask_fixed = pad(mask_fixed, (0, w-h, 0, 0))
                else:
                    fixed = pad(fixed, (0, 0, 0, h - w))
                    mask_fixed = pad(mask_fixed, (0, 0, 0, h-w))
            fixed = to_tensor(resize(fixed, imsize[1:])).float()[None]
            mask = torch.Tensor(resize(mask_fixed, imsize[1:]) > 0).float()[None, None]     
            batch_fixed = torch.cat([batch_fixed, torch.cat([fixed, mask], dim=1)], dim=0)
            moving = chunk[i+1]
            mask_moving= mask_chunk[i+1]
            if moving.shape[-1] == 3:
                moving = color.rgb2gray(moving)
            if h / w != 1.:
                if h < w:
                    moving = pad(moving, (0, w - h, 0, 0))
                    mask_moving = pad(mask_moving, (0, w - h, 0, 0))
                else:
                    moving = pad(moving, (0, 0, 0, h - w))
                    mask_moving = pad(mask_moving, (0, h - w, 0, 0))

            moving = to_tensor(resize(moving, imsize[1:])).float()[None]
            mask = torch.Tensor(resize(mask_moving, imsize[1:]) > 0).float()[None, None]
            batch_moving = torch.cat([batch_moving, torch.cat([moving, mask], dim=1)], dim=0)
    return batch_fixed, batch_moving, im_sizes


def save256(new_seq, path, name):
    new_seq = new_seq * 255
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
            if h / w != 1.:
                if h < w:
                    d = w - h
                    tmp = resize(defxy, (w, w))
                    defxy = tmp[:-d]
                else:
                    d = h - w
                    tmp = resize(defxy, (h, h))
                    defxy = tmp[:,:-d]
               
            else:
                defxy = resize(defxy, (w, h))
        
            if i != 1:
                defxy1 = ff_1_to_k(defxy1, defxy)
            else:
                defxy1 = defxy.copy()

            new_def.append(defxy)

            im_new = SP(torch.tensor(im[None, None, :, :], dtype=torch.float),
                        torch.tensor(defxy1[None].transpose((0,3,1,2)), dtype=torch.float)).squeeze()

            im_new = np.uint8(im_new.numpy())
            new_seq.append(im_new)

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

    filenames = glob(image_path + '*eq*1.tif')
    mask_filenames = [image_path + '/masks/'+ f.split('/')[-1].split('.tif')[0] + '_body.tif' for f in filenames]
    for file, mask_file in zip(filenames, mask_filenames):
        seq = io.imread(file)
        defs = np.zeros((tuple(im_size) + (2,)))
        masks = 1. - io.imread(mask_file)
        chunks = [seq[i:min(i + batch_size + 1, len(seq))] for i in range(0, len(seq)-1, batch_size)]
        mask_chunks = [masks[i:min(i + batch_size + 1, len(masks))] for i in range(0, len(masks)-1, batch_size)]
        print(len(chunks), len(chunks[0]), len(seq), file)
        first = True
        new_seq = []
        for i, (chunk, mask_chunk) in enumerate(zip(chunks, mask_chunks)):
            batch_fixed, batch_moving, imsizes = GetBatch(chunk, im_size, mask_chunk)
            if use_gpu:
                batch_fixed = batch_fixed.cuda()
                batch_moving = batch_moving.cuda()
            registered, deformations = voxelmorph(batch_moving, batch_fixed)
            registered = registered.detach().cpu().numpy()[:, 0]
            batch_fixed = batch_fixed.detach().cpu().numpy()[:, 0]
            batch_moving = batch_moving.detach().cpu().numpy()[:, 0]
            deformations = deformations.detach().cpu().numpy().transpose((0, 2, 3, 1))*40.
            # print(registered.shape, deformations.shape)
            if first:
                new_seq = np.concatenate([batch_fixed[0].squeeze()[None],
                                          registered], axis=0)
                first = False
            else:
                new_seq = np.concatenate([new_seq, registered], axis=0)
            defs = np.concatenate([defs, deformations], axis=0)

            im1 = (batch_fixed * 255).astype('uint8').squeeze()[None]
            im2 = (batch_moving * 255).astype('uint8').squeeze()[None]
            im3 = (registered * 255).astype('uint8').squeeze()[None]
            im_zero = np.zeros_like(im3)
            im = np.concatenate([im1, im2, im_zero], axis=0).transpose((1, 2, 0))
            imp = np.concatenate([im1, im3, im_zero], axis=0).transpose((1, 2, 0))
            im = np.concatenate([im, imp], axis=0)
            io.imsave(f'./tmp_{name}/{i}.jpg', im)
        # print(new_seq.shape)
        # input()
        save(seq, defs, save_path, file.split('/')[-1])
        save256(new_seq, save_path + '/256/', file.split('/')[-1])

            # # print(registered_image.max(), registered_image.min())
            # save_images(batch_fixed, registered_images, deformations, imsizes, chunk, save_path)


if __name__ == "__main__":
    name = 'cross-corr_0411'
    os.makedirs(f'./tmp_{name}', exist_ok=True)
    # predict('./snapshots/ssim1/vm_1000', '/data/sim/Notebooks/VM/data/viz/fwd/ini', (1, 256, 256),
    #         1, '/data/sim/Notebooks/VM/data/viz/fwd/check/proposed/')
    predict(f'/data/sim/DefReg/snapshots/{name}/vm_440',
            '/data/sim/Notebooks/VM/data/', (1, 256, 256),
            1, '/data/sim/Notebooks/VM/data/registered/result/cross-corr_0411/')
