import os
import numpy as np
from glob import glob
import skimage.io as io
from skimage import color
from skimage.transform import resize
from tqdm import tqdm
import torch
from torchvision.transforms import ToTensor
import albumentations as A

from models.defregnet_model import DefRegNet
from basic_nets.spatial_transform import SpatialTransformation
from utils.data_process import pad_image, match_histograms, normalize_mean_std


use_gpu = torch.cuda.is_available()
devices = ['cpu', 'cuda']
device = devices[use_gpu]


def preprocess_image_pair(image1, image2, im_size,
                          use_masks=False, mask1=None, mask2=None):
    to_tensor = ToTensor()
    resizer = A.Resize(im_size)

    if image1.shape[-1] == 3:
        image1 = color.rgb2gray(image1)

    if image2.shape[-1] == 3:
        image1 = color.rgb2gray(image1)

    image1 = normalize_mean_std(image1)
    image2 = normalize_mean_std(image2)

    image1, image2, _ = match_histograms(image1, image2, random_switch=False)
    h, w = image1.shape

    if h != w:
        if h < w:
            image1 = pad_image(image1, (0, w - h, 0, 0))
            image2 = pad_image(image2, (0, w - h, 0, 0))
            if use_masks:
                mask1 = pad_image(mask1, (0, w - h, 0, 0))
                mask2 = pad_image(mask2, (0, w - h, 0, 0))
        else:
            image1 = pad_image(image1, (0, 0, 0, h - w))
            image2 = pad_image(image2, (0, 0, 0, h - w))
            if use_masks:
                mask1 = pad_image(mask1, (0, 0, 0, h - w))
                mask2 = pad_image(mask2, (0, 0, 0, h - w))

    data1 = {'image': image1}
    if use_masks:
        data1['mask'] = mask1
    data1 = resizer(*data1)
    image1 = data1['image']
    if use_masks:
        mask1 = data1['mask']

    data2 = {'image': image2}
    if use_masks:
        data2['mask'] = mask2
    data2 = resizer(*data2)
    image2 = data2['image']
    if use_masks:
        mask2 = data2['mask']

    image1 = to_tensor(image1).float()
    image2 = to_tensor(image2).float()

    if use_masks:
        image1 = torch.cat([image1, torch.Tensor(mask1).float()[None]], 0)
        image2 = torch.cat([image2, torch.Tensor(mask2).float()[None]], 0)
    return image1, image2


def save256(new_seq, path, name):
    # new_seq = new_seq * 255
    new_seq.astype(np.uint8)
    os.makedirs(path, exist_ok=True)
    io.imsave(path + name, new_seq)


def save(seq, deformations, path, name):
    os.makedirs(path, exist_ok=True)
    os.makedirs(path + '/deformations/', exist_ok=True)
    new_seq = []
    new_def = []
    SP = SpatialTransformation()

    for i, im in enumerate(seq):
        if im.shape[-1] == 3:
            im = color.rgb2gray(im) * 255
            im = im.astype(np.uint8)
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
                    defxy = tmp[:, :-d]
            else:
                defxy = resize(defxy, (h, w))

            new_def.append(defxy)
            im_new = SP(torch.tensor(im[None, None, :, :], dtype=torch.float),
                        torch.tensor(defxy.transpose((2, 0, 1))[None], dtype=torch.float)).squeeze()
            im_new = np.uint8(im_new.numpy())
            new_seq.append(im_new)

    io.imsave(path + name, np.array(new_seq, dtype=np.uint8))
    np.save(path + '/deformations/' + name.split('.tif')[0], new_def)


def show(moving, fixed, reg):
    import cv2
    mov = moving[0] * 255
    fix = fixed[0] * 255
    reg = reg[0] * 255
    im1 = np.zeros((256, 256, 3), dtype=np.uint8)
    im2 = np.zeros_like(im1)
    im1[:,:,0] = np.uint8(fix)
    im2[:, :, 0] = np.uint8(fix)
    im1[:, :, 1] = np.uint8(mov)
    im2[:, :, 1] = np.uint8(reg)
    im = np.concatenate([im1, im2], axis=1)
    print(im.shape)
    cv2.imwrite('../old/test.jpg', im)
    input()


def predict(model_path, image_path, im_size, save_path):

    model_dict = torch.load(model_path)
    if 'model' not in model_dict:
        model = DefRegNet(1, im_size[0])
    else:
        model = model_dict['model']
    model.load_state_dict(model_dict['model_state_dict'])

    model.cuda()
    model.eval()

    print("Model loaded successfully!")

    filenames = glob(image_path + '*eq*.tif')

    for file in filenames:
        defs = np.zeros(im_size + (2, ))
        seq = io.imread(file)
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
