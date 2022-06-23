import torch
from torchvision.transforms import ToTensor
from torch.utils import data
from skimage.transform import resize
import skimage.io as io
import numpy as np


class Dataset(data.Dataset):
    """
    Dataset class for converting the data into batches.
    The data.Dataset class is a pyTorch class which help
    in speeding up  this process with effective parallelization
    """
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, im_size=(256, 256), use_mask=False, size_file=None):
        """Initialization"""
        self.list_IDs = list_IDs
        self.im_size = im_size
        self.use_mask = use_mask
        if size_file is None:
            self.shape = im_size
        else:
            self.shape = {}
            with open(size_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                vals = line.split('\t')
                name = vals[0]
                shape = tuple(map(int, vals[1:]))
                self.shape[name] = shape

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.list_IDs)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        ID = self.list_IDs[index]

        to_tensor = ToTensor()
        arr = np.load(ID)
        if isinstance(self.shape, dict):
            h, w = self.shape[ID.split('/')[-1].split('_')[0]]
            arr = arr.reshape(4, h, w)
        else:
            h, w = self.shape
            arr = arr.reshape(4, h, w)

        # Load data and get label
        # fixed_image = io.imread(ID + '_1.jpg', as_gray=True)
        fixed_image = arr[0].astype('uint8')
        if h / w != 1.:
            ch, cw = h//2, w//2
            cr = min(h, w)
            fixed_image = fixed_image[ch-cr//2:ch+cr, cw-cr//2: cw+cr]
        fixed_image = resize(fixed_image, self.im_size)
        fixed_image = to_tensor(fixed_image).float()

        # print(fixed_image.shape, fixed_image.max(), fixed_image.min())

        # moving_image = io.imread(ID + '_2.jpg', as_gray=True)
        moving_image = arr[1].astype('uint8')
        if h / w != 1.:
            ch, cw = h//2, w//2
            cr = min(h, w)
            moving_image = moving_image[ch-cr//2:ch+cr, cw-cr//2: cw+cr]
        moving_image = resize(moving_image, self.im_size)
        moving_image = to_tensor(moving_image).float()
        # print(moving_image.shape)

        deformation = arr[2:].transpose(1, 2, 0)
        if h / w != 1.:
            ch, cw = h//2, w//2
            cr = min(h, w)
            deformation = deformation[ch-cr//2:ch+cr, cw-cr//2: cw+cr]
        deformation = resize(deformation, self.im_size)
        deformation = torch.Tensor(deformation)

        if self.use_mask:
            name = "".join([x + '/' for x in ID.split('/')[:-2]]) + 'masks/' + ID.split('/')[-1]
            fixed_image_mask = io.imread(name + '_1.jpg', as_gray=True) / 255.
            fixed_image_mask = resize(fixed_image_mask, self.im_size)
            fixed_image_mask = torch.Tensor(fixed_image_mask[None])
            # print (fixed_image.shape, fixed_image_mask.shape)
            # fixed_image = torch.cat([fixed_image, fixed_image_mask], dim=0)
            fixed_image *= fixed_image_mask

            moving_image_mask = io.imread(name + '_2.jpg', as_gray=True) / 255.
            moving_image_mask = resize(moving_image_mask, self.im_size)
            moving_image_mask = torch.Tensor(moving_image_mask[None])
            # moving_image = torch.cat([moving_image, moving_image_mask], dim=0)
            moving_image *= moving_image_mask

        return fixed_image, moving_image, deformation

from glob import glob

path = '/home/nadya/Projects/VoxelMorph/data/pairs/fwd/*.npy'
paths = glob(path)
dataset = Dataset(paths, (256, 256), size_file='/home/nadya/Projects/VoxelMorph/data/sizes.txt')
fixed, moving, deform = dataset[-1]
print(fixed.shape, moving.shape, deform.shape)
from voxelmorph2d import SpatialTransformation
SP = SpatialTransformation()
movingN = SP(moving[:,:,:,None], deform[None]).squeeze()
movingN = np.uint8(movingN.numpy()*255)
print(movingN.shape)
fixed = np.uint8(fixed.numpy().squeeze() * 255)
moving = np.uint8(moving.numpy().squeeze() * 255)

print(fixed.shape, moving.shape, deform.shape)
print(fixed.max())
from matplotlib import pyplot as plt
plt.imshow(np.stack([fixed, moving, np.zeros(fixed.shape, dtype='int')], axis=-1), cmap='gray')
plt.waitforbuttonpress()


plt.figure()
plt.imshow(np.stack([fixed, movingN, np.zeros(fixed.shape, dtype='int')], axis=-1), cmap='gray')
plt.waitforbuttonpress()
plt.close()