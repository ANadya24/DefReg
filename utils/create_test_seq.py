import numpy as np
import skimage
import torch.nn.functional as F
from scipy import ndimage
import torch
from src.voxelmorph2d import SpatialTransformation
import cv2
from glob import glob
from skimage import io, transform
import pickle

# Helper that generates power-law power spectrum
def Pkgen(n):
    def Pk(k):
        return np.power(k, -n)
    return Pk


# Draw samples from a normal distribution
def distrib(shape):
    a = np.random.normal(loc=0, scale=1, size=shape)
    b = np.random.normal(loc=0, scale=1, size=shape)
    return a + 1j * b


if __name__ == '__main__':
    sequences = glob('./VoxelMorph/data/*.tif')
    for name in filter(lambda name: name.find('SeqB1') != -1, sequences):
        img = io.imread(name)[0]
        img = ndimage.gaussian_filter(img, 1.)
    # img = cv2.imread('cat.jpg', 0).astype(np.float32)/255.
    img = (img - img.min())/ (img.max() - img.min())
    # img = cv2.resize(img, (400, 300))
    # cv2.imshow("", img)
    # cv2.waitKey(0)
    # cv2.d
    # estroyAllWindows()

    shape = [s//8 for s in img.shape]
    seq = np.zeros((10,) + img.shape, dtype='uint8')
    seq[0] =  ndimage.gaussian_filter(io.imread(name)[0], 1.)

    deformations = np.zeros((10,) + img.shape + (2,), dtype='float32')
    print(seq.shape, deformations.shape)
    for i in range(1, 10):
        # field_x = ndimage.gaussian_filter(np.random.uniform(-10., 10., shape), 2.)#generate_field(distrib, Pkgen(3), shape)*10
        # field_y = ndimage.gaussian_filter(np.random.uniform(-10., 10., shape), 2.)#generate_field(distrib, Pkgen(3), shape)*10
        field_x = np.random.uniform(-7., 7., shape)
        field_y = np.random.uniform(-7., 7., shape)
        field_x = transform.resize(field_x, img.shape)
        field_y = transform.resize(field_y, img.shape)
        field_x = ndimage.gaussian_filter(field_x, 5.)
        field_y = ndimage.gaussian_filter(field_x, 5.)
        # print(field_x.shape)

        # mapx_base, mapy_base = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        # mapx = mapx_base + field_x
        #
        # mapy = mapy_base + field_y
        # print(mapy.shape)
        # deformed = cv2.remap(img, mapx.astype(np.float32), mapy.astype(np.float32),
        #                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        deform = np.stack([field_x, field_y], -1)

        map = torch.cat([torch.tensor(field_x[:, :, None], dtype=torch.float32),
                         torch.tensor(field_y[:, :, None], dtype=torch.float32)], dim=2)[None]
        # print(map.max(), map.min())
        tensor = torch.tensor(img[None, :, :, None])
        # print(tensor.shape)
        deformed = SpatialTransformation()(tensor, map)
        # deformed = F.grid_sample(torch.tensor(img[None, None]),
        #                          torch.stack((torch.tensor(mapx, dtype=torch.float32),
        #                                       torch.tensor(mapy, dtype=torch.float32)), dim=2)[None],
        #                          mode='bilinear', padding_mode='reflection')
        # print(deformed.numpy().max(), deformed.numpy().min(), deformed.numpy().squeeze().shape)
        deformations[i] = deform
        seq[i] = deformed.numpy().squeeze() * 255.
        # cv2.imwrite("./VoxelMorph/cell.jpg", np.uint8(deformed.numpy().squeeze()*255.))
        # cv2.imwrite("./VoxelMorph/cell1.jpg", np.uint8(img * 255.))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    io.imsave('/home/nadya/Projects/VoxelMorph/test_seq/testSeq_bcw.tiff', seq[::-1])
    with open('/home/nadya/Projects/VoxelMorph/test_seq/testSeq_bcw', 'wb') as f:
        pickle.dump(deformations[::-1], f)