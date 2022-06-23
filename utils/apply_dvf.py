from FieldGenerator import generate_field
import numpy as np
import skimage
import torch.nn.functional as F
import torch
from voxelmorph2d import SpatialTransformation
import cv2

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
    img = cv2.imread('cat.jpg', 0).astype(np.float32)/255.
    img = cv2.resize(img, (400, 300))
    # cv2.imshow("", img)
    # cv2.waitKey(0)
    # cv2.d
    # estroyAllWindows()

    shape = img.shape

    field_x = generate_field(distrib, Pkgen(3), shape)*10
    field_y = generate_field(distrib, Pkgen(3), shape)*10
    # print(field_x.shape)

    # mapx_base, mapy_base = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    # mapx = mapx_base + field_x
    #
    # mapy = mapy_base + field_y
    # print(mapy.shape)
    # deformed = cv2.remap(img, mapx.astype(np.float32), mapy.astype(np.float32),
    #                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
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
    cv2.imwrite("result.jpg", np.uint8(deformed.numpy().squeeze()*255.))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
