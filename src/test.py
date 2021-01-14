import pytorch_ssim
import torch
from torch.autograd import Variable
from torch import optim
from scipy.ndimage.interpolation import map_coordinates
import cv2
import numpy as np

# npImg1 = cv2.imread("/mnt/tech5/VoxelMorph/cat.jpg")
# npImg1 = cv2.cvtColor(npImg1, cv2.COLOR_BGR2GRAY)[:,:,None]
# print(npImg1.shape)
# img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0)/255.0
# img2 = torch.rand(img1.size())
#
# if torch.cuda.is_available():
#     img1 = img1.cuda()
#     img2 = img2.cuda()
#
#
# img1 = Variable(img1,  requires_grad=True)
# img2 = Variable(img2, requires_grad = True)
#
# print(img1.shape)
# print(img2.shape)
# # Functional: pytorch_ssim.ssim(img1, img2, window_size = 11, size_average = True)
# ssim_value = 1-pytorch_ssim.ssim(img1, img2).item()
# print("Initial ssim:", ssim_value)
#
# # Module: pytorch_ssim.SSIM(window_size = 11, size_average = True)
# ssim_loss = pytorch_ssim.SSIM()
#
# optimizer = optim.Adam([img2], lr=0.01)
#
# while ssim_value > 0.05:
#     optimizer.zero_grad()
#     ssim_out = 1-ssim_loss(img1, img2)
#     ssim_value = ssim_out.item()
#     print(ssim_value)
#     ssim_out.backward()
#     optimizer.step()
#     cv2.imshow('op',np.transpose(img2.cpu().detach().numpy()[0],(1,2,0)))
#     cv2.waitKey()

res = np.load('./VoxelMorph/data/pairs/SeqB1_0.npy')
res = res.reshape(4, 356, 287)
im1 = res[0].astype('uint8')
im2 = res[1].astype('uint8')
def_x, def_y = res[2], res[3]
h, w = im1.shape
zero_im = np.zeros((h, w, 3), 'uint8')
# f = lambda x,y : ( x+0.8*np.exp(-x**2-y**2),y )
grid_x, grid_y = np.meshgrid(np.arange(0, w), np.arange(0, h))
distx, disty = grid_x - def_x, grid_y - def_y

indices = np.reshape(disty, (-1, 1)), np.reshape(distx, (-1, 1))

im2_new = map_coordinates(im2.copy(), indices, order=1, mode='reflect').reshape(im2.shape[:2])
zero_im[:,:, 2] = im1
zero_im[:, :, 0] = im2_new

cv2.imshow('name', zero_im)
cv2.waitKey(0)
cv2.destroyAllWindows()