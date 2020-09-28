import numpy as np
import cv2
import os
from skimage import io


def validate_images(images1, images2, images3, val_dir='val_images/', epoch=0, train=False):
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    images1 = images1.cpu().detach().numpy()
    images2 = images2.cpu().detach().numpy()
    images3 = images3.cpu().detach().numpy()
    for i, (im1, im2, im3) in enumerate(zip(images1, images2, images3)):
        im1 *= 255.
        im2 *= 255.
        im3 *= 255.
        im1 = im1[0].astype('uint8')
        im2 = im2[0].astype('uint8')
        im3 = im3[0].astype('uint8')
        if train:
            io.imsave(val_dir + f'train_epoch{epoch}_{i}_1.jpg', im1)
            io.imsave(val_dir + f'train_epoch{epoch}_{i}_2.jpg', im2)
            io.imsave(val_dir + f'train_epoch{epoch}_{i}_3.jpg', im3)
        else:
            io.imsave(val_dir + f'val_epoch{epoch}_{i}_1.jpg', im1)
            io.imsave(val_dir + f'val_epoch{epoch}_{i}_2.jpg', im2)
            io.imsave(val_dir + f'val_epoch{epoch}_{i}_3.jpg', im3)

        i += 1
