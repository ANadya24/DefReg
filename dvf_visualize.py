import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.ndimage.interpolation import map_coordinates
import cv2
from PIL import Image
from skimage import io
import javabridge
import bioformats
import numpy as np
import matplotlib.pyplot as plt
import imageio


def plot_grid(x,y, ax=None, **kwargs):
    ax = ax or plt.gca()
    segs1 = np.stack((x,y), axis=2)
    segs2 = segs1.transpose(1,0,2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()


def draw_grid(im, grid_size):
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(255,))


def make_gif(image_sequence, name='test.gif'):
    imageio.mimsave(name, image_sequence)


if __name__ == "__main__":
    javabridge.start_vm(class_path=bioformats.JARS)
    name = '/home/nadya/Projects/VoxelMorph/data/SeqB1.tif'
    def_x_name = '/home/nadya/Projects/VoxelMorph/data/out/SeqB1_ffXsh_bcw.ics'
    def_y_name = '/home/nadya/Projects/VoxelMorph/data/out/SeqB1_ffYsh_bcw.ics'
    grid_step = 20
    seq = io.imread(name)
    im = seq[0]
    new_seq = []
    new_seq.append(im)
    print(len(seq))
    for i in range(1, len(seq)):
        draw_grid(im, grid_step)
        # if i == 0:
        #     continue
        h, w = im.shape
        # print(im.shape, im.dtype)
        def_x = bioformats.load_image(def_x_name, z=i)
        def_y = bioformats.load_image(def_y_name, z=i)
        # print(def_x.shape)

        # f = lambda x,y : ( x+0.8*np.exp(-x**2-y**2),y )
        grid_x, grid_y = np.meshgrid(np.arange(0, w), np.arange(0, h))
        distx, disty = grid_x + def_x, grid_y + def_y

        indices = np.reshape(disty, (-1, 1)), np.reshape(distx, (-1, 1))

        imm = map_coordinates(im.copy(), indices, order=1, mode='reflect').reshape(im.shape[:2])
        new_seq.append(imm)

    make_gif(new_seq, './VoxelMorph/test.gif')

    # plt.figure(figsize=(16, 14))
    # plt.imshow(np.c_[seq, im2], cmap='gray')
    # plt.show()
    javabridge.kill_vm()
    #######################################################################
    # fig, ax = plt.subplots()
    # ax.imshow(seq)
    # grid_x, grid_y = np.meshgrid(np.linspace(0, 287, 287 // step), np.linspace(0, 356, 356 // step))
    # plot_grid(grid_x, grid_y, ax=ax,  color="lightgrey")
    # distx, disty = grid_x + def_x[::step, ::step], grid_y + def_y[::step, ::step]
    # plot_grid(distx, disty, ax=ax, color="C0")
    # plt.show()
    #######################################################################


    # plt.show()
