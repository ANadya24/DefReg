import cv2
from skimage import io
import javabridge
import bioformats
import numpy as np
import imageio
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.ndimage.interpolation import map_coordinates


def plot_grid(x, y, ax=None, **kwargs):
    """
    plot deformation field
    """
    ax = ax or plt.gca()
    segs1 = np.stack((x, y), axis=2)
    segs2 = segs1.transpose((1, 0, 2))
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()


def draw_grid(im, grid_size):
    """
    Draw grid lines on the image
    """
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(255,))


def make_gif(image_sequence, name='test.gif'):
    """
    Creation of gir from image sequence
    """
    imageio.mimsave(name, image_sequence)


if __name__ == "__main__":
    javabridge.start_vm(class_path=bioformats.JARS)
    name = '/home/nadya/Projects/VoxelMorph/data/SeqB1.tif'
    def_x_name = '/home/nadya/Projects/VoxelMorph/data/out/SeqB1_ffXsh_bcw.ics'
    def_y_name = '/home/nadya/Projects/VoxelMorph/data/out/SeqB1_ffYsh_bcw.ics'
    grid_step = 20
    seq = io.imread(name)
    im = seq[0]
    new_seq = [im]
    print(len(seq))
    for i in range(1, len(seq)):
        draw_grid(im, grid_step)
        h, w = im.shape
        def_x = bioformats.load_image(def_x_name, z=i)
        def_y = bioformats.load_image(def_y_name, z=i)
        grid_x, grid_y = np.meshgrid(np.arange(0, w), np.arange(0, h))
        distx, disty = grid_x + def_x, grid_y + def_y

        indices = np.reshape(disty, (-1, 1)), np.reshape(distx, (-1, 1))

        imm = map_coordinates(im.copy(), indices, order=1, mode='reflect').reshape(im.shape[:2])
        new_seq.append(imm)

    make_gif(new_seq, './VoxelMorph/test.gif')

    javabridge.kill_vm()
