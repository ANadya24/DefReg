from skimage import io
import javabridge
import bioformats
import numpy as np
import os
from glob import glob
from ffRemap import *
import imagesize
from tqdm import tqdm


if __name__ == "__main__":
    javabridge.start_vm(class_path=bioformats.JARS)
    sequences = glob('./VoxelMorph/data/Series009*.tif')
    print(len(sequences))

    for name in sequences:  # filter(lambda name: name.find('Seq') != -1, sequences)
        def_x_name = '/home/nadya/Projects/VoxelMorph/data/deformations/ics/' \
                     + name.split('/')[-1].split('.tif')[0] + '_ffXsh_fwd.ics'
        def_y_name = '/home/nadya/Projects/VoxelMorph/data/deformations/ics/' \
                     + name.split('/')[-1].split('.tif')[0] + '_ffYsh_fwd.ics'
        seq = io.imread(name)
        deform = np.empty(seq.shape + (2,))
        print(deform.shape)
        print(name, def_x_name)
        for i in tqdm(range(len(seq))):
            deform[i, :, :, 0] = bioformats.load_image(def_x_name, z=i)[None]
            deform[i, :, :, 1] = bioformats.load_image(def_y_name, z=i)[None]

        tmp = deform[0].copy()
        deform = deform[1:]
        deform = deform[::-1]
        deform = np.concatenate([tmp[None], deform], 0)

        named = './VoxelMorph/data/deformations/numpy/' + name.split('/')[-1].split('.tif')[0] + '_fwd.npy'
        np.save(named, deform)

        def_x_name = './VoxelMorph/data/deformations/ics/' + name.split('/')[-1].split('.tif')[0] + '_ffXsh_bcw.ics'
        def_y_name = './VoxelMorph/data/deformations/ics/' + name.split('/')[-1].split('.tif')[0] + '_ffYsh_bcw.ics'
        seq = io.imread(name)
        deform = np.empty(seq.shape + (2,))
        print(deform.shape)
        print(name)
        for i in tqdm(range(len(seq))):
            deform[i, :, :, 0] = bioformats.load_image(def_x_name, z=i)[None]
            deform[i, :, :, 1] = bioformats.load_image(def_y_name, z=i)[None]
        # deform = deform[::-1]
        named = './VoxelMorph/data/deformations/numpy/' + name.split('/')[-1].split('.tif')[0] + '_bcw.npy'
        np.save(named, deform)

    javabridge.kill_vm()