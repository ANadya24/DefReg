from skimage import io
import javabridge
import bioformats
import numpy as np
import os
from glob import glob
import imagesize


if __name__ == "__main__":
    javabridge.start_vm(class_path=bioformats.JARS)
    sequences = glob('./VoxelMorph/data/*.tif')
    # with open('sizes.txt', 'w') as f:
    #     for name in sequences:
    #         w, h = imagesize.get(name)
    #         f.write(name.split('/')[-1].split('.tif')[0] + f'\t{h}\t{w}\n')
    print(len(sequences))
    out_dir = './VoxelMorph/data/pairs/'
    os.makedirs(out_dir, exist_ok=True)

    for name in sequences: #filter(lambda name: name.find('Seq') != -1, sequences)
        def_x_name = './VoxelMorph/data/deformations/' + name.split('/')[-1].split('.tif')[0] + '_ffXsh_bcw.ics'
        def_y_name = './VoxelMorph/data/deformations/' + name.split('/')[-1].split('.tif')[0] + '_ffYsh_bcw.ics'
        seq = io.imread(name)
        print(name)
        for i in range(0, len(seq)-1):
            for j in range(i+2, i+7):
            # j= i+1
                im1 = seq[i].astype('float32')[None]
                im2 = seq[j].astype('float32')[None]
                def_x = bioformats.load_image(def_x_name, z=i+1)[None]
                def_y = bioformats.load_image(def_y_name, z=i+1)[None]
                if (j - i) != 1:
                    k = i + 2
                    while k != j:
                        def_x += bioformats.load_image(def_x_name, z=k)[None]
                        def_y += bioformats.load_image(def_y_name, z=k)[None]
                        k += 1

                print(def_y.max(), def_y.min())
                print(im1.shape, im1.dtype, def_x.shape, def_x.dtype)
                res = np.vstack([im1, im2, def_x, def_y])
                print(res.shape, res.dtype)
                if j - i == 1:
                    np.save(name.split('.tif')[0] + f'_{i}', res)
                else:
                    np.save(name.split('.tif')[0] + f'_{i}_diff{j}', res)

    javabridge.kill_vm()