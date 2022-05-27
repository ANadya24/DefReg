import numpy as np
from tqdm import tqdm
import pickle
from glob import glob
from skimage import io, color
import os
DIRECTION = 'FWD'

if __name__ == "__main__":
    sequences = glob('./VoxelMorph/data/*.tif')
    print(sequences)
    test = set([sequences[0], sequences[4]])
    train = set(sequences).difference(test)
    print(train, test)
    out = {}
    for name in tqdm(train):
        def_name = './VoxelMorph/data/deformations/numpy/' + name.split('/')[-1].split('.tif')[0] + '_fwd.npy'
        deformations = np.load(def_name)

        # name1 = f'./VoxelMorph/data/viz/{DIRECTION.lower()}/init_' + name.split('/')[-1]
        seq = io.imread(name)
        if seq.shape[-1] == 3:
            seq = (color.rgb2gray(seq) * 255).astype('uint8')
        print(seq.shape, seq.dtype, seq.max())
        out[name.split('/')[-1]] = {'imseq': seq, 'defs': deformations}
    os.makedirs('./VoxelMorph/dataset/', exist_ok=True)
    with open('./VoxelMorph/dataset/train_set.pkl', 'wb') as file:
        pickle.dump(out, file)

    out = {}
    for name in tqdm(test):
        def_name = './VoxelMorph/data/deformations/numpy/' + name.split('/')[-1].split('.tif')[0] + '_fwd.npy'
        deformations = np.load(def_name)
        # name1 = f'./VoxelMorph/data/viz/{DIRECTION.lower()}/init_' + name.split('/')[-1]
        seq = io.imread(name)
        if seq.shape[-1] == 3:
            seq = (color.rgb2gray(seq) * 255).astype('uint8')
        print(seq.shape, seq.dtype, seq.max())
        out[name.split('/')[-1]] = {'imseq': seq, 'defs': deformations}
    with open('./VoxelMorph/dataset/test_set.pkl', 'wb') as file:
        pickle.dump(out, file)