from skimage import io
import os
from glob import glob
import shutil


def revert_seq(seqname, postfix='_bcw'):
    seq = io.imread(seqname)
    print(seq.shape, seq.dtype)
    seq = seq[::-1]
    lexs = seqname.split('.')
    name = lexs[0] + postfix + '.' + lexs[1]
    io.imsave(name, seq)
    print(name)
    return name


if __name__ == '__main__':
    path = '/home/nadya/Projects/VoxelMorph/data/masks/'
    new_path = path + 'backward seqs/'
    os.makedirs(new_path, exist_ok=True)

    for seqname in glob(path + '*.tif'):
        name = revert_seq(seqname)
        shutil.move(name, new_path+name.split('/')[-1])