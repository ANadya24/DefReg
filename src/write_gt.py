from skimage import io, color
from glob import glob
from ffRemap import *
from scipy import io as spio


def adjust01(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


def WarpCoords2(poi, V, out_size):
    h = out_size[1]
    w = out_size[2]
    indices = poi[0, :, 0].reshape(-1, 1), poi[0, :, 1].reshape(-1, 1)
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    indices = np.column_stack([np.reshape(x, (-1, 1)), np.reshape(y, (-1, 1))])
    print(x.shape, V[0, :, :, 0].shape, poi[0].shape, indices.shape, 356 * 287)
    vec_x = interpolate.griddata(indices, V[0, :, :, 0].reshape(-1), poi[0], method='linear')
    vec_y = interpolate.griddata(indices, V[0, :, :, 1].reshape(-1), poi[0], method='linear')
    # vec_x = interpolation.map_coordinates(indices, V[0, :, :, 0], order=1, mode='reflect')
    # vec_y = interpolation.map_coordinates(indices, V[0, :, :, 1], order=1, mode='reflect')
    # print(vec_x.min(), vec_x.max(), V.min(), V.max())
    p = np.array(poi)
    p[0, :, 0] += vec_x.squeeze()
    p[0, :, 1] += vec_y.squeeze()
    return p


def WarpCoords(poi, V, out_size):
    num_batch = out_size[0]
    out_height = out_size[1]
    out_width = out_size[2]

    V = np.transpose(V, [0, 3, 1, 2])  # [n, 2, h, w]
    cy = poi[:, :, 1] - np.floor(poi[:, :, 1])
    cx = poi[:, :, 0] - np.floor(poi[:, :, 0])

    idx = np.floor(poi[:, :, 0]).astype('int')
    idy = np.floor(poi[:, :, 1]).astype('int')
    vy00 = np.zeros((num_batch, poi.shape[1]))
    vx00 = np.zeros((num_batch, poi.shape[1]))
    vy01 = np.zeros((num_batch, poi.shape[1]))
    vx01 = np.zeros((num_batch, poi.shape[1]))
    vy10 = np.zeros((num_batch, poi.shape[1]))
    vx10 = np.zeros((num_batch, poi.shape[1]))
    vy11 = np.zeros((num_batch, poi.shape[1]))
    vx11 = np.zeros((num_batch, poi.shape[1]))

    for b in range(num_batch):
        iy = idy[b]
        ix = idx[b]
        vy00[b] = V[b, 1, iy, ix]
        vy01[b] = V[b, 1, iy, ix + 1]
        vy10[b] = V[b, 1, iy + 1, ix]
        vy11[b] = V[b, 1, iy + 1, ix + 1]

        vx00[b] = V[b, 0, iy, ix]
        vx01[b] = V[b, 0, iy, ix + 1]
        vx10[b] = V[b, 0, iy + 1, ix]
        vx11[b] = V[b, 0, iy + 1, ix + 1]

    ys = (vy11 * cx * cy + vy10 * cy * (1 - cx) + vy01 * cx * (1 - cy) + vy00 * (1 - cx) * (
            1 - cy))
    xs = (vx11 * cx * cy + vx10 * cy * (1 - cx) + vx01 * cx * (1 - cy) + vx00 * (1 - cx) * (
            1 - cy))
    p = np.array(poi)

    p[:, :, 0] += xs
    p[:, :, 1] += ys
    p[:, :, 0] = np.clip(p[:, :, 0], 0, out_width - 1)
    p[:, :, 1] = np.clip(p[:, :, 1], 0, out_height - 1)

    return p

DIRECTION = 'FWD'
GT = True

if __name__ == "__main__":

    sequences = glob('./VoxelMorph/data/*.tif')
    print(len(sequences))

    for name in filter(lambda name: name.find('SeqB') != -1, sequences):
        if GT:
            if DIRECTION == 'BCW':
                def_name = './VoxelMorph/data/deformations/numpy/' + name.split('/')[-1].split('.tif')[0] + '_bcw.npy'
            else:
                def_name = './VoxelMorph/data/deformations/numpy/' + name.split('/')[-1].split('.tif')[0] + '_fwd.npy'
        else:
            if DIRECTION == 'BCW':
                def_name = './VoxelMorph/data/registered/result/bcw/deformations/' + name.split('/')[-1].split('.tif')[0] + '.npy'
            else:
                def_name = './VoxelMorph/data/registered/result/fwd/deformations/' + name.split('/')[-1].split('.tif')[0] + '.npy'
        deformations = np.load(def_name)

        # name1 = f'./VoxelMorph/data/viz/{DIRECTION.lower()}/init_' + name.split('/')[-1]
        seq = io.imread(name)
        if seq.shape[-1] == 3:
            seq = (color.rgb2gray(seq) * 255).astype('uint8')
        print(seq.shape, seq.dtype, seq.max())
        new_seq = [seq[0]]
        init_def = None
        print(name)

        for i in range(1, len(seq)):
            print(i)
            im = seq[i].astype('float32')
            h, w = im.shape

            deform = deformations[i]
            if DIRECTION == 'BCW':
                if GT:
                    if i != 1:
                        deform = ff_1_to_k(init_def, deform)
                    init_def = deform
                im_new = backward_warp(im, deform)
            else:
                if GT:
                    if i != 1:
                        deform = ff_1_to_k(init_def, deform)
                    init_def = deform
                im_new = forward_warp(im, deform)

            new_seq.append(im_new)
        if GT:
            if DIRECTION == 'BCW':
                io.imsave(f'./VoxelMorph/data/registered/gt/{DIRECTION.lower()}/' + name.split('/')[-1], np.array(new_seq, 'uint8').squeeze())
            else:
                io.imsave(f'./VoxelMorph/data/registered/gt/{DIRECTION.lower()}/' + name.split('/')[-1], np.array(new_seq, 'uint8').squeeze())
        else:
            if DIRECTION == 'BCW':
                io.imsave(f'./VoxelMorph/data/registered/result/{DIRECTION.lower()}/' + name.split('/')[-1],
                          np.array(new_seq, 'uint8').squeeze())
            else:
                io.imsave(f'./VoxelMorph/data/registered/result/{DIRECTION.lower()}/' + name.split('/')[-1],
                          np.array(new_seq, 'uint8').squeeze())

