from skimage import io, filters
import javabridge
import bioformats
from glob import glob
import pickle
from utils.ffRemap import *
from matplotlib import pyplot as plt


def normalize(im):
    img = im.astype('float32') - im.min()
    img = img * 255 / img.max()
    return img


test_name = "12"  # "12

if __name__ == "__main__":

    javabridge.start_vm(class_path=bioformats.JARS)
    sequences = glob('./VoxelMorph/data/*.tif')

    if test_name == "123":

        print("Do with backward warp...")

        for name in filter(lambda name: name.find('SeqB1') != -1, sequences):
            def_x_name = './VoxelMorph/data/deformations/ics/' + name.split('/')[-1].split('.tif')[0] + '_ffXsh_bcw.ics'
            def_y_name = './VoxelMorph/data/deformations/ics/' + name.split('/')[-1].split('.tif')[0] + '_ffYsh_bcw.ics'
            seq = io.imread(name)
            print(seq.shape, seq.dtype)
            # new_seq = [seq[20]]
            # new_seq2 = [seq[20]]
            new_seq = [filters.gaussian(seq[20], 1.)]
            new_seq2 = [filters.gaussian(seq[20], 1.)]
            init_def = None

            for i in range(20, 0, -1):
                im = (new_seq[0])#.astype('float32')
                im2 = (new_seq2[20 - i])#.astype('float32')
                h, w = im.shape

                def_x = bioformats.load_image(def_x_name, z=i)[None]
                def_y = bioformats.load_image(def_y_name, z=i)[None]
                # print(def_x.min(), def_y.max())

                deform = np.concatenate([def_x.transpose(1, 2, 0), def_y.transpose(1, 2, 0)], axis=-1)
                im_new2 = backward_warp(im2, deform)
                # print(im_new2.max())

                if i != 20:
                    deform = ff_1_to_k(init_def, deform)
                init_def = deform
                print(deform.min(), deform.max())

                im_new = backward_warp(im, deform)
                # print(im_new.max())

                new_seq.append(im_new)
                new_seq2.append(im_new2)

            diff  = np.sqrt(((new_seq[-1].astype('float') - new_seq2[-1].astype('float')) ** 2))

            plt.subplot(1, 2, 1)
            plt.imshow(new_seq[-1], cmap='gray')
            plt.title('Summarized (with \n interpolation) deformation \n application')
            plt.subplot(1, 2, 2)
            plt.imshow(new_seq2[-1], cmap='gray')
            plt.title('Sequential application \n of deformations')
            plt.waitforbuttonpress(0)
            plt.savefig('./VoxelMorph/tmp_results/images_bcw_seq_len20.png')
            plt.figure()
            print(diff.min(), diff.max())
            plt.imshow(diff, cmap='gray')
            plt.colorbar()
            plt.waitforbuttonpress(0)
            plt.savefig('./VoxelMorph/tmp_results/l2_difference_bcw_seq_len20.png')
            plt.close()

        print("Do with forward warp...")

        for name in filter(lambda name: name.find('SeqB1') != -1, sequences):
            def_x_name = './VoxelMorph/data/deformations/ics/' + name.split('/')[-1].split('.tif')[0] + '_ffXsh_fwd.ics'
            def_y_name = './VoxelMorph/data/deformations/ics/' + name.split('/')[-1].split('.tif')[0] + '_ffYsh_fwd.ics'
            seq = io.imread(name)
            print('Sequence shape and type are:', seq.shape, seq.dtype)
            new_seq = [filters.gaussian(seq[0], 1.)]
            new_seq2 = [filters.gaussian(seq[0], 1.)]
            init_def = None

            for i in range(1, 20):
                im = (new_seq[0]).astype('float32')
                # print(im[170:172, 140:142])
                im2 = (new_seq2[i - 1]).astype('float32')
                h, w = im.shape
                print(f'Number {i}/{len(seq)}, slice {len(seq) - i}')
                def_x = bioformats.load_image(def_x_name, z=len(seq) - i)[None]
                def_y = bioformats.load_image(def_y_name, z=len(seq) - i)[None]
                print(f'Def min : {def_x.min()}, max {def_y.max()}')

                deform = np.concatenate([def_x.transpose(1, 2, 0), def_y.transpose(1, 2, 0)], axis=-1)
                im_new2 = forward_warp(im2, deform)
                # print(f'Sequential approach: im max {im_new2.max()}')

                # print(deform[170:172, 140:142, 0])
                # print(deform[170:172, 140:142, 1])
                if i != 1:
                    deform = ff_1_to_k(init_def, deform)
                # print(deform[170:172, 140:142, 0])
                # print(deform[170:172, 140:142, 1])
                init_def = deform.copy()

                # x, y = np.meshgrid(np.arange(140, 142), np.arange(170, 172))
                # xx = x + np.floor(deform[170:172, 140:142, 0]).astype('int')
                # yy = y + np.floor(deform[170:172, 140:142, 1]).astype('int')
                # print( deform[170:172, 140:142, 0] - np.floor(deform[170:172, 140:142, 0]),
                #        deform[170:172, 140:142, 1] - np.floor(deform[170:172, 140:142, 1]))
                # print(im[xx.reshape(-1, 1), yy.reshape(-1,1)].reshape(2,2))
                # xxx = x + np.ceil(deform[170:172, 140:142, 0]).astype('int')
                # yyy = y + np.ceil(deform[170:172, 140:142, 1]).astype('int')
                # print(xx, yy)
                # print(im[xxx.reshape(-1, 1), yyy.reshape(-1, 1)].reshape(2, 2))
                # print(im[xx.reshape(-1, 1), yyy.reshape(-1, 1)].reshape(2, 2))
                # print(im[xxx.reshape(-1, 1), yy.reshape(-1, 1)].reshape(2, 2))

                im_new = forward_warp(im, deform)
                # print(im_new[170:172, 140:142])
                # print(f' Sum approach: im max {im_new.max()}')

                new_seq.append(im_new)
                new_seq2.append(im_new2)

            diff = abs(new_seq[-1].astype('float') - new_seq2[-1].astype('float'))

            plt.subplot(1, 2, 1)
            plt.imshow(new_seq[-1], cmap='gray')
            plt.title('Summarized (with \n interpolation) deformation application')
            plt.subplot(1, 2, 2)
            plt.imshow(new_seq2[-1], cmap='gray')
            plt.title('Sequential application \n of deformations')
            plt.waitforbuttonpress(0)
            plt.savefig('/home/nadya/Projects/VoxelMorph/tmp_results/images_fwd_seq_len20.png')
            plt.figure()
            print(f' Difference min is {diff.min()}, max is {diff.max()}')
            plt.imshow(diff, cmap='gray')
            plt.colorbar()
            plt.waitforbuttonpress(0)
            plt.savefig('/home/nadya/Projects/VoxelMorph/tmp_results/abs_difference_fwd_seq_len20.png')
            plt.close()

    elif test_name == "12":
        print("Do with backward warp...")

        for name in filter(lambda name: name.find('Series009') != -1, sequences):
            def_x_name = './VoxelMorph/data/deformations/ics/' + name.split('/')[-1].split('.tif')[0] + '_ffXsh_bcw.ics'
            def_y_name = './VoxelMorph/data/deformations/ics/' + name.split('/')[-1].split('.tif')[0] + '_ffYsh_bcw.ics'
            seq = io.imread(name)
            print(seq.shape, seq.dtype)
            im1 = (seq[2])
            im2 = (seq[3])
            print(im1.max(), im2.max())

            def_x = bioformats.load_image(def_x_name, z=3)[None]
            def_y = bioformats.load_image(def_y_name, z=3)[None]
            print(def_x.min(), def_y.max())

            deform = np.concatenate([def_x.transpose(1, 2, 0), def_y.transpose(1, 2, 0)], axis=-1)
            im_new2 = backward_warp(im2, deform)
            print(im1.max(), im_new2.max())

            diff = abs(im_new2.astype('float') - im1.astype('float'))

            plt.subplot(1, 2, 1)
            plt.imshow(im1, cmap='gray')
            plt.subplot(1, 2, 2)
            plt.imshow(im_new2, cmap='gray')
            plt.waitforbuttonpress(0)
            plt.figure()
            print(diff.min(), diff.max())
            plt.imshow(diff, cmap='gray')
            plt.waitforbuttonpress(0)
            plt.close()
    elif test_name == "test_Seq":

        print("Do with backward warp...")

        name = './VoxelMorph/test_seq/testSeq_bcw.tiff'
        def_name = './VoxelMorph/test_seq/testSeq_bcw'
        with open(def_name, 'rb') as f:
            deformations = pickle.load(f)
        seq = io.imread(name)
        print('Sequence shape and type are:', seq.shape, seq.dtype)
        new_seq = [seq[9]]
        new_seq2 = [seq[9]]
        init_def = None

        for i in range(9, 0, -1):
            im = (new_seq[0])  # .astype('float32')
            im2 = (new_seq2[9 - i])  # .astype('float32')
            h, w = im.shape

            deform = deformations[i]
            im_new2 = backward_warp(im2, deform)
            # print(im_new2.max())

            if i != 9:
                deform = ff_1_to_k(init_def, deform)
            init_def = deform
            print(deform.min(), deform.max())

            im_new = backward_warp(im, deform)
            # print(im_new.max())

            new_seq.append(im_new)
            new_seq2.append(im_new2)

        diff = np.sqrt(((new_seq[-1].astype('float') - new_seq2[-1].astype('float')) ** 2))

        plt.subplot(1, 2, 1)
        plt.imshow(new_seq[-1], cmap='gray')
        plt.title('Summarized (with \n interpolation) deformation \n application')
        plt.subplot(1, 2, 2)
        plt.imshow(new_seq2[-1], cmap='gray')
        plt.title('Sequential application \n of deformations')
        plt.waitforbuttonpress(0)
        plt.savefig('./VoxelMorph/tmp_results/images_bcw_test_seq_len10.png')
        plt.figure()
        print(diff.min(), diff.max())
        plt.imshow(diff, cmap='gray')
        plt.colorbar()
        plt.waitforbuttonpress(0)
        plt.savefig('./VoxelMorph/tmp_results/l2_difference_bcw_test_seq_len10.png')
        plt.close()

        print("Do with forward warp...")

        name = './VoxelMorph/test_seq/testSeq_fwd.tiff'
        def_name =  './VoxelMorph/test_seq/testSeq_fwd'
        with open(def_name, 'rb') as f:
            deformations = pickle.load(f)
        seq = io.imread(name)
        print('Sequence shape and type are:', seq.shape, seq.dtype)
        new_seq = [seq[0]]
        new_seq2 = [seq[0]]
        init_def = None

        for i in range(1, 10):
            im = (new_seq[0]).astype('float32')
            # print(im[170:172, 140:142])
            im2 = (new_seq2[i - 1]).astype('float32')
            h, w = im.shape
            print(f'Number {i}/{len(seq)}, slice {len(seq) - i}')
            deform = deformations[i]
            im_new2 = forward_warp(im2, deform)
            # print(f'Sequential approach: im max {im_new2.max()}')

            # print(deform[170:172, 140:142, 0])
            # print(deform[170:172, 140:142, 1])
            if i != 1:
                deform = ff_1_to_k(init_def, deform)
            # print(deform[170:172, 140:142, 0])
            # print(deform[170:172, 140:142, 1])
            init_def = deform.copy()

            im_new = forward_warp(im, deform)
            # print(im_new[170:172, 140:142])
            # print(f' Sum approach: im max {im_new.max()}')

            new_seq.append(im_new)
            new_seq2.append(im_new2)

        diff = abs(new_seq[-1].astype('float') - new_seq2[-1].astype('float'))

        plt.subplot(1, 2, 1)
        plt.imshow(new_seq[-1], cmap='gray')
        plt.title('Summarized (with \n interpolation) deformation application')
        plt.subplot(1, 2, 2)
        plt.imshow(new_seq2[-1], cmap='gray')
        plt.title('Sequential application \n of deformations')
        plt.waitforbuttonpress(0)
        plt.savefig('/home/nadya/Projects/VoxelMorph/tmp_results/images_fwd_test_seq_len10.png')
        plt.figure()
        print(f' Difference min is {diff.min()}, max is {diff.max()}')
        plt.imshow(diff, cmap='gray')
        plt.colorbar()
        plt.waitforbuttonpress(0)
        plt.savefig('/home/nadya/Projects/VoxelMorph/tmp_results/abs_difference_fwd_test_seq_len10.png')
        plt.close()
    javabridge.kill_vm()


