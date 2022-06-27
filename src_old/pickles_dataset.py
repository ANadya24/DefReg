import torch
from torchvision.transforms import ToTensor
from torch.utils import data
from skimage.transform import resize
import pickle
import cv2
from utils.ffRemap import *
from utils.data_process import pad_image, normalize_min_max


class Dataset(data.Dataset):
    def __init__(self, path, im_size=(1, 256, 256), smooth=False, train=True, size_file=None, shuffle=False,
                 use_masks=True, use_crop=False, use_mul=True, use_extra=False):
        """Initialization"""
        with open(path, 'rb') as file:
            self.data = pickle.load(file)
            print(self.data.keys())
        if use_masks:
            with open(path.split('.pkl')[0] + '_body.pkl', 'rb') as file:
                self.masks = pickle.load(file)
                print(self.masks.keys())

        self.length = sum([len(self.data[it]['imseq']) for it in self.data])
        print('Dataset length is ', self.length)
        self.seq = []

        for d in self.data.keys():
            for i, im in enumerate(self.data[d]['imseq']):
                self.seq.append((d, i))

        # self.seq = self.seq[:5]
        # self.length = len(self.seq)
        self.use_masks = use_masks
        self.use_crop = use_crop
        self.use_mul = use_mul
        self.use_extra = use_extra
        
        self.im_size = im_size[1:]
        self.smooth = smooth
        self.train = train
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.seq)

        # if self.train:
        #     self.aug_pipe = A.Compose([A.HorizontalFlip(p=1),# A.VerticalFlip(p=0.3),
        #                           #A.ShiftScaleRotate(shift_limit=0.0225, scale_limit=0.1, rotate_limit=15, p=0.2)
        #                           ], additional_targets={'image2': 'image', 'keypoints2': 'keypoints'},
        #                               keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        if size_file is None:
            self.shape = im_size
        else:
            self.shape = {}
            with open(size_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                vals = line.split('\t')
                name = vals[0]
                shape = tuple(map(int, vals[1:]))
                self.shape[name] = shape

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        seqID, it = self.seq[index]
        endSeq = len(self.data[seqID]['imseq']) - 1
        if self.train:
            if seqID.find('Series') >= 0:
                inter = np.arange(max(it - 7, 0), min(it + 8, endSeq))
                inter = np.delete(inter, [5, 6, 7, 8, 9])
                it2 = np.random.choice(inter)
                if it2 > endSeq:
                    it2 -= 2
            else:
                it2 = np.random.choice(np.arange(max(it-5, 0), min(it+6, endSeq)))
#                it2 = it+1
                if it2 > endSeq:
                    it2 -= 2
        else:
            it2 = min(it + 1, endSeq)

        if it2 < it:
            tmp = it
            it = it2
            it2 = tmp

        to_tensor = ToTensor()
        if isinstance(self.shape, dict):
            h, w = self.shape[seqID.split('/')[-1].split('.')[0]]
        else:
            h, w = self.shape

        # Load data and get label
        fixed_image = self.data[seqID]['imseq'][it].astype('uint8')
        if self.use_crop:
            x0 = np.random.randint(0, w - self.im_size[1])
            y0 = np.random.randint(0, h - self.im_size[0])
            fixed_image = fixed_image[y0: y0 + self.im_size[0], x0:x0 + self.im_size[1]]
        else:
            if h < w:
                fixed_image = pad_image(fixed_image, (0, w - h, 0, 0))
            else:
                fixed_image = pad_image(fixed_image, (0, 0, 0, h - w))
            fixed_image = resize(fixed_image, self.im_size)

        moving_image = self.data[seqID]['imseq'][it2].astype('uint8')
        if self.use_crop:
            moving_image = moving_image[y0: y0 + self.im_size[0], x0:x0 + self.im_size[1]]
        else:
            if h < w:
                moving_image = pad_image(moving_image, (0, w - h, 0, 0))
            else:
                moving_image = pad_image(moving_image, (0, 0, 0, h - w))
            moving_image = resize(moving_image, self.im_size)

        if self.use_masks:
            fixed_mask = self.masks[seqID]['imseq'][it].astype('uint8')
            if self.use_crop:
                fixed_mask = fixed_mask[y0: y0 + self.im_size[0], x0:x0 + self.im_size[1]]
            else:
                if h < w:
                    fixed_mask = pad_image(fixed_mask, (0, w - h, 0, 0))
                else:
                    fixed_mask = pad_image(fixed_mask, (0, 0, 0, h - w))

                fixed_mask = resize(fixed_mask, self.im_size, 0)

            moving_mask = self.masks[seqID]['imseq'][it2].astype('uint8')
            if self.use_crop:
                moving_mask = moving_mask[y0: y0 + self.im_size[0], x0:x0 + self.im_size[1]]
            else:
                if h < w:
                    moving_mask = pad_image(moving_mask, (0, w - h, 0, 0))
                else:
                    moving_mask = pad_image(moving_mask, (0, 0, 0, h - w))

                moving_mask = resize(moving_mask, self.im_size, 0)

        if it == it2:
            deformation = self.data[seqID]['defs'][0]
        else:
            deformation = self.data[seqID]['defs'][it + 1]
            for d in range(it + 2, it2 + 1):
                tmp = self.data[seqID]['defs'][d]
                deformation = ff_1_to_k(deformation, tmp)
        if self.use_crop:
            deformation = deformation[y0: y0 + self.im_size[0], x0:x0 + self.im_size[1]]
        else:
            if h < w:
                deformation = pad_image(deformation, (0, w - h, 0, 0, 0, 0))
            else:
                deformation = pad_image(deformation, (0, 0, 0, h - w, 0, 0))

            deformation = resize(deformation, self.im_size)

        if self.train:
            # hh, ww = deformation.shape[:2]
            # x, y = np.meshgrid(np.arange(ww), np.arange(hh))
            # x_shape, y_shape = x.shape, y.shape
            # x_grid = x + deformation[:, :, 0]
            # y_grid = y + deformation[:, :, 1]
            # indices = np.column_stack([np.reshape(x_grid, (-1, 1)),
            #                            np.reshape(y_grid, (-1, 1))])
            # grid_indices = np.column_stack([np.reshape(x, (-1, 1)),
            #                                 np.reshape(y, (-1, 1))])
            # data = {"image":fixed_image, "image2": moving_image, "keypoints": indices.reshape(-1, 2),
            #         "keypoints2": grid_indices.reshape(-1, 2)}
            # augmented = self.aug_pipe(**data)
            # fixed_image, moving_image, deformation, grid = augmented["image"], augmented["image2"],\
            #                                                np.array(augmented["keypoints"]), \
            #                                                np.array(augmented["keypoints2"])
            #
            # x_grid = deformation[:, 0].reshape(x_shape) - grid[:, 0].reshape(x_shape)
            # y_grid = deformation[:, 1].reshape(y_shape) - grid[:, 1].reshape(y_shape)
            # deformation = np.concatenate([x_grid[:,:, None], y_grid[:,:, None]], axis=-1)
            ############ROTATION############
#            if np.random.rand() < 0.4:
#                angle = np.random.randint(-20, 20)
#                # print(angle)
#                # scale = np.random.randint(50, 64) / max_d
#                # ang = 360 * np.random.rand()
#                # tx = np.random.randint(-32, 33)
#                # ty = np.random.randint(-32, 33)
#                h, w = moving_image.shape[:2]
#                M0 = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.)
#                t = np.array([0, 0, 1])
#                M0 = np.vstack([M0, t])
#                # M1 = np.float32([[1, 0, w // 2],
#                #                  [0, 1, h // 2], [0, 0, 1]])
#                M = M0
#                # M = M0
#                moving_image = cv2.warpAffine(moving_image, M[:2], (w, h))
#                fixed_image = cv2.warpAffine(fixed_image, M[:2], (w, h))
#                if self.use_masks:
#                    moving_mask = cv2.warpAffine(moving_mask, M[:2], (w, h))
#                    fixed_mask = cv2.warpAffine(fixed_mask, M[:2], (w, h))

            if np.random.rand() < 0.5:
                # print('a')
                fixed_image = fixed_image[:, ::-1].copy()
                moving_image = moving_image[:, ::-1].copy()
                if self.use_masks:
                    fixed_mask = fixed_mask[:, ::-1].copy()
                    moving_mask = moving_mask[:, ::-1].copy()
                deformation = deformation[:, ::-1].copy()
                deformation[:, :, 0] *= -1
                # print(fixed_image.shape, moving_image.shape, deformation.shape)

            if np.random.rand() < 0.5:
                # print('b')
                fixed_image = fixed_image[::-1].copy()
                moving_image = moving_image[::-1].copy()
                if self.use_masks:
                    fixed_mask = fixed_mask[::-1].copy()
                    moving_mask = moving_mask[::-1].copy()
                deformation = deformation[::-1].copy()
                deformation[:, :, 1] *= -1

            if np.random.rand() < 0.05:
                moving_image = fixed_image.copy()
                if self.use_masks:
                    moving_mask = fixed_mask.copy()
                deformation = deformation * 0.
        if self.smooth:
            moving_image = cv2.medianBlur(np.uint8(255*normalize_min_max(moving_image)), 5)
            fixed_image = cv2.medianBlur(np.uint8(255*normalize_min_max(fixed_image)), 5)
        if self.use_extra:
            extra_info = abs(fixed_image - moving_image)
            extra_info = to_tensor(extra_info)
        moving_image = to_tensor(moving_image).float()
        deformation = torch.Tensor(deformation).permute((2, 0, 1))
        fixed_image = to_tensor(fixed_image).float() 
        if self.use_masks:
            fixed_mask = torch.Tensor(fixed_mask > 0).float()[None]
            moving_mask = torch.Tensor(moving_mask > 0).float()[None]
            
            if self.use_mul:
                fixed_image = fixed_image * fixed_mask
                moving_image = moving_image * moving_mask
            else:
                fixed_image = torch.cat([fixed_image, fixed_mask], dim=0)
                moving_image = torch.cat([moving_image, moving_mask], dim=0)

        if self.use_extra:
            fixed_image = torch.cat([fixed_image, extra_info], dim=0)

        return fixed_image, moving_image, deformation


if __name__ == '__main__':
    path = '/data/sim/Notebooks/VM/dataset/train_set.pkl'
    dataset = Dataset(path, (1, 256, 256), size_file='sizes.txt',
                      smooth=True, train=True, shuffle=True, use_masks=True, use_mul=False)
    fixed, moving, deform = dataset[0]
    print(deform.min(), deform.max())
    from basic_nets.spatial_transform import SpatialTransformation

    SP = SpatialTransformation()
    print(deform.shape, moving.shape)
#    plt.imshow(np.concatenate([deform[0], deform[1]], axis=1), cmap='gray')
#    plt.waitforbuttonpress()
    cv2.imwrite('test0.jpg', np.concatenate([deform[0], deform[1]], axis=1))
    movingN = SP(moving[None, :, :, :], deform[None])
    movingN = np.uint8(movingN.numpy() * 255).squeeze()
    print(movingN.shape)
    fixed = np.uint8(fixed.numpy().squeeze() * 255)
    moving = np.uint8(moving.numpy().squeeze() * 255)

    print(fixed.shape, moving.shape, deform.shape)
    print(fixed.max())
#    plt.imshow(np.stack([fixed, moving, np.zeros(fixed.shape, dtype='int')], axis=-1), cmap='gray')
#    plt.waitforbuttonpress()
    cv2.imwrite('test1.jpg', np.stack([fixed[0], moving[0], np.zeros(fixed[0].shape, dtype='int')], axis=-1))
    cv2.imwrite('masktest1.jpg', np.stack([fixed[1], moving[1], np.zeros(fixed[1].shape, dtype='int')], axis=-1))

#    plt.figure()
#    plt.imshow(np.stack([fixed, movingN, np.zeros(fixed.shape, dtype='int')], axis=-1), cmap='gray')
#    plt.waitforbuttonpress()
#    plt.close()

    cv2.imwrite('test2.jpg', np.stack([fixed[0], movingN[0], np.zeros(fixed[0].shape, dtype='int')], axis=-1))
    cv2.imwrite('masktest2.jpg', np.stack([fixed[1], movingN[1], np.zeros(fixed[1].shape, dtype='int')], axis=-1))