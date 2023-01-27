import numpy as np
from torchvision.transforms import ToTensor
from torch.utils import data
import torch
from skimage import io, color, filters
from scipy import io as spio
import albumentations as A
from utils.data_process import pad_image, match_histograms, normalize_mean_std, normalize_min_max

MAX_LINE_LEN = 20
np.random.seed(1234)


class SeqDataset(data.Dataset):
    def __init__(self, image_sequences, input_seq_len=3,
                 im_size=(1, 256, 256),
                 train=True, register_limit=5, gauss_sigma=-1,
                 use_masks=True, use_crop=False, multiply_mask=True,
                 ):
        """

        :param: image_sequences:
        :param: image_keypoints:
        :param: im_size:
        :param: train:
        :param: register_limit:
        :param: use_masks:
        :param: use_crop:
        """
        self.image_sequences = []
        self.image_keypoints = []
        self.points_length = []
        self.image_masks = []
        self.use_masks = use_masks
        self.multiply_mask = multiply_mask
        self.use_crop = use_crop
        self.train = train
        self.gauss_sigma = gauss_sigma
        self.im_size = im_size[1:]
        self.input_seq_len = input_seq_len

        for sequence_path in image_sequences:

            seq = io.imread(sequence_path)
            if seq.shape[-1] == 3 and im_size[0] == 1:
                seq = color.rgb2gray(seq)
            self.image_sequences.append(seq)

            if self.use_masks:
                mask_seq = io.imread(sequence_path.replace('.tif', '_mask.tif'))
                if mask_seq.shape[-1] == 3:
                    mask_seq = mask_seq.sum(-1)
                mask_seq = 1. - np.clip(np.array(mask_seq, dtype=np.float32), 0., 1.)
                self.image_masks.append(mask_seq)

        self.seq_numeration = []
        for seq_idx, _ in enumerate(self.image_sequences):
            for i, _ in enumerate(self.image_sequences[seq_idx]):
                self.seq_numeration.append((seq_idx, i))

        self.length = len(self.seq_numeration)

        if isinstance(register_limit, int):
            self.register_limit = [register_limit] * len(self.image_sequences)
        else:
            assert len(register_limit) == len(self.image_sequences), 'limit value must be assigned either \
            by integer or by the list of values for each mage sequence accordingly'

            self.register_limit = register_limit

        if self.train:
            np.random.shuffle(self.seq_numeration)

        if self.train:
            add_target = {}
            for i in range(self.input_seq_len):
                add_target.update({f'image{i}': 'image',
                                   f'mask{i}': 'mask'})
            self.aug_pipe = A.Compose([A.OneOf([A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
                                                A.ShiftScaleRotate(shift_limit=0.0225, scale_limit=0.1,
                                                                   rotate_limit=5)], p=0.2)],
                                      additional_targets=add_target)

        self.to_tensor = ToTensor()

        self.resize = A.Resize(*self.im_size)


    def __len__(self):
        return self.length

    def reset(self):
        np.random.shuffle(self.seq_numeration)

    def __getitem__(self, index):
        seq_idx, it = self.seq_numeration[index]
        current_seq_len = len(self.image_sequences[seq_idx])

        images = [self.image_sequences[seq_idx][it].squeeze().astype(np.float32)]
        if self.use_masks:
            masks = [self.image_masks[seq_idx][it].squeeze()]

        # if not self.train and self.return_points:
        #     points_len = self.points_length[seq_idx]
        #     points = [self.image_keypoints[seq_idx][it]]

        for _ in range(self.input_seq_len - 1):
            if self.train:
                next_it = np.random.randint(max(it - self.register_limit[seq_idx], 0),
                                            min(it + self.register_limit[seq_idx] + 1, current_seq_len - 1))
            else:
                next_it = min(it + 1, current_seq_len - 1)

            images.append(self.image_sequences[seq_idx][next_it].squeeze().astype(np.float32))
            if self.use_masks:
                masks.append(self.image_masks[seq_idx][next_it].squeeze())


            it = next_it
        for i in range(len(images)):
            images[i] = normalize_min_max(images[i])

        for i in range(1, self.input_seq_len):
            images[i-1], images[i], _ = match_histograms(images[i-1], images[i], random_switch=False)

        if self.use_masks:
            for i in range(len(images)):
                images[i] = np.stack([images[i], masks[i]], -1)

        h, w = images[0].shape[:2]

        if self.use_crop:
            x0 = np.random.randint(0, w - self.im_size[1])
            y0 = np.random.randint(0, h - self.im_size[0])
            for i in range(len(images)):
                images[i] = images[i][y0: y0 + self.im_size[0], x0:x0 + self.im_size[1]]
        else:

            if h != w:
                if h < w:
                    pad_params = (0, w - h, 0, 0)
                else:
                    pad_params = (0, 0, 0, h - w)

                if len(images[0].shape) > 2:
                    pad_params = pad_params + (0, 0)

                for i in range(len(images)):
                    images[i] = pad_image(images[i], pad_params)

        for i in range(len(images)):
            resize_dict = {'image': images[i]}
            if self.use_masks:
                resize_dict['mask'] = masks[i]
            data = self.resize(**resize_dict)
            images[i] = data['image'].astype('float32')
            if self.use_masks:
                masks[i] = data['mask']

        if self.train:
            if np.random.rand() < 0.5:
                for i in range(len(images)):
                    images[i] = images[i][:, ::-1]

            if np.random.rand() < 0.5:
                for i in range(len(images)):
                    images[i] = images[i][::-1]

            aug_dict = {'image': images[0]}
            for i in range(1, len(images)):
                aug_dict.update({f'image{i}': images[i].astype(np.float32)})
            data = self.aug_pipe(**aug_dict)
            images[0] = data['image'].astype('float32')
            for i in range(1, len(images)):
                images[i] = data[f'image{i}'].astype('float32')

        if self.gauss_sigma > 0.:
            if self.use_masks and self.multiply_mask:
                for i in range(len(images)):
                    images[i][:, :, 0] = filters.gaussian(images[i][:, :, 0], self.gauss_sigma)
            else:
                for i in range(len(images)):
                    images[i] = filters.gaussian(images[i], self.gauss_sigma)

        if self.use_masks and self.multiply_mask:
            for i in range(len(images)):
                images[i] = images[i][:, :, 0] * images[i][:, :, 1]

        for i in range(len(images)):
            images[i] = self.to_tensor(images[i]).float()
        
        return torch.stack(images, 0)
