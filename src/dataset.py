import numpy as np
import torch
from torchvision.transforms import ToTensor
from torch.utils import data
from skimage import io, color
from scipy import io as spio
import albumentations as A
from utils.data_process import pad_image, match_histograms, normalize_mean_std, normalize_min_max

MAX_LINE_LEN = 1300


class Dataset(data.Dataset):
    def __init__(self, image_sequences, image_keypoints,
                 im_size=(1, 256, 256),
                 train=True, register_limit=5,
                 use_masks=True, use_crop=False, multiply_mask=True):
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
        self.im_size = im_size[1:]

        for sequence_path, keypoint_path in zip(image_sequences, image_keypoints):
            if keypoint_path == '':
                assert self.train is True
            else:
                assert sequence_path.split('/')[-1].split('.')[0] == \
                       keypoint_path.split('/')[-1].split('.')[0], 'Keypoint and sequence files must be ordered!'

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

            if keypoint_path == '' or self.train:
                continue
                # self.image_keypoints.append({'inner': np.zeros((len(seq), 4, 2)),
                #                              'bound': np.zeros((len(seq), 8, 2)),
                #                              'lines': (np.zeros((len(seq), MAX_LINE_LEN, 2)), [1, 1, 1, 1])})
            else:
                poi = spio.loadmat(keypoint_path)
                bound = np.stack(poi['spotsB'][0].squeeze())
                inner = np.stack(poi['spotsI'][0].squeeze())

                bound = bound[:, :, :2]
                inner = inner[:, :, :2]

                line1 = np.stack(poi['lines'][:, 0])
                line2 = np.stack(poi['lines'][:, 1])
                line3 = np.stack(poi['lines'][:, 2])
                line4 = np.stack(poi['lines'][:, 3])

                len1 = len(line1[0])
                len2 = len(line2[0])
                len3 = len(line3[0])
                len4 = len(line4[0])

                lines = np.concatenate((line1, line2, line3, line4), axis=1)
                lines = np.pad(lines, np.array([0, 0, 0, MAX_LINE_LEN - lines.shape[1], 0, 0]).reshape(-1, 2))
                lines_lengths = [len1, len2, len3, len4]
                
                inner_len = len(inner[0])
                bound_len = len(bound[0])
                points_len = np.array([inner_len, bound_len, *lines_lengths])
                points = np.concatenate([inner, bound, lines], axis=1)
                self.image_keypoints.append(points)
                self.points_length.append(points_len)
                
                # self.image_keypoints.append({'inner': inner, 'bound': bound, 'lines': (lines, lines_lengths)})

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
            self.aug_pipe = A.Compose([A.OneOf([A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
                                                # A.RandomResizedCrop(self.im_size[1], self.im_size[0]),
                                                A.ShiftScaleRotate(shift_limit=0.0225, scale_limit=0.1,
                                                                   rotate_limit=5)], p=0.2)],
                                      additional_targets={'image2': 'image',# 'keypoints2': 'keypoints',
                                                          'mask2': 'mask'},)
                                      #keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        self.to_tensor = ToTensor()
        
        if self.train:
            self.resize = A.Resize(*self.im_size)
        else:                    
            self.resize = A.Compose([A.Resize(*self.im_size)],
                                    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        seq_idx, it = self.seq_numeration[index]
        current_seq_len = len(self.image_sequences[seq_idx])

        if self.train:
            it2 = np.random.randint(max(it - self.register_limit[seq_idx], 0),
                                    min(it + self.register_limit[seq_idx] + 1, current_seq_len - 1))
        else:
            it2 = min(it + 1, current_seq_len - 1)

        image1 = self.image_sequences[seq_idx][it].squeeze().astype(np.float32)
        image2 = self.image_sequences[seq_idx][it2].squeeze().astype(np.float32)

        # image1 = normalize_mean_std(image1)
        # image2 = normalize_mean_std(image2)
        image1 = normalize_min_max(image1)
        image2 = normalize_min_max(image2)

        image1, image2, swap_flag = match_histograms(image1, image2, random_switch=self.train)
        if swap_flag:
            tmp = it
            it = it2
            it2 = tmp

        if self.use_masks:
            mask1 = self.image_masks[seq_idx][it].squeeze()
            mask2 = self.image_masks[seq_idx][it2].squeeze()
            image1 = np.stack([image1, mask1], -1)
            image2 = np.stack([image2, mask2], -1)
            
        if not self.train:
            points_len = self.points_length[seq_idx]
            points1 = self.image_keypoints[seq_idx][it]
            points2 = self.image_keypoints[seq_idx][it2]
            
        h, w = image1.shape[:2]

        if self.use_crop:
            x0 = np.random.randint(0, w - self.im_size[1])
            y0 = np.random.randint(0, h - self.im_size[0])
            image1 = image1[y0: y0 + self.im_size[0], x0:x0 + self.im_size[1]]
            image2 = image2[y0: y0 + self.im_size[0], x0:x0 + self.im_size[1]]
            # if self.use_masks:
            #     mask1 = mask1[y0: y0 + self.im_size[0], x0:x0 + self.im_size[1]]
            #     mask2 = mask2[y0: y0 + self.im_size[0], x0:x0 + self.im_size[1]]
            if not self.train:
                points1 -= np.array([x0, y0]).reshape(1, 2)
                points2 -= np.array([x0, y0]).reshape(1, 2)

        else:

            if h != w:
                if h < w:
                    pad_params = (0, w-h, 0, 0)
                else:
                    pad_params = (0, 0, 0, h - w)
                    
                if len(image1.shape) > 2:
                    pad_params = pad_params + (0,0)
                    
                image1 = pad_image(image1, pad_params)
                image2 = pad_image(image2, pad_params)
                # if self.use_masks:
                #     mask1 = pad_image(mask1, pad_params)
                #     mask2 = pad_image(mask2, pad_params)
               
        resize_dict = {'image': image1}
        if not self.train:
            resize_dict['keypoints'] = points1
        if self.use_masks:
            resize_dict['mask'] = mask1
        data1 = self.resize(**resize_dict)

        image1 = data1['image']
        if not self.train:
            points1 = np.array(data1['keypoints'], dtype=np.float32)
            points1 = np.clip(points1, 0., self.im_size[1] - 1)
        # if self.use_masks:
        #     mask1 = data1['mask']

        resize_dict = {'image': image2}
        if not self.train:
            resize_dict['keypoints'] = points2
        # if self.use_masks:
        #     resize_dict['mask'] = mask2
        data2 = self.resize(**resize_dict)

        image2 = data2['image']
        if not self.train:
            points2 = np.array(data2['keypoints'], dtype=np.float32)
            points2 = np.clip(points2, 0., self.im_size[1] - 1)
        # if self.use_masks:
        #     mask2 = data2['mask']

        if self.train:
            h, w = self.im_size
            if np.random.rand() < 0.5:
                image1 = image1[:, ::-1].copy()
                image2 = image2[:, ::-1].copy()
                # if self.use_masks:
                #     mask1 = mask1[:, ::-1].copy()
                #     mask2 = mask2[:, ::-1].copy()
                # points1[:, 0] = w - 1 - points1[:, 0]
                # points2[:, 0] = w - 1 - points2[:, 0]

            if np.random.rand() < 0.5:
                image1 = image1[::-1].copy()
                image2 = image2[::-1].copy()
                # if self.use_masks:
                #     mask1 = mask1[::-1].copy()
                #     mask2 = mask2[::-1].copy()
                # points1[:, 1] = h - 1 - points1[:, 1]
                # points2[:, 1] = h - 1 - points2[:, 1]
            # if self.use_masks:
            #     data = self.aug_pipe(image=image1.astype(np.float32),
            #                          mask=mask1.astype(np.float32),  # keypoints=points1,
            #                          image2=image2.astype(np.float32),
            #                          mask2=mask2.astype(np.float32), )  # keypoints2=points2)
            #     image1, mask1 = data['image'], data['mask']
            #     image2, mask2 = data['image2'], data['mask2']
            # else:
            data = self.aug_pipe(image=image1.astype(np.float32),  # keypoints=points1,
                                 image2=image2.astype(np.float32), )  # keypoints2=points2)
            image1 = data['image']
            image2 = data['image2']
        if self.use_masks and self.multiply_mask:
            image1 = image1[:,:,0] * image1[:,:,1]
            image2 = image2[:,:,0] * image2[:,:,1]

        image1 = self.to_tensor(image1).float()
        image2 = self.to_tensor(image2).float()

        # if self.use_masks:
        #     image1 = torch.cat([image1, torch.Tensor(mask1).float()[None]], 0)
        #     image2 = torch.cat([image2, torch.Tensor(mask2).float()[None]], 0)

        if self.train:
            return image1, image2

        return image1, image2, points1, points2, points_len.astype(np.int32)
