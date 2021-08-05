"""The functions for building the Kinetics400 dataset class for pretraining
Code partially borrowed from
https://github.com/YihengZhang-CV/SeCo-Sequence-Contrastive-Learning/blob/main/dataset/dataset_kinetics_v2.py.

MIT License
Copyright (c) 2020 YihengZhang-CV
"""


import torch.utils.data
import os
import copy
import random
import torch
import numpy as np
from PIL import Image
from .dataset_builder import DATASETS


def set_rng(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@DATASETS.register_module()
class KineticsClipFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root, split='train', **kwargs):
        super(KineticsClipFolderDataset, self).__init__()
        if '##' in root:  # super resource
            data_root_split = root.split('##')
            assert len(data_root_split) == 2
            root = data_root_split[0]
            self.dataset_frame_root_ssd = os.path.join(data_root_split[1], 'data')
            assert '#' not in self.dataset_frame_root_ssd
            assert os.path.exists(self.dataset_frame_root_ssd)
        else:
            self.dataset_frame_root_ssd = None
        # dataset root
        if '#' in root:  # multiple data resources
            self.dataset_root = root.split('#')
        else:
            self.dataset_root = [root]
        for p in self.dataset_root:
            if not os.path.exists(p):
                print(p)
                assert False
        self.dataset_root_num = len(self.dataset_root)
        print('using {} data sources'.format(self.dataset_root_num))
        # data frame root
        self.dataset_frame_root = [os.path.join(p, split) for p in self.dataset_root]
        for p in self.dataset_frame_root:
            assert os.path.exists(p)
        # data list file
        assert split in ('train', 'val')
        self.dataset_list_file = os.path.join(self.dataset_root[0], split + '.txt')
        assert os.path.exists(self.dataset_list_file)
        # load vid samples
        self.samples = self._load_list(self.dataset_list_file)
        self.transform = None

    def _get_aug_frame(self, frame_root, frame_idx):
        frame = Image.open(os.path.join(frame_root, 'frame_{:05d}.jpg'.format(frame_idx)))
        frame.convert('RGB')
        if self.transform is not None:
            frame_aug = self.transform(frame)
        else:
            frame_aug = frame
        return frame_aug

    def _load_list(self, list_root):
        with open(list_root, 'r') as f:
            lines = f.readlines()
        vids = []
        for k, l in enumerate(lines):
            lsp = l.strip().split(' ')
            # path, frame, label
            if self.dataset_frame_root_ssd is not None and os.path.exists(
                    os.path.join(self.dataset_frame_root_ssd, lsp[0])):
                vid_root = os.path.join(self.dataset_frame_root_ssd, lsp[0])
            else:
                vid_root = os.path.join(self.dataset_frame_root[k % self.dataset_root_num], lsp[0])
            vid_root, _ = os.path.splitext(vid_root)
            # use splitetxt twice because there are some video root like: abseiling/9EnSwbXxu5g.mp4.webm
            vid_root, _ = os.path.splitext(vid_root)
            vids.append((vid_root, int(lsp[1]), int(lsp[2])))

        return vids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        raise NotImplementedError


@DATASETS.register_module()
class KineticsClipFolderDatasetMultiFrames(KineticsClipFolderDataset):
    def __init__(self, root, transform=None, split='train', sample_num=0):
        super(KineticsClipFolderDatasetMultiFrames, self).__init__(root, split)
        self.transform = transform
        self.sample_num = sample_num
        assert self.transform is not None

    def __getitem__(self, item):
        frame_root, frame_num, cls = self.samples[item]
        sample_num = frame_num if self.sample_num <= 0 or self.sample_num > frame_num else self.sample_num
        frame_indices = np.round(np.linspace(1, frame_num, num=sample_num)).astype(np.int64)
        frames = torch.cat([self._get_aug_frame(frame_root, frame_indices[i]) for i in range(sample_num)], dim=0)
        return frames, cls


@DATASETS.register_module()
class KineticsClipFolderDatasetOrderTSN(KineticsClipFolderDataset):
    def __init__(self, root, transform=None, split='train_list'):
        super(KineticsClipFolderDatasetOrderTSN, self).__init__(root, split)
        self.transform = transform
        assert self.transform is not None

        self.num_segments = 3

    def __getitem__(self, item):
        frame_root, frame_num, cls = self.samples[item]

        initial_seed = random.randint(0, 2 ** 31)
        set_rng(initial_seed)

        ###### Step 1: TSN samples ######
        # segments (base on num_images_to_return)
        frame_indices = np.round(np.linspace(1, frame_num, num=frame_num)).astype(np.int64)
        segments_length = frame_num // self.num_segments

        segments = []

        for i in range(self.num_segments):
            start_idx = i * segments_length
            if i == self.num_segments - 1:
                segment = frame_indices[start_idx:]
            else:
                end = (i + 1) * segments_length
                segment = frame_indices[start_idx:end]

            segments.append(segment)

        # sample frames from each segments
        key_images = []
        queue_images = []

        # debug
        key_ids = []
        queue_ids = []

        for segment in segments:
            image_path_inds = np.random.choice(segment, 2, replace=False)
            for ii, ind in enumerate(image_path_inds):
                image = self._get_aug_frame(frame_root, ind).unsqueeze(dim=0)
                if ii == 0:
                    key_images.append(image)
                    key_ids.append(ind)
                else:
                    queue_images.append(image)
                    queue_ids.append(ind)

        if len(key_images) < self.num_segments:
            return None

        ###### Step 2: SeCo samples ######
        rand_segment = random.randint(0, 1)
        if rand_segment == 0:
            frame1_aug1 = queue_images[0].squeeze(dim=0)
            frame1_aug2 = self._get_aug_frame(frame_root, queue_ids[0])
            frame2_aug = queue_images[1].squeeze(dim=0)
            frame3_aug = queue_images[2].squeeze(dim=0)
        else:
            frame1_aug1 = queue_images[2].squeeze(dim=0)
            frame1_aug2 = self._get_aug_frame(frame_root, queue_ids[2])
            frame2_aug = queue_images[0].squeeze(dim=0)
            frame3_aug = queue_images[1].squeeze(dim=0)

        ###### Step 3: Order samples ######
        # 4 labels: 0 (00), 1 (10), 2 (01), 3 (11)
        rand_shuffle1 = random.randint(0, 1)
        rand_shuffle2 = random.randint(0, 1)

        if rand_shuffle1:
            queue_images, queue_ids = shuffle_list(queue_images, queue_ids)
            if rand_shuffle2:
                key_images, key_ids = shuffle_list(key_images, key_ids)
                order_label = 3  # label: 11
            else:
                order_label = 1  # label: 10
        else:
            if rand_shuffle2:
                key_images, key_ids = shuffle_list(key_images, key_ids)
                order_label = 2  # label: 01
            else:
                order_label = 0  # label: 00

        # tsn q and k
        tsn_q = torch.cat(queue_images, dim=0)
        tsn_k = torch.cat(key_images, dim=0)

        return frame1_aug1, frame1_aug2, frame2_aug, frame3_aug, order_label, tsn_q, tsn_k


def shuffle_list(l, l_idx):
    l_idx_forward = copy.copy(l_idx)
    l_idx_backward = copy.copy(l_idx)
    l_idx_backward.reverse()

    i = 0
    while True:
        seed = random.randint(0, 2 ** 31)
        set_rng(seed)
        random.shuffle(l)
        set_rng(seed)
        random.shuffle(l_idx)

        # after shuffling, still keep the order
        if l_idx != l_idx_forward and l_idx != l_idx_backward:
            break

    return l, l_idx
