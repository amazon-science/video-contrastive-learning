"""The functions for defining the dataset class of downstream tasks, such as UCF101, HMDB51, etc.
Code partially borrowed from
https://github.com/YihengZhang-CV/SeCo-Sequence-Contrastive-Learning/blob/main/dataset/dataset_kinetics_v2.py.

MIT License
Copyright (c) 2020 YihengZhang-CV
"""

import torch.utils.data
import os
import random
import math
import torch
import numpy as np
from PIL import Image
from .dataset_builder import DATASETS


@DATASETS.register_module()
class DownstreamClipFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root, split='split_1', mode='train', data_source='ucf',**kwargs):
        super(DownstreamClipFolderDataset, self).__init__()

        self.dataset_root = root
        # data frame root
        self.dataset_frame_root = os.path.join(self.dataset_root, 'rawframes')
        assert os.path.exists(self.dataset_frame_root)

        # data list file
        assert mode in ('train', 'val', 'test')
        assert split in ('split_1', 'split_2', 'split_3')
        self.data_source = data_source

        if data_source == 'ucf':
            self.dataset_list_file = os.path.join(self.dataset_root, 'ucfTrainTestlist',
                                                  'ucf101_' + mode + '_' + split + '_rawframes' + '.txt')
        elif data_source == 'hmdb':
            self.dataset_list_file = os.path.join(self.dataset_root, 'testTrainMulti_7030_splits',
                                                  'hmdb51_' + mode + '_' + split + '_rawframes' + '.txt')
        elif data_source == 'sthv2':
            self.dataset_list_file = os.path.join(self.dataset_root,
                                                  'sthv2_' + mode + '_list' + '_rawframes' + '.txt')
        elif data_source == 'anet':
            self.dataset_list_file = os.path.join(self.dataset_root,
                                                  'anet_' + mode + '_video' + '.txt')
        assert os.path.exists(self.dataset_list_file)

        # load vid samples
        self.samples = self._load_list(self.dataset_list_file)
        self.transform = None

    def _load_list(self, list_root):
        with open(list_root, 'r') as f:
            lines = f.readlines()
        vids = []
        for k, l in enumerate(lines):
            lsp = l.strip().split(' ')
            # path, frame, label
            vid_root = os.path.join(self.dataset_frame_root, lsp[0])
            vid_root, _ = os.path.splitext(vid_root)
            # use splitetxt twice because there are some video root like: abseiling/9EnSwbXxu5g.mp4.webm
            vid_root, _ = os.path.splitext(vid_root)
            vids.append((vid_root, int(lsp[1]), int(lsp[2])))

        return vids

    def _get_aug_frame(self, frame_root, frame_idx):
        frame = Image.open(os.path.join(frame_root, 'img_{:05d}.jpg'.format(frame_idx)))
        frame.convert('RGB')
        if self.transform is not None:
            frame_aug = self.transform(frame)
        else:
            frame_aug = frame
        return frame_aug

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        raise NotImplementedError


@DATASETS.register_module()
class DownstreamClipFolderDatasetTSNMultiFrames(DownstreamClipFolderDataset):
    def __init__(self, root, transform=None, split='split_1',
                 mode='train', data_source='ucf', sample_num=0):
        super(DownstreamClipFolderDatasetTSNMultiFrames, self).__init__(root, split, mode, data_source)
        self.transform = transform
        self.sample_num = sample_num
        assert self.transform is not None

    def __getitem__(self, item):
        frame_root, frame_num, cls = self.samples[item]
        sample_num = 3 if self.sample_num <= 0 or self.sample_num > frame_num else self.sample_num

        frame_indices = np.round(np.linspace(1, frame_num, num=frame_num)).astype(np.int64)
        if self.data_source == 'anet':
            frame_indices = np.round(np.linspace(0, frame_num-1, num=frame_num)).astype(np.int64)

        segments_length = frame_num // sample_num
        segments = []

        for i in range(sample_num):
            start_idx = i * segments_length
            if i == sample_num - 1:
                segment = frame_indices[start_idx:]
            else:
                end = (i + 1) * segments_length
                segment = frame_indices[start_idx:end]

            segments.append(segment)

        images = []
        images_ids = []

        for segment in segments:
            image_path_ind = np.random.choice(segment, 1)[0]
            image = self._get_aug_frame(frame_root, image_path_ind).unsqueeze(dim=0)
            images.append(image)
            images_ids.append(image_path_ind)

        if len(images) < sample_num:
            return None

        clips = torch.cat(images, dim=0)

        return clips, cls


@DATASETS.register_module()
class DownstreamDatasetMultiFrames(DownstreamClipFolderDataset):
    def __init__(self, root, transform=None, split='train',
                 mode='train', data_source='ucf', sample_num=0):
        super(DownstreamDatasetMultiFrames, self).__init__(root, split, mode, data_source)
        self.transform = transform
        self.sample_num = sample_num
        assert self.transform is not None

    def __getitem__(self, item):
        frame_root, frame_num, cls = self.samples[item]
        sample_num = frame_num if self.sample_num <= 0 or self.sample_num > frame_num else self.sample_num

        frame_indices = np.round(np.linspace(1, frame_num, num=sample_num)).astype(np.int64)
        if self.data_source == 'anet':
            frame_indices = np.round(np.linspace(0, frame_num-1, num=frame_num)).astype(np.int64)

        frames = torch.cat([self._get_aug_frame(frame_root, frame_indices[i]).unsqueeze(dim=0) for i in range(sample_num)], dim=0)
        return frames, cls
