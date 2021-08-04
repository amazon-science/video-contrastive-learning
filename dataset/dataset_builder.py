"""The functions for building the dataset class
Code borrowed from
https://github.com/YihengZhang-CV/SeCo-Sequence-Contrastive-Learning/blob/main/dataset/dataset_builder.py.
"""
from mmcv.utils import Registry

DATASETS = Registry('datasets')


def build_dataset(dataset, root, split, transform, **kwargs):
    return DATASETS.get(dataset)(root, transform, split, **kwargs)
