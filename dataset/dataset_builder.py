from mmcv.utils import Registry

DATASETS = Registry('datasets')


def build_dataset(dataset, root, split, transform, **kwargs):
    return DATASETS.get(dataset)(root, transform, split, **kwargs)
