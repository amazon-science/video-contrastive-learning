"""The functions for VCLR video retrieval downstream (extract feature of videos)
Code partially borrowed from
https://github.com/YihengZhang-CV/SeCo-Sequence-Contrastive-Learning/blob/main/eval_svm_feature_extract.py.

MIT License
Copyright (c) 2020 YihengZhang-CV
"""

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.distributed as dist
import torchvision.transforms as transforms

from dataset import build_dataset
from models.resnet_mlp import resnet50
from utils.logger import setup_logger
from utils.util import load_pretrained
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

import os
import argparse
import json
import numpy as np


def parse_option():
    parser = argparse.ArgumentParser('retrieval eval')

    # data
    parser.add_argument('--data_dir', type=str, required=True, help='root director of dataset')
    parser.add_argument('--dataset', type=str,
                        default='DownstreamDatasetMultiFrames', help='dataset to training')
    parser.add_argument('--datasplit', type=str, default='split_1')
    parser.add_argument('--datamode', type=str, default='train')
    parser.add_argument('--data-source', type=str, default='ucf')
    parser.add_argument('--datasamplenum', type=int, default=30)

    # aug
    parser.add_argument('--resize', type=int, default=0)
    parser.add_argument('--cropsize', type=int, default=224)

    # feat dim: 6: 2048; 7: 128
    parser.add_argument('--layer', type=int, default=6)

    # io
    parser.add_argument('--pretrained_model', type=str, required=True, help="pretrained model path")
    parser.add_argument('--output_dir', type=str, default='./eval_output', help='output director')

    # msic
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    args = parser.parse_args()
    return args


def get_loader(args):
    val_transform_list = []
    if args.resize > 0:
        val_transform_list.append(transforms.Resize(args.resize))
    val_transform_list.append(transforms.CenterCrop(args.cropsize))
    val_transform_list.append(transforms.ToTensor())
    val_transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    val_transform = transforms.Compose(val_transform_list)
    clipdataset = build_dataset(
        dataset=args.dataset,
        root=args.data_dir,
        split=args.datasplit,
        transform=val_transform,
        mode=args.datamode,
        data_source=args.data_source,
        sample_num=args.datasamplenum,
    )
    dataloader = DataLoader(clipdataset,
                            batch_size=1,
                            num_workers=8,
                            sampler=DistributedSampler(clipdataset, shuffle=False),
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False)
    return dataloader, len(clipdataset)


def main(args):
    data_loader, total_num = get_loader(args)
    logger.info('using data: {}'.format(len(data_loader)))

    model_config_dict = dict(
        num_classes=128,
        mlp=True,
    )
    model = resnet50(**model_config_dict).cuda()
    model = DistributedDataParallel(model, device_ids=[args.local_rank])
    load_pretrained(args, model)
    model.eval()

    logger.info('model init done')

    all_feat = []
    all_feat_cls = np.zeros([len(data_loader)], dtype=np.int32)

    with torch.no_grad():
        for idx, (data, cls) in enumerate(data_loader):
            logger.info('{}/{}'.format(idx, len(data_loader)))
            # data: B * S * C * H * W
            data = data.cuda()
            feat = model(data, layer=args.layer, tsn_mode=True).view(-1)

            all_feat.append(feat.data.cpu().numpy())
            all_feat_cls[idx] = cls.item()

    all_feat = np.stack(all_feat, axis=0)
    np.save(os.path.join(args.output_dir, 'feature_{}_{}.npy'.format(args.datamode, args.local_rank)), all_feat)
    np.save(os.path.join(args.output_dir, 'feature_{}_cls_{}.npy'.format(args.datamode, args.local_rank)), all_feat_cls)

    if dist.get_rank() == 0:
        np.save(os.path.join(args.output_dir, 'vid_num_{}.npy'.format(args.datamode)), np.array([total_num]))


if __name__ == '__main__':
    args = parse_option()

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=args.output_dir, distributed_rank=dist.get_rank(), name="vclr")
    if dist.get_rank() == 0:
        path = os.path.join(args.output_dir, "config.json")
        with open(path, "w") as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    main(args)
