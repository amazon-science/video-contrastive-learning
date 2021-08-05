"""The functions for some tools
Code partially borrowed from
https://github.com/YihengZhang-CV/SeCo-Sequence-Contrastive-Learning/blob/main/seco/util.py.

MIT License
Copyright (c) 2020 YihengZhang-CV
"""

import argparse
import random
import torch
import torch.distributed as dist
import os

from torchvision import transforms
from dataset import build_dataset
from PIL import ImageFilter


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype)
                for _ in range(dist.get_world_size())]
    dist.all_gather(out_list, x, async_op=False)
    return torch.cat(out_list, dim=0)


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


class DistributedShuffle:
    @staticmethod
    def forward_shuffle(x, epoch=None):
        x_all = dist_collect(x)
        forward_inds, backward_inds = DistributedShuffle.get_shuffle_ids(x_all.shape[0], epoch)
        forward_inds_local = DistributedShuffle.get_local_id(forward_inds)
        return x_all[forward_inds_local], backward_inds

    @staticmethod
    def backward_shuffle(x, backward_inds, return_local=True):
        x_all = dist_collect(x)
        if return_local:
            backward_inds_local = DistributedShuffle.get_local_id(backward_inds)
            return x_all[backward_inds], x_all[backward_inds_local]
        else:
            return x_all[backward_inds]

    @staticmethod
    def get_local_id(ids):
        return ids.chunk(dist.get_world_size())[dist.get_rank()]

    @staticmethod
    def get_shuffle_ids(bsz, epoch):
        if epoch is not None:
            torch.manual_seed(epoch)
        forward_inds = torch.randperm(bsz).long().cuda()
        if epoch is None:
            torch.distributed.broadcast(forward_inds, src=0)
        backward_inds = torch.zeros(forward_inds.shape[0]).long().cuda()
        value = torch.arange(bsz).long().cuda()
        backward_inds.index_copy_(0, forward_inds, value)
        return forward_inds, backward_inds


def set_bn_train(model):
    def set_bn_train_helper(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()
    model.eval()
    model.apply(set_bn_train_helper)


@torch.no_grad()
def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data = p2.data * m + p1.data * (1 - m)


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def load_pretrained(args, model, logger=print):
    ckpt = torch.load(args.pretrained_model, map_location='cpu')
    if len(ckpt) == 3:  # moco initialization
        ckpt = {k[17:]: v for k, v in ckpt['state_dict'].items() if k.startswith('module.encoder_q')}
        for fc in ('fc_inter', 'fc_intra', 'fc_order', 'fc_tsn'):
            ckpt[fc + '.0.weight'] = ckpt['fc.0.weight']
            ckpt[fc + '.0.bias'] = ckpt['fc.0.bias']
            ckpt[fc + '.2.weight'] = ckpt['fc.2.weight']
            ckpt[fc + '.2.bias'] = ckpt['fc.2.bias']
    else:
        ckpt = ckpt['model']
    [misskeys, unexpkeys] = model.load_state_dict(ckpt, strict=False)
    logger('Missing keys: {}'.format(misskeys))
    logger('Unexpect keys: {}'.format(unexpkeys))
    logger("==> loaded checkpoint '{}'".format(args.pretrained_model))


def load_checkpoint(args, model, model_ema, contrast, contrast_tsn, optimizer, scheduler, logger=print):
    logger("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume, map_location='cpu')
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    model_ema.load_state_dict(checkpoint['model_ema'])
    contrast.load_state_dict(checkpoint['contrast'])
    contrast_tsn.load_state_dict(checkpoint['contrast_tsn'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(args, epoch, model, model_ema, contrast, contrast_tsn, optimizer, scheduler, logger=print):
    logger('==> Saving...')
    state = {
        'opt': args,
        'model': model.state_dict(),
        'model_ema': model_ema.state_dict(),
        'contrast': contrast.state_dict(),
        'contrast_tsn': contrast_tsn.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, os.path.join(args.output_dir, 'current.pth'))
    if epoch % args.save_freq == 0:
        torch.save(state, os.path.join(args.output_dir, 'ckpt_epoch_{}.pth'.format(epoch)))


def get_loader(args):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.cropsize, scale=(args.crop, 1.)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = build_dataset(args.dataset, args.data_dir, transform=train_transform, split = args.datasplit)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        sampler=train_sampler, drop_last=True)
    return train_loader


def parse_option():
    parser = argparse.ArgumentParser('training')

    # dataset
    parser.add_argument('--data_dir', type=str, required=True, help='root director of dataset')
    parser.add_argument('--dataset', type=str, default='KineticsClipFolderDatasetOrderTSN', help='dataset to training')
    parser.add_argument('--datasplit', type=str, default='train')

    parser.add_argument('--crop', type=float, default=0.2, help='minimum crop')
    parser.add_argument('--cropsize', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')

    # model
    parser.add_argument('--model_mlp', action='store_true', default=False)

    # loss function
    parser.add_argument('--alpha', type=float, default=0.999, help='exponential moving average weight')
    parser.add_argument('--nce_k', type=int, default=131072, help='num negative sampler')
    parser.add_argument('--nce_t', type=float, default=0.10, help='NCE temperature')
    parser.add_argument('--nce_t_intra', type=float, default=0.10, help='NCE temperature')

    # optimization
    parser.add_argument('--base_lr', type=float, default=0.1,
                        help='base learning when batch size = 256. final lr is determined by linear scale')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=["cosine"], help="learning rate scheduler")
    parser.add_argument('--warmup_epoch', type=int, default=5, help='warmup epoch')
    parser.add_argument('--warmup_multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--lr_decay_epochs', type=int, default=[120, 160, 200], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--epochs', type=int, default=400, help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, default=1, help='used for resume')

    # io
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained_model', default='', type=str, metavar='PATH',
                        help='path to pretrained weights like imagenet (default: none)')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--output_dir', type=str, default='./output', help='output director')

    # misc
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--broadcast_buffer", action='store_true', default=False, help='broadcast_buffer for DistributedDataParallel')
    parser.add_argument("--rng_seed", type=int, default=-1, help='manual seed')

    args = parser.parse_args()
    return args
