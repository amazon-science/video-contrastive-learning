"""The functions for VCLR self-supervised pretraining
Code partially borrowed from
https://github.com/YihengZhang-CV/SeCo-Sequence-Contrastive-Learning/blob/main/train_inter_intra_order.py.

MIT License
Copyright (c) 2020 YihengZhang-CV
"""

import os
import json
import mmcv

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from models.resnet_mlp import resnet50 as resnet50mlp
from models.Contrast import MemorySeCo, MemoryVCLR, NCESoftmaxLoss
from utils.logger import setup_logger
from utils.util import AverageMeter, set_bn_train, moment_update
from utils.util import load_checkpoint, save_checkpoint, load_pretrained, get_loader, parse_option
from utils.util import DistributedShuffle
from utils.lr_scheduler import get_scheduler


def build_model(args):
    model_config_dict = dict(
        num_classes=128,
        mlp=args.model_mlp,
        intra_out=True,
        order_out=True,
        tsn_out=True,
    )
    model = resnet50mlp(**model_config_dict).cuda()
    model_ema = resnet50mlp(**model_config_dict).cuda()
    if args.pretrained_model:
        load_pretrained(args, model)
    # copy weights from `model' to `model_ema'
    moment_update(model, model_ema, 0)
    return model, model_ema


def main(args, writer):
    train_loader = get_loader(args)
    n_data = len(train_loader.dataset)
    logger.info("length of training dataset: {}".format(n_data))

    model, model_ema = build_model(args)
    logger.info('{}'.format(model))
    contrast = MemorySeCo(128, args.nce_k, args.nce_t, args.nce_t_intra).cuda()
    contrast_tsn = MemoryVCLR(128, args.nce_k, args.nce_t).cuda()
    criterion = NCESoftmaxLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.batch_size * dist.get_world_size() / 256 * args.base_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, len(train_loader), args)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=args.broadcast_buffer)
    logger.info('Distributed Enabled')

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume)
        load_checkpoint(args, model, model_ema, contrast, contrast_tsn, optimizer, scheduler, logger.info)

    # routine
    logger.info('Training')
    timer = mmcv.Timer()
    for epoch in range(args.start_epoch, args.epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        loss = train_vclr(epoch, train_loader, model, model_ema, contrast, contrast_tsn, criterion, optimizer,
                          scheduler, writer, args)
        logger.info('epoch {}, total time {:.2f}, loss={}'.format(epoch, timer.since_last_check(), loss))
        if dist.get_rank() == 0:
            save_checkpoint(args, epoch, model, model_ema, contrast, contrast_tsn, optimizer, scheduler, logger.info)
        dist.barrier()


def train_vclr(epoch, train_loader, model, model_ema, contrast, contrast_tsn, criterion, optimizer, scheduler, writer, args):
    model.train()
    set_bn_train(model_ema)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    timer = mmcv.Timer()
    for idx, (xq, x1, x2, x3, order_label, tsn_q, tsn_k) in enumerate(train_loader):
        xq = xq.cuda(non_blocking=True)  # query
        x1 = x1.cuda(non_blocking=True)  # same frame diff aug
        x2 = x2.cuda(non_blocking=True)  # diff frame 1
        x3 = x3.cuda(non_blocking=True)  # diff frame 2
        order_label = order_label.cuda(non_blocking=True)
        tsn_q = tsn_q.cuda(non_blocking=True)
        tsn_k = tsn_k.cuda(non_blocking=True)
        # forward keys
        with torch.no_grad():
            x1_shuffled, x1_backward_inds = DistributedShuffle.forward_shuffle(x1)
            x2_shuffled, x2_backward_inds = DistributedShuffle.forward_shuffle(x2)
            x3_shuffled, x3_backward_inds = DistributedShuffle.forward_shuffle(x3)
            x1_feat_inter, x1_feat_intra = model_ema(x1_shuffled)
            x2_feat_inter, x2_feat_intra = model_ema(x2_shuffled)
            x3_feat_inter, x3_feat_intra = model_ema(x3_shuffled)
            x1_feat_inter_all, x1_feat_inter = DistributedShuffle.backward_shuffle(x1_feat_inter, x1_backward_inds)
            x1_feat_intra_all, x1_feat_intra = DistributedShuffle.backward_shuffle(x1_feat_intra, x1_backward_inds)
            x2_feat_inter_all, x2_feat_inter = DistributedShuffle.backward_shuffle(x2_feat_inter, x2_backward_inds)
            x2_feat_intra_all, x2_feat_intra = DistributedShuffle.backward_shuffle(x2_feat_intra, x2_backward_inds)
            x3_feat_inter_all, x3_feat_inter = DistributedShuffle.backward_shuffle(x3_feat_inter, x3_backward_inds)
            x3_feat_intra_all, x3_feat_intra = DistributedShuffle.backward_shuffle(x3_feat_intra, x3_backward_inds)

            # tsn, o3n
            tsn_k_shuffle, tsn_k_backward_inds = DistributedShuffle.forward_shuffle(tsn_k)
            tsn_k_feat, o3n_k = model_ema(tsn_k_shuffle, tsn_mode=True)
            tsn_k_feat_all, tsn_k_feat = DistributedShuffle.backward_shuffle(tsn_k_feat, tsn_k_backward_inds)
            o3n_k_feat_all, o3n_k_feat = DistributedShuffle.backward_shuffle(o3n_k, tsn_k_backward_inds)

        # forward query
        xq_feat_inter, xq_feat_intra = model(xq)
        tsn_q_feat, o3n_q_feat, xq_logit_order = model(tsn_q, order_feat=o3n_k_feat, tsn_mode=True)

        out_inter = contrast(xq_feat_inter,
                             x1_feat_inter, x2_feat_inter, x3_feat_inter,
                             torch.cat([x1_feat_inter_all, x2_feat_inter_all, x3_feat_inter_all], dim=0), inter=True)
        out_intra = contrast(xq_feat_intra,
                             x1_feat_intra, x2_feat_intra, x3_feat_intra, None, inter=False)
        out_tsn = contrast_tsn(tsn_q_feat,
                               tsn_k_feat, tsn_k_feat_all)

        # loss calc
        loss_inter = criterion(out_inter)
        loss_intra = criterion(out_intra)
        loss_order = torch.nn.functional.cross_entropy(xq_logit_order, order_label)
        loss_tsn = criterion(out_tsn)
        loss = loss_inter + loss_intra + loss_order + loss_tsn
        # backward
        optimizer.zero_grad()
        loss.backward()
        # update params
        optimizer.step()
        scheduler.step()
        moment_update(model, model_ema, args.alpha)
        # update meters
        loss_meter.update(loss.item())
        batch_time.update(timer.since_last_check())
        # print info
        if idx % args.print_freq == 0:
            logger.info(
                'Train: [{:>3d}]/[{:>4d}/{:>4d}] BT={:>0.3f}/{:>0.3f} Loss={:>0.3f} {:>0.3f} {:>0.3f} {:>0.3f} {:>0.3f}/{:>0.3f}'.format(
                    epoch, idx, len(train_loader),
                    batch_time.val, batch_time.avg,
                    loss.item(), loss_inter.item(), loss_intra.item(), loss_order.item(), loss_tsn.item(), loss_meter.avg,
                ))

        # summary to tensorboard
        if dist.get_rank() == 0:
            n_iter = idx + len(train_loader) * (epoch - 1)
            writer.add_scalar('Loss/loss', loss.item(), n_iter)
            writer.add_scalar('Loss/loss_avg', loss_meter.avg, n_iter)
            writer.add_scalar('Loss/loss_inter', loss_inter.item(), n_iter)
            writer.add_scalar('Loss/loss_intra', loss_intra.item(), n_iter)
            writer.add_scalar('Loss/loss_order', loss_order.item(), n_iter)
            writer.add_scalar('Loss/loss_tsn', loss_tsn.item(), n_iter)

            currlr = 0.0
            for param_group in optimizer.param_groups:
                currlr = param_group['lr']
                break
            writer.add_scalar('lr', currlr, n_iter)

    return loss_meter.avg


if __name__ == '__main__':
    args = parse_option()
    if args.rng_seed >= 0:
        torch.manual_seed(args.rng_seed)
        torch.cuda.manual_seed_all(args.rng_seed)
    print(args.local_rank)

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=args.output_dir, distributed_rank=dist.get_rank(), name="vclr")
    if dist.get_rank() == 0:
        path = os.path.join(args.output_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    log_dir = os.path.join('model', 'tensorboards')
    writer = SummaryWriter(log_dir=log_dir)

    main(args, writer)
