"""The functions for the learning rate scheduler
Code borrowed from
https://github.com/YihengZhang-CV/SeCo-Sequence-Contrastive-Learning/blob/main/seco/lr_scheduler.py.

MIT License
Copyright (c) 2020 YihengZhang-CV
"""

from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
import math


class WarmUpCosineAnnealingLR(CosineAnnealingLR):
    def __init__(self, optimizer, warm_multiplier, warm_duration, cos_duration, eta_min=0, last_epoch=-1):
        assert warm_duration >= 0
        assert warm_multiplier > 1.0
        self.warm_m = float(warm_multiplier)
        self.warm_d = warm_duration
        self.cos_duration = cos_duration
        self.cos_eta_min = eta_min
        super(WarmUpCosineAnnealingLR, self).__init__(optimizer, self.cos_duration, eta_min, last_epoch)

    def get_lr(self):
        if self.warm_d == 0:
            return super(WarmUpCosineAnnealingLR, self).get_lr()
        else:
            if not self._get_lr_called_within_step:
                warnings.warn("To get the last learning rate computed by the scheduler, "
                              "please use `get_last_lr()`.", UserWarning)
            if self.last_epoch == 0:
                return [lr / self.warm_m for lr in self.base_lrs]
                # return self.base_lrs / self.warm_m
            elif self.last_epoch <= self.warm_d:
                return [(self.warm_d + (self.warm_m - 1) * self.last_epoch) / (self.warm_d + (self.warm_m - 1) * (self.last_epoch - 1)) * group['lr'] for group in self.optimizer.param_groups]
            else:
                cos_last_epoch = self.last_epoch - self.warm_d
                if cos_last_epoch == 0:
                    return self.base_lrs
                elif (cos_last_epoch - 1 - self.cos_duration) % (2 * self.cos_duration) == 0:
                    return [group['lr'] + (base_lr - self.cos_eta_min) *
                            (1 - math.cos(math.pi / self.cos_duration)) / 2
                            for base_lr, group in
                            zip(self.base_lrs, self.optimizer.param_groups)]
                return [(1 + math.cos(math.pi * cos_last_epoch / self.cos_duration)) /
                        (1 + math.cos(math.pi * (cos_last_epoch - 1) / self.cos_duration)) *
                        (group['lr'] - self.cos_eta_min) + self.cos_eta_min
                        for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        if self.warm_d == 0:
            return super(WarmUpCosineAnnealingLR, self)._get_closed_form_lr()
        else:
            if self.last_epoch <= self.warm_d:
                return [base_lr * (self.warm_d + (self.warm_m - 1) * self.last_epoch) / (self.warm_d * self.warm_m) for base_lr in self.base_lrs]
            else:
                cos_last_epoch = self.last_epoch - self.warm_d
                return [self.cos_eta_min + (base_lr - self.cos_eta_min) *
                    (1 + math.cos(math.pi * cos_last_epoch / self.cos_duration)) / 2
                    for base_lr in self.base_lrs]


def get_scheduler(optimizer, n_iter_per_epoch, args):
    if "cosine" in args.lr_scheduler:
        return WarmUpCosineAnnealingLR(
            optimizer=optimizer,
            warm_multiplier=args.warmup_multiplier,
            warm_duration=args.warmup_epoch * n_iter_per_epoch,
            cos_duration=(args.epochs - args.warmup_epoch) * n_iter_per_epoch,
            eta_min=0.000001,
        )
    else:
        raise NotImplementedError("scheduler {} not supported".format(args.lr_scheduler))