"""The functions for MoCoV2 contrastive loss
Code partially borrowed from
https://github.com/YihengZhang-CV/SeCo-Sequence-Contrastive-Learning/blob/main/seco/Contrast.py.

MIT License
Copyright (c) 2020 YihengZhang-CV
"""

import torch
import torch.nn as nn
import math


class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        label = torch.zeros([x.shape[0]]).long().to(x.device)
        return self.criterion(x, label)


class MemorySeCo(nn.Module):
    """Fixed-size queue with momentum encoder"""

    def __init__(self, feature_dim, queue_size, temperature=0.10, temperature_intra=0.10):
        super(MemorySeCo, self).__init__()
        self.queue_size = queue_size
        self.temperature = temperature
        self.temperature_intra = temperature_intra
        self.index = 0

        # noinspection PyCallingNonCallable
        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(feature_dim / 3)
        memory = torch.rand(self.queue_size, feature_dim, requires_grad=False).mul_(2 * stdv).add_(-stdv)
        self.register_buffer('memory', memory)

    def forward(self, q, k_sf, k_df1, k_df2, k_all, inter=True):
        l_pos_sf = (q * k_sf.detach()).sum(dim=-1, keepdim=True)  # shape: (batchSize, 1)
        l_pos_df1 = (q * k_df1.detach()).sum(dim=-1, keepdim=True)  # shape: (batchSize, 1)
        l_pos_df2 = (q * k_df2.detach()).sum(dim=-1, keepdim=True)  # shape: (batchSize, 1)
        if inter:
            l_neg = torch.mm(q, self.memory.clone().detach().t())
            out = torch.cat((torch.cat((l_pos_sf, l_pos_df1, l_pos_df2), dim=0), l_neg.repeat(3, 1)), dim=1)
            out = torch.div(out, self.temperature).contiguous()
            with torch.no_grad():
                all_size = k_all.shape[0]
                out_ids = torch.fmod(torch.arange(all_size, dtype=torch.long).cuda() + self.index, self.queue_size)
                self.memory.index_copy_(0, out_ids, k_all)
                self.index = (self.index + all_size) % self.queue_size
        else:
            # out intra-frame similarity
            out = torch.div(torch.cat((l_pos_sf.repeat(2, 1), torch.cat((l_pos_df1, l_pos_df2), dim=0)), dim=-1),
                            self.temperature_intra).contiguous()

        return out


class MemoryVCLR(nn.Module):
    """Fixed-size queue with momentum encoder"""
    def __init__(self, feature_dim, queue_size, temperature=0.10):
        super(MemoryVCLR, self).__init__()
        self.queue_size = queue_size
        self.temperature = temperature
        self.index = 0

        # noinspection PyCallingNonCallable
        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(feature_dim / 3)
        memory = torch.rand(self.queue_size, feature_dim, requires_grad=False).mul_(2 * stdv).add_(-stdv)
        self.register_buffer('memory', memory)

    def forward(self, q, k, k_all):
        l_pos = (q * k.detach()).sum(dim=-1, keepdim=True)  # shape: (batchSize, 1)
        l_neg = torch.mm(q, self.memory.clone().detach().t())
        out = torch.cat((l_pos, l_neg), dim=1)
        out = torch.div(out, self.temperature).contiguous()
        with torch.no_grad():
            all_size = k_all.shape[0]
            out_ids = torch.fmod(torch.arange(all_size, dtype=torch.long).cuda() + self.index, self.queue_size)
            self.memory.index_copy_(0, out_ids, k_all)
            self.index = (self.index + all_size) % self.queue_size

        return out
