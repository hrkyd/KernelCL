import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class Kernel(nn.Module):
    def __init__(self, args):
        super(Kernel, self).__init__()
        self.kernel = args.type
        self.args = args

    def forward(self, z1, z2):
        if self.kernel == 'gaussian':
            sim_mat = torch.exp( -((torch.cdist(z1, z2, p=2) ** 2) / (self.args.band_width)))
        elif self.kernel == 'quadratic':
            sim_mat = torch.matmul(z1, z2.T) ** 2
        else:
            raise NotImplementedError
        return sim_mat

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.kernel = Kernel(args)
        self.weight = args.weight
        self.normalize = not args.no_normalize

    def forward(self, z1, z2):
        if self.normalize:
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
        else:
            raise AttributeError('Feature vectors should be normalized')

        sim_mat = self.kernel(z1, z2)
        pos = torch.mean(torch.diag(sim_mat, 0))
        neg = torch.mean(torch.triu(sim_mat, diagonal=1) + torch.tril(sim_mat, diagonal=-1)) * z1.shape[0] / (z1.shape[0] - 1)

        return - pos + self.weight * neg, pos, neg

