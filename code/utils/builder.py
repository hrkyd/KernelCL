# This code is based on the original implementation of "simsiam/builder.py" on GitHub (https://github.com/facebookresearch/simsiam)
#################################################################
# The information about the licence of the original "simsiam":
# Copyright (c) Facebook, Inc. and its affiliates
# Released under the CC-BY-NC 4.0 license 
# https://github.com/facebookresearch/simsiam/blob/main/LICENSE
#################################################################

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, base_encoder, dim=2048, pred_dim=512, no_maxpool=False, conv1_type='imagenet', proj_layer=3, method_type='gaussian'):
        super(Model, self).__init__()

        self.method_type = method_type

        # create the encoder
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True, conv1_type=conv1_type, no_maxpool=no_maxpool)

        print("=> Build a projection head")
        print("number of layers for project head => {}".format(proj_layer))
        if proj_layer == 1:
            # build a 1-layer projector
            pass
        elif proj_layer == 2:
            # build a 2-layer projector
            prev_dim = self.encoder.fc.weight.shape[1]
            self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                            nn.BatchNorm1d(prev_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(prev_dim, dim)) # output layer
        elif proj_layer == 3:
            # build a 3-layer projector
            prev_dim = self.encoder.fc.weight.shape[1]
            self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                            nn.BatchNorm1d(prev_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(prev_dim, prev_dim, bias=False),
                                            nn.BatchNorm1d(prev_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(prev_dim, dim)) # output layer
        elif proj_layer == 4:
            # build a 4-layer projector
            prev_dim = self.encoder.fc.weight.shape[1]
            self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                            nn.BatchNorm1d(prev_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(prev_dim, prev_dim, bias=False),
                                            nn.BatchNorm1d(prev_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(prev_dim, prev_dim, bias=False),
                                            nn.BatchNorm1d(prev_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(prev_dim, dim)) # output layer
        elif proj_layer == 5:
            # build a 5-layer projector
            prev_dim = self.encoder.fc.weight.shape[1]
            self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                            nn.BatchNorm1d(prev_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(prev_dim, prev_dim, bias=False),
                                            nn.BatchNorm1d(prev_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(prev_dim, prev_dim, bias=False),
                                            nn.BatchNorm1d(prev_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(prev_dim, prev_dim, bias=False),
                                            nn.BatchNorm1d(prev_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(prev_dim, dim)) # output layer
        else:
            raise NotImplementedError


    def forward(self, x1, x2):
        if self.method_type in ['gaussian', 'quadratic']:
            z1 = self.encoder(x1)
            z2 = self.encoder(x2)

            return z1, z2
        else:
            raise NotImplementedError