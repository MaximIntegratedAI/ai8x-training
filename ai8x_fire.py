###################################################################################################
#
# Copyright (C) 2019-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Contains the limits of the AI85 implementation and custom PyTorch modules that take
the limits into account.
"""

import torch
import torch.nn as nn
import ai8x


class Fire(nn.Module):
    """
    AI8X - Fire Layer
    """
    def __init__(self, in_planes, squeeze_planes, expand1x1_planes, expand3x3_planes,
                 bias=True):
        super(Fire, self).__init__()
        self.squeeze_layer = ai8x.FusedConv2dReLU(in_channels=in_planes,
                                                  out_channels=squeeze_planes, kernel_size=1,
                                                  bias=bias)
        self.expand1x1_layer = ai8x.FusedConv2dReLU(in_channels=squeeze_planes,
                                                    out_channels=expand1x1_planes, kernel_size=1,
                                                    bias=bias)
        self.expand3x3_layer = ai8x.FusedConv2dReLU(in_channels=squeeze_planes,
                                                    out_channels=expand3x3_planes, kernel_size=3,
                                                    padding=1, bias=bias)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.squeeze_layer(x)
        return torch.cat([
            self.expand1x1_layer(x),
            self.expand3x3_layer(x)
        ], 1)
