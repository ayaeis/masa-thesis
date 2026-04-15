import math

import torch
import torch.nn as nn


class GhostConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        ratio=2,
        bias=True,
    ):
        super().__init__()
        ratio = max(2, int(ratio))
        init_channels = int(math.ceil(out_channels / ratio))
        cheap_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Conv1d(
            in_channels,
            init_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.cheap_operation = None
        if cheap_channels > 0:
            self.cheap_operation = nn.Conv1d(
                init_channels,
                cheap_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=init_channels,
                bias=bias,
            )

        self.out_channels = out_channels

    def forward(self, x):
        primary = self.primary_conv(x)
        if self.cheap_operation is None:
            return primary
        cheap = self.cheap_operation(primary)
        out = torch.cat([primary, cheap], dim=1)
        return out[:, : self.out_channels, :]


class GhostConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        ratio=2,
        bias=True,
    ):
        super().__init__()
        ratio = max(2, int(ratio))
        init_channels = int(math.ceil(out_channels / ratio))
        cheap_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Conv2d(
            in_channels,
            init_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.cheap_operation = None
        if cheap_channels > 0:
            self.cheap_operation = nn.Conv2d(
                init_channels,
                cheap_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=init_channels,
                bias=bias,
            )

        self.out_channels = out_channels

    def forward(self, x):
        primary = self.primary_conv(x)
        if self.cheap_operation is None:
            return primary
        cheap = self.cheap_operation(primary)
        out = torch.cat([primary, cheap], dim=1)
        return out[:, : self.out_channels, :, :]
