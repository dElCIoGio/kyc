"""Checkpoint-compatible MiniFASNet layers adapted from Silent-Face-Anti-Spoofing.

Source: Minivision Silent-Face-Anti-Spoofing ``src/model_lib/MiniFASNet.py``
(Apache-2.0).  This is limited to the V2 and V1SE inference architectures.
Names intentionally match the upstream modules so published checkpoints load
strictly.  This module is imported only by the optional PyTorch loader.
"""

from __future__ import annotations

import torch
from torch.nn import AdaptiveAvgPool2d, BatchNorm1d, BatchNorm2d, Conv2d, Linear, Module, PReLU, ReLU, Sequential, Sigmoid


class Flatten(Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.view(input.size(0), -1)


class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.conv = Conv2d(in_c, out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.conv = Conv2d(in_c, out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class Depth_Wise(Module):
    def __init__(self, c1, c2, c3, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super().__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        self.conv = Conv_block(c1_in, c1_out)
        self.conv_dw = Conv_block(c2_in, c2_out, groups=c2_in, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(c3_in, c3_out)
        self.residual = residual

    def forward(self, x):
        short_cut = x if self.residual else None
        x = self.project(self.conv_dw(self.conv(x)))
        return short_cut + x if self.residual else x


class Residual(Module):
    def __init__(self, c1, c2, c3, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.model = Sequential(*[
            Depth_Wise(c1[index], c2[index], c3[index], residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups)
            for index in range(num_block)
        ])

    def forward(self, x):
        return self.model(x)


class SEModule(Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.bn1 = BatchNorm2d(channels // reduction)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.bn2 = BatchNorm2d(channels)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.sigmoid(self.bn2(self.fc2(self.relu(self.bn1(self.fc1(self.avg_pool(x)))))))
        return module_input * x


class Depth_Wise_SE(Depth_Wise):
    def __init__(self, c1, c2, c3, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, se_reduct=8):
        super().__init__(c1, c2, c3, residual, kernel, stride, padding, groups)
        self.se_module = SEModule(c3[1], se_reduct)

    def forward(self, x):
        short_cut = x if self.residual else None
        x = self.project(self.conv_dw(self.conv(x)))
        if self.residual:
            x = self.se_module(x)
            return short_cut + x
        return x


class ResidualSE(Module):
    def __init__(self, c1, c2, c3, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1), se_reduct=4):
        super().__init__()
        blocks = []
        for index in range(num_block):
            block = Depth_Wise_SE if index == num_block - 1 else Depth_Wise
            kwargs = {"se_reduct": se_reduct} if block is Depth_Wise_SE else {}
            blocks.append(block(c1[index], c2[index], c3[index], residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups, **kwargs))
        self.model = Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class MiniFASNet(Module):
    def __init__(self, keep, embedding_size, conv6_kernel=(7, 7), drop_p=0.0, num_classes=3, img_channel=3):
        super().__init__()
        self.embedding_size = embedding_size
        self.conv1 = Conv_block(img_channel, keep[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(keep[0], keep[1], kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=keep[1])
        self.conv_23 = Depth_Wise((keep[1], keep[2]), (keep[2], keep[3]), (keep[3], keep[4]), kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[3])
        self.conv_3 = self._residual(keep, 4, 16, 4)
        self.conv_34 = Depth_Wise((keep[16], keep[17]), (keep[17], keep[18]), (keep[18], keep[19]), kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[19])
        self.conv_4 = self._residual(keep, 6, 37, 19)
        self.conv_45 = Depth_Wise((keep[37], keep[38]), (keep[38], keep[39]), (keep[39], keep[40]), kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[40])
        self.conv_5 = self._residual(keep, 2, 46, 40)
        self.conv_6_sep = Conv_block(keep[46], keep[47])
        self.conv_6_dw = Linear_block(keep[47], keep[48], groups=keep[48], kernel=conv6_kernel)
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
        self.drop = torch.nn.Dropout(p=drop_p)
        self.prob = Linear(embedding_size, num_classes, bias=False)

    @staticmethod
    def _triples(keep, end, count):
        start = end - count * 3
        return ([(keep[i], keep[i + 1]) for i in range(start, end, 3)],
                [(keep[i + 1], keep[i + 2]) for i in range(start, end, 3)],
                [(keep[i + 2], keep[i + 3]) for i in range(start, end, 3)])

    def _residual(self, keep, count, end, groups):
        c1, c2, c3 = self._triples(keep, end, count)
        return Residual(c1, c2, c3, count, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        out = self.conv1(x); out = self.conv2_dw(out); out = self.conv_23(out); out = self.conv_3(out)
        out = self.conv_34(out); out = self.conv_4(out); out = self.conv_45(out); out = self.conv_5(out)
        out = self.conv_6_sep(out); out = self.conv_6_dw(out); out = self.conv_6_flatten(out)
        if self.embedding_size != 512:
            out = self.linear(out)
        return self.prob(self.drop(self.bn(out)))


class MiniFASNetSE(MiniFASNet):
    def __init__(self, keep, embedding_size, conv6_kernel=(7, 7), drop_p=0.75, num_classes=3, img_channel=3):
        super().__init__(keep, embedding_size, conv6_kernel, drop_p, num_classes, img_channel)
        self.conv_3 = self._residual_se(keep, 4, 16, 4)
        self.conv_4 = self._residual_se(keep, 6, 37, 19)
        self.conv_5 = self._residual_se(keep, 2, 46, 40)

    def _residual_se(self, keep, count, end, groups):
        c1, c2, c3 = self._triples(keep, end, count)
        return ResidualSE(c1, c2, c3, count, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1))


_KEEP = {
    "MiniFASNetV2": [32, 32, 103, 103, 64, 13, 13, 64, 13, 13, 64, 13, 13, 64, 13, 13, 64, 231, 231, 128, 231, 231, 128, 52, 52, 128, 26, 26, 128, 77, 77, 128, 26, 26, 128, 26, 26, 128, 308, 308, 128, 26, 26, 128, 26, 26, 128, 512, 512],
    "MiniFASNetV1SE": [32, 32, 103, 103, 64, 13, 13, 64, 26, 26, 64, 13, 13, 64, 52, 52, 64, 231, 231, 128, 154, 154, 128, 52, 52, 128, 26, 26, 128, 52, 52, 128, 26, 26, 128, 26, 26, 128, 308, 308, 128, 26, 26, 128, 26, 26, 128, 512, 512],
}


def build_model(architecture: str, input_height: int, input_width: int):
    kernel = ((input_height + 15) // 16, (input_width + 15) // 16)
    if architecture == "MiniFASNetV2":
        return MiniFASNet(_KEEP[architecture], 128, kernel, 0.2, 3, 3)
    if architecture == "MiniFASNetV1SE":
        return MiniFASNetSE(_KEEP[architecture], 128, kernel, 0.75, 3, 3)
    raise ValueError("Unsupported MiniFASNet architecture")
