import torch
import torch.nn as nn

import heads


class ConvBNActivation(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):

        super(ConvBNActivation, self).__init__()

        if stride == 1:
            padding = 'same'
        else:
            padding = (kernel_size - stride) // 2

        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups
            ),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x = self.layers(x)

        return x


class SEBlock(nn.Module):

    def __init__(self, channels, se_ratio):

        super(SEBlock, self).__init__()

        self.channels = channels
        self.gap = nn.AdaptiveAvgPool1d(output_size=(1,))

        self.layers = nn.Sequential(
            nn.Linear(in_features=channels, out_features=channels // se_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channels // se_ratio, out_features=channels),
            nn.Sigmoid()
        )

    def forward(self, x):

        x_squeeze = self.gap(x).view(-1, self.channels)
        x = torch.mul(x, torch.unsqueeze(self.layers(x_squeeze), dim=-1))

        return x


class ResBlock(nn.Module):

    def __init__(self, channels, kernel_size, se_ratio):

        super(ResBlock, self).__init__()

        self.layers = nn.Sequential(
            ConvBNActivation(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1),
            ConvBNActivation(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1),
            SEBlock(channels=channels, se_ratio=se_ratio)
        )

    def forward(self, x):

        x_residual = self.layers(x)
        x = x + x_residual

        return x


class Seq2SeqSeResNet(nn.Module):

    def __init__(
            self,
            in_channels, stem_channels, block_channels, block_kernel_sizes, block_strides,
            res_kernel_sizes, res_se_ratios, res_block_depth
    ):

        super(Seq2SeqSeResNet, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=stem_channels, kernel_size=1, stride=1, groups=1, padding='same'),
            nn.BatchNorm1d(num_features=stem_channels),
        )

        self.blocks = nn.ModuleList()
        for i in range(len(block_channels)):

            if i == 0:
                in_channels = stem_channels
            else:
                in_channels = block_channels[i - 1]

            out_channels = block_channels[i]
            kernel_size = block_kernel_sizes[i]
            stride = block_strides[i]

            block = []
            block.append(ConvBNActivation(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride))

            res_kernel_size = res_kernel_sizes[i]
            res_se_ratio = res_se_ratios[i]

            for j in range(res_block_depth):
                block.append(ResBlock(channels=out_channels, kernel_size=res_kernel_size, se_ratio=res_se_ratio))

            self.blocks.append(nn.Sequential(*block))

        self.head = nn.Conv1d(in_channels=block_channels[-1], out_channels=14, kernel_size=1, stride=1, padding='same')

    def forward(self, x):

        x = self.stem(x)

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

        outputs = self.head(x)
        outputs = torch.cat((
            outputs[:, :6, :].view(-1, 360),
            outputs[:, 6:, :].mean(dim=-1)
        ), dim=-1)

        return outputs
