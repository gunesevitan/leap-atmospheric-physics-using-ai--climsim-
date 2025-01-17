import torch
import torch.nn as nn


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


class UNetSeResNet(nn.Module):

    def __init__(
            self,
            in_channels, stem_channels, down_channels, down_kernel_sizes, down_strides,
            res_kernel_sizes, res_se_ratios, res_block_depth,
    ):

        super(UNetSeResNet, self).__init__()

        self.down_kernel_sizes = down_kernel_sizes
        self.down_strides = down_strides

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=stem_channels, kernel_size=1, stride=1, groups=1, padding='same'),
            nn.BatchNorm1d(num_features=stem_channels),
        )

        self.down_layers = nn.ModuleList()
        for i in range(len(down_channels)):

            if i == 0:
                in_channels = stem_channels
            else:
                in_channels = down_channels[i - 1]

            out_channels = down_channels[i]
            kernel_size = down_kernel_sizes[i]
            stride = down_strides[i]

            block = []
            block.append(ConvBNActivation(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride))

            res_kernel_size = res_kernel_sizes[i]
            res_se_ratio = res_se_ratios[i]

            for j in range(res_block_depth):
                block.append(ResBlock(channels=out_channels, kernel_size=res_kernel_size, se_ratio=res_se_ratio))

            self.down_layers.append(nn.Sequential(*block))

        self.up_layers = nn.ModuleList()
        for i in range(len(down_channels) - 1, -1, -1):

            if i == len(down_channels) - 1:
                in_channels = down_channels[i]
                out_channels = down_channels[i - 1]
            elif i == 0:
                in_channels = down_channels[i] * 2
                out_channels = down_channels[i]
            else:
                in_channels = down_channels[i] * 2
                out_channels = down_channels[i - 1]

            self.up_layers.append(
                ConvBNActivation(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    groups=1
                )
            )

        self.head = nn.Conv1d(in_channels=down_channels[0], out_channels=14, kernel_size=1, stride=1, padding='same')

    def forward(self, x):

        outputs = []
        #print(x.shape)
        x = self.stem(x)
        #print(x.shape)

        for i in range(len(self.down_layers)):
            x_out = self.down_layers[i](x)

            if i < len(self.down_layers) - 1:
                outputs.append(x_out)

            x = x_out

        for j, i in enumerate(range(len(self.up_layers) - 1, -1, -1)):

            if i < len(self.up_layers) - 1:
                x = torch.cat((x, outputs[i]), dim=1)

            x = self.up_layers[j](x)

        outputs = self.head(x)
        outputs = torch.cat((
            outputs[:, :6, :].view(-1, 360),
            outputs[:, 6:14, :].mean(dim=-1)
        ), dim=-1)

        return outputs


if __name__ == '__main__':

    import preprocessing
    import numpy as np
    import sys
    sys.path.append('..')
    import settings


    x = torch.rand(8, 40, 60)

    x = x.cuda()
    m = UNetSeResNet(
        in_channels=40,
        stem_channels=64,
        down_channels=[64, 64, 64, 64, 64, 64],
        down_kernel_sizes=[3, 3, 5, 5, 7, 7],
        down_strides=[1, 1, 1, 1, 1, 1],
        res_kernel_sizes=[3, 3, 5, 5, 7, 7],
        res_se_ratios=[2, 2, 2, 2, 2, 2],
        res_block_depth=3

    )
    m.to('cuda')
    print(f'{sum(p.numel() for p in m.parameters())} parameters')
    y = m(x)
