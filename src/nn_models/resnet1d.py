import torch
import torch.nn as nn
import torch.nn.functional as F

import heads


class ResNet1DBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, downsampling):

        super(ResNet1DBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.activation = nn.LeakyReLU(inplace=False)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.pooling = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        self.downsampling = downsampling

    def forward(self, x):

        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pooling(x)

        identity = self.downsampling(identity)
        outputs = x + identity
        outputs = self.activation(outputs)

        return outputs


class ResNet1D(nn.Module):

    def __init__(self, in_channels, res_channels, res_blocks, kernels, fixed_kernel_size):

        super(ResNet1D, self).__init__()

        self.in_channels = in_channels
        self.res_channels = res_channels
        self.res_blocks = res_blocks
        self.kernels = kernels

        self.parallel_conv = nn.ModuleList()
        for i, kernel_size in enumerate(self.kernels):
            sep_conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.res_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                bias=True
            )
            self.parallel_conv.append(sep_conv)

        self.bn1 = nn.BatchNorm1d(num_features=self.res_channels)
        self.activation = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv1d(in_channels=self.res_channels, out_channels=self.res_channels, kernel_size=fixed_kernel_size, stride=2, padding=2, bias=True)
        self.res_blocks = self._make_resnet_layer(kernel_size=fixed_kernel_size, stride=1, padding=fixed_kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(num_features=self.res_channels)

        self.head = nn.Conv1d(in_channels=self.res_channels, out_channels=14, kernel_size=1, stride=1, padding='same')

    def _make_resnet_layer(self, kernel_size, stride, padding=0):

        layers = []

        for i in range(self.res_blocks):
            downsampling = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
            res_block = ResNet1DBlock(
                in_channels=self.res_channels,
                out_channels=self.res_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                downsampling=downsampling
            )
            layers.append(res_block)

        return nn.Sequential(*layers)

    def forward(self, x):

        x = torch.cat((
            x[:, :360].view(x.shape[0], -1, 60),
            torch.unsqueeze(x[:, 360:376], dim=-1).repeat(repeats=(1, 1, 60)),
            x[:, 376:].view(x.shape[0], -1, 60)
        ), dim=1)

        out_sep = []

        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)

        x = torch.cat(out_sep, dim=2)

        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv1(x)

        x = self.res_blocks(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = torch.squeeze(F.adaptive_avg_pool1d(x, output_size=(60,)), dim=-1)

        outputs = self.head(x)
        outputs = torch.cat((
            outputs[:, :6, :].view(-1, 360),
            outputs[:, 6:, :].mean(dim=-1)
        ), dim=-1)

        return outputs
