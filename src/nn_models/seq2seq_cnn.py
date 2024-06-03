import torch
import torch.nn as nn
import torch.nn.functional as F


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
            nn.ReLU()
        )

    def forward(self, x):

        x = self.layers(x)

        return x


class SEBlock(nn.Module):

    def __init__(self, channels, se_ratio):

        super(SEBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=(1,)),
            nn.Conv1d(in_channels=channels, out_channels=channels // se_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=channels // se_ratio, out_channels=channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = torch.mul(x, self.layers(x))

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


class Seq2SeqCNN(nn.Module):

    def __init__(
            self,
            in_channels, stem_channels, down_channels, down_kernel_sizes, down_strides,
            res_kernel_sizes, res_se_ratios, res_block_depth
    ):

        super(Seq2SeqCNN, self).__init__()

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

        self.up_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=down_channels[i] + down_channels[i - 1],
                    out_channels=down_channels[i - 1],
                    kernel_size=3,
                    stride=1,
                    padding='same',
                    bias=True
                ),
                nn.BatchNorm1d(down_channels[i - 1]),
                nn.ReLU(inplace=True)
            ) for i in range(len(down_channels) - 1, 0, -1)

        ])
        self.head = nn.Conv1d(in_channels=down_channels[0], out_channels=14, kernel_size=1, stride=1, padding='same')

    def forward(self, x):

        x = torch.cat((
            x[:, :360].view(x.shape[0], -1, 60),
            torch.unsqueeze(x[:, 360:376], dim=-1).repeat(repeats=(1, 1, 60)),
            x[:, 376:].view(x.shape[0], -1, 60)
        ), dim=1)

        outputs = []

        x = self.stem(x)
        #print(f'x stem {x.shape}')

        for i in range(len(self.down_layers)):
            x_out = self.down_layers[i](x)

            if i < len(self.down_layers) - 1:
                outputs.append(x_out)

            x = x_out
            #print(f'down x {i} {x_out.shape}')

        #print(f'{len(outputs)} {outputs[0].shape} {outputs[1].shape}')

        for j, i in enumerate(range(len(self.up_layers) - 1, -1, -1)):
            x = torch.cat((x, outputs[i]), dim=1)
            #print(f'up cat x {x.shape}')
            x = self.up_layers[j](x)
            #print(f'up x {i} {x.shape}')

        outputs = self.head(x)
        #print(f'head x {outputs.shape}')
        outputs = torch.cat((
            outputs[:, :6, :].view(-1, 360),
            outputs[:, 6:, :].mean(dim=-1)
        ), dim=-1)
        #print(f'x out {outputs.shape}')

        return outputs


if __name__ == '__main__':

    x = torch.rand(8, 556)
    m = Seq2SeqCNN(
        in_channels=25,
        stem_channels=32,
        down_channels=[32, 64, 128],
        down_kernel_sizes=[3, 3, 3],
        down_strides=[1, 1, 1],
        res_kernel_sizes=[1, 1, 1],
        res_se_ratios=[2, 2, 2],
        res_block_depth=1

    )

    y = m(x)