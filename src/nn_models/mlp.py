import torch.nn as nn

import heads


class MLPBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim):

        super(MLPBlock, self).__init__()

        self.l1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU()
        self.l2 = nn.Linear(hidden_dim, input_dim, bias=True)

    def forward(self, x):

        x_i = x
        x = self.l1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.l2(x)
        x = x_i + x

        return x


class MLP(nn.Module):

    def __init__(self, input_dim, mlp_hidden_dim, n_blocks):

        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            *[
                MLPBlock(
                    input_dim=input_dim,
                    hidden_dim=mlp_hidden_dim,
                ) for _ in range(n_blocks)
            ]
        )
        self.head = heads.SingleOutputHead(input_dim=input_dim)

    def forward(self, x):

        x = self.mlp(x)
        outputs = self.head(x)

        return outputs
