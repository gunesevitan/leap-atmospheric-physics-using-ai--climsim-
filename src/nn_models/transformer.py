import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import heads


def positional_encoding(length, embed_dim):

    dim = embed_dim // 2

    position = np.arange(length)[:, np.newaxis]
    dim = np.arange(dim)[np.newaxis, :] / dim

    angle = 1 / (10000 ** dim)
    angle = position * angle

    pos_embed = np.concatenate([np.sin(angle), np.cos(angle)], axis=-1)
    pos_embed = torch.from_numpy(pos_embed).float()

    return pos_embed


class MLP(nn.Module):

    def __init__(self, embed_dim, hidden_dim):

        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=embed_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class TransformerBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, out_dim):

        super(TransformerBlock, self).__init__()

        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.mlp = MLP(embed_dim=embed_dim, hidden_dim=out_dim)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(out_dim)

    def forward(self, x):

        attention_output, _ = self.attention(x, x, x)
        x = x + attention_output
        x = self.layer_norm1(x)
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.layer_norm2(x)

        return x


class Transformer(nn.Module):

    def __init__(self, input_dim, seq_len, embed_dim, num_heads, num_blocks, pooling_type, dropout_rate):

        super(Transformer, self).__init__()

        self.stem = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.encoder = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                out_dim=embed_dim,
            ) for _ in range(num_blocks)
        ])

        self.positional_embeddings = torch.nn.Parameter(positional_encoding(seq_len, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros((1, embed_dim)))

        self.pooling_type = pooling_type
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.head = heads.MultiOutputHead(input_dim=embed_dim)

    def forward(self, x):

        x = torch.cat((
            x[:, :360].view(x.shape[0], -1, 60),
            torch.unsqueeze(x[:, 360:376], dim=-1).repeat(repeats=(1, 1, 60)),
            x[:, 376:].view(x.shape[0], -1, 60),
        ), dim=1)

        x = x.permute(0, 2, 1)
        x = self.stem(x)

        # Add positional embeddings and concatenate cls token
        x += self.positional_embeddings
        x = torch.cat([
            self.cls_token.unsqueeze(0).repeat(x.size(0), 1, 1),
            x
        ], 1)

        # Pass it to transformer encoder
        for block in self.encoder:
            x = block(x)

        if self.pooling_type == 'avg':
            x = torch.mean(x[:, 1:, :], dim=1)
        elif self.pooling_type == 'max':
            x = torch.max(x[:, 1:, :], dim=1)[0]
        elif self.pooling_type == 'cls':
            x = x[:, 0, :]

        x = self.dropout(x)
        outputs = self.head(x)
        outputs = torch.cat(outputs, dim=1)

        return outputs


if __name__ == '__main__':

    import pandas as pd


    features = np.random.rand(1024, 556)
    targets = np.random.rand(1024, 368)

    #features = np.concatenate([
    #    features[:, :360].reshape(features.shape[0], -1, 60),
    #    np.repeat(np.expand_dims(features[:, 360:376], axis=-1), repeats=60, axis=-1),
    #    features[:, 376:].reshape(features.shape[0], -1, 60),
    #], axis=1)

    import torch_datasets
    d = torch_datasets.TabularInMemoryDataset(features, targets)

    from torch.utils.data import DataLoader
    l = DataLoader(dataset=d, batch_size=4)
    m = Transformer(
        input_dim=25,
        seq_len=60,
        embed_dim=256,

        num_heads=8,
        num_blocks=2,
        pooling_type='avg',
        dropout_rate=0.1,
    )

    for x, y in l:

        yy = m(x)
        exit()

