import numpy as np
import torch
import torch.nn as nn


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

        self.pooling_type = pooling_type
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.head = nn.Linear(embed_dim, 14)

    def forward(self, x):

        x = x.permute(0, 2, 1)
        x = self.stem(x)

        x += self.positional_embeddings

        for block in self.encoder:
            x = block(x)

        outputs = self.head(x).permute(0, 2, 1)
        outputs = torch.cat((
            outputs[:, :6, :].reshape(-1, 360),
            outputs[:, 6:, :].mean(dim=-1)
        ), dim=-1)

        return outputs
