import torch
import torch.nn as nn

import heads


class LSTM(nn.Module):

    def __init__(self, input_dim, lstm_dim, n_layers, dropout, bidirectional, pooling):

        super(LSTM, self).__init__()

        self.pooling = pooling

        self.stem = nn.Sequential(
            nn.Linear(input_dim, lstm_dim),
            nn.LayerNorm(lstm_dim)
        )
        self.lstm = nn.LSTM(
            input_size=lstm_dim,
            hidden_size=lstm_dim,
            num_layers=n_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.head = heads.SingleOutputHead(input_dim=lstm_dim * 2 if bidirectional else lstm_dim)

    def forward(self, x):

        x = x.permute(0, 2, 1)
        x = self.stem(x)

        x, _ = self.lstm(x)

        if self.pooling == 'avg':
            x = x.mean(dim=1)
        elif self.pooling == 'max':
            x = x.max(dim=1)[0]
        else:
            raise ValueError(f'Invalid pooling {self.pooling}')

        outputs = self.head(x)

        return outputs


class GRU(nn.Module):

    def __init__(self, input_dim, gru_dim, n_layers, dropout, bidirectional, pooling):

        super(GRU, self).__init__()

        self.pooling = pooling

        self.stem = nn.Sequential(
            nn.Linear(input_dim, gru_dim),
            nn.LayerNorm(gru_dim)
        )
        self.gru = nn.GRU(
            input_size=gru_dim,
            hidden_size=gru_dim,
            num_layers=n_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.head = heads.SingleOutputHead(input_dim=gru_dim * 2 if bidirectional else gru_dim)

    def forward(self, x):

        x = x.permute(0, 2, 1)
        x = self.stem(x)

        x, _ = self.gru(x)

        if self.pooling == 'avg':
            x = x.mean(dim=1)
        elif self.pooling == 'max':
            x = x.max(dim=1)[0]
        else:
            raise ValueError(f'Invalid pooling {self.pooling}')

        outputs = self.head(x)

        return outputs
