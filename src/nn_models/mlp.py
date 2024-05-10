import torch
import torch.nn as nn

import heads


class MLPBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim):

        super(MLPBlock, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim, bias=True),
        )

    def forward(self, x):
        return self.mlp(x)


class LEAPModel(nn.Module):

    def __init__(self, input_dim, mlp_hidden_dim, n_blocks):

        super(LEAPModel, self).__init__()

        self.mlp = nn.Sequential(
            *[
                MLPBlock(
                    input_dim=input_dim,
                    hidden_dim=mlp_hidden_dim,
                ) for _ in range(n_blocks)
            ]
        )

        self.head = heads.MultiOutputHead(input_dim=input_dim)

    def forward(self, x):

        #x = torch.cat([v for v in x.values()], dim=1)

        x = self.mlp(x)

        outputs = (
            ptend_t_outout, ptend_q0001_outout, ptend_q0002_outout,
            ptend_q0003_outout, ptend_u_outout, ptend_v_outout,
            cam_out_NETSW_outout, cam_out_FLWDS_outout, cam_out_PRECSC_outout, cam_out_PRECC_outout,
            cam_out_SOLS_outout, cam_out_SOLL_outout, cam_out_SOLSD_outout, cam_out_SOLLD_outout
        ) = self.head(x)

        outputs = torch.cat(outputs, dim=1)

        #return (
        #    ptend_t_outout, ptend_q0001_outout, ptend_q0002_outout,
        #    ptend_q0003_outout, ptend_u_outout, ptend_v_outout,
        #    cam_out_NETSW_outout, cam_out_FLWDS_outout, cam_out_PRECSC_outout, cam_out_PRECC_outout,
        #    cam_out_SOLS_outout, cam_out_SOLL_outout, cam_out_SOLSD_outout, cam_out_SOLLD_outout
        #)

        #outputs = self.head(x)
        #outputs = torch.cat(outputs, dim=1)

        return outputs
