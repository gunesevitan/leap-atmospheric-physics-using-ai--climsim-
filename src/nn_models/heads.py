import torch
import torch.nn as nn


class SingleOutputHead(nn.Module):

    def __init__(self, input_dim):

        super(SingleOutputHead, self).__init__()

        self.head = nn.Linear(input_dim, 368, bias=True)

    def forward(self, x):

        outputs = self.head(x)

        return outputs


class MultiOutputHead(nn.Module):

    def __init__(self, input_dim):

        super(MultiOutputHead, self).__init__()

        self.ptend_t_head = nn.Conv1d(input_dim, out_channels=60, kernel_size=(1,), bias=True)
        self.ptend_q0001_head = nn.Conv1d(input_dim, out_channels=60, kernel_size=(1,), bias=True)
        self.ptend_q0002_head = nn.Conv1d(input_dim, out_channels=60, kernel_size=(1,), bias=True)
        self.ptend_q0003_head = nn.Conv1d(input_dim, out_channels=60, kernel_size=(1,), bias=True)
        self.ptend_u_head = nn.Conv1d(input_dim, out_channels=60, kernel_size=(1,), bias=True)
        self.ptend_v_head = nn.Conv1d(input_dim, out_channels=60, kernel_size=(1,), bias=True)
        self.cam_out_NETSW_head = nn.Linear(input_dim, 1, bias=True)
        self.cam_out_FLWDS_head = nn.Linear(input_dim, 1, bias=True)
        self.cam_out_PRECSC_head = nn.Linear(input_dim, 1, bias=True)
        self.cam_out_PRECC_head = nn.Linear(input_dim, 1, bias=True)
        self.cam_out_SOLS_head = nn.Linear(input_dim, 1, bias=True)
        self.cam_out_SOLL_head = nn.Linear(input_dim, 1, bias=True)
        self.cam_out_SOLSD_head = nn.Linear(input_dim, 1, bias=True)
        self.cam_out_SOLLD_head = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):

        x_conv = torch.unsqueeze(x, dim=1).permute(0, 2, 1)

        ptend_t_outout = torch.squeeze(self.ptend_t_head(x_conv), -1)
        ptend_q0001_outout = torch.squeeze(self.ptend_q0001_head(x_conv), -1)
        ptend_q0002_outout = torch.squeeze(self.ptend_q0002_head(x_conv), -1)
        ptend_q0003_outout = torch.squeeze(self.ptend_q0003_head(x_conv), -1)
        ptend_u_outout = torch.squeeze(self.ptend_u_head(x_conv), -1)
        ptend_v_outout = torch.squeeze(self.ptend_v_head(x_conv), -1)
        cam_out_NETSW_outout = self.cam_out_NETSW_head(x)
        cam_out_FLWDS_outout = self.cam_out_FLWDS_head(x)
        cam_out_PRECSC_outout = self.cam_out_PRECSC_head(x)
        cam_out_PRECC_outout = self.cam_out_PRECC_head(x)
        cam_out_SOLS_outout = self.cam_out_SOLS_head(x)
        cam_out_SOLL_outout = self.cam_out_SOLL_head(x)
        cam_out_SOLSD_outout = self.cam_out_SOLSD_head(x)
        cam_out_SOLLD_outout = self.cam_out_SOLLD_head(x)

        return (
            ptend_t_outout, ptend_q0001_outout, ptend_q0002_outout,
            ptend_q0003_outout, ptend_u_outout, ptend_v_outout,
            cam_out_NETSW_outout, cam_out_FLWDS_outout, cam_out_PRECSC_outout, cam_out_PRECC_outout,
            cam_out_SOLS_outout, cam_out_SOLL_outout, cam_out_SOLSD_outout, cam_out_SOLLD_outout
        )
