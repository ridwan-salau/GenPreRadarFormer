import math
import torch
import torch.nn as nn


class MNet(nn.Module):
    def __init__(self, in_chirps, out_channels, conv_op=None):
        super(MNet, self).__init__()
        self.t_conv3d = nn.Conv3d(in_channels=2, out_channels=out_channels, kernel_size=(3, 1, 1), stride=(2, 1, 1),
                                padding=(1, 0, 0))
        
        t_conv_out = math.floor((in_chirps + 2 * 1 - (3 - 1) - 1) / 2 + 1)
        self.t_maxpool = nn.MaxPool3d(kernel_size=(t_conv_out, 1, 1))

    def forward(self, x):
        batch_size, n_channels, win_size, in_chirps, w, h = x.shape
        inp_t = x.transpose(2,1)
        inp_t = inp_t.reshape(-1, *inp_t.shape[2:])
        out_t = self.t_conv3d(inp_t)
        out_t = self.t_maxpool(out_t).squeeze().reshape(batch_size, win_size, -1, w, h)
        return out_t

class MNetPlus(MNet):
    def __init__(self, in_chirps, out_channels, win_size, conv_op=None):
        super(MNetPlus, self).__init__(in_chirps, out_channels)
        self.conv3d_1 = nn.Conv3d(in_channels=win_size, out_channels=win_size, kernel_size=(3, 1, 1), stride=(2, 1, 1),
                                padding=(1, 0, 0))
        self.conv3d_2 = nn.Conv3d(in_channels=win_size, out_channels=win_size, kernel_size=(3, 1, 1), stride=(2, 1, 1),
                                padding=(1, 0, 0))
        self.conv3d_3 = nn.Conv3d(in_channels=win_size, out_channels=win_size, kernel_size=(3, 1, 1), stride=(2, 1, 1),
                                padding=(1, 0, 0))
        self.conv3d_4 = nn.Conv3d(in_channels=win_size, out_channels=win_size, kernel_size=(2, 1, 1), stride=(1, 1, 1),
                                padding=(0, 0, 0))

    def forward(self, x):
        out_mnet = super().forward(x)
        out = self.conv3d_1(out_mnet)
        out = self.conv3d_2(out)
        out = self.conv3d_3(out)
        out = self.conv3d_4(out)
        
        return out

if __name__ == '__main__':
    batch_size = 4
    in_channels = 2
    win_size = 32
    in_chirps = 4
    w = 128
    h = 128
    out_channels = 32
    mnet = MNetPlus(in_chirps=in_chirps, out_channels=out_channels, win_size=win_size).cuda()
    input = torch.randn(batch_size, in_channels, win_size, in_chirps, w, h).cuda()
    output = mnet(input)
    print(output.shape, output.device, output.dtype)
