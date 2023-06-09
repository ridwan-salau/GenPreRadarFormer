import torch
import torch.nn as nn
import sys, os
sys.path.append(os.path.abspath("."))

from .backbones.MaxVIT2 import RadarStackedHourglass
from .modules.mnet import MNet, MNetPlus




class MaxVIT2(nn.Module):
    def __init__(self, 
                in_channels, 
                n_class, 
                stacked_num=1, 
                mnet_cfg=None, 
                dcn=True,
                out_head = 1,
                win_size = 16,
                patch_size = 8, 
                hidden_size = 516, 
                receptive_field = [[3,3,3,3],[3,3,3,3]],
                num_layers = 12,
                mnet_plus_out_channels = None):
        super(MaxVIT2, self).__init__()
        self.conv_op = nn.Conv3d
        if mnet_cfg is not None:
            in_chirps_mnet, out_channels_mnet = mnet_cfg
            if mnet_plus_out_channels:
                in_channels = mnet_plus_out_channels
                self.mnet = MNetPlus(in_chirps_mnet, out_channels_mnet, win_size=win_size)
            else:
                in_channels = out_channels_mnet
                self.mnet = MNet(in_chirps_mnet, out_channels_mnet, conv_op=self.conv_op)
            self.with_mnet = True
            self.stacked_hourglass = RadarStackedHourglass(in_channels, n_class, stacked_num=stacked_num,
                                                win_size = win_size, patch_size = patch_size, hidden_size = hidden_size,
                                                num_layers = num_layers, receptive_field = receptive_field,
                                                out_head = out_head)
        else:
            self.with_mnet = False
            self.stacked_hourglass = RadarStackedHourglass(in_channels, n_class, stacked_num=stacked_num,
                                                           conv_op=self.conv_op)

    def forward(self, x):
        if self.with_mnet:
            x = self.mnet(x)
        out = self.stacked_hourglass(x)
        return out


if __name__ == '__main__':
    kwargs={
        'in_channels': 4, 'n_class': 3, 'stacked_num': 1,
        'mnet_cfg': (4, 32), 'dcn': False, 'win_size': 32,
        'patch_size': 8, 'hidden_size': 1160,
        'receptive_field': [[3, 5, 7, 9], [3, 3, 3, 3]],
        'out_head': 1, 'num_layers': 16
    }
    testModel = MaxVIT2(**kwargs).cuda().bfloat16()
    x = torch.zeros((1, 2, 32, 4, 128, 128)).cuda().bfloat16()
    print(x.shape)
    out = testModel(x)
    print(out.shape)
