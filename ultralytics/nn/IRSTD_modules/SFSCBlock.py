import torch
import torch.nn as nn
from collections import OrderedDict

from ultralytics.nn.modules.conv import Conv



class ScharrConv(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.conv_x = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False)
        self.conv_y = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False)


        scharr_x = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]])
        scharr_y = torch.tensor([[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]])

        self.conv_x.weight.data = scharr_x.view(1, 1, 3, 3).repeat(in_ch, 1, 1, 1)
        self.conv_y.weight.data = scharr_y.view(1, 1, 3, 3).repeat(in_ch, 1, 1, 1)


        for p in self.parameters():
            p.requires_grad = False


        self.bn = nn.BatchNorm2d(in_ch)

    def forward(self, x):

        edge = torch.abs(self.conv_x(x)) + torch.abs(self.conv_y(x))
        return self.bn(edge)



class SpectralGating(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape


        x = x.float()
        fft = torch.fft.rfft2(x, norm='ortho')

        gate = self.fc(x)


        fft_weighted = fft * gate

        return torch.fft.irfft2(fft_weighted, s=(h, w), norm='ortho')



class InfraredFreqFusion(nn.Module):
    def __init__(self, ch):
        super().__init__()


        self.edge_branch = ScharrConv(ch)

        self.freq_branch = SpectralGating(ch)

        self.spatial_branch = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        )



        self.reduce = nn.Sequential(
            nn.Conv2d(ch * 3, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        )


        self.align = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, groups=ch, bias=False),  # DWConv
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        )


        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        edge = self.edge_branch(x)
        freq = self.freq_branch(x)
        spatial = self.spatial_branch(x)


        fused = torch.cat([edge, freq, spatial], dim=1)

        fused = self.reduce(fused)

        fused = self.align(fused)

        return x + fused * self.att(fused)



class DWConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, act='relu'):
        super().__init__()

        self.dw = nn.Conv2d(ch_in, ch_in, kernel_size, stride, kernel_size // 2, groups=ch_in, bias=False)

        self.pw = nn.Conv2d(ch_in, ch_out, 1, 1, 0, bias=False)

        self.bn = nn.BatchNorm2d(ch_out)


        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'silu':
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return self.act(self.bn(x))



class SFSCBlock(nn.Module):

    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d', **kwargs):
        super().__init__()
        self.shortcut = shortcut

        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', Conv(ch_in, ch_out, 1, 1, act=False))
                ]))
            else:
                self.short = Conv(ch_in, ch_out, 1, stride, act=False)



        self.branch2a = DWConvNormLayer(ch_in, ch_out, stride=stride, act=act)

        self.branch2b = InfraredFreqFusion(ch_out)

        if act == 'relu':
            self.act_final = nn.ReLU(inplace=True)
        elif act == 'silu':
            self.act_final = nn.SiLU(inplace=True)
        else:
            self.act_final = nn.Identity()

    def forward(self, x):

        out = self.branch2a(x)
        out = self.branch2b(out)


        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        return self.act_final(out + short)