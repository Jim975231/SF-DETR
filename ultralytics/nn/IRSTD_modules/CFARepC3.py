import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv



class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_w * a_h



class LightFDE(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, x_high):
        b, c, h, w = x_high.shape
        fft = torch.fft.rfft2(x_high.float(), norm='ortho')

        freq_h = torch.fft.fftfreq(h, device=x_high.device).view(-1, 1)
        freq_w = torch.fft.rfftfreq(w, device=x_high.device).view(1, -1)
        radius = torch.sqrt(freq_h ** 2 + freq_w ** 2)
        high_pass_mask = radius / radius.max()  

        delta_fft = fft * (self.gain * high_pass_mask)
        high_freq_patch = torch.fft.irfft2(delta_fft, s=(h, w), norm='ortho')

        return self.bn(self.conv(high_freq_patch))


class CFARepC3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c_half = c1 // 2
        c_hidden = int(c2 * e)


        self.fde = LightFDE(self.c_half)

        # 主路
        self.cv1 = Conv(c1, c_hidden, 1, 1)
        self.cv2 = Conv(c1, c_hidden, 1, 1)
        self.cv3 = Conv(2 * c_hidden, c2, 1)


        self.m = nn.Sequential(*(
            nn.Sequential(
                Conv(c_hidden, c_hidden, 3, g=g),
                CoordAtt(c_hidden, c_hidden)
            ) for _ in range(n)
        ))

    def forward(self, x):

        x_low = x[:, :self.c_half, :, :]
        x_high = x[:, self.c_half:, :, :]

        f_patch = self.fde(x_high)

        x_high_enhanced = x_high + f_patch

        x_fused = torch.cat([x_low, x_high_enhanced], dim=1)

        return self.cv3(torch.cat([
            self.m(self.cv1(x_fused)),
            self.cv2(x_fused)
        ], dim=1))
