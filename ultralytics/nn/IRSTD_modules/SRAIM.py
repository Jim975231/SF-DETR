import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyAnomalyMiner(nn.Module):


    def __init__(self):
        super().__init__()
        self.spectral_smooth = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.constant_(self.spectral_smooth.weight, 1 / 9.0)
        self.norm = nn.InstanceNorm2d(1, affine=True)
        self.mask_proj = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_spatial = torch.mean(x, dim=1, keepdim=True)

        fft_x = torch.fft.fft2(x_spatial, norm='ortho')

        amp = torch.sqrt(fft_x.real ** 2 + fft_x.imag ** 2 + 1e-12)
        phase = torch.angle(fft_x)


        log_amp = torch.log(amp)
        smooth_log_amp = self.spectral_smooth(log_amp)
        spectral_residual = log_amp - smooth_log_amp

        spectral_residual = torch.clamp(spectral_residual, min=-10.0, max=10.0)


        complex_res = torch.polar(torch.exp(spectral_residual), phase)
        saliency_map = torch.abs(torch.fft.ifft2(complex_res, norm='ortho'))


        saliency_map = self.norm(saliency_map)
        freq_mask = self.mask_proj(saliency_map)

        return freq_mask


class SRAIM(nn.Module):


    def __init__(self, c1, c2=None, num_heads=8, *args, **kwargs):
        super().__init__()


        self.freq_miner = FrequencyAnomalyMiner()
        self.freq_align = nn.Conv2d(c1, c1, kernel_size=1, bias=False)
        self.freq_norm = nn.BatchNorm2d(c1)


        self.gamma = nn.Parameter(torch.full((1,), 1e-4))


        self.transformer = nn.TransformerEncoderLayer(
            d_model=c1,
            nhead=num_heads,
            dim_feedforward=c1 * 3,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

    def forward(self, x):

        b, c, h, w = x.shape

        freq_mask = self.freq_miner(x)
        x_freq = x * freq_mask
        x_freq_aligned = self.freq_norm(self.freq_align(x_freq))



        pos_embed = self.build_2d_sincos_position_embedding(w, h, c).to(x.device)


        x_flat = x.flatten(2).permute(0, 2, 1)


        x_spatial_flat = self.transformer(x_flat + pos_embed)
        x_spatial = x_spatial_flat.permute(0, 2, 1).view(b, c, h, w).contiguous()


        out = x_spatial + self.gamma * x_freq_aligned

        return out

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]
