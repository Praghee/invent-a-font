import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x, temb):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        t = self.act(temb)
        h = h + self.time_proj(t)[:, :, None, None]
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        out = h + self.skip(x)
        return out

class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        return out

class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ups1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.ups1(x)
        out = self.conv1(x)
        return out