import torch
import torch.nn as nn
from models.embeddings import timestep_embedding

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

class UNet(nn.Module):
    def __init__(self, time_dim=128):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.init_cov = nn.Conv2d(1, 32, 3, padding=1)
        self.down1 = ResBlock(32, 32, time_dim)
        self.ds1 = Downsample(32)
        self.down2 = ResBlock(32, 64, time_dim)
        self.ds2 = Downsample(64)
        self.mid1 = ResBlock(64, 128, time_dim)
        self.mid2 = ResBlock(128, 128, time_dim)
        self.us2 = Upsample(128)
        self.up2 = ResBlock(192, 64, time_dim)
        self.us1 = Upsample(64)
        self.up1 = ResBlock(96, 32, time_dim)
        self.out_norm = nn.GroupNorm(8, 32)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x, t):
        temb = timestep_embedding(t, self.time_dim)
        temb = self.time_mlp(temb)
        h = self.init_cov(x)
        h = self.down1(h, temb)
        skip1 = h
        h = self.ds1(h)
        h = self.down2(h, temb)
        skip2 = h
        h = self.ds2(h)
        h = self.mid1(h, temb)
        h = self.mid2(h, temb)
        h = self.us2(h)
        h = torch.cat([h, skip2], dim=1)
        h = self.up2(h, temb)
        h = self.us1(h)
        h = torch.cat([h, skip1], dim=1)
        h = self.up1(h, temb)
        h = self.out_norm(h)
        h = self.out_act(h)
        h = self.out_conv(h)
        return h


model = UNet()
# x = torch.randn(4, 1, 32, 32)
# t = torch.tensor([0, 10, 500, 999])
# print(model(x, t).shape)
for name, p in model.named_parameters():
    if p.grad is None:
        print("no grad:", name)