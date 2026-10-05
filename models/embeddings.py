import torch

def timestep_embedding(t, dim):
    half = dim // 2
    i = torch.arange(half, dtype=torch.float32, device=t.device)
    freqs = 10000 ** (-i / half)
    angles = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    return emb