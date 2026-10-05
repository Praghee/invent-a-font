import torch

def make_schedule(T, schedule="linear", beta_start=1e-4, beta_end=0.02, s=0.008):
    if schedule=="linear":
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

    elif schedule=="cosine":
        steps = torch.linspace(0, T, T + 1)
        f = torch.cos((((steps / T) + s) / (1 + s)) * (torch.pi / 2)) ** 2
        alpha_bars_full = f / f[0]
        betas = torch.clamp(1 - (alpha_bars_full[1:]/alpha_bars_full[:-1]), max=0.999)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

    else:
        raise ValueError(f"unknown schedule: {schedule}")

    return betas, alphas, alpha_bars