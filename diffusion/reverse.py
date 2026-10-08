import torch

def reverse_mean(xt, eps, i, betas, alphas, alpha_bars):
    coef_x = 1 / torch.sqrt(alphas[i])
    coef_eps = betas[i] / torch.sqrt(1 - alpha_bars[i])
    return coef_x * (xt - coef_eps * eps)

def ddpm_step(model, xt, i, betas, alphas, alpha_bars):
    B = xt.shape[0]
    t = torch.full((B,), i, dtype=torch.long, device=xt.device)
    with torch.no_grad():
        eps = model(xt, t)
    mean = reverse_mean(xt, eps, i,  betas, alphas, alpha_bars)
    return mean + torch.sqrt(betas[i]) * torch.randn_like(xt) if i > 0 else mean

def ddpm_sample(model, n, betas, alphas, alpha_bars):
    T = len(alpha_bars)
    model.eval()
    device = next(model.parameters()).device
    x = torch.randn(n, 1, 32, 32, device=device)
    with torch.no_grad():
        for i in reversed(range(T)):
            x = ddpm_step(model, x, i, betas, alphas, alpha_bars)
    return x

def ddim_step(xt, eps, i, i_prev, alpha_bars):
    ab = alpha_bars[i]
    ab_prev = torch.ones_like(ab) if i_prev < 0 else alpha_bars[i_prev]
    x0_hat = (xt - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
    x0_hat = x0_hat.clamp(-1, 1)
    eps = (xt - torch.sqrt(ab) * x0_hat) / torch.sqrt(1 - ab)
    x_prev = torch.sqrt(ab_prev) * x0_hat + torch.sqrt(1 - ab_prev) * eps
    return x_prev

def ddim_sample(model, n, alpha_bars, n_steps=50):
    device = next(model.parameters()).device
    T = len(alpha_bars)
    steps = torch.linspace(T - 1, 0, n_steps).long().tolist()
    model.eval()
    x = torch.randn(n, 1, 32, 32, device=device)
    with torch.no_grad():
        for k, i in enumerate(steps):
            i_prev = steps[k + 1] if k + 1 < len(steps) else -1
            t = torch.full((n,), i, dtype=torch.long, device=device)
            eps = model(x, t)
            x = ddim_step(x, eps, i, i_prev, alpha_bars)
    return x

def generate(model, n, schedule, method="ddpm", n_steps=50):
    device = next(model.parameters()).device
    betas, alphas, alpha_bars = [t.to(device) for t in schedule]
    if method == "ddpm":
        return ddpm_sample(model, n, betas, alphas, alpha_bars)
    if method == "ddim":
        return ddim_sample(model, n, alpha_bars, n_steps)
    raise ValueError(f"unknown sampling method {method!r}, use 'ddpm' or 'ddim'")