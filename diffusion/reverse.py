import torch

def predict_eps(model, xt, t, y=None, w=1.0):
    with torch.no_grad():
        if y is None:
            return model(xt, t)                        # blank card only
        if w == 1.0:
            return model(xt, t, y)                     # plain conditional, one pass
        blank = torch.full_like(y, model.num_classes)  # blank card id
        eps_u, eps_c = model(torch.cat([xt, xt]), torch.cat([t, t]), torch.cat([blank, y])).chunk(2)
        return eps_u + w * (eps_c - eps_u)

def reverse_mean(xt, eps, i, betas, alphas, alpha_bars):
    coef_x = 1 / torch.sqrt(alphas[i])
    coef_eps = betas[i] / torch.sqrt(1 - alpha_bars[i])
    return coef_x * (xt - coef_eps * eps)

def start_noise(n, device, xT=None):
    if xT is None:
        return torch.randn(n, 1, 32, 32, device=device)
    if xT.shape != (n, 1, 32, 32):
        raise ValueError(f"xT must have shape {(n, 1, 32, 32)}, got {tuple(xT.shape)}")
    return xT.to(device)

def ddpm_step(model, xt, i, betas, alphas, alpha_bars, y=None, w=1.0):
    B = xt.shape[0]
    t = torch.full((B,), i, dtype=torch.long, device=xt.device)
    eps = predict_eps(model, xt, t, y, w)
    mean = reverse_mean(xt, eps, i, betas, alphas, alpha_bars)
    return mean + torch.sqrt(betas[i]) * torch.randn_like(xt) if i > 0 else mean

def ddpm_sample(model, n, betas, alphas, alpha_bars, y=None, w=1.0, xT=None):
    device = next(model.parameters()).device
    xt = start_noise(n, device, xT)
    for i in reversed(range(len(betas))):
        xt = ddpm_step(model, xt, i, betas, alphas, alpha_bars, y, w)
    return xt

def ddim_step(xt, eps, i, i_prev, alpha_bars, clip_x0=False):
    ab = alpha_bars[i]
    ab_prev = torch.ones_like(ab) if i_prev < 0 else alpha_bars[i_prev]
    x0_hat = (xt - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
    if clip_x0:
        x0_hat = x0_hat.clamp(-1, 1)
        eps = (xt - torch.sqrt(ab) * x0_hat) / torch.sqrt(1 - ab)
    return torch.sqrt(ab_prev) * x0_hat + torch.sqrt(1 - ab_prev) * eps

def ddim_sample(model, n, alpha_bars, n_steps=50, clip_x0=True, y=None, w=1.0, xT=None, return_traj=False):
    device = next(model.parameters()).device
    T = len(alpha_bars)
    steps = torch.linspace(T - 1, 0, n_steps).long().tolist()
    xt = start_noise(n, device, xT)
    traj = [xt.cpu()] if return_traj else None
    for k, i in enumerate(steps):
        i_prev = steps[k + 1] if k + 1 < len(steps) else -1
        t = torch.full((n,), i, dtype=torch.long, device=device)
        eps = predict_eps(model, xt, t, y, w)
        xt = ddim_step(xt, eps, i, i_prev, alpha_bars, clip_x0)
        if return_traj:
            traj.append(xt.cpu())
    return (xt, traj) if return_traj else xt

def generate(model, n, schedule, method="ddpm", n_steps=50, y=None, w=1.0, xT=None, return_traj=False):
    device = next(model.parameters()).device
    if y is not None:
        y = y.to(device)
        if len(y) != n:
            raise ValueError(f"y has {len(y)} labels but n={n}")
    betas, alphas, alpha_bars = [t.to(device) for t in schedule]
    if method == "ddpm":
        if return_traj:
            raise ValueError("return_traj is only supported for ddim")
        return ddpm_sample(model, n, betas, alphas, alpha_bars, y, w, xT)
    if method == "ddim":
        return ddim_sample(model, n, alpha_bars, n_steps, y=y, w=w, xT=xT, return_traj=return_traj)
    raise ValueError(f"unknown sampling method {method!r}, use 'ddpm' or 'ddim'")