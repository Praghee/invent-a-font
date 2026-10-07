import torch

def reverse_mean(xt, eps, i, betas, alphas, alpha_bars):
    coef_x = 1 / torch.sqrt(alphas[i])
    coef_eps = betas[i] / torch.sqrt(1 - alpha_bars[i])
    return coef_x * (xt - coef_eps * eps)

def p_sample(model, xt, i, betas, alphas, alpha_bars):
    B = xt.shape[0]
    t = torch.full((B,), i, dtype=torch.long, device=xt.device)
    with torch.no_grad():
        eps = model(xt, t)
    mean = reverse_mean(xt, eps, i,  betas, alphas, alpha_bars)
    return mean + torch.sqrt(betas[i]) * torch.rand_like(xt) if i > 0 else mean