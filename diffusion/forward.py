import torch

def q_sample(x0, t, noise, alpha_bars):
    a_bar = alpha_bars[t]
    sqrt_a_bar = torch.sqrt(a_bar).reshape(-1, 1 ,1, 1)
    sqrt_one_minus_a_bar = torch.sqrt(1 - a_bar).reshape(-1, 1, 1, 1)
    xt = (sqrt_a_bar * x0) + (sqrt_one_minus_a_bar * noise)
    return xt
