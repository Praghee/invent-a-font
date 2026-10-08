import math
import torch
import torch.nn.functional as F

@torch.no_grad()
def judge_probs(judge, x, batch_size=256):
    judge.eval()
    device = next(judge.parameters()).device
    out = []
    for i in range(0, len(x), batch_size):
        xb = x[i : i + batch_size].to(device).clamp(-1, 1)
        out.append(F.softmax(judge(xb), dim=1).cpu())
    return torch.cat(out)

def readability(probs, thresh=0.9):
    conf = probs.max(dim=1).values
    return {"conf": conf.mean().item(), "sure": (conf >= thresh).float().mean().item()}

def coverage(probs, min_conf=0.5):
    conf, pred = probs.max(dim=1)
    counts = torch.bincount(pred[conf >= min_conf], minlength=26).float()
    if counts.sum() == 0:
        return {"letters": 0, "evenness": 0.0}
    p = counts / counts.sum()
    p = p[p > 0]
    entropy = -(p * p.log()).sum().item()
    return {"letters": int((counts > 0).sum()), "evenness": entropy / math.log(26)}

@torch.no_grad()
def novelty(imgs, ref, copy_thresh=0.1):
    a = imgs.clamp(-1, 1).flatten(1)
    b = ref.flatten(1)
    d = torch.cdist(a, b).min(dim=1).values / (a.shape[1] ** 0.5)
    return {"nn_dist": d.mean().item(), "copy_rate": (d < copy_thresh).float().mean().item()}