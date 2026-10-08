import json
import os
import torch

from models.unet import UNet
from models.judge import JudgeNet
from diffusion.reverse import sample
from diffusion.schedule import make_schedule
from data.dataset import FontDataset
from data.dataset import dataset_to_tensors, font_split
from utils.metrics import coverage, judge_probs, novelty, readability
from utils.viz import show_grid

def get_samples(device):
    path = f"outputs/samples_{cfg['model_name']}.pt"
    if os.path.exists(path):
        print("loading cached samples:", path)
        return torch.load(path)
    ckpt_path = f"checkpoints/{cfg['model_name']}.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    betas, alphas, alpha_bars = [t.to(device) for t in make_schedule(cfg["T"])]
    model = UNet().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.eval()
    chunks = []
    for k in range(cfg["n_samples"] // cfg["chunk"]):
        chunks.append(sample(model, cfg["chunk"], betas, alphas, alpha_bars).cpu())
        print("generated", (k + 1) * cfg["chunk"])
    samples = torch.cat(chunks)
    torch.save(samples, path)
    return samples

def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    torch.manual_seed(cfg["seed"])
    os.makedirs("outputs", exist_ok=True)

    ds = FontDataset(cfg["fonts_dir"])
    X, _ = dataset_to_tensors(ds)
    train_idx, val_idx = font_split(len(ds) // 26, cfg["num_val_fonts"], cfg["seed"])
    X_train, X_val = X[train_idx], X[val_idx]

    judge = JudgeNet()
    judge.load_state_dict(torch.load("checkpoints/judge.pt", map_location=device))
    judge.to(device).eval()

    samples = get_samples(device)
    print("samples", tuple(samples.shape), "min", round(samples.min().item(), 3),
          "max", round(samples.max().item(), 3), "mean", round(samples.mean().item(), 3),
          "| real mean", round(X.mean().item(), 3))
    show_grid(samples[:16], "outputs/grid.png", ncols=4)

    noise = torch.randn_like(samples).clamp(-1, 1)
    rows = {
        "real, unseen fonts": (X_val, X_train),
        "diffusion samples": (samples, X),
        "pure noise": (noise, X),
    }
    results = {}
    for name, (imgs, ref) in rows.items():
        probs = judge_probs(judge, imgs)
        results[name] = {**readability(probs), **coverage(probs), **novelty(imgs, ref)}

    cols = ["conf", "sure", "letters", "evenness", "nn_dist", "copy_rate"]
    print("| set | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for name, r in results.items():
        vals = [f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols]
        print(f"| {name} | " + " | ".join(vals) + " |")

    with open(f"outputs/metrics_{cfg['model_name']}.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    cfg = {
        "fonts_dir": "data/raw/fonts",
        "model_name": "test",
        "T": 1000,
        "n_samples": 500,
        "chunk": 100,
        "num_val_fonts": 20,
        "seed": 0,
    }
    main(cfg)