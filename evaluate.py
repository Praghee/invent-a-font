import json
import os
import time
import torch
from models.unet import UNet
from models.judge import JudgeNet
from utils.viz import show_grid
from utils.metrics import coverage, judge_probs, novelty, readability
from data.dataset import FontDataset, dataset_to_tensors, font_split
from diffusion.reverse import generate
from diffusion.schedule import make_schedule

def load_model(cfg, device):
    ckpt_path = f"checkpoints/{cfg['model_name']}.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    model = UNet().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    return model

def get_samples(cfg, device, tag):
    path = f"outputs/samples_{cfg['model_name']}_{tag}.pt"
    if os.path.exists(path):
        print("loading cached samples:", path)
        return torch.load(path)
    torch.manual_seed(cfg["seed"])
    schedule = make_schedule(cfg["T"])
    model = load_model(cfg, device)
    chunks = []
    for k in range(cfg["n_samples"] // cfg["chunk"]):
        x = generate(model, cfg["chunk"], schedule, cfg["sampling"], cfg["ddim_steps"])
        chunks.append(x.cpu())
        print("generated", (k + 1) * cfg["chunk"])
    samples = torch.cat(chunks)
    torch.save(samples, path)
    return samples

def load_assets(cfg, device):
    ds = FontDataset(cfg["fonts_dir"])
    X, _ = dataset_to_tensors(ds)
    train_idx, val_idx = font_split(len(ds) // 26, cfg["num_val_fonts"], cfg["seed"])
    judge = JudgeNet()
    judge.load_state_dict(torch.load("checkpoints/judge.pt", map_location=device))
    judge.to(device).eval()
    return X, X[train_idx], X[val_idx], judge

def score(judge, imgs, ref):
    probs = judge_probs(judge, imgs)
    return {**readability(probs), **coverage(probs), **novelty(imgs, ref)}

def time_it(model, schedule, method, steps, n=16):
    t0 = time.time()
    generate(model, n, schedule, method, steps)
    return time.time() - t0

def main(cfg):
    if cfg["sampling"] not in ("ddpm", "ddim", "compare"):
        raise ValueError(f"sampling must be 'ddpm', 'ddim' or 'compare', got {cfg['sampling']!r}")
    compare = cfg["sampling"] == "compare"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| mode:", cfg["sampling"])
    os.makedirs("outputs", exist_ok=True)

    X, X_train, X_val, judge = load_assets(cfg, device)

    if compare:
        runs = cfg["compare_runs"]
        schedule = make_schedule(cfg["T"])
        model = load_model(cfg, device)
        generate(model, 2, schedule, "ddim", 2)          # warm-up, so timings are fair
    else:
        runs = [(cfg["sampling"], cfg["ddim_steps"])]

    results, first_rows = {}, []
    for method, steps in runs:
        run_cfg = dict(cfg, sampling=method, ddim_steps=steps)
        tag = "ddpm" if method == "ddpm" else f"ddim{steps}"
        samples = get_samples(run_cfg, device, tag)
        print(tag, "samples", tuple(samples.shape), "min", round(samples.min().item(), 3),
              "max", round(samples.max().item(), 3), "mean", round(samples.mean().item(), 3),
              "| real mean", round(X.mean().item(), 3))
        show_grid(samples[:16], f"outputs/grid_{tag}.png", ncols=4)
        results[tag] = score(judge, samples, X)
        if compare:
            results[tag]["sec_per_16"] = time_it(model, schedule, method, steps)
        first_rows.append(samples[:8])

    torch.manual_seed(cfg["seed"])
    noise = torch.randn(cfg["n_samples"], 1, 32, 32).clamp(-1, 1)
    results["real, unseen fonts"] = score(judge, X_val, X_train)
    results["pure noise"] = score(judge, noise, X)

    cols = ["conf", "sure", "letters", "evenness", "nn_dist", "copy_rate"]
    if compare:
        cols = ["sec_per_16"] + cols
    print("| set | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for name, r in results.items():
        vals = [f"{r[c]:.3f}" if isinstance(r.get(c), float) else str(r.get(c, "-")) for c in cols]
        print(f"| {name} | " + " | ".join(vals) + " |")

    out = f"compare_{cfg['model_name']}" if compare else f"metrics_{cfg['model_name']}_{tag}"
    with open(f"outputs/{out}.json", "w") as f:
        json.dump(results, f, indent=2)
    if compare:
        show_grid(torch.cat(first_rows), "outputs/compare_grid.png", ncols=8)

if __name__ == "__main__":
    cfg = {
        "fonts_dir": "data/raw/fonts",
        "model_name": "test",
        "sampling": "compare",          # "ddpm" | "ddim" | "compare"
        "ddim_steps": 50,               # used when sampling == "ddim"
        "compare_runs": [("ddpm", 1000), ("ddim", 10), ("ddim", 25), ("ddim", 50), ("ddim", 100)],
        "T": 1000,
        "n_samples": 500,
        "chunk": 100,
        "num_val_fonts": 20,
        "seed": 0,
    }
    main(cfg)