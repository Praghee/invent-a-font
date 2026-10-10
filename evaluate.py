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
    model = UNet(num_classes=26 if cfg["conditional"] else None).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.eval()
    return model


def make_labels(cfg):
    # 20 images per letter, ordered A..A, B..B, ..., Z..Z
    if not cfg["conditional"]:
        return None
    return torch.arange(26).repeat_interleave(cfg["n_samples"] // 26)


def get_samples(cfg, device, tag, w):
    path = f"outputs/samples_{cfg['model_name']}_{tag}.pt"
    if os.path.exists(path):
        print("loading cached samples:", path)
        return torch.load(path)
    torch.manual_seed(cfg["seed"])
    schedule = make_schedule(cfg["T"])
    model = load_model(cfg, device)
    y_all = make_labels(cfg)
    chunks = []
    for k in range(cfg["n_samples"] // cfg["chunk"]):
        lo, hi = k * cfg["chunk"], (k + 1) * cfg["chunk"]
        y = y_all[lo:hi] if y_all is not None else None
        x = generate(model, cfg["chunk"], schedule, cfg["sampling"], cfg["ddim_steps"], y=y, w=w)
        chunks.append(x.cpu())
        print("generated", hi)
    samples = torch.cat(chunks)
    torch.save(samples, path)
    return samples


def load_assets(cfg, device):
    ds = FontDataset(cfg["fonts_dir"])
    X, L = dataset_to_tensors(ds)
    train_idx, val_idx = font_split(len(ds) // 26, cfg["num_val_fonts"], cfg["seed"])
    judge = JudgeNet()
    judge.load_state_dict(torch.load("checkpoints/judge.pt", map_location=device))
    judge.to(device).eval()
    return X, X[train_idx], X[val_idx], L[val_idx], judge


def score(judge, imgs, ref, y=None):
    probs = judge_probs(judge, imgs)
    r = {**readability(probs), **coverage(probs), **novelty(imgs, ref)}
    if y is not None:
        probs, y = probs.cpu(), y.cpu()
        r["acc"] = (probs.argmax(1) == y).float().mean().item()
        r["p_true"] = probs[torch.arange(len(y)), y].mean().item()
        r["per_letter"] = [
            (probs[y == k].argmax(1) == k).float().mean().item() for k in range(26)
        ]
    return r


def time_it(model, schedule, method, steps, w, conditional, n=16):
    y = torch.arange(n) % 26 if conditional else None
    t0 = time.time()
    generate(model, n, schedule, method, steps, y=y, w=w)
    return time.time() - t0


def preview(samples, cfg):
    # conditional: one image per letter (A..Z); unconditional: first 16
    if cfg["conditional"]:
        return samples[:: cfg["n_samples"] // 26]
    return samples[:16]


def main(cfg):
    mode = cfg["sampling"]
    if mode not in ("ddpm", "ddim", "compare", "sweep"):
        raise ValueError(f"sampling must be 'ddpm', 'ddim', 'compare' or 'sweep', got {mode!r}")
    cond = cfg["conditional"]
    if mode == "sweep" and not cond:
        raise ValueError("sweep needs conditional=True")
    if cond and (cfg["n_samples"] % 26 or cfg["n_samples"] % cfg["chunk"]):
        raise ValueError("n_samples must be divisible by 26 and by chunk (e.g. 520 and 130)")

    multi = mode in ("compare", "sweep")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| mode:", mode, "| conditional:", cond)
    os.makedirs("outputs", exist_ok=True)

    X, X_train, X_val, L_val, judge = load_assets(cfg, device)
    w0 = cfg["guidance"] if cond else 1.0

    # each run = (method, steps, w)
    if mode == "compare":
        runs = [(m, s, w0) for m, s in cfg["compare_runs"]]
    elif mode == "sweep":
        runs = [("ddim", cfg["ddim_steps"], w) for w in cfg["guidance_sweep"]]
    else:
        runs = [(mode, cfg["ddim_steps"], w0)]

    if multi:
        schedule = make_schedule(cfg["T"])
        model = load_model(cfg, device)
        generate(model, 2, schedule, "ddim", 2)          # warm-up, so timings are fair

    y_gen = make_labels(cfg)
    results, first_rows = {}, []
    for method, steps, w in runs:
        run_cfg = dict(cfg, sampling=method, ddim_steps=steps)
        tag = "ddpm" if method == "ddpm" else f"ddim{steps}"
        if cond:
            tag += f"_w{w:g}"
        samples = get_samples(run_cfg, device, tag, w)
        print(tag, "samples", tuple(samples.shape), "min", round(samples.min().item(), 3),
              "max", round(samples.max().item(), 3), "mean", round(samples.mean().item(), 3),
              "| real mean", round(X.mean().item(), 3))
        show_grid(preview(samples, cfg), f"outputs/grid_{cfg['model_name']}_{tag}.png",
                  ncols=13 if cond else 4)
        results[tag] = score(judge, samples, X, y_gen)
        if mode == "compare":
            results[tag]["sec_per_16"] = time_it(model, schedule, method, steps, w, cond)
        first_rows.append(preview(samples, cfg)[:8])

    torch.manual_seed(cfg["seed"])
    noise = torch.randn(cfg["n_samples"], 1, 32, 32).clamp(-1, 1)
    results["real, unseen fonts"] = score(judge, X_val, X_train, L_val if cond else None)
    results["pure noise"] = score(judge, noise, X, torch.arange(cfg["n_samples"]) % 26 if cond else None)

    cols = ["conf", "sure", "letters", "evenness", "nn_dist", "copy_rate"]
    if cond:
        cols = ["acc", "p_true"] + cols
    if mode == "compare":
        cols = ["sec_per_16"] + cols
    print("| set | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for name, r in results.items():
        vals = [f"{r[c]:.3f}" if isinstance(r.get(c), float) else str(r.get(c, "-")) for c in cols]
        print(f"| {name} | " + " | ".join(vals) + " |")

    if cond:
        print("\nper-letter accuracy (A..Z):")
        for name, r in results.items():
            if "per_letter" in r:
                print(f"{name:>24}: " + " ".join(f"{a:.2f}" for a in r["per_letter"]))

    if mode == "compare":
        out = f"compare_{cfg['model_name']}"
    elif mode == "sweep":
        out = f"sweep_{cfg['model_name']}"
    else:
        out = f"metrics_{cfg['model_name']}_{tag}"
    with open(f"outputs/{out}.json", "w") as f:
        json.dump(results, f, indent=2)
    if multi:
        show_grid(torch.cat(first_rows), f"outputs/{out}_grid.png", ncols=8)


if __name__ == "__main__":
    cfg = {
        "fonts_dir": "data/raw/fonts",
        "model_name": "cond_best",
        "conditional": True,
        "guidance": 3.0,                          # w for ddpm / ddim / compare
        "guidance_sweep": [0.0, 1.0, 2.0, 3.0, 5.0],
        "sampling": "ddim",                       # "ddpm" | "ddim" | "compare" | "sweep"
        "ddim_steps": 100,
        "compare_runs": [("ddpm", 1000), ("ddim", 10), ("ddim", 25), ("ddim", 50), ("ddim", 100)],
        "T": 1000,
        "n_samples": 520,                         # 20 per letter when conditional
        "chunk": 130,
        "num_val_fonts": 20,
        "seed": 0,
    }
    main(cfg)