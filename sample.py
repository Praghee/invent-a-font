import os
import time
import torch
from models.unet import UNet
from utils.viz import show_grid
from diffusion.reverse import generate
from diffusion.schedule import make_schedule


def main(cfg):
    device = cfg["device"]
    cond = cfg["conditional"]
    tag = "ddpm" if cfg["sampling"] == "ddpm" else f"ddim{cfg['ddim_steps']}"
    if cond:
        tag += f"_w{cfg['guidance']:g}"

    ckpt_path = f"checkpoints/{cfg['model_name']}.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file {ckpt_path} not found!")

    model = UNet(num_classes=26 if cond else None).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.eval()

    # which letters to draw
    if cond:
        letters = cfg["letters"].upper()
        n = len(letters)
        y = torch.tensor([ord(c) - ord("A") for c in letters], device=device)
        ncols = 13 if n == 26 else min(n, 8)
    else:
        n, y, ncols = cfg["batch_size"], None, 4

    torch.manual_seed(cfg["seed"])
    schedule = make_schedule(cfg["T"])
    t0 = time.time()
    x = generate(model, n, schedule, cfg["sampling"], cfg["ddim_steps"], y=y, w=cfg["guidance"])
    print(f"{tag} sampling took {time.time() - t0:.1f} s")

    os.makedirs("outputs", exist_ok=True)
    show_grid(x.cpu(), f"outputs/sample_{cfg['model_name']}_{tag}.png", ncols=ncols)


if __name__ == "__main__":
    cfg = {
        "T": 1000,
        "batch_size": 16,            # used only when conditional is False
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_name": "cond_best",
        "conditional": True,
        "letters": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "guidance": 3.0,             # w: 1.0 = plain conditional, higher = stronger push
        "sampling": "ddim",
        "ddim_steps": 100,
        "seed": 0,
    }

    main(cfg)