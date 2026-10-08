import os
import time
import torch
from models.unet import UNet
from utils.viz import show_grid
from diffusion.reverse import generate
from diffusion.schedule import make_schedule

def main(cfg):
    device = cfg["device"]
    tag = "ddpm" if cfg["sampling"] == "ddpm" else f"ddim{cfg['ddim_steps']}"
    ckpt_path = f"checkpoints/{cfg['model_name']}.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file {ckpt_path} not found!")

    model = UNet().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])

    torch.manual_seed(cfg["seed"])
    schedule = make_schedule(cfg["T"])
    t0 = time.time()
    x = generate(model, cfg["batch_size"], schedule, cfg["sampling"], cfg["ddim_steps"])
    print(f"{tag} sampling took {time.time() - t0:.1f} s")

    os.makedirs("outputs", exist_ok=True)
    show_grid(x.cpu(), f"outputs/sample_{tag}.png", ncols=4)

if __name__ == "__main__":
    cfg = {
        "T" : 1000,
        "batch_size" : 16,
        "device" : "cuda" if torch.cuda.is_available() else "cpu",      
        "model_name": "test",
        "sampling": "ddim",
        "ddim_steps": 100,
        "seed": 0
    }

    main(cfg)