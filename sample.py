import os
import torch
from models.unet import UNet
from utils.viz import show_grid
from diffusion.reverse import sample
from diffusion.schedule import make_schedule

def main(cfg):
    T = cfg["T"]
    device = cfg["device"]
    batch_size = cfg["batch_size"]
    model_name = cfg["model_name"]

    model = UNet()
    model.to(device)
    if os.path.exists(f"checkpoints/{model_name}.pt"):
        ckpt = torch.load(f"checkpoints/{model_name}.pt", map_location=device)
        model.load_state_dict(ckpt["model"])

    betas, alphas, alpha_bars = make_schedule(T)
    x = sample(model, batch_size, betas, alphas, alpha_bars)
    show_grid(x, "sample.png", ncols=4)


if __name__ == "__main__":
    cfg = {
        "T" : 1000,
        "batch_size" : 16,
        "device" : "cuda" if torch.cuda.is_available() else "cpu",      
        "model_name": "test"
    }

    main(cfg)