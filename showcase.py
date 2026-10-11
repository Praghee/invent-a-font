import os
import torch
from PIL import Image
from models.unet import UNet
from utils.viz import show_grid
from data.dataset import FontDataset, dataset_to_tensors
from diffusion.reverse import generate
from diffusion.schedule import make_schedule

def load_model(cfg, device):
    ckpt_path = f"checkpoints/{cfg['model_name']}.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    model = UNet(num_classes=26).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.eval()
    return model

def alphabet_sheet(model, schedule, cfg):
    y = torch.arange(26)
    rows = []
    for seed in cfg["seeds"]:
        g = torch.Generator().manual_seed(seed)
        xT = torch.randn(1, 1, 32, 32, generator=g).repeat(26, 1, 1, 1)   # one noise = one typeface
        x = generate(model, 26, schedule, "ddim", cfg["ddim_steps"], y=y, w=cfg["guidance"], xT=xT)
        rows.append(x.cpu())
    sheet = torch.cat(rows)                                               # (6 * 26, 1, 32, 32)
    path = f"outputs/alphabet_sheet_{cfg['model_name']}.png"
    show_grid(sheet, path, ncols=13)
    print("saved", path, tuple(sheet.shape))
    return sheet

def guidance_strip(model, schedule, cfg):
    letters = cfg["strip_letters"]
    y = torch.tensor([ord(c) - ord("A") for c in letters])
    xT = torch.cat([
        torch.randn(1, 1, 32, 32, generator=torch.Generator().manual_seed(s))
        for s in cfg["strip_seeds"]
    ])                                                                    # one noise per row
    cols = []
    for w in cfg["strip_w"]:
        x = generate(model, len(letters), schedule, "ddim", cfg["ddim_steps"], y=y, w=w, xT=xT)
        cols.append(x.cpu())
    grid = torch.stack(cols, 1).flatten(0, 1)                             # (rows * W, 1, 32, 32)
    path = f"outputs/guidance_strip_{cfg['model_name']}.png"
    show_grid(grid, path, ncols=len(cfg["strip_w"]))
    print("saved", path, tuple(grid.shape))

def to_frame(x, scale=4):                                                 # x: (26, 1, 32, 32) in [-1, 1]
    img = ((x.clamp(-1, 1) + 1) / 2 * 255).byte().squeeze(1)
    img = img.view(2, 13, 32, 32).permute(0, 2, 1, 3).reshape(64, 13 * 32)
    img = img.repeat_interleave(scale, 0).repeat_interleave(scale, 1)
    return Image.fromarray(img.numpy())

def denoise_gif(model, schedule, cfg):
    y = torch.arange(26)
    g = torch.Generator().manual_seed(cfg["gif_seed"])
    xT = torch.randn(1, 1, 32, 32, generator=g).repeat(26, 1, 1, 1)
    _, traj = generate(model, 26, schedule, "ddim", cfg["ddim_steps"],
                       y=y, w=cfg["guidance"], xT=xT, return_traj=True)
    frames = [to_frame(x) for x in traj]
    frames += [frames[-1]] * 20                                           # hold the final alphabet
    path = f"outputs/denoise_{cfg['model_name']}.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=50, loop=0)
    print("saved", path, len(frames), "frames")

def copy_check(sheet, cfg, thr=0.1):
    X, _ = dataset_to_tensors(FontDataset(cfg["fonts_dir"]))              # all 2600 real glyphs
    a, b = sheet.flatten(1), X.flatten(1)
    d = torch.cdist(a, b) / a.shape[1] ** 0.5                             # (156, 2600) RMS pixel distance
    nn_d, nn_i = d.min(1)
    flagged = (nn_d < thr).nonzero().flatten().tolist()
    print(f"nearest-training distance: mean {nn_d.mean():.3f}, min {nn_d.min():.3f}")
    print(f"near copies (< {thr}): {len(flagged)} of {len(sheet)}")
    for k in flagged:
        print(f"  typeface {k // 26 + 1}, letter {chr(65 + k % 26)}: dist {nn_d[k]:.3f}")
    if flagged:
        pairs = torch.stack([t for k in flagged for t in (sheet[k], X[nn_i[k]])])
        show_grid(pairs, f"outputs/copy_check_{cfg['model_name']}.png", ncols=8)   # generated | real

def main(cfg):
    device = cfg["device"]
    os.makedirs("outputs", exist_ok=True)
    model = load_model(cfg, device)
    schedule = make_schedule(cfg["T"])
    sheet = alphabet_sheet(model, schedule, cfg)
    copy_check(sheet, cfg)
    guidance_strip(model, schedule, cfg)
    denoise_gif(model, schedule, cfg)

if __name__ == "__main__":
    cfg = {
        "fonts_dir": "data/raw/fonts",
        "T": 1000,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_name": "cond_best",
        "guidance": 3.0,
        "ddim_steps": 100,
        "seeds": [1, 2, 3, 4, 5, 6],
        "strip_letters": "AGRS",
        "strip_seeds": [1, 2, 3, 4],
        "strip_w": [0.0, 1.0, 2.0, 3.0, 5.0],
        "gif_seed": 5,
    }
    main(cfg)