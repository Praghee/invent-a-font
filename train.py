import os
import time
import torch
from models.unet import UNet
from data.dataset import FontDataset
from diffusion.forward import q_sample
from diffusion.schedule import make_schedule

def main(cfg):
    lr = cfg["lr"]
    T = cfg["T"]
    size = cfg["size"]
    shuffle = cfg["shuffle"]
    num_epochs = cfg["num_epochs"]
    batch_size = cfg["batch_size"]
    device = cfg["device"]
    cond = cfg["conditional"]
    num_classes = cfg["num_classes"]
    ckpt_path = f"checkpoints/{cfg['ckpt']}.pt"

    print("Loading Dataset and Model!")
    ds = FontDataset("data/raw/fonts", size=size)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    _, _, alpha_bars = make_schedule(T)
    model = UNet(num_classes=num_classes if cond else None)
    model.to(device)
    alpha_bars = alpha_bars.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    print("Starting Training!")
    start_epoch = 1
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from epoch {start_epoch - 1}")

    model.train()
    for epoch in range(start_epoch, start_epoch + num_epochs):
        start = time.time()
        total = 0
        for x0, y in loader:
            x0, y = x0.to(device), y.to(device)
            batch = x0.shape[0] 
            noise = torch.randn_like(x0)
            t = torch.randint(0, T, (batch,)).to(device)
            xt = q_sample(x0, t, noise, alpha_bars)
            if cond:
                drop = torch.rand(batch, device=device) < cfg["p_uncond"]
                y = torch.where(drop, torch.full_like(y, num_classes), y)
                pred = model(xt, t, y)
            else:
                pred = model(xt, t)
            loss = torch.mean((pred - noise) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * batch
        epoch_loss = total / len(ds)
        end = time.time()

        ckpt = {
            "model" : model.state_dict(),
            "opt" : opt.state_dict(),
            "epoch" : epoch,
            "loss" : epoch_loss
        }
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(ckpt, ckpt_path)
        print(f"Loss on epoch {epoch}: {epoch_loss} and took {(end - start):.3f} seconds per epoch")

if __name__ == "__main__":
    cfg = {
        "lr": 1e-3,
        "T" : 1000,
        "size" : 32,
        "shuffle" : True,
        "num_epochs" : 100,
        "batch_size" : 16,
        "device" : "cuda" if torch.cuda.is_available() else "cpu",    
        "conditional": True,
        "num_classes": 26,
        "p_uncond": 0.1,            # chance that a training example gets the blank card.
        "ckpt": "cond_last"
    }
    
    main(cfg)