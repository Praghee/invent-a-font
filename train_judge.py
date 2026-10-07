import os
import torch
import torch.nn as nn
from models.judge import JudgeNet
from data.dataset import FontDataset
from torch.utils.data import TensorDataset, DataLoader

def accuracy(net, loader):
    net.eval()
    device = next(net.parameters()).device
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = net(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.shape[0]
    return correct / total

def main(cfg):
    lr = cfg["lr"]
    size = cfg["size"]
    num_epochs = cfg["num_epochs"]
    batch_size = cfg["batch_size"]
    device = cfg["device"]

    print("Loading Dataset and Model!")
    ds = FontDataset("data/raw/fonts", size=size)
    font_order = torch.randperm(len(ds) // 26, generator=torch.Generator().manual_seed(0))
    train_fonts = font_order[:80].tolist()
    val_fonts = font_order[80:].tolist()

    train_idx, val_idx = [], []
    for f in train_fonts:
        train_idx.extend([f * 26 + letter for letter in range(26)])
    for f in val_fonts:
        val_idx.extend([f * 26 + letter for letter in range(26)])

    X = torch.stack([ds[i] if torch.is_tensor(ds[i]) else ds[i][0] for i in range(len(ds))])
    Y = torch.arange(len(ds)) % 26
    train_ds = TensorDataset(X[train_idx], Y[train_idx])
    val_ds   = TensorDataset(X[val_idx],   Y[val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    net = JudgeNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    print("Starting Training!")
    for epoch in range(1, num_epochs + 1):
        net.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = net(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.shape[0]
        train_loss = total_loss / len(train_ds)
        train_acc = accuracy(net, train_loader)
        val_acc = accuracy(net, val_loader)
        print(f"Epoch: {epoch}; Train_Loss: {round(train_loss, 4)}, Train_Acc: {round(train_acc, 4)}, Val_Loss: {round(val_acc, 4)}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(net.state_dict(), "checkpoints/judge.pt")

if __name__ == "__main__":
    cfg = {
        "lr": 1e-4,
        "size" : 32,
        "shuffle" : True,
        "num_epochs" : 100,
        "batch_size" : 16,
        "device" : "cuda" if torch.cuda.is_available() else "cpu"        
    }
    
    main(cfg)