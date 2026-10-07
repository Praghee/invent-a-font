import matplotlib.pyplot as plt
from data.glyphs import inv_normalize

def show_grid(x, path, ncols=8):
    imgs = inv_normalize(x.clamp(-1, 1)).round().byte().squeeze(1).cpu().numpy()
    n = len(imgs)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.2, nrows * 1.2))
    for k, ax in enumerate(axes.flat):
        ax.axis("off")
        if k < n:
            ax.imshow(imgs[k], cmap="gray", vmin=0, vmax=255)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()