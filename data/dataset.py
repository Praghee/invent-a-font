import os
import torch
import string
from data.glyphs import glyph_to_tensor

def font_split(num_fonts=100, num_val=20, seed=0, per_font=26):
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(num_fonts, generator=g).tolist()
    train_fonts = order[: num_fonts - num_val]
    val_fonts = order[num_fonts - num_val :]
    train_idx = [f * per_font + l for f in train_fonts for l in range(per_font)]
    val_idx = [f * per_font + l for f in val_fonts for l in range(per_font)]
    return train_idx, val_idx

def dataset_to_tensors(ds):
    imgs = []
    for i in range(len(ds)):
        item = ds[i]
        imgs.append(item if torch.is_tensor(item) else item[0])
    return torch.stack(imgs), torch.arange(len(ds)) % 26

class FontDataset(torch.utils.data.Dataset):
    def __init__(self, font_dir, size=32):
        fonts = sorted([f for f in os.listdir(font_dir) if f.lower().endswith('.ttf')])
        self.font_names, data, labels = list(), list(), list()
        for font in fonts:
            self.font_names.append(font[:-4])
            for label, char in enumerate(string.ascii_uppercase):
                font_path = os.path.join(font_dir, font)
                data.append(glyph_to_tensor(char, font_path, size))
                labels.append(label)
        self.data = torch.stack(data, dim=0)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]