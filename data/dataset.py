import os
import torch
import string
from data.glyphs import glyph_to_tensor

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