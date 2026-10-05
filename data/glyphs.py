import torch 
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def render_glyph(char, font_path, size=32):
    font = ImageFont.truetype(font_path, int(size * 0.8))
    img = Image.new("L", (size, size), color=0)      # "L" = grayscale, 0 = black background
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = draw.textbbox((0, 0), char, font=font)
    w, h = right - left, bottom - top
    x = (size - w) / 2 - left                        # centre horizontally
    y = (size - h) / 2 - top                         # centre vertically
    draw.text((x, y), char, fill=255, font=font)     # white letter
    return np.array(img) 

def normalize(img):
    return (img / 127.5) - 1.0

def inv_normalize(nimg):
    return 127.5 * (nimg + 1.0)

def glyph_to_tensor(char, font_path, size=32):
    img = render_glyph(char, font_path, size)
    img_norm = normalize(img.astype(np.float32))
    x_tensor = torch.from_numpy(img_norm).unsqueeze(0)
    return x_tensor