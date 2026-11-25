import numpy as np
from PIL import Image, ImageOps

SIZE = 64

def preprocess(image):
    # Convert palette/RGB values to luminance; preserve higher-bit grayscale.
    gray = image.convert("F")
    gray = ImageOps.contain(gray, (SIZE, SIZE), method=Image.Resampling.BILINEAR)
    pixels = np.asarray(gray, dtype=np.float32)
    low, high = np.percentile(pixels, [1, 99])
    if high <= low or not np.isfinite([low, high]).all():
        raise ValueError("The image has insufficient grayscale variation.")
    scaled = np.clip((pixels - low) / (high - low), 0, 1)
    canvas = np.zeros((SIZE, SIZE), dtype=np.float32)
    h, w = scaled.shape
    y, x = (SIZE-h)//2, (SIZE-w)//2
    canvas[y:y+h, x:x+w] = scaled
    return canvas
