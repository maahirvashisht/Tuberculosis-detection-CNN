"""Web demo for the trained full CNN and both independent hospital cohorts."""
import json
import warnings
from pathlib import Path
from flask import Flask, request, render_template
import numpy as np
from PIL import Image, UnidentifiedImageError
from imaging import preprocess
from model import forward, sigmoid

BASE = Path(__file__).resolve().parent
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 32_000_000
report = json.loads((BASE / "results/summary.json").read_text())
with np.load(BASE / "results/full.npz", allow_pickle=False) as f:
    params = {key:f[key] for key in f.files}

def page(result=None, error=None):
    return render_template("index.html", report=report, result=result, error=error)

@app.get("/")
def home():
    return page()

@app.get("/health")
def health():
    return {"status":"ready"}

@app.errorhandler(413)
def too_large(_):
    return page(error="Choose a PNG or JPEG smaller than 12 MB."), 413

@app.post("/")
def predict():
    uploaded = request.files.get("image")
    if not uploaded:
        return page(error="Choose a public dataset X-ray first."),400
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error",Image.DecompressionBombWarning)
            with Image.open(uploaded.stream) as image:
                if image.format not in {"PNG","JPEG"}:
                    raise ValueError("Unsupported image format")
                x = preprocess(image)
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombWarning, Image.DecompressionBombError):
        return page(error="Use a readable PNG or JPEG with visible grayscale variation and at most 32 megapixels."),400
    score = float(sigmoid(forward(x[None],params)[0])[0])
    return page(result={"score":score, "positive":score>=report["threshold"]})
