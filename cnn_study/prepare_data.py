import argparse
import csv
import hashlib
import io
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
from imaging import preprocess

BASE = Path(__file__).resolve().parent

def main(source, destination):
    with (BASE / "data" / "manifest.csv").open() as f:
        records = list(csv.DictReader(f))
    if len({r["case"] for r in records}) != len(records):
        raise ValueError("Duplicate case ID in manifest")
    source = Path(source)
    archive = zipfile.ZipFile(source) if source.is_file() else None
    arrays = []
    try:
        for i, row in enumerate(records):
            raw = archive.read(row["file"]) if archive else (source / row["file"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError("Unexpected image bytes for " + row["case"])
            with Image.open(io.BytesIO(raw)) as im:
                arrays.append(preprocess(im))
            if (i + 1) % 100 == 0:
                print(f"Prepared {i+1} radiographs", flush=True)
    finally:
        if archive:
            archive.close()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, images=np.stack(arrays),
                        cases=[r["case"] for r in records])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Kaggle ZIP or its extracted root")
    parser.add_argument("--out", default="image_cache.npz")
    args = parser.parse_args()
    main(args.source, args.out)
