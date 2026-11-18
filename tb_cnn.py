import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)

IMG = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
MODEL_PATH = "tb_cnn_best.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"


class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 32), block(32, 64),
                                      block(64, 128), block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.5), nn.Linear(256, 1))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(name):
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(m.fc.in_features, 1))
        return m
    return SimpleCNN()


def get_loaders(root, bs):
    train_tf = transforms.Compose([
        transforms.Resize((IMG, IMG)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG, IMG)),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def folder(path, tf):
        return datasets.ImageFolder(path, tf)

    if os.path.isdir(os.path.join(root, "train")):
        tr = folder(os.path.join(root, "train"), train_tf)
        classes = tr.classes
        if os.path.isdir(os.path.join(root, "val")) and os.path.isdir(os.path.join(root, "test")):
            va = folder(os.path.join(root, "val"), eval_tf)
            te = folder(os.path.join(root, "test"), eval_tf)
            return tr, va, te, classes
        base_dir = os.path.join(root, "train")
    else:
        base_dir = root
        classes = folder(base_dir, eval_tf).classes

    full_train = folder(base_dir, train_tf)
    full_eval = folder(base_dir, eval_tf)
    n = len(full_train)
    g = torch.Generator().manual_seed(42)
    idx = torch.randperm(n, generator=g).tolist()
    n_tr, n_va = int(0.7 * n), int(0.15 * n)
    tr = Subset(full_train, idx[:n_tr])
    va = Subset(full_eval, idx[n_tr:n_tr + n_va])
    te = Subset(full_eval, idx[n_tr + n_va:])
    return tr, va, te, classes


def labels_of(ds):
    if isinstance(ds, Subset):
        return [ds.dataset.targets[i] for i in ds.indices]
    return ds.targets


def run_epoch(model, loader, loss_fn, opt=None):
    train = opt is not None
    model.train(train)
    tot, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.float().to(device)
            out = model(x).squeeze(1)
            loss = loss_fn(out, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot += loss.item() * len(y)
            correct += ((out > 0).float() == y).sum().item()
            n += len(y)
    return tot / n, correct / n


def evaluate(model, loader, classes):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            probs += torch.sigmoid(model(x.to(device)).squeeze(1)).cpu().tolist()
            ys += y.tolist()
    probs, ys = np.array(probs), np.array(ys)
    preds = (probs > 0.5).astype(int)
    print("\n=== Test results ===")
    print(classification_report(ys, preds, target_names=classes, digits=3))
    print("Confusion matrix (rows=true, cols=pred):\n", confusion_matrix(ys, preds))
    print("ROC-AUC: %.4f" % roc_auc_score(ys, probs))


def train(args):
    tr, va, te, classes = get_loaders(args.data, args.bs)
    print("Classes:", classes, "| train/val/test:", len(tr), len(va), len(te))

    ys = np.array(labels_of(tr))
    pos_weight = torch.tensor((ys == 0).sum() / max((ys == 1).sum(), 1)).float().to(device)

    tl = DataLoader(tr, args.bs, shuffle=True, num_workers=2)
    vl = DataLoader(va, args.bs, num_workers=2)
    tel = DataLoader(te, args.bs, num_workers=2)

    model = build_model(args.model).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)

    best, bad = 1e9, 0
    for ep in range(1, args.epochs + 1):
        trl, tra = run_epoch(model, tl, loss_fn, opt)
        vll, vla = run_epoch(model, vl, loss_fn)
        sched.step(vll)
        print(f"ep {ep:02d} | train loss {trl:.4f} acc {tra:.3f} | val loss {vll:.4f} acc {vla:.3f}")
        if vll < best:
            best, bad = vll, 0
            torch.save({"model": model.state_dict(), "name": args.model,
                        "classes": classes}, MODEL_PATH)
        else:
            bad += 1
            if bad >= 7:
                print("Early stopping")
                break

    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    evaluate(model, tel, classes)


def predict(path):
    from PIL import Image
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model = build_model(ckpt["name"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tf = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor(),
                             transforms.Normalize(MEAN, STD)])
    x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.sigmoid(model(x)).item()
    print(f"P(TB) = {p:.3f} -> {ckpt['classes'][int(p > 0.5)]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", choices=["simple", "resnet18"], default="simple")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--predict", help="path to a single X-ray image")
    a = ap.parse_args()
    predict(a.predict) if a.predict else train(a)