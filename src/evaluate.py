"""Step 3: evaluate best checkpoints on the held-out test split + measure inference cost.

Usage:
    python src/evaluate.py --model resnet50 [--model efficientnet_b0 ...]
    python src/evaluate.py --all

Writes results/<model>/{test_metrics.json, test_probs.npy, confusion_matrix.png}
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)

import config as C
import data
import models


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, targets = [], []
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x.to(device, non_blocking=True))
        probs.append(out.float().softmax(1).cpu())
        targets.append(y)
    return torch.cat(probs).numpy(), torch.cat(targets).numpy()


@torch.no_grad()
def latency(model, device, batch=1, iters=50, warmup=10):
    model.eval()
    x = torch.randn(batch, 3, C.IMG_SIZE, C.IMG_SIZE, device=device)
    for _ in range(warmup):
        with torch.autocast("cuda", dtype=torch.float16):
            model(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.autocast("cuda", dtype=torch.float16):
            model(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000  # ms per batch


def metrics_from_probs(probs, targets):
    preds = probs.argmax(1)
    rep = classification_report(targets, preds, target_names=C.CLASSES, output_dict=True, zero_division=0)
    return dict(
        accuracy=float(accuracy_score(targets, preds)),
        balanced_accuracy=float(balanced_accuracy_score(targets, preds)),
        macro_f1=float(f1_score(targets, preds, average="macro")),
        weighted_f1=float(f1_score(targets, preds, average="weighted")),
        macro_auc=float(roc_auc_score(targets, probs, multi_class="ovr", average="macro")),
        per_class={c: {k: float(rep[c][k]) for k in ("precision", "recall", "f1-score", "support")} for c in C.CLASSES},
        confusion=confusion_matrix(targets, preds).tolist(),
    )


def plot_confusion(cm, title, path):
    cm = np.array(cm)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(C.CLASSES)), C.CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(C.CLASSES)), C.CLASSES)
    for i in range(len(C.CLASSES)):
        for j in range(len(C.CLASSES)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def evaluate_model(key, device, test_loader):
    out_dir = C.RESULTS_DIR / key
    model = models.create(key, pretrained=False)
    ck = torch.load(out_dir / "best.pt", map_location="cpu")
    model.load_state_dict(ck["model"])
    model.to(device)

    probs, targets = predict(model, test_loader, device)
    np.save(out_dir / "test_probs.npy", probs)
    np.save(C.RESULTS_DIR / "test_targets.npy", targets)
    m = metrics_from_probs(probs, targets)
    m.update(
        model=key, params_M=models.count_params(model) / 1e6, gflops=models.count_flops(model, device),
        latency_ms_b1=latency(model, device, 1), latency_ms_b32=latency(model, device, 32),
        ckpt_size_mb=(out_dir / "best.pt").stat().st_size / 2**20, best_epoch=ck["epoch"],
    )
    m["throughput_img_s_b32"] = 32 / (m["latency_ms_b32"] / 1000)
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    plot_confusion(m["confusion"], f"{C.MODEL_LABELS[key]} (test)", C.FIGURES_DIR / f"cm_{key}.png")
    print(f"[{key}] acc={m['accuracy']:.4f} bal_acc={m['balanced_accuracy']:.4f} "
          f"macroF1={m['macro_f1']:.4f} AUC={m['macro_auc']:.4f} "
          f"lat_b1={m['latency_ms_b1']:.1f}ms  {m['throughput_img_s_b32']:.0f} img/s")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", choices=list(C.MODELS))
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    keys = list(C.MODELS) if a.all or not a.model else a.model
    C.FIGURES_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda")
    _, _, test_loader, _ = data.make_loaders(num_workers=4)
    for k in keys:
        evaluate_model(k, device, test_loader)
