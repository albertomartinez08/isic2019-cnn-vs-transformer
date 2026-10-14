"""Step 5: interpretability. Grad-CAM for the CNNs and for Swin-T (with token reshaping).

Produces figures/gradcam_grid.png: rows = sample test images (one per class, correctly or
incorrectly classified), columns = original + one heat-map per model.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import config as C
import data
import models


def pick_samples(n_per_class=1):
    df = data.load_splits()["test"].reset_index(drop=True)
    targets = np.load(C.RESULTS_DIR / "test_targets.npy")
    preds = {k: np.load(C.RESULTS_DIR / k / "test_probs.npy").argmax(1) for k in C.MODELS}
    rng = np.random.RandomState(C.SEED)
    rows = []
    for ci, c in enumerate(C.CLASSES):
        idx = np.where(targets == ci)[0]
        rng.shuffle(idx)
        for i in idx[:n_per_class]:
            rows.append(dict(i=int(i), image=df.loc[i, "image"], true=c,
                             **{f"pred_{k}": C.CLASSES[preds[k][i]] for k in C.MODELS}))
    return rows


def main():
    device = torch.device("cuda")
    tf = data.build_transforms(train=False)
    samples = pick_samples()
    cams = {}
    for k in C.MODELS:
        model = models.create(k, pretrained=False)
        model.load_state_dict(torch.load(C.RESULTS_DIR / k / "best.pt", map_location="cpu")["model"])
        model.to(device).eval()
        layers, reshape = models.gradcam_target_layer(k, model)
        cam = GradCAM(model=model, target_layers=layers, reshape_transform=reshape)
        cams[k] = []
        for s in samples:
            img = Image.open(C.RESIZED_DIR / f"{s['image']}.jpg").convert("RGB")
            x = tf(img).unsqueeze(0).to(device)
            ci = C.CLASSES.index(s["true"])
            g = cam(input_tensor=x, targets=[ClassifierOutputTarget(ci)])[0]
            rgb = np.asarray(img.resize((C.IMG_SIZE + 32, C.IMG_SIZE + 32)).crop((16, 16, 16 + C.IMG_SIZE, 16 + C.IMG_SIZE))) / 255.0
            cams[k].append((rgb, show_cam_on_image(rgb.astype(np.float32), g, use_rgb=True)))
        del model, cam
        torch.cuda.empty_cache()

    n = len(samples)
    cols = 1 + len(C.MODELS)
    fig, axes = plt.subplots(n, cols, figsize=(2.3 * cols, 2.3 * n))
    for r, s in enumerate(samples):
        axes[r, 0].imshow(cams["resnet50"][r][0]); axes[r, 0].set_ylabel(s["true"], fontsize=10, fontweight="bold")
        if r == 0:
            axes[r, 0].set_title("Input", fontsize=9)
        for c, k in enumerate(C.MODELS, start=1):
            axes[r, c].imshow(cams[k][r][1])
            ok = s[f"pred_{k}"] == s["true"]
            axes[r, c].set_title(f"{C.MODEL_LABELS[k]}\npred: {s[f'pred_{k}']} {'✓' if ok else '✗'}" if r == 0
                                 else f"pred: {s[f'pred_{k}']} {'✓' if ok else '✗'}",
                                 fontsize=8, color="green" if ok else "red")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    C.FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(C.FIGURES_DIR / "gradcam_grid.png", dpi=180)
    with open(C.RESULTS_DIR / "gradcam_samples.json", "w") as f:
        json.dump(samples, f, indent=2)
    print("saved figures/gradcam_grid.png")


if __name__ == "__main__":
    main()
