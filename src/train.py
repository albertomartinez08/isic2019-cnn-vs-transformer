"""Step 2: fine-tune one backbone with the shared protocol.

Usage:
    python src/train.py --model resnet50
    python src/train.py --model efficientnet_b0
    python src/train.py --model swin_t

Writes results/<model>/{best.pt, history.json, train_summary.json}
"""
import argparse
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

import config as C
import data
import models


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(step, total, warmup, base):
    if step < warmup:
        return base * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp):
    model.eval()
    losses, preds, targets = [], [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = model(x)
            loss = criterion(out, y)
        losses.append(loss.item() * len(y))
        preds.append(out.argmax(1).cpu())
        targets.append(y.cpu())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return {
        "loss": float(sum(losses) / len(targets)),
        "acc": float((preds == targets).mean()),
        "macro_f1": float(f1_score(targets, preds, average="macro")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(C.MODELS))
    ap.add_argument("--epochs", type=int, default=C.TRAIN["epochs"])
    ap.add_argument("--batch-size", type=int, default=C.TRAIN["batch_size"])
    ap.add_argument("--lr", type=float, default=C.TRAIN["lr"])
    ap.add_argument("--workers", type=int, default=C.TRAIN["num_workers"])
    ap.add_argument("--subset", type=float, default=1.0,
                    help="fraction of each split to use (e.g. 0.1 for a quick smoke test); results go to results/demo_<model>")
    a = ap.parse_args()

    seed_everything(C.SEED)
    device = torch.device("cuda")
    out_dir = C.RESULTS_DIR / (a.model if a.subset >= 1.0 else f"demo_{a.model}")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _, splits = data.make_loaders(a.batch_size, a.workers, a.subset)
    model = models.create(a.model).to(device)
    n_params = models.count_params(model)
    gflops = models.count_flops(model, device)
    print(f"[{a.model}] params={n_params/1e6:.2f}M  GFLOPs={gflops:.2f}")

    weights = data.class_weights(splits["train"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=C.TRAIN["label_smoothing"])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=C.TRAIN["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=C.TRAIN["amp"])
    steps_per_epoch = len(train_loader)
    total_steps = a.epochs * steps_per_epoch
    warmup_steps = C.TRAIN["warmup_epochs"] * steps_per_epoch

    history, best_f1, best_epoch, bad_epochs = [], -1.0, -1, 0
    torch.cuda.reset_peak_memory_stats(device)
    t_start = time.time()
    step = 0
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss, n_seen = 0.0, 0
        pbar = tqdm(train_loader, desc=f"{a.model} epoch {epoch}/{a.epochs}", leave=False)
        for x, y in pbar:
            lr = cosine_lr(step, total_steps, warmup_steps, a.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=C.TRAIN["amp"]):
                out = model(x)
                loss = criterion(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * len(y)
            n_seen += len(y)
            step += 1
            pbar.set_postfix(loss=f"{run_loss/n_seen:.3f}", lr=f"{lr:.1e}")
        train_time = time.time() - t0
        val = evaluate(model, val_loader, criterion, device, C.TRAIN["amp"])
        rec = dict(epoch=epoch, train_loss=run_loss / n_seen, epoch_time_s=train_time,
                   val_loss=val["loss"], val_acc=val["acc"], val_macro_f1=val["macro_f1"],
                   lr_end=lr, peak_vram_mb=torch.cuda.max_memory_allocated(device) / 2**20)
        history.append(rec)
        print(f"[{a.model}] epoch {epoch}: train_loss={rec['train_loss']:.4f} val_loss={val['loss']:.4f} "
              f"val_acc={val['acc']:.4f} val_f1={val['macro_f1']:.4f} ({train_time:.0f}s)")
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if val["macro_f1"] > best_f1:
            best_f1, best_epoch, bad_epochs = val["macro_f1"], epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val}, out_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= C.TRAIN["patience"]:
                print(f"[{a.model}] early stopping at epoch {epoch}")
                break

    total_time = time.time() - t_start
    # epochs needed to reach 95 % of the best validation macro-F1 (convergence speed)
    e95 = next(h["epoch"] for h in history if h["val_macro_f1"] >= 0.95 * best_f1)
    summary = dict(
        model=a.model, timm_name=C.MODELS[a.model], params_M=n_params / 1e6, gflops=gflops,
        epochs_run=len(history), best_epoch=best_epoch, epochs_to_95pct_best=e95,
        best_val_macro_f1=best_f1, best_val_acc=history[best_epoch - 1]["val_acc"],
        total_train_time_min=total_time / 60, mean_epoch_time_s=float(np.mean([h["epoch_time_s"] for h in history])),
        peak_vram_mb=torch.cuda.max_memory_allocated(device) / 2**20,
        batch_size=a.batch_size, lr=a.lr, protocol=C.TRAIN,
    )
    with open(out_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
