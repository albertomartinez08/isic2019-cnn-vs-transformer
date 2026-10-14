"""Step 6: aggregate every result into tables (CSV + LaTeX) and figures for the paper/slides.

Outputs:
    results/summary.csv, results/summary_table.tex, results/per_class_f1.tex
    figures/curves.png, figures/acc_vs_cost.png, figures/per_class_f1.png, figures/class_dist.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C

COLORS = {"resnet50": "#1f77b4", "efficientnet_b0": "#2ca02c", "swin_t": "#d62728", "ensemble": "#7f7f7f"}


def load():
    rows = []
    for k in list(C.MODELS) + ["ensemble"]:
        p = C.RESULTS_DIR / k / "test_metrics.json"
        if not p.exists():
            continue
        m = json.load(open(p))
        tr = json.load(open(C.RESULTS_DIR / k / "train_summary.json")) if k != "ensemble" else {}
        rows.append(dict(
            key=k, model=C.MODEL_LABELS[k], family=C.MODEL_FAMILY[k],
            accuracy=m["accuracy"], balanced_accuracy=m["balanced_accuracy"], macro_f1=m["macro_f1"],
            weighted_f1=m["weighted_f1"], macro_auc=m["macro_auc"],
            params_M=m["params_M"], gflops=m["gflops"], latency_ms_b1=m["latency_ms_b1"],
            throughput_img_s=m["throughput_img_s_b32"],
            train_time_min=tr.get("total_train_time_min", np.nan), epoch_time_s=tr.get("mean_epoch_time_s", np.nan),
            peak_vram_mb=tr.get("peak_vram_mb", np.nan), best_epoch=tr.get("best_epoch", np.nan),
            epochs_to_95=tr.get("epochs_to_95pct_best", np.nan), epochs_run=tr.get("epochs_run", np.nan),
            per_class={c: m["per_class"][c]["f1-score"] for c in C.CLASSES},
        ))
    return pd.DataFrame(rows)


def latex_tables(df):
    cols = ["model", "family", "accuracy", "balanced_accuracy", "macro_f1", "macro_auc",
            "params_M", "gflops", "latency_ms_b1", "throughput_img_s", "train_time_min", "peak_vram_mb", "best_epoch", "epochs_to_95"]
    t = df[cols].copy()
    fmt = {"accuracy": "{:.3f}", "balanced_accuracy": "{:.3f}", "macro_f1": "{:.3f}", "macro_auc": "{:.3f}",
           "params_M": "{:.1f}", "gflops": "{:.2f}", "latency_ms_b1": "{:.1f}", "throughput_img_s": "{:.0f}",
           "train_time_min": "{:.1f}", "peak_vram_mb": "{:.0f}", "best_epoch": "{:.0f}", "epochs_to_95": "{:.0f}"}
    lines = [r"\begin{tabular}{llcccccccccccc}", r"\toprule",
             r"Model & Type & Acc. & Bal. Acc. & Macro-F1 & AUC & Params (M) & GFLOPs & Lat. (ms) & img/s & Train (min) & VRAM (MB) & Best ep. & Ep. 95\% \\",
             r"\midrule"]
    for _, r in t.iterrows():
        vals = [r["model"], r["family"]] + [fmt[c].format(r[c]) if pd.notna(r[c]) else "--" for c in cols[2:]]
        lines.append(" & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (C.RESULTS_DIR / "summary_table.tex").write_text("\n".join(lines), encoding="utf-8")

    # per-class F1
    lines = [r"\begin{tabular}{l" + "c" * len(C.CLASSES) + "}", r"\toprule",
             "Model & " + " & ".join(C.CLASSES) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        lines.append(r["model"] + " & " + " & ".join(f"{r['per_class'][c]:.2f}" for c in C.CLASSES) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (C.RESULTS_DIR / "per_class_f1.tex").write_text("\n".join(lines), encoding="utf-8")


def fig_curves():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for k in C.MODELS:
        p = C.RESULTS_DIR / k / "history.json"
        if not p.exists():
            continue
        h = pd.DataFrame(json.load(open(p)))
        axes[0].plot(h.epoch, h.train_loss, "--", color=COLORS[k], alpha=0.7)
        axes[0].plot(h.epoch, h.val_loss, "-", color=COLORS[k], label=C.MODEL_LABELS[k])
        axes[1].plot(h.epoch, h.val_acc, "-o", ms=3, color=COLORS[k], label=C.MODEL_LABELS[k])
        axes[2].plot(h.epoch, h.val_macro_f1, "-o", ms=3, color=COLORS[k], label=C.MODEL_LABELS[k])
    axes[0].set_title("Loss (dashed: train, solid: val)"); axes[1].set_title("Validation accuracy"); axes[2].set_title("Validation macro-F1")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
    axes[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "curves.png", dpi=200); plt.close(fig)


def fig_cost(df):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for _, r in df.iterrows():
        axes[0].scatter(r.params_M, r.macro_f1, s=90, color=COLORS[r.key]); axes[0].annotate(r.model, (r.params_M, r.macro_f1), (5, 5), textcoords="offset points", fontsize=8)
        axes[1].scatter(r.latency_ms_b1, r.macro_f1, s=90, color=COLORS[r.key]); axes[1].annotate(r.model, (r.latency_ms_b1, r.macro_f1), (5, 5), textcoords="offset points", fontsize=8)
    axes[0].set_xlabel("Parameters (M)"); axes[1].set_xlabel("Latency, batch 1 (ms, fp16)")
    for ax in axes:
        ax.set_ylabel("Test macro-F1"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "acc_vs_cost.png", dpi=200); plt.close(fig)


def fig_per_class(df):
    x = np.arange(len(C.CLASSES)); w = 0.8 / len(df)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for i, (_, r) in enumerate(df.iterrows()):
        ax.bar(x + i * w - 0.4 + w / 2, [r.per_class[c] for c in C.CLASSES], w, label=r.model, color=COLORS[r.key])
    ax.set_xticks(x, C.CLASSES); ax.set_ylabel("Test F1"); ax.set_ylim(0, 1); ax.legend(fontsize=8, ncol=4); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "per_class_f1.png", dpi=200); plt.close(fig)


def fig_class_dist():
    s = json.load(open(C.RESULTS_DIR / "dataset_stats.json"))
    fig, ax = plt.subplots(figsize=(7, 3))
    x = np.arange(len(C.CLASSES))
    bottom = np.zeros(len(C.CLASSES))
    for split, col in [("train", "#4c72b0"), ("val", "#dd8452"), ("test", "#55a868")]:
        v = np.array([s[split].get(c, 0) for c in C.CLASSES]); ax.bar(x, v, bottom=bottom, label=split, color=col); bottom += v
    for i, t in enumerate(bottom):
        ax.text(i, t + 150, f"{int(t)}", ha="center", fontsize=8)
    ax.set_xticks(x, C.CLASSES); ax.set_ylabel("images"); ax.legend(); ax.set_title("ISIC 2019 class distribution and split")
    fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "class_dist.png", dpi=200); plt.close(fig)


def latex_macros(df):
    """Numbers used inline in the paper, e.g. \\ResnetAcc, \\SwinParams, \\BestModel."""
    short = {"resnet50": "Resnet", "efficientnet_b0": "Effnet", "swin_t": "Swin", "ensemble": "Ens"}
    L = []
    for _, r in df.iterrows():
        s = short[r.key]
        L += [f"\\newcommand{{\\{s}Acc}}{{{100*r.accuracy:.1f}}}", f"\\newcommand{{\\{s}BalAcc}}{{{100*r.balanced_accuracy:.1f}}}",
              f"\\newcommand{{\\{s}Fone}}{{{100*r.macro_f1:.1f}}}", f"\\newcommand{{\\{s}Auc}}{{{r.macro_auc:.3f}}}",
              f"\\newcommand{{\\{s}Params}}{{{r.params_M:.1f}}}", f"\\newcommand{{\\{s}Gflops}}{{{r.gflops:.2f}}}",
              f"\\newcommand{{\\{s}Lat}}{{{r.latency_ms_b1:.1f}}}", f"\\newcommand{{\\{s}Thr}}{{{r.throughput_img_s:.0f}}}"]
        if pd.notna(r.train_time_min):
            L += [f"\\newcommand{{\\{s}Train}}{{{r.train_time_min:.0f}}}", f"\\newcommand{{\\{s}Epoch}}{{{r.epoch_time_s:.0f}}}",
                  f"\\newcommand{{\\{s}Vram}}{{{r.peak_vram_mb/1024:.1f}}}", f"\\newcommand{{\\{s}BestEp}}{{{int(r.best_epoch)}}}",
                  f"\\newcommand{{\\{s}EpNinetyfive}}{{{int(r.epochs_to_95)}}}", f"\\newcommand{{\\{s}EpRun}}{{{int(r.epochs_run)}}}"]
        for c in C.CLASSES:
            L.append(f"\\newcommand{{\\{s}F{c}}}{{{100*r.per_class[c]:.1f}}}")
    single = df[df.key != "ensemble"]
    best = single.loc[single.macro_f1.idxmax()]
    L.append(f"\\newcommand{{\\BestModel}}{{{best.model}}}")
    L.append(f"\\newcommand{{\\BestFone}}{{{100*best.macro_f1:.1f}}}")
    s = json.load(open(C.RESULTS_DIR / "dataset_stats.json"))
    for k, v in s["n"].items():
        L.append(f"\\newcommand{{\\N{k}}}{{{v:,}}}")
    L.append(f"\\newcommand{{\\Ntotal}}{{{sum(s['n'].values()):,}}}")
    (C.RESULTS_DIR / "macros.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    C.FIGURES_DIR.mkdir(exist_ok=True)
    df = load()
    df.drop(columns=["per_class"]).to_csv(C.RESULTS_DIR / "summary.csv", index=False)
    latex_tables(df)
    latex_macros(df)
    fig_curves(); fig_cost(df); fig_per_class(df); fig_class_dist()
    print(df.drop(columns=["per_class"]).to_string())
