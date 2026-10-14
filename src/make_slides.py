"""Step 7: build the 10-minute presentation (slides/presentation.pptx) from results and figures.

Design: navy/coral/teal palette, title band on every slide, stat cards, model cards, real lesion
thumbnails, highlighted results table, call-out boxes, numbered conclusions and a QR code.
"""
import json

import pandas as pd
import qrcode
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

import config as C

AUTHOR = "José Alberto Martínez Navarro"
COURSE = "Aprendizaje Profundo"
REPO_URL = "https://github.com/albertomartinez08/isic2019-cnn-vs-transformer"

NAVY = RGBColor(0x14, 0x24, 0x40)
NAVY2 = RGBColor(0x1F, 0x3A, 0x5F)
CORAL = RGBColor(0xE4, 0x57, 0x2E)
TEAL = RGBColor(0x17, 0xA3, 0x98)
GOLD = RGBColor(0xF2, 0xB1, 0x34)
INK = RGBColor(0x22, 0x28, 0x33)
GREY = RGBColor(0x6B, 0x74, 0x80)
LIGHT = RGBColor(0xF3, 0xF5, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
MODEL_COLOR = {"resnet50": RGBColor(0x1F, 0x77, 0xB4), "efficientnet_b0": RGBColor(0x2C, 0xA0, 0x2C),
               "swin_t": RGBColor(0xD6, 0x27, 0x28), "ensemble": RGBColor(0x6B, 0x74, 0x80)}

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
W, H = 13.333, 7.5
BLANK = prs.slide_layouts[6]
slide_no = [0]


# ----------------------------------------------------------------------------- primitives
def rect(slide, l, t, w, h, fill, shape=MSO_SHAPE.RECTANGLE, line=None):
    s = slide.shapes.add_shape(shape, Inches(l), Inches(t), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line; s.line.width = Pt(1)
    s.shadow.inherit = False
    return s


def text(slide, l, t, w, h, lines, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, font="Calibri", spacing=4):
    """lines: str or list of str | (str, dict) with per-paragraph overrides."""
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.08); tf.margin_top = tf.margin_bottom = Inches(0.04)
    if isinstance(lines, str):
        lines = [lines]
    for i, item in enumerate(lines):
        s, opt = (item, {}) if isinstance(item, str) else item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = opt.get("align", align); p.space_after = Pt(opt.get("spacing", spacing))
        r = p.add_run(); r.text = s
        r.font.size = Pt(opt.get("size", size)); r.font.bold = opt.get("bold", bold)
        r.font.color.rgb = opt.get("color", color); r.font.name = font
    return tb


def bullets(slide, l, t, w, h, items, size=16, color=INK, gap=6):
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.08)
    for i, it in enumerate(items):
        s, lvl = (it, 0) if isinstance(it, str) else it
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        r1 = p.add_run(); r1.text = ("▪ " if lvl == 0 else "      – "); r1.font.color.rgb = CORAL if lvl == 0 else GREY
        r1.font.size = Pt(size - 2 * lvl); r1.font.bold = True
        r2 = p.add_run(); r2.text = s; r2.font.size = Pt(size - 2 * lvl); r2.font.color.rgb = color; r2.font.name = "Calibri"
    return tb


def picture(slide, path, l, t, w=None, h=None):
    kw = {}
    if w: kw["width"] = Inches(w)
    if h: kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(path), Inches(l), Inches(t), **kw)


def card(slide, l, t, w, h, title, body, accent=TEAL, title_size=13, body_size=20, body_color=NAVY):
    rect(slide, l, t, w, h, LIGHT, MSO_SHAPE.ROUNDED_RECTANGLE)
    rect(slide, l, t, 0.09, h, accent)
    text(slide, l + 0.15, t + 0.08, w - 0.25, 0.4, title, size=title_size, color=GREY, bold=True)
    text(slide, l + 0.15, t + 0.42, w - 0.25, h - 0.5, body, size=body_size, color=body_color, bold=True,
         anchor=MSO_ANCHOR.TOP)


def callout(slide, l, t, w, h, title, body, accent=CORAL, size=13):
    rect(slide, l, t, w, h, WHITE, MSO_SHAPE.ROUNDED_RECTANGLE, line=accent)
    text(slide, l + 0.12, t + 0.06, w - 0.24, 0.35, title, size=size, color=accent, bold=True)
    text(slide, l + 0.12, t + 0.38, w - 0.24, h - 0.45, body, size=size - 1, color=INK)


def content_slide(title, kicker=None):
    slide_no[0] += 1
    s = prs.slides.add_slide(BLANK)
    rect(s, 0, 0, W, 1.05, NAVY)
    rect(s, 0, 1.05, W, 0.06, CORAL)
    text(s, 0.5, 0.18, 11.5, 0.7, title, size=26, color=WHITE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    if kicker:
        text(s, 0.5, 0.66, 11.5, 0.35, kicker, size=13, color=RGBColor(0xBF, 0xD7, 0xEA))
    # footer
    rect(s, 0, 7.12, W, 0.38, LIGHT)
    text(s, 0.5, 7.14, 8, 0.32, f"{AUTHOR}  ·  {COURSE}  ·  ISIC 2019: CNNs vs Transformers",
         size=10, color=GREY, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 12.0, 7.14, 0.9, 0.32, f"{slide_no[0]}", size=11, color=GREY, bold=True, align=PP_ALIGN.RIGHT,
         anchor=MSO_ANCHOR.MIDDLE)
    return s


def notes(slide, t):
    slide.notes_slide.notes_text_frame.text = t


# ----------------------------------------------------------------------------- data
df = pd.read_csv(C.RESULTS_DIR / "summary.csv")
stats = json.load(open(C.RESULTS_DIR / "dataset_stats.json"))
single = df[df.key != "ensemble"].reset_index(drop=True)
best = single.loc[single.macro_f1.idxmax()]
ens = df[df.key == "ensemble"].iloc[0]
row = {r.key: r for _, r in df.iterrows()}
samples = json.load(open(C.RESULTS_DIR / "gradcam_samples.json"))
F = C.FIGURES_DIR
N = sum(stats["n"].values())
test_m = {k: json.load(open(C.RESULTS_DIR / k / "test_metrics.json")) for k in list(C.MODELS) + ["ensemble"]}
combos = json.load(open(C.RESULTS_DIR / "ensemble" / "all_combinations.json"))

# ============================================================================= 1. Title
slide_no[0] += 1
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, W, H, NAVY)
rect(s, 0, 0, 0.35, H, CORAL)
rect(s, 8.9, 0, W - 8.9, H, NAVY2)
text(s, 0.9, 0.9, 8.0, 0.5, "FINAL PROJECT  ·  " + COURSE.upper(), size=13, color=GOLD, bold=True)
text(s, 0.9, 1.5, 8.0, 2.6, ["CNNs vs Vision Transformers", "for Skin-Lesion Classification"],
     size=38, color=WHITE, bold=True, spacing=0)
text(s, 0.9, 3.9, 7.8, 1.1, "A controlled comparison of ResNet-50, EfficientNet-B0 and Swin-T on ISIC 2019: "
     "accuracy, computational cost and interpretability under one identical protocol",
     size=16, color=RGBColor(0xBF, 0xD7, 0xEA))
text(s, 0.9, 5.6, 7.8, 0.5, AUTHOR, size=20, color=WHITE, bold=True)
text(s, 0.9, 6.05, 7.8, 0.4, f"{COURSE}  ·  October 2026", size=14, color=RGBColor(0xBF, 0xD7, 0xEA))
# right panel: lesion thumbnails 2x4 + chips
x0, y0, sz, g = 9.5, 1.3, 1.38, 0.16
for i, smp in enumerate(samples[:8]):
    r_, c_ = divmod(i, 2)
    l_, t_ = x0 + c_ * (sz + g), y0 + r_ * (sz + g)
    picture(s, C.RESIZED_DIR / f"{smp['image']}.jpg", l_, t_, w=sz, h=sz)
    rect(s, l_, t_ + sz - 0.32, sz, 0.32, RGBColor(0x14, 0x24, 0x40))
    text(s, l_, t_ + sz - 0.33, sz, 0.32, smp["true"], size=10, color=WHITE, bold=True, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
for i, (lab, val) in enumerate([("images", f"{N:,}"), ("classes", "8"), ("models", "3 + ensemble")]):
    rect(s, 9.35 + i * 1.3, 0.55, 1.2, 0.5, NAVY, MSO_SHAPE.ROUNDED_RECTANGLE, line=TEAL)
    text(s, 9.35 + i * 1.3, 0.55, 1.2, 0.5, [(val, {"size": 12, "bold": True, "color": WHITE, "align": PP_ALIGN.CENTER, "spacing": 0}),
                                              (lab, {"size": 8, "color": TEAL, "align": PP_ALIGN.CENTER})], anchor=MSO_ANCHOR.MIDDLE)
notes(s, "Hello. My project compares three ImageNet-pretrained backbones on the ISIC 2019 dermoscopy dataset under one identical protocol, and looks at cost and interpretability, not only accuracy.")

# ============================================================================= 2. Problem
s = content_slide("Problem and objective", "Why another comparison, and what is different about this one")
callout(s, 0.5, 1.4, 6.0, 1.55, "Clinical task", "Classify a dermoscopic image into 8 diagnoses: melanoma, nevus, basal cell carcinoma, "
        "actinic keratosis, benign keratosis, dermatofibroma, vascular lesion, squamous cell carcinoma.", TEAL, 14)
callout(s, 0.5, 3.1, 6.0, 1.55, "Practical question", "CNN or Transformer for a clinical prototype, and at what price in latency, "
        "memory and training time?", CORAL, 14)
callout(s, 0.5, 4.8, 6.0, 1.55, "Gap in the literature", "Published ISIC results use different splits, resolutions, ensembles and metadata, "
        "so they cannot be compared; cost is rarely reported.", GOLD, 14)
rect(s, 6.9, 1.4, 5.95, 4.95, NAVY, MSO_SHAPE.ROUNDED_RECTANGLE)
text(s, 7.15, 1.55, 5.5, 0.5, "OBJECTIVE", size=13, color=GOLD, bold=True)
text(s, 7.15, 1.95, 5.5, 1.0, "Hold everything fixed and compare ResNet-50, EfficientNet-B0 and Swin-T on:", size=16, color=WHITE, bold=True)
bullets(s, 7.15, 3.0, 5.5, 3.3, [
    "Accuracy, balanced accuracy, macro-F1, AUC, per-class F1",
    "Parameters, FLOPs, VRAM, training time, epochs to converge, latency",
    "Grad-CAM evidence maps",
    "A soft-voting ensemble: are the families complementary?",
], size=15, color=WHITE)
notes(s, "The motivation is practical: a deployment decision depends on cost as much as on accuracy, and the literature rarely reports both under the same protocol.")

# ============================================================================= 3. Dataset
s = content_slide("Dataset: ISIC 2019", f"{N:,} dermoscopic images · 8 classes · strong imbalance (NV ≈ 50 %)")
picture(s, F / "class_dist.png", 0.5, 1.35, w=6.6)
for i, k in enumerate(["train", "val", "test"]):
    card(s, 0.5 + i * 2.25, 4.45, 2.1, 1.15, k.upper(), f"{stats['n'][k]:,}", accent=[TEAL, GOLD, CORAL][i], body_size=22)
callout(s, 0.5, 5.75, 6.6, 1.25, "No leakage", "Split stratified by class and grouped by lesion_id: the same lesion never appears "
        "in two splits (checked by an assertion in prepare_data.py).", TEAL, 13)
# lesion strip
text(s, 7.5, 1.35, 5.4, 0.35, "One test image per class", size=13, color=GREY, bold=True)
x0, y0, sz, g = 7.5, 1.75, 1.22, 0.12
for i, smp in enumerate(samples[:8]):
    r_, c_ = divmod(i, 4)
    picture(s, C.RESIZED_DIR / f"{smp['image']}.jpg", x0 + c_ * (sz + g), y0 + r_ * (sz + g + 0.3), w=sz, h=sz)
    text(s, x0 + c_ * (sz + g), y0 + r_ * (sz + g + 0.3) + sz, sz, 0.3, smp["true"], size=11, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
bullets(s, 7.5, 4.95, 5.4, 2.1, [
    "Pre-resize to 256 px once (442 MB cache instead of 9 GB)",
    "Train: random 224 crops, flips, rotation ±20°, colour jitter",
    "Eval: centre 224 crop · ImageNet normalisation",
], size=13)
notes(s, "ISIC 2019 merges HAM10000, BCN20000 and MSK. I split by lesion id so that repeated images of the same lesion cannot leak from train to test, which inflates scores in many papers.")

# ============================================================================= 4. Models + protocol
s = content_slide("Three backbones, one protocol", "Everything that is not the architecture is identical")
specs = [("resnet50", "Residual CNN", "Reference backbone: skip connections, 4 stages, global pooling"),
         ("efficientnet_b0", "Efficient CNN", "Compound-scaled, depthwise-separable convolutions, mobile-oriented"),
         ("swin_t", "Hierarchical Transformer", "Shifted-window self-attention with a feature pyramid; cost ≈ ResNet-50")]
for i, (k, fam, desc) in enumerate(specs):
    l = 0.5 + i * 4.15
    rect(s, l, 1.4, 3.95, 2.75, LIGHT, MSO_SHAPE.ROUNDED_RECTANGLE)
    rect(s, l, 1.4, 3.95, 0.55, MODEL_COLOR[k], MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, l + 0.1, 1.42, 3.75, 0.5, C.MODEL_LABELS[k], size=18, color=WHITE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    text(s, l + 0.15, 2.05, 3.7, 0.35, fam, size=12, color=MODEL_COLOR[k], bold=True)
    text(s, l + 0.15, 2.4, 3.7, 0.8, desc, size=12, color=INK)
    r = row[k]
    text(s, l + 0.15, 3.25, 1.8, 0.8, [(f"{r.params_M:.1f} M", {"size": 20, "bold": True, "color": NAVY, "spacing": 0}),
                                       ("parameters", {"size": 10, "color": GREY})])
    text(s, l + 2.0, 3.25, 1.8, 0.8, [(f"{r.gflops:.2f}", {"size": 20, "bold": True, "color": NAVY, "spacing": 0}),
                                      ("GFLOPs / image", {"size": 10, "color": GREY})])
# pipeline
text(s, 0.5, 4.4, 12, 0.35, "SHARED FINE-TUNING PROTOCOL", size=12, color=GREY, bold=True)
steps = [("ImageNet-1k weights", "timm · new 8-way head · full fine-tuning"),
         ("AdamW", "lr 1e-4 · wd 0.05 · batch 32"),
         ("Schedule", "1 warm-up epoch + cosine · ≤ 12 epochs"),
         ("Loss", "CE weighted 1/√freq · label smoothing 0.1"),
         ("Early stopping", "patience 4 on validation macro-F1"),
         ("Hardware", "RTX 4070 Laptop 8 GB · AMP fp16 · seed 42")]
sw, gap = 1.95, 0.12
for i, (a, b) in enumerate(steps):
    l = 0.5 + i * (sw + gap)
    rect(s, l, 4.8, sw, 1.55, NAVY if i % 2 == 0 else NAVY2, MSO_SHAPE.CHEVRON if i else MSO_SHAPE.PENTAGON)
    text(s, l + 0.22, 4.9, sw - 0.5, 0.45, a, size=12, color=GOLD, bold=True)
    text(s, l + 0.22, 5.3, sw - 0.5, 1.0, b, size=10, color=WHITE)
text(s, 0.5, 6.5, 12, 0.5, "Best-validation checkpoint evaluated once on the test split. Swin-T was chosen over ViT-B so that its cost matches ResNet-50.",
     size=12, color=GREY)
notes(s, "Everything that is not the architecture is identical, so differences can be attributed to the backbone.")

# ============================================================================= 5. Results table
s = content_slide("Results on the test split", "Accuracy and cost side by side · best value per column highlighted")
cols = ["model", "accuracy", "balanced_accuracy", "macro_f1", "macro_auc", "params_M", "gflops", "latency_ms_b1", "train_time_min", "peak_vram_mb", "epochs_to_95"]
hdr = ["Model", "Accuracy", "Bal. acc.", "Macro-F1", "AUC", "Params (M)", "GFLOPs", "Latency b1 (ms)", "Train (min)", "VRAM (MB)", "Ep. to 95 %"]
fmt = {"accuracy": "{:.3f}", "balanced_accuracy": "{:.3f}", "macro_f1": "{:.3f}", "macro_auc": "{:.3f}", "params_M": "{:.1f}",
       "gflops": "{:.2f}", "latency_ms_b1": "{:.1f}", "train_time_min": "{:.0f}", "peak_vram_mb": "{:.0f}", "epochs_to_95": "{:.0f}"}
better_high = {"accuracy", "balanced_accuracy", "macro_f1", "macro_auc"}
rows_n, cols_n = len(df) + 1, len(cols)
tbl = s.shapes.add_table(rows_n, cols_n, Inches(0.5), Inches(1.4), Inches(12.33), Inches(0.45 * rows_n)).table
tbl.columns[0].width = Inches(2.3)
for j in range(1, cols_n):
    tbl.columns[j].width = Inches((12.33 - 2.3) / (cols_n - 1))
for j, h in enumerate(hdr):
    c = tbl.cell(0, j); c.text = h; c.fill.solid(); c.fill.fore_color.rgb = NAVY
    p = c.text_frame.paragraphs[0]; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER if j else PP_ALIGN.LEFT
for i, (_, r) in enumerate(df.iterrows(), start=1):
    for j, col in enumerate(cols):
        c = tbl.cell(i, j); v = r[col]
        c.fill.solid(); c.fill.fore_color.rgb = LIGHT if i % 2 else WHITE
        c.text = fmt.get(col, "{}").format(v) if pd.notna(v) else "–"
        p = c.text_frame.paragraphs[0]; p.font.size = Pt(12); p.alignment = PP_ALIGN.CENTER if j else PP_ALIGN.LEFT
        p.font.color.rgb = INK
        if j == 0:
            p.font.bold = True; p.font.color.rgb = MODEL_COLOR[r.key]
        elif col in fmt and pd.notna(v):
            series = single[col] if r.key != "ensemble" else None
            is_best = (col in better_high and v == df[col].max()) or (col not in better_high and v == single[col].min())
            if is_best:
                p.font.bold = True; p.font.color.rgb = CORAL
card(s, 0.5, 4.1, 2.95, 1.35, "BEST SINGLE MODEL", f"{best.model}\n{100*best.macro_f1:.1f} % macro-F1", accent=MODEL_COLOR[best.key], body_size=18)
card(s, 3.6, 4.1, 2.95, 1.35, "ENSEMBLE OF 3", f"{100*ens.macro_f1:.1f} % macro-F1\nAUC {ens.macro_auc:.3f}", accent=GREY, body_size=18)
card(s, 6.7, 4.1, 2.95, 1.35, "COST OF THE TRANSFORMER", f"{row['swin_t'].latency_ms_b1/row['resnet50'].latency_ms_b1:.1f}× latency · {row['swin_t'].peak_vram_mb/row['resnet50'].peak_vram_mb:.1f}× VRAM\nvs ResNet-50", accent=CORAL, body_size=16)
card(s, 9.8, 4.1, 3.03, 1.35, "CHEAPEST TO TRAIN", f"EfficientNet-B0\n{row['efficientnet_b0'].train_time_min:.0f} min · {row['efficientnet_b0'].peak_vram_mb/1024:.1f} GB", accent=MODEL_COLOR["efficientnet_b0"], body_size=18)
callout(s, 0.5, 5.65, 12.33, 1.3, "Read the table in two halves",
        f"Decision metrics (accuracy, balanced accuracy, macro-F1) rank Swin-T > EfficientNet-B0 > ResNet-50. "
        f"AUC ranks them the other way round ({row['resnet50'].macro_auc:.3f} > {row['efficientnet_b0'].macro_auc:.3f} > {row['swin_t'].macro_auc:.3f}): "
        "Swin-T fits the training set harder and gives sharper posteriors — better hard decisions, slightly worse probability ranking. A triage tool should be calibrated.", NAVY, 13)
notes(s, "Read the table from left to right: the accuracy columns are close, the cost columns are not. Point out the AUC inversion.")

# ============================================================================= 6. Curves
s = content_slide("Convergence under the same schedule", "Validation curves per epoch · dashed = training loss")
picture(s, F / "curves.png", 0.5, 1.35, w=12.33)
for i, (_, r) in enumerate(single.iterrows()):
    card(s, 0.5 + i * 4.15, 5.05, 3.95, 1.25, C.MODEL_LABELS[r.key].upper(),
         f"best epoch {int(r.best_epoch)} · 95 % of best at epoch {int(r.epochs_to_95)}\n{r.epoch_time_s:.0f} s / epoch · {r.train_time_min:.0f} min total",
         accent=MODEL_COLOR[r.key], body_size=13)
text(s, 0.5, 6.4, 12.3, 0.6, "Swin-T is already at 0.60 macro-F1 after 2 epochs; the CNNs need 5–6. Training loss keeps falling while validation loss "
     "flattens: incipient over-fitting kept in check by early stopping.", size=13, color=INK)
notes(s, "All three converge within the 12-epoch budget thanks to ImageNet pre-training; note which one gets there first and how the validation loss behaves.")

# ============================================================================= 7. Cost
s = content_slide("Accuracy vs computational cost", "FLOPs are not latency")
picture(s, F / "acc_vs_cost.png", 0.5, 1.35, w=8.0)
card(s, 8.8, 1.4, 4.0, 1.3, "EFFICIENTNET-B0 vs RESNET-50", f"{row['resnet50'].gflops/row['efficientnet_b0'].gflops:.0f}× fewer FLOPs, yet {row['efficientnet_b0'].latency_ms_b1:.0f} ms vs {row['resnet50'].latency_ms_b1:.0f} ms", accent=MODEL_COLOR["efficientnet_b0"], body_size=15)
card(s, 8.8, 2.85, 4.0, 1.3, "SWIN-T vs RESNET-50", f"same FLOPs, {row['swin_t'].latency_ms_b1/row['resnet50'].latency_ms_b1:.1f}× latency, {row['swin_t'].peak_vram_mb/row['resnet50'].peak_vram_mb:.1f}× training VRAM", accent=MODEL_COLOR["swin_t"], body_size=15)
card(s, 8.8, 4.3, 4.0, 1.3, "THROUGHPUT (batch 32, fp16)", f"ResNet {row['resnet50'].throughput_img_s:.0f} · Swin {row['swin_t'].throughput_img_s:.0f} · EffNet {row['efficientnet_b0'].throughput_img_s:.0f} img/s", accent=NAVY, body_size=14)
callout(s, 0.5, 5.3, 12.33, 1.65, "Why the cheapest model by FLOPs is not the fastest on a GPU",
        "Depthwise-separable convolutions are memory-bound and launch many small kernels that cannot use tensor cores; EfficientNet's savings "
        "appear on CPUs and mobile NPUs, the hardware it was designed for. Deployment decisions must use latency measured on the target device, not parameter counts.", CORAL, 13)
notes(s, "This is the figure I would show to an engineering manager: the choice is a budget decision.")

# ============================================================================= 8. Per class
s = content_slide("Where the models differ: per-class behaviour", "Macro-F1 is driven by the rare and overlapping classes")
picture(s, F / "per_class_f1.png", 0.5, 1.35, w=7.6)
picture(s, F / f"cm_{best.key}.png", 8.3, 1.3, h=4.1)
callout(s, 0.5, 4.4, 3.65, 2.55, "Strong", "NV (F1 0.85–0.88), BCC and VASC: frequent or visually distinctive.", TEAL, 13)
callout(s, 4.3, 4.4, 3.8, 2.55, "Weak", "AK, SCC, DF: rare and morphologically overlapping. Two persistent confusions, also hard for dermatologists: MEL↔NV and AK↔BKL↔SCC.", CORAL, 13)
callout(s, 8.3, 5.55, 4.5, 1.4, "Swin-T is the most uniform", "Best F1 in 7 of 8 classes; the only backbone with DF above 0.6. Class weighting lifted rare-class recall without hurting NV.", MODEL_COLOR["swin_t"], 12)
notes(s, "Macro-F1 is dominated by the rare classes; class weighting helps but does not remove the MEL/NV and keratosis confusions that are also hard for dermatologists.")

# ============================================================================= 9. Grad-CAM
s = content_slide("Interpretability: Grad-CAM evidence maps", "Target = true class · one test image per class · green/red = correct/incorrect")
picture(s, F / "gradcam_grid.png", 0.5, 1.3, h=5.7)
callout(s, 3.7, 1.4, 4.4, 1.7, "CNNs: compact, centred blobs", "ResNet-50 and EfficientNet-B0 fire on the pigmented core. Adequate for NV, BCC, DF, VASC; misses the periphery that separates MEL from NV.", MODEL_COLOR["resnet50"], 12)
callout(s, 3.7, 3.25, 4.4, 1.7, "Swin-T: multi-focal maps", "Follows the irregular border and the asymmetric pigment distribution — the clinical ABCD cues (two hot spots on the melanoma, a ring on the vascular lesion).", MODEL_COLOR["swin_t"], 12)
callout(s, 3.7, 5.1, 4.4, 1.85, "Informative failures", "BKL: all three fire on the darkest region and predict melanoma. SCC: EfficientNet-B0 spreads evidence over hair and scale artefacts → segment the lesion or remove artefacts before deployment.", CORAL, 12)
rect(s, 8.4, 1.4, 4.45, 5.55, NAVY, MSO_SHAPE.ROUNDED_RECTANGLE)
text(s, 8.6, 1.55, 4.1, 0.4, "WHAT THIS BUYS A CLINICIAN", size=12, color=GOLD, bold=True)
bullets(s, 8.6, 2.0, 4.1, 4.9, [
    "A sanity check that the decision comes from the lesion, not from rulers, hair or vignetting",
    "A way to audit each error class: wrong region vs right region, wrong label",
    "Evidence that the Transformer uses lesion-wide texture, consistent with its advantage on DF and AK",
    "Required by the project brief: interpretability beyond ‘which model is best’",
], size=13, color=WHITE, gap=10)
notes(s, "The CNNs focus on compact lesion regions; Swin spreads attention over wider context. Failures often correspond to evidence on hair, ruler marks or borders.")

# ============================================================================= 10. Ensemble / discussion
s = content_slide("Are the two families complementary?", "Soft-voting ensembles on the test split")
pairs = [("resnet50+efficientnet_b0", "CNN + CNN"), ("resnet50+swin_t", "ResNet-50 + Swin-T"),
         ("efficientnet_b0+swin_t", "EfficientNet-B0 + Swin-T"), ("resnet50+efficientnet_b0+swin_t", "All three")]
for i, (key, lab) in enumerate(pairs):
    m = combos[key]
    acc = CORAL if "swin" in key else GREY
    card(s, 0.5 + i * 3.1, 1.4, 2.95, 1.6, lab.upper(), f"{100*m['macro_f1']:.1f} % macro-F1\nAUC {m['macro_auc']:.3f}", accent=acc, body_size=17)
callout(s, 0.5, 3.2, 6.0, 1.75, "Nearly all the gain comes from CNN + Transformer",
        f"The two CNNs together reach only {100*combos['resnet50+efficientnet_b0']['macro_f1']:.1f} %; adding Swin-T to EfficientNet-B0 gives "
        f"{100*combos['efficientnet_b0+swin_t']['macro_f1']:.1f} %, almost the full ensemble ({100*ens.macro_f1:.1f} %). The families make different mistakes.", CORAL, 13)
callout(s, 6.85, 3.2, 6.0, 1.75, "Price of the ensemble",
        f"{ens.params_M:.0f} M parameters, {ens.latency_ms_b1:.0f} ms per image at batch 1 — fine for offline screening of a clinic's daily images, "
        "too slow for an interactive tool.", NAVY, 13)
callout(s, 0.5, 5.15, 12.35, 1.8, "Context and limitations",
        "Balanced accuracies of 0.60–0.68 are in the range of single ImageNet-pretrained models at 224 px; the challenge winner reached 0.63–0.74 with "
        "multi-resolution ensembles and metadata on a harder, out-of-distribution test set. Single seed per model (differences < 1 point are noise), "
        "fixed 224 px, no metadata, no test-time augmentation, no per-model learning-rate search.", GREY, 12)
notes(s, "Go beyond 'which is best': cost, interpretability and generalisation each tell a different story.")

# ============================================================================= 11. Conclusions
s = content_slide("Conclusions and recommendation", "Three findings, one decision rule")
concl = [("1", "The Transformer wins on decisions", f"With ImageNet pre-training Swin-T is the most accurate and most class-uniform backbone on 25k images ({100*best.macro_f1:.1f} % macro-F1), contradicting the view that transformers need huge medical datasets."),
         ("2", "That accuracy has a measurable price", f"{row['swin_t'].latency_ms_b1/row['resnet50'].latency_ms_b1:.1f}× the latency and {row['swin_t'].peak_vram_mb/row['resnet50'].peak_vram_mb:.1f}× the training memory of ResNet-50; and the cheapest model by FLOPs is not the fastest on a GPU. Decide with measured latency, not parameter counts."),
         ("3", "The families are complementary", f"CNN + Transformer soft voting gives the best macro-F1 ({100*ens.macro_f1:.1f} %) and AUC ({ens.macro_auc:.3f}) of the study because the two families make different errors.")]
for i, (n, t_, b) in enumerate(concl):
    top = 1.35 + i * 1.3
    rect(s, 0.5, top, 0.75, 1.15, [TEAL, CORAL, GOLD][i], MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, 0.5, top, 0.75, 1.15, n, size=30, color=WHITE, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 1.4, top, 6.4, 0.4, t_, size=15, color=NAVY, bold=True)
    text(s, 1.4, top + 0.38, 6.4, 0.8, b, size=11, color=INK)
rect(s, 8.2, 1.35, 4.65, 3.85, NAVY, MSO_SHAPE.ROUNDED_RECTANGLE)
text(s, 8.4, 1.5, 4.3, 0.4, "RECOMMENDATION BY BUDGET", size=12, color=GOLD, bold=True)
for i, (ctx, mdl, why) in enumerate([("CPU / edge device", "EfficientNet-B0", "4 M params, 1.5 GB to train"),
                                     ("GPU available", "Swin-T", "best accuracy per class"),
                                     ("Offline screening", "CNN + Transformer ensemble", "accuracy dominates, latency does not")]):
    top = 1.95 + i * 1.05
    rect(s, 8.4, top, 4.25, 0.9, NAVY2, MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, 8.5, top + 0.05, 1.7, 0.8, ctx, size=11, color=RGBColor(0xBF, 0xD7, 0xEA), anchor=MSO_ANCHOR.MIDDLE)
    text(s, 10.15, top + 0.05, 2.45, 0.8, [(mdl, {"size": 13, "bold": True, "color": WHITE, "spacing": 0}), (why, {"size": 9, "color": TEAL})], anchor=MSO_ANCHOR.MIDDLE)
callout(s, 0.5, 5.35, 7.3, 1.6, "Future work", "Several seeds · 384–512 px · patient metadata · test-time augmentation · calibration for triage · "
        "self-supervised pre-training on dermoscopy (DINO/MAE) · external validation on the official ISIC test set and PAD-UFES-20.", GREY, 12)
callout(s, 8.2, 5.35, 4.65, 1.6, "Reproducible", "run_all.ps1 regenerates every number and figure; Colab notebook with a 442 MB fast path and a 3-minute demo mode.", TEAL, 12)
notes(s, "Close with the recommendation and the reproducibility statement.")

# ============================================================================= 12. Thank you / Q&A
slide_no[0] += 1
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, W, H, NAVY)
rect(s, 0, 0, 0.35, H, CORAL)
text(s, 0.9, 1.3, 7.5, 1.0, "Thank you — questions?", size=36, color=WHITE, bold=True)
text(s, 0.9, 2.4, 7.5, 0.5, AUTHOR + "  ·  " + COURSE, size=16, color=RGBColor(0xBF, 0xD7, 0xEA))
bullets(s, 0.9, 3.3, 7.6, 3.5, [
    "Code, logs, figures, paper and slides: " + REPO_URL.replace("https://", ""),
    "Open in Colab from the README · 442 MB fast path · --subset 0.1 demo in 3 min",
    "Paper: IEEE conference format, 6 pages, 23 references",
    "Data: ISIC 2019 (public), lesion-grouped split with seed 42",
], size=14, color=WHITE, gap=10)
qr = qrcode.QRCode(box_size=10, border=2); qr.add_data(REPO_URL); qr.make(fit=True)
qr_path = C.PROJECT_DIR / "slides" / "qr_repo.png"
qr.make_image(fill_color="#142440", back_color="white").save(qr_path)
rect(s, 9.2, 1.6, 3.4, 4.3, WHITE, MSO_SHAPE.ROUNDED_RECTANGLE)
picture(s, qr_path, 9.4, 1.8, w=3.0, h=3.0)
text(s, 9.3, 4.85, 3.2, 0.9, "Scan for the repository", size=13, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
notes(s, "Leave the QR on screen during questions.")

out = C.PROJECT_DIR / "slides" / "presentation.pptx"
out.parent.mkdir(exist_ok=True)
prs.save(out)
print("saved", out, "·", len(prs.slides), "slides")
