"""Step 1: unzip ISIC 2019, build a leakage-free stratified split and pre-resize images.

Usage:
    python src/prepare_data.py            # does everything that is still missing
    python src/prepare_data.py --skip-resize

Outputs:
    results/splits.csv                 image, label, label_idx, split, lesion_id
    <DATA_DIR>/resized256/*.jpg        shorter side 256 px, center-cropped to 256x256
    results/dataset_stats.json         class counts per split
"""
import argparse
import json
import zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

import config as C


def unzip_if_needed():
    if C.RAW_IMG_DIR.exists() and any(C.RAW_IMG_DIR.glob("*.jpg")):
        n = len(list(C.RAW_IMG_DIR.glob("*.jpg")))
        print(f"[prepare] raw images already extracted ({n} files)")
        return
    zpath = C.RAW_DIR / "ISIC_2019_Training_Input.zip"
    print(f"[prepare] extracting {zpath} ...")
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(C.RAW_DIR)
    print("[prepare] extraction done")


def build_split():
    gt = pd.read_csv(C.GT_CSV)
    meta = pd.read_csv(C.META_CSV)
    gt = gt.drop(columns=[c for c in gt.columns if c == "UNK"])
    labels = gt[C.CLASSES].values.argmax(1)
    df = pd.DataFrame({"image": gt["image"], "label_idx": labels})
    df["label"] = [C.CLASSES[i] for i in labels]
    df = df.merge(meta[["image", "lesion_id", "age_approx", "anatom_site_general", "sex"]], on="image", how="left")
    # Several images can belong to the same lesion (HAM10000 subset). Group them so that
    # the same lesion never appears in two splits.
    df["group"] = df["lesion_id"].fillna(df["image"])

    rng = C.SEED
    # 70/15/15 via two StratifiedGroupKFold passes (approximately stratified, group-safe)
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=rng)
    folds = np.zeros(len(df), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(df, df["label_idx"], df["group"])):
        folds[idx] = k
    split = np.where(folds < 7, "train", np.where(folds < 8, "val", "test"))
    # folds 0-6 -> train (70%), 7 -> val (10%)... we want 15/15: use folds 7,8 (val) partially.
    # Simpler and exact enough: 7 folds train, 1.5 folds val, 1.5 folds test.
    split = np.array(["train"] * len(df), dtype=object)
    split[folds == 7] = "val"
    split[folds == 8] = "test"
    # fold 9: split by group hash into val/test halves
    f9 = np.where(folds == 9)[0]
    groups9 = df.loc[f9, "group"].values
    uniq = np.unique(groups9)
    rs = np.random.RandomState(rng)
    rs.shuffle(uniq)
    val_groups = set(uniq[: len(uniq) // 2])
    split[f9] = ["val" if g in val_groups else "test" for g in groups9]
    df["split"] = split

    df.to_csv(C.SPLIT_CSV, index=False)
    stats = {s: df[df.split == s]["label"].value_counts().to_dict() for s in ["train", "val", "test"]}
    stats["total"] = df["label"].value_counts().to_dict()
    stats["n"] = {s: int((df.split == s).sum()) for s in ["train", "val", "test"]}
    with open(C.RESULTS_DIR / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("[prepare] split sizes:", stats["n"])
    print(pd.crosstab(df["split"], df["label"]))
    # leakage check
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        inter = set(df[df.split == a]["group"]) & set(df[df.split == b]["group"])
        assert not inter, f"lesion leakage between {a} and {b}: {len(inter)}"
    print("[prepare] no lesion overlap between splits")
    return df


def _resize_one(args):
    src, dst = args
    if dst.exists():
        return
    try:
        im = Image.open(src).convert("RGB")
        w, h = im.size
        s = C.PRE_RESIZE / min(w, h)
        im = im.resize((max(C.PRE_RESIZE, round(w * s)), max(C.PRE_RESIZE, round(h * s))), Image.BICUBIC)
        w, h = im.size
        left, top = (w - C.PRE_RESIZE) // 2, (h - C.PRE_RESIZE) // 2
        im = im.crop((left, top, left + C.PRE_RESIZE, top + C.PRE_RESIZE))
        im.save(dst, quality=95)
    except Exception as e:  # noqa
        print("resize failed", src, e)


def resize_all(df):
    C.RESIZED_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(C.RAW_IMG_DIR / f"{img}.jpg", C.RESIZED_DIR / f"{img}.jpg") for img in df["image"]]
    jobs = [j for j in jobs if not j[1].exists()]
    print(f"[prepare] resizing {len(jobs)} images to {C.PRE_RESIZE}px ...")
    with Pool(8) as pool:
        for i, _ in enumerate(pool.imap_unordered(_resize_one, jobs, chunksize=64)):
            if i % 2000 == 0:
                print(f"  {i}/{len(jobs)}")
    print("[prepare] resize done:", len(list(C.RESIZED_DIR.glob('*.jpg'))), "files")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-resize", action="store_true")
    a = ap.parse_args()
    C.RESULTS_DIR.mkdir(exist_ok=True)
    n_cached = len(list(C.RESIZED_DIR.glob("*.jpg"))) if C.RESIZED_DIR.exists() else 0
    if n_cached >= 25331:
        # Fast path: the 256-px cache (≈600 MB) is already present, the 9 GB raw zip is not needed.
        print(f"[prepare] resized cache complete ({n_cached} files); skipping unzip/resize")
        build_split()
    else:
        unzip_if_needed()
        df = build_split()
        if not a.skip_resize:
            resize_all(df)
