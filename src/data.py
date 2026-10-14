"""Dataset, transforms and DataLoaders shared by every model (identical protocol)."""
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

import config as C


def build_transforms(train: bool):
    norm = T.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD)
    if train:
        return T.Compose([
            T.RandomResizedCrop(C.IMG_SIZE, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(20),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
            T.ToTensor(),
            norm,
        ])
    return T.Compose([
        T.Resize(C.IMG_SIZE),
        T.CenterCrop(C.IMG_SIZE),
        T.ToTensor(),
        norm,
    ])


class ISICDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool):
        self.df = df.reset_index(drop=True)
        self.tf = build_transforms(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(C.RESIZED_DIR / f"{row.image}.jpg").convert("RGB")
        return self.tf(img), int(row.label_idx)


def load_splits():
    df = pd.read_csv(C.SPLIT_CSV)
    return {s: df[df.split == s] for s in ["train", "val", "test"]}


def class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """Inverse square-root frequency weights (softened re-balancing)."""
    counts = train_df["label_idx"].value_counts().sort_index().values.astype(float)
    w = 1.0 / counts ** 0.5
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def make_loaders(batch_size=None, num_workers=None, subset: float = 1.0):
    """subset < 1 keeps a stratified fraction of every split (quick demo / smoke test)."""
    cfg = C.TRAIN
    bs = batch_size or cfg["batch_size"]
    nw = cfg["num_workers"] if num_workers is None else num_workers
    splits = load_splits()
    if subset < 1.0:
        splits = {k: v.groupby("label_idx", group_keys=False).apply(
            lambda g: g.sample(frac=subset, random_state=C.SEED)) for k, v in splits.items()}
    kw = dict(num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
    train = DataLoader(ISICDataset(splits["train"], True), batch_size=bs, shuffle=True, drop_last=True, **kw)
    val = DataLoader(ISICDataset(splits["val"], False), batch_size=bs * 2, shuffle=False, **kw)
    test = DataLoader(ISICDataset(splits["test"], False), batch_size=bs * 2, shuffle=False, **kw)
    return train, val, test, splits
