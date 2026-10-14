"""Shared configuration for the ISIC 2019 comparative study."""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths. The raw dataset lives outside the (OneDrive-synced) project folder.
# Override with the ISIC_DATA_DIR environment variable if you move it.
# ---------------------------------------------------------------------------
import os

DATA_DIR = Path(os.environ.get("ISIC_DATA_DIR", r"C:\Users\josep\datasets\ISIC2019"))
RAW_DIR = DATA_DIR / "raw"
RAW_IMG_DIR = RAW_DIR / "ISIC_2019_Training_Input"
RESIZED_DIR = DATA_DIR / "resized256"          # pre-resized JPEGs (shorter side 256, center crop)
GT_CSV = RAW_DIR / "ISIC_2019_Training_GroundTruth.csv"
META_CSV = RAW_DIR / "ISIC_2019_Training_Metadata.csv"

PROJECT_DIR = Path(__file__).resolve().parents[1]
SPLIT_CSV = PROJECT_DIR / "results" / "splits.csv"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------
CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
CLASS_NAMES = {
    "MEL": "Melanoma",
    "NV": "Melanocytic nevus",
    "BCC": "Basal cell carcinoma",
    "AK": "Actinic keratosis",
    "BKL": "Benign keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular lesion",
    "SCC": "Squamous cell carcinoma",
}
NUM_CLASSES = len(CLASSES)

# ---------------------------------------------------------------------------
# Experimental protocol (identical for every model)
# ---------------------------------------------------------------------------
SEED = 42
IMG_SIZE = 224
PRE_RESIZE = 256
SPLIT_FRACS = (0.70, 0.15, 0.15)   # train / val / test, grouped by lesion_id, stratified by class

MODELS = {
    # key -> timm model name
    "resnet50": "resnet50.a1_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "swin_t": "swin_tiny_patch4_window7_224.ms_in1k",
}
MODEL_LABELS = {
    "resnet50": "ResNet-50",
    "efficientnet_b0": "EfficientNet-B0",
    "swin_t": "Swin-T",
    "ensemble": "Ensemble (soft voting)",
}
MODEL_FAMILY = {"resnet50": "CNN", "efficientnet_b0": "CNN", "swin_t": "Transformer", "ensemble": "Hybrid"}

TRAIN = dict(
    epochs=12,
    batch_size=32,
    lr=1e-4,
    weight_decay=0.05,
    warmup_epochs=1,
    label_smoothing=0.1,
    patience=4,            # early stopping on validation macro-F1
    num_workers=6,
    amp=True,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
