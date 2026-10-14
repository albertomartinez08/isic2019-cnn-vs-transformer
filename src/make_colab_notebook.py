"""Builds notebooks/ISIC2019_CNN_vs_Transformer.ipynb: a Colab-ready notebook that reproduces the
whole pipeline (download -> split -> train 3 models -> evaluate -> Grad-CAM -> tables).

The notebook reuses the project scripts verbatim (they are copied into the Colab runtime), so
there is a single source of truth for the experimental protocol.
"""
import json
from pathlib import Path

import config as C

SRC = C.PROJECT_DIR / "src"
scripts = ["config.py", "prepare_data.py", "data.py", "models.py", "train.py", "evaluate.py", "ensemble.py", "gradcam.py", "summarize.py"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}


cells = [
    md("# CNNs vs Vision Transformers on ISIC 2019\n\n"
       "Comparative evaluation of **ResNet-50, EfficientNet-B0 and Swin-T** (ImageNet-pretrained, identical fine-tuning "
       "protocol) on the 8-class ISIC 2019 dermoscopy dataset. Final project, Deep Learning (UPE, Oct 2026).\n\n"
       "**Runtime → Change runtime type → GPU (T4 or better).** A T4 needs roughly 25-35 min per model.\n\n"
       "Pipeline: 1) install, 2) download ISIC 2019 (9.1 GB, public), 3) leakage-free split + 256-px cache, "
       "4) train the three backbones, 5) test metrics + cost, 6) ensemble, 7) Grad-CAM, 8) tables & figures."),
    code("!pip -q install timm grad-cam torchmetrics\nimport torch; print(torch.__version__, torch.cuda.get_device_name(0))"),
    md("## 1. Project code\nThe scripts below are the same files as in the repository (`src/`)."),
    code("import os, pathlib\nos.makedirs('src', exist_ok=True); os.makedirs('results', exist_ok=True); os.makedirs('figures', exist_ok=True)\n"
         "os.environ['ISIC_DATA_DIR'] = '/content/isic2019'   # dataset location on the Colab disk"),
]
for name in scripts:
    body = (SRC / name).read_text(encoding="utf-8")
    cells.append(code(f"%%writefile src/{name}\n{body}"))

cells += [
    md("## 2a. Fast path (recommended for reviewers): 256-px cache, ≈600 MB instead of 9 GB\n"
       "The pre-resized images (`resized256.zip`) and the two label CSVs are shared on Google Drive. "
       "Run this cell and **skip 2b**. `prepare_data.py` detects the cache and only rebuilds the split."),
    code("# Google Drive folder: https://drive.google.com/drive/folders/152UjJSMcixu5zDfhYwLnEinIV8gwSw4I\nZIP_ID = '1jzoumLJDOob85qd_8MlgMzdXZi8beIvk'        # resized256.zip (442 MB)\nGT_ID  = '1uQcn9lb5dZUJYE_hFLKOQEvib5MkHpHR'        # ISIC_2019_Training_GroundTruth.csv\nMETA_ID = '1Z_Rrds8raaBmNqmD_CuwlXvJmY2wrf9j'       # ISIC_2019_Training_Metadata.csv\n!pip -q install gdown && mkdir -p /content/isic2019/raw\n!cd /content/isic2019/raw && gdown --id $GT_ID -O ISIC_2019_Training_GroundTruth.csv && gdown --id $META_ID -O ISIC_2019_Training_Metadata.csv\n!cd /content/isic2019 && gdown --id $ZIP_ID -O resized256.zip && unzip -q -o resized256.zip && ls resized256 | wc -l"),
    md("### Quick smoke test (≈3 min on a T4): 10 % of the data, 2 epochs\n"
       "`--subset 0.1` writes to `results/demo_<model>` and does not touch the real results."),
    code("!cd /content && python src/prepare_data.py && python src/train.py --model efficientnet_b0 --subset 0.1 --epochs 2 --workers 2"),
    md("## 2b. Full path: download ISIC 2019 (public, no account, 9.1 GB)\nOnly needed if you skipped 2a."),
    code("!mkdir -p /content/isic2019/raw && cd /content/isic2019/raw && \\\n"
         "  wget -q -c https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv && \\\n"
         "  wget -q -c https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv && \\\n"
         "  wget -c https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip && ls -la"),
    md("## 3. Split (grouped by lesion, stratified by class) and 256-px cache"),
    code("!cd /content && python src/prepare_data.py"),
    md("## 4. Train the three backbones with the identical protocol"),
    code("!cd /content && python src/train.py --model resnet50 --workers 2"),
    code("!cd /content && python src/train.py --model efficientnet_b0 --workers 2"),
    code("!cd /content && python src/train.py --model swin_t --workers 2"),
    md("## 5. Test metrics, latency, FLOPs and confusion matrices"),
    code("!cd /content && python src/evaluate.py --all"),
    md("## 6. Soft-voting ensemble (creativity)"),
    code("!cd /content && python src/ensemble.py"),
    md("## 7. Interpretability: Grad-CAM"),
    code("!cd /content && python src/gradcam.py\nfrom IPython.display import Image, display; display(Image('figures/gradcam_grid.png'))"),
    md("## 8. Summary tables and figures"),
    code("!cd /content && python src/summarize.py\nimport pandas as pd; pd.read_csv('results/summary.csv')"),
    code("from IPython.display import Image, display\nfor f in ['curves.png','acc_vs_cost.png','per_class_f1.png','cm_resnet50.png','cm_efficientnet_b0.png','cm_swin_t.png']:\n    display(Image('figures/'+f))"),
    md("## 9. Save outputs to Drive (optional)"),
    code("# from google.colab import drive; drive.mount('/content/drive')\n# !cp -r results figures /content/drive/MyDrive/isic2019_project/"),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = C.PROJECT_DIR / "notebooks" / "ISIC2019_CNN_vs_Transformer.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("saved", out)
