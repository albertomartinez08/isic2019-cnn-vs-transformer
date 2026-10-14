#!/usr/bin/env bash
# Full pipeline (Linux / Colab). Run from the project root.
set -e
python src/prepare_data.py
for m in resnet50 efficientnet_b0 swin_t; do python src/train.py --model "$m"; done
python src/evaluate.py --all
python src/ensemble.py
python src/gradcam.py
python src/summarize.py
