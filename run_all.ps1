# Full pipeline (Windows PowerShell). Run from the project root.
$ErrorActionPreference = "Stop"
python src/prepare_data.py
foreach ($m in @("resnet50", "efficientnet_b0", "swin_t")) { python src/train.py --model $m }
python src/evaluate.py --all
python src/ensemble.py
python src/gradcam.py
python src/summarize.py
