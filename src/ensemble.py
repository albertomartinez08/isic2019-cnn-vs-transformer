"""Step 4 (creativity): soft-voting ensemble of the three fine-tuned models on the test split.

Also reports the CNN-only ensemble and the best CNN + Transformer pair so the paper can discuss
whether the Transformer adds complementary information.
"""
import itertools
import json

import numpy as np

import config as C
from evaluate import metrics_from_probs, plot_confusion

if __name__ == "__main__":
    targets = np.load(C.RESULTS_DIR / "test_targets.npy")
    probs = {k: np.load(C.RESULTS_DIR / k / "test_probs.npy") for k in C.MODELS}
    costs = {k: json.load(open(C.RESULTS_DIR / k / "test_metrics.json")) for k in C.MODELS}

    results = {}
    for r in (2, 3):
        for combo in itertools.combinations(C.MODELS, r):
            p = np.mean([probs[k] for k in combo], axis=0)
            m = metrics_from_probs(p, targets)
            m["members"] = list(combo)
            m["params_M"] = sum(costs[k]["params_M"] for k in combo)
            m["gflops"] = sum(costs[k]["gflops"] for k in combo)
            m["latency_ms_b1"] = sum(costs[k]["latency_ms_b1"] for k in combo)
            m["latency_ms_b32"] = sum(costs[k]["latency_ms_b32"] for k in combo)
            m["throughput_img_s_b32"] = 32 / (m["latency_ms_b32"] / 1000)
            results["+".join(combo)] = m
            print(f"{'+'.join(combo):40s} acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} AUC={m['macro_auc']:.4f}")

    full = results["+".join(C.MODELS)]
    out = C.RESULTS_DIR / "ensemble"
    out.mkdir(exist_ok=True)
    full["model"] = "ensemble"
    with open(out / "test_metrics.json", "w") as f:
        json.dump(full, f, indent=2)
    with open(out / "all_combinations.json", "w") as f:
        json.dump(results, f, indent=2)
    plot_confusion(full["confusion"], "Ensemble of 3 (test)", C.FIGURES_DIR / "cm_ensemble.png")
