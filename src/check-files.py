
import os
import pandas as pd
import torch
from model import Net

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT, "data", "processed")
RESULTS_DIR = os.path.join(ROOT, "results")
MODELS_DIR = os.path.join(ROOT, "models")

EPS = [0.5, 1, 2, 5, 10, 100]

NAMES = ["baseline_e50", "baseline_early"] + [
    f"dpsgd_eps{e}" for e in EPS
]

data = [
    "target_train.csv",
    "target_test.csv",
    "shadow_train.csv",
    "shadow_test.csv"
]

results = [
    "multiseed_results.csv",
    "simple_attacks.csv",
    "lira_results.csv",
    "attribute_inference.csv",
    "exposure_baseline_e50.csv",
    "exposure_features_baseline_e50.csv"
]

models = [
    f"{n}_seed{s}.pt"
    for n in NAMES
    for s in range(1, 6)
] + ["shadow_e50_seed1.pt"]


def missing(folder, files):
    return [
        f for f in files
        if not os.path.exists(os.path.join(folder, f))
    ]


for folder, files in [
    (DATA_DIR, data),
    (RESULTS_DIR, results),
    (MODELS_DIR, models)
]:
    miss = missing(folder, files)

    print(
        f"{folder}: {len(files) - len(miss)}/{len(files)} found",
        "| MISSING:" if miss else "",
        miss if miss else ""
    )


# Validate model only if required files exist
data_file = os.path.join(DATA_DIR, "target_train.csv")
model_file = os.path.join(MODELS_DIR, "baseline_e50_seed1.pt")
results_file = os.path.join(RESULTS_DIR, "multiseed_results.csv")

if all(os.path.exists(f) for f in [
    data_file, model_file, results_file
]):

    input_dim = pd.read_csv(data_file, nrows=1).shape[1] - 1

    m = Net(input_dim)

    m.load_state_dict(
        torch.load(model_file, map_location="cpu")
    )

    print("\nModel loads fine, input_dim =", input_dim)

    print(
        pd.read_csv(results_file)
        .groupby("model")["attack_auc"]
        .mean()
        .round(3)
    )

else:
    print("\nModel validation skipped due to missing files.")
