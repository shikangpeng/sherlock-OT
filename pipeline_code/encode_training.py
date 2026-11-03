#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from tqdm import tqdm

FEATURES = ("lv", "mv", "hv", "sm")
HRF_SHIFT = 3 
DROP_FIRST_N = 26 # cartoon part
MAX_COMPONENTS = 30 
CV_SPLITS = 3

SPLITS = {
    "forward": {"train_part": 1, "test_part": 2, "feature_suffix": "", "output_suffix": ""},
    "backward": {"train_part": 2, "test_part": 1, "feature_suffix": "_rev", "output_suffix": "_rev"},
}


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def compute_rank_accuracy(pred, y_true):
    """Mean normalized rank of the matching TR based on voxel-pattern correlation."""
    pred_z = (pred - pred.mean(axis=1, keepdims=True)) / pred.std(axis=1, ddof=1, keepdims=True)
    true_z = (y_true - y_true.mean(axis=1, keepdims=True)) / y_true.std(axis=1, ddof=1, keepdims=True)
    corr_matrix = (pred_z @ true_z.T) / (pred.shape[1] - 1)

    n_test = pred.shape[0]
    ranks = [np.where(np.argsort(corr_matrix[i])[::-1] == i)[0][0] for i in range(n_test)]
    return float(np.mean(1 - np.asarray(ranks) / (n_test - 1)))


def tune_and_fit_pls_models(x_train, x_test, y_train, y_test, max_components=MAX_COMPONENTS, cv_splits=CV_SPLITS):
    """Select PLS components by training-partition CV, refit, and evaluate on the held-out partition."""
    models = {}
    kfold = KFold(n_splits=cv_splits, shuffle=False)

    for roi, y_tr in tqdm(y_train.items(), desc="ROIs", unit="ROI"):
        y_te = y_test[roi]
        accuracy_curve = []

        for n_comp in range(1, max_components + 1):
            fold_scores = []
            for train_idx, val_idx in kfold.split(x_train):
                pls = PLSRegression(n_components=n_comp, scale=False) # the features were standardized beforehand
                pls.fit(x_train[train_idx], y_tr[train_idx])
                fold_scores.append(compute_rank_accuracy(pls.predict(x_train[val_idx]), y_tr[val_idx]))
            accuracy_curve.append(float(np.mean(fold_scores)))

        best_n = int(np.argmax(accuracy_curve)) + 1
        best_pls = PLSRegression(n_components=best_n, scale=False)
        best_pls.fit(x_train, y_tr)
        heldout_accuracy = compute_rank_accuracy(best_pls.predict(x_test), y_te)

        models[roi] = {
            "model": best_pls,
            "n_components": best_n,
            "accuracy_curve": accuracy_curve,
            "best_cv_accuracy": accuracy_curve[best_n - 1],
            "heldout_accuracy": heldout_accuracy,
        }

    return models


def run_split(sub, split_name, base_dir, out_dir):
    split = SPLITS[split_name]
    train_part = split["train_part"]
    test_part = split["test_part"]
    feature_suffix = split["feature_suffix"]
    output_suffix = split["output_suffix"]

    print(f"\n=== Subject {sub:02d} | {split_name} ===")
    y_train_raw = load_pickle(base_dir / "data" / f"sub-{sub:02d}" / f"sub-{sub:02d}_part{train_part}.pkl")
    y_test_raw = load_pickle(base_dir / "data" / f"sub-{sub:02d}" / f"sub-{sub:02d}_part{test_part}.pkl")
    y_train = {roi: data[DROP_FIRST_N + HRF_SHIFT:] for roi, data in y_train_raw.items()} 
    y_test = {roi: data[DROP_FIRST_N + HRF_SHIFT:] for roi, data in y_test_raw.items()}

    for feature in FEATURES:
        print(f"Subject {sub:02d} | {split_name} | feature {feature}")
        x_train_raw = load_pickle(base_dir / "feature" / "raw" / f"{feature}_part{train_part}z{feature_suffix}.pkl")
        x_test_raw = load_pickle(base_dir / "feature" / "raw" / f"{feature}_part{test_part}z{feature_suffix}.pkl")
        x_train = x_train_raw[DROP_FIRST_N:-HRF_SHIFT]
        x_test = x_test_raw[DROP_FIRST_N:-HRF_SHIFT]

        models = tune_and_fit_pls_models(x_train, x_test, y_train, y_test)
        save_pickle(models, out_dir / f"feature-{feature}_pls_model{output_suffix}.pkl")

        components = {roi: result["n_components"] for roi, result in models.items()}
        save_pickle(components, out_dir / f"feature-{feature}_pls_comps{output_suffix}.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", type=int, required=True)
    parser.add_argument("--direction", choices=("forward", "backward", "both"), default="both")
    parser.add_argument("--base-dir", type=Path, default=Path("/project/sherlock/encoding"))
    parser.add_argument("--scratch-dir", type=Path, default=Path("/scratch/sherlock/encoding"))
    args = parser.parse_args()

    out_dir = args.scratch_dir / "model_info_raw_nocartoon" / f"sub-{args.sub:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    directions = SPLITS if args.direction == "both" else (args.direction,)
    for split_name in directions:
        run_split(args.sub, split_name, args.base_dir, out_dir)


if __name__ == "__main__":
    main()