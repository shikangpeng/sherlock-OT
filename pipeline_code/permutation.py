#!/usr/bin/env python3

import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from tqdm import tqdm

FEATURES = ("lv", "mv", "hv", "sm")
HRF_SHIFT = 3
DROP_FIRST_N = 26

SPLITS = {
    "forward": {"test_part": 2, "feature_suffix": "", "model_suffix": "", "output_suffix": ""},
    "backward": {"test_part": 1, "feature_suffix": "_rev", "model_suffix": "_rev", "output_suffix": "_rev"},
}

_PRED_Z = None
_TRUE_Z = None
_FEATURES = None


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def zscore_rows(x):
    return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, ddof=1, keepdims=True)

def rank_stats_from_z(pred_z, true_z):
    """Rank the matching TR against all observed TRs using voxel-pattern correlation."""
    corr_matrix = (pred_z @ true_z.T) / (pred_z.shape[1] - 1)
    order = np.argsort(corr_matrix, axis=1)[:, ::-1]
    inv = np.empty_like(order)
    inv[np.arange(order.shape[0])[:, None], order] = np.arange(order.shape[1])[None, :]
    ranks = inv[np.arange(order.shape[0]), np.arange(order.shape[0])] + 1
    n_tr = len(ranks)
    rank_accuracy = np.mean(1 - (ranks - 1) / (n_tr - 1))
    return {"normalized_rank_percentage": float(rank_accuracy), "mean_rank": float(ranks.mean()), "n_tr_used": n_tr}

def make_block_perm_order(n_tr, block_len, rng):
    """Shuffle contiguous blocks after a random circular offset."""
    n_blocks = n_tr // block_len
    offset = int(rng.integers(0, block_len))
    rolled = (np.arange(n_tr) + offset) % n_tr
    blocks = rolled.reshape(n_blocks, block_len)
    return blocks[rng.permutation(n_blocks)].reshape(-1)

def predict_patterns(x_test, feature, sub, model_dir, model_suffix):
    model_path = model_dir / f"sub-{sub:02d}" / f"feature-{feature}_pls_model{model_suffix}.pkl"
    models = load_pickle(model_path)
    return {roi: result["model"].predict(x_test) for roi, result in models.items()}

def evaluate_order(pred_z, true_z, features, row_order=None):
    out = {feature: {} for feature in features}
    for feature in features:
        for roi, pred in pred_z[feature].items():
            pred_use = pred if row_order is None else pred[row_order]
            out[feature][roi] = rank_stats_from_z(pred_use, true_z[roi])
    return out

def init_worker(pred_z, true_z, features):
    global _PRED_Z, _TRUE_Z, _FEATURES
    _PRED_Z = pred_z
    _TRUE_Z = true_z
    _FEATURES = features

def worker_perm(args):
    perm, child_seed, n_tr, block_len = args
    rng = np.random.default_rng(child_seed)
    row_order = make_block_perm_order(n_tr, block_len, rng)
    return perm, evaluate_order(_PRED_Z, _TRUE_Z, _FEATURES, row_order)

def run_direction(sub, direction, base_dir, scratch_dir, n_perm, block_len, n_jobs):
    split = SPLITS[direction]
    test_part = split["test_part"]
    feature_suffix = split["feature_suffix"]
    model_suffix = split["model_suffix"]
    output_suffix = split["output_suffix"]

    print(f"\n=== Subject {sub:02d} | {direction} block null ===")
    model_dir = base_dir / "model_info_raw_nocartoon"
    out_dir = scratch_dir / "model_nullDist_nocartoon" / f"sub-{sub:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_test_raw = load_pickle(base_dir / "data" / f"sub-{sub:02d}" / f"sub-{sub:02d}_part{test_part}.pkl")
    y_test = {roi: data[DROP_FIRST_N + HRF_SHIFT:] for roi, data in y_test_raw.items()}

    preds_all = {}
    for feature in FEATURES:
        x_test_raw = load_pickle(base_dir / "feature" / "raw" / f"{feature}_part{test_part}z{feature_suffix}.pkl")
        x_test = x_test_raw[DROP_FIRST_N:-HRF_SHIFT]
        preds_all[feature] = predict_patterns(x_test, feature, sub, model_dir, model_suffix)

    n_test = min(data.shape[0] for data in y_test.values())
    perm_base = (n_test // block_len) * block_len
    print(f"sub-{sub:02d}: n_test={n_test}, perm_base={perm_base}, blocks={perm_base // block_len}")

    true_z = {roi: zscore_rows(data[:perm_base]) for roi, data in y_test.items()}
    pred_z = {
        feature: {roi: zscore_rows(pred[:perm_base]) for roi, pred in preds.items()}
        for feature, preds in preds_all.items()
    }

    observed = evaluate_order(pred_z, true_z, FEATURES)
    save_pickle(observed, out_dir / f"sub-{sub:02d}_observed{output_suffix}.pkl")

    child_seeds = np.random.SeedSequence(sub).spawn(n_perm)
    work = [(perm, child_seeds[perm], perm_base, block_len) for perm in range(n_perm)]
    null_distribution = {feature: {} for feature in FEATURES}

    if n_jobs == 1:
        init_worker(pred_z, true_z, FEATURES)
        results = map(worker_perm, work)
        iterator = tqdm(results, total=n_perm, desc=f"sub-{sub:02d} {direction} null")
        for perm, acc in iterator:
            for feature in FEATURES:
                null_distribution[feature][perm] = acc[feature]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=init_worker, initargs=(pred_z, true_z, FEATURES)) as executor:
            results = executor.map(worker_perm, work, chunksize=8)
            for perm, acc in tqdm(results, total=n_perm, desc=f"sub-{sub:02d} {direction} null"):
                for feature in FEATURES:
                    null_distribution[feature][perm] = acc[feature]

    save_pickle(null_distribution, out_dir / f"sub-{sub:02d}_nullDistribution_all{output_suffix}.pkl")

    metadata = {
        "subject": sub,
        "direction": direction,
        "n_perm": n_perm,
        "block_len": block_len,
        "features": FEATURES,
        "hrf_shift": HRF_SHIFT,
        "drop_first_n": DROP_FIRST_N,
        "n_test": n_test,
        "perm_base": perm_base,
        "null_method": "block_permutation_of_prediction_rows",
        "observed_uses_perm_base": True,
    }
    save_pickle(metadata, out_dir / f"sub-{sub:02d}_nullDist_index{output_suffix}.pkl")
    print(f"Saved null distribution to {out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", type=int, required=True)
    parser.add_argument("--direction", choices=("forward", "backward", "both"), default="forward")
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--block-len", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--base-dir", type=Path, default=Path("/project/sherlock/encoding"))
    parser.add_argument("--scratch-dir", type=Path, default=Path("/scratch/sherlock/encoding"))
    args = parser.parse_args()

    directions = SPLITS if args.direction == "both" else (args.direction,)
    for direction in directions:
        run_direction(args.sub, direction, args.base_dir, args.scratch_dir, args.n_perm, args.block_len, args.n_jobs)

if __name__ == "__main__":
    main()