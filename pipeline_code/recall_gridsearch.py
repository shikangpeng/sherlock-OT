#!/usr/bin/env python3
"""Subject-level grid search for FGW coupling parameters."""

import argparse
import os
import pickle
from pathlib import Path
import numpy as np
import ot
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import ParameterGrid

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

BASE_DIR = Path("/project/sherlock/schaefer")
SCRATCH_DIR = Path("/scratch/sherlock/schaefer")
HRF_SHIFT = 3
MAX_ITER = 2000

def load_annotation():
    anno1 = np.load(BASE_DIR / "recall/movie_data/fused_part1.npy")[26:, :]
    anno2 = np.load(BASE_DIR / "recall/movie_data/fused_part2.npy")[26:, :]
    return np.vstack((anno1, anno2))

def normalize_rows(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def softmax_weights(x):
    weights = np.exp(x - np.max(x))
    return weights / weights.sum()

def build_feature_mixer(ar_embed, annotation_embed, alpha, epsilon, max_iter):
    c_rec = pairwise_distances(ar_embed, metric="cosine")
    c_ann = pairwise_distances(annotation_embed, metric="cosine")
    for cost in (c_rec, c_ann):
        cost /= np.median(cost[np.triu_indices_from(cost, k=1)]) + 1e-12

    m_feat = pairwise_distances(ar_embed, annotation_embed, metric="cosine")
    m_feat /= np.median(m_feat) + 1e-12

    n_rec, n_enc = ar_embed.shape[0], annotation_embed.shape[0]
    p = np.full(n_rec, 1.0 / n_rec)
    q = np.full(n_enc, 1.0 / n_enc)
    transport = ot.gromov.entropic_fused_gromov_wasserstein(
        m_feat, c_rec, c_ann, p, q, alpha=alpha, loss_fun="square_loss",
        epsilon=epsilon, max_iter=max_iter, tol=1e-9, solver="PGD"
    )
    row_sums = transport.sum(axis=1, keepdims=True)
    if np.any(row_sums == 0):
        raise RuntimeError("FGW returned a zero-mass row")
    transport /= row_sums

    z_transport = (transport - transport.mean(axis=0)) / (transport.std(axis=0, ddof=0) + 1e-8)
    return np.vstack([softmax_weights(row) for row in z_transport])

def load_full_matrix(feature):
    with open(BASE_DIR / f"encoding/feature/raw/{feature}_z_full.pkl", "rb") as f:
        feature_matrix = pickle.load(f)
    part1, part2 = feature_matrix[:946, :], feature_matrix[946:, :]
    return np.vstack((part1[26:], part2[26:]))

def model_prediction(feature_data, pls_models):
    predictions = {}
    for roi, model_info in pls_models.items():
        model = None if model_info is None else model_info.get("model")
        predictions[roi] = None if model is None else model.predict(feature_data)
    return predictions

def model_rank_accuracy(y_pred, y_raw, *, offset, min_std=0, eps=1e-8):
    results = {}
    for roi, pred in y_pred.items():
        raw = y_raw.get(roi)
        if pred is None or raw is None:
            results[roi] = None
            continue

        pred_aligned = pred
        raw_aligned = raw[offset:offset + pred.shape[0]]
        if raw_aligned.shape[0] != pred_aligned.shape[0]:
            raise ValueError(f"ROI {roi}: predicted and observed time series do not align")

        mask = (pred_aligned.std(axis=1) > min_std) & (raw_aligned.std(axis=1) > min_std)
        if not mask.any():
            results[roi] = None
            continue
        pred_aligned, raw_aligned = pred_aligned[mask], raw_aligned[mask]

        pred_z = (pred_aligned - pred_aligned.mean(axis=1, keepdims=True)) / (pred_aligned.std(axis=1, keepdims=True) + eps)
        raw_z = (raw_aligned - raw_aligned.mean(axis=1, keepdims=True)) / (raw_aligned.std(axis=1, keepdims=True) + eps)
        corr = pred_z @ raw_z.T / pred_z.shape[1]

        order = np.argsort(-corr, axis=1)
        inverse_order = np.empty_like(order)
        inverse_order[np.arange(order.shape[0])[:, None], order] = np.arange(order.shape[1])[None, :]
        ranks = inverse_order[np.arange(corr.shape[0]), np.arange(corr.shape[0])] + 1

        n_ranked = ranks.size
        normalized_rank = np.ones_like(ranks, dtype=float) if n_ranked == 1 else 1 - (ranks - 1) / (n_ranked - 1)
        results[roi] = {
            "normalized_rank_accuracy": float(normalized_rank.mean()),
            "mean_rank": float(ranks.mean()),
            "n_tr_used": int(n_ranked),
            "offset_used": int(offset),
        }
    return results

def apply_hrf_shift(x):
    return x[:-HRF_SHIFT]

def mean_roi_accuracy(roi_results):
    values = np.array([
        result["normalized_rank_accuracy"] for result in roi_results.values() if result is not None
    ], dtype=float)
    return float(np.nanmean(values))

def evaluate_feature_mixer(mixer, feature, y_raw, tr_onset, pls_models):
    aligned_features = mixer @ load_full_matrix(feature)
    feature_data = apply_hrf_shift(aligned_features)
    predictions = model_prediction(feature_data, pls_models)
    return mean_roi_accuracy(model_rank_accuracy(predictions, y_raw, offset=tr_onset))

def eval_param_for_subject(params, subject, annotation_embed, features, max_iter):
    recall_path = BASE_DIR / f"recall/recall_blocks/sub-{subject:02d}/sub-{subject:02d}_recall_data_embeds.csv"
    ar_embed = normalize_rows(pd.read_csv(recall_path).values)
    mixer = build_feature_mixer(ar_embed, annotation_embed, params["alpha"], params["epsilon"], max_iter)

    with open(BASE_DIR / f"encoding/data/sub-{subject:02d}/sub-{subject:02d}_recall.pkl", "rb") as f:
        y_raw = pickle.load(f)

    tr_onsets = pd.read_csv(BASE_DIR / "recall/recall_blocks/onsets.csv")
    tr_onset = int(tr_onsets.iloc[subject - 1]["TR"] - 1)

    feature_scores = []
    for feature in features:
        model_path = BASE_DIR / f"encoding/model_info_raw/sub-{subject:02d}/feature-{feature}_pls_full.pkl"
        with open(model_path, "rb") as f:
            pls_models = pickle.load(f)

        y_raw_masked = {}
        for roi, raw in y_raw.items():
            model_info = pls_models.get(roi)
            model = None if model_info is None else model_info.get("model")
            y_raw_masked[roi] = None if model is None else raw[:, model_info["voxel_mask"]]

        feature_scores.append(evaluate_feature_mixer(mixer, feature, y_raw_masked, tr_onset, pls_models))

    level_scores = dict.fromkeys(("low", "mid", "high", "sem"), np.nan)
    if len(feature_scores) == 4:
        level_scores.update(zip(level_scores, feature_scores))

    return {
        "subject": int(subject),
        "alpha": params["alpha"],
        "epsilon": params["epsilon"],
        "low_level_mean_acc": float(level_scores["low"]),
        "mid_level_mean_acc": float(level_scores["mid"]),
        "high_level_mean_acc": float(level_scores["high"]),
        "sem_level_mean_acc": float(level_scores["sem"]),
        "mean_acc_all_features": float(np.mean(feature_scores)),
        "n_features": len(features),
    }

def grid_per_subject(subject, features, param_grid, n_jobs_params, max_iter=MAX_ITER):
    annotation_embed = normalize_rows(load_annotation())
    out_dir = SCRATCH_DIR / "recall/gridSearch"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"simple_grid_sub-{subject:02d}.csv"
    param_list = list(ParameterGrid(param_grid))

    def evaluate(params):
        return eval_param_for_subject(params, subject, annotation_embed, features, max_iter)

    if n_jobs_params != 1:
        rows = Parallel(n_jobs=n_jobs_params, backend="loky", prefer="processes")(
            delayed(evaluate)(params) for params in param_list
        )
    else:
        rows = [evaluate(params) for params in param_list]

    columns = [
        "subject", "alpha", "epsilon", "low_level_mean_acc", "mid_level_mean_acc",
        "high_level_mean_acc", "sem_level_mean_acc", "mean_acc_all_features", "n_features"
    ]
    pd.DataFrame(rows)[columns].to_csv(out_csv, index=False)
    print(f"sub-{subject:02d}: wrote {len(rows)} rows to {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", type=int, required=True)
    parser.add_argument("--n-jobs-params", type=int, required=True)
    args = parser.parse_args()

    features = ["lv", "mv", "hv", "sm"]
    param_grid = {
        "alpha": np.round(np.linspace(0, 1, 11), 6),
        "epsilon": np.round(np.linspace(0.5, 8.0, 16), 6),
    }
    grid_per_subject(args.sub, features, param_grid, args.n_jobs_params)


if __name__ == "__main__":
    main()