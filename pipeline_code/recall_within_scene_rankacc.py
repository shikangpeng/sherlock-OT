#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subject-level OT within-scene rank-accuracy analysis.

Each recall TR is ranked only against observed TRs from the same labeled
scene, holding event identity constant.
"""

import argparse
import glob
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from tqdm import tqdm

HRF_LAG_TR = 3
FEATURES = ["lv", "mv", "hv", "sm"]
filename = "withinscene_rankacc"

_CELLS = None
_PER_PERM_DIR = None


def load_full_matrix(feature):
    with open(f"{base_dir}/encoding/feature/raw/{feature}_z_full.pkl", "rb") as f:
        matrix = pickle.load(f)
    run1, run2 = matrix[:946, :], matrix[946:, :]
    return np.vstack((run1[26:], run2[26:])) #remove the 26 cartoon TRs in movies because they have no corresponding recall TRs

def load_pls(feature, sub):
    path = f"{base_dir}/encoding/model_info_raw/sub-{sub:02d}/feature-{feature}_pls_full.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)

def predict_recall_patterns(plan, feature, pls):
    """Predict BOLD patterns with a fixed 3-TR HRF lag."""
    X = (plan @ load_full_matrix(feature))[:-HRF_LAG_TR]
    predictions = {}
    for roi, data in pls.items():
        predictions[roi] = None if data is None or data.get("model") is None else data["model"].predict(X)
    return predictions

def load_recall_scene_rows(sub):
    df = pd.read_csv(f"{base_dir}/recall/recall_blocks/scene_TRs_recall.csv")
    sub_df = df[df["Sub"] == sub]
    return [(int(r.Scenes), int(r.Onset), int(r.Offset)) for r in sub_df.itertuples()]


def build_scene_labels(n_rec, recall_rows):
    recalled_mask = np.zeros(n_rec, dtype=bool)
    scene_label = np.full(n_rec, -1, dtype=int)

    for scene_id, r_on, r_off in recall_rows:
        r_s, r_e = max(r_on - 1, 0), min(r_off, n_rec)
        if r_e <= r_s:
            continue

        recalled_mask[r_s:r_e] = True
        scene_label[r_s:r_e] = scene_id

    return recalled_mask, scene_label

def make_within_scene_perm_order(scene_ids, rng):
    order = np.arange(scene_ids.size)
    for scene_id in np.unique(scene_ids):
        idx = np.flatnonzero(scene_ids == scene_id)
        if idx.size > 1:
            order[idx] = idx[rng.permutation(idx.size)]
    return order

def load_subject_recall(sub):
    path = f"{base_dir}/encoding/data/sub-{sub:02d}/sub-{sub:02d}_recall.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)

def _zscore_rows(X, eps=1e-8):
    return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + eps)

def _within_scene_rank_acc(Pz, Rz, scene_ids):
    """Mean normalized diagonal rank, with each query ranked only within its scene."""
    if Pz.shape[0] < 2:
        return np.nan

    total_rank_sum = 0.0
    total_rows = 0
    for scene_id in np.unique(scene_ids):
        idx = np.flatnonzero(scene_ids == scene_id)
        m = idx.size
        if m < 2:
            continue

        P, R = Pz[idx], Rz[idx]
        corr = P @ R.T / P.shape[1]
        order = np.argsort(-corr, axis=1)
        inverse = np.empty_like(order)
        inverse[np.arange(m)[:, None], order] = np.arange(m)[None, :]
        ranks = inverse[np.arange(m), np.arange(m)] + 1
        total_rank_sum += float((1.0 - (ranks - 1) / (m - 1)).sum())
        total_rows += m

    return np.nan if total_rows == 0 else total_rank_sum / total_rows

def precompute_cells(rois, y_raw, pred_ot, feature_list, bold_offset, recalled_mask, scene_label,
                     eps=1e-8, min_scene_rows=2):
    cells = {}
    for roi in rois:
        observed = y_raw.get(roi)
        if observed is None or observed.ndim != 2:
            continue

        for feature in feature_list:
            pred = pred_ot[feature].get(roi)
            if pred is None or observed.shape[1] != pred.shape[1]:
                continue

            common = min(observed.shape[0] - bold_offset, pred.shape[0], recalled_mask.size)
            if common < min_scene_rows:
                continue

            obs = observed[bold_offset:bold_offset + common]
            pred = pred[:common]
            rmask, labels = recalled_mask[:common], scene_label[:common]
            valid = (obs.std(axis=1) > 0) & (pred.std(axis=1) > 0) & rmask
            idx = np.flatnonzero(valid)
            if idx.size < min_scene_rows:
                continue

            scene_ids = labels[idx]
            _, counts = np.unique(scene_ids, return_counts=True)
            if not np.any(counts >= min_scene_rows):
                continue

            cells[(roi, feature)] = {
                "Pz": _zscore_rows(pred[idx], eps),
                "Rz": _zscore_rows(obs[idx], eps),
                "scene": scene_ids,
            }
    return cells

def rows_from_cells(cells, sub, perm_idx, is_null, rng=None):
    rows = []
    for roi, feature in sorted(cells):
        cell = cells[(roi, feature)]
        scene_ids = cell["scene"]
        row_order = make_within_scene_perm_order(scene_ids, rng) if is_null else None
        Pz = cell["Pz"][row_order] if is_null else cell["Pz"]
        rank_acc = _within_scene_rank_acc(Pz, cell["Rz"], scene_ids)

        rows.append({
            "subject": sub, "roi": roi + 1, "perm": perm_idx, "is_null": is_null,
            "method": "OT", "feature": feature, "rank_acc": rank_acc,
        })
    return rows

def _init_perm_worker(cells, per_perm_dir):
    global _CELLS, _PER_PERM_DIR
    _CELLS, _PER_PERM_DIR = cells, per_perm_dir

def _perm_worker(args):
    perm_idx, child_seed, sub = args
    rng = np.random.default_rng(child_seed)
    rows = rows_from_cells(_CELLS, sub, perm_idx, is_null=True, rng=rng)
    with open(f"{_PER_PERM_DIR}/perm_{perm_idx:05d}.pkl", "wb") as f:
        pickle.dump(pd.DataFrame(rows), f, protocol=pickle.HIGHEST_PROTOCOL)
    return perm_idx

def subject_permutation_null(sub, n_perm=1000, n_jobs=1, save_merged=False, min_scene_rows=2):
    print(f"==== Processing sub-{sub:02d} ====")

    with open(f"{fgw_dir}/sub-{sub:02d}/sub-{sub:02d}_FGW.pkl", "rb") as f:
        plan_ot = np.asarray(pickle.load(f), dtype=float)
    n_rec, n_enc = plan_ot.shape
    print(f"sub-{sub:02d}: OT plan {n_rec} recall x {n_enc} encoding TRs")

    recall_rows = load_recall_scene_rows(sub)
    recalled_mask, scene_label = build_scene_labels(n_rec, recall_rows)

    source_mask = recalled_mask[:-HRF_LAG_TR]
    source_scene = scene_label[:-HRF_LAG_TR]

    y_raw = load_subject_recall(sub)
    tr_onsets = pd.read_csv(f"{base_dir}/recall/recall_blocks/sub_onsetTRs.csv")
    recall_bold_start = int(tr_onsets.loc[sub - 1, "TR"] - 1)
    bold_offset = recall_bold_start

    pred_ot = {}
    for feature in FEATURES:
        pls = load_pls(feature, sub)
        pred_ot[feature] = predict_recall_patterns(plan_ot, feature, pls)

    cells = precompute_cells(
        sorted(y_raw), y_raw, pred_ot, FEATURES, bold_offset, source_mask, source_scene,
        min_scene_rows=min_scene_rows,
    )

    out_dir = f"{scratch_dir}/recall/model_info_raw/{filename}/sub-{sub:02d}"
    per_perm_dir = f"{out_dir}/permutation_files"
    os.makedirs(per_perm_dir, exist_ok=True)

    observed = pd.DataFrame(rows_from_cells(cells, sub, -1, is_null=False))
    observed.to_pickle(f"{out_dir}/sub-{sub:02d}_observed_rankacc.pkl")

    diagnostics = []
    for (roi, feature), cell in cells.items():
        scene_ids = cell["scene"]
        unique, counts = np.unique(scene_ids, return_counts=True)
        multi = counts[counts >= min_scene_rows]
        diagnostics.append({
            "subject": sub, "roi": roi + 1, "feature": feature, "n_S": int(scene_ids.size),
            "n_scenes_total": int(unique.size), "n_scenes_multiTR": int(multi.size),
            "n_rows_scored": int(multi.sum()), "mean_scene_size": float(multi.mean()) if multi.size else np.nan,
        })
    pd.DataFrame(diagnostics).to_pickle(f"{out_dir}/sub-{sub:02d}_scene_diag.pkl")

    child_seeds = np.random.SeedSequence(sub).spawn(n_perm)
    work = [(perm, child_seeds[perm], sub) for perm in range(n_perm)]

    if n_jobs <= 1:
        _init_perm_worker(cells, per_perm_dir)
        for item in tqdm(work, desc=f"sub-{sub:02d} null"):
            _perm_worker(item)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_perm_worker, initargs=(cells, per_perm_dir)) as executor:
            for _ in tqdm(executor.map(_perm_worker, work, chunksize=8), total=n_perm, desc=f"sub-{sub:02d} null"):
                pass

    if save_merged:
        paths = sorted(glob.glob(f"{per_perm_dir}/perm_*.pkl"))
        merged = [pd.read_pickle(path) for path in paths]
        pd.concat(merged, ignore_index=True).to_pickle(f"{out_dir}/sub-{sub:02d}_null_rankacc.pkl")

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", type=int, required=True)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--min-scene-rows", type=int, default=2)
    parser.add_argument("--save-merged", action="store_true")
    args = parser.parse_args()

    base_dir = "/project/sherlock/schaefer"
    scratch_dir = "/scratch/sherlock/schaefer"
    fgw_dir = f"{base_dir}/recall/outcome/raw/model_outcome"
    subject_permutation_null(
        sub=args.sub, n_perm=args.n_perm, n_jobs=args.n_jobs,
        save_merged=args.save_merged, min_scene_rows=args.min_scene_rows,
    )