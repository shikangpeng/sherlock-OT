#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Group aggregation for the fixed-HRF within-scene rank-accuracy analysis.

"""

import argparse
import glob
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import fdrcorrection

filename = "withinscene_rankacc"
group_filename = "withinscene_rankacc_groupstats"


def empirical_p_greater(observed, null_values):
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if not np.isfinite(observed) or null_values.size == 0:
        return np.nan
    return (1 + np.sum(null_values >= observed)) / (1 + null_values.size)

def load_observed(sub, root):
    path = f"{root}/sub-{sub:02d}/sub-{sub:02d}_observed_rankacc.pkl"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    observed = pd.read_pickle(path)
    return observed[observed["method"] == "OT"].copy()

def collect_null(sub, root):
    perm_dir = f"{root}/sub-{sub:02d}/permutation_files"
    paths = sorted(glob.glob(f"{perm_dir}/perm_*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No permutation files found in {perm_dir}")

    null_values = defaultdict(list)
    cols = ["roi", "feature", "method", "rank_acc"]

    for path in paths:
        df = pd.read_pickle(path)[cols]
        df = df[df["method"] == "OT"]

        for row in df.itertuples(index=False):
            value = float(row.rank_acc) if pd.notna(row.rank_acc) else np.nan
            null_values[(int(row.roi), str(row.feature))].append(value)

    return null_values

def subject_results(sub, observed, null_values):
    rows = []

    for row in observed.itertuples(index=False):
        values = np.asarray(
            null_values.get((int(row.roi), str(row.feature)), []),
            dtype=float,
        )
        values = values[np.isfinite(values)]

        obs = float(row.rank_acc) if pd.notna(row.rank_acc) else np.nan
        null_median = float(np.median(values)) if values.size else np.nan
        effect = obs - null_median if np.isfinite(obs) and np.isfinite(null_median) else np.nan

        rows.append({
            "subject": sub,
            "roi": int(row.roi),
            "feature": str(row.feature),
            "method": "OT",
            "obs_rank_acc": obs,
            "null_median_rank_acc": null_median,
            "rank_acc_minus_null_median": effect,
            "emp_p_greater": empirical_p_greater(obs, values),
            "null_n_perm": int(values.size),
        })

    return pd.DataFrame(rows)

def group_wilcoxon(df, value_col, group_keys, alternative, min_subjects):
    rows = []

    for key_values, group in df.groupby(group_keys, sort=True):
        values = group[value_col].dropna().astype(float).to_numpy()

        if values.size < min_subjects:
            stat, p = np.nan, np.nan
        else:
            try:
                stat, p = wilcoxon(
                    values,
                    alternative=alternative,
                    zero_method="wilcox",
                    method="auto",
                )
            except ValueError:
                stat, p = np.nan, np.nan

        key_values = key_values if isinstance(key_values, tuple) else (key_values,)
        record = dict(zip(group_keys, key_values))
        record.update({
            "n_subjects": int(values.size),
            "median_value": float(np.median(values)) if values.size else np.nan,
            "mean_value": float(np.mean(values)) if values.size else np.nan,
            "n_pos": int(np.sum(values > 0)),
            "n_neg": int(np.sum(values < 0)),
            "n_zero": int(np.sum(values == 0)),
            "wilcoxon_stat": stat,
            "p": p,
        })
        rows.append(record)

    out = pd.DataFrame(rows)
    out["q"] = np.nan
    fdr_keys = [key for key in group_keys if key != "roi"]

    for _, indices in out.groupby(fdr_keys).groups.items():
        indices = list(indices)
        valid = out.loc[indices, "p"].notna()

        if valid.any():
            valid_indices = np.asarray(indices)[valid.to_numpy()]
            out.loc[valid_indices, "q"] = fdrcorrection(
                out.loc[valid_indices, "p"].to_numpy()
            )[1]

    return out

def main(subs, min_subjects):
    root = f"{scratch_dir}/recall/model_info_raw/{filename}"
    out_dir = f"{scratch_dir}/recall/model_info_raw/{group_filename}"
    os.makedirs(out_dir, exist_ok=True)

    subject_frames = []

    for sub in subs:
        print(f"Loading sub-{sub:02d}")
        observed = load_observed(sub, root)
        null_values = collect_null(sub, root)
        subject_frames.append(subject_results(sub, observed, null_values))

    subject_df = pd.concat(subject_frames, ignore_index=True)

    group_df = group_wilcoxon(
        subject_df,
        "rank_acc_minus_null_median",
        ["feature", "method", "roi"],
        "greater",
        min_subjects,
    )

    raw_summary = (
        subject_df
        .groupby(["feature", "method", "roi"], as_index=False)
        .agg(
            median_obs_rank_acc=("obs_rank_acc", "median"),
            mean_obs_rank_acc=("obs_rank_acc", "mean"),
        )
    )

    group_df = group_df.merge(
        raw_summary,
        on=["feature", "method", "roi"],
        how="left",
        validate="one_to_one",
    )

    prefix = f"subs-{len(subs):02d}"
    outputs = {
        "subject_per_method": subject_df,
        "group_per_method": group_df,
    }

    for name, frame in outputs.items():
        frame.to_csv(f"{out_dir}/{prefix}_{name}.csv", index=False)
        with open(f"{out_dir}/{prefix}_{name}.pkl", "wb") as f:
            pickle.dump(frame, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved to: {out_dir}")

    summary = group_df.groupby("feature").agg(
        median_accuracy=("median_obs_rank_acc", "median"),
        median_effect=("median_value", "median"),
        n_roi_q05=("q", lambda values: int((values < 0.05).sum())),
        n_roi=("roi", "count"),
    ).reset_index()

    print("\n=== OT within-scene rank accuracy, by feature ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subs", type=int, nargs="+", required=True)
    parser.add_argument("--min_subjects", type=int, default=12)
    args = parser.parse_args()

    scratch_dir = "/scratch/sherlock/schaefer"
    main(subs=args.subs, min_subjects=args.min_subjects)