from pathlib import Path

import pandas as pd

GRID_DIR = Path("/project/sherlock/recall/gridSearch")
SUBJECTS = range(1, 17)

grids = {
    sub: pd.read_csv(GRID_DIR / f"simple_grid_sub-{sub:02d}.csv")
    for sub in SUBJECTS
}

selected = []

for test_sub in SUBJECTS:
    training = pd.concat(
        [grids[sub] for sub in SUBJECTS if sub != test_sub],
        ignore_index=True,
    )

    performance = (
        training.groupby(["alpha", "epsilon"], as_index=False)
        ["mean_acc_all_features"]
        .mean()
        .rename(columns={"mean_acc_all_features": "mean_training_accuracy"})
        .sort_values(
            ["mean_training_accuracy", "alpha", "epsilon"],
            ascending=[False, True, True],
        )
    )

    best = performance.iloc[0]
    selected.append({
        "subject": test_sub,
        "alpha": best["alpha"],
        "epsilon": best["epsilon"],
        "mean_training_accuracy": best["mean_training_accuracy"],
    })

selected = pd.DataFrame(selected)