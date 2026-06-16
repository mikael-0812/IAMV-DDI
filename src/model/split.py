import os
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, train_test_split


def split_5_fold(
    input_csv="",
    output_dir="",
    seed=123,
    n_splits=5
):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)

    indices = np.arange(len(df))

    splitter = ShuffleSplit(
        n_splits=n_splits,
        train_size=0.8,
        test_size=0.2,
        random_state=seed
    )

    summary = []

    for fold, (train_idx, temp_idx) in enumerate(splitter.split(indices), start=1):
        valid_idx, test_idx = train_test_split(
            temp_idx,
            test_size=1/2,
            random_state=seed + fold,
            shuffle=True
        )

        fold_dir = os.path.join(output_dir, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        df.iloc[train_idx].to_csv(
            os.path.join(fold_dir, "train.csv"),
            index=False
        )

        df.iloc[valid_idx].to_csv(
            os.path.join(fold_dir, "valid.csv"),
            index=False
        )

        df.iloc[test_idx].to_csv(
            os.path.join(fold_dir, "test.csv"),
            index=False
        )

        summary.append({
            "fold": fold,
            "train": len(train_idx),
            "valid": len(valid_idx),
            "test": len(test_idx),
            "total": len(df),
            "train_ratio": len(train_idx) / len(df),
            "valid_ratio": len(valid_idx) / len(df),
            "test_ratio": len(test_idx) / len(df)
        })

        print(
            f"Fold {fold}: "
            f"train={len(train_idx)}, "
            f"valid={len(valid_idx)}, "
            f"test={len(test_idx)}"
        )

    pd.DataFrame(summary).to_csv(
        os.path.join(output_dir, "split_summary.csv"),
        index=False
    )

    print(f"\nSaved splits to: {output_dir}")


if __name__ == "__main__":
    split_5_fold(
    input_csv="dataset/ZhangDDI/ZhangDDI_ddi.csv",
    output_dir="ZhangDDI",
    )