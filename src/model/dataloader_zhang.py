from __future__ import annotations

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Optional, List
from sklearn.model_selection import StratifiedKFold

# idx for drug
idx_map = pd.read_csv('dataset/ZhangDDI/drug_list_zhang.csv')
drugid2idx = dict(zip(idx_map["drugbank_id"], idx_map["idx"]))

class DDIBinaryPairDataset(Dataset):
    def __init__(
        self,
        path: str,
        drugid2idx: dict,
        valid_set: Optional[List[int]] = None,
        delimiter: str = ",",
        col_h: str = "drugbank_id_1",
        col_t: str = "drugbank_id_2",
        col_y: str = "label",
        directed: bool = True,
        strict: bool = True,
    ):
        df = pd.read_csv(path, sep=delimiter)

        required_cols = [col_h, col_t, col_y]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing columns {missing}. Current columns: {list(df.columns)}"
            )

        before = len(df)
        df = df[[col_h, col_t, col_y]].copy()

        # Map DrugBank ID -> integer index
        df["h_idx"] = df[col_h].map(drugid2idx)
        df["t_idx"] = df[col_t].map(drugid2idx)
        df = df.dropna(subset=["h_idx", "t_idx", col_y])

        df["h_idx"] = df["h_idx"].astype(int)
        df["t_idx"] = df["t_idx"].astype(int)
        df[col_y] = df[col_y].astype(float)
        df = df[df[col_y].isin([0, 1, 0.0, 1.0])]

        if valid_set is not None:
            valid = set(int(x) for x in valid_set)
            df = df[
                df["h_idx"].isin(valid) & df["t_idx"].isin(valid)
            ].reset_index(drop=True)

        after = len(df)

        if strict and after < before:
            print(f"[warn] dropped {before - after} rows due to missing mapping/cache/label")

        self.h = df["h_idx"].to_numpy(np.int64)
        self.t = df["t_idx"].to_numpy(np.int64)
        self.y = df[col_y].to_numpy(np.float32)
        self.r = np.zeros(len(df), dtype=np.int64)

        self.directed = directed

    def __len__(self):
        return len(self.h)

    def __getitem__(self, i: int):
        return (
            int(self.h[i]),
            int(self.t[i]),
            int(self.r[i]),
            float(self.y[i]),
        )

def collate_ddi_binary_directed_file(batch):
    h, t, r, y = zip(*batch)
    h = torch.tensor(h, dtype=torch.long)
    t = torch.tensor(t, dtype=torch.long)
    r = torch.tensor(r, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.float32)
    return h, t, r, y

def split_indices(n: int, val_frac: float = 0.1, seed: int = 33):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_val = int(round(n * val_frac))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    return train_idx, val_idx


def split_indices_3way(
    n: int,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 33,
):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))

    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    return train_idx, val_idx, test_idx

def make_loader(ds, batch_size, shuffle, num_workers):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_ddi_binary_directed_file,
    )

def make_dataset(path, drugid2idx, valid_set):
    return DDIBinaryPairDataset(
        path=path,
        drugid2idx=drugid2idx,
        valid_set=valid_set,
        col_h="drugbank_id_1",
        col_t="drugbank_id_2",
        col_y="label",
    )

def split_train_val_from_indices(indices, val_frac: float = 0.1, seed: int = 33):
    indices = np.asarray(indices)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_val = int(round(len(indices) * val_frac))

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    return train_idx, val_idx


def build_loaders_binary_directed_from_file(
    train_path: str,
    val_path: Optional[str],
    test_path: Optional[str],
    batch_size: int,
    valid_set: Optional[List[int]],
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 33,
    num_workers: int = 0,
):
    train_full = make_dataset(train_path, drugid2idx, valid_set)

    if val_path is None and test_path is None:
        tr_idx, va_idx, te_idx = split_indices_3way(
            len(train_full),
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed,
        )

        train_ds = Subset(train_full, tr_idx.tolist())
        val_ds = Subset(train_full, va_idx.tolist())
        test_ds = Subset(train_full, te_idx.tolist())

    elif val_path is None:
        tr_idx, va_idx = split_indices(len(train_full), val_frac, seed)

        train_ds = Subset(train_full, tr_idx.tolist())
        val_ds = Subset(train_full, va_idx.tolist())
        test_ds = make_dataset(test_path, drugid2idx, valid_set)

    else:
        train_ds = train_full
        val_ds = make_dataset(val_path, drugid2idx, valid_set)
        test_ds = make_dataset(test_path, drugid2idx, valid_set) if test_path else None

    train_loader = make_loader(train_ds, batch_size, True, num_workers)
    val_loader = make_loader(val_ds, batch_size, False, num_workers)
    test_loader = make_loader(test_ds, batch_size, False, num_workers) if test_ds is not None else None

    return train_loader, val_loader, test_loader


def build_loaders_binary_directed_cv(
    train_path: str,
    val_path: Optional[str],
    test_path: Optional[str],
    batch_size: int,
    valid_set: Optional[List[int]],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 33,
    num_workers: int = 0,
    use_cv: bool = True,
    cv_folds: int = 5,
    fold: int = 1,
):
    full_ds = make_dataset(train_path, drugid2idx, valid_set)
    train_ds = full_ds
    val_ds = make_dataset(val_path, drugid2idx, valid_set)
    test_ds = make_dataset(test_path, drugid2idx, valid_set) if test_path else None

    print(
        f"[file split] train={len(train_ds)} "
        f"val={len(val_ds)} "
        f"test={len(test_ds) if test_ds is not None else 0}"
    )

    train_loader = make_loader(train_ds, batch_size, True, num_workers)
    val_loader = make_loader(val_ds, batch_size, False, num_workers)
    test_loader = make_loader(test_ds, batch_size, False, num_workers) if test_ds is not None else None

    return train_loader, val_loader, test_loader