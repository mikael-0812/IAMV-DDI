from __future__ import annotations

import csv
import random
from typing import List, Tuple, Optional
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Subset
from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader, Subset

# idx for drug
idx_map = pd.read_csv('data/DrugBank/id_map.csv')
drugid2idx = dict(zip(idx_map["drug_id"], idx_map["idx"]))

def filter_df_by_valid_set_directed(df, valid_set, col_h="h", col_t="t", col_neg="neg_idx"):
    if valid_set is None:
        return df
    valid = set(int(x) for x in valid_set)
    mask = df[col_h].isin(valid) & df[col_t].isin(valid) & df[col_neg].isin(valid)
    return df[mask].reset_index(drop=True)

class DDIDataset(Dataset):
    """
    Each row in CSV provides:
      d1, d2, type, Neg samples (e.g. 'DB01022$t' or 'DB00188$h')

    __getitem__ returns ints:
      h, t, r, neg, ntype_is_h
    """
    def __init__(
        self,
        path: str,
        drugid2idx: dict,
        valid_set: Optional[List[int]] = None,
        delimiter: str = ",",
        has_header: bool = True,

        col_h: str = "d1",
        col_t: str = "d2",
        col_r: str = "type",
        col_neg: str = "split",
        strict: bool = True,
    ):
        if has_header:
            df = pd.read_csv(path, sep=delimiter)
        else:
            df = pd.read_csv(path, sep=delimiter, header=None, names=[col_h, col_t, col_r, col_neg])

        def parse_neg(x: str):
            if not isinstance(x, str) or "$" not in x:
                return None, None
            neg_id, ntype = x.split("$", 1)
            ntype = ntype.strip().lower()
            if ntype not in ("h", "t"):
                return None, None
            return neg_id.strip(), ntype

        neg_parsed = df[col_neg].apply(parse_neg)
        df["neg_id"] = neg_parsed.apply(lambda z: z[0])
        df["ntype"] = neg_parsed.apply(lambda z: z[1])

        # Map IDs -> indices
        df[col_h] = df[col_h].map(drugid2idx)
        df[col_t] = df[col_t].map(drugid2idx)
        df["neg_idx"] = df["neg_id"].map(drugid2idx)

        # Drop invalid
        before = len(df)
        df = df.dropna(subset=[col_h, col_t, col_r, "neg_idx", "ntype"])
        # sau khi map indices + dropna
        if valid_set is not None:
            valid = set(int(x) for x in valid_set)
            df = df[df[col_h].isin(valid) & df[col_t].isin(valid) & df["neg_idx"].isin(valid)].reset_index(drop=True)

        after = len(df)
        if strict and after < before:
            print(f"[warn] dropped {before-after} rows due to missing mapping/neg parse")

        self.h = df[col_h].astype(int).to_numpy(np.int64)
        self.t = df[col_t].astype(int).to_numpy(np.int64)
        self.r = df[col_r].astype(int).to_numpy(np.int64)
        self.neg = df["neg_idx"].astype(int).to_numpy(np.int64)
        self.ntype_is_h = (df["ntype"].astype(str).str.lower() == "h").to_numpy(np.bool_)

    def __len__(self):
        return len(self.h)

    def __getitem__(self, i: int):
        return (
            int(self.h[i]),
            int(self.t[i]),
            int(self.r[i]),
            int(self.neg[i]),
            bool(self.ntype_is_h[i]),
        )


def collate_ddi_binary_directed_file(batch):
    # batch item: (h, t, r, neg, ntype_is_h)
    h_list, t_list, r_list, y_list = [], [], [], []

    for (h, t, r, neg, ntype_is_h) in batch:
        # if h not in valid_set or t not in valid_set or neg not in valid_set:
        #     continue

        # positive
        h_list.append(h); t_list.append(t); r_list.append(r); y_list.append(1.0)

        # negative: corrupt head or tail, keep relation r
        if ntype_is_h:
            h_list.append(neg); t_list.append(t)
        else:
            h_list.append(h); t_list.append(neg)
        r_list.append(r); y_list.append(0.0)

    h = torch.tensor(h_list, dtype=torch.long)
    t = torch.tensor(t_list, dtype=torch.long)
    r = torch.tensor(r_list, dtype=torch.long)
    y = torch.tensor(y_list, dtype=torch.float32)
    return h, t, r, y



def split_indices(n: int, val_frac: float, seed: int):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(n * val_frac))
    return idx[n_val:], idx[:n_val]

def build_loaders_binary_directed(
    train_path: str,
    val_path: str,
    test_path: str,
    batch_size: int,
    valid_set: None,
    val_frac: float = 0.1,
    seed: int = 33,
    num_workers: int = 0,
):
    train_full = DDIDataset(train_path, drugid2idx, valid_set)

    if val_path is None:
        tr_idx, va_idx = split_indices(len(train_full), val_frac, seed)

        train_ds = Subset(train_full, tr_idx.tolist())
        val_ds = Subset(train_full, va_idx.tolist())

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            collate_fn=collate_ddi_binary_directed_file
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            collate_fn=collate_ddi_binary_directed_file
        )

    else:
        train_loader = DataLoader(
            train_full, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            collate_fn=collate_ddi_binary_directed_file
        )

        val_ds = DDIDataset(val_path, drugid2idx, valid_set)
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            collate_fn=collate_ddi_binary_directed_file
        )

    test_loader = None
    if test_path:
        test_ds = DDIDataset(test_path, drugid2idx, valid_set)
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            collate_fn=collate_ddi_binary_directed_file
        )
    return train_loader, val_loader, test_loader


class DDIDatasetMultiClass(Dataset):
    """
    Multi-class DDIE dataset.

    Each CSV row provides at least:
      d1, d2, type

    __getitem__ returns ints:
      h, t, r, y

    where:
      - h: head drug index
      - t: tail drug index
      - r: relation/event id (same as y, kept for compatibility)
      - y: class id in [0 .. num_rel-1]
    """
    def __init__(
        self,
        path: str,
        drugid2idx: dict,
        valid_set: Optional[List[int]] = None,
        delimiter: str = ",",
        has_header: bool = True,
        col_h: str = "d1",
        col_t: str = "d2",
        col_r: str = "type",
        strict: bool = True,
    ):
        if has_header:
            df = pd.read_csv(path, sep=delimiter)
        else:
            df = pd.read_csv(path, sep=delimiter, header=None, names=[col_h, col_t, col_r])

        # Map IDs -> indices
        df[col_h] = df[col_h].map(drugid2idx)
        df[col_t] = df[col_t].map(drugid2idx)

        before = len(df)

        # Drop invalid rows
        df = df.dropna(subset=[col_h, col_t, col_r])

        # Filter by valid_set if provided
        if valid_set is not None:
            valid = set(int(x) for x in valid_set)
            df = df[df[col_h].isin(valid) & df[col_t].isin(valid)].reset_index(drop=True)

        after = len(df)
        if strict and after < before:
            print(f"[warn] dropped {before - after} rows due to missing mapping/filtering")

        self.h = df[col_h].astype(int).to_numpy(np.int64)
        self.t = df[col_t].astype(int).to_numpy(np.int64)
        self.r = df[col_r].astype(int).to_numpy(np.int64)
        self.y = df[col_r].astype(int).to_numpy(np.int64)   # multi-class target

    def __len__(self):
        return len(self.h)

    def __getitem__(self, i: int):
        return (
            int(self.h[i]),
            int(self.t[i]),
            int(self.r[i]),
            int(self.y[i]),
        )


def collate_ddi_multiclass_file(batch):
    """
    batch item: (h, t, r, y)
    return:
      h: LongTensor [B]
      t: LongTensor [B]
      r: LongTensor [B]
      y: LongTensor [B]
    """
    h_list, t_list, r_list, y_list = [], [], [], []

    for (h, t, r, y) in batch:
        h_list.append(h)
        t_list.append(t)
        r_list.append(r)
        y_list.append(y)

    h = torch.tensor(h_list, dtype=torch.long)
    t = torch.tensor(t_list, dtype=torch.long)
    r = torch.tensor(r_list, dtype=torch.long)
    y = torch.tensor(y_list, dtype=torch.long)
    return h, t, r, y


def split_indices(n: int, val_frac: float, seed: int):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(n * val_frac))
    return idx[n_val:], idx[:n_val]


def build_loaders_multiclass_from_file(
    train_path: str,
    val_path: Optional[str],
    test_path: Optional[str],
    batch_size: int,
    valid_set: Optional[List[int]] = None,
    val_frac: float = 0.1,
    seed: int = 33,
    num_workers: int = 0,
    delimiter: str = ",",
    has_header: bool = True,
    col_h: str = "d1",
    col_t: str = "d2",
    col_r: str = "type",
):
    train_full = DDIDatasetMultiClass(
        path=train_path,
        drugid2idx=drugid2idx,
        valid_set=valid_set,
        delimiter=delimiter,
        has_header=has_header,
        col_h=col_h,
        col_t=col_t,
        col_r=col_r,
    )

    if val_path is None:
        tr_idx, va_idx = split_indices(len(train_full), val_frac, seed)

        train_ds = Subset(train_full, tr_idx.tolist())
        val_ds = Subset(train_full, va_idx.tolist())

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_ddi_multiclass_file,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_ddi_multiclass_file,
        )
    else:
        train_loader = DataLoader(
            train_full,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_ddi_multiclass_file,
        )

        val_ds = DDIDatasetMultiClass(
            path=val_path,
            drugid2idx=drugid2idx,
            valid_set=valid_set,
            delimiter=delimiter,
            has_header=has_header,
            col_h=col_h,
            col_t=col_t,
            col_r=col_r,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_ddi_multiclass_file,
        )

    test_loader = None
    if test_path:
        test_ds = DDIDatasetMultiClass(
            path=test_path,
            drugid2idx=drugid2idx,
            valid_set=valid_set,
            delimiter=delimiter,
            has_header=has_header,
            col_h=col_h,
            col_t=col_t,
            col_r=col_r,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_ddi_multiclass_file,
        )

    return train_loader, val_loader, test_loader