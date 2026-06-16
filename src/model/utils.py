# src/model/utils.py
from __future__ import annotations
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, Union
import torch
import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, accuracy_score, recall_score, \
    precision_score, balanced_accuracy_score
from sklearn.preprocessing import label_binarize
from torch_geometric.data import Batch

TensorOrTensors = Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]

def is_better(curr: float, best: Optional[float]):
    if np.isnan(curr):
        return False
    return best is None or curr > best

def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-logits))

def _to_cpu_detached(
    x,
    float_dtype: Optional[torch.dtype] = None,
    pin: bool = False,
    make_contiguous: bool = False,
):
    if x is None:
        return None

    if torch.is_tensor(x):
        t = x.detach()
        if float_dtype is not None and t.is_floating_point():
            t = t.to(dtype=float_dtype)

        t = t.to("cpu")
        if make_contiguous:
            t = t.contiguous()
        if pin:
            t = t.pin_memory()

        return t

    if isinstance(x, (tuple, list)):
        return type(x)(_to_cpu_detached(t, float_dtype=float_dtype, pin=pin, make_contiguous=make_contiguous) for t in x)

    if isinstance(x, dict):
        return {k: _to_cpu_detached(v, float_dtype=float_dtype, pin=pin, make_contiguous=make_contiguous) for k, v in x.items()}

    raise TypeError(type(x))

def _to_device(x, device, non_blocking: bool = True):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.to(device, non_blocking=non_blocking)
    if isinstance(x, (tuple, list)):
        return type(x)(_to_device(t, device, non_blocking=non_blocking) for t in x)
    if isinstance(x, dict):
        return {k: _to_device(v, device, non_blocking=non_blocking) for k, v in x.items()}
    raise TypeError(type(x))

class LRUEncCache:
    def __init__(self, max_items=2048, store_on_cpu=True, cpu_dtype=None, pin_memory=True, non_blocking=True):
        self.max_items = int(max_items)
        self.store_on_cpu = bool(store_on_cpu)
        self.cpu_dtype = cpu_dtype            # dùng như float_dtype
        self.pin_memory = bool(pin_memory)
        self.non_blocking = bool(non_blocking)
        self._data = OrderedDict()

    def get(self, key, device=None):
        if key not in self._data:
            return None
        val = self._data.pop(key)
        self._data[key] = val
        if device is None:
            return val
        return _to_device(val, device, non_blocking=self.non_blocking)

    def put(self, key, value):
        if key in self._data:
            self._data.pop(key)

        if self.store_on_cpu:
            # chỉ cast float tensors, giữ nguyên long/bool
            value = _to_cpu_detached(value, float_dtype=self.cpu_dtype, pin=self.pin_memory)

        self._data[key] = value
        while len(self._data) > self.max_items:
            self._data.popitem(last=False)

def kge_binary_metrics(probs: np.ndarray, y_true: np.ndarray, thr: float = 0.5):
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    y = np.asarray(y_true, dtype=np.int32).reshape(-1)

    pred = (probs >= thr).astype(np.int32)

    out = {
        "acc": float(metrics.accuracy_score(y, pred)),
        "recall": float(metrics.recall_score(y, pred, zero_division=0)),
        "precision": float(metrics.precision_score(y, pred, zero_division=0)),
        "f1": float(metrics.f1_score(y, pred, zero_division=0)),
        "thr": float(thr),
        "n": int(len(y)),
        "pos_rate": float(y.mean()) if len(y) else float("nan"),
        "pred_pos_rate": float(pred.mean()) if len(pred) else float("nan"),
        "tn_fp_fn_tp": tuple(int(x) for x in metrics.confusion_matrix(y, pred, labels=[0, 1]).ravel())
    }

    if len(np.unique(y)) == 2:
        out["auroc"] = float(metrics.roc_auc_score(y, probs))
        out["aupr"]  = float(metrics.average_precision_score(y, probs))
    else:
        out["auroc"] = float("nan")
        out["aupr"]  = float("nan")

    return out

def softmax_np(logits_np: np.ndarray) -> np.ndarray:
    logits_np = logits_np - logits_np.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_np)
    return exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), 1e-12, None)

def paper_macro_accuracy_ovr(y_true, y_pred, num_classes: int):
    """
    Original paper-style macro Acc:
      1/C * sum_i (TP_i + TN_i) / (TP_i + TN_i + FP_i + FN_i)
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    acc_list = []

    for c in range(num_classes):
        true_c = y_true == c
        pred_c = y_pred == c

        tp = np.sum(true_c & pred_c)
        tn = np.sum(~true_c & ~pred_c)
        fp = np.sum(~true_c & pred_c)
        fn = np.sum(true_c & ~pred_c)

        denom = tp + tn + fp + fn
        acc_c = (tp + tn) / denom if denom > 0 else np.nan
        acc_list.append(acc_c)

    return float(np.nanmean(acc_list))


def fair_macro_accuracy(y_true, y_pred, num_classes: int):
    """
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    acc_list = []

    for c in range(num_classes):
        true_c = y_true == c
        pred_c = y_pred == c

        tp = np.sum(true_c & pred_c)
        fn = np.sum(true_c & ~pred_c)

        denom = tp + fn
        acc_c = tp / denom if denom > 0 else np.nan
        acc_list.append(acc_c)

    return float(np.nanmean(acc_list))


def compute_metrics_multiclass(logits, target, num_classes: int):
    if isinstance(logits, torch.Tensor):
        logits_np = logits.detach().cpu().numpy()
    else:
        logits_np = np.asarray(logits)

    if isinstance(target, torch.Tensor):
        y_true = target.detach().cpu().numpy()
    else:
        y_true = np.asarray(target)

    y_true = y_true.reshape(-1).astype(np.int64)
    y_pred = logits_np.argmax(axis=1).reshape(-1)
    y_prob = softmax_np(logits_np)

    labels = np.arange(num_classes)

    overall_acc = accuracy_score(y_true, y_pred)

    paper_macro_acc = paper_macro_accuracy_ovr(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=num_classes,
    )

    balanced_acc = fair_macro_accuracy(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=num_classes,
    )

    balanced_acc_sklearn = balanced_accuracy_score(y_true, y_pred)

    macro_precision = precision_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    y_bin = label_binarize(y_true, classes=labels)

    auroc_list = []
    aupr_list = []

    for c in labels:
        y_true_c = y_bin[:, c]
        y_score_c = y_prob[:, c]

        if y_true_c.sum() > 0 and y_true_c.sum() < len(y_true_c):
            auroc_list.append(roc_auc_score(y_true_c, y_score_c))
            aupr_list.append(average_precision_score(y_true_c, y_score_c))

    macro_auroc = float(np.mean(auroc_list)) if len(auroc_list) > 0 else np.nan
    macro_aupr = float(np.mean(aupr_list)) if len(aupr_list) > 0 else np.nan

    return {
        "acc": float(overall_acc),
        "paper_macro_acc": float(paper_macro_acc),
        "balanced_acc": float(balanced_acc),
        "balanced_acc_sklearn": float(balanced_acc_sklearn),

        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "macro_auroc": macro_auroc,
        "macro_aupr": macro_aupr,

        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "logits": logits_np,
        "n": int(len(y_true)),
    }

