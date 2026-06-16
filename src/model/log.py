import os

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, roc_auc_score

from src.model.utils import compute_metrics_multiclass


def save_multiclass_prediction_artifacts(
    save_path: str,
    logits,
    y_true,
    h_idx=None,
    t_idx=None,
    vH=None,
    vT=None,
    fold: int = -1,
    epoch: int = -1,
    split: str = "final_test",
    num_classes: int = 86,
    task_loss: float = np.nan,
    cl_loss: float = np.nan,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    m = compute_metrics_multiclass(
        logits=logits,
        target=y_true,
        num_classes=num_classes,
    )

    payload = {"fold": np.array(fold), "epoch": np.array(epoch), "split": np.array(split), "y_true": m["y_true"],
               "y_pred": m["y_pred"], "y_prob": m["y_prob"], "logits": m["logits"], "acc": np.array(m["acc"]),
               "macro_precision": np.array(m["macro_precision"]), "macro_recall": np.array(m["macro_recall"]),
               "macro_f1": np.array(m["macro_f1"]), "paper_macro_acc": np.array(m["paper_macro_acc"]),
               "balanced_acc": np.array(m["balanced_acc"]), "balanced_acc_sklearn": np.array(m["balanced_acc_sklearn"]),
               "macro_auroc": np.array(m["macro_auroc"]), "macro_aupr": np.array(m["macro_aupr"]),
               "task_loss": np.array(task_loss), "cl_loss": np.array(cl_loss), "macro_acc": np.array(m["acc"])}

    if h_idx is not None:
        payload["h_idx"] = np.asarray(h_idx).astype(np.int64)

    if t_idx is not None:
        payload["t_idx"] = np.asarray(t_idx).astype(np.int64)

    if vH is not None:
        payload["vH"] = np.asarray(vH).astype(np.float32)

    if vT is not None:
        payload["vT"] = np.asarray(vT).astype(np.float32)

    if vH is not None and vT is not None:
        vH_np = np.asarray(vH).astype(np.float32)
        vT_np = np.asarray(vT).astype(np.float32)

        pair_embeddings = np.concatenate(
            [
                vH_np,
                vT_np,
                np.abs(vH_np - vT_np),
                vH_np * vT_np,
            ],
            axis=1,
        )

        payload["pair_embeddings"] = pair_embeddings

    np.savez(save_path, **payload)

    # ============================================================
    # Per-class metrics for radar chart
    # ============================================================
    labels = np.arange(num_classes)

    precision, recall, f1, support = precision_recall_fscore_support(
        m["y_true"],
        m["y_pred"],
        labels=labels,
        zero_division=0,
    )

    rows = []

    for c in labels:
        y_true_c = (m["y_true"] == c)
        y_pred_c = (m["y_pred"] == c)

        tp = int(np.sum(y_true_c & y_pred_c))
        tn = int(np.sum(~y_true_c & ~y_pred_c))
        fp = int(np.sum(~y_true_c & y_pred_c))
        fn = int(np.sum(y_true_c & ~y_pred_c))

        denom_paper = tp + tn + fp + fn
        paper_acc_c = (tp + tn) / denom_paper if denom_paper > 0 else np.nan

        denom_balanced = tp + fn
        class_acc_c = tp / denom_balanced if denom_balanced > 0 else np.nan

        y_true_bin = y_true_c.astype(int)
        y_score_c = m["y_prob"][:, c]

        if y_true_bin.sum() > 0 and y_true_bin.sum() < len(y_true_bin):
            auc_c = roc_auc_score(y_true_bin, y_score_c)
            aupr_c = average_precision_score(y_true_bin, y_score_c)
        else:
            auc_c = np.nan
            aupr_c = np.nan

        rows.append({
            "fold": fold,
            "epoch": epoch,
            "split": split,
            "class_id": int(c),

            # confusion values
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "support": int(support[c]),

            # per-class classification metrics
            "precision": float(precision[c]),
            "recall": float(recall[c]),
            "f1": float(f1[c]),

            # two accuracy variants for radar
            "paper_acc": float(paper_acc_c) if not np.isnan(paper_acc_c) else np.nan,
            "class_acc": float(class_acc_c) if not np.isnan(class_acc_c) else np.nan,

            # ranking metrics
            "auc": float(auc_c) if not np.isnan(auc_c) else np.nan,
            "aupr": float(aupr_c) if not np.isnan(aupr_c) else np.nan,
        })

    per_class_path = save_path.replace(".npz", "_per_class_metrics.csv")
    pd.DataFrame(rows).to_csv(per_class_path, index=False)

    print(
        f"[save-preds] split={split} epoch={epoch + 1:03d} "
        f"path={save_path} n={m['n']}"
    )
    print(f"[save-per-class] path={per_class_path}")

    print(
        f"[saved-metrics] "
        f"acc={m['acc']:.4f} "
        f"paper_macro_acc={m['paper_macro_acc']:.4f} "
        f"balanced_acc={m['balanced_acc']:.4f} "
        f"macro_precision={m['macro_precision']:.4f} "
        f"macro_recall={m['macro_recall']:.4f} "
        f"macro_f1={m['macro_f1']:.4f} "
        f"macro_auroc={m['macro_auroc']:.4f} "
        f"macro_aupr={m['macro_aupr']:.4f}"
    )

    return m