from __future__ import annotations

import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.dataloader import build_loaders_binary_directed
from src.model.dataloader_zhang import build_loaders_binary_directed_cv
from src.model.encoder import _encode_unique_drugs_in_batch
from src.model.model_ddi import DDIHeadV2
from src.model.pretrain_loader import (
    apply_freeze_schedule,
    filter_drug3d_cache_processed,
    load_drug2d_cache_from_data_pt,
    load_pretrained_model,
)
from src.model.utils import LRUEncCache, kge_binary_metrics, sigmoid_np

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def count_trainable(module: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )

def save_ckpt(path: str, **kwargs) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(kwargs, path)

class WeightedMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.n = 0

    def update(self, value: float, n: int) -> None:
        self.sum += float(value) * int(n)
        self.n += int(n)

    def avg(self) -> float:
        return self.sum / max(1, self.n)

    def reset(self) -> None:
        self.sum = 0.0
        self.n = 0

def save_eval_predictions(save_dir: str, run_name: str, split: str, epoch: int, y_true: np.ndarray, y_score: np.ndarray, logits: np.ndarray, metrics: dict, fold: int = -1, save_each_epoch: bool = False) -> str:
    
    pred_dir = os.path.join(save_dir, "predictions", run_name)
    os.makedirs(pred_dir, exist_ok=True)

    tag = f"fold{fold}_" if fold >= 0 else ""

    if save_each_epoch:
        filename = f"{tag}{split}_epoch{epoch + 1:03d}_preds.npz"
    else:
        filename = f"{tag}{split}_preds.npz"

    save_path = os.path.join(pred_dir, filename)

    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score).astype(float).reshape(-1)
    logits = np.asarray(logits).astype(float).reshape(-1)

    threshold = float(metrics.get("threshold", 0.5))
    y_pred = (y_score >= threshold).astype(int)

    payload = {
        "y_true": y_true,
        "y_score": y_score,
        "logits": logits,
        "y_pred": y_pred,
        "threshold": np.array(threshold),

        "acc": np.array(float(metrics.get("acc", np.nan))),
        "f1": np.array(float(metrics.get("f1", np.nan))),
        "precision": np.array(float(metrics.get("precision", np.nan))),
        "recall": np.array(float(metrics.get("recall", np.nan))),
        "auroc": np.array(float(metrics.get("auroc", np.nan))),
        "aupr": np.array(float(metrics.get("aupr", np.nan))),
        "pos_rate": np.array(float(metrics.get("pos_rate", np.nan))),
        "pred_pos_rate": np.array(float(metrics.get("pred_pos_rate", np.nan))),
        "task_loss": np.array(float(metrics.get("task_loss", np.nan))),
    }

    # Optional embeddings for t-SNE / UMAP
    if "vH" in metrics:
        payload["vH"] = np.asarray(metrics["vH"]).astype(np.float32)

    if "vT" in metrics:
        payload["vT"] = np.asarray(metrics["vT"]).astype(np.float32)

    if "pair_embeddings" in metrics:
        payload["pair_embeddings"] = np.asarray(metrics["pair_embeddings"]).astype(np.float32)

    np.savez(save_path, **payload)

    return save_path

def make_optimizer(args, tokenmae_2d, egnn_3d, ddi_head):
    return torch.optim.AdamW(
        [
            {"params": ddi_head.parameters(), "lr": args.lr_head},
            {"params": tokenmae_2d.parameters(), "lr": args.lr_enc2d},
            {"params": egnn_3d.parameters(), "lr": args.lr_enc3d},
        ],
        weight_decay=args.weight_decay,
    )

def build_token_lists(h_idx, t_idx, h2_map, h3_map):
    h2_h_list, h3_h_list, h2_t_list, h3_t_list = [], [], [], []
    for i in range(h_idx.numel()):
        h = int(h_idx[i].item())
        t = int(t_idx[i].item())
        h2_h_list.append(h2_map[h])
        h3_h_list.append(h3_map[h])
        h2_t_list.append(h2_map[t])
        h3_t_list.append(h3_map[t])
    return h2_h_list, h3_h_list, h2_t_list, h3_t_list


def forward_batch(args, epoch: int, batch, tokenmae_2d, egnn_3d, ddi_head, drug2d, drug3d, device, lru_cache=None):
    h_idx, t_idx, r, y = batch
    h_idx = h_idx.to(device, non_blocking=True)
    t_idx = t_idx.to(device, non_blocking=True)
    r = r.to(device, non_blocking=True).long()
    y = y.to(device, non_blocking=True).float().view(-1)

    enc2d_trainable = any(p.requires_grad for p in tokenmae_2d.parameters())
    enc3d_trainable = any(p.requires_grad for p in egnn_3d.parameters())
    enc_trainable = enc2d_trainable or enc3d_trainable

    if enc_trainable:
        h2_map, h3_map, _, _, cache_hits, enc_time = _encode_unique_drugs_in_batch(h_idx, t_idx, epoch=epoch, args=args, tokenmae_2d=tokenmae_2d, egnn_3d=egnn_3d, drug2d=drug2d, drug3d=drug3d, device=device, lru_cache=lru_cache)
    else:
        with torch.no_grad():
            h2_map, h3_map, _, _, cache_hits, enc_time = _encode_unique_drugs_in_batch(h_idx, t_idx, epoch=epoch, args=args, tokenmae_2d=tokenmae_2d, egnn_3d=egnn_3d, drug2d=drug2d, drug3d=drug3d, device=device, lru_cache=lru_cache)

        h2_map = {k: v.detach() for k, v in h2_map.items()}
        h3_map = {k: v.detach() for k, v in h3_map.items()}
    h2_h_list, h3_h_list, h2_t_list, h3_t_list = build_token_lists(h_idx, t_idx, h2_map, h3_map)

    head_start = time.time()
    logits, vH, vT = ddi_head(h2_h_list, h3_h_list, h2_t_list, h3_t_list, r)
    head_time = time.time() - head_start
    logits = logits.view(-1)

    loss = F.binary_cross_entropy_with_logits(logits, y)

    if args.log_cache:
        print(
            f"enc={enc_time:.3f}s head={head_time:.3f}s "
            f"cache_hit={cache_hits}/{len(h2_map)} "
            f"cache_size={len(lru_cache) if lru_cache is not None else 0}"
        )

    emb_dict = {
        "vH": vH.detach(),
        "vT": vT.detach(),
    }

    pair_embeddings = torch.cat(
        [
            vH,
            vT,
            torch.abs(vH - vT),
            vH * vT,
        ],
        dim=1,
    )

    emb_dict["pair_embeddings"] = pair_embeddings.detach()

    return loss, logits, y, emb_dict


@torch.no_grad()
def evaluate(loader, args, epoch: int, tokenmae_2d, egnn_3d, ddi_head, drug2d, drug3d, device, use_amp: bool = False, lru_cache=None):
    tokenmae_2d.eval()
    egnn_3d.eval()
    ddi_head.eval()

    all_logits, all_y = [], []
    loss_meter = WeightedMeter()

    all_vH = []
    all_vT = []
    all_pair_embeddings = []

    for batch in loader:
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss, logits, y, emb_dict = forward_batch(
                args=args,
                epoch=epoch,
                batch=batch,
                tokenmae_2d=tokenmae_2d,
                egnn_3d=egnn_3d,
                ddi_head=ddi_head,
                drug2d=drug2d,
                drug3d=drug3d,
                device=device,
                lru_cache=lru_cache,
            )

        bs = int(y.numel())
        loss_meter.update(loss.item(), bs)

        all_logits.append(logits.detach().float().cpu())
        all_y.append(y.detach().long().cpu())

        if "vH" in emb_dict:
            all_vH.append(emb_dict["vH"].detach().float().cpu())

        if "vT" in emb_dict:
            all_vT.append(emb_dict["vT"].detach().float().cpu())

        if "pair_embeddings" in emb_dict:
            all_pair_embeddings.append(
                emb_dict["pair_embeddings"].detach().float().cpu()
            )

    logits_np = torch.cat(all_logits, dim=0).numpy()
    y_np = torch.cat(all_y, dim=0).numpy()
    probs_np = sigmoid_np(logits_np)

    metrics = kge_binary_metrics(probs_np, y_np, thr=args.thr)
    metrics["task_loss"] = float(loss_meter.avg())
    metrics["threshold"] = float(args.thr)
    metrics["y_true"] = y_np
    metrics["y_score"] = probs_np
    metrics["logits"] = logits_np

    # Save embeddings for t-SNE / UMAP
    if len(all_vH) > 0:
        metrics["vH"] = torch.cat(all_vH, dim=0).numpy().astype(np.float32)

    if len(all_vT) > 0:
        metrics["vT"] = torch.cat(all_vT, dim=0).numpy().astype(np.float32)

    if len(all_pair_embeddings) > 0:
        metrics["pair_embeddings"] = (
            torch.cat(all_pair_embeddings, dim=0)
            .numpy()
            .astype(np.float32)
        )

    return metrics


def print_metrics(prefix: str, metrics_dict: dict) -> None:
    print(
        f"{prefix} "
        f"task={metrics_dict['task_loss']:.4f} "
        f"acc={metrics_dict['acc']:.4f} "
        f"f1={metrics_dict['f1']:.4f} "
        f"precision={metrics_dict['precision']:.4f} "
        f"recall={metrics_dict['recall']:.4f} "
        f"auroc={metrics_dict['auroc']:.4f} "
        f"aupr={metrics_dict['aupr']:.4f} "
        f"pos={metrics_dict['pos_rate']:.4f} "
        f"pred_pos={metrics_dict['pred_pos_rate']:.4f} "
        f"thr={metrics_dict.get('threshold', 0.5):.2f} "
        f"n={len(metrics_dict['y_true']) if 'y_true' in metrics_dict else -1}"
    )

def metric_value(metrics_dict: dict, key: str) -> float:
    value = float(metrics_dict[key])
    return value if math.isfinite(value) else -float("inf")

def build_args():
    dataset_parser = argparse.ArgumentParser(add_help=False)
    dataset_parser.add_argument("--dataset", choices=["drugbank", "zhang"], default="drugbank")
    known_args, _ = dataset_parser.parse_known_args()
    dataset = known_args.dataset
    is_zhang = dataset == "zhang"
    parser = argparse.ArgumentParser("Binary DDI finetuning with 2D+3D encoders", parents=[dataset_parser])

    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, default="")
    parser.add_argument("--test_path", type=str, default="")
    parser.add_argument("--id_map_path", type=str, required=True)
    parser.add_argument("--cache_2d_path", type=str)
    parser.add_argument("--cache_3d_path", type=str, required=True)

    # Pretrained checkpoints
    parser.add_argument("--ckpt_2d", type=str, required=True)
    parser.add_argument("--ckpt_3d", type=str, required=True)

    # Runtime
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=500)

    # Model
    parser.add_argument("--num_labels", type=int, default=1 if is_zhang else 86)
    parser.add_argument("--d_fuse", type=int, default=128)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--intra_layers", type=int, default=1)
    parser.add_argument("--inter_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Fine-tuning schedule
    parser.add_argument("--freeze_epochs", type=int, default=200)
    parser.add_argument("--unfreeze_2d", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--unfreeze_3d", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--unfreeze_2d_last_n", type=int, default=0)

    # Cache
    parser.add_argument("--lru_size", type=int, default=4096)
    parser.add_argument("--lru_device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--use_final_cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_2d_token_cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_3d_prefix_cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_cache", action="store_true", default=False)

    # Optimizer
    parser.add_argument("--lr_enc2d", type=float, default=1e-5)
    parser.add_argument("--lr_enc3d", type=float, default=1e-5)
    parser.add_argument("--lr_head", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    # Evaluation / checkpoint
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--metric_best", type=str, default="acc", choices=["aupr", "auroc", "f1", "acc", "precision", "recall"])
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="checkpoints_ddi")
    parser.add_argument("--run_name", type=str, default="binary_ddi")

    # Prediction logging for ROC/PR curves
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--save_preds", dest="save_preds", action="store_true", default=True)
    parser.add_argument("--no_save_preds", dest="save_preds", action="store_false")
    parser.add_argument("--save_preds_each_epoch", action="store_true", default=False)

    return parser.parse_args()


def main() -> None:
    args = build_args()
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    use_amp = bool(args.amp and device.startswith("cuda"))
    print(f"Device: {device}")


    drug2d = load_drug2d_cache_from_data_pt(args.cache_2d_path, args.id_map_path)
    drug3d = torch.load(args.cache_3d_path, map_location="cpu", weights_only=False)
    drug3d = filter_drug3d_cache_processed(drug3d, min_nodes=0)

    common = sorted(set(drug2d.keys()) & set(drug3d.keys()))
    drug2d = {k: drug2d[k] for k in common}
    drug3d = {k: drug3d[k] for k in common}
    print(f"[cache_intersection] common={len(common)}")

    lru_cache = None
    if args.lru_size > 0:
        lru_cache = LRUEncCache(max_items=args.lru_size, store_on_cpu=(args.lru_device == "cpu"), cpu_dtype=torch.float32, pin_memory=True, non_blocking=True)

    loader_builder = (
        build_loaders_binary_directed_cv
        if args.dataset == "zhang"
        else build_loaders_binary_directed
    )
    train_loader, val_loader, test_loader = loader_builder(train_path=args.train_path, val_path=args.val_path if args.val_path else None, test_path=args.test_path, valid_set=set(common), batch_size=args.batch_size, seed=args.seed, num_workers=args.num_workers)

    tokenmae_2d, egnn_3d = load_pretrained_model(args)
    tokenmae_2d.to(device)
    egnn_3d.to(device)

    ddi_head = DDIHeadV2(num_rel=args.num_labels, d_fuse=args.d_fuse, K=args.K, nhead=args.nhead, intra_layers=args.intra_layers, inter_layers=args.inter_layers, dropout=args.dropout).to(device)

    optimizer = make_optimizer(args, tokenmae_2d, egnn_3d, ddi_head)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_score = -float("inf")
    best_epoch = -1
    bad_epochs = 0
    global_step = 0
    best_path = os.path.join(args.save_dir, f"{args.run_name}_best.pt")

    for epoch in range(args.epochs):
        print("Epoch:", epoch)
        apply_freeze_schedule(args, epoch, tokenmae_2d, egnn_3d)
        encoders_frozen = epoch < args.freeze_epochs

        tokenmae_2d.eval() if encoders_frozen else tokenmae_2d.train()
        egnn_3d.eval() if encoders_frozen else egnn_3d.train()
        ddi_head.train()

        loss_meter = WeightedMeter()
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            batch_size = int(batch[-1].size(0))

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, _, _, _ = forward_batch(args, epoch, batch, tokenmae_2d, egnn_3d, ddi_head, drug2d, drug3d, device, lru_cache=lru_cache)

            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(ddi_head.parameters())
                    + list(tokenmae_2d.parameters())
                    + list(egnn_3d.parameters()),
                    args.grad_clip,
                )
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), batch_size)
            global_step += 1

            if step % args.log_every == 0:
                print(
                    f"[train] step={global_step} "
                    f"loss={loss_meter.avg():.4f} "
                )
                loss_meter.reset()

        val_metrics = evaluate(val_loader, args, epoch, tokenmae_2d, egnn_3d, ddi_head, drug2d, drug3d, device, use_amp=use_amp, lru_cache=lru_cache)
        print_metrics(f"[val {epoch + 1:03d}]", val_metrics)
        if args.save_preds and args.save_preds_each_epoch:
            save_eval_predictions(save_dir=args.save_dir, run_name=args.run_name, split="val", epoch=epoch, y_true=val_metrics["y_true"], y_score=val_metrics["y_score"], logits=val_metrics["logits"], metrics=val_metrics, fold=args.fold, save_each_epoch=True)

        current_score = metric_value(val_metrics, args.metric_best)
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            bad_epochs = 0
            save_ckpt(best_path, epoch=epoch, args=vars(args), tokenmae_2d=tokenmae_2d.state_dict(), egnn_3d=egnn_3d.state_dict(),  ddi_head=ddi_head.state_dict(), best_metric_name=args.metric_best, best_metric_value=best_score)
            print(f"[best] saved => {best_path} ({args.metric_best}={best_score:.6f})")
            if args.save_preds:
                save_eval_predictions(save_dir=args.save_dir, run_name=args.run_name, split="best_val", epoch=epoch, y_true=val_metrics["y_true"], y_score=val_metrics["y_score"], logits=val_metrics["logits"], metrics=val_metrics, fold=args.fold, save_each_epoch=False)
        else:
            bad_epochs += 1
            print(f"[early-stop] bad_epochs={bad_epochs}/{args.patience}, best_epoch={best_epoch + 1}")
            if bad_epochs >= args.patience:
                break

    if test_loader is not None and os.path.exists(best_path):
        print("\n[final] evaluating best checkpoint on test set ...")
        ckpt = torch.load(best_path, map_location="cpu")
        tokenmae_2d.load_state_dict(ckpt["tokenmae_2d"])
        egnn_3d.load_state_dict(ckpt["egnn_3d"])
        ddi_head.load_state_dict(ckpt["ddi_head"])

        tokenmae_2d.to(device).eval()
        egnn_3d.to(device).eval()
        ddi_head.to(device).eval()

        final_metrics = evaluate(test_loader, args, int(ckpt["epoch"]), tokenmae_2d, egnn_3d, ddi_head, drug2d, drug3d, device, use_amp=False, lru_cache=lru_cache)
        print_metrics("[final-test]", final_metrics)
        if args.save_preds:
            save_eval_predictions(save_dir=args.save_dir, run_name=args.run_name, split="final_test", epoch=int(ckpt["epoch"]), y_true=final_metrics["y_true"], y_score=final_metrics["y_score"], logits=final_metrics["logits"], metrics=final_metrics, fold=args.fold, save_each_epoch=False)

    print("Done.")

if __name__ == "__main__":
    main()
