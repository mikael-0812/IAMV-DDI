from __future__ import annotations

import os
import argparse
import warnings
from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.model.log import save_multiclass_prediction_artifacts
from src.model.loss import make_ce_loss
from src.model.pretrain_loader import load_pretrained_model, apply_freeze_schedule
from src.model.utils import LRUEncCache, compute_metrics_multiclass, is_better
from src.model.dataloader import build_loaders_multiclass_from_file
from src.model.model_ddi import DDIHead
from src.model.models import DDIHead as AblationDDIHead
from src.model.encoder import _encode_unique_drugs_in_batch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_trainable(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def save_ckpt(path: str, **kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(kwargs, path)

class WeightedMeter:
    def __init__(self):
        self.sum = 0.0
        self.n = 0

    def update(self, val: float, n: int):
        self.sum += float(val) * int(n)
        self.n += int(n)

    def avg(self):
        return self.sum / max(1, self.n)

    def reset(self):
        self.sum = 0.0
        self.n = 0

def build_args(
    default_ablation_mode: str = "full",
    default_metric_best: str = "acc",
):
    p = argparse.ArgumentParser("DDI multiclass finetuning with 2D+3D encoders")

    # paths
    p.add_argument("--ckpt_2d", type=str, required=True)
    p.add_argument("--ckpt_3d", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="checkpoints_ddi")
    p.add_argument("--run_name", type=str, default="ddi_multiclass")
    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--val_path", type=str, default="")
    p.add_argument("--test_path", type=str, default="")

    # cache paths
    p.add_argument("--cache_2d_path", type=str, default="src/ckpt/cache_drug/data.pt")
    p.add_argument("--cache_3d_path", type=str, default="src/ckpt/cache_drug/drug3d_cache_final.pt")
    p.add_argument("--id_map_path", type=str, default="src/data/id_map.csv")

    # runtime
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--amp", action="store_true", default=False)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=500)

    # model
    p.add_argument("--num_labels", type=int, default=86)
    p.add_argument("--d_fuse", type=int, default=128)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--inter_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--ablation_mode",
        type=str,
        default=default_ablation_mode,
        choices=["full", "wo_2d", "wo_3d", "wo_intra_fusion", "wo_inter_fusion"],
    )

    # finetune
    p.add_argument("--freeze_epochs", type=int, default=200)
    p.add_argument("--unfreeze_2d", action="store_true", default=True)
    p.add_argument("--unfreeze_3d", action="store_true", default=True)
    p.add_argument("--unfreeze_2d_last_n", type=int, default=0)

    # LRU cache
    p.add_argument("--lru_size", type=int, default=4096)
    p.add_argument("--lru_device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--lru_dtype", type=str, default="fp32", choices=["fp16", "fp32"])
    p.add_argument("--use_final_cache", action="store_true", default=True)
    p.add_argument("--use_2d_token_cache", action="store_true", default=True)
    p.add_argument("--use_3d_prefix_cache", action="store_true", default=True)
    p.add_argument("--log_cache", action="store_true", default=False)

    # optimizer
    p.add_argument("--lr_enc2d", type=float, default=1e-5)
    p.add_argument("--lr_enc3d", type=float, default=1e-5)
    p.add_argument("--lr_head", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)

    # eval / checkpoint
    p.add_argument("--metric_best", type=str, default=default_metric_best,
        choices=[
            "acc",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "macro_auroc",
            "macro_aupr",
        ])
    p.add_argument("--patience", type=int, default=15)

    # prediction saving
    p.add_argument("--fold", type=int, default=-1)
    p.add_argument("--save_final_preds", action="store_true", default=True)
    p.add_argument("--save_embeddings", action="store_true", default=True)

    return p.parse_args()

def load_drug2d_cache_from_data_pt(data_pt_path: str, id_map_csv_path: str):
    obj = torch.load(data_pt_path, map_location="cpu")

    if not (isinstance(obj, (tuple, list)) and len(obj) == 2):
        raise ValueError(f"Expected (data, slices) in {data_pt_path}, got {type(obj)}")

    data, slices = obj
    id_map = pd.read_csv(id_map_csv_path)
    idx_list = id_map["idx"].astype(int).tolist()

    keys = list(data.keys()) if callable(getattr(data, "keys", None)) else list(data.keys)

    def get_item(pos: int) -> Data:
        out = Data()
        for key in keys:
            item = data[key]
            s = slices[key]
            start, end = int(s[pos]), int(s[pos + 1])
            if key == "edge_index":
                out[key] = item[:, start:end]
            else:
                out[key] = item[start:end]
        return out

    cache = {}

    for pos, did in enumerate(idx_list):
        data_i = get_item(pos)

        if getattr(data_i, "x", None) is None or data_i.x.size(0) < 1:
            continue

        cache[int(did)] = data_i

    return cache

def filter_drug3d_cache_processed(drug3d: dict, min_nodes: int = 0):
    kept = {}
    skipped = 0

    for k, v in drug3d.items():
        try:
            h64, x3, edges, eattr = v

            if x3 is None or x3.size(0) < min_nodes:
                skipped += 1
                continue

            kept[int(k)] = (h64, x3, edges, eattr)

        except Exception:
            skipped += 1

    print(f"[drug3d_cache] kept={len(kept)} skipped<{min_nodes}={skipped}")
    return kept

def make_optimizer(args, tokenmae_2d, egnn_3d, ddi_head):
    head_params = list(ddi_head.parameters())
    enc2d_params = list(tokenmae_2d.parameters())
    enc3d_params = list(egnn_3d.parameters())
    return torch.optim.AdamW(
        [
            {"params": head_params, "lr": args.lr_head, "weight_decay": args.weight_decay},
            {"params": enc2d_params, "lr": args.lr_enc2d, "weight_decay": args.weight_decay},
            {"params": enc3d_params, "lr": args.lr_enc3d, "weight_decay": args.weight_decay},
        ]
    )

def train_step(args, epoch: int, batch, tokenmae_2d, egnn_3d, ddi_head, drug2d, drug3d, device, ce_loss_fn, lru_cache=None,):
    h_idx, t_idx, r, y = batch
    h_idx = h_idx.to(device, non_blocking=True)
    t_idx = t_idx.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True).long().view(-1)

    batch_size = y.numel()

    H2_map, H3_map, z2, z3, cache_hits, t_enc = _encode_unique_drugs_in_batch(h_idx, t_idx, epoch=epoch, args=args, tokenmae_2d=tokenmae_2d, egnn_3d=egnn_3d, drug2d=drug2d, drug3d=drug3d, device=device, lru_cache=lru_cache)
    lam = 0.0
    L_cl = torch.zeros((), device=device)

    H2_H_list, H3_H_list, H2_T_list, H3_T_list = [], [], [], []

    for i in range(batch_size):
        h = int(h_idx[i].item())
        t = int(t_idx[i].item())

        H2_H_list.append(H2_map[h])
        H3_H_list.append(H3_map[h])
        H2_T_list.append(H2_map[t])
        H3_T_list.append(H3_map[t])

    logits, _, _ = ddi_head(H2_H_list, H3_H_list, H2_T_list, H3_T_list)
    L_task = ce_loss_fn(logits, y)
    loss = L_task + lam * L_cl
    return loss, L_task.detach(), L_cl.detach(), lam


@torch.no_grad()
def eval_epoch(loader, args, epoch, tokenmae_2d, egnn_3d, ddi_head, drug2d, drug3d, device, ce_loss_fn, use_amp: bool = False, lru_cache=None, save_pred_path: Optional[str] = None, split: str = "val", fold: int = -1, save_embeddings: bool = False,):
    tokenmae_2d.eval()
    egnn_3d.eval()
    ddi_head.eval()
    # proj2d.eval()
    # proj3d.eval()

    all_logits = []
    all_y = []
    all_h_idx = []
    all_t_idx = []
    all_vH = []
    all_vT = []

    task_meter = WeightedMeter()
    cl_meter = WeightedMeter()

    for batch in loader:
        h_idx, t_idx, r, y = batch

        h_idx = h_idx.to(device, non_blocking=True)
        t_idx = t_idx.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long().view(-1)

        batch_size = y.numel()

        with torch.cuda.amp.autocast(enabled=use_amp):
            H2_map, H3_map, z2, z3, _, _ = _encode_unique_drugs_in_batch(h_idx, t_idx, epoch=epoch, args=args, tokenmae_2d=tokenmae_2d, egnn_3d=egnn_3d, drug2d=drug2d, drug3d=drug3d, device=device, lru_cache=lru_cache,)
            H2_H_list, H3_H_list, H2_T_list, H3_T_list = [], [], [], []

            for i in range(batch_size):
                h = int(h_idx[i].item())
                t = int(t_idx[i].item())

                H2_H_list.append(H2_map[h])
                H3_H_list.append(H3_map[h])
                H2_T_list.append(H2_map[t])
                H3_T_list.append(H3_map[t])

            logits, vH, vT = ddi_head(H2_H_list, H3_H_list, H2_T_list, H3_T_list)
            L_task = ce_loss_fn(logits, y)

            L_cl = torch.zeros((), device=device)

        all_logits.append(logits.detach().float().cpu())
        all_y.append(y.detach().long().cpu())
        all_h_idx.append(h_idx.detach().long().cpu())
        all_t_idx.append(t_idx.detach().long().cpu())

        if save_embeddings:
            all_vH.append(vH.detach().float().cpu())
            all_vT.append(vT.detach().float().cpu())

        task_meter.update(L_task.item(), batch_size)
        cl_meter.update(L_cl.item(), batch_size)

    logits_all = torch.cat(all_logits, dim=0)
    y_all = torch.cat(all_y, dim=0)
    h_all = torch.cat(all_h_idx, dim=0)
    t_all = torch.cat(all_t_idx, dim=0)

    metrics = compute_metrics_multiclass(logits_all, y_all, num_classes=args.num_labels,)

    vH_all = torch.cat(all_vH, dim=0) if save_embeddings and all_vH else None
    vT_all = torch.cat(all_vT, dim=0) if save_embeddings and all_vT else None

    if save_pred_path is not None:
        save_multiclass_prediction_artifacts(
            save_path=save_pred_path,
            logits=logits_all.numpy(),
            y_true=y_all.numpy(),
            h_idx=h_all.numpy(),
            t_idx=t_all.numpy(),
            vH=vH_all.numpy() if vH_all is not None else None,
            vT=vT_all.numpy() if vT_all is not None else None,
            fold=fold,
            epoch=epoch,
            split=split,
            num_classes=args.num_labels,
            task_loss=task_meter.avg(),
            cl_loss=cl_meter.avg(),
        )

    out = {
        "task_loss": float(task_meter.avg()),
        "cl_loss": float(cl_meter.avg()),
        "acc": metrics["acc"],
        "paper_macro_acc": metrics["paper_macro_acc"],
        "balanced_acc": metrics["balanced_acc"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "macro_auroc": metrics["macro_auroc"],
        "macro_aupr": metrics["macro_aupr"],
        "n": metrics["n"],
    }

    return out

def main(
    default_ablation_mode: str = "full",
    default_metric_best: str = "acc",
):
    args = build_args(
        default_ablation_mode=default_ablation_mode,
        default_metric_best=default_metric_best,
    )
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"

    drug2d = load_drug2d_cache_from_data_pt(args.cache_2d_path, args.id_map_path)
    drug3d = torch.load(args.cache_3d_path, map_location="cpu", weights_only=False)
    drug3d = filter_drug3d_cache_processed(drug3d, min_nodes=0)
    #
    common = sorted(set(drug2d.keys()) & set(drug3d.keys()))
    # drug2d = {k: drug2d[k] for k in common}
    # drug3d = {k: drug3d[k] for k in common}
    #
    # print(f"[cache_intersection] common={len(common)}")

    if args.lru_size > 0:
        lru_cache = LRUEncCache(
            max_items=args.lru_size,
            store_on_cpu=(args.lru_device == "cpu"),
            cpu_dtype=torch.float32,
            pin_memory=True,
            non_blocking=True,
        )
    else:
        lru_cache = None

    # dataloader
    train_loader, val_loader, test_loader = build_loaders_multiclass_from_file(train_path=args.train_path, val_path=args.val_path if args.val_path else None, test_path=args.test_path if args.test_path else None, batch_size=args.batch_size, valid_set=common, val_frac=0.2, seed=args.seed, num_workers=args.num_workers,)

    # model
    tokenmae_2d, egnn_3d = load_pretrained_model(args)
    tokenmae_2d.to(device)
    egnn_3d.to(device)

    head_cls = DDIHead if args.ablation_mode == "full" else AblationDDIHead
    ddi_head = head_cls(
        num_rel=args.num_labels,
        d_fuse=args.d_fuse,
        nhead=args.nhead,
        inter_layers=args.inter_layers,
        dropout=args.dropout,
        mlp_hidden=256,
        **(
            {"ablation_mode": args.ablation_mode}
            if args.ablation_mode != "full"
            else {}
        ),
    ).to(device)

    ce_loss_fn = make_ce_loss(args, device, train_loader=train_loader)

    opt = make_optimizer(args, tokenmae_2d, egnn_3d, ddi_head)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.startswith("cuda")))
    best_score = None
    best_epoch = -1
    bad_epochs = 0

    best_path = os.path.join(args.save_dir, f"{args.run_name}_best.pt")

    for epoch in range(args.epochs):
        print(f"[epoch] {epoch}")
        apply_freeze_schedule(args, epoch, tokenmae_2d, egnn_3d)
        fully_frozen = epoch < args.freeze_epochs
        if fully_frozen:
            tokenmae_2d.eval()
            egnn_3d.eval()
        else:
            tokenmae_2d.train()
            egnn_3d.train()

        ddi_head.train()
        # proj2d.train()
        # proj3d.train()

        loss_meter = WeightedMeter()
        task_meter = WeightedMeter()
        cl_meter = WeightedMeter()

        for step, batch in enumerate(train_loader, start=1):
            opt.zero_grad(set_to_none=True)
            _, _, _, y_cpu = batch
            batch_size = int(y_cpu.size(0))

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                loss, L_task, L_cl, lam = train_step(args=args, epoch=epoch, batch=batch, tokenmae_2d=tokenmae_2d, egnn_3d=egnn_3d, ddi_head=ddi_head, drug2d=drug2d, drug3d=drug3d, device=device, ce_loss_fn=ce_loss_fn, lru_cache=lru_cache)
            scaler.scale(loss).backward()

            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(opt)

                torch.nn.utils.clip_grad_norm_(
                    list(ddi_head.parameters()) + list(tokenmae_2d.parameters()) + list(egnn_3d.parameters()), args.grad_clip,)
            scaler.step(opt)
            scaler.update()

            loss_meter.update(loss.item(), batch_size)
            task_meter.update(L_task.item(), batch_size)
            cl_meter.update(L_cl.item(), batch_size)

            if step % args.log_every == 0:
                print(
                    f"[train] step={step} "
                    f"loss={loss_meter.avg():.4f} "
                    f"task={task_meter.avg():.4f} "
                    f"cl={cl_meter.avg():.4f} "
                    f"lam={lam:.4f}"
                )

                loss_meter.reset()
                task_meter.reset()
                cl_meter.reset()

        # validation
        val_metrics = eval_epoch(loader=val_loader, args=args, epoch=epoch, tokenmae_2d=tokenmae_2d, egnn_3d=egnn_3d, ddi_head=ddi_head, drug2d=drug2d, drug3d=drug3d, device=device, ce_loss_fn=ce_loss_fn, use_amp=scaler.is_enabled(), lru_cache=lru_cache)

        print(f"[val {epoch + 1:03d}] "
            f"task={val_metrics['task_loss']:.4f} "
            f"acc={val_metrics['acc']:.4f} "
            f"balanced_acc={val_metrics['balanced_acc']:.4f} "
            f"macro_f1={val_metrics['macro_f1']:.4f} "
            f"macro_precision={val_metrics['macro_precision']:.4f} "
            f"macro_recall={val_metrics['macro_recall']:.4f} "
            f"macro_auroc={val_metrics['macro_auroc']:.4f} "
            f"macro_aupr={val_metrics['macro_aupr']:.4f} ")
        curr = float(val_metrics[args.metric_best])

        # test_metrics = eval_epoch(loader=test_loader, args=args, epoch=epoch, tokenmae_2d=tokenmae_2d, egnn_3d=egnn_3d, ddi_head=ddi_head, drug2d=drug2d, drug3d=drug3d, device=device, ce_loss_fn=ce_loss_fn, use_amp=scaler.is_enabled(), lru_cache=lru_cache)
        # print(
        #     f"[test {epoch + 1:03d}] "
        #     f"task={test_metrics['task_loss']:.4f} "
        #     f"acc={test_metrics['acc']:.4f} "
        #     f"balanced_acc={test_metrics['balanced_acc']:.4f} "
        #     f"macro_f1={test_metrics['macro_f1']:.4f} "
        #     f"macro_precision={test_metrics['macro_precision']:.4f} "
        #     f"macro_recall={test_metrics['macro_recall']:.4f} "
        #     f"macro_auroc={test_metrics['macro_auroc']:.4f} "
        #     f"macro_aupr={test_metrics['macro_aupr']:.4f} "
        # )
        
        # save best
        if is_better(curr, best_score):
            best_score = curr
            best_epoch = epoch
            bad_epochs = 0

            save_ckpt(best_path, epoch=epoch, args=vars(args), tokenmae_2d=tokenmae_2d.state_dict(), egnn_3d=egnn_3d.state_dict(), ddi_head=ddi_head.state_dict(), best_metric_name=args.metric_best, best_metric_value=best_score)
            print(f"[best] saved => {best_path} ({args.metric_best}={best_score:.6f})")
        else:
            bad_epochs += 1
            print(f"[early-stop] bad_epochs={bad_epochs}/{args.patience}, best_epoch={best_epoch + 1}")
            if bad_epochs >= args.patience:
                print(f"[early-stop] no improvement for {args.patience} epochs.")
                break
    if test_loader is not None and os.path.exists(best_path):
        print("[final] evaluating best checkpoint on test set")
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        tokenmae_2d.load_state_dict(ckpt["tokenmae_2d"])
        egnn_3d.load_state_dict(ckpt["egnn_3d"])
        ddi_head.load_state_dict(ckpt["ddi_head"])

        pred_path = None
        if args.save_final_preds:
            pred_dir = os.path.join(args.save_dir, "predictions", args.run_name)
            pred_path = os.path.join(pred_dir, f"fold{args.fold}_final_test_preds.npz")

        final_metrics = eval_epoch(
            loader=test_loader,
            args=args,
            epoch=int(ckpt["epoch"]),
            tokenmae_2d=tokenmae_2d,
            egnn_3d=egnn_3d,
            ddi_head=ddi_head,
            drug2d=drug2d,
            drug3d=drug3d,
            device=device,
            ce_loss_fn=ce_loss_fn,
            use_amp=False,
            lru_cache=lru_cache,
            save_pred_path=pred_path,
            split="final_test",
            fold=args.fold,
            save_embeddings=args.save_embeddings,
        )
        print(
            f"[final-test] acc={final_metrics['acc']:.4f} "
            f"balanced_acc={final_metrics['balanced_acc']:.4f} "
            f"macro_f1={final_metrics['macro_f1']:.4f} "
            f"macro_auroc={final_metrics['macro_auroc']:.4f} "
            f"macro_aupr={final_metrics['macro_aupr']:.4f}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
