import argparse
import torch
import torch.nn.functional as F
import os
import re
import time
import warnings
from typing import Optional
import numpy as np
import pandas as pd
import torch.nn as nn
from torch_geometric.data import Data
from collections import OrderedDict

from src.pretrained_2D.model import TokenMAE
from src.pretrained_3D.egnn import EGNN

def build_tokenmae_args_from_cli():
    args = argparse.Namespace()

    args.trans_encoder_layer = 4
    args.trans_decoder_layer = 1
    args.custom_trans = True
    args.transformer_norm_input = True
    args.drop_mask_tokens = True
    args.nonpara_tokenizer = True
    args.gnn_token_layer = 1
    args.loss = "mse"
    args.gnn_type = "gin"
    args.decoder_input_norm = True
    args.eps = 0.5

    args.gnn_emb_dim = 300
    args.gnn_dropout = 0.0
    args.gnn_JK = "last"
    args.gnn_activation = "relu"
    args.decoder_jk = "last"
    args.loss_all_nodes = False
    args.zero_mask = False

    args.pe_type = "none"
    args.laplacian_norm = "none"
    args.max_freqs = 20
    args.eigvec_norm = "L2"
    args.raw_norm_type = "none"
    args.kernel_times = []
    args.kernel_times_func = "none"
    args.layers = 3
    args.post_layers = 2
    args.dim_pe = 28
    args.phi_hidden_dim = 32
    args.phi_out_dim = 32

    return args

def instantiate_tokenmae_2d(device="cuda"):
    args = build_tokenmae_args_from_cli()

    model = TokenMAE(
        gnn_encoder_layer=5,
        gnn_token_layer=args.gnn_token_layer,
        gnn_decoder_layer=3,
        gnn_emb_dim=args.gnn_emb_dim,
        nonpara_tokenizer=args.nonpara_tokenizer,
        gnn_JK=args.gnn_JK,
        gnn_dropout=args.gnn_dropout,
        gnn_type=args.gnn_type,

        d_model=128,
        trans_encoder_layer=args.trans_encoder_layer,
        trans_decoder_layer=args.trans_decoder_layer,
        nhead=4,
        dim_feedforward=512,
        transformer_dropout=0.0,
        transformer_activation=F.relu,
        transformer_norm_input=args.transformer_norm_input,
        custom_trans=args.custom_trans,
        drop_mask_tokens=args.drop_mask_tokens,

        use_trans_decoder=False,
        pe_type=args.pe_type,
        args=args,
    ).to(device)

    return model


def load_tokenmae_checkpoint(model, ckpt_path):
    obj = torch.load(ckpt_path, map_location="cpu")

    print("Checkpoint path:", ckpt_path)
    print("Loaded object type:", type(obj))
    if isinstance(obj, OrderedDict):
        sd = obj

    elif isinstance(obj, dict):
        if "state_dict" in obj:
            sd = obj["state_dict"]
        elif "model_state_dict" in obj:
            sd = obj["model_state_dict"]
        elif "model" in obj:
            sd = obj["model"]
        elif "model_state" in obj:
            sd = obj["model_state"]
        else:
            maybe_keys = list(obj.keys())
            if len(maybe_keys) > 0 and all(isinstance(k, str) for k in maybe_keys):
                sd = obj
            else:
                raise ValueError(f"Cannot find model state_dict keys in checkpoint. Available keys: {list(obj.keys())}")
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")

    print("First checkpoint keys:", list(sd.keys())[:10])

    incompat = model.load_state_dict(sd, strict=True)
    print("Missing keys:", incompat.missing_keys)
    print("Unexpected keys:", incompat.unexpected_keys)

    return model

def strip_prefix(sd, prefix: str):
    if not any(k.startswith(prefix) for k in sd.keys()):
        return sd
    out = OrderedDict()
    for k, v in sd.items():
        out[k[len(prefix):]] = v
    return out

def extract_state_dict(ckpt):
    """
    - OrderedDict (state_dict)
    - dict checkpoint training keys: epoch, optimizer, egnn/encoder...
    """
    if isinstance(ckpt, OrderedDict):
        return ckpt

    if isinstance(ckpt, dict):
        if "egnn" in ckpt and isinstance(ckpt["egnn"], (dict, OrderedDict)):
            return ckpt["egnn"]
        if "encoder" in ckpt and isinstance(ckpt["encoder"], (dict, OrderedDict)):
            return ckpt["encoder"]
        # fallback
        for k in ("state_dict", "model", "model_state_dict"):
            if k in ckpt and isinstance(ckpt[k], (dict, OrderedDict)):
                return ckpt[k]

    raise ValueError(f"Cannot find state_dict inside checkpoint. Top-level keys: {list(ckpt.keys())[:30]}")

def load_egnn_3d(ckpt_path, device, model):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # checkpoint training dict
    sd = extract_state_dict(ckpt)

    sd = strip_prefix(sd, "module.")
    sd = strip_prefix(sd, "egnn.")

    missing, unexpected = model.load_state_dict(sd, strict=True)
    print(f"[3D EGNN] strict=True loaded | missing={len(missing)} unexpected={len(unexpected)}")
    return model

def freeze(m):
    for p in m.parameters():
        p.requires_grad = False

def load_pretrained_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_2d = args.ckpt_2d
    ckpt_3d = args.ckpt_3d

    tokenmae_2d = instantiate_tokenmae_2d(device=device)
    tokenmae_2d = load_tokenmae_checkpoint(tokenmae_2d, ckpt_2d)

    egnn_3d = EGNN(
        in_node_nf=64,
        hidden_nf=128,
        out_node_nf=128,
        in_edge_nf=13,
        device=device,
        n_layers=4
    ).to(device)

    egnn_3d = load_egnn_3d(ckpt_3d, device, egnn_3d)

    # freeze(tokenmae_2d)
    # freeze(egnn_3d)

    return tokenmae_2d, egnn_3d

def unfreeze_tokenmae_transformer_blocks(tokenmae_2d: nn.Module, last_n: Optional[int] = None):
    layer_ids = []

    for name, _ in tokenmae_2d.named_parameters():
        match = re.match(r"encoder\.trans_enc\.transformer\.layers\.(\d+)\.", name)
        if match:
            layer_ids.append(int(match.group(1)))

    layer_ids = sorted(set(layer_ids))

    if not layer_ids:
        print("[warn] cannot find transformer layers under encoder.trans_enc.transformer.layers.*")
        return

    if last_n is None or int(last_n) <= 0:
        allow = set(layer_ids)
    else:
        allow = set(layer_ids[-int(last_n):])

    for name, p in tokenmae_2d.named_parameters():
        match = re.match(r"encoder\.trans_enc\.transformer\.layers\.(\d+)\.", name)
        if match and int(match.group(1)) in allow:
            p.requires_grad = True

def unfreeze_egnn_gcl3_and_out(egnn_3d: nn.Module):
    for name, p in egnn_3d.named_parameters():
        if name.startswith("gcl_3.") or name.startswith("embedding_out."):
            p.requires_grad = True

def freeze_all(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False

def apply_freeze_schedule(args, epoch: int, tokenmae_2d: nn.Module, egnn_3d: nn.Module):
    freeze_all(tokenmae_2d)
    freeze_all(egnn_3d)
    if epoch < args.freeze_epochs:
        return
    if args.unfreeze_2d:
        unfreeze_tokenmae_transformer_blocks(
            tokenmae_2d,
            last_n=args.unfreeze_2d_last_n,
        )
    if args.unfreeze_3d:
        unfreeze_egnn_gcl3_and_out(egnn_3d)

def load_drug2d_cache_from_data_pt(data_pt_path: str, id_map_csv_path: str):
    obj = torch.load(data_pt_path, map_location="cpu")

    if not (isinstance(obj, (tuple, list)) and len(obj) == 2):
        raise ValueError(f"Expected (data, slices) in {data_pt_path}, got {type(obj)}")

    data, slices = obj
    id_map = pd.read_csv(id_map_csv_path)
    idx_list = id_map["idx"].astype(int).tolist()

    if "x" not in slices:
        raise KeyError("slices has no 'x'. Cannot infer num_graphs.")

    num_graphs = len(slices["x"]) - 1

    if len(idx_list) != num_graphs:
        raise ValueError(f"id_map size {len(idx_list)} != num_graphs {num_graphs}")

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


