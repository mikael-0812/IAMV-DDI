import torch
import torch.nn as nn
import torch.nn.functional as F

def egnn_prefix_upto(egnn_3d, h64, x3, edges, eattr, upto_layer_exclusive: int):
    h = egnn_3d.embedding_in(h64)
    x = x3
    for i in range(0, upto_layer_exclusive):
        gcl = egnn_3d._modules[f"gcl_{i}"]
        h, x, _ = gcl(h, edges, x, edge_attr=eattr)
    return h, x

def egnn_suffix_from(egnn_3d, h, x, edges, eattr, start_layer: int):
    for i in range(start_layer, egnn_3d.n_layers):
        gcl = egnn_3d._modules[f"gcl_{i}"]
        h, x, _ = gcl(h, edges, x, edge_attr=eattr)
    h = egnn_3d.embedding_out(h)
    return h


def pool_mean(H, batch=None):
    if batch is None:
        return H.mean(dim=0, keepdim=True)

    num_graphs = int(batch.max().item()) + 1
    out = torch.zeros((num_graphs, H.size(-1)), device=H.device, dtype=H.dtype)
    out = out.index_add(0, batch, H)
    counts = torch.bincount(batch, minlength=num_graphs).clamp(min=1).unsqueeze(-1)
    return out / counts


def prepare_tokenmae_finetune_inputs(data):
    dev = data.x.device
    if not hasattr(data, "batch") or data.batch is None:
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=dev)
    data.mask_tokens = torch.zeros(data.x.size(0), dtype=torch.bool, device=dev)
    data.x_masked = data.x
    return data


def encode_2d_encoder128(tokenmae_2d, data):
    data = prepare_tokenmae_finetune_inputs(data)
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

    h0 = tokenmae_2d.tokenizer(x, edge_index, edge_attr)  # (N,300)

    pe = tokenmae_2d.pos_encoder(data)
    h = tokenmae_2d.encoder(tokenmae_2d.gnn_act(h0), edge_index, edge_attr, data.batch, data.mask_tokens, pe)
    return h  # (N,128)


def encode_3d_encoder128(egnn_3d, h64, x3, edges, edge_attr):
    H, _ = egnn_3d(h64, x3, edges, edge_attr)  # (N,128)
    return H

from typing import Dict, Optional, Tuple, List
import torch
import torch.nn.functional as F
import time


def _pool_tokens_mean(H: torch.Tensor) -> torch.Tensor:
    """
    H: (L, D) token-level embeddings for a single drug
    return: (1, D) pooled global embedding
    """
    return H.mean(dim=0, keepdim=True)


def _encode_one_drug(
    did: int,
    *,
    tokenmae_2d,
    egnn_3d,
    drug2d: Dict[int, torch.Tensor],
    drug3d: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    device: str,
    lru_cache: Optional["LRUEncCache"],
    use_final_cache: bool,
    use_2d_token_cache: bool,
    use_3d_prefix_cache: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """
    Returns:
      H2: (L2, D)
      H3: (L3, D)
      z2: (1, D) pooled
      z3: (1, D) pooled
      hit_final_cache: bool
    """
    # -- final cache: store (H2, H3) on CPU or CUDA depending on your LRU config
    if use_final_cache and lru_cache is not None:
        cached = lru_cache.get(("final", did), device=device)
        if cached is not None:
            H2, H3 = cached
            return H2, H3, _pool_tokens_mean(H2), _pool_tokens_mean(H3), True

    # 2D
    g2d = prepare_tokenmae_finetune_inputs(drug2d[did].to(device))

    if use_2d_token_cache and lru_cache is not None:
        tok = lru_cache.get(("2d_tok", did), device=device)
        if tok is None:
            x, edge_index, edge_attr = g2d.x, g2d.edge_index, g2d.edge_attr
            with torch.no_grad():
                h0 = tokenmae_2d.tokenizer(x, edge_index, edge_attr)
                pe = tokenmae_2d.pos_encoder(g2d)  # can be None
            h0 = h0.float()
            if pe is not None:
                pe = pe.float()
            lru_cache.put(("2d_tok", did), (h0, pe))
        else:
            h0, pe = tok

        H2 = tokenmae_2d.encoder(
            tokenmae_2d.gnn_act(h0),
            g2d.edge_index, g2d.edge_attr,
            g2d.batch, g2d.mask_tokens, pe
        )
    else:
        H2 = encode_2d_encoder128(tokenmae_2d, g2d)

    # ---------------------- 3D
    if use_3d_prefix_cache and lru_cache is not None:
        # store prefix on CPU to reuse when partial-unfreeze
        pref = lru_cache.get(("3d_pref", did), device=None)
        if pref is not None:
            h_pref_cpu, x_pref_cpu, edges_cpu, eattr_cpu = pref
            h_pref = h_pref_cpu.to(device, non_blocking=True)
            x_pref = x_pref_cpu.to(device, non_blocking=True)
            edges  = edges_cpu.to(device, non_blocking=True)
            eattr  = eattr_cpu.to(device, non_blocking=True)
            H3 = egnn_suffix_from(egnn_3d, h_pref, x_pref, edges, eattr, start_layer=3)
        else:
            h64_cpu, x3_cpu, edges_cpu, eattr_cpu = drug3d[did]
            h64   = h64_cpu.to(device, non_blocking=True)
            x3    = x3_cpu.to(device, non_blocking=True)
            edges = edges_cpu.to(device, non_blocking=True)
            eattr = eattr_cpu.to(device, non_blocking=True)

            with torch.no_grad():
                h_pref, x_pref = egnn_prefix_upto(
                    egnn_3d, h64, x3, edges, eattr, upto_layer_exclusive=3
                )
            h_pref_cpu = h_pref.detach().to("cpu")
            x_pref_cpu = x_pref.detach().to("cpu")
            lru_cache.put(("3d_pref", did), (h_pref_cpu, x_pref_cpu, edges_cpu, eattr_cpu))

            H3 = egnn_suffix_from(egnn_3d, h_pref, x_pref, edges, eattr, start_layer=3)
    else:
        h64_cpu, x3_cpu, edges_cpu, eattr_cpu = drug3d[did]
        H3 = encode_3d_encoder128(
            egnn_3d,
            h64_cpu.to(device, non_blocking=True),
            x3_cpu.to(device, non_blocking=True),
            edges_cpu.to(device, non_blocking=True),
            eattr_cpu.to(device, non_blocking=True),
        )

    # save final cache if enabled
    if use_final_cache and lru_cache is not None:
        lru_cache.put(("final", did), (H2, H3))

    return H2, H3, _pool_tokens_mean(H2), _pool_tokens_mean(H3), False


def _encode_unique_drugs_in_batch(
    h_idx: torch.Tensor,
    t_idx: torch.Tensor,
    *,
    epoch: int,
    args,
    tokenmae_2d,
    egnn_3d,
    drug2d,
    drug3d,
    device: str,
    lru_cache: Optional["LRUEncCache"],
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], torch.Tensor, torch.Tensor, int, float]:
    """
    Returns:
      H2_map: did -> (L2, D)
      H3_map: did -> (L3, D)
      z2: (m, D)
      z3: (m, D)
      cache_hits_final: int
      t_enc: float seconds
    """
    uniq = torch.unique(torch.cat([h_idx, t_idx], dim=0)).tolist()

    fully_frozen = epoch < args.freeze_epochs

    use_final_cache     = bool(args.use_final_cache and fully_frozen and lru_cache is not None)
    use_2d_token_cache  = bool(args.use_2d_token_cache and (not fully_frozen) and lru_cache is not None)
    use_3d_prefix_cache = False

    H2_map: Dict[int, torch.Tensor] = {}
    H3_map: Dict[int, torch.Tensor] = {}
    z2_list: List[torch.Tensor] = []
    z3_list: List[torch.Tensor] = []
    cache_hits = 0

    t0 = time.time()
    for did in uniq:
        did = int(did)
        H2, H3, z2, z3, hit = _encode_one_drug(
            did,
            tokenmae_2d=tokenmae_2d,
            egnn_3d=egnn_3d,
            drug2d=drug2d,
            drug3d=drug3d,
            device=device,
            lru_cache=lru_cache,
            use_final_cache=use_final_cache,
            use_2d_token_cache=use_2d_token_cache,
            use_3d_prefix_cache=use_3d_prefix_cache,
        )
        cache_hits += int(hit)
        H2_map[did] = H2
        H3_map[did] = H3
        z2_list.append(z2)
        z3_list.append(z3)

    t_enc = time.time() - t0
    z2 = torch.cat(z2_list, dim=0)  # (m, D)
    z3 = torch.cat(z3_list, dim=0)  # (m, D)
    return H2_map, H3_map, z2, z3, cache_hits, t_enc

