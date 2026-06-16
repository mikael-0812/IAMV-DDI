import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.cross_attention import CrossAttnBlock, QueryPooling, pad_tokens

class IntraDrugFusion(nn.Module):
    """
    Inputs:
      H2_list: list[(L2_i,128)]
      H3_list: list[(L3_i,128)]
    Outputs:
      T: (B, K, Df)
      g: (B, Df)
    """
    def __init__(self, d_in2=128, d_in3=128, d_fuse=256, K=16, nhead=8, nlayers=1, dropout=0.1):
        super().__init__()
        self.proj2 = nn.Linear(d_in2, d_fuse)
        self.proj3 = nn.Linear(d_in3, d_fuse)

        self.xattn_2_to_3 = nn.ModuleList([CrossAttnBlock(d_fuse, nhead, dropout) for _ in range(nlayers)])
        self.xattn_3_to_2 = nn.ModuleList([CrossAttnBlock(d_fuse, nhead, dropout) for _ in range(nlayers)])

        self.pool2 = QueryPooling(d_model=d_fuse, num_queries=K, nhead=nhead, dropout=dropout)
        self.pool3 = QueryPooling(d_model=d_fuse, num_queries=K, nhead=nhead, dropout=dropout)

        self.gate = nn.Sequential(
            nn.Linear(2 * d_fuse, d_fuse),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fuse, 1),
            nn.Sigmoid()
        )

        self.ln = nn.LayerNorm(d_fuse)

    def forward(self, H2_list, H3_list):
        X2_list = [self.proj2(h) for h in H2_list]
        X3_list = [self.proj3(h) for h in H3_list]
        X2, m2 = pad_tokens(X2_list)
        X3, m3 = pad_tokens(X3_list)

        X2_old = X2

        for blk in self.xattn_2_to_3:
            X2 = blk(X2, X3, key_padding_mask=m3)
        for blk in self.xattn_3_to_2:
            X3 = blk(X3, X2_old, key_padding_mask=m2)

        P2 = self.pool2(X2, key_padding_mask=m2)  # (B,K,Df)
        P3 = self.pool3(X3, key_padding_mask=m3)

        s2 = P2.mean(dim=1)
        s3 = P3.mean(dim=1)
        a = self.gate(torch.cat([s2, s3], dim=-1))  # (B,1)

        T = a.unsqueeze(1) * P2 + (1 - a.unsqueeze(1)) * P3
        T = self.ln(T)
        g = T.mean(dim=1)
        return T, g

class InterDrugInteraction(nn.Module):
    def __init__(self, d_model=256, nhead=8, nlayers=2, dropout=0.1):
        super().__init__()
        self.nlayers = nlayers
        self.ab = nn.ModuleList([CrossAttnBlock(d_model, nhead, dropout) for _ in range(nlayers)])
        self.ba = nn.ModuleList([CrossAttnBlock(d_model, nhead, dropout) for _ in range(nlayers)])

    def forward(self, TA, TB):
        A, B = TA, TB
        if self.nlayers == 0:
            return A, B
        else:
            for blk in self.ab:
                A = blk(A, B, key_padding_mask=None)
            for blk in self.ba:
                B = blk(B, A, key_padding_mask=None)
            return A, B

class AttnPool(nn.Module):
    """
    Attention pooling over K tokens.
    X: (B, K, D) -> v: (B, D)
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        # x: (B, K, D)
        score = self.scorer(x)              # (B, K, 1)
        weight = torch.softmax(score, dim=1)
        v = (x * weight).sum(dim=1)         # (B, D)
        return v

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=128, out_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, p=2, dim=-1)

class DDIHeadV2(nn.Module):
    """
      - Conditioned pooling over K tokens (for head & tail)
      - FiLM gate on pair embedding: emb = pair * gamma + beta
      - Interaction features: [vH, vT, |vH-vT|, vH*vT]
      - Output: 1 logit / sample (B,)
    """
    def __init__(self, num_rel: int = 86, d_fuse: int = 128, K: int = 8, nhead: int = 16, intra_layers: int = 1, inter_layers: int = 1, dropout: float = 0.1, layernorm: bool = True, mlp_hidden: int = 256, film_on: str = "feat",):
        super().__init__()
        self.d_fuse = d_fuse
        self.inter_layers = inter_layers

        self.intra = IntraDrugFusion(128, 128, d_fuse=d_fuse, K=K, nhead=nhead, nlayers=intra_layers, dropout=dropout)
        self.inter = InterDrugInteraction(d_model=d_fuse, nhead=nhead, nlayers=inter_layers, dropout=dropout)
        self.pool_h = AttnPool(d_model=d_fuse, dropout=dropout)
        self.pool_t = AttnPool(d_model=d_fuse, dropout=dropout)

        # base vectors normalization
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_fuse) if layernorm else nn.Identity()

        # interaction features
        # feat = [vH, vT, |vH-vT|, vH*vT] => 4*d_fuse
        feat_dim = 4 * d_fuse
        self.rel_film = nn.Embedding(num_rel, 2 * feat_dim)

        # classifier
        self.fc_layers = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, H2_H_list, H3_H_list, H2_T_list, H3_T_list, r):
        r = r.long()

        TH, _ = self.intra(H2_H_list, H3_H_list)  # (B,K,D)
        TT, _ = self.intra(H2_T_list, H3_T_list)  # (B,K,D)

        if self.inter_layers > 0:
            TH, TT = self.inter(TH, TT)

        vH = self.pool_h(TH)   # (B, D)
        vT = self.pool_t(TT)   # (B, D)

        vH = self.ln(self.drop(vH))
        vT = self.ln(self.drop(vT))

        # interactions
        feat = torch.cat([vH, vT, (vH - vT).abs(), vH * vT], dim=-1)  # (B,4D)

        logits = self.fc_layers(feat).squeeze(-1)  # (B,)
        return logits, vH, vT


class DDIHead(nn.Module):
    """
    For fair comparison with MMFF-DDI-style multi-class event prediction:
      - Output: num_rel logits / sample => (B, num_rel)
    """
    def __init__(self, num_rel: int = 86, d_fuse: int = 128, K: int = 16, nhead: int = 8, intra_layers: int = 1, inter_layers: int = 1, dropout: float = 0.1, layernorm: bool = True, mlp_hidden: int = 256):
        super().__init__()
        self.d_fuse = d_fuse
        self.inter_layers = inter_layers
        self.num_rel = num_rel

        self.intra = IntraDrugFusionAbl(128, 128, d_fuse=d_fuse, K=K, nhead=nhead, nlayers=intra_layers, dropout=dropout)
        self.inter = InterDrugInteraction(d_model=d_fuse, nhead=nhead, nlayers=inter_layers, dropout=dropout)

        self.pool_h = AttnPool(d_model=d_fuse, dropout=dropout)
        self.pool_t = AttnPool(d_model=d_fuse, dropout=dropout)

        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_fuse) if layernorm else nn.Identity()

        feat_dim = 4 * d_fuse

        self.fc_layers = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_rel),
        )

    def forward(self, H2_H_list, H3_H_list, H2_T_list, H3_T_list):
        TH, _ = self.intra(H2_H_list, H3_H_list)  # (B, K, D)
        TT, _ = self.intra(H2_T_list, H3_T_list)  # (B, K, D)

        if self.inter_layers > 0:
            TH, TT = self.inter(TH, TT)

        vH = self.pool_h(TH)   # (B, D)
        vT = self.pool_t(TT)   # (B, D)

        vH = self.ln(self.drop(vH))
        vT = self.ln(self.drop(vT))

        feat = torch.cat(
            [vH, vT, (vH - vT).abs(), vH * vT],
            dim=-1
        )  # (B, 4D)

        logits = self.fc_layers(feat)  # (B, num_rel)
        return logits, vH, vT


class IntraDrugFusionAbl(nn.Module):
    """
    Inputs:
      H2_list: list[(L2_i,128)]
      H3_list: list[(L3_i,128)]
    Outputs:
      T: (B, K, Df)
      g: (B, Df)
    """
    def __init__(
        self,
        d_in2=128,
        d_in3=128,
        d_fuse=256,
        K=16,
        nhead=8,
        nlayers=1,
        dropout=0.1,
        fusion_mode="concat",   # "gate" | "mean" | "sum" | "concat"
    ):
        super().__init__()
        assert fusion_mode in {"gate", "mean", "sum", "concat"}
        self.fusion_mode = fusion_mode

        self.proj2 = nn.Linear(d_in2, d_fuse)
        self.proj3 = nn.Linear(d_in3, d_fuse)

        self.xattn_2_to_3 = nn.ModuleList(
            [CrossAttnBlock(d_fuse, nhead, dropout) for _ in range(nlayers)]
        )
        self.xattn_3_to_2 = nn.ModuleList(
            [CrossAttnBlock(d_fuse, nhead, dropout) for _ in range(nlayers)]
        )

        self.pool2 = QueryPooling(d_model=d_fuse, num_queries=K, nhead=nhead, dropout=dropout)
        self.pool3 = QueryPooling(d_model=d_fuse, num_queries=K, nhead=nhead, dropout=dropout)

        if fusion_mode == "gate":
            self.gate = nn.Sequential(
                nn.Linear(2 * d_fuse, d_fuse),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_fuse, 1),
                nn.Sigmoid()
            )
        else:
            self.gate = None

        if fusion_mode == "concat":
            self.fuse_proj = nn.Sequential(
                nn.Linear(2 * d_fuse, d_fuse),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.fuse_proj = None

        self.ln = nn.LayerNorm(d_fuse)

    def forward(self, H2_list, H3_list):
        X2_list = [self.proj2(h) for h in H2_list]
        X3_list = [self.proj3(h) for h in H3_list]
        X2, m2 = pad_tokens(X2_list)
        X3, m3 = pad_tokens(X3_list)

        X2_old = X2

        for blk in self.xattn_2_to_3:
            X2 = blk(X2, X3, key_padding_mask=m3)
        for blk in self.xattn_3_to_2:
            X3 = blk(X3, X2_old, key_padding_mask=m2)

        P2 = self.pool2(X2, key_padding_mask=m2)  # (B, K, D)
        P3 = self.pool3(X3, key_padding_mask=m3)  # (B, K, D)

        if self.fusion_mode == "mean":
            T = 0.5 * (P2 + P3)
        elif self.fusion_mode == "sum":
            T = P2 + P3
        elif self.fusion_mode == "concat":
            T = self.fuse_proj(torch.cat([P2, P3], dim=-1))
        elif self.fusion_mode == "gate":
            s2 = P2.mean(dim=1)
            s3 = P3.mean(dim=1)
            a = self.gate(torch.cat([s2, s3], dim=-1))   # (B, 1)
            T = a.unsqueeze(1) * P2 + (1 - a.unsqueeze(1)) * P3
        else:
            raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")

        T = self.ln(T)
        g = T.mean(dim=1)
        return T, g

