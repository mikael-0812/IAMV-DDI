import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from src.model.model_ddi import AttnPool, InterDrugInteraction, IntraDrugFusionAbl, IntraDrugFusion


class SingleViewDrugTokens(nn.Module):
    """
    Used for w/o 2D or w/o 3D ablation.
    Input: list of token tensors [(L_i, 128)]
    Output: T: (B, K, d_fuse)
    """
    def __init__(
        self,
        d_in: int = 128,
        d_fuse: int = 128,
        K: int = 16,
        nhead: int = 8,
        dropout: float = 0.1,
        layernorm: bool = True,
    ):
        super().__init__()
        self.K = K
        self.proj = nn.Linear(d_in, d_fuse)

        self.query = nn.Parameter(torch.randn(K, d_fuse) * 0.02)

        self.q_attn = nn.MultiheadAttention(
            embed_dim=d_fuse,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_fuse) if layernorm else nn.Identity()

    def _pad_tokens(self, token_list):
        lengths = [x.size(0) for x in token_list]
        X = pad_sequence(token_list, batch_first=True)  # (B, Lmax, Din)

        B, Lmax, _ = X.shape
        device = X.device

        mask = torch.zeros(B, Lmax, dtype=torch.bool, device=device)
        for i, l in enumerate(lengths):
            if l < Lmax:
                mask[i, l:] = True

        return X, mask

    def forward(self, H_list):
        X, mask = self._pad_tokens(H_list)   # (B, L, 128)
        X = self.proj(X)                     # (B, L, D)

        B = X.size(0)
        Q = self.query.unsqueeze(0).expand(B, -1, -1)  # (B, K, D)

        T, _ = self.q_attn(
            query=Q,
            key=X,
            value=X,
            key_padding_mask=mask,
            need_weights=False,
        )

        T = self.ln(self.drop(T))
        g = T.mean(dim=1)

        return T, g

class DDIHead(nn.Module):
    """
    Multi-class DDI event prediction with ablation modes.

    ablation_mode:
      - "full"              : full model
      - "wo_2d"             : remove 2D branch, use only 3D
      - "wo_3d"             : remove 3D branch, use only 2D
      - "wo_intra_fusion"   : remove intra-drug 2D-3D cross-fusion
      - "wo_inter_fusion"   : remove inter-drug cross-attention
    """
    def __init__(
        self,
        num_rel: int = 86,
        d_fuse: int = 128,
        K: int = 16,
        nhead: int = 8,
        intra_layers: int = 1,
        inter_layers: int = 1,
        dropout: float = 0.1,
        layernorm: bool = True,
        mlp_hidden: int = 256,
        ablation_mode: str = "full",
    ):
        super().__init__()

        valid_modes = {
            "full",
            "wo_2d",
            "wo_3d",
            "wo_intra_fusion",
            "wo_inter_fusion",
        }

        if ablation_mode not in valid_modes:
            raise ValueError(
                f"Unknown ablation_mode={ablation_mode}. "
                f"Choose from {sorted(valid_modes)}"
            )

        self.d_fuse = d_fuse
        self.num_rel = num_rel
        self.ablation_mode = ablation_mode

        # ======================================================
        # Intra-drug encoder
        # ======================================================
        if ablation_mode == "wo_2d":
            # Remove 2D branch, keep only 3D tokens
            self.intra = SingleViewDrugTokens(
                d_in=128,
                d_fuse=d_fuse,
                K=K,
                nhead=nhead,
                dropout=dropout,
                layernorm=layernorm,
            )

        elif ablation_mode == "wo_3d":
            # Remove 3D branch, keep only 2D tokens
            self.intra = SingleViewDrugTokens(
                d_in=128,
                d_fuse=d_fuse,
                K=K,
                nhead=nhead,
                dropout=dropout,
                layernorm=layernorm,
            )

        else:
            # full / w/o intra / w/o inter
            effective_intra_layers = intra_layers

            if ablation_mode == "wo_intra_fusion":
                # Disable 2D-3D cross-attention inside a drug
                effective_intra_layers = 0

            self.intra = IntraDrugFusion(
                128,
                128,
                d_fuse=d_fuse,
                K=K,
                nhead=nhead,
                nlayers=effective_intra_layers,
                dropout=dropout,
            )

        # ======================================================
        # Inter-drug interaction module
        # ======================================================
        effective_inter_layers = inter_layers

        if ablation_mode == "wo_inter_fusion":
            effective_inter_layers = 0

        self.inter_layers = effective_inter_layers

        self.inter = InterDrugInteraction(
            d_model=d_fuse,
            nhead=nhead,
            nlayers=effective_inter_layers,
            dropout=dropout,
        )

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

    def encode_drug_pair(
        self,
        H2_H_list,
        H3_H_list,
        H2_T_list,
        H3_T_list,
    ):
        """
        Return:
          TH: head drug tokens, (B, K, D)
          TT: tail drug tokens, (B, K, D)
        """
        if self.ablation_mode == "wo_2d":
            # only 3D
            TH, _ = self.intra(H3_H_list)
            TT, _ = self.intra(H3_T_list)

        elif self.ablation_mode == "wo_3d":
            # only 2D
            TH, _ = self.intra(H2_H_list)
            TT, _ = self.intra(H2_T_list)

        else:
            # full / wo_intra_fusion / wo_inter_fusion
            TH, _ = self.intra(H2_H_list, H3_H_list)
            TT, _ = self.intra(H2_T_list, H3_T_list)

        return TH, TT

    def forward(self, H2_H_list, H3_H_list, H2_T_list, H3_T_list):
        TH, TT = self.encode_drug_pair(
            H2_H_list,
            H3_H_list,
            H2_T_list,
            H3_T_list,
        )

        # Inter-drug cross-attention
        if self.inter_layers > 0:
            TH, TT = self.inter(TH, TT)

        vH = self.pool_h(TH)   # (B, D)
        vT = self.pool_t(TT)   # (B, D)

        vH = self.ln(self.drop(vH))
        vT = self.ln(self.drop(vT))

        feat = torch.cat(
            [
                vH,
                vT,
                torch.abs(vH - vT),
                vH * vT,
            ],
            dim=-1,
        )  # (B, 4D)

        logits = self.fc_layers(feat)  # (B, num_rel)

        return logits, vH, vT