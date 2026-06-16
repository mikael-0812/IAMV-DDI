import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any

def pad_tokens(token_list: List[torch.Tensor], pad_value=0.0):
    """
    token_list: list of (L_i, D)
    Returns:
      X: (B, Lmax, D)
      pad_mask: (B, Lmax) True=PAD (for key_padding_mask)
    """
    B = len(token_list)
    D = token_list[0].size(-1)
    Lmax = max(t.size(0) for t in token_list)
    X = token_list[0].new_full((B, Lmax, D), float(pad_value))
    pad_mask = torch.ones((B, Lmax), dtype=torch.bool, device=token_list[0].device)
    for i, t in enumerate(token_list):
        L = t.size(0)
        X[i, :L] = t
        pad_mask[i, :L] = False
    return X, pad_mask


class QueryPooling(nn.Module):
    def __init__(self, d_model=256, num_queries=16, nhead=8, dropout=0.1, return_attn=False):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.empty(num_queries, d_model))
        nn.init.normal_(self.queries, std=0.02)

        self.q_ln  = nn.LayerNorm(d_model)     # Pre-LN for Q
        self.kv_ln = nn.LayerNorm(d_model)     # Pre-LN for X
        self.attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.drop = nn.Dropout(dropout)

        #FFN block
        self.ffn_ln = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model),
        )

        self.return_attn = return_attn

    def forward(self, X, key_padding_mask=None, return_attn=None):
        """
        X: (B, L, D)
        key_padding_mask: (B, L) with True=PAD(ignore)  [đúng chuẩn PyTorch MHA]
        """
        if return_attn is None:
            return_attn = self.return_attn

        B = X.size(0)
        Q = self.queries.unsqueeze(0).expand(B, -1, -1)   # (B, K, D)

        # Pre-LN
        Qn = self.q_ln(Q)
        Xn = self.kv_ln(X)

        out, attn_w = self.attn(
            Qn, Xn, Xn,
            key_padding_mask=key_padding_mask,
            need_weights=return_attn,
            average_attn_weights=False
        )
        Q = Q + self.drop(out)
        Q2 = self.ffn_ln(Q)
        Q = Q + self.drop(self.ffn(Q2))

        if return_attn:
            return Q, attn_w
        return Q

class CrossAttnBlock(nn.Module):
    def __init__(self, d_model=256, nhead=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, Q, K, key_padding_mask=None):
        h, _ = self.attn(Q, K, K, key_padding_mask=key_padding_mask, need_weights=False)
        Q = self.ln1(Q + self.drop(h))
        Q = self.ln2(Q + self.drop(self.ffn(Q)))
        return Q