"""Causal Inception-Conformer Multi-Task Network Architecture (Plan 3 - Hardened).

Designed for high-frequency financial forecasting (45-minute continuous cause -> 5-minute effect).
Combines:
1. Strictly Causal Multi-Scale Inception-1D Convolutions (k=3, 5, 15 with causal padding (k-1)*d).
2. Conformer Multi-Head Self-Attention & Macaron-style Depthwise Convolutions.
3. MultiScaleInputEmbedding (Continuous Multi-Scale Projections + RBF Soft Bins + 15m Group Embeddings + Macro Group Tensor).
4. Multi-Task Specialized Prediction Heads:
   - 9-Bin Ordinal Distribution Head (or Cumulative Link CDF)
   - Auxiliary Continuous Return Regression Head (E[r | x])
   - Cost-Aware 3-Class Direction Gate (Long, Dead Zone, Short)
   - Conditional Variance / Volatility Head (Var[r | x])
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from src.continuous_features import MultiScaleInputEmbedding
except ImportError:
    from continuous_features import MultiScaleInputEmbedding


class CausalConv1d(nn.Module):
    """1D Convolution with strictly causal left-padding to prevent future lookahead."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, L)
        x_padded = F.pad(x, (self.causal_padding, 0))
        return self.conv(x_padded)


class CausalInceptionBlock(nn.Module):
    """Multi-Scale Causal Inception Block capturing 3m micro-wicks, 5m bars, and 15m group trends."""

    def __init__(self, channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        branch_ch = channels // 4
        
        # Branch 1 (k=3): Microstructural wicks & tick reversals
        self.branch3 = nn.Sequential(
            CausalConv1d(channels, branch_ch, kernel_size=3),
            nn.GroupNorm(num_groups=min(4, branch_ch), num_channels=branch_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Branch 2 (k=5): 5-Minute candle formation & momentum
        self.branch5 = nn.Sequential(
            CausalConv1d(channels, branch_ch, kernel_size=5),
            nn.GroupNorm(num_groups=min(4, branch_ch), num_channels=branch_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Branch 3 (k=15): 15-Minute group trend & macro momentum
        self.branch15 = nn.Sequential(
            CausalConv1d(channels, branch_ch, kernel_size=15),
            nn.GroupNorm(num_groups=min(4, branch_ch), num_channels=branch_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Branch 4: Causal Dilated Conv for wider receptive field (k=3, dilation=5 -> effective k=11)
        self.branch_dilated = nn.Sequential(
            CausalConv1d(channels, branch_ch, kernel_size=3, dilation=5),
            nn.GroupNorm(num_groups=min(4, branch_ch), num_channels=branch_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 1x1 Projection back to full channels
        self.proj_out = nn.Conv1d(branch_ch * 4, channels, kernel_size=1)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x_in = x.transpose(1, 2)  # (B, channels, L)
        
        b3 = self.branch3(x_in)
        b5 = self.branch5(x_in)
        b15 = self.branch15(x_in)
        bd = self.branch_dilated(x_in)
        
        cat = torch.cat([b3, b5, b15, bd], dim=1)  # (B, channels, L)
        out = self.proj_out(cat).transpose(1, 2)   # (B, L, channels)
        
        return self.norm(res + self.dropout(out))


class ConformerBlock(nn.Module):
    """Macaron-Style Conformer Encoder Block."""

    def __init__(self, d_model: int, nhead: int = 4, d_ff: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        
        # 1. Half-step FFN 1
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        # 2. Multi-Head Attention
        self.norm_mha = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop_mha = nn.Dropout(dropout)
        
        # 3. Causal Depthwise Convolution Module
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv_module = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, kernel_size=1),  # Pointwise
            nn.GLU(dim=1),
            CausalConv1d(d_model, d_model, kernel_size=5, groups=d_model),  # Depthwise
            nn.GroupNorm(num_groups=min(4, d_model), num_channels=d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),      # Pointwise
            nn.Dropout(dropout),
        )
        
        # 4. Half-step FFN 2
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. First FFN half-step
        x = x + 0.5 * self.ffn1(x)
        
        # 2. Multi-Head Self-Attention
        norm_x = self.norm_mha(x)
        attn_out, _ = self.mha(norm_x, norm_x, norm_x, attn_mask=mask)
        x = x + self.drop_mha(attn_out)
        
        # 3. Conv Module
        norm_x = self.norm_conv(x).transpose(1, 2)
        conv_out = self.conv_module(norm_x).transpose(1, 2)
        x = x + conv_out
        
        # 4. Second FFN half-step
        x = x + 0.5 * self.ffn2(x)
        
        return self.final_norm(x)


class AttentivePooling(nn.Module):
    """Learned cross-sequence attentive pooling compressing (B, L, D) to (B, D)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.key_proj = nn.Linear(d_model, d_model)
        self.scale = 1.0 / (d_model ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        keys = self.key_proj(x)
        scores = torch.sum(self.query * keys, dim=-1) * self.scale
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = torch.sum(x * weights, dim=1)
        return pooled


class HybridInceptionConformer(nn.Module):
    """Complete Multi-Task Inception-Conformer Backbone (Plan 3)."""

    def __init__(
        self,
        continuous_dim: int = 28,
        group_feat_dim: int = 12,
        num_target_bins: int = 9,
        rbf_bins: int = 9,
        rbf_embed_dim: int = 32,
        d_model: int = 128,
        nhead: int = 4,
        num_conformer_layers: int = 2,
        d_ff: int = 256,
        lookback_bars: int = 45,
        num_groups: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lookback_bars = lookback_bars
        self.num_target_bins = num_target_bins
        
        # 1. Multi-Scale Input Embedding + RBF Soft Bins + Group/Segment IDs + Macro Group Tensor
        self.embedding = MultiScaleInputEmbedding(
            continuous_dim=continuous_dim,
            group_feat_dim=group_feat_dim,
            rbf_bins=rbf_bins,
            rbf_embed_dim=rbf_embed_dim,
            d_model=d_model,
            lookback_bars=lookback_bars,
            num_groups=num_groups,
            dropout=dropout,
        )
        
        # 2. Causal Multi-Scale Inception Block (3m, 5m, 15m, dilated)
        self.inception = CausalInceptionBlock(channels=d_model, dropout=dropout)
        
        # 3. Conformer Layers (Self-Attention + Depthwise Conv)
        self.conformer_layers = nn.ModuleList([
            ConformerBlock(d_model=d_model, nhead=nhead, d_ff=d_ff, dropout=dropout)
            for _ in range(num_conformer_layers)
        ])
        
        # 4. Attentive Temporal Pooling
        self.pooler = AttentivePooling(d_model=d_model)
        
        # 5. Shared Representation Trunk
        self.trunk = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        
        # 6. Multi-Task Prediction Heads
        self.head_dist = nn.Linear(d_model, num_target_bins)
        
        self.head_reg = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        
        self.head_dir = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )
        
        self.head_vol = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x_cont: torch.Tensor,
        z_1m: torch.Tensor,
        group_ids: Optional[torch.Tensor] = None,
        group_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_cont: (B, L, continuous_dim) continuous feature sequence
            z_1m: (B, L) normalized 1-min returns
            group_ids: Optional (B, L) group integer IDs
            group_features: Optional (B, group_feat_dim) macro group transition statistics
        Returns:
            Dictionary with multi-task prediction outputs.
        """
        # 1. Embed inputs (fuses features, RBF soft bins, segment IDs, and group transition tensor)
        x = self.embedding(x_cont, z_1m, group_ids, group_features)  # (B, L, d_model)
        
        # 2. Causal Inception Feature Extraction
        x = self.inception(x)  # (B, L, d_model)
        
        # 3. Conformer Encoders (with causal attention mask: bar T attends only to bars 0..T)
        seq_len = x.shape[1]
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        )
        for layer in self.conformer_layers:
            x = layer(x, mask=causal_mask)
            
        # 4. Attentive Pooling & Shared Trunk
        h_pool = self.pooler(x)  # (B, d_model)
        h = self.trunk(h_pool)   # (B, d_model)
        
        # 5. Multi-Task Heads
        logits = self.head_dist(h)
        probs = F.softmax(logits, dim=-1)
        
        pred_return = self.head_reg(h)
        
        dir_logits = self.head_dir(h)
        dir_probs = F.softmax(dir_logits, dim=-1)
        
        pred_vol = F.softplus(self.head_vol(h)) + 1e-4
        
        return {
            "logits": logits,
            "probs": probs,
            "pred_return": pred_return,
            "dir_logits": dir_logits,
            "dir_probs": dir_probs,
            "pred_vol": pred_vol,
        }
