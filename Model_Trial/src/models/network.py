"""
Phase 6 — Deep Residual Learner Network Architecture (Branch B + Cascade Heads).

Implements Mechanism III from Whitepaper §6 and Build Plan Phase 6:
  - Multi-width Conv1D blocks (kernel widths 3, 5, 7) for multi-scale indicator interactions [F3]
  - 2-layer BiLSTM sequence learner [F3, B1]
  - Learned Attention Pooling layer across sequence steps
  - Head 1: Residual Return Head (predicts r30 - r_hat_lin)
  - Head 2: Cascade Magnitude & RV Head (consumes trunk ⊕ r̂30 ⊕ G6 volatility state) [E3, P9]
  - Head 3: Quantile Head (predicts 9 pinball quantiles q0.1...q0.9 of |r30|)

Regularization & capacity:
  - Bounded parameter footprint (~75k parameters) to prevent overfitting on financial residuals [C1, P11]
  - Dropout + LayerNorm + Weight Decay from the start.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelOutputs:
    """Strongly-typed container for all composite model head outputs."""
    residual_hat: torch.Tensor       # (B, 1) predicted residual
    r30_hat: torch.Tensor            # (B, 1) final continuous return = r_hat_lin + residual_hat
    mag30_hat: torch.Tensor          # (B, 1) predicted log-magnitude
    rv30_hat: torch.Tensor           # (B, 1) predicted log-RV
    quantiles_hat: torch.Tensor      # (B, 9) predicted quantiles q0.1 ... q0.9 of |r30|
    attention_weights: torch.Tensor  # (B, L) learned attention weights over lookback sequence


class MultiWidthConvBlock(nn.Module):
    """
    Multi-width 1D temporal convolution block capturing short, medium, and
    long-range indicator interactions simultaneously across time [F3].
    """

    def __init__(
        self,
        in_features: int,
        out_channels_per_kernel: int = 32,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.15,
    ):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=in_features,
                out_channels=out_channels_per_kernel,
                kernel_size=k,
                padding=k // 2,  # Same padding to preserve sequence length L
            )
            for k in kernel_sizes
        ])
        total_channels = out_channels_per_kernel * len(kernel_sizes)
        self.norm = nn.LayerNorm(total_channels)
        self.dropout = nn.Dropout(dropout)
        self.total_channels = total_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F) -> transpose to (B, F, L) for Conv1D
        x_t = x.transpose(1, 2)
        conv_outs = [F.gelu(conv(x_t)) for conv in self.convs]
        # Concatenate along channel dimension: (B, total_channels, L)
        cat = torch.cat(conv_outs, dim=1)
        # Transpose back: (B, L, total_channels)
        cat_t = cat.transpose(1, 2)
        out = self.dropout(self.norm(cat_t))
        return out


class AttentionPooling(nn.Module):
    """
    Learned attention pooling layer that collapses the sequence dimension L
    into a single summary context vector h_trunk.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L, H)
        scores = self.attn(x)                       # (B, L, 1)
        weights = F.softmax(scores, dim=1)           # (B, L, 1)
        context = torch.sum(x * weights, dim=1)      # (B, H)
        return context, weights.squeeze(-1)


class ResidualLearnerNetwork(nn.Module):
    """
    Complete PyTorch sequence architecture for Branch B residual learning,
    cascade magnitude/RV forecasting, and quantile interval estimation.
    """

    def __init__(
        self,
        in_features: int,
        g6_dim: int = 10,
        conv_channels: int = 32,
        lstm_hidden: int = 32,
        lstm_layers: int = 2,
        dropout: float = 0.2,
        n_quantiles: int = 9,
    ):
        super().__init__()
        self.in_features = in_features
        self.g6_dim = g6_dim
        self.n_quantiles = n_quantiles

        # 1. Multi-width Conv1D feature extractor
        self.conv_block = MultiWidthConvBlock(
            in_features=in_features,
            out_channels_per_kernel=conv_channels,
            kernel_sizes=(3, 5, 7),
            dropout=dropout,
        )

        # 2. 2-layer Bidirectional LSTM
        conv_out_dim = self.conv_block.total_channels  # 32 * 3 = 96
        self.lstm = nn.LSTM(
            input_size=conv_out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        trunk_dim = lstm_hidden * 2  # 64-dim bidirectional representation

        # 3. Attention pooling
        self.attention = AttentionPooling(hidden_dim=trunk_dim)
        self.trunk_norm = nn.LayerNorm(trunk_dim)
        self.trunk_dropout = nn.Dropout(dropout)

        # ── HEAD 1: Residual Return Head ──────────────────────────────────
        self.residual_head = nn.Sequential(
            nn.Linear(trunk_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # ── HEAD 2: Cascade Magnitude & Realized Volatility Head [E3, P9] ──
        # Inputs: trunk (64) + r̂30 (1) + G6 volatility state (g6_dim)
        cascade_in_dim = trunk_dim + 1 + g6_dim
        self.cascade_head = nn.Sequential(
            nn.Linear(cascade_in_dim, 48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48, 2),  # Outputs (mag30_hat, rv30_hat)
        )

        # ── HEAD 3: Quantile Head (Pinball interval estimation) ───────────
        quant_in_dim = trunk_dim + 1
        self.quantile_head = nn.Sequential(
            nn.Linear(quant_in_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_quantiles),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize linear and conv weights using Kaiming normal."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,              # (B, L, F)
        r_hat_lin: torch.Tensor,      # (B, 1) from Branch A
        g6_features: torch.Tensor,    # (B, F_g6) G6 state
    ) -> ModelOutputs:
        """
        Full forward pass through Branch B trunk and cascade heads.
        """
        # 1. Multi-width Conv1D interaction extraction
        conv_feat = self.conv_block(x)                         # (B, L, 96)

        # 2. 2-layer BiLSTM
        lstm_out, _ = self.lstm(conv_feat)                     # (B, L, 64)

        # 3. Attention pooling
        h_trunk, attn_weights = self.attention(lstm_out)       # (B, 64), (B, L)
        h_trunk = self.trunk_dropout(self.trunk_norm(h_trunk)) # (B, 64)

        # 4. Head 1: Residual & Final Return
        residual_hat = self.residual_head(h_trunk)             # (B, 1)
        r30_hat = r_hat_lin + residual_hat                     # (B, 1)

        # 5. Head 2: Cascade Magnitude & Volatility
        # Concatenate trunk ⊕ r̂30 ⊕ G6 volatility features
        cascade_input = torch.cat([h_trunk, r30_hat, g6_features], dim=-1)
        cascade_out = self.cascade_head(cascade_input)         # (B, 2)
        mag30_hat = cascade_out[:, 0:1]
        rv30_hat = cascade_out[:, 1:2]

        # 6. Head 3: Quantiles on |r30|
        quant_input = torch.cat([h_trunk, r30_hat], dim=-1)
        raw_quantiles = self.quantile_head(quant_input)        # (B, 9)
        # Ensure strictly non-negative quantile predictions via softplus
        quantiles_hat = F.softplus(raw_quantiles)

        return ModelOutputs(
            residual_hat=residual_hat,
            r30_hat=r30_hat,
            mag30_hat=mag30_hat,
            rv30_hat=rv30_hat,
            quantiles_hat=quantiles_hat,
            attention_weights=attn_weights,
        )
