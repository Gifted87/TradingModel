"""
Phase 6 — Residual Learner PyTorch Training Engine.

REWRITE (v2):
  - Early stopping / checkpoint selection is now on the DIRECTION task
    (validation standardized return loss) instead of the composite total,
    which was dominated by magnitude/QLIKE terms and blind to direction
    (selection_metric="return"; "total" restores old behavior).
  - DataLoader uses the dataset's batched-gather collate (no per-sample
    tensor construction). Works with SequenceDataset, ConcatSequenceDataset,
    and torch Subset wrappers of both.
  - Optional MLflow logging per Build Plan Phase 6.5: params include fold ID
    and feature-registry version; per-epoch loss components logged as metrics.
    Guarded import — training proceeds without mlflow installed.
  - val_split="chronological" (default, single symbol) or "random" (used by
    the multi-symbol transfer test).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .dataset import BatchData, SequenceDataset
from .loss import CompositeMultiTaskLoss, LossBreakdown
from .network import ResidualLearnerNetwork

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    epochs: int = 15
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 4
    min_delta: float = 1e-4
    max_grad_norm: float = 1.0
    val_split_fraction: float = 0.15
    val_split: str = "chronological"        # "chronological" | "random"
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    # Model selection: which validation loss component drives early stopping.
    # "return" = the direction task (default; fixes the old total-loss trap).
    selection_metric: str = "return"        # "return" | "total" | "mag" | "rv"
    # MLflow (Build Plan Phase 6.5) — all optional
    mlflow_experiment: Optional[str] = None
    mlflow_run_name: Optional[str] = None
    fold_id: Optional[str] = None
    registry_version: Optional[str] = None


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_breakdown: LossBreakdown
    val_loss: float
    val_breakdown: LossBreakdown
    learning_rate: float


class ResidualLearnerTrainer:
    """Training, validation, direction-aware early stopping, checkpointing."""

    def __init__(
        self,
        config: Optional[TrainerConfig] = None,
        loss_fn: Optional[CompositeMultiTaskLoss] = None,
    ):
        self.config = config or TrainerConfig()
        self.loss_fn = loss_fn or CompositeMultiTaskLoss()
        self.device = torch.device(self.config.device)
        self.loss_fn.to(self.device)
        self.best_val_score_: float = float("inf")

    # ── helpers ────────────────────────────────────────────────────────────

    def _selection_score(self, bd: LossBreakdown) -> float:
        m = self.config.selection_metric
        if m == "return":
            return bd.return_loss
        if m == "mag":
            return bd.mag_loss
        if m == "rv":
            return bd.rv_loss
        return bd.total_loss

    def _start_mlflow(self, network: ResidualLearnerNetwork, n_tr: int, n_val: int):
        cfg = self.config
        if not cfg.mlflow_experiment:
            return None
        try:
            import mlflow
            mlflow.set_experiment(cfg.mlflow_experiment)
            mlflow.start_run(run_name=cfg.mlflow_run_name)
            mlflow.log_params({
                "fold_id": cfg.fold_id or "",
                "registry_version": cfg.registry_version or "",
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "learning_rate": cfg.learning_rate,
                "weight_decay": cfg.weight_decay,
                "selection_metric": cfg.selection_metric,
                "n_train_seq": n_tr,
                "n_val_seq": n_val,
                "n_params": sum(p.numel() for p in network.parameters()),
            })
            return mlflow
        except Exception as exc:  # never let tracking break training
            logger.warning(f"MLflow unavailable ({exc!r}); continuing without tracking.")
            return None

    # ── main loop ──────────────────────────────────────────────────────────

    def train(
        self,
        network: ResidualLearnerNetwork,
        train_dataset,                      # SequenceDataset | ConcatSequenceDataset
        val_dataset=None,
    ) -> Tuple[ResidualLearnerNetwork, List[EpochMetrics]]:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        network.to(self.device)

        collate = train_dataset.collate

        if val_dataset is None:
            n_total = len(train_dataset)
            n_val = max(20, int(n_total * self.config.val_split_fraction))
            n_tr = n_total - n_val
            if self.config.val_split == "random":
                perm = np.random.permutation(n_total)
                tr_subset = Subset(train_dataset, perm[n_val:].tolist())
                val_subset = Subset(train_dataset, perm[:n_val].tolist())
            else:  # chronological — no temporal leakage
                tr_subset = Subset(train_dataset, list(range(n_tr)))
                val_subset = Subset(train_dataset, list(range(n_tr, n_total)))
        else:
            tr_subset = train_dataset
            val_subset = val_dataset

        train_loader = DataLoader(
            tr_subset, batch_size=self.config.batch_size, shuffle=True,
            collate_fn=collate, num_workers=self.config.num_workers,
            pin_memory=(self.config.device == "cuda"),
        )
        val_loader = DataLoader(
            val_subset, batch_size=self.config.batch_size, shuffle=False,
            collate_fn=collate, num_workers=self.config.num_workers,
        )

        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs, eta_min=1e-5
        )

        mlf = self._start_mlflow(network, len(tr_subset), len(val_subset))

        best_score = float("inf")
        best_state = None
        patience_counter = 0
        history: List[EpochMetrics] = []

        logger.info(
            f"Training Branch B on {self.device} ({len(tr_subset):,} train, "
            f"{len(val_subset):,} val, ≤{self.config.epochs} epochs, "
            f"selection={self.config.selection_metric})..."
        )

        try:
            for epoch in range(1, self.config.epochs + 1):
                # ── train ──────────────────────────────────────────────────
                network.train()
                tr_tot, tr_ret, tr_mag, tr_rv, tr_q = [], [], [], [], []
                for batch in train_loader:
                    batch = self._to_device(batch)
                    optimizer.zero_grad()
                    outputs = network(batch.sequences, batch.r_hat_lin, batch.g6_features)
                    loss, bd = self.loss_fn(outputs, batch)
                    loss.backward()
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(network.parameters(), self.config.max_grad_norm)
                    optimizer.step()
                    tr_tot.append(bd.total_loss); tr_ret.append(bd.return_loss)
                    tr_mag.append(bd.mag_loss); tr_rv.append(bd.rv_loss); tr_q.append(bd.quantile_loss)

                current_lr = scheduler.get_last_lr()[0]
                scheduler.step()

                # ── validate ───────────────────────────────────────────────
                network.eval()
                v_tot, v_ret, v_mag, v_rv, v_q = [], [], [], [], []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = self._to_device(batch)
                        outputs = network(batch.sequences, batch.r_hat_lin, batch.g6_features)
                        _, bd = self.loss_fn(outputs, batch)
                        v_tot.append(bd.total_loss); v_ret.append(bd.return_loss)
                        v_mag.append(bd.mag_loss); v_rv.append(bd.rv_loss); v_q.append(bd.quantile_loss)

                tr_bd = LossBreakdown(float(np.mean(tr_tot)), float(np.mean(tr_ret)),
                                      float(np.mean(tr_mag)), float(np.mean(tr_rv)), float(np.mean(tr_q)))
                v_bd = LossBreakdown(float(np.mean(v_tot)), float(np.mean(v_ret)),
                                     float(np.mean(v_mag)), float(np.mean(v_rv)), float(np.mean(v_q)))
                history.append(EpochMetrics(epoch, tr_bd.total_loss, tr_bd, v_bd.total_loss, v_bd, current_lr))

                if mlf is not None:
                    try:
                        mlf.log_metrics({
                            "train_total": tr_bd.total_loss, "train_return": tr_bd.return_loss,
                            "val_total": v_bd.total_loss, "val_return": v_bd.return_loss,
                            "val_mag": v_bd.mag_loss, "val_rv": v_bd.rv_loss,
                            "lr": current_lr,
                        }, step=epoch)
                    except Exception:
                        pass

                logger.debug(
                    f"Epoch {epoch:2d}/{self.config.epochs} | train {tr_bd.total_loss:.4f} "
                    f"(ret {tr_bd.return_loss:.4f}) | val {v_bd.total_loss:.4f} "
                    f"(ret {v_bd.return_loss:.4f}) | lr {current_lr:.2e}"
                )

                # ── direction-aware early stopping ─────────────────────────
                score = self._selection_score(v_bd)
                if score < (best_score - self.config.min_delta):
                    best_score = score
                    best_state = copy.deepcopy(network.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        logger.debug(f"Early stopping at epoch {epoch} (best {self.config.selection_metric}={best_score:.5f}).")
                        break
        finally:
            if mlf is not None:
                try:
                    mlf.end_run()
                except Exception:
                    pass

        if best_state is not None:
            network.load_state_dict(best_state)
        self.best_val_score_ = float(best_score)
        network.eval()
        return network, history

    def _to_device(self, batch: BatchData) -> BatchData:
        return BatchData(
            sequences=batch.sequences.to(self.device),
            r_hat_lin=batch.r_hat_lin.to(self.device),
            g6_features=batch.g6_features.to(self.device),
            r30=batch.r30.to(self.device),
            residual=batch.residual.to(self.device),
            mag30=batch.mag30.to(self.device),
            rv30=batch.rv30.to(self.device),
            abs_r30=batch.abs_r30.to(self.device),
        )