"""
Eunoia Raiden v2 (恵雷) — Loss Functions
SHV Groups AGI Research Division
training/losses.py

Three loss components used across Phase 1 and Phase 2 training:

  policy_loss               : supervised cross-entropy on token sequences
  value_loss                : MSE on scalar value head prediction
  execution_loss_reinforce  : REINFORCE with EMA baseline advantage
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def policy_loss(
    logits:  torch.Tensor,   # [B, T, vocab_size]
    targets: torch.Tensor,   # [B, T]  — token ids, -100 = ignore
) -> torch.Tensor:
    """Autoregressive cross-entropy over the program token sequence."""
    return F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=-100)


def value_loss(
    pred:   torch.Tensor,   # [B, 1]
    actual: torch.Tensor,   # [B, 1]
) -> torch.Tensor:
    """MSE between predicted task-success probability and observed reward."""
    return F.mse_loss(pred, actual)


def execution_loss_reinforce(
    log_probs: torch.Tensor,   # [T]  per-token log-probabilities
    reward:    float,
    baseline:  float,
) -> torch.Tensor:
    """
    REINFORCE with EMA baseline.
    Advantage = reward - baseline.
    Loss = -(mean log-prob) * advantage
    Dividing by sequence length normalises gradient magnitude across programs
    of different lengths.
    """
    advantage = float(reward) - float(baseline)
    seq_len   = max(log_probs.shape[0], 1)
    return -(log_probs.sum() / seq_len) * advantage