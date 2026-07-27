"""
Eunoia Raiden v2 (恵雷) — 300M Slot Attention Bottleneck
SHV Groups AGI Research Division
core/slot_attention.py

FIX BUG-008: Slot initialisation is now deterministic during eval/inference.
  Previously torch.randn was called unconditionally in forward(), making every
  inference call non-reproducible and breaking TTA (each augmentation got
  different random slot noise, invalidating cross-augmentation comparison).
  Fix: noise = zeros when self.training is False.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple


class SlotBottleneck(nn.Module):
    """
    Input  : FloatTensor [B, N_objects, 512]
    Returns: (slots [B, 16, 512], entropy_loss scalar < 0)

    Slot initialisation:
      Training  : slots ~ N(mu, sigma²)  — stochastic, encourages diversity
      Eval/infer: slots = mu             — deterministic, reproducible
    """

    def __init__(self, num_slots: int = 16, hidden_dim: int = 512, iters: int = 5):
        super().__init__()
        self.num_slots  = num_slots
        self.hidden_dim = hidden_dim
        self.iters      = iters
        self.scale      = hidden_dim ** -0.5

        self.slots_mu       = nn.Parameter(torch.randn(1, num_slots, hidden_dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, num_slots, hidden_dim))

        self.to_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gru  = nn.GRUCell(hidden_dim, hidden_dim)

        self.norm_inputs = nn.LayerNorm(hidden_dim)
        self.norm_slots  = nn.LayerNorm(hidden_dim)

        # Relational pass — all-pairs attention between slots
        self.rel_q    = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rel_k    = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rel_v    = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rel_norm = nn.LayerNorm(hidden_dim)
        self.rel_ff   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = inputs.shape
        assert D == self.hidden_dim, \
            f"SlotBottleneck: expected hidden_dim={self.hidden_dim}, got {D}"

        inputs_n = self.norm_inputs(inputs)
        k = self.to_k(inputs_n)
        v = self.to_v(inputs_n)

        sigma = torch.exp(self.slots_logsigma).expand(B, -1, -1)

        # BUG-008 FIX: stochastic only during training; deterministic at eval
        if self.training:
            noise = torch.randn(B, self.num_slots, D, device=inputs.device,
                                dtype=inputs.dtype)
        else:
            noise = torch.zeros(B, self.num_slots, D, device=inputs.device,
                                dtype=inputs.dtype)

        slots = self.slots_mu.expand(B, -1, -1) + sigma * noise

        entropy_loss = torch.zeros(1, device=inputs.device, dtype=inputs.dtype)

        for _ in range(self.iters):
            prev  = slots
            q     = self.to_q(self.norm_slots(slots))
            dots  = torch.einsum("bkd,bnd->bkn", q, k) * self.scale
            attn  = dots.softmax(dim=1) + 1e-8
            attn_n = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
            H     = -(attn_n * torch.log(attn_n + 1e-8)).sum(dim=-1).mean()
            entropy_loss = entropy_loss + (-H)
            updates = torch.einsum("bkn,bnd->bkd", attn_n, v)
            slots = self.gru(
                updates.reshape(B * self.num_slots, D),
                prev.reshape(B * self.num_slots, D),
            ).reshape(B, self.num_slots, D)

        # Relational pass
        rs   = self.rel_q(slots)
        rk   = self.rel_k(slots)
        rv   = self.rel_v(slots)
        ra   = torch.einsum("bkd,bjd->bkj", rs, rk) * (self.hidden_dim ** -0.5)
        ra   = ra.softmax(dim=-1)
        rmsg = torch.einsum("bkj,bjd->bkd", ra, rv)
        slots = self.rel_norm(slots + rmsg)
        slots = slots + self.rel_ff(slots)

        return slots, entropy_loss