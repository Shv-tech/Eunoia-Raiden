"""
Eunoia Raiden v2 (恵雷) — 300M Cross-Example Inductive Core
SHV Groups AGI Research Division
core/inductive_core.py

FIX BUG-026: Encoder–decoder split for MCTS encoder caching.
  Added encode() and decode() public methods so MCTS can call encode() once
  per task and decode() once per rollout (12,500×), eliminating ~17 seconds
  of redundant 16-layer encoder computation per evaluation task.
  forward() is kept intact for Phase 1 supervised training.

FIX BUG-014: pair_idx overflow guard.
  Added clamp on pair_idx and [:3] slice on train_pairs so passing more than
  3 pairs cannot exceed pair_emb range (0-3).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class IOSlotEmbedder(nn.Module):
    """
    Projects 512-dim slot tensors to 1024-dim and adds three
    learned positional signals (modal, pair, slot-position).
    pair_idx 0-2 = train pairs, 3 = test input.
    """

    def __init__(self, num_slots: int = 16, hidden_dim: int = 1024,
                 slot_dim: int = 512):
        super().__init__()
        self.up_proj      = nn.Linear(slot_dim, hidden_dim)
        self.modal_emb    = nn.Embedding(2, hidden_dim)
        self.pair_emb     = nn.Embedding(4, hidden_dim)   # 4 slots: pair 0,1,2 + test=3
        self.slot_pos_emb = nn.Parameter(
            torch.randn(1, num_slots, hidden_dim) * 0.02
        )

    def forward(self, slots: torch.Tensor, is_output: bool,
                pair_idx: int) -> torch.Tensor:
        # BUG-014 FIX: clamp pair_idx to valid range
        pair_idx = max(0, min(3, int(pair_idx)))
        x = self.up_proj(slots)
        m = self.modal_emb(torch.tensor(int(is_output), device=x.device))
        p = self.pair_emb(torch.tensor(pair_idx, device=x.device))
        return x + m + p + self.slot_pos_emb


class InductiveCore(nn.Module):
    """
    300M cross-example inductive reasoning core.

    Constructor args:
        vocab_size         : 128  (structured token vocabulary)
        hidden_dim         : 1024
        num_encoder_layers : 16
        num_decoder_layers : 6
        nhead              : 16
        ffn_dim            : 4096
        num_slots          : 16
        slot_dim           : 512  (must match SlotBottleneck hidden_dim)

    Public API:
        forward(train_pairs, test_input, program_prefix) → (logits, value)
            Full pass — used by Phase 1 trainer (needs encoder gradients).

        encode(train_pairs, test_input) → mem
            Encoder-only pass.  Call ONCE per task before MCTS rollouts.

        decode(mem, program_prefix) → (logits, value)
            Decoder-only pass.  Call once per MCTS rollout.
            Requires pre-computed mem from encode().
    """

    def __init__(
        self,
        vocab_size:         int   = 128,
        hidden_dim:         int   = 1024,
        num_encoder_layers: int   = 16,
        num_decoder_layers: int   = 6,
        nhead:              int   = 16,
        ffn_dim:            int   = 4096,
        num_slots:          int   = 16,
        slot_dim:           int   = 512,
        dropout:            float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embedder = IOSlotEmbedder(
            num_slots=num_slots,
            hidden_dim=hidden_dim,
            slot_dim=slot_dim,
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead,
            dim_feedforward=ffn_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_encoder_layers,
            enable_nested_tensor=False,
        )

        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nhead,
            dim_feedforward=ffn_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            dec_layer, num_layers=num_decoder_layers
        )

        self.token_embedding     = nn.Embedding(vocab_size, hidden_dim)
        self.positional_encoding = nn.Parameter(
            torch.randn(1, 256, hidden_dim) * 0.02
        )

        self.policy_head = nn.Linear(hidden_dim, vocab_size)
        self.value_head  = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        # Hybrid direct grid-prediction head
        self.grid_head = nn.Sequential(
            nn.Linear(hidden_dim, 2048),
            nn.SiLU(),
            nn.Linear(2048, 30 * 30 * 10),
        )

        self.sos_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

    # ── Internal helper ───────────────────────────────────────────────────────
    def _build_joint(
        self,
        train_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        test_input:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Concatenate slot embeddings for up to 3 train pairs + test input.
        Returns joint tensor [B, S, hidden_dim].
        BUG-014 FIX: slices to max 3 train pairs so pair_idx never exceeds 2.
        """
        sequence: List[torch.Tensor] = []
        for i, (in_slots, out_slots) in enumerate(train_pairs[:3]):
            pi = min(i, 2)   # clamp: valid pair_emb indices = 0, 1, 2
            sequence.append(self.embedder(in_slots,  is_output=False, pair_idx=pi))
            sequence.append(self.embedder(out_slots, is_output=True,  pair_idx=pi))
        sequence.append(self.embedder(test_input, is_output=False, pair_idx=3))
        return torch.cat(sequence, dim=1)   # [B, 7*num_slots, hidden]

    def _build_tgt(
        self,
        mem: torch.Tensor,
        program_prefix: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B = mem.size(0)
        if program_prefix is None or program_prefix.numel() == 0:
            return self.sos_token.expand(B, -1, -1)
        T   = program_prefix.size(1)
        emb = (self.token_embedding(program_prefix)
               + self.positional_encoding[:, :T, :])
        return torch.cat([self.sos_token.expand(B, -1, -1), emb], dim=1)

    # ── BUG-026 FIX: encode / decode split ───────────────────────────────────

    def encode(
        self,
        train_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        test_input:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the 16-layer encoder once per task.
        Returns mem [B, S, hidden_dim] — reuse this for all MCTS rollouts.

        Call with torch.no_grad() during MCTS search.
        Call with grad enabled during Phase 2 gradient computation.
        """
        joint = self._build_joint(train_pairs, test_input)
        return self.encoder(joint)

    def decode(
        self,
        mem:            torch.Tensor,
        program_prefix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the 6-layer decoder given pre-computed encoder memory.
        Returns (logits [B, T, vocab_size], value [B, 1]).

        This is the method called 12,500× per MCTS search.  The encoder is
        NOT re-run here — pass the mem from encode() above.
        """
        value   = self.value_head(mem.mean(dim=1))
        tgt     = self._build_tgt(mem, program_prefix)
        mask    = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1), device=mem.device
        )
        dec_out = self.decoder(tgt, mem, tgt_mask=mask)
        logits  = self.policy_head(dec_out)
        return logits, value

    # ── Full forward pass (Phase 1 training) ──────────────────────────────────

    def forward(
        self,
        train_pairs:    List[Tuple[torch.Tensor, torch.Tensor]],
        test_input:     torch.Tensor,
        program_prefix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full encode → decode pass.
        Returns (logits [B, T, vocab_size], value [B, 1]).
        Used by Phase 1 trainer which needs gradients through the encoder.
        """
        mem    = self.encode(train_pairs, test_input)
        logits, value = self.decode(mem, program_prefix)
        return logits, value

    # ── Direct grid prediction (hybrid fallback) ──────────────────────────────

    def predict_grid(
        self,
        train_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        test_input:  torch.Tensor,
        out_h: int, out_w: int,
    ) -> torch.Tensor:
        """
        Direct grid prediction fallback (used when MCTS reward < 0.7).
        Returns LongTensor [H, W] of predicted colors.
        """
        mem    = self.encode(train_pairs, test_input)
        # Weight test-input slots more heavily for direct prediction
        global_state = mem.mean(dim=1)   # [B, hidden]
        raw    = self.grid_head(global_state)   # [B, 30*30*10]
        raw    = raw.view(-1, 30, 30, 10)
        colors = raw[:, :out_h, :out_w, :].argmax(dim=-1)   # [B, H, W]
        return colors[0]