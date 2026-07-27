"""
Eunoia Raiden v2 (恵雷) — Phase 2 Trainer (300M)
SHV Groups AGI Research Division
training/phase2_trainer.py

FIX BUG-024: CRITICAL — raw bytecode is NO LONGER cast to token tensors.
  list(best_bc) produced raw byte values 0-255 which blew up nn.Embedding(128)
  on any byte > 127 (e.g. HALT=0xFF=255).  MCTS now returns (bytecode,
  token_ids) and the trainer uses the token_ids directly.

FIX BUG-001: Model interface fixed — gradient forward pass now calls
  model.inductive_core.encode() + decode() with slot tensors instead of
  model(slot_tensors) which expected graph-tuples and would TypeError.

FIX BUG-010: Slot entropy loss is now included in Phase 2 loss.
  Previously it was computed but silently discarded, letting slots collapse
  during REINFORCE fine-tuning.

FIX BUG-022: ShapedReward wired in for W&B / logging.
"""

from __future__ import annotations

import os
import torch
import numpy as np
from torch.optim import AdamW
from typing import Dict, List, Optional, Tuple

from training.config import TrainingConfig
from training.losses import execution_loss_reinforce, value_loss
from engine.mcts_search import MCTSSearch
from engine.sandbox import ExecutionSandbox
from engine.shaped_reward import compute_dense_reward
from core.gnn_perception import GridParser


class Phase2Trainer:
    def __init__(self, model, config: TrainingConfig, arc_loader):
        self.model      = model
        self.cfg        = config.phase2
        self.device     = torch.device(config.device)
        self.use_bf16   = config.use_bfloat16
        self.ckpt_dir   = config.checkpoint_dir
        self.loader     = arc_loader

        self.optimizer = AdamW(
            model.parameters(),
            lr=self.cfg.learning_rate,
            betas=(0.9, 0.95),
        )
        self.baseline    = 0.0
        self.global_step = 0
        self._all_tasks: Optional[List[Dict]] = None

        # Logging accumulators
        self._log_rewards: List[float] = []

    # ── Grid → slot tensor helper ──────────────────────────────────────────
    def _grid_to_slots(
        self, grid: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (slots [1, num_slots, slot_dim], entropy scalar < 0)."""
        nf, ei, et = GridParser.parse(grid, self.device)
        gnn        = self.model.perception(nf, ei, et)
        slots, ent = self.model.bottleneck(gnn.unsqueeze(0))
        return slots, ent

    # ── Single-task processing ─────────────────────────────────────────────
    def _process_task(self, task: dict) -> Tuple[float, Optional[torch.Tensor]]:
        try:
            train_raw = task.get("train", [])
            if len(train_raw) < 3 or not task.get("test"):
                return 0.0, None

            pair_grids = [
                (np.array(p["input"],  dtype=np.uint8),
                 np.array(p["output"], dtype=np.uint8))
                for p in train_raw[:3]
            ]
            test_in = np.array(task["test"][0]["input"], dtype=np.uint8)

            # ── Step 1: pre-compute slots for MCTS (no_grad) ──────────────
            with torch.no_grad():
                proc_pairs_nograd: List[Tuple] = []
                for in_np, out_np in pair_grids:
                    in_s,  _ = self._grid_to_slots(in_np)
                    out_s, _ = self._grid_to_slots(out_np)
                    proc_pairs_nograd.append((in_s, out_s))
                test_s_nograd, _ = self._grid_to_slots(test_in)

                # BUG-026 FIX: encode once, pass cached mem to MCTS
                mem_nograd = self.model.inductive_core.encode(
                    proc_pairs_nograd, test_s_nograd
                )

            # ── Step 2: MCTS search ────────────────────────────────────────
            mcts = MCTSSearch(
                policy_model=self.model,
                train_pair_grids=pair_grids,
                max_rollouts=self.cfg.mcts_rollouts,
                c_puct=self.cfg.c_puct,
                cached_mem=mem_nograd,
            )
            # BUG-024 FIX: unpack (bytecode, token_ids) — never use list(bytes)
            best_bc, best_token_ids = mcts.search(
                proc_pairs_nograd, test_s_nograd
            )

            # ── Step 3: score against all training pairs ───────────────────
            total_r = 0.0
            all_exact = True
            for in_np, out_np in pair_grids:
                with ExecutionSandbox(in_np) as sb:
                    r, exact = sb.score(best_bc, out_np)
                    total_r += r
                    if not exact:
                        all_exact = False
            reward = total_r / len(pair_grids)

            # BUG-022 FIX: structured reward logging
            shaped = compute_dense_reward(reward, all_exact)
            self._log_rewards.append(shaped.total)

            # Update EMA baseline
            a             = self.cfg.baseline_ema_alpha
            self.baseline = a * self.baseline + (1 - a) * reward

            # ── Step 4: forward pass WITH gradients for REINFORCE loss ─────
            # Re-run GNN+slots with grad so encoder/bottleneck are trained.
            proc_pairs_grad: List[Tuple] = []
            total_ent = torch.zeros(1, device=self.device)
            for in_np, out_np in pair_grids:
                in_s,  ie = self._grid_to_slots(in_np)
                out_s, oe = self._grid_to_slots(out_np)
                proc_pairs_grad.append((in_s, out_s))
                total_ent = total_ent + ie + oe
            test_s_grad, te = self._grid_to_slots(test_in)
            total_ent = total_ent + te

            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=self.use_bf16 and self.device.type == "cuda"
            ):
                # BUG-001 FIX: call encode+decode on slot tensors directly.
                mem_grad = self.model.inductive_core.encode(
                    proc_pairs_grad, test_s_grad
                )

                # BUG-024 FIX: use DSL token ids (range 0-127), not raw bytes
                prog_tokens = torch.tensor(
                    [best_token_ids], dtype=torch.long, device=self.device
                )
                prefix  = prog_tokens[:, :-1] if prog_tokens.size(1) > 1 else None
                targets = prog_tokens[:, 1:]

                logits, pred_v = self.model.inductive_core.decode(mem_grad, prefix)

            # Policy log-probs
            lp_all = torch.log_softmax(logits, dim=-1)
            if targets.numel() > 0 and targets.size(1) > 0:
                tidx   = targets[0]
                trange = torch.arange(
                    min(lp_all.size(1), tidx.size(0)), device=self.device
                )
                lp = lp_all[0, trange, tidx[:trange.size(0)]]
            else:
                # Degenerate: single HALT token — use its log-prob
                halt_tok = torch.tensor(
                    [2], dtype=torch.long, device=self.device  # TOKEN_HALT=2
                )
                lp = lp_all[0, 0, halt_tok]

            l_exec = execution_loss_reinforce(lp, reward, self.baseline)
            l_val  = value_loss(
                pred_v,
                torch.tensor([[reward]], device=self.device)
            )
            # BUG-010 FIX: include entropy in Phase 2 loss to prevent slot collapse
            l_ent  = self.cfg.lambda_entropy * (total_ent / 7.0)
            loss   = self.cfg.lambda_e * l_exec + self.cfg.lambda_v * l_val + l_ent

            return reward, loss

        except Exception as e:
            print(f"[Phase2] Task error: {type(e).__name__}: {e}")
            return 0.0, None

    # ── Training step ──────────────────────────────────────────────────────
    def train_step(self) -> float:
        self.model.train()
        if self._all_tasks is None:
            self._all_tasks = self.loader.load_and_augment()

        tasks = self.loader.get_batch(self._all_tasks, self.cfg.tasks_per_step)
        self.optimizer.zero_grad()
        total_r = 0.0
        valid   = 0

        for task in tasks:
            r, loss = self._process_task(task)
            if loss is None:
                continue
            (loss / self.cfg.tasks_per_step).backward()
            total_r += r
            valid   += 1

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_norm)
        self.optimizer.step()
        self.global_step += 1
        mean_r = total_r / max(valid, 1)

        if self.global_step % self.cfg.log_every == 0:
            avg_shaped = (
                sum(self._log_rewards) / len(self._log_rewards)
                if self._log_rewards else 0.0
            )
            print(
                f"[Phase2] Step {self.global_step:>5} | "
                f"Reward={mean_r:.4f} | Baseline={self.baseline:.4f} | "
                f"ShapedAvg={avg_shaped:.4f} | Tasks={valid}/{len(tasks)}"
            )
            self._log_rewards.clear()

        return mean_r

    # ── Checkpointing ──────────────────────────────────────────────────────
    def save_checkpoint(self, tag: str) -> str:
        os.makedirs(self.ckpt_dir, exist_ok=True)
        path = os.path.join(self.ckpt_dir, f"phase2_{tag}.pt")
        torch.save({
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "global_step":     self.global_step,
            "baseline":        self.baseline,
        }, path)
        print(f"[Phase2] Checkpoint -> {path}")
        return path

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.global_step = ckpt.get("global_step", 0)
        self.baseline    = ckpt.get("baseline", 0.0)
        print(f"[Phase2] Loaded <- {path}")