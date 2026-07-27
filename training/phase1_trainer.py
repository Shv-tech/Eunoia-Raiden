"""
Eunoia Raiden v2 (恵雷) — Phase 1 Trainer (300M)
SHV Groups AGI Research Division
training/phase1_trainer.py

FIX BUG-015: _task_to_inputs now handles both task-dict formats:
  - SyntheticARCDataset: task["train_pairs"], task["test_input"]
  - ARCLoader raw format: task["train"][i]["input"], task["test"][0]["input"]
  Mixing the two no longer silently KeyErrors.
"""

from __future__ import annotations

import os
import torch
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from training.config import TrainingConfig
from training.losses import policy_loss
from core.gnn_perception import GridParser


class Phase1Trainer:
    def __init__(self, model, config: TrainingConfig, dataloader):
        self.model      = model
        self.cfg        = config.phase1
        self.device     = torch.device(config.device)
        self.use_bf16   = config.use_bfloat16
        self.ckpt_dir   = config.checkpoint_dir
        self.dataloader = dataloader

        self.optimizer = AdamW(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        steps_per_epoch   = max(1, len(dataloader) // self.cfg.grad_accum_steps)
        total_optim_steps = self.cfg.epochs * steps_per_epoch

        warmup = LinearLR(self.optimizer, start_factor=1e-6, end_factor=1.0,
                          total_iters=self.cfg.warmup_steps)
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, total_optim_steps - self.cfg.warmup_steps),
            eta_min=self.cfg.lr_min,
        )
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup, cosine],
            milestones=[self.cfg.warmup_steps],
        )
        self.global_step = 0

    def _lambda_entropy(self, step: int, total: int) -> float:
        p = min(1.0, step / max(total, 1))
        return (self.cfg.lambda_entropy_start
                - p * (self.cfg.lambda_entropy_start - self.cfg.lambda_entropy_end))

    # BUG-015 FIX: support both SyntheticARCDataset and ARCLoader task dicts
    def _task_to_inputs(self, task: dict):
        # ── Determine format ────────────────────────────────────────────────
        if "train_pairs" in task:
            # SyntheticARCDataset format: list of (in_np, out_np)
            raw_pairs = task["train_pairs"]
            train_pairs_np = [(in_np, out_np) for in_np, out_np in raw_pairs]
        elif "train" in task:
            # ARCLoader raw JSON format
            train_pairs_np = [
                (np.array(p["input"],  dtype=np.uint8),
                 np.array(p["output"], dtype=np.uint8))
                for p in task["train"][:3]
            ]
        else:
            raise ValueError("Task dict has neither 'train_pairs' nor 'train' key")

        if "test_input" in task:
            test_np = task["test_input"]
            if not isinstance(test_np, np.ndarray):
                test_np = np.array(test_np, dtype=np.uint8)
        elif "test" in task and task["test"]:
            test_np = np.array(task["test"][0]["input"], dtype=np.uint8)
        else:
            raise ValueError("Task dict has neither 'test_input' nor 'test' key")

        # ── Convert to graph tuples ─────────────────────────────────────────
        train_pairs = []
        for in_np, out_np in train_pairs_np:
            in_g  = GridParser.parse(np.asarray(in_np,  dtype=np.uint8), self.device)
            out_g = GridParser.parse(np.asarray(out_np, dtype=np.uint8), self.device)
            train_pairs.append((in_g, out_g))

        test_g = GridParser.parse(test_np, self.device)

        # ── Program token ids ────────────────────────────────────────────────
        prog = task["program_bytes"]
        # program_bytes contains raw DSL bytes; we need the opcode token sequence.
        # Phase1 uses supervised learning on token sequences so we must use
        # the token vocabulary (not raw byte values) to avoid Embedding index OOB.
        # We reconstruct token ids from the raw bytes by matching opcode bytes.
        from dsl.primitives import OPCODE_ARG_BYTES, TOKEN_SOS, OPCODE_TO_TOKEN, OpCode
        token_ids = [TOKEN_SOS]
        i = 0
        while i < len(prog):
            byte = prog[i]
            try:
                opcode = OpCode(byte)
            except ValueError:
                i += 1
                continue
            token_ids.append(OPCODE_TO_TOKEN.get(opcode, 2))
            n_args = OPCODE_ARG_BYTES.get(opcode, 0)
            i += 1 + n_args
            if opcode == OpCode.HALT:
                break

        tokens = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        return train_pairs, test_g, tokens

    def train_epoch(self, epoch_idx: int):
        self.model.train()
        n_grids = 7   # 3 in + 3 out + 1 test

        steps_per_epoch   = max(1, len(self.dataloader) // self.cfg.grad_accum_steps)
        total_optim_steps = self.cfg.epochs * steps_per_epoch

        self.optimizer.zero_grad()
        accum = 0

        for step, task in enumerate(self.dataloader):
            gs  = self.global_step
            lam = self._lambda_entropy(gs, total_optim_steps)

            try:
                pairs, test_g, tokens = self._task_to_inputs(task)
            except Exception as e:
                print(f"[Phase1] Skip malformed task: {e}")
                continue

            if tokens.size(1) < 2:
                continue

            prefix  = tokens[:, :-1]
            targets = tokens[:,  1:]

            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=self.use_bf16 and self.device.type == "cuda"
            ):
                logits, _val, entropy = self.model(
                    pairs, test_g, program_prefix=prefix
                )
                l_pol = policy_loss(logits, targets)
                l_ent = entropy / n_grids
                loss  = (l_pol + lam * l_ent) / self.cfg.grad_accum_steps

            loss.backward()
            accum += 1

            if accum >= self.cfg.grad_accum_steps:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                accum = 0

                if self.global_step % self.cfg.log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    print(
                        f"[Phase1] E{epoch_idx} S{self.global_step:>6} | "
                        f"Loss={loss.item()*self.cfg.grad_accum_steps:.4f} | "
                        f"λ={lam:.5f} | LR={lr:.2e}"
                    )

        if accum > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.max_norm
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.global_step += 1

    def save_checkpoint(self, tag: str) -> str:
        os.makedirs(self.ckpt_dir, exist_ok=True)
        path = os.path.join(self.ckpt_dir, f"phase1_{tag}.pt")
        torch.save({
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "global_step":     self.global_step,
        }, path)
        print(f"[Phase1] Checkpoint -> {path}")
        return path

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.global_step = ckpt.get("global_step", 0)
        print(f"[Phase1] Loaded <- {path}")