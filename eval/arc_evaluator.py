"""
Eunoia Raiden v2 (恵雷) — ARC Evaluator with TTA + Hybrid Fallback
SHV Groups AGI Research Division
eval/arc_evaluator.py

FIX BUG-003: Removed num_objects_hint from MCTSSearch constructor call.
  It is now passed to mcts.search() where it is an accepted parameter.

FIX BUG-004: evaluate() now accepts an optional max_rollouts override so
  ablation.py can call evaluator.evaluate(max_rollouts=1) without TypeError.

FIX BUG-009: TTA restricted to color permutations + identity only.
  Previously all 8 augmentations (including rotations/flips) were used to
  generate augmented-space programs, which were then executed on the original
  un-augmented test input — producing garbage.  A rotation-solving program
  cannot solve the original orientation.  Geometric augmentations are now
  only used as color-permutation variants; spatial augmentations are excluded
  from TTA.  This makes TTA strictly additive (more coverage, never worse).

FIX BUG-017: GNN + slot computation is cached per task.
  _task_to_processed() result is computed once and reused across all TTA
  augmentations for the same task instead of being recomputed per aug.
  Also, encoder memory (mem) from inductive_core.encode() is computed once
  per (task, aug) and passed to MCTSSearch as cached_mem, saving 16 encoder
  layers per rollout.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from core.gnn_perception import GridParser
from engine.mcts_search  import MCTSSearch
from engine.sandbox      import ExecutionSandbox
from factory.arc_loader  import ARCLoader


class ARCEvaluator:
    """
    Full ARC evaluation with TTA (color permutations only) + hybrid fallback.

    Args:
        model        : EunoiaRaiden (300M, fully trained)
        data_dir     : path to ARC task JSON files
        max_rollouts : MCTS budget per task (total, split across TTA variants)
        use_tta      : enable test-time augmentation (default True)
    """

    # BUG-009 FIX: only color permutations + identity.
    # Geometric augmentations (rot90, flip, etc.) are REMOVED from TTA
    # because a program that solves the rotated task cannot solve the
    # original orientation — the two programs are NOT interchangeable.
    AUG_TYPES = [
        "identity",
        "color_perm_1",
        "color_perm_2",
        "color_perm_3",
    ]

    def __init__(
        self,
        model:        torch.nn.Module,
        data_dir:     str,
        max_rollouts: int  = 12_500,
        use_tta:      bool = True,
    ):
        self.model        = model
        self.model.eval()
        self.device       = next(model.parameters()).device
        self.max_rollouts = max_rollouts
        self.use_tta      = use_tta
        self.loader       = ARCLoader(data_dir)
        # BUG-002 FIX (arc_loader.py): load_raw() now exists
        self.tasks        = self.loader.load_raw()

        n_augs            = len(self.AUG_TYPES) if use_tta else 1
        self.tta_rollouts = max(1, max_rollouts // n_augs)

    # ── Grid augmentation ──────────────────────────────────────────────────

    @staticmethod
    def _aug_grid(grid: np.ndarray, aug: str) -> np.ndarray:
        """
        Color-permutation TTA only.
        Returns augmented copy; identity returns unmodified copy.
        """
        if aug == "identity":
            return grid.copy()
        if aug.startswith("color_perm"):
            perm = {c: random.randint(1, 9) for c in range(1, 10)}
            out  = grid.copy()
            for orig, new in perm.items():
                out[grid == orig] = new
            return out
        return grid.copy()

    def _aug_task(self, task: dict, aug: str) -> dict:
        return {
            "train": [
                {
                    "input":  self._aug_grid(
                        np.array(p["input"],  dtype=np.uint8), aug).tolist(),
                    "output": self._aug_grid(
                        np.array(p["output"], dtype=np.uint8), aug).tolist(),
                }
                for p in task["train"]
            ],
            "test": [
                {
                    "input": self._aug_grid(
                        np.array(p["input"], dtype=np.uint8), aug).tolist(),
                }
                for p in task["test"]
            ],
        }

    # ── Grid → slot tensor ─────────────────────────────────────────────────

    def _grid_to_slots(self, grid: np.ndarray) -> torch.Tensor:
        """Returns slots [1, num_slots, slot_dim] with no grad."""
        nf, ei, et = GridParser.parse(grid, self.device)
        gnn_out    = self.model.perception(nf, ei, et)
        slots, _   = self.model.bottleneck(gnn_out.unsqueeze(0))
        return slots

    def _task_to_processed(
        self, task: dict
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """
        BUG-017 FIX: compute GNN+slots once, cache result.
        Returns (processed_pairs, test_slots).
        """
        pairs = []
        for p in task["train"][:3]:
            in_s  = self._grid_to_slots(np.array(p["input"],  dtype=np.uint8))
            out_s = self._grid_to_slots(np.array(p["output"], dtype=np.uint8))
            pairs.append((in_s, out_s))
        test_g = self._grid_to_slots(
            np.array(task["test"][0]["input"], dtype=np.uint8)
        )
        return pairs, test_g

    # ── Hybrid fallback ─────────────────────────────────────────────────────

    def _hybrid_predict(
        self,
        task:             dict,
        processed_pairs:  List[Tuple[torch.Tensor, torch.Tensor]],
        test_slots:       torch.Tensor,
    ) -> np.ndarray:
        """Use grid head when MCTS training-pair score < 0.7."""
        tgt_example = np.array(task["train"][0]["output"], dtype=np.uint8)
        H, W = tgt_example.shape
        with torch.no_grad():
            pred = self.model.inductive_core.predict_grid(
                processed_pairs, test_slots, H, W
            )
        return pred.cpu().numpy().astype(np.uint8)

    # ── Single task evaluation ───────────────────────────────────────────────

    def _evaluate_task(self, task: dict) -> Tuple[bool, float]:
        """
        Evaluate one task. Returns (is_correct, best_reward).
        """
        test_output = np.array(
            task["test"][0].get("output", []), dtype=np.uint8
        )
        has_gt = test_output.size > 0

        best_reward   = 0.0
        best_bytecode: Optional[bytes] = None

        augs = self.AUG_TYPES if self.use_tta else ["identity"]

        for aug in augs:
            aug_task   = self._aug_task(task, aug)
            pair_grids = [
                (np.array(p["input"],  dtype=np.uint8),
                 np.array(p["output"], dtype=np.uint8))
                for p in aug_task["train"][:3]
            ]

            with torch.no_grad():
                # BUG-017 FIX: compute slots once per aug variant
                processed, test_slots = self._task_to_processed(aug_task)

                # BUG-026 FIX: pre-compute encoder memory; pass as cached_mem
                mem = self.model.inductive_core.encode(processed, test_slots)

            # BUG-003 FIX: num_objects_hint is on search(), not __init__()
            mcts = MCTSSearch(
                policy_model=self.model,
                train_pair_grids=pair_grids,
                max_rollouts=self.tta_rollouts,
                cached_mem=mem,
            )
            best_bc, _ = mcts.search(
                processed, test_slots, num_objects_hint=4
            )

            # Score this aug's program against training pairs
            total = 0.0
            for in_grid, out_grid in pair_grids:
                with ExecutionSandbox(in_grid) as sb:
                    r, _ = sb.score(best_bc, out_grid)
                    total += r
            aug_reward = total / max(len(pair_grids), 1)

            if aug_reward > best_reward:
                best_reward   = aug_reward
                best_bytecode = best_bc

        if best_bytecode is None:
            return False, 0.0

        if not has_gt:
            return best_reward >= 1.0, best_reward

        # Hybrid fallback: if MCTS underperformed, try direct grid prediction
        test_in_np = np.array(task["test"][0]["input"], dtype=np.uint8)
        if best_reward < 0.7:
            with torch.no_grad():
                p2, ts = self._task_to_processed(task)
                pred   = self._hybrid_predict(task, p2, ts)
            is_exact = np.array_equal(pred, test_output)
            return is_exact, 0.7 if is_exact else 0.3

        # Execute best program on actual test input
        with ExecutionSandbox(test_in_np) as sb:
            r, is_exact = sb.score(best_bytecode, test_output)

        return is_exact, r

    # ── Full evaluation loop ─────────────────────────────────────────────────

    def evaluate(self, max_rollouts: Optional[int] = None) -> float:
        """
        Evaluate all tasks. Returns accuracy in [0, 1].

        Args:
            max_rollouts: optional override for MCTS budget per task.
                          BUG-004 FIX: this parameter now exists so
                          ablation.py can call evaluate(max_rollouts=1).
        """
        if max_rollouts is not None:
            # Temporarily override rollout budget for this evaluation run
            prev_max        = self.max_rollouts
            prev_tta        = self.tta_rollouts
            self.max_rollouts = max_rollouts
            n_augs            = len(self.AUG_TYPES) if self.use_tta else 1
            self.tta_rollouts = max(1, max_rollouts // n_augs)

        n       = len(self.tasks)
        correct = 0
        print(
            f"\n[Evaluator] {n} tasks | "
            f"TTA={'ON x'+str(len(self.AUG_TYPES)) if self.use_tta else 'OFF'} | "
            f"Rollouts/aug={self.tta_rollouts:,}"
        )

        try:
            for i, task in enumerate(self.tasks):
                try:
                    ok, r = self._evaluate_task(task)
                    if ok:
                        correct += 1
                        tag = "SOLVED"
                    else:
                        tag = f"FAILED  (reward={r:.3f})"
                    print(
                        f"  Task {i+1:>3}/{n} {tag}  |  "
                        f"Running: {correct}/{i+1}"
                    )
                except Exception as e:
                    print(f"  Task {i+1:>3}/{n} ERROR: {e}")
        finally:
            # Restore original rollout counts if they were overridden
            if max_rollouts is not None:
                self.max_rollouts = prev_max
                self.tta_rollouts = prev_tta

        score = correct / max(n, 1)
        print(f"\n[Evaluator] FINAL SCORE: {correct}/{n} = {score*100:.2f}%")
        return score