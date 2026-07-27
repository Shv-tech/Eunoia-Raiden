"""
factory/arc_loader.py

FIX BUG-002: Added load_raw() — loads official ARC tasks without augmentation.
  ARCEvaluator called self.loader.load_raw() which did not exist, causing an
  AttributeError at evaluator construction.
"""
from __future__ import annotations
import json
import copy
import random
import numpy as np
from pathlib import Path
from typing import List, Dict


def _augment_grid(grid: np.ndarray, aug: str) -> np.ndarray:
    if aug == "rot90":   return np.rot90(grid, k=1).copy()
    if aug == "rot180":  return np.rot90(grid, k=2).copy()
    if aug == "rot270":  return np.rot90(grid, k=3).copy()
    if aug == "flip_h":  return np.fliplr(grid).copy()
    if aug == "flip_v":  return np.flipud(grid).copy()
    return grid.copy()


def _color_permutation(grid: np.ndarray, perm: List[int]) -> np.ndarray:
    """Apply a consistent color permutation (keeps 0 as background)."""
    out = np.zeros_like(grid)
    for old_c, new_c in enumerate(perm):
        if old_c == 0:
            continue
        out[grid == old_c] = new_c
    out[grid == 0] = 0
    return out


class ARCLoader:
    GEOMETRIC_AUGS = ["rot90", "rot180", "rot270", "flip_h", "flip_v"]

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    # ── BUG-002 FIX ───────────────────────────────────────────────────────────
    def load_raw(self) -> List[Dict]:
        """
        Load all official ARC task JSON files WITHOUT any augmentation.
        Used by ARCEvaluator for clean hold-out scoring.
        Each returned dict has 'train' and 'test' keys per the ARC format.
        """
        raw_tasks: List[Dict] = []
        for f in sorted(self.data_dir.glob("*.json")):
            with open(f) as fh:
                raw_tasks.append(json.load(fh))
        if not raw_tasks:
            print(f"[ARCLoader] WARNING: No .json files found in {self.data_dir}")
        else:
            print(f"[ARCLoader] Loaded {len(raw_tasks)} raw tasks from {self.data_dir}")
        return raw_tasks

    def _apply_aug_to_task(self, task: Dict, aug: str) -> Dict:
        """Apply a geometric augmentation to every grid in a task."""
        def aug_pair(pair):
            return {
                'input':  _augment_grid(np.array(pair['input'],  dtype=np.uint8), aug).tolist(),
                'output': _augment_grid(np.array(pair['output'], dtype=np.uint8), aug).tolist(),
            }
        return {
            'train': [aug_pair(p) for p in task['train']],
            'test':  [aug_pair(p) for p in task['test']],
        }

    def _apply_color_perm_to_task(self, task: Dict, perm: List[int]) -> Dict:
        """Apply a color permutation to every grid in a task."""
        def perm_pair(pair):
            ig = _color_permutation(np.array(pair['input'],  dtype=np.uint8), perm).tolist()
            og = _color_permutation(np.array(pair['output'], dtype=np.uint8), perm).tolist()
            return {'input': ig, 'output': og}
        return {
            'train': [perm_pair(p) for p in task['train']],
            'test':  [perm_pair(p) for p in task['test']],
        }

    def load_and_augment(self) -> List[Dict]:
        """
        Loads all JSON task files and applies 5 geometric + 2 color augmentations
        per task → ~8x expansion.
        """
        raw_tasks = self.load_raw()
        if not raw_tasks:
            return []

        all_tasks = []
        colors = list(range(1, 10))

        for task in raw_tasks:
            all_tasks.append(task)

            for aug in self.GEOMETRIC_AUGS:
                all_tasks.append(self._apply_aug_to_task(task, aug))

            for _ in range(2):
                perm_colors = colors.copy()
                random.shuffle(perm_colors)
                perm = [0] + perm_colors
                all_tasks.append(self._apply_color_perm_to_task(task, perm))

        print(f"[ARCLoader] Augmented: {len(raw_tasks)} tasks → {len(all_tasks)} total.")
        return all_tasks

    def get_batch(self, all_tasks: List[Dict], batch_size: int = 8) -> List[Dict]:
        """Sample a random batch from pre-loaded tasks."""
        return random.sample(all_tasks, min(batch_size, len(all_tasks)))