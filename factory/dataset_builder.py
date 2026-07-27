"""
Eunoia Raiden (恵雷) — Synthetic Dataset Builder & PyTorch Dataset
SHV Groups AGI Research Division
factory/dataset_builder.py

Three public components:

  DatasetBuilder          — generates N synthetic tasks via multiprocessing
                            and writes sharded Parquet files to disk.

  SyntheticARCDataset     — PyTorch Dataset that reads the Parquet shards
                            and returns one task dict per item.

  build_phase1_dataloader — convenience function that wraps the dataset
                            in a DataLoader ready for Phase1Trainer.
"""

from __future__ import annotations

import os
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader

from factory.reverse_generator import ReverseARCGenerator


# ══════════════════════════════════════════════════════════════════════════════
# PYARROW SCHEMA
# Every task is stored as a flat row of binary blobs (one per grid) plus
# metadata. Grids are serialised as raw uint8 bytes (H*W bytes each).
# ══════════════════════════════════════════════════════════════════════════════
_SCHEMA = pa.schema([
    ("program_bytes",  pa.binary()),
    ("complexity_bin", pa.int8()),
    ("grid_rows",      pa.int8()),
    ("grid_cols",      pa.int8()),
    ("train_in_0",     pa.binary()),
    ("train_out_0",    pa.binary()),
    ("train_in_1",     pa.binary()),
    ("train_out_1",    pa.binary()),
    ("train_in_2",     pa.binary()),
    ("train_out_2",    pa.binary()),
    ("test_in",        pa.binary()),
    ("test_out",       pa.binary()),
])


def _task_to_row(task: Dict) -> Optional[Dict]:
    """Convert a generator task dict to a flat Parquet row."""
    try:
        train_pairs = task["train_pairs"]
        test_pair   = task["test_pair"]
        if len(train_pairs) < 3:
            return None

        in0, out0 = train_pairs[0]
        in1, out1 = train_pairs[1]
        in2, out2 = train_pairs[2]
        t_in, t_out = test_pair

        H, W = in0.shape
        return {
            "program_bytes":  task["program_bytes"],
            "complexity_bin": int(task["complexity_bin"]),
            "grid_rows":      int(H),
            "grid_cols":      int(W),
            "train_in_0":     in0.astype(np.uint8).tobytes(),
            "train_out_0":    out0.astype(np.uint8).tobytes(),
            "train_in_1":     in1.astype(np.uint8).tobytes(),
            "train_out_1":    out1.astype(np.uint8).tobytes(),
            "train_in_2":     in2.astype(np.uint8).tobytes(),
            "train_out_2":    out2.astype(np.uint8).tobytes(),
            "test_in":        t_in.astype(np.uint8).tobytes(),
            "test_out":       t_out.astype(np.uint8).tobytes(),
        }
    except Exception:
        return None


def _worker_fn(chunk_size: int) -> List[Dict]:
    """Generate chunk_size valid tasks in a worker process."""
    gen   = ReverseARCGenerator(use_rust=False)
    tasks = []
    while len(tasks) < chunk_size:
        task = gen.generate_task()
        if task is not None:
            row = _task_to_row(task)
            if row is not None:
                tasks.append(row)
    return tasks


# ══════════════════════════════════════════════════════════════════════════════
# DATASET BUILDER
# ══════════════════════════════════════════════════════════════════════════════
class DatasetBuilder:
    """
    Generates num_tasks synthetic ARC tasks and writes them as
    snappy-compressed Parquet shards under output_dir/.

    Usage:
        DatasetBuilder(num_tasks=1_000_000, num_workers=16).build("data/synthetic")
    """

    def __init__(
        self,
        num_tasks:   int = 1_000_000,
        num_workers: int = 16,
        shard_size:  int = 50_000,
    ):
        self.num_tasks   = num_tasks
        self.num_workers = num_workers
        self.shard_size  = shard_size

    def build(self, output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        print(
            f"[DatasetBuilder] Generating {self.num_tasks:,} tasks "
            f"across {self.num_workers} workers ..."
        )

        chunk_size = max(1, self.num_tasks // self.num_workers)

        # Use 'fork' on Linux (A100 server); fall back to 'spawn' on Windows
        start_method = "fork" if os.name != "nt" else "spawn"
        ctx = mp.get_context(start_method)

        with ctx.Pool(self.num_workers) as pool:
            chunks = pool.map(_worker_fn, [chunk_size] * self.num_workers)

        all_rows = [row for chunk in chunks for row in chunk]
        print(f"[DatasetBuilder] Generated {len(all_rows):,} valid tasks.")

        # Write sharded Parquet files
        shard_idx = 0
        for start in range(0, len(all_rows), self.shard_size):
            batch = all_rows[start : start + self.shard_size]
            table = pa.table(
                {col: [r[col] for r in batch] for col in _SCHEMA.names},
                schema=_SCHEMA,
            )
            shard_path = out / f"shard_{shard_idx:04d}.parquet"
            pq.write_table(table, shard_path, compression="snappy")
            print(f"  Wrote {shard_path}  ({len(batch):,} rows)")
            shard_idx += 1

        print(f"[DatasetBuilder] Done. {shard_idx} shards -> {output_dir}/")

    # Keep old name for backward compatibility
    def build_dataset(self, output_path: str) -> None:
        self.build(output_path)


# ══════════════════════════════════════════════════════════════════════════════
# PYTORCH DATASET
# ══════════════════════════════════════════════════════════════════════════════
class SyntheticARCDataset(Dataset):
    """
    Reads Parquet shards and returns one task dict per item.

    Each item:
        {
            "train_pairs"   : [(in_np, out_np), (in_np, out_np), (in_np, out_np)]
            "test_input"    : np.ndarray uint8 [H, W]
            "program_bytes" : bytes   -- ground-truth DSL bytecode
            "complexity_bin": int     -- 1-5
        }
    """

    def __init__(
        self,
        data_dir:        str,
        complexity_bins: Optional[List[int]] = None,
    ):
        self.data_dir = Path(data_dir)

        parquet_files = sorted(self.data_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No .parquet files found in {data_dir}. "
                "Run DatasetBuilder.build() first."
            )

        self._table = pq.read_table(str(self.data_dir), schema=_SCHEMA)

        # Filter by complexity bin if requested
        if complexity_bins:
            bins_set = set(complexity_bins)
            mask = pa.array(
                [int(b) in bins_set
                 for b in self._table["complexity_bin"].to_pylist()]
            )
            self._table = self._table.filter(mask)

        print(
            f"[SyntheticARCDataset] {len(self._table):,} tasks loaded "
            f"(bins={complexity_bins or 'all'})"
        )

    def __len__(self) -> int:
        return len(self._table)

    def __getitem__(self, idx: int) -> Dict:
        row = {col: self._table[col][idx].as_py() for col in _SCHEMA.names}

        H = int(row["grid_rows"])
        W = int(row["grid_cols"])

        def _to_grid(blob: bytes) -> np.ndarray:
            return np.frombuffer(blob, dtype=np.uint8).reshape(H, W).copy()

        return {
            "train_pairs": [
                (_to_grid(row["train_in_0"]), _to_grid(row["train_out_0"])),
                (_to_grid(row["train_in_1"]), _to_grid(row["train_out_1"])),
                (_to_grid(row["train_in_2"]), _to_grid(row["train_out_2"])),
            ],
            "test_input":     _to_grid(row["test_in"]),
            "program_bytes":  row["program_bytes"],
            "complexity_bin": int(row["complexity_bin"]),
        }


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE DATALOADER — used by Phase1Trainer
# ══════════════════════════════════════════════════════════════════════════════
def build_phase1_dataloader(
    data_dir:        str,
    complexity_bins: Optional[List[int]] = None,
    batch_size:      int  = 1,
    num_workers:     int  = 4,
    shuffle:         bool = True,
) -> DataLoader:
    """
    Build a DataLoader for Phase 1 training.

    batch_size=1 is the correct default because GNN graphs have variable
    numbers of nodes per task. Gradient accumulation in Phase1Trainer
    (grad_accum_steps=512) creates the effective large batch.

    Args:
        data_dir        : path to directory containing .parquet shards
        complexity_bins : bin levels to include e.g. [1, 2]  (None = all)
        batch_size      : keep at 1 unless you implement graph batching
        num_workers     : parallel data loading workers
        shuffle         : shuffle dataset each epoch

    Returns:
        DataLoader yielding one task dict per iteration
    """
    dataset = SyntheticARCDataset(
        data_dir,
        complexity_bins=complexity_bins,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda x: x[0],        # pass single dict directly
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )