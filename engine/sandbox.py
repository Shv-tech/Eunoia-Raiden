"""
Eunoia Raiden (恵雷) — Execution Sandbox (Python wrapper)
SHV Groups AGI Research Division
engine/sandbox.py

Thin Python wrapper around the Rust FFI functions.
Each ExecutionSandbox holds exactly one Rust task handle.
Always use as a context manager to guarantee handle release.

No bugs found in audit — included verbatim for completeness.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


class ExecutionSandbox:
    """
    Opens a Rust task handle for a given input grid.
    Releases the handle when .close() or __exit__ is called.

    Usage:
        with ExecutionSandbox(input_grid) as sb:
            reward, exact = sb.score(program_bytes, target_grid)
    """

    def __init__(self, input_grid: np.ndarray):
        try:
            import insgr_rust as _rust  # type: ignore
            self._rust = _rust
        except ImportError:
            raise RuntimeError(
                "insgr_rust.so not found. Compile first with:  python dsl/build.py"
            )

        grid = np.asarray(input_grid, dtype=np.uint8)
        if grid.ndim != 2:
            raise ValueError(f"Expected 2-D grid, got shape {grid.shape}")

        rows, cols   = grid.shape
        self._handle = self._rust.parse_task(grid.tobytes(), rows, cols)
        self._active = True

    def score(
        self,
        program_bytes: bytes,
        target_grid:   np.ndarray,
    ) -> Tuple[float, bool]:
        """
        Execute program_bytes on the stored input grid and score vs target.

        Returns:
            reward   : float in [0.0, 1.0]
            is_exact : bool — True iff every cell matches exactly
        """
        if not self._active:
            raise RuntimeError(
                "Attempted to use a released ExecutionSandbox. "
                "Was .close() called prematurely?"
            )
        tgt = np.asarray(target_grid, dtype=np.uint8)
        if tgt.ndim != 2:
            raise ValueError(f"Expected 2-D target grid, got shape {tgt.shape}")
        tr, tc = tgt.shape
        return self._rust.execute_and_score(
            self._handle,
            program_bytes,
            tgt.tobytes(),
            tr,
            tc,
        )

    def close(self) -> None:
        if self._active:
            self._rust.release_task(self._handle)
            self._active = False

    def __enter__(self) -> "ExecutionSandbox":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        if getattr(self, "_active", False):
            try:
                self._rust.release_task(self._handle)
            except Exception:
                pass
            self._active = False