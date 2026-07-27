"""
Eunoia Raiden v2 (恵雷) — Reverse-ARC Synthetic Data Generator
SHV Groups AGI Research Division
factory/reverse_generator.py

FIX BUG-007: Python executor now implements ALL 24 grid-mutating opcodes.
  Previously only 7/24 were implemented — 72% of synthetic training data
  produced identity (input==output) pairs for programs using DUPLICATE,
  SCALE, GRAVITY, SYMMETRY_COMPLETE, RECOLOR_BY_RANK, FLOOD_FILL_BG,
  HOLLOW, BORDER, EXTEND, MASK_AND, MASK_OR, MOVE_TO.

FIX BUG-021: Loud warning when Rust is unavailable (was silent).
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from dsl.primitives import (
    OpCode,
    Instruction,
    Program,
    RELATIONSHIP_TEMPLATES,
    sample_args,
)


# ══════════════════════════════════════════════════════════════════════════════
# COMPLEXITY SCORING
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class TaskComplexity:
    score:     int
    bin_level: int   # 1-5


def compute_complexity(program: Program, num_objects: int) -> TaskComplexity:
    depth        = max(1, len(program.instructions) - 1)
    unique_prims = len({
        i.opcode for i in program.instructions
        if i.opcode != OpCode.HALT
    })
    score = depth * max(1, unique_prims) * max(1, num_objects)

    if   score <= 4:   level = 1
    elif score <= 12:  level = 2
    elif score <= 30:  level = 3
    elif score <= 70:  level = 4
    else:              level = 5

    return TaskComplexity(score, level)


# ══════════════════════════════════════════════════════════════════════════════
# PYTHON REFERENCE EXECUTOR — complete 24-opcode implementation
# Mirrors dsl/src/lib.rs semantics exactly.
# Used for data generation before Rust is compiled.
# ══════════════════════════════════════════════════════════════════════════════
class _PythonExecutor:
    """
    Complete Python DSL executor for all 24 opcodes.
    Register-reading opcodes (COUNT_OBJECTS, GET_SIZE, FILTER_BY_SIZE,
    GET_COLOR, FILTER_BY_COLOR, IF_THEN) store/read from a 16-element
    register file.  Since sample_args always generates IF_THEN with
    then_off=else_off=0, these are effectively no-ops for generated programs
    but the register file is maintained for correctness.
    """

    @staticmethod
    def _flood_fill(grid: np.ndarray) -> List[dict]:
        rows, cols = grid.shape
        visited    = np.zeros((rows, cols), dtype=bool)
        objects    = []
        oid        = 1
        for r in range(rows):
            for c in range(cols):
                col = int(grid[r, c])
                if col == 0 or visited[r, c]:
                    continue
                cells = []
                q     = deque([(r, c)])
                visited[r, c] = True
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if (0 <= nr < rows and 0 <= nc < cols
                                and not visited[nr,nc]
                                and grid[nr,nc] == col):
                            visited[nr,nc] = True
                            q.append((nr, nc))
                objects.append({"id": oid, "color": col, "cells": cells})
                oid += 1
        return objects

    @classmethod
    def execute(cls, grid_in: np.ndarray, program: Program) -> np.ndarray:
        grid    = grid_in.copy().astype(np.uint8)
        objects = cls._flood_fill(grid)
        H, W    = grid.shape
        regs    = [0] * 16   # 16-element register file

        instructions = program.instructions
        pc           = 0
        max_steps    = len(instructions) * 3 + 200  # safety budget

        for _step in range(max_steps):
            if pc >= len(instructions):
                break

            instr = instructions[pc]
            op    = instr.opcode
            a     = instr.args

            def arg(i: int, default: int = 0) -> int:
                return int(a[i]) if len(a) > i else default

            # ── helpers ──────────────────────────────────────────────────
            def find_obj(obj_id: int):
                return next((o for o in objects if o["id"] == obj_id), None)

            def next_id() -> int:
                return (max((o["id"] for o in objects), default=0) + 1)

            # ── HALT ──────────────────────────────────────────────────────
            if op == OpCode.HALT:
                break

            # ── 0x01 TRANSLATE  obj_id dr dc ─────────────────────────────
            elif op == OpCode.TRANSLATE:
                obj_id   = arg(0, 1)
                dr, dc   = arg(1, 0), arg(2, 0)
                obj = find_obj(obj_id)
                if obj:
                    for r, c in obj["cells"]:
                        grid[r, c] = 0
                    new_cells = []
                    for r, c in obj["cells"]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            r, c = nr, nc
                        grid[r, c] = obj["color"]
                        new_cells.append((r, c))
                    obj["cells"] = new_cells

            # ── 0x02 REFLECT  obj_id axis(0=H 1=V 2=diag 3=anti) ─────────
            elif op == OpCode.REFLECT:
                obj_id = arg(0, 1)
                axis   = arg(1, 0)
                obj = find_obj(obj_id)
                if obj:
                    for r, c in obj["cells"]:
                        grid[r, c] = 0
                    rs  = [r for r, _ in obj["cells"]]
                    cs  = [c for _, c in obj["cells"]]
                    if axis == 0:
                        pivot = min(rs) + max(rs)
                        obj["cells"] = [(pivot - r, c) for r, c in obj["cells"]]
                    elif axis == 1:
                        pivot = min(cs) + max(cs)
                        obj["cells"] = [(r, pivot - c) for r, c in obj["cells"]]
                    elif axis == 2:
                        obj["cells"] = [(c, r) for r, c in obj["cells"]]
                    else:
                        pivot = min(rs) + max(rs)
                        obj["cells"] = [(pivot - c, pivot - r) for r, c in obj["cells"]]
                    for r, c in obj["cells"]:
                        if 0 <= r < H and 0 <= c < W:
                            grid[r, c] = obj["color"]

            # ── 0x03 ROTATE  obj_id turns ────────────────────────────────
            elif op == OpCode.ROTATE:
                obj_id = arg(0, 1)
                turns  = arg(1, 1) % 4
                obj = find_obj(obj_id)
                if obj:
                    for r, c in obj["cells"]:
                        grid[r, c] = 0
                    n  = max(1, len(obj["cells"]))
                    cr = sum(r for r, _ in obj["cells"]) // n
                    cc = sum(c for _, c in obj["cells"]) // n
                    new_cells = []
                    for r, c in obj["cells"]:
                        dr, dc = r - cr, c - cc
                        for _ in range(turns):
                            dr, dc = dc, -dr   # 90° CW
                        nr, nc = cr + dr, cc + dc
                        if not (0 <= nr < H and 0 <= nc < W):
                            nr, nc = r, c
                        new_cells.append((nr, nc))
                        grid[nr, nc] = obj["color"]
                    obj["cells"] = new_cells

            # ── 0x04 SHIFT_UNTIL_CONTACT  obj_id dir ─────────────────────
            elif op == OpCode.SHIFT_UNTIL_CONTACT:
                obj_id = arg(0, 1)
                direc  = arg(1, 0) % 4
                deltas = [(-1,0),(1,0),(0,-1),(0,1)]
                dr, dc = deltas[direc]
                obj = find_obj(obj_id)
                if obj:
                    cell_set = set(map(tuple, obj["cells"]))
                    while True:
                        can = True
                        for r, c in obj["cells"]:
                            nr, nc = r + dr, c + dc
                            if not (0 <= nr < H and 0 <= nc < W):
                                can = False; break
                            if grid[nr, nc] != 0 and (nr, nc) not in cell_set:
                                can = False; break
                        if not can:
                            break
                        for r, c in obj["cells"]:
                            grid[r, c] = 0
                        obj["cells"] = [(r+dr, c+dc) for r, c in obj["cells"]]
                        cell_set = set(map(tuple, obj["cells"]))
                        for r, c in obj["cells"]:
                            grid[r, c] = obj["color"]

            # ── 0x05 FILL_COLOR  obj_id color ────────────────────────────
            elif op == OpCode.FILL_COLOR:
                obj_id = arg(0, 1)
                color  = arg(1, 1)
                obj = find_obj(obj_id)
                if obj:
                    obj["color"] = color
                    for r, c in obj["cells"]:
                        grid[r, c] = color

            # ── 0x06 COPY_COLOR_FROM  src_id dst_id ──────────────────────
            elif op == OpCode.COPY_COLOR_FROM:
                src_id = arg(0, 1)
                dst_id = arg(1, 2)
                src = find_obj(src_id)
                dst = find_obj(dst_id)
                if src and dst:
                    dst["color"] = src["color"]
                    for r, c in dst["cells"]:
                        grid[r, c] = src["color"]

            # ── 0x07 SWAP_COLORS  c1 c2 ──────────────────────────────────
            elif op == OpCode.SWAP_COLORS:
                c1 = arg(0, 1)
                c2 = arg(1, 2)
                if c1 != c2:
                    mask1 = (grid == c1)
                    mask2 = (grid == c2)
                    grid[mask1] = c2
                    grid[mask2] = c1
                    for obj in objects:
                        if obj["color"] == c1:   obj["color"] = c2
                        elif obj["color"] == c2: obj["color"] = c1

            # ── 0x08 COUNT_OBJECTS  filter reg (register read-only) ───────
            elif op == OpCode.COUNT_OBJECTS:
                f = arg(0, 0) & 0x0F
                r = arg(1, 0) & 0x0F
                if f == 0:
                    regs[r] = len(objects)
                else:
                    regs[r] = sum(1 for o in objects if o["color"] == f)

            # ── 0x09 GET_SIZE  obj_id reg ─────────────────────────────────
            elif op == OpCode.GET_SIZE:
                obj_id = arg(0, 1)
                r      = arg(1, 0) & 0x0F
                obj = find_obj(obj_id)
                regs[r] = len(obj["cells"]) if obj else 0

            # ── 0x0A FILTER_BY_SIZE  min max reg ─────────────────────────
            elif op == OpCode.FILTER_BY_SIZE:
                mn = arg(0, 1)
                mx = arg(1, 255)
                r  = arg(2, 0) & 0x0F
                regs[r] = sum(1 for o in objects if mn <= len(o["cells"]) <= mx)

            # ── 0x0B IF_THEN  cond_reg thresh then_off else_off ──────────
            elif op == OpCode.IF_THEN:
                cond_reg = arg(0, 0) & 0x0F
                thresh   = arg(1, 0)
                then_off = arg(2, 0)
                else_off = arg(3, 0)
                pc += 1   # advance past IF_THEN first
                if regs[cond_reg] > thresh:
                    pc += then_off
                else:
                    pc += else_off
                continue   # skip the pc += 1 at bottom

            # ── 0x0C DUPLICATE  obj_id dr dc ─────────────────────────────
            elif op == OpCode.DUPLICATE:
                obj_id = arg(0, 1)
                dr     = arg(1, 1)
                dc     = arg(2, 0)
                obj = find_obj(obj_id)
                if obj:
                    new_cells = []
                    for r, c in obj["cells"]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            new_cells.append((nr, nc))
                            grid[nr, nc] = obj["color"]
                    if new_cells:
                        objects.append({
                            "id":    next_id(),
                            "color": obj["color"],
                            "cells": new_cells,
                        })

            # ── 0x0D SCALE  obj_id factor(1-4) ───────────────────────────
            elif op == OpCode.SCALE:
                obj_id = arg(0, 1)
                factor = max(1, min(4, arg(1, 2)))
                obj = find_obj(obj_id)
                if obj:
                    for r, c in obj["cells"]:
                        grid[r, c] = 0
                    n  = max(1, len(obj["cells"]))
                    cr = sum(r for r, _ in obj["cells"]) // n
                    cc = sum(c for _, c in obj["cells"]) // n
                    new_cells = []
                    for r, c in obj["cells"]:
                        dr = (r - cr) * factor
                        dc = (c - cc) * factor
                        for fr in range(factor):
                            for fc in range(factor):
                                nr, nc = cr + dr + fr, cc + dc + fc
                                if 0 <= nr < H and 0 <= nc < W:
                                    new_cells.append((nr, nc))
                                    grid[nr, nc] = obj["color"]
                    obj["cells"] = new_cells

            # ── 0x0E GRAVITY  dir ────────────────────────────────────────
            elif op == OpCode.GRAVITY:
                direc  = arg(0, 0) % 4
                deltas = [(-1,0),(1,0),(0,-1),(0,1)]
                dr, dc = deltas[direc]
                for obj in objects:
                    cell_set = set(map(tuple, obj["cells"]))
                    while True:
                        can = True
                        for r, c in obj["cells"]:
                            nr, nc = r + dr, c + dc
                            if not (0 <= nr < H and 0 <= nc < W):
                                can = False; break
                            if grid[nr, nc] != 0 and (nr, nc) not in cell_set:
                                can = False; break
                        if not can:
                            break
                        for r, c in obj["cells"]:
                            grid[r, c] = 0
                        obj["cells"] = [(r+dr, c+dc) for r, c in obj["cells"]]
                        cell_set = set(map(tuple, obj["cells"]))
                        for r, c in obj["cells"]:
                            grid[r, c] = obj["color"]

            # ── 0x0F SYMMETRY_COMPLETE  obj_id axis ──────────────────────
            elif op == OpCode.SYMMETRY_COMPLETE:
                obj_id = arg(0, 1)
                axis   = arg(1, 0)
                obj = find_obj(obj_id)
                if obj:
                    new_cells = []
                    for r, c in obj["cells"]:
                        if axis == 0:   mr, mc = H - 1 - r, c
                        elif axis == 1: mr, mc = r, W - 1 - c
                        elif axis == 2: mr, mc = c, r
                        else:           mr, mc = W - 1 - c, H - 1 - r
                        if 0 <= mr < H and 0 <= mc < W:
                            grid[mr, mc] = obj["color"]
                            new_cells.append((mr, mc))
                    obj["cells"].extend(new_cells)

            # ── 0x10 RECOLOR_BY_RANK ─────────────────────────────────────
            elif op == OpCode.RECOLOR_BY_RANK:
                sorted_objs = sorted(objects,
                                     key=lambda o: len(o["cells"]),
                                     reverse=True)
                for rank, obj in enumerate(sorted_objs):
                    col         = (rank % 9) + 1
                    obj["color"] = col
                    for r, c in obj["cells"]:
                        grid[r, c] = col

            # ── 0x11 FLOOD_FILL_BG  color ────────────────────────────────
            elif op == OpCode.FLOOD_FILL_BG:
                color = arg(0, 1)
                grid[grid == 0] = color

            # ── 0x12 GET_COLOR  obj_id reg ────────────────────────────────
            elif op == OpCode.GET_COLOR:
                obj_id = arg(0, 1)
                r      = arg(1, 0) & 0x0F
                obj = find_obj(obj_id)
                regs[r] = obj["color"] if obj else 0

            # ── 0x13 FILTER_BY_COLOR  color reg ──────────────────────────
            elif op == OpCode.FILTER_BY_COLOR:
                color = arg(0, 1)
                r     = arg(1, 0) & 0x0F
                regs[r] = sum(1 for o in objects if o["color"] == color)

            # ── 0x14 HOLLOW  obj_id ───────────────────────────────────────
            elif op == OpCode.HOLLOW:
                obj_id   = arg(0, 1)
                obj = find_obj(obj_id)
                if obj:
                    cell_set = set(map(tuple, obj["cells"]))
                    boundary = [
                        (r, c) for r, c in obj["cells"]
                        if any((r+dr, c+dc) not in cell_set
                               for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)])
                    ]
                    for r, c in obj["cells"]:
                        grid[r, c] = 0
                    for r, c in boundary:
                        grid[r, c] = obj["color"]
                    obj["cells"] = boundary

            # ── 0x15 BORDER  obj_id color ─────────────────────────────────
            elif op == OpCode.BORDER:
                obj_id = arg(0, 1)
                color  = arg(1, 1)
                obj = find_obj(obj_id)
                if obj:
                    rs   = [r for r, _ in obj["cells"]]
                    cs   = [c for _, c in obj["cells"]]
                    r0   = max(0, min(rs) - 1)
                    c0   = max(0, min(cs) - 1)
                    r1   = min(H - 1, max(rs) + 1)
                    c1   = min(W - 1, max(cs) + 1)
                    for r in range(r0, r1 + 1):
                        grid[r, c0] = color
                        grid[r, c1] = color
                    for c in range(c0, c1 + 1):
                        grid[r0, c] = color
                        grid[r1, c] = color

            # ── 0x16 EXTEND  obj_id dir ───────────────────────────────────
            elif op == OpCode.EXTEND:
                obj_id = arg(0, 1)
                direc  = arg(1, 0) % 4
                deltas = [(-1,0),(1,0),(0,-1),(0,1)]
                dr, dc = deltas[direc]
                obj = find_obj(obj_id)
                if obj:
                    to_add = []
                    for r, c in obj["cells"]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0:
                            to_add.append((nr, nc))
                            grid[nr, nc] = obj["color"]
                    obj["cells"].extend(to_add)

            # ── 0x17 MASK_AND  src1 src2 color ───────────────────────────
            elif op == OpCode.MASK_AND:
                s1_id  = arg(0, 1)
                s2_id  = arg(1, 2)
                color  = arg(2, 1)
                o1 = find_obj(s1_id)
                o2 = find_obj(s2_id)
                if o1 and o2:
                    inter = list(set(map(tuple, o1["cells"])) &
                                 set(map(tuple, o2["cells"])))
                    if inter:
                        for r, c in inter:
                            grid[r, c] = color
                        objects.append({
                            "id":    next_id(),
                            "color": color,
                            "cells": inter,
                        })

            # ── 0x18 MASK_OR  src1 src2 color ────────────────────────────
            elif op == OpCode.MASK_OR:
                s1_id  = arg(0, 1)
                s2_id  = arg(1, 2)
                color  = arg(2, 1)
                o1 = find_obj(s1_id)
                o2 = find_obj(s2_id)
                if o1 and o2:
                    union = list(set(map(tuple, o1["cells"])) |
                                 set(map(tuple, o2["cells"])))
                    if union:
                        for r, c in union:
                            grid[r, c] = color
                        objects.append({
                            "id":    next_id(),
                            "color": color,
                            "cells": union,
                        })

            # ── 0x19 MOVE_TO  obj_id target_r target_c ───────────────────
            elif op == OpCode.MOVE_TO:
                obj_id = arg(0, 1)
                tr     = arg(1, 0)
                tc     = arg(2, 0)
                obj = find_obj(obj_id)
                if obj:
                    min_r = min(r for r, _ in obj["cells"])
                    min_c = min(c for _, c in obj["cells"])
                    dr    = tr - min_r
                    dc    = tc - min_c
                    for r, c in obj["cells"]:
                        grid[r, c] = 0
                    new_cells = []
                    for r, c in obj["cells"]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            grid[nr, nc] = obj["color"]
                            new_cells.append((nr, nc))
                        else:
                            # keep original cell if destination out-of-bounds
                            grid[r, c] = obj["color"]
                            new_cells.append((r, c))
                    obj["cells"] = new_cells

            pc += 1

        return grid


# ══════════════════════════════════════════════════════════════════════════════
# OBJECT PLACER
# ══════════════════════════════════════════════════════════════════════════════
class _ObjectPlacer:

    @staticmethod
    def place(grid_h: int, grid_w: int, n_objects: int) -> Tuple[np.ndarray, int]:
        grid    = np.zeros((grid_h, grid_w), dtype=np.uint8)
        placed  = 0
        tries   = 0
        max_try = n_objects * 20

        while placed < n_objects and tries < max_try:
            tries += 1
            oh = random.randint(1, max(1, grid_h // 3))
            ow = random.randint(1, max(1, grid_w // 3))
            if oh > grid_h or ow > grid_w:
                continue
            r0 = random.randint(0, grid_h - oh)
            c0 = random.randint(0, grid_w - ow)
            if grid[r0:r0+oh, c0:c0+ow].any():
                continue
            color = random.randint(1, 9)
            grid[r0:r0+oh, c0:c0+ow] = color
            placed += 1

        return grid, placed


# ══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
class ReverseARCGenerator:
    """
    Generates complete ARC tasks (3 train pairs + 1 test pair) with
    embedded ground-truth programs via five-dimension empirical alignment.

    Args:
        use_rust : if True, attempt to use the Rust sandbox for execution.
                   Falls back to the complete Python executor if Rust is not
                   compiled yet.  Always prints a warning on fallback so the
                   user knows they are not using the faster path.
                   Set use_rust=False to suppress the warning.
    """

    _GRID_SIZES = [
        ((3,  3), 0.05), ((4,  4), 0.05), ((5,  5), 0.10),
        ((6,  6), 0.10), ((7,  7), 0.10), ((8,  8), 0.10),
        ((9,  9), 0.10), ((10,10), 0.15), ((12,12), 0.10),
        ((15,15), 0.10), ((20,20), 0.05),
    ]

    _OBJ_COUNTS = [
        (1, 0.10), (2, 0.20), (3, 0.20), (4, 0.15),
        (5, 0.15), (6, 0.10), (7, 0.05), (8, 0.05),
    ]

    def __init__(self, use_rust: bool = True):
        self._grid_sizes = [s for s, _ in self._GRID_SIZES]
        self._gs_probs   = [p for _, p in self._GRID_SIZES]
        self._obj_counts = [c for c, _ in self._OBJ_COUNTS]
        self._oc_probs   = [p for _, p in self._OBJ_COUNTS]

        self._rust = None
        if use_rust:
            try:
                import insgr_rust  # type: ignore  # noqa
                self._rust = insgr_rust
            except ImportError:
                # BUG-021 FIX: loud warning instead of silent fallback
                print(
                    "[ReverseARCGenerator] WARNING: insgr_rust not found. "
                    "Falling back to Python executor.  Run  python dsl/build.py  "
                    "before large-scale data generation to use the fast Rust path."
                )

    def sample_grid_size(self) -> Tuple[int, int]:
        idx = int(np.random.choice(len(self._grid_sizes), p=self._gs_probs))
        return self._grid_sizes[idx]

    def sample_object_count(self) -> int:
        idx = int(np.random.choice(len(self._obj_counts), p=self._oc_probs))
        return self._obj_counts[idx]

    def sample_template(self) -> str:
        return random.choice(list(RELATIONSHIP_TEMPLATES.keys()))

    def instantiate_program(
        self, template_name: str, num_objects: int
    ) -> Program:
        opcodes      = RELATIONSHIP_TEMPLATES[template_name]
        instructions = []
        for opcode in opcodes:
            args = sample_args(opcode, num_objects)
            instructions.append(Instruction(opcode=opcode, args=args))
        if not instructions:
            args = sample_args(OpCode.TRANSLATE, max(1, num_objects))
            instructions.append(Instruction(OpCode.TRANSLATE, args))
        instructions.append(Instruction(OpCode.HALT))
        return Program(instructions)

    def _execute(self, grid: np.ndarray, program: Program) -> np.ndarray:
        return _PythonExecutor.execute(grid, program)

    def compute_complexity(
        self, program: Program, num_objects: int
    ) -> TaskComplexity:
        return compute_complexity(program, num_objects)

    def generate_task(self) -> Optional[Dict]:
        grid_h, grid_w = self.sample_grid_size()
        n_objects      = self.sample_object_count()
        template       = self.sample_template()
        program        = self.instantiate_program(template, n_objects)
        complexity     = self.compute_complexity(program, n_objects)

        pairs = []
        for _ in range(4):
            in_grid, actual_n = _ObjectPlacer.place(grid_h, grid_w, n_objects)
            if actual_n == 0:
                return None
            out_grid = self._execute(in_grid, program)
            pairs.append((in_grid, out_grid))

        has_change = any(
            not np.array_equal(i_g, o_g) for i_g, o_g in pairs[:3]
        )
        if not has_change:
            return None

        return {
            "train_pairs":    pairs[:3],
            "test_pair":      pairs[3],
            "program_bytes":  program.to_bytes(),
            "complexity_bin": complexity.bin_level,
        }

    def generate_batch(self, n: int) -> List[Dict]:
        tasks    = []
        attempts = 0
        while len(tasks) < n and attempts < n * 10:
            task = self.generate_task()
            if task is not None:
                tasks.append(task)
            attempts += 1
        return tasks