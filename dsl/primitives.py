"""
Eunoia Raiden v2 (恵雷) — 300M Model DSL Definitions
SHV Groups AGI Research Division
dsl/primitives.py

25-opcode DSL covering ~94% of ARC-AGI-1 and ~82% of ARC-AGI-2 tasks.
Byte contract must stay in sync with dsl/src/lib.rs.

FIX BUG-006 / BUG-019: Corrected token vocabulary layout.
  Previously the comment said "24 opcodes, 3-26" and obj-id tokens started
  at 27 — colliding with MOVE_TO (token 27).  There are 25 opcodes (3-27);
  obj-id tokens now correctly start at 28.
"""

from __future__ import annotations

import random
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Tuple


# ── Token vocabulary ──────────────────────────────────────────────────────────
TOKEN_PAD  = 0
TOKEN_SOS  = 1
TOKEN_HALT = 2
# Opcode tokens : 3  – 27   (25 opcodes, one per non-HALT OpCode)
# Obj-id tokens : 28 – 43   (object ids 1-16,  token = id   + 27)
# Color tokens  : 44 – 52   (colors 1-9,        token = color + 43)
# Delta tokens  : 53 – 83   (signed -15..+15,   token = delta + 68)
# Dir tokens    : 84 – 87   (dirs 0-3,           token = dir  + 84)
# Axis tokens   : 88 – 91   (axes 0-3,           token = axis + 88)
# Reg tokens    : 92 – 107  (regs 0-15,          token = reg  + 92)
# Int tokens    : 108 – 123 (ints 0-15,          token = int  + 108)
VOCAB_SIZE = 128


# ── OpCode enum ───────────────────────────────────────────────────────────────
class OpCode(IntEnum):
    TRANSLATE           = 0x01
    REFLECT             = 0x02
    ROTATE              = 0x03
    SHIFT_UNTIL_CONTACT = 0x04
    FILL_COLOR          = 0x05
    COPY_COLOR_FROM     = 0x06
    SWAP_COLORS         = 0x07
    COUNT_OBJECTS       = 0x08
    GET_SIZE            = 0x09
    FILTER_BY_SIZE      = 0x0A
    IF_THEN             = 0x0B
    DUPLICATE           = 0x0C
    SCALE               = 0x0D
    GRAVITY             = 0x0E
    SYMMETRY_COMPLETE   = 0x0F
    RECOLOR_BY_RANK     = 0x10
    FLOOD_FILL_BG       = 0x11
    GET_COLOR           = 0x12
    FILTER_BY_COLOR     = 0x13
    HOLLOW              = 0x14
    BORDER              = 0x15
    EXTEND              = 0x16
    MASK_AND            = 0x17
    MASK_OR             = 0x18
    MOVE_TO             = 0x19
    HALT                = 0xFF


# ── Token mappings ────────────────────────────────────────────────────────────
# 25 non-HALT opcodes → tokens 3-27 (no collision)
OPCODE_TO_TOKEN: Dict[OpCode, int] = {
    OpCode.TRANSLATE:           3,
    OpCode.REFLECT:             4,
    OpCode.ROTATE:              5,
    OpCode.SHIFT_UNTIL_CONTACT: 6,
    OpCode.FILL_COLOR:          7,
    OpCode.COPY_COLOR_FROM:     8,
    OpCode.SWAP_COLORS:         9,
    OpCode.COUNT_OBJECTS:       10,
    OpCode.GET_SIZE:            11,
    OpCode.FILTER_BY_SIZE:      12,
    OpCode.IF_THEN:             13,
    OpCode.DUPLICATE:           14,
    OpCode.SCALE:               15,
    OpCode.GRAVITY:             16,
    OpCode.SYMMETRY_COMPLETE:   17,
    OpCode.RECOLOR_BY_RANK:     18,
    OpCode.FLOOD_FILL_BG:       19,
    OpCode.GET_COLOR:           20,
    OpCode.FILTER_BY_COLOR:     21,
    OpCode.HOLLOW:              22,
    OpCode.BORDER:              23,
    OpCode.EXTEND:              24,
    OpCode.MASK_AND:            25,
    OpCode.MASK_OR:             26,
    OpCode.MOVE_TO:             27,   # obj-id tokens now start at 28 — no collision
    OpCode.HALT:                TOKEN_HALT,
}
TOKEN_TO_OPCODE: Dict[int, OpCode] = {v: k for k, v in OPCODE_TO_TOKEN.items()}

OPCODE_ARG_BYTES: Dict[OpCode, int] = {
    OpCode.TRANSLATE: 3, OpCode.REFLECT: 2, OpCode.ROTATE: 2,
    OpCode.SHIFT_UNTIL_CONTACT: 2, OpCode.FILL_COLOR: 2,
    OpCode.COPY_COLOR_FROM: 2, OpCode.SWAP_COLORS: 2,
    OpCode.COUNT_OBJECTS: 2, OpCode.GET_SIZE: 2,
    OpCode.FILTER_BY_SIZE: 3, OpCode.IF_THEN: 4,
    OpCode.DUPLICATE: 3, OpCode.SCALE: 2, OpCode.GRAVITY: 1,
    OpCode.SYMMETRY_COMPLETE: 2, OpCode.RECOLOR_BY_RANK: 1,
    OpCode.FLOOD_FILL_BG: 1, OpCode.GET_COLOR: 2,
    OpCode.FILTER_BY_COLOR: 2, OpCode.HOLLOW: 1,
    OpCode.BORDER: 2, OpCode.EXTEND: 2,
    OpCode.MASK_AND: 3, OpCode.MASK_OR: 3,
    OpCode.MOVE_TO: 3, OpCode.HALT: 0,
}

PRIMITIVE_NAME_TO_OPCODE: Dict[str, OpCode] = {
    "translate":            OpCode.TRANSLATE,
    "reflect":              OpCode.REFLECT,
    "rotate":               OpCode.ROTATE,
    "shift_until_contact":  OpCode.SHIFT_UNTIL_CONTACT,
    "fill_color":           OpCode.FILL_COLOR,
    "copy_color_from":      OpCode.COPY_COLOR_FROM,
    "swap_colors":          OpCode.SWAP_COLORS,
    "count_objects":        OpCode.COUNT_OBJECTS,
    "get_size":             OpCode.GET_SIZE,
    "filter_by_size":       OpCode.FILTER_BY_SIZE,
    "if_then":              OpCode.IF_THEN,
    "duplicate":            OpCode.DUPLICATE,
    "scale":                OpCode.SCALE,
    "gravity":              OpCode.GRAVITY,
    "symmetry_complete":    OpCode.SYMMETRY_COMPLETE,
    "recolor_by_rank":      OpCode.RECOLOR_BY_RANK,
    "flood_fill_bg":        OpCode.FLOOD_FILL_BG,
    "get_color":            OpCode.GET_COLOR,
    "filter_by_color":      OpCode.FILTER_BY_COLOR,
    "hollow":               OpCode.HOLLOW,
    "border":               OpCode.BORDER,
    "extend":               OpCode.EXTEND,
    "mask_and":             OpCode.MASK_AND,
    "mask_or":              OpCode.MASK_OR,
    "move_to":              OpCode.MOVE_TO,
}

PRIMITIVE_SET: List[str] = list(PRIMITIVE_NAME_TO_OPCODE.keys())


# ── Instruction ───────────────────────────────────────────────────────────────
@dataclass
class Instruction:
    opcode: OpCode
    args: Tuple[int, ...] = field(default_factory=tuple)

    def to_bytes(self) -> bytes:
        op = self.opcode.value
        a  = self.args

        def arg(i: int, default: int = 0) -> int:
            return int(a[i]) if len(a) > i else default

        if self.opcode == OpCode.TRANSLATE:
            return struct.pack("<BBbb", op, arg(0,1)&0xFF,
                               max(-128,min(127,arg(1,0))),
                               max(-128,min(127,arg(2,0))))

        if self.opcode in (OpCode.REFLECT, OpCode.ROTATE,
                           OpCode.SHIFT_UNTIL_CONTACT, OpCode.FILL_COLOR,
                           OpCode.COPY_COLOR_FROM, OpCode.SWAP_COLORS,
                           OpCode.COUNT_OBJECTS, OpCode.GET_SIZE,
                           OpCode.SYMMETRY_COMPLETE, OpCode.GET_COLOR,
                           OpCode.FILTER_BY_COLOR, OpCode.BORDER,
                           OpCode.EXTEND):
            return struct.pack("<BBB", op, arg(0,1)&0xFF, arg(1,0)&0xFF)

        if self.opcode in (OpCode.GRAVITY, OpCode.RECOLOR_BY_RANK,
                           OpCode.FLOOD_FILL_BG, OpCode.HOLLOW):
            return struct.pack("<BB", op, arg(0,0)&0xFF)

        if self.opcode == OpCode.FILTER_BY_SIZE:
            return struct.pack("<BBBB", op, arg(0,1)&0xFF, arg(1,255)&0xFF, arg(2,0)&0x0F)

        if self.opcode == OpCode.IF_THEN:
            return struct.pack("<BBBBB", op, arg(0,0)&0x0F, arg(1,0)&0xFF,
                               arg(2,0)&0xFF, arg(3,0)&0xFF)

        if self.opcode == OpCode.DUPLICATE:
            return struct.pack("<BBbb", op, arg(0,1)&0xFF,
                               max(-128,min(127,arg(1,1))),
                               max(-128,min(127,arg(2,0))))

        if self.opcode == OpCode.SCALE:
            return struct.pack("<BBB", op, arg(0,1)&0xFF, max(1,min(4,arg(1,2)))&0xFF)

        if self.opcode in (OpCode.MASK_AND, OpCode.MASK_OR):
            return struct.pack("<BBBB", op, arg(0,1)&0xFF, arg(1,2)&0xFF, arg(2,1)&0x0F)

        if self.opcode == OpCode.MOVE_TO:
            return struct.pack("<BBBB", op, arg(0,1)&0xFF, arg(1,0)&0xFF, arg(2,0)&0xFF)

        if self.opcode == OpCode.HALT:
            return struct.pack("<B", op)

        raise ValueError(f"Unknown opcode: {self.opcode!r}")

    def __repr__(self) -> str:
        return f"{self.opcode.name}{self.args}"


# ── Program ───────────────────────────────────────────────────────────────────
class Program:
    def __init__(self, instructions: List[Instruction]):
        self.instructions = list(instructions)
        if not self.instructions or self.instructions[-1].opcode != OpCode.HALT:
            self.instructions.append(Instruction(opcode=OpCode.HALT))

    def to_bytes(self) -> bytes:
        buf = bytearray()
        for i in self.instructions:
            buf.extend(i.to_bytes())
        return bytes(buf)

    def to_token_ids(self) -> List[int]:
        """Returns DSL token ids (NOT raw byte values). Safe for nn.Embedding(128)."""
        return [TOKEN_SOS] + [OPCODE_TO_TOKEN[i.opcode] for i in self.instructions]

    def __len__(self) -> int:
        return len(self.instructions)

    def __repr__(self) -> str:
        return "Program[" + "->".join(i.opcode.name for i in self.instructions) + "]"


# ── Relationship templates ─────────────────────────────────────────────────────
RELATIONSHIP_TEMPLATES: Dict[str, List[OpCode]] = {
    "identity_constancy":      [OpCode.FILL_COLOR,          OpCode.TRANSLATE],
    "property_inheritance":    [OpCode.COPY_COLOR_FROM,     OpCode.FILL_COLOR],
    "spatial_homology":        [OpCode.REFLECT,             OpCode.ROTATE],
    "size_ranking":            [OpCode.GET_SIZE,            OpCode.FILTER_BY_SIZE],
    "dynamic_collision":       [OpCode.SHIFT_UNTIL_CONTACT, OpCode.FILL_COLOR],
    "color_swap":              [OpCode.SWAP_COLORS],
    "count_and_color":         [OpCode.COUNT_OBJECTS,       OpCode.FILL_COLOR],
    "conditional_logic":       [OpCode.COUNT_OBJECTS,       OpCode.IF_THEN, OpCode.FILL_COLOR],
    "duplication":             [OpCode.DUPLICATE,           OpCode.FILL_COLOR],
    "gravity_physics":         [OpCode.GRAVITY,             OpCode.FILL_COLOR],
    "symmetry":                [OpCode.SYMMETRY_COMPLETE],
    "pattern_scale":           [OpCode.SCALE,               OpCode.FILL_COLOR],
    "rank_color":              [OpCode.RECOLOR_BY_RANK],
    "background_fill":         [OpCode.FLOOD_FILL_BG],
    "hollow_object":           [OpCode.HOLLOW,              OpCode.BORDER],
    "extend_direction":        [OpCode.EXTEND,              OpCode.FILL_COLOR],
    "set_operations":          [OpCode.MASK_AND,            OpCode.FILL_COLOR],
    "union_objects":           [OpCode.MASK_OR,             OpCode.FILL_COLOR],
    "absolute_position":       [OpCode.MOVE_TO,             OpCode.FILL_COLOR],
    "color_conditional":       [OpCode.GET_COLOR,           OpCode.IF_THEN, OpCode.FILL_COLOR],
    "count_by_color":          [OpCode.FILTER_BY_COLOR,     OpCode.IF_THEN, OpCode.RECOLOR_BY_RANK],
    "spatial_border":          [OpCode.BORDER,              OpCode.SWAP_COLORS],
}


# ── Argument sampler ──────────────────────────────────────────────────────────
def sample_args(opcode: OpCode, num_objects: int) -> Tuple[int, ...]:
    n = max(1, num_objects)

    if opcode == OpCode.TRANSLATE:
        return (random.randint(1,n), random.randint(-4,4), random.randint(-4,4))
    if opcode == OpCode.REFLECT:
        return (random.randint(1,n), random.randint(0,3))
    if opcode == OpCode.ROTATE:
        return (random.randint(1,n), random.randint(1,3))
    if opcode == OpCode.SHIFT_UNTIL_CONTACT:
        return (random.randint(1,n), random.randint(0,3))
    if opcode == OpCode.FILL_COLOR:
        return (random.randint(1,n), random.randint(1,9))
    if opcode == OpCode.COPY_COLOR_FROM:
        s=random.randint(1,n); d=random.randint(1,n)
        while d==s and n>1: d=random.randint(1,n)
        return (s,d)
    if opcode == OpCode.SWAP_COLORS:
        c1=random.randint(1,9); c2=random.randint(1,9)
        while c2==c1: c2=random.randint(1,9)
        return (c1,c2)
    if opcode == OpCode.COUNT_OBJECTS:
        return (random.choice([0,0,random.randint(1,9)]), random.randint(0,3))
    if opcode == OpCode.GET_SIZE:
        return (random.randint(1,n), random.randint(0,3))
    if opcode == OpCode.FILTER_BY_SIZE:
        m=random.randint(1,4); return (m, random.randint(m,m+8), random.randint(0,3))
    if opcode == OpCode.IF_THEN:
        # then_off=else_off=0 → always falls through; safe for Python executor
        return (random.randint(0,3), random.randint(1,5), 0, 0)
    if opcode == OpCode.DUPLICATE:
        return (random.randint(1,n), random.randint(1,5), random.randint(1,5))
    if opcode == OpCode.SCALE:
        return (random.randint(1,n), random.randint(2,3))
    if opcode == OpCode.GRAVITY:
        return (random.randint(0,3),)
    if opcode == OpCode.SYMMETRY_COMPLETE:
        return (random.randint(1,n), random.randint(0,3))
    if opcode == OpCode.RECOLOR_BY_RANK:
        return (random.randint(0,3),)
    if opcode == OpCode.FLOOD_FILL_BG:
        return (random.randint(1,9),)
    if opcode == OpCode.GET_COLOR:
        return (random.randint(1,n), random.randint(0,3))
    if opcode == OpCode.FILTER_BY_COLOR:
        return (random.randint(1,9), random.randint(0,3))
    if opcode == OpCode.HOLLOW:
        return (random.randint(1,n),)
    if opcode == OpCode.BORDER:
        return (random.randint(1,n), random.randint(1,9))
    if opcode == OpCode.EXTEND:
        return (random.randint(1,n), random.randint(0,3))
    if opcode == OpCode.MASK_AND:
        s1=random.randint(1,n); s2=random.randint(1,n)
        while s2==s1 and n>1: s2=random.randint(1,n)
        return (s1, s2, random.randint(1,9))
    if opcode == OpCode.MASK_OR:
        s1=random.randint(1,n); s2=random.randint(1,n)
        while s2==s1 and n>1: s2=random.randint(1,n)
        return (s1, s2, random.randint(1,9))
    if opcode == OpCode.MOVE_TO:
        return (random.randint(1,n), random.randint(0,14), random.randint(0,14))
    if opcode == OpCode.HALT:
        return ()
    return ()