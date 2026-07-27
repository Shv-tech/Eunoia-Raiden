"""
test_mcts.py

FIX BUG-012:
  1. MCTSSearch(policy_model=None, sandbox=None) — 'sandbox' is not a valid
     constructor parameter; train_pair_grids is required.  Fixed.
  2. search.get_ucb_score(parent, child) — method does not exist.
     Correct name is _ucb(child, parent_visits).  Fixed.
  3. Test now constructs MCTSSearch with a minimal valid signature and
     accesses _ucb() correctly.
"""

import math
from unittest.mock import MagicMock

import numpy as np
import torch

from engine.mcts_search import MCTSSearch, MCTSNode
from dsl.primitives import Instruction, Program, OpCode


def _make_mock_model():
    """Build a minimal mock that satisfies MCTSSearch's model interface."""
    mock_core = MagicMock()
    # encode() returns a [1, 112, 1024] tensor
    mock_core.encode.return_value = torch.zeros(1, 112, 1024)
    # decode() returns (logits [1,1,128], value [1,1])
    logits = torch.zeros(1, 1, 128)
    value  = torch.tensor([[0.5]])
    mock_core.decode.return_value = (logits, value)

    mock_model = MagicMock()
    mock_model.inductive_core = mock_core
    mock_model.parameters.return_value = iter([torch.zeros(1)])  # for .device
    return mock_model


def test_ucb_exploration():
    """
    UCB correctly balances exploitation vs exploration.
    A child with visit_count=0 and high prior must outscore a child with
    high Q-value but non-zero visits when parent_visits is large.
    """
    model = _make_mock_model()

    # BUG-012 FIX: correct constructor signature
    search = MCTSSearch(
        policy_model=model,
        train_pair_grids=[],          # empty — no sandbox needed for this test
        max_rollouts=1,
        cached_mem=torch.zeros(1, 112, 1024),
    )

    parent = MCTSNode([], prior=1.0)
    parent.visit_count = 10

    child_exploit        = MCTSNode([Instruction(OpCode.TRANSLATE, (1, 1, 0))],
                                    prior=0.1)
    child_exploit.visit_count = 5
    child_exploit.value_sum   = 4.0   # Q = 0.8

    child_explore        = MCTSNode([Instruction(OpCode.FILL_COLOR, (1, 3))],
                                    prior=0.9)
    child_explore.visit_count = 0
    child_explore.value_sum   = 0.0   # Q = 0.0

    # BUG-012 FIX: correct method name and signature
    ucb_exploit = search._ucb(child_exploit, parent.visit_count)
    ucb_explore = search._ucb(child_explore, parent.visit_count)

    assert ucb_explore > ucb_exploit, (
        f"UCB should favour unexplored high-prior child. "
        f"explore={ucb_explore:.4f}  exploit={ucb_exploit:.4f}"
    )
    print(f"MCTS UCB: exploit={ucb_exploit:.4f} explore={ucb_explore:.4f}  PASS")


def test_node_to_token_ids():
    """
    MCTSNode.to_token_ids() must return DSL vocab tokens (0–127),
    never raw byte values like 0xFF.
    """
    from dsl.primitives import OPCODE_TO_TOKEN, TOKEN_SOS, TOKEN_HALT

    instrs = [
        Instruction(OpCode.TRANSLATE,  (1, 2, -1)),
        Instruction(OpCode.FILL_COLOR, (1, 5)),
        Instruction(OpCode.HALT),
    ]
    node   = MCTSNode(instrs, prior=1.0)
    tokens = node.to_token_ids()

    assert tokens[0] == TOKEN_SOS, "First token must be SOS"
    for tok in tokens:
        assert 0 <= tok < 128, (
            f"Token {tok} is out of vocab range [0,127]. "
            "Raw bytes must never be used as token ids."
        )
    print(f"MCTS Node tokens={tokens}  (all in [0,127])  PASS")


if __name__ == "__main__":
    test_ucb_exploration()
    test_node_to_token_ids()