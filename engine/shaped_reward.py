"""
engine/shaped_reward.py

FIX BUG-022: This module is now imported and used in Phase2Trainer.
  Previously it was implemented but never imported anywhere, making reward
  component logging completely absent from training runs.

Decomposes the scalar Rust reward into trackable components for logging.
The Rust sandbox computes:
    R = 0.3 * [syntax_valid] + 0.5 * (correct_cells / total_cells) + 0.2 * [exact_match]
"""
from dataclasses import dataclass


@dataclass
class ShapedReward:
    total:             float
    syntax_component:  float
    partial_component: float
    exact_match:       bool


def compute_dense_reward(raw_reward: float, exact_match: bool) -> ShapedReward:
    """
    Reverse-decomposes the Rust sandbox reward scalar into its three
    additive components so each can be logged independently.

    raw_reward == 0.0 indicates syntax-invalid or timed-out program.
    """
    if raw_reward <= 0.0:
        return ShapedReward(0.0, 0.0, 0.0, False)

    syntax_component  = 0.3
    exact_bonus       = 0.2 if exact_match else 0.0
    partial_component = max(0.0, raw_reward - syntax_component - exact_bonus)

    return ShapedReward(
        total=raw_reward,
        syntax_component=syntax_component,
        partial_component=partial_component,
        exact_match=exact_match,
    ) 