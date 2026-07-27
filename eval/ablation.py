"""
ablation.py — Empirical ablation runner.

FIX BUG-004: evaluator.evaluate(max_rollouts=rollout_budget) now works.
  ARCEvaluator.evaluate() previously took no parameters; a max_rollouts
  argument has been added so ablation runs can override the search budget.
"""

import argparse
import torch
from eval.arc_evaluator import ARCEvaluator
from core.inductive_core import InductiveCore
from train import EunoiaRaiden
from training.config import TrainingConfig


def run_ablation(disable_mcts: bool, disable_relational_pass: bool):
    print(
        f"[Ablation] Config: MCTS Disabled={disable_mcts}, "
        f"Relational Pass Disabled={disable_relational_pass}"
    )

    config = TrainingConfig()
    device = torch.device("cpu")

    model = EunoiaRaiden().to(device)
    model.eval()
    # model.load_state_dict(torch.load("checkpoints/phase2_final.pt")["model_state"])

    rollout_budget = 1 if disable_mcts else 12_500

    evaluator = ARCEvaluator(
        model=model,
        data_dir=config.arc_data_dir,
        max_rollouts=rollout_budget,
        use_tta=not disable_mcts,   # TTA is meaningless if MCTS is off
    )

    # BUG-004 FIX: evaluate() now accepts max_rollouts parameter
    score = evaluator.evaluate(max_rollouts=rollout_budget)
    print(f"[Ablation] Score: {score*100:.2f}%")
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_mcts",       action="store_true")
    parser.add_argument("--no_relational", action="store_true")
    args = parser.parse_args()
    run_ablation(args.no_mcts, args.no_relational)