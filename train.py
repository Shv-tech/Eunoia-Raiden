"""
Eunoia Raiden v2 (恵雷) — 300M Parameter ARC-AGI Engine
SHV Groups AGI Research Division
train.py

FIX BUG-023: Phase 3 no longer mutates config.phase2 in-place.
  Previously run_phase3() overwrote config.phase2 fields and attempted
  restoration in a finally block that was missing — a mid-run crash left
  config permanently corrupted for any subsequent phase.  Phase 3 now
  creates its own Phase2Config instance, leaving the original untouched.

Usage:
    python train.py                                     # full pipeline
    python train.py --gpu h100
    python train.py --phase 1
    python train.py --phase 2 --resume checkpoints/phase1_final.pt
    python train.py --phase 3 --resume checkpoints/phase2_final.pt
    python train.py --eval   --resume checkpoints/phase3_final.pt
    python train.py --count-params
    python train.py --generate-data --num-tasks 2000000

GPU profiles:
    --gpu a40           A40 48GB  $0.20/hr  (default, best cost)
    --gpu h100          H100 80GB $2.69/hr  (fastest)
    --gpu rtx_pro_6000  RTX Pro 6000 96GB
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from core.gnn_perception     import GNNPerception, GridParser
from core.slot_attention     import SlotBottleneck
from core.inductive_core     import InductiveCore
from factory.dataset_builder import DatasetBuilder, build_phase1_dataloader
from factory.arc_loader      import ARCLoader
from training.config         import TrainingConfig, Phase2Config
from training.phase1_trainer import Phase1Trainer
from training.phase2_trainer import Phase2Trainer


# ══════════════════════════════════════════════════════════════════════════════
# EUNOIA RAIDEN v2 — 300M Parameter Model
# ══════════════════════════════════════════════════════════════════════════════
class EunoiaRaiden(nn.Module):
    """
    300M parameter universal inductive reasoning engine.

    Architecture:
      GNNPerception   : 512-dim, 4 layers           → ~22M params
      SlotBottleneck  : 16 slots, 512-dim, 5 iters  → ~8M  params
      InductiveCore   : 1024-dim, 16+6 layers       → ~270M params
      Total                                         → ~300M params

    forward() returns (logits, value, total_entropy) — always a 3-tuple.
    """

    def __init__(self):
        super().__init__()
        self.perception = GNNPerception(
            feature_dim=128,
            hidden_dim=512,
            num_layers=4,
        )
        self.bottleneck = SlotBottleneck(
            num_slots=16,
            hidden_dim=512,
            iters=5,
        )
        self.inductive_core = InductiveCore(
            vocab_size=128,
            hidden_dim=1024,
            num_encoder_layers=16,
            num_decoder_layers=6,
            nhead=16,
            ffn_dim=4096,
            num_slots=16,
            slot_dim=512,
        )

    def _graph_to_slots(
        self, graph_tuple
    ):
        nf, ei, et     = graph_tuple
        gnn_out        = self.perception(nf, ei, et)
        slots, entropy = self.bottleneck(gnn_out.unsqueeze(0))
        return slots, entropy

    def forward(self, train_pairs, test_input, program_prefix=None):
        """
        Args:
            train_pairs    : list of up to 3 × (graph_tuple, graph_tuple)
            test_input     : graph_tuple  (nf, ei, et)
            program_prefix : optional LongTensor [1, T] of DSL token ids

        Returns:
            logits        : [1, T, vocab_size]
            value         : [1, 1]
            total_entropy : scalar tensor (negative — minimise to maximise diversity)
        """
        processed  : list = []
        total_ent  = torch.zeros(1, device=next(self.parameters()).device)

        for in_g, out_g in train_pairs:
            in_s,  ie = self._graph_to_slots(in_g)
            out_s, oe = self._graph_to_slots(out_g)
            processed.append((in_s, out_s))
            total_ent = total_ent + ie + oe

        test_s, te = self._graph_to_slots(test_input)
        total_ent  = total_ent + te

        logits, value = self.inductive_core(
            processed, test_s, program_prefix=program_prefix
        )
        return logits, value, total_ent


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM INIT
# ══════════════════════════════════════════════════════════════════════════════
def _print_banner():
    print("=" * 66)
    print("  EUNOIA RAIDEN (恵雷)  —  SHV Groups Eunoia Labs")
    print("=" * 66)


def _verify_rust() -> bool:
    try:
        import insgr_rust  # type: ignore  # noqa
        print("[System] Rust sandbox verified.")
        return True
    except ImportError:
        pass
    print("[System] Compiling Rust sandbox ...")
    r = subprocess.run([sys.executable, "dsl/build.py"])
    if r.returncode != 0:
        print("[FATAL] Rust compilation failed.")
        return False
    try:
        import insgr_rust  # type: ignore  # noqa
        print("[System] Rust sandbox compiled successfully.")
        return True
    except ImportError:
        print("[FATAL] Rust compiled but import still failed. Check maturin output.")
        return False


def _build_model(device: torch.device, config: TrainingConfig) -> EunoiaRaiden:
    model = EunoiaRaiden().to(device)

    if config.compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[System] torch.compile active (H100 mode).")
        except Exception as e:
            print(f"[System] torch.compile skipped: {e}")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Model] Total parameters : {total:>15,}")
    print(f"[Model] Trainable        : {trainable:>15,}")
    print()
    for name, mod in [("perception",     model.perception),
                      ("bottleneck",     model.bottleneck),
                      ("inductive_core", model.inductive_core)]:
        p = sum(x.numel() for x in mod.parameters())
        print(f"  {name:20s}: {p:>12,}")
    print()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# DATASET GENERATION
# ══════════════════════════════════════════════════════════════════════════════
def generate_dataset(config: TrainingConfig, num_tasks: int = 2_000_000):
    if os.path.exists(config.data_dir):
        shards = list(Path(config.data_dir).glob("*.parquet"))
        if shards:
            print(
                f"[DataGen] Dataset exists ({len(shards)} shards). "
                "Delete to regenerate."
            )
            return

    print(f"[DataGen] Generating {num_tasks:,} synthetic tasks ...")
    t0 = time.time()
    DatasetBuilder(
        num_tasks=num_tasks,
        num_workers=min(16, os.cpu_count() or 4),
    ).build(config.data_dir)
    print(f"[DataGen] Done in {(time.time()-t0)/60:.1f} min.")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Supervised on synthetic tasks
# ══════════════════════════════════════════════════════════════════════════════
def run_phase1(
    model:  EunoiaRaiden,
    config: TrainingConfig,
    resume: str | None = None,
) -> str:
    print("\n" + "-"*66)
    print("  PHASE 1 — The Apprentice")
    print("  Supervised learning on 2M synthetic Reverse-ARC tasks")
    print("-"*66)

    epoch_bins = [[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5]]

    trainer = Phase1Trainer(
        model=model, config=config,
        dataloader=build_phase1_dataloader(
            config.data_dir, complexity_bins=epoch_bins[0],
            num_workers=config.num_data_workers,
        ),
    )
    if resume:
        trainer.load_checkpoint(resume)

    for epoch in range(config.phase1.epochs):
        bins = epoch_bins[min(epoch, len(epoch_bins) - 1)]
        trainer.dataloader = build_phase1_dataloader(
            config.data_dir, complexity_bins=bins,
            num_workers=config.num_data_workers, shuffle=True,
        )
        print(f"\n[Phase1] Epoch {epoch} | bins={bins}")
        t0 = time.time()
        trainer.train_epoch(epoch)
        print(f"[Phase1] Epoch {epoch} done in {(time.time()-t0)/3600:.2f}h")
        if config.phase1.save_every_epoch:
            trainer.save_checkpoint(f"epoch{epoch}")

    path = trainer.save_checkpoint("final")
    print(f"[Phase1] Complete -> {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — MCTS REINFORCE on ARC-AGI-1
# ══════════════════════════════════════════════════════════════════════════════
def run_phase2(
    model:     EunoiaRaiden,
    config:    TrainingConfig,
    resume:    str | None = None,
    num_steps: int = 10_000,
) -> str:
    print("\n" + "-"*66)
    print("  PHASE 2 — The Reasoner (ARC-AGI-1)")
    print("  MCTS-in-the-loop REINFORCE on real ARC tasks")
    print("-"*66)

    if not os.path.exists(config.arc_data_dir):
        print(f"[FATAL] ARC data not found: {config.arc_data_dir}")
        sys.exit(1)

    trainer = Phase2Trainer(
        model=model, config=config,
        arc_loader=ARCLoader(config.arc_data_dir),
    )
    if resume:
        trainer.load_checkpoint(resume)

    t0 = time.time()
    rw: list = []
    for step in range(num_steps):
        r = trainer.train_step()
        rw.append(r)
        if (step + 1) % config.phase2.save_every == 0:
            trainer.save_checkpoint(f"step{trainer.global_step}")
            avg = sum(rw) / len(rw)
            print(
                f"[Phase2] [{(time.time()-t0)/3600:.2f}h] "
                f"Avg reward={avg:.4f} (last {len(rw)} steps)"
            )
            rw.clear()

    path = trainer.save_checkpoint("final")
    print(f"[Phase2] Complete -> {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Fine-tuning on ARC-AGI-2
# ══════════════════════════════════════════════════════════════════════════════
def run_phase3(
    model:  EunoiaRaiden,
    config: TrainingConfig,
    resume: str | None = None,
) -> str:
    """
    BUG-023 FIX: Phase 3 builds its OWN Phase2Config from Phase3Config values
    instead of mutating config.phase2 in-place.  The original config is never
    touched, so a mid-run crash cannot corrupt it.
    """
    print("\n" + "-"*66)
    print("  PHASE 3 — The Champion (ARC-AGI-2)")
    print("  Fine-tuning on harder ARC-AGI-2 style tasks")
    print("-"*66)

    arc2_dir = config.arc2_data_dir
    if not os.path.exists(arc2_dir):
        print(f"[Phase3] ARC-AGI-2 data not found at {arc2_dir}.")
        print("[Phase3] Falling back to ARC-AGI-1 data with Phase3 hyperparams.")
        arc2_dir = config.arc_data_dir

    # BUG-023 FIX: build a fresh Phase2Config with Phase3 values — never mutate
    p3_as_p2           = Phase2Config()
    p3_as_p2.learning_rate      = config.phase3.learning_rate
    p3_as_p2.mcts_rollouts      = config.phase3.mcts_rollouts
    p3_as_p2.lambda_v           = config.phase3.lambda_v
    p3_as_p2.lambda_e           = config.phase3.lambda_e
    p3_as_p2.lambda_entropy     = config.phase3.lambda_entropy
    p3_as_p2.baseline_ema_alpha = config.phase3.baseline_ema_alpha
    p3_as_p2.c_puct             = config.phase3.c_puct
    p3_as_p2.max_norm           = config.phase3.max_norm
    p3_as_p2.tasks_per_step     = config.phase3.tasks_per_step
    p3_as_p2.save_every         = config.phase3.save_every
    p3_as_p2.log_every          = config.phase3.log_every

    phase3_config           = copy.deepcopy(config)
    phase3_config.phase2    = p3_as_p2
    # NOTE: config.phase2 is NEVER modified — original config is preserved.

    trainer = Phase2Trainer(
        model=model, config=phase3_config,
        arc_loader=ARCLoader(arc2_dir),
    )
    if resume:
        trainer.load_checkpoint(resume)

    t0 = time.time()
    rw: list = []
    for step in range(config.phase3.num_steps):
        r = trainer.train_step()
        rw.append(r)
        if (step + 1) % config.phase3.save_every == 0:
            trainer.save_checkpoint(f"phase3_step{trainer.global_step}")
            avg = sum(rw) / len(rw)
            print(f"[Phase3] [{(time.time()-t0)/3600:.2f}h] Avg reward={avg:.4f}")
            rw.clear()

    path = trainer.save_checkpoint("phase3_final")
    print(f"[Phase3] Complete -> {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def run_evaluation(
    model:        EunoiaRaiden,
    config:       TrainingConfig,
    max_rollouts: int  = 12_500,
    use_tta:      bool = True,
) -> float:
    from eval.arc_evaluator import ARCEvaluator
    print("\n" + "-"*66)
    print(
        f"  EVALUATION | TTA={'ON' if use_tta else 'OFF'} | "
        f"Rollouts={max_rollouts:,}"
    )
    print("-"*66)
    ev = ARCEvaluator(
        model=model, data_dir=config.arc_data_dir,
        max_rollouts=max_rollouts, use_tta=use_tta,
    )
    return ev.evaluate()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def _parse_args():
    p = argparse.ArgumentParser(
        description="Eunoia Raiden v2 — 300M ARC-AGI Engine"
    )
    p.add_argument("--phase", type=int, choices=[1, 2, 3], default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--no-tta", action="store_true",
                   help="Disable test-time augmentation during evaluation")
    p.add_argument("--generate-data", action="store_true")
    p.add_argument("--count-params", action="store_true")
    p.add_argument("--num-tasks", type=int, default=2_000_000)
    p.add_argument("--phase2-steps", type=int, default=10_000)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--gpu", type=str, default="a40",
        choices=["a40", "h100", "rtx_pro_6000", "cpu"],
        help="GPU profile to use",
    )
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    _print_banner()
    args   = _parse_args()
    config = TrainingConfig(gpu_type=args.gpu)
    config.apply_gpu_profile()

    if args.device:
        config.device = args.device
    device = torch.device(
        config.device if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cpu":
        print("[System] WARNING: CPU mode — bfloat16 disabled.")
        config.use_bfloat16 = False

    print(f"[System] Device  : {device}")
    print(f"[System] GPU     : {config.gpu_type}")
    print(f"[System] bf16    : {config.use_bfloat16}")
    print(f"[System] compile : {config.compile_model}")
    print()

    # ── Parameter count only ─────────────────────────────────────────────
    if args.count_params:
        m = EunoiaRaiden()
        t = sum(p.numel() for p in m.parameters())
        print(f"Total parameters: {t:,}")
        for name in ["perception", "bottleneck", "inductive_core"]:
            mod = getattr(m, name)
            p   = sum(x.numel() for x in mod.parameters())
            print(f"  {name:20s}: {p:>12,}")
        return

    # ── Rust sandbox ─────────────────────────────────────────────────────
    if not _verify_rust():
        sys.exit(1)

    # ── Data generation only ─────────────────────────────────────────────
    if args.generate_data:
        generate_dataset(config, num_tasks=args.num_tasks)
        return

    # ── Build model ───────────────────────────────────────────────────────
    model  = _build_model(device, config)
    p1_ckpt = args.resume

    # ── Evaluation only ───────────────────────────────────────────────────
    if args.eval:
        if not args.resume:
            print("[FATAL] --eval requires --resume")
            sys.exit(1)
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        run_evaluation(model, config, use_tta=not args.no_tta)
        return

    # ── Phase 1 ───────────────────────────────────────────────────────────
    if args.phase is None or args.phase == 1:
        shards = (
            list(Path(config.data_dir).glob("*.parquet"))
            if os.path.exists(config.data_dir) else []
        )
        if not shards:
            print("[System] No synthetic data found. Generating ...")
            generate_dataset(config, args.num_tasks)
        p1_ckpt = run_phase1(model, config, resume=p1_ckpt)

    # ── Phase 2 ───────────────────────────────────────────────────────────
    p2_ckpt = args.resume if args.phase == 2 else p1_ckpt
    if args.phase is None or args.phase == 2:
        p2_ckpt = run_phase2(
            model, config, resume=p2_ckpt, num_steps=args.phase2_steps
        )

    # ── Phase 3 ───────────────────────────────────────────────────────────
    p3_ckpt = args.resume if args.phase == 3 else p2_ckpt
    if args.phase is None or args.phase == 3:
        run_phase3(model, config, resume=p3_ckpt)

    # ── Final evaluation ──────────────────────────────────────────────────
    if args.phase is None:
        print("\n[System] Running final evaluation with TTA ...")
        score = run_evaluation(model, config, use_tta=True)
        print(f"\n[System] Final ARC score: {score*100:.2f}%")

    print("\n[System] Eunoia Raiden v2 — Training complete.")


if __name__ == "__main__":
    main()