"""
Eunoia Raiden v2 (恵雷) — 300M Training Configuration
SHV Groups AGI Research Division
training/config.py

FIX BUG-010: lambda_entropy added to Phase2Config.
  Previously slot entropy regularisation was computed in Phase2Trainer but
  never used in the loss because there was no config field for it.
  Slots would collapse during REINFORCE fine-tuning as a result.

FIX BUG-020: Removed misleading comment on grad_accum_steps saying
  "effective batch 256" — the actual effective batch size depends on which
  GPU profile is applied via apply_gpu_profile().
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Phase1Config:
    # Per-sample batch (GNN graphs have variable node counts — keep at 1)
    batch_size:           int   = 1
    # Effective batch = grad_accum_steps * batch_size.
    # Overridden by apply_gpu_profile() — see GPUProfile.grad_accum_steps.
    grad_accum_steps:     int   = 256
    learning_rate:        float = 1e-4
    lr_min:               float = 5e-6
    warmup_steps:         int   = 20_000
    weight_decay:         float = 0.01
    epochs:               int   = 3
    max_norm:             float = 1.0
    lambda_entropy_start: float = 0.01
    lambda_entropy_end:   float = 0.001
    log_every:            int   = 50
    save_every_epoch:     bool  = True


@dataclass
class Phase2Config:
    learning_rate:      float = 2e-5
    mcts_rollouts:      int   = 50
    lambda_v:           float = 0.5
    lambda_e:           float = 1.0
    # BUG-010 FIX: entropy coefficient for slot diversity during RL fine-tuning.
    # Without this, slot attention can collapse after a few thousand REINFORCE
    # steps because nothing penalises degenerate slot assignments.
    lambda_entropy:     float = 0.001
    baseline_ema_alpha: float = 0.99
    c_puct:             float = 0.5
    max_norm:           float = 0.3
    tasks_per_step:     int   = 4
    log_every:          int   = 10
    save_every:         int   = 100


@dataclass
class Phase3Config:
    """ARC-AGI-2 style harder task fine-tuning."""
    learning_rate:      float = 1e-5
    mcts_rollouts:      int   = 100
    lambda_v:           float = 0.3
    lambda_e:           float = 1.0
    lambda_entropy:     float = 0.001
    baseline_ema_alpha: float = 0.995
    c_puct:             float = 0.8
    max_norm:           float = 0.2
    tasks_per_step:     int   = 4
    num_steps:          int   = 5_000
    log_every:          int   = 10
    save_every:         int   = 100


@dataclass
class GPUProfile:
    name:              str
    vram_gb:           int
    grad_accum_steps:  int
    use_bfloat16:      bool
    use_flash_attn:    bool
    num_data_workers:  int
    compile_model:     bool
    notes:             str = ""


GPU_PROFILES = {
    "a40": GPUProfile(
        name="A40 (48GB)",
        vram_gb=48,
        grad_accum_steps=256,
        use_bfloat16=True,
        use_flash_attn=False,
        num_data_workers=6,
        compile_model=False,
        notes="Best cost/performance. ~$0.20/hr. 300M fits with grad_accum=256.",
    ),
    "h100": GPUProfile(
        name="H100 (80GB)",
        vram_gb=80,
        grad_accum_steps=512,
        use_bfloat16=True,
        use_flash_attn=True,
        num_data_workers=12,
        compile_model=True,
        notes="Fastest GPU. ~$2.69/hr. torch.compile + FlashAttn active.",
    ),
    "rtx_pro_6000": GPUProfile(
        name="RTX Pro 6000 (96GB)",
        vram_gb=96,
        grad_accum_steps=512,
        use_bfloat16=True,
        use_flash_attn=False,
        num_data_workers=8,
        compile_model=False,
        notes="Most VRAM. Good for large batch experiments.",
    ),
    "cpu": GPUProfile(
        name="CPU (testing only)",
        vram_gb=0,
        grad_accum_steps=4,
        use_bfloat16=False,
        use_flash_attn=False,
        num_data_workers=2,
        compile_model=False,
        notes="For smoke tests only. Never train the 300M model on CPU.",
    ),
}


@dataclass
class TrainingConfig:
    phase1:         Phase1Config = field(default_factory=Phase1Config)
    phase2:         Phase2Config = field(default_factory=Phase2Config)
    phase3:         Phase3Config = field(default_factory=Phase3Config)

    device:         str  = "cuda"
    gpu_type:       str  = "a40"
    use_bfloat16:   bool = True
    compile_model:  bool = False

    checkpoint_dir: str  = "checkpoints"
    data_dir:       str  = "data/synthetic"
    arc_data_dir:   str  = "data/arc_train"
    arc2_data_dir:  str  = "data/arc2_train"

    def apply_gpu_profile(self) -> None:
        """
        Apply GPU-specific optimisations.
        Always call this after construction to set grad_accum, bf16, compile,
        and data-worker counts from the selected GPU profile.
        """
        profile = GPU_PROFILES.get(self.gpu_type, GPU_PROFILES["a40"])
        self.use_bfloat16            = profile.use_bfloat16
        self.compile_model           = profile.compile_model
        self.phase1.grad_accum_steps = profile.grad_accum_steps
        self.phase2.tasks_per_step   = max(2, profile.grad_accum_steps // 64)
        print(f"[Config] GPU profile applied: {profile.name}")
        print(
            f"[Config] bfloat16={self.use_bfloat16} | "
            f"compile={self.compile_model} | "
            f"grad_accum={self.phase1.grad_accum_steps} | "
            f"data_workers={profile.num_data_workers}"
        )
        print(f"[Config] Notes: {profile.notes}")

    @property
    def num_data_workers(self) -> int:
        return GPU_PROFILES.get(self.gpu_type, GPU_PROFILES["a40"]).num_data_workers