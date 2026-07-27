# check_params.py
# Run with: python check_params.py
#
# FIX BUG-005: InductiveCore constructor now uses the correct parameter names.
#   Old (broken): InductiveCore(vocab_size=256, hidden_dim=512, num_layers=5)
#     - InductiveCore has no 'num_layers' param → TypeError on construction
#     - vocab_size=256 is wrong (model uses 128) → misleading param count
#   Fixed: uses num_encoder_layers, num_decoder_layers, vocab_size=128 to
#   match the production EunoiaRaiden architecture exactly.

import torch
import torch.nn as nn
from core.gnn_perception import GNNPerception
from core.slot_attention import SlotBottleneck
from core.inductive_core import InductiveCore


class EunoiaRaiden(nn.Module):
    """Mirrors the production model in train.py for parameter counting."""
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
        # BUG-005 FIX: correct parameter names, vocab_size=128
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


model = EunoiaRaiden()

total = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total:,}")
print()
for name, module in model.named_children():
    params = sum(p.numel() for p in module.parameters())
    print(f"  {name:20s} : {params:>12,}")

print()
print("Expected breakdown:")
print("  perception           :      ~22,000,000")
print("  bottleneck           :       ~8,000,000")
print("  inductive_core       :     ~270,000,000")
print("  Total                :     ~300,000,000")