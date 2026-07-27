"""
test_cross_example.py

FIX BUG-011:
  1. Slot tensors were [1, 12, 256] but InductiveCore(hidden_dim=512) has
     IOSlotEmbedder.up_proj = Linear(slot_dim=512, ...) — Linear(256→1024)
     would receive 256-dim input and crash.  Fixed to slot_dim=512.
  2. Assert checked logits.shape[-1] == 120, but vocab_size defaults to 128.
     Fixed to 128.
  3. hidden_dim=512 was passed to InductiveCore but that sets the transformer
     model dimension.  The production model uses hidden_dim=1024.  Test now
     uses a small but self-consistent config that matches the real constructor
     parameter names (num_encoder_layers, num_decoder_layers) not num_layers.
"""

import torch
from core.inductive_core import InductiveCore


def test_delta_sequence_length():
    """Verifies cross-example sequence assembly and forward pass shape."""

    VOCAB      = 128
    HIDDEN     = 512     # small for fast CPU test
    SLOT_DIM   = 256     # must match what we pass in
    NUM_SLOTS  = 8
    NHEAD      = 8       # hidden // 64 = 8

    core = InductiveCore(
        vocab_size=VOCAB,
        hidden_dim=HIDDEN,
        num_encoder_layers=2,
        num_decoder_layers=1,
        nhead=NHEAD,
        ffn_dim=1024,
        num_slots=NUM_SLOTS,
        slot_dim=SLOT_DIM,
    )
    core.eval()

    # 3 training pairs and 1 test input; slots match slot_dim above
    train_pairs = [
        (torch.randn(1, NUM_SLOTS, SLOT_DIM),
         torch.randn(1, NUM_SLOTS, SLOT_DIM))
        for _ in range(3)
    ]
    test_input = torch.randn(1, NUM_SLOTS, SLOT_DIM)

    with torch.no_grad():
        logits, value = core(train_pairs, test_input)

    assert logits.shape[-1] == VOCAB, \
        f"Vocab projection mismatch: expected {VOCAB}, got {logits.shape[-1]}"
    assert value.shape == (1, 1), \
        f"Value head should be [1,1], got {value.shape}"
    assert 0.0 <= value.item() <= 1.0, \
        f"Value should be in [0,1] (Sigmoid output), got {value.item()}"

    print("Cross-Example Induction: Sequence Assembly & Forward Pass  PASS")


def test_encode_decode_split():
    """Verifies that encode() + decode() produces the same output as forward()."""

    VOCAB     = 128
    HIDDEN    = 512
    SLOT_DIM  = 256
    NUM_SLOTS = 8
    NHEAD     = 8

    core = InductiveCore(
        vocab_size=VOCAB,
        hidden_dim=HIDDEN,
        num_encoder_layers=2,
        num_decoder_layers=1,
        nhead=NHEAD,
        ffn_dim=1024,
        num_slots=NUM_SLOTS,
        slot_dim=SLOT_DIM,
    )
    core.eval()

    train_pairs = [
        (torch.randn(1, NUM_SLOTS, SLOT_DIM),
         torch.randn(1, NUM_SLOTS, SLOT_DIM))
        for _ in range(3)
    ]
    test_input = torch.randn(1, NUM_SLOTS, SLOT_DIM)
    prefix     = torch.tensor([[3, 7, 5]], dtype=torch.long)

    with torch.no_grad():
        logits_fwd,  value_fwd  = core(train_pairs, test_input, prefix)
        mem                      = core.encode(train_pairs, test_input)
        logits_dec,  value_dec  = core.decode(mem, prefix)

    assert torch.allclose(logits_fwd, logits_dec, atol=1e-5), \
        "encode()+decode() diverges from forward()"
    assert torch.allclose(value_fwd,  value_dec,  atol=1e-5), \
        "Value head diverges between forward() and decode()"

    print("Encode/Decode Split: Output Consistency               PASS")


if __name__ == "__main__":
    test_delta_sequence_length()
    test_encode_decode_split()