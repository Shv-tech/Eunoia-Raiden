"""
Eunoia Raiden (恵雷) — Full System Smoke Test
SHV Groups AGI Research Division
smoke_test.py

Updated for all API changes from the 26-bug audit fix:
  - InductiveCore uses num_encoder_layers / num_decoder_layers (not num_layers)
  - vocab_size=128 throughout
  - forward() returns 3-tuple (logits, value, entropy)
  - Program.to_token_ids() used instead of list(bytes) for token sequences
  - encode() + decode() split verified
  - Slot eval determinism verified

Run BEFORE renting any GPU.
Expected runtime: ~60-90 seconds on CPU.
All 7 tests must print PASS before training is safe to launch.
"""

import torch
import numpy as np
from core.gnn_perception   import GridParser, GNNPerception
from core.slot_attention   import SlotBottleneck
from core.inductive_core   import InductiveCore
from dsl.primitives        import (Program, Instruction, OpCode,
                                   sample_args, VOCAB_SIZE,
                                   OPCODE_TO_TOKEN, TOKEN_SOS)
from factory.reverse_generator import ReverseARCGenerator
from training.config       import TrainingConfig

print("=" * 58)
print("  Eunoia Raiden — Smoke Test  (26-bug-fix build)")
print("=" * 58)

device     = torch.device("cpu")
config     = TrainingConfig()
all_passed = True


# ── Test 1: DSL serialisation ──────────────────────────────────────────────
print("\n[1/7] DSL serialisation & token safety...")
try:
    instr = Instruction(OpCode.TRANSLATE, args=(1, 2, -3))
    b = instr.to_bytes()
    assert len(b) == 4, f"TRANSLATE should be 4 bytes, got {len(b)}"

    instr2 = Instruction(OpCode.FILL_COLOR, args=(1, 5))
    b2 = instr2.to_bytes()
    assert len(b2) == 3, f"FILL_COLOR should be 3 bytes, got {len(b2)}"

    prog = Program([instr, instr2, Instruction(OpCode.HALT)])
    pb   = prog.to_bytes()
    assert pb[-1] == 0xFF, "Program must end with HALT (0xFF)"

    # Token ids must all be in [0, VOCAB_SIZE)
    token_ids = prog.to_token_ids()
    assert token_ids[0] == TOKEN_SOS, "First token must be SOS"
    for tok in token_ids:
        assert 0 <= tok < VOCAB_SIZE, (
            f"Token {tok} is out of range [0,{VOCAB_SIZE}). "
            "Raw byte values must never be used as token ids."
        )

    # MOVE_TO token must not collide with obj-id 1 token (BUG-006 fix)
    move_to_tok = OPCODE_TO_TOKEN[OpCode.MOVE_TO]
    obj1_tok    = 28  # obj-id 1: token = 1 + 27 = 28 per new layout
    assert move_to_tok != obj1_tok, (
        f"MOVE_TO token ({move_to_tok}) must not collide with obj-id-1 "
        f"token ({obj1_tok})"
    )

    for op in OpCode:
        if op == OpCode.HALT:
            continue
        args = sample_args(op, num_objects=3)
        Instruction(opcode=op, args=args).to_bytes()

    print(f"   PASS — DSL correct (vocab_size={VOCAB_SIZE}, "
          f"MOVE_TO token={move_to_tok}, obj-id-1 token={obj1_tok})")
except AssertionError as e:
    print(f"   FAIL — {e}")
    all_passed = False


# ── Test 2: GridParser ─────────────────────────────────────────────────────
print("\n[2/7] GridParser (flood-fill)...")
try:
    grid = np.array([
        [1, 1, 0, 2],
        [1, 0, 0, 2],
        [0, 0, 3, 0],
    ], dtype=np.uint8)

    nf, ei, et = GridParser.parse(grid, device)
    assert nf.shape[1] == 128, f"Node features should be 128-dim, got {nf.shape}"
    assert nf.shape[0] == 3,   f"Should find 3 objects, got {nf.shape[0]}"
    assert ei.shape[0] == 2,   "edge_index should have 2 rows"
    assert nf.dtype == torch.float32

    empty = np.zeros((5, 5), dtype=np.uint8)
    nf_e, ei_e, _ = GridParser.parse(empty, device)
    assert nf_e.shape == (1, 128), f"Empty grid dummy node wrong shape: {nf_e.shape}"
    assert ei_e.shape[1] == 0,    "Empty grid should have no edges"

    print(f"   PASS — Found {nf.shape[0]} objects, {ei.shape[1]} edges")
except AssertionError as e:
    print(f"   FAIL — {e}")
    all_passed = False


# ── Test 3: GNN forward pass ───────────────────────────────────────────────
print("\n[3/7] GNN forward pass...")
try:
    gnn     = GNNPerception(feature_dim=128, hidden_dim=256, num_layers=2)
    gnn_out = gnn(nf, ei, et)
    assert gnn_out.shape == (3, 256), f"Expected [3,256], got {gnn_out.shape}"
    assert not torch.isnan(gnn_out).any(), "GNN output contains NaN"

    single_nf = torch.randn(1, 128)
    single_ei = torch.zeros(2, 0, dtype=torch.long)
    single_et = torch.zeros(0, dtype=torch.long)
    out_s     = gnn(single_nf, single_ei, single_et)
    assert out_s.shape == (1, 256), f"Single object wrong shape: {out_s.shape}"

    print(f"   PASS — GNN output shape: {gnn_out.shape}")
except AssertionError as e:
    print(f"   FAIL — {e}")
    all_passed = False


# ── Test 4: Slot Attention ─────────────────────────────────────────────────
print("\n[4/7] Slot Attention (eval determinism)...")
try:
    slot = SlotBottleneck(num_slots=12, hidden_dim=256, iters=3)

    # Train mode: must produce different slots across calls
    slot.train()
    inp = gnn_out.unsqueeze(0)
    with torch.no_grad():
        s1, ent1 = slot(inp)
        s2, _    = slot(inp)
    assert not torch.allclose(s1, s2, atol=1e-8), \
        "Training mode should be stochastic"

    # Eval mode: must produce identical slots across calls (BUG-008 fix)
    slot.eval()
    with torch.no_grad():
        s3, ent3 = slot(inp)
        s4, _    = slot(inp)
    assert torch.allclose(s3, s4, atol=1e-6), \
        "Eval mode should be deterministic"
    assert s3.shape == (1, 12, 256), f"Expected [1,12,256], got {s3.shape}"
    assert ent3.item() < 0, "Entropy loss must be negative"

    print(f"   PASS — Slots {s3.shape}, entropy={ent3.item():.4f}, "
          f"train=stochastic, eval=deterministic")
except AssertionError as e:
    print(f"   FAIL — {e}")
    all_passed = False


# ── Test 5: InductiveCore encode/decode split ──────────────────────────────
print("\n[5/7] InductiveCore encode/decode split...")
try:
    SLOT_DIM  = 256
    NUM_SLOTS = 12
    VOCAB     = 128

    core = InductiveCore(
        vocab_size=VOCAB,
        hidden_dim=512,
        num_encoder_layers=2,
        num_decoder_layers=1,
        nhead=8,
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
    test_slots = torch.randn(1, NUM_SLOTS, SLOT_DIM)
    prefix     = torch.tensor([[3, 7]], dtype=torch.long)

    with torch.no_grad():
        logits_fwd, val_fwd = core(train_pairs, test_slots, prefix)
        mem                  = core.encode(train_pairs, test_slots)
        logits_dec, val_dec = core.decode(mem, prefix)

    assert torch.allclose(logits_fwd, logits_dec, atol=1e-5), \
        "encode()+decode() output diverges from forward()"
    assert torch.allclose(val_fwd, val_dec, atol=1e-5), \
        "Value head diverges between forward() and decode()"
    assert logits_fwd.shape[-1] == VOCAB, \
        f"Logits last dim should be {VOCAB}, got {logits_fwd.shape}"
    assert 0 <= val_fwd.item() <= 1, \
        f"Value should be in [0,1], got {val_fwd.item()}"

    print(f"   PASS — logits={logits_fwd.shape}, value={val_fwd.item():.4f}, "
          f"encode/decode consistent")
except AssertionError as e:
    print(f"   FAIL — {e}")
    all_passed = False
except Exception as e:
    print(f"   ERROR — {type(e).__name__}: {e}")
    all_passed = False


# ── Test 6: Full EunoiaRaiden forward pass ─────────────────────────────────
print("\n[6/7] Full EunoiaRaiden forward pass (3-tuple output)...")
try:
    from train import EunoiaRaiden
    model = EunoiaRaiden()
    model.eval()

    vocab_size = model.inductive_core.vocab_size
    assert vocab_size == 128, f"Model vocab_size should be 128, got {vocab_size}"

    def make_pair():
        g = np.random.randint(0, 5, (8, 8), dtype=np.uint8)
        return GridParser.parse(g, device)

    train_pairs = [(make_pair(), make_pair()) for _ in range(3)]
    test_input  = make_pair()
    prefix      = torch.tensor([[3, 7, 5]], dtype=torch.long)

    with torch.no_grad():
        result = model(train_pairs, test_input, program_prefix=prefix)

    assert len(result) == 3, \
        f"forward() must return 3-tuple (logits, value, entropy), got {len(result)}"
    logits, value, entropy = result

    assert logits.shape[-1] == vocab_size, \
        f"Logits last dim should be {vocab_size}, got {logits.shape}"
    assert 0 <= value.item() <= 1, \
        f"Value should be in [0,1], got {value.item()}"
    assert entropy.item() < 0, \
        f"Total entropy should be negative, got {entropy.item()}"
    assert not torch.isnan(logits).any(),  "Logits contain NaN"
    assert not torch.isnan(value),         "Value contains NaN"

    # No-prefix pass
    with torch.no_grad():
        l2, v2, e2 = model(train_pairs, test_input, program_prefix=None)
    assert l2.shape[-1] == vocab_size

    total = sum(p.numel() for p in model.parameters())
    print(f"   PASS — logits={logits.shape}, value={value.item():.4f}, "
          f"entropy={entropy.item():.4f}")
    print(f"   Model total parameters: {total:,}")

except AssertionError as e:
    print(f"   FAIL — {e}")
    all_passed = False
except Exception as e:
    print(f"   ERROR — {type(e).__name__}: {e}")
    all_passed = False


# ── Test 7: Reverse-ARC generator ─────────────────────────────────────────
print("\n[7/7] Reverse-ARC generator (5 tasks, full 24-opcode executor)...")
try:
    gen   = ReverseARCGenerator(use_rust=False)
    tasks = gen.generate_batch(5)

    assert len(tasks) == 5, f"Expected 5 tasks, got {len(tasks)}"

    changed_count = 0
    for i, t in enumerate(tasks):
        assert "train_pairs"   in t, f"Task {i} missing train_pairs"
        assert "program_bytes" in t, f"Task {i} missing program_bytes"
        assert "test_pair"     in t, f"Task {i} missing test_pair"
        assert len(t["train_pairs"]) == 3, \
            f"Task {i} should have 3 train pairs, got {len(t['train_pairs'])}"
        assert 1 <= t["complexity_bin"] <= 5, \
            f"Task {i} complexity_bin out of range: {t['complexity_bin']}"

        in_g, out_g = t["train_pairs"][0]
        assert in_g.shape == out_g.shape, \
            f"Task {i} grid shape mismatch: {in_g.shape} vs {out_g.shape}"
        assert in_g.dtype == np.uint8

        pb = t["program_bytes"]
        assert len(pb) > 0,    f"Task {i} program_bytes empty"
        assert pb[-1] == 0xFF, f"Task {i} program must end with HALT"

        # Check at least one pair shows a change (executor is not all no-ops)
        if any(not np.array_equal(ig, og) for ig, og in t["train_pairs"]):
            changed_count += 1

    # All 5 tasks must show at least one non-trivial transformation
    assert changed_count == 5, (
        f"Only {changed_count}/5 tasks show grid changes. "
        "Python executor may still have missing opcode implementations."
    )

    print(f"   PASS — Generated {len(tasks)} tasks, all show grid transformations")

except AssertionError as e:
    print(f"   FAIL — {e}")
    all_passed = False
except Exception as e:
    print(f"   ERROR — {type(e).__name__}: {e}")
    all_passed = False


# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*58}")
if all_passed:
    print("  ALL 7 TESTS PASSED")
    print()
    print("  Pre-flight checklist:")
    print("  [ ] Run: python dsl/build.py        (compile Rust sandbox)")
    print("  [ ] Place ARC JSON files in data/arc_train/")
    print("  [ ] Run: python train.py --gpu a40  (start training)")
    print(f"{'='*58}")
else:
    print("  ONE OR MORE TESTS FAILED")
    print("  Fix all failures before renting GPU time.")
    print(f"{'='*58}")