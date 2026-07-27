# Eunoia Raiden — Live Architecture Dashboard

A local, interactive dashboard that reads your **actual** `EunoiaRaiden` model,
`training/config.py`, and `dsl/primitives.py` at runtime — every number on
screen comes from introspecting the real objects, not from hand-typed data.
Edit the codebase, save, and the dashboard updates within a couple of
seconds.

### What you get
- **Architecture** — 3D view of every `nn.Module` in the model (`perception`
  → `bottleneck` → `inductive_core` → individual encoder/decoder layers),
  sphere size = real parameter count, click any node for its exact param
  count and tensor shapes.
- **Codebase Graph** — an Obsidian-style force graph of `core/ dsl/ engine/
  factory/ training/ eval/`, built by parsing real `import` statements with
  `ast`. Node size = lines of code.
- **DSL** — live opcode table and relationship-template list parsed from
  `dsl/primitives.py`.
- **Config** — live `Phase1Config` / `Phase2Config` / `Phase3Config` /
  `GPU_PROFILES` values from `training/config.py`.

Theme is a clean light UI (white/off-white, near-black text, Inter +
JetBrains Mono, one accent color per module) — no cyberpunk styling.

## Install

Drop the `visualize/` folder into the repo root, next to `train.py`, so the
layout looks like:

```
your-repo/
  core/  dsl/  engine/  factory/  training/  eval/
  train.py
  visualize/
    introspect.py
    dashboard.html
    server.py
    README.md
```

No new dependencies beyond what training already needs (`torch`, `numpy`).
The dashboard itself is plain HTML/JS loaded from CDN (three.js, D3) — your
browser fetches those, not the training environment.

## Run

```bash
python visualize/server.py
```

This opens `http://127.0.0.1:8765/dashboard.html` in your browser
automatically (`--no-browser` to skip that) and starts a background scan
that re-runs introspection whenever a `.py` file under `core/ dsl/ engine/
factory/ training/ eval/` or the root scripts changes (default: checked
every 2s, `--interval` to change it).

```bash
python visualize/server.py --port 9000 --interval 1
```

To just generate a one-off snapshot without the server (e.g. for CI or a
quick check):

```bash
python visualize/introspect.py
```

writes `visualize/data.json`, which `dashboard.html` polls.

## Notes / honesty about limits

- **Model construction cost.** `EunoiaRaiden()` allocates ~354M float32
  parameters. On CPU this takes several seconds per re-scan — that's real
  `torch.randn` initialization cost, not overhead in this script. On the
  GPU box you train on it'll be faster; if you want snappier local
  iteration, bump `--interval` up so it isn't re-instantiating on every
  keystroke-adjacent save.
- **What "live" means.** The model is *re-constructed from scratch* on each
  scan (fresh random init) purely to read off its architecture (module
  tree, parameter shapes, counts). It does **not** load a checkpoint, so
  parameter *values* aren't meaningful here — only the *structure* (shapes,
  counts, module graph) is. If you want the dashboard to reflect a trained
  checkpoint's actual weight statistics (norms, gradient history, etc.)
  that's a different, heavier feature — say the word and I'll build a
  `--checkpoint path.pt` mode for it.
- **If `torch` import fails** (e.g. running this on a machine without the
  training deps installed), the dashboard shows a clear "model unavailable"
  status with the real Python exception, rather than silently faking data.