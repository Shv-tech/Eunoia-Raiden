"""
Eunoia Raiden — Live Introspection
visualize/introspect.py

Imports the REAL model from train.py, walks the actual nn.Module tree,
and emits visualize/data.json. Nothing in this file hand-codes a parameter
count or layer shape — every number comes from torch introspection or from
parsing the actual source files. Re-run any time the codebase changes and
the dashboard will reflect it.

Usage:
    python visualize/introspect.py
    python visualize/introspect.py --watch          # regenerate on file change
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUT_PATH = Path(__file__).resolve().parent / "data.json"

# Packages we scan for the codebase graph. Add to this if the repo grows.
PACKAGES = ["core", "dsl", "engine", "factory", "training", "eval"]
ROOT_FILES = ["train.py", "build.py", "ablation.py", "check_params.py", "smoke_test.py"]


# ─────────────────────────────────────────────────────────────────────────
# 1. MODEL INTROSPECTION — real torch modules, real parameter counts
# ─────────────────────────────────────────────────────────────────────────

def _module_node(name: str, mod, path: str) -> dict:
    """One node per nn.Module, with its OWN (non-recursive) param count
    plus a recursive total, so the treemap/3d view can show both."""
    own_params = sum(p.numel() for p in mod.parameters(recurse=False))
    total_params = sum(p.numel() for p in mod.parameters())
    trainable = sum(p.numel() for p in mod.parameters() if p.requires_grad)
    shapes = [list(p.shape) for p in mod.parameters(recurse=False)]
    return {
        "name": name,
        "path": path,
        "type": mod.__class__.__name__,
        "own_params": int(own_params),
        "total_params": int(total_params),
        "trainable_params": int(trainable),
        "param_shapes": shapes,
        "children": [],
    }


def build_model_tree(model, max_depth: int = 6) -> dict:
    root = _module_node("EunoiaRaiden", model, "")
    stack = [(model, root, "", 0)]
    while stack:
        mod, node, prefix, depth = stack.pop()
        if depth >= max_depth:
            continue
        for child_name, child_mod in mod.named_children():
            child_path = f"{prefix}.{child_name}" if prefix else child_name
            child_node = _module_node(child_name, child_mod, child_path)
            node["children"].append(child_node)
            stack.append((child_mod, child_node, child_path, depth + 1))
    return root


def flatten_edges(tree: dict, edges: list, parent_path: str | None = None):
    """Parent -> child edges for the 3D architecture graph."""
    if parent_path is not None:
        edges.append({"source": parent_path, "target": tree["path"]})
    for c in tree["children"]:
        flatten_edges(c, edges, tree["path"])


def try_build_real_model():
    """Attempt to import and instantiate the actual EunoiaRaiden model.
    Falls back gracefully (with a flag) if torch/deps aren't installed
    in the environment running this script — the dashboard will show
    a clear 'stale/unavailable' banner rather than fake data."""
    try:
        import torch  # noqa
        from train import EunoiaRaiden
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

    try:
        model = EunoiaRaiden()
        model.eval()
        return model, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ─────────────────────────────────────────────────────────────────────────
# 2. CONFIG INTROSPECTION — real values from training/config.py
# ─────────────────────────────────────────────────────────────────────────

def introspect_config() -> dict:
    try:
        from training.config import TrainingConfig, GPU_PROFILES
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    cfg = TrainingConfig()
    gpu_profiles = {
        k: {
            "name": v.name, "vram_gb": v.vram_gb,
            "grad_accum_steps": v.grad_accum_steps,
            "use_bfloat16": v.use_bfloat16,
            "use_flash_attn": v.use_flash_attn,
            "compile_model": v.compile_model,
            "notes": v.notes,
        }
        for k, v in GPU_PROFILES.items()
    }
    return {
        "phase1": vars(cfg.phase1),
        "phase2": vars(cfg.phase2),
        "phase3": vars(cfg.phase3),
        "gpu_profiles": gpu_profiles,
    }


# ─────────────────────────────────────────────────────────────────────────
# 3. DSL INTROSPECTION — real opcode table from dsl/primitives.py
# ─────────────────────────────────────────────────────────────────────────

def introspect_dsl() -> dict:
    try:
        from dsl.primitives import (
            OpCode, OPCODE_ARG_BYTES, PRIMITIVE_SET,
            RELATIONSHIP_TEMPLATES, VOCAB_SIZE,
        )
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    opcodes = [
        {"name": op.name, "value": hex(op.value), "arg_bytes": OPCODE_ARG_BYTES.get(op, 0)}
        for op in OpCode
    ]
    templates = {k: [op.name for op in v] for k, v in RELATIONSHIP_TEMPLATES.items()}
    return {
        "vocab_size": VOCAB_SIZE,
        "num_opcodes": len(PRIMITIVE_SET),
        "opcodes": opcodes,
        "templates": templates,
    }


# ─────────────────────────────────────────────────────────────────────────
# 4. CODEBASE GRAPH — real import graph, parsed via ast (Obsidian-style)
# ─────────────────────────────────────────────────────────────────────────

def _module_key(pkg: str, stem: str) -> str:
    return f"{pkg}.{stem}" if pkg else stem


def introspect_codebase_graph() -> dict:
    nodes = {}
    edges = []

    all_files = []
    for pkg in PACKAGES:
        pkg_dir = REPO_ROOT / pkg
        if pkg_dir.is_dir():
            for f in sorted(pkg_dir.glob("*.py")):
                if f.name == "__init__.py":
                    continue
                all_files.append((pkg, f))
    for fname in ROOT_FILES:
        f = REPO_ROOT / fname
        if f.exists():
            all_files.append(("", f))

    key_by_stem = {}
    for pkg, f in all_files:
        key = _module_key(pkg, f.stem)
        key_by_stem[f.stem] = key

    for pkg, f in all_files:
        key = _module_key(pkg, f.stem)
        src = f.read_text(encoding="utf-8", errors="ignore")
        loc = len([l for l in src.splitlines() if l.strip()])
        try:
            tree = ast.parse(src)
        except SyntaxError:
            tree = None

        classes, functions = [], []
        imports_local = set()

        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.FunctionDef) and isinstance(
                    getattr(node, "parent_is_module", True), bool
                ):
                    pass  # placeholder, real functions collected below
            # top-level only, cleaner graph
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    if node.name not in classes:
                        pass
                elif isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    mod = node.module.split(".")[0]
                    sub = node.module.split(".")[-1]
                    if mod in PACKAGES:
                        imports_local.add(sub)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        if top in key_by_stem:
                            imports_local.add(top)

        nodes[key] = {
            "id": key,
            "package": pkg or "root",
            "file": f.name,
            "loc": loc,
            "classes": sorted(set(classes)),
            "functions": functions,
            "size_hash": hashlib.md5(src.encode()).hexdigest()[:8],
        }
        for imp in imports_local:
            if imp in key_by_stem and key_by_stem[imp] != key:
                edges.append({"source": key, "target": key_by_stem[imp]})

    # de-dup edges
    seen = set()
    uniq_edges = []
    for e in edges:
        t = (e["source"], e["target"])
        if t not in seen:
            seen.add(t)
            uniq_edges.append(e)

    return {"nodes": list(nodes.values()), "edges": uniq_edges}


# ─────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────

def build_snapshot() -> dict:
    model, err = try_build_real_model()

    if model is not None:
        model_tree = build_model_tree(model)
        edges = []
        flatten_edges(model_tree, edges)
        total_params = model_tree["total_params"]
        model_status = "live"
    else:
        model_tree, edges, total_params = None, [], None
        model_status = "unavailable"

    snapshot = {
        "generated_at": time.time(),
        "model_status": model_status,
        "model_error": err,
        "total_params": total_params,
        "model_tree": model_tree,
        "model_edges": edges,
        "config": introspect_config(),
        "dsl": introspect_dsl(),
        "codebase_graph": introspect_codebase_graph(),
    }
    return snapshot


def write_snapshot():
    snap = build_snapshot()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(snap, indent=2))
    ts = time.strftime("%H:%M:%S")
    status = snap["model_status"]
    tp = snap["total_params"]
    tp_str = f"{tp:,}" if tp else "n/a"
    print(f"[{ts}] wrote {OUT_PATH.relative_to(REPO_ROOT)} | model={status} | params={tp_str}", flush=True)


def _hash_all_sources() -> str:
    h = hashlib.md5()
    for pkg in PACKAGES:
        d = REPO_ROOT / pkg
        if d.is_dir():
            for f in sorted(d.glob("*.py")):
                h.update(f.read_bytes())
    for fname in ROOT_FILES:
        f = REPO_ROOT / fname
        if f.exists():
            h.update(f.read_bytes())
    return h.hexdigest()


def watch(interval: float = 2.0):
    print("[watch] watching repo for changes... (Ctrl+C to stop)", flush=True)
    last_hash = None
    while True:
        h = _hash_all_sources()
        if h != last_hash:
            write_snapshot()
            last_hash = h
        time.sleep(interval)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="regenerate data.json whenever source files change")
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()

    if args.watch:
        watch(args.interval)
    else:
        write_snapshot()