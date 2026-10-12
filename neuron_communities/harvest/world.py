"""Assemble the attribution harvest blocks into a world: per-layer CSR matrices of top-k neuron sets.

Reads ``<harvest-dir>/blocks/block_*.npz`` (the ``attribution`` stage) and writes

  <out-dir>/merged/L{L:02d}.npz   indptr (int64, n_prompts + 1), indices (uint16, neuron ids, exactly k per
                                  prompt in top-k order), values (float32, the variant's score — gradient
                                  products underflow fp16; used by weighted censuses), layer
  <out-dir>/manifest.npz, dedup_kept.npz, dedup_drop.npz   symlinks (or copies, --copy) to the manifest dir
                                  (dedup_drop.npz only when the manifest dir has one)
  <out-dir>/meta.json             the manifest's meta.json plus an ``attr_source`` record
  <harvest-dir>/harvest_summary.json   crosscheck, teacher-valid fraction and mean top-1 log-prob over all blocks

Rows follow manifest order (block ``manifest_lo`` offsets); full manifest coverage is asserted, and the layer count
and ``k`` are taken from the blocks. This is the layout the census, assignment and evaluation stages read. Top-k is
nested (top-k order), so the first ``k' < k`` entries of each row are the top-k' set.

  python -m neuron_communities world --harvest-dir HARVEST --manifest-dir WORLD --out-dir WORLD_act_ga5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

DEFAULT_VARIANT = "act_ga"


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--harvest-dir", type=Path, required=True, help="attribution harvest dir (with blocks/)")
    ap.add_argument("--manifest-dir", type=Path, required=True,
                    help="manifest dir the harvest ran on (manifest.npz, dedup_kept.npz, dedup_drop.npz, meta.json)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="world dir to write (default: <harvest-dir>_<variant><k> next to the harvest dir)")
    ap.add_argument("--variant", default=DEFAULT_VARIANT, help="harvested variant to materialise (the attribution stage writes act_ga)")
    ap.add_argument("--copy", action="store_true", help="copy manifest / dedup files instead of symlinking")


def link(src: Path, dst: Path, copy: bool = False) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    if copy:
        shutil.copyfile(src, dst)
    else:
        os.symlink(os.path.relpath(src, dst.parent), dst)


def run(args: argparse.Namespace) -> None:
    t0 = time.monotonic()
    root, base, v = args.harvest_dir, args.manifest_dir, args.variant
    n = len(np.load(base / "manifest.npz")["row_index"])
    blocks = sorted((root / "blocks").glob("block_*.npz"))
    assert blocks, f"no blocks under {root / 'blocks'}"
    with np.load(blocks[0]) as z:
        K = int(z["k"][0]); variants = [str(x) for x in z["variants"]]
        assert v in variants, f"variant {v!r} not in the harvest ({variants})"
        n_layers = int(z[f"idx_{v}"].shape[1])
    idx_all = np.zeros((n, n_layers, K), np.uint16)
    val_all = np.zeros((n, n_layers, K), np.float32)
    seen = np.zeros(n, bool)
    checks, rels, valid, lps = [], [], [], []
    for f in blocks:
        with np.load(f) as z:
            lo = int(z["manifest_lo"][0]); m = len(z["row_index"])
            idx_all[lo:lo + m] = z[f"idx_{v}"]; val_all[lo:lo + m] = z[f"val_{v}"]
            seen[lo:lo + m] = True
            checks.append(z["crosscheck"]); rels.append(z["crosscheck_rel_err"]); valid.append(z["teacher_valid"])
            lps.append(z["logp_ga"])
    assert seen.all(), f"missing rows: {int((~seen).sum()):,}"
    score_layer = min(18, n_layers - 1)
    summary = {"n": n, "k": K, "layers": n_layers, "blocks": len(blocks), "crosscheck_topk_overlap_mean": float(np.concatenate(checks).mean()),
               "crosscheck_grad_rel_err_max": float(np.concatenate(rels).max()), "teacher_valid_frac": float(np.concatenate(valid).mean()),
               "mean_logp_ga": float(np.nanmean(np.concatenate(lps))),
               f"top1_score_L{score_layer}": {v: float(val_all[:, score_layer, 0].mean())}}
    (root / "harvest_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

    W = args.out_dir if args.out_dir is not None else root.parent / f"{root.name}_{v}{K}"
    (W / "merged").mkdir(parents=True, exist_ok=True)
    indptr = np.arange(n + 1, dtype=np.int64) * K
    for L in range(n_layers):
        np.savez_compressed(W / "merged" / f"L{L:02d}.npz", indptr=indptr,
                            indices=np.ascontiguousarray(idx_all[:, L]).reshape(-1),
                            values=np.ascontiguousarray(val_all[:, L]).reshape(-1), layer=np.array([L]))
    for name in ("manifest.npz", "dedup_kept.npz", "dedup_drop.npz"):
        if name == "dedup_drop.npz" and not (base / name).exists():
            continue                                   # optional (pass 2 of dedup not run): readers treat it as empty
        link(base / name, W / name, args.copy)
    base_meta = json.loads((base / "meta.json").read_text()) if (base / "meta.json").exists() else {}
    meta = dict(base_meta)
    meta["attr_source"] = {"harvest": str(root), "manifest_dir": str(base), "variant": v, "k": K,
                           "note": "top-k per layer under the forward x backward score (attribution stage); every prompt has exactly k neurons per layer"}
    (W / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"world {W} written ({time.monotonic() - t0:.0f}s)", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
