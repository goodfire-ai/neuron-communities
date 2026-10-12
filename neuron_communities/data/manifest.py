"""``sample`` stage: freeze a uniform random prompt manifest over a tokens dataset.

A prompt is a (document, target position) pair: the prompt text is the document prefix up to and including the
target token. Positions are sampled uniformly, without replacement, over all positions >= 1 of every document
(position 0 is the BOS token); the order is shuffled so that contiguous blocks of prompts mix documents.

Writes <out>/manifest.npz (row_index, position) and <out>/meta.json (dataset, n, seed, total_valid_positions, ...).
The manifest order is frozen: every later stage refers to prompts by their index into it, and harvest blocks are
contiguous slices of this order.

  python -m neuron_communities sample --dataset wiki_tokens --out wiki250k --n 250000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neuron_communities import config
from neuron_communities.data import io


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--dataset", required=True, help="tokens dataset directory (from the tokens stage)")
    ap.add_argument("--out", required=True, help="manifest directory to create")
    ap.add_argument("--n", type=int, default=250_000, help="number of prompts to sample")
    ap.add_argument("--seed", type=int, default=0)


def run(a: argparse.Namespace) -> None:
    DATASET, OUT = Path(a.dataset), Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    counts = io.read_counts(DATASET)
    valid = (counts - 1).clip(min=0)
    total = int(valid.sum())
    print(f"{len(counts):,} docs, {total:,} valid positions")
    rng = np.random.default_rng(a.seed)
    flat = np.sort(rng.choice(total, size=a.n, replace=False))
    doc_start = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(valid, out=doc_start[1:])
    row = np.searchsorted(doc_start, flat, side="right") - 1
    pos = (flat - doc_start[row] + 1).astype(np.int64)
    assert (pos >= 1).all() and (pos < counts[row]).all()
    order = rng.permutation(a.n)
    row, pos = row[order], pos[order]
    np.savez_compressed(OUT / "manifest.npz", row_index=row, position=pos)
    (OUT / "meta.json").write_text(json.dumps({
        "dataset": str(DATASET), "n": a.n, "seed": a.seed, "total_valid_positions": total,
        "selection": "uniform without replacement over all positions >= 1",
        "note": f"manifest order is frozen; blocks are contiguous {config.BLOCK_SIZE}-slices of this order"}))
    print(f"manifest frozen: {a.n:,} positions, {len(np.unique(row)):,} distinct docs, mean position {pos.mean():.0f}")
