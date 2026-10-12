"""Stage ``necessity``: KL(clean || community zero-ablated) at the target position for every (prompt, community) pair.

For each prompt of the shard (shards split the prompt set of ``pairs.npz`` round-robin), one trunk forward + cached-tail
precompute at the lowest member layer among the prompt's communities, then one batched target-position-only row per
community in which all of its members are zero-ablated: members of the split layer itself are removed by subtracting
their writes ``W_down[:, n] * act_n`` from the target residual, members of deeper layers through the tail's per-layer
MLP masks. Context positions are never touched.

Input ``pairs.npz``: prompt (kept-row index of the world), cand (index into the candidates).
Output per shard: ``<out>/nec_shard<k>.npz`` with prompt, cand, kl_drop (float32) for every listed pair of the shard.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from neuron_communities import config
from neuron_communities.data.io import World, load_candidates
from neuron_communities.tail import Trunk



def drop_kls(trunk: Trunk, lo: int, tk: torch.Tensor, t_row, t_layer, t_neuron, K: int, chunk: int, bufs=None) -> np.ndarray:
    """KL(clean || members of community k dropped) for k in range(K); members given as (row, layer, neuron) triples."""
    DEV = trunk.device
    ctx, rp, rp_t, neurons_base, W_down, base_lp, base_p = trunk.clean(tk, lo)
    tail = trunk.tail(lo)
    kls = np.empty(K, np.float32)
    for c0 in range(0, K, chunk):
        B = min(chunk, K - c0)
        sel = (t_row >= c0) & (t_row < c0 + B)
        r, l, n = t_row[sel] - c0, t_layer[sel], t_neuron[sel]
        hidden = rp_t.unsqueeze(0).repeat(B, 1)
        effs = {}
        restore = []
        for L in torch.unique(l).tolist():
            m = l == L
            if L == lo:                                  # split layer: subtract the members' writes
                contrib = W_down[:, n[m]] * neurons_base[n[m]]
                delta = torch.zeros(B, W_down.shape[0], device=DEV)
                delta.index_add_(0, r[m], contrib.T)
                hidden -= delta
            else:
                if bufs is not None and L in bufs:
                    e = bufs[L][:B]; restore.append((L, r[m], n[m]))
                else:
                    e = torch.ones(B, config.D_MLP, device=DEV)
                e[r[m], n[m]] = 0.0; effs[L] = e
        lp = torch.log_softmax(tail.step_logits(hidden.to(rp.dtype), ctx, mlp_masks=effs or None).float(), -1)
        kls[c0: c0 + B] = (base_p * (base_lp - lp)).sum(-1).cpu().numpy()
        for L, r_, n_ in restore:
            bufs[L][r_, n_] = 1.0
    return kls


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--world", required=True, help="world dir (manifest.npz, dedup_kept.npz, meta.json)")
    ap.add_argument("--candidates", default=None, help="candidates.json.gz (default <world>/candidates/candidates.json.gz)")
    ap.add_argument("--pairs", default=None, help="pairs.npz with prompt (kept row) and cand (default <world>/assignments/pairs.npz)")
    ap.add_argument("--out", default=None, help="output dir (default <world>/necessity)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts of the shard (smoke tests)")
    ap.add_argument("--chunk", type=int, default=2048, help="intervention rows per tail batch")
    ap.add_argument("--model", default=config.MODEL_ID)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dataset", default=None, help="tokens dataset of the manifest (default: meta.json of the world)")
    ap.add_argument("--dataset-root", default=None, help="base dir for relative dataset paths recorded in meta.json (default: cwd)")


def run(a: argparse.Namespace) -> None:
    W = Path(a.world)
    out = Path(a.out) if a.out else W / "necessity"
    out.mkdir(parents=True, exist_ok=True)
    final = out / f"nec_shard{a.shard:03d}.npz"
    if final.exists():
        print(f"shard {a.shard}: done already", flush=True); return
    cands = load_candidates(a.candidates or W / "candidates" / "candidates.json.gz")
    world = World(W, dataset=a.dataset, dataset_root=a.dataset_root)
    DEV = a.device
    with np.load(a.pairs or W / "assignments" / "pairs.npz") as z:
        P, C = z["prompt"].astype(np.int64), z["cand"].astype(np.int64)
    order = np.lexsort((C, P)); P, C = P[order], C[order]
    prompts = np.unique(P)
    starts = np.searchsorted(P, prompts); ends = np.r_[starts[1:], len(P)]
    sel = np.arange(len(prompts))[a.shard::a.shards]
    if a.limit:
        sel = sel[: a.limit]
    n_pairs = int(sum(ends[s] - starts[s] for s in sel))
    print(f"shard {a.shard}/{a.shards}: {len(sel):,} prompts, {n_pairs:,} pairs", flush=True)
    cm = [np.asarray(c["members"], np.int64) for c in cands]
    trunk = Trunk(a.model, DEV)
    o_p = np.empty(n_pairs, np.int32); o_c = np.empty(n_pairs, np.int32); o_kl = np.empty(n_pairs, np.float32)
    w = 0
    t0 = time.monotonic()
    for n_done, s in enumerate(sel):
        p = int(prompts[s]); cs = C[starts[s]: ends[s]]
        K = len(cs)
        g = np.concatenate([cm[int(c)] for c in cs])
        rows = np.repeat(np.arange(K), [len(cm[int(c)]) for c in cs])
        t_row = torch.as_tensor(rows, device=DEV)
        t_layer = torch.as_tensor(g // config.D_MLP, device=DEV); t_neuron = torch.as_tensor(g % config.D_MLP, device=DEV)
        lo = int((g // config.D_MLP).min())
        with torch.no_grad():
            kls = drop_kls(trunk, lo, world.toks_kept(p), t_row, t_layer, t_neuron, K, a.chunk)
        o_p[w: w + K] = p; o_c[w: w + K] = cs; o_kl[w: w + K] = kls; w += K
        if (n_done + 1) % 200 == 0:
            print(f"  {n_done + 1:,}/{len(sel):,} prompts, {w:,} pairs ({(time.monotonic() - t0) / 60:.1f} min)", flush=True)
    tmp = out / f".nec_shard{a.shard:03d}.tmp.npz"
    np.savez_compressed(tmp, prompt=o_p[:w], cand=o_c[:w], kl_drop=o_kl[:w])
    os.replace(tmp, final)
    print(f"shard {a.shard} done: {len(sel):,} prompts, {w:,} pairs ({(time.monotonic() - t0) / 60:.1f} min)", flush=True)
