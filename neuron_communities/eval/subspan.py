"""Stage ``subspan``: sufficiency and layer necessity for every contiguous sub-span of a necessary pair's member layers.

For each (prompt, community) pair and every contiguous sub-span [a, b] of the community's involved layers (its sorted
member layers; non-member layers in between stay clean), two target-position-only rows:
  only : in layers L[a..b] keep only the members (+ the hub neuron of each layer), all else clean
  off  : in layers L[a..b] keep only the hubs (the layer-necessity readout for that sub-span)
A sub-span is 'explained' by the community when only <= TAU_SUFF and off >= TAU_LAYER; the widest such sub-span is the
community's certified extent on that prompt. Testing sub-spans removes the width penalty of a whole-involved-layers
sufficiency readout (emptying many layers vs two). Full-span values are the row with a = 0, b = k-1.

The hub neurons (``--hubs``) are the always-on neurons that every "only" / "off" row keeps alive; without them even a
perfect community cannot be sufficient, because removing them breaks the model on every prompt.

Input ``pairs.npz``: prompt (kept-row indices of the world), cand (indices into the candidates) — the necessary pairs
written by ``aggregate necessity``.
Output per shard: ``<out>/subspan_shard<k>.npz`` with prompt, cand, a, b, kl_only, kl_off (float16).
"""
from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from neuron_communities import config
from neuron_communities.data.io import World, load_candidates
from neuron_communities.tail import Trunk



def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--world", required=True, help="world dir (manifest.npz, dedup_kept.npz, meta.json)")
    ap.add_argument("--candidates", default=None, help="candidates.json.gz (default <world>/candidates/candidates.json.gz)")
    ap.add_argument("--pairs", default=None, help="necessary pairs (default <world>/subspan/pairs.npz)")
    ap.add_argument("--out", default=None, help="output dir (default <world>/subspan)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=1024, help="intervention rows per tail batch")
    ap.add_argument("--max-layers", type=int, default=config.MAX_SPAN_LAYERS, help="skip communities with more involved layers")
    ap.add_argument("--hubs", default=config.hub_string(), help="always-on hub neurons kept alive, 'L2:4786,L3:6390,...'")
    ap.add_argument("--model", default=config.MODEL_ID)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dataset", default=None, help="tokens dataset of the manifest (default: meta.json of the world)")
    ap.add_argument("--dataset-root", default=None, help="base dir for relative dataset paths recorded in meta.json (default: cwd)")


def run(a: argparse.Namespace) -> None:
    DEV = a.device
    HUBS = config.parse_hubs(a.hubs)
    W = Path(a.world)
    out = Path(a.out) if a.out else W / "subspan"
    out.mkdir(parents=True, exist_ok=True)
    final = out / f"subspan_shard{a.shard:03d}.npz"
    if final.exists():
        print(f"shard {a.shard}: done already", flush=True); return
    cands = load_candidates(a.candidates or W / "candidates" / "candidates.json.gz")
    z = np.load(a.pairs or W / "subspan" / "pairs.npz"); P, C = z["prompt"].astype(np.int64), z["cand"].astype(np.int64)
    by_prompt: dict[int, list[int]] = defaultdict(list)
    for p, c in zip(P.tolist(), C.tolist()):
        by_prompt[p].append(c)
    prompts = sorted(by_prompt)[a.shard::a.shards]
    print(f"shard {a.shard}/{a.shards}: {len(prompts):,} prompts, {sum(len(by_prompt[p]) for p in prompts):,} pairs", flush=True)

    world = World(W, dataset=a.dataset, dataset_root=a.dataset_root)

    # per community: sorted member layers, members per layer, sub-span list
    cinfo = {}
    for c in set(C.tolist()):
        byl = defaultdict(list)
        for g in cands[c]["members"]:
            byl[g // config.D_MLP].append(g % config.D_MLP)
        L = sorted(byl)
        if len(L) > a.max_layers:
            continue
        spans = [(x, y) for x in range(len(L)) for y in range(x, len(L))]
        cinfo[c] = (L, {l: np.array(v) for l, v in byl.items()}, spans)

    trunk = Trunk(a.model, DEV)
    o_p, o_c, o_a, o_b, o_only, o_off = [], [], [], [], [], []
    t0 = time.monotonic()
    for n_done, p in enumerate(prompts):
        cs = [c for c in by_prompt[p] if c in cinfo]
        if not cs:
            continue
        lo = min(cinfo[c][0][0] for c in cs)
        tail = trunk.tail(lo)
        tk = world.toks_kept(p)
        # rows: (cand, a, b, kind) ; kind 0 = only members+hubs, 1 = hubs only
        rows = [(c, x, y, kind) for c in cs for (x, y) in cinfo[c][2] for kind in (0, 1)]
        with torch.no_grad():
            ctx, rp, rp_t, neurons_base, W_down, base_lp, base_p = trunk.clean(tk, lo)
            clean_base = W_down @ neurons_base
            kls = np.empty(len(rows), np.float32)
            for c0 in range(0, len(rows), a.chunk):
                batch = rows[c0: c0 + a.chunk]; B = len(batch)
                hidden = rp_t.unsqueeze(0).repeat(B, 1)
                effs: dict[int, torch.Tensor] = {}
                for r, (c, x, y, kind) in enumerate(batch):
                    L, byl, _ = cinfo[c]
                    for l in L[x: y + 1]:
                        e = torch.zeros(config.D_MLP, device=DEV)
                        if kind == 0:
                            e[torch.as_tensor(byl[l], device=DEV)] = 1.0
                        if l in HUBS:
                            e[HUBS[l]] = 1.0
                        if l > lo:
                            if l not in effs:
                                effs[l] = torch.ones(B, config.D_MLP, device=DEV)
                            effs[l][r] = e
                        else:
                            hidden[r] += W_down @ (neurons_base * e) - clean_base
                lp = torch.log_softmax(tail.step_logits(hidden.to(rp.dtype), ctx, mlp_masks=effs or None).float(), -1)
                kls[c0: c0 + B] = (base_p * (base_lp - lp)).sum(-1).cpu().numpy()
        for r, (c, x, y, kind) in enumerate(rows):
            if kind == 0:
                o_p.append(p); o_c.append(c); o_a.append(x); o_b.append(y); o_only.append(kls[r]); o_off.append(kls[r + 1])
        if (n_done + 1) % 20 == 0:
            print(f"  {n_done + 1:,}/{len(prompts):,} prompts ({(time.monotonic() - t0) / 60:.1f} min)", flush=True)
    tmp = out / f".subspan_shard{a.shard:03d}.tmp.npz"
    np.savez_compressed(tmp, prompt=np.array(o_p, np.int32), cand=np.array(o_c, np.int32), a=np.array(o_a, np.int8), b=np.array(o_b, np.int8),
                        kl_only=np.array(o_only, np.float16), kl_off=np.array(o_off, np.float16))
    os.replace(tmp, final)
    print(f"shard {a.shard} done: {len(o_p):,} sub-span rows ({(time.monotonic() - t0) / 60:.1f} min)", flush=True)
