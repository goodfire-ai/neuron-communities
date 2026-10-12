"""Stage ``span-local``: span-local necessity for every (prompt, community, sub-span) row and re-certification.

The ``necessity`` stage ablates the whole community; a sub-span certified by the ``subspan`` stage may therefore owe its
necessity to members outside the span. Span-local necessity closes that gap: for every row the sub-span stage evaluated
(the pairs on which the whole community is necessary, times every contiguous sub-span [a, b] of the member layers),
KL(clean || only the community members inside L[a..b] zero-ablated) at the target position; members in other layers and
everything else run clean. A sub-span then certifies a prompt when span-local necessity >= TAU_NEC, sufficiency
("only members + hubs" in the span) <= TAU_SUFF and layer necessity ("hubs only" in the span) >= TAU_LAYER.

Three steps:
  prep       <world>/subspan/rows_sorted.npz -> <dir>/spec.json (manifest dir, token datasets by row offset, members of
             every community with rows) and <dir>/in/rows_<k>.npz (prompt, cand, a, b of the shard's rows, sorted by prompt;
             shards split the prompt set round-robin). Rows are given explicitly, so the GPU step evaluates exactly what
             the pipeline evaluated.
  rows       one shard: cached-prefix replay as in ``subspan`` (one prefix forward per prompt, all rows of that prompt
             batched in the tail) -> <dir>/out/kl_<k>.npz with prompt, cand, a, b, kl (float32) in the input order
  aggregate  align the shards with rows_sorted.npz and write a span-local statistics world under <out>:
               candidates, manifest.npz, dedup_*.npz, merged   symlinks to the source world; meta.json a copy with a note
               subspan/nec_span.npz        kl per row of the SOURCE rows_sorted (same order), NaN where no value
               subspan/rows_sorted.npz     the rows certified under the span-local definition (a, b, only, off, nec, ns=True, prompt, cand)
               screen/subspan_table.json.gz {cand: [[lo, hi, n_in_span, n_ns, n_layer_nec, n_nec, median_only], ...]} with span-local counts
               coverage/tail_stats.json, certified_communities.npz, certified_prompts.npy, summary.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from neuron_communities import config
from neuron_communities.data.io import World, dataset_segments, load_candidates, prompt_sources, resolve_path, source_names
from neuron_communities.tail import Trunk

t0 = time.monotonic()


def log(m):
    print(f"[{time.monotonic() - t0:5.0f}s] {m}", flush=True)


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("step", choices=["prep", "rows", "aggregate"])
    ap.add_argument("--world", required=True, help="world dir with manifest.npz, dedup_kept.npz, meta.json, candidates/, subspan/rows_sorted.npz")
    ap.add_argument("--dir", default=None, help="span-necessity scan dir with spec.json, in/, out/ (default <world>/span_necessity)")
    ap.add_argument("--shards", type=int, default=192, help="prep: number of input shards")
    ap.add_argument("--shard", type=int, default=0, help="rows: the shard to evaluate")
    ap.add_argument("--chunk", type=int, default=512, help="rows: intervention rows per tail batch")
    ap.add_argument("--model", default=config.MODEL_ID)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dataset-root", default=None, help="base dir for relative dataset / manifest paths recorded in meta.json and spec.json (default: cwd)")
    ap.add_argument("--out", default=None, help="aggregate: the span-local statistics world to write (required)")
    ap.add_argument("--tau-nec", type=float, default=config.TAU_NEC)
    ap.add_argument("--tau-suff", type=float, default=config.TAU_SUFF)
    ap.add_argument("--tau-layer", type=float, default=config.TAU_LAYER)


def prep(a: argparse.Namespace) -> None:
    W = Path(a.world); out = Path(a.dir) if a.dir else W / "span_necessity"; (out / "in").mkdir(parents=True, exist_ok=True)
    meta = json.loads((W / "meta.json").read_text())
    datasets = [[o, p] for o, p in dataset_segments(meta)]
    with np.load(W / "subspan/rows_sorted.npz") as r:
        P = r["prompt"].astype(np.int32); C = r["cand"].astype(np.int32); A = r["a"].astype(np.int8); B = r["b"].astype(np.int8)
    log(f"{len(P):,} rows, {len(np.unique(C)):,} communities")
    cands = load_candidates(W / "candidates/candidates.json.gz")
    present = np.unique(C)
    spec = {"world": str(W), "manifest_dir": str(W), "datasets": datasets, "n_rows": int(len(P)), "shards": a.shards,
            "communities": {str(int(ci)): sorted(cands[int(ci)]["members"]) for ci in present.tolist()}}
    (out / "spec.json").write_text(json.dumps(spec))
    log("spec written")
    uniq, inv = np.unique(P, return_inverse=True)
    shard_of_prompt = np.arange(len(uniq)) % a.shards           # round-robin over the sorted prompt set
    shard = shard_of_prompt[inv]
    order = np.argsort(shard, kind="stable")
    bounds = np.flatnonzero(np.diff(np.concatenate([[-1], shard[order], [a.shards]])))
    for k in range(a.shards):
        s0, e0 = bounds[k], bounds[k + 1]
        idx = order[s0:e0]
        idx = idx[np.argsort(P[idx], kind="stable")]
        np.savez(out / "in" / f"rows_{k:03d}.npz", prompt=P[idx], cand=C[idx], a=A[idx], b=B[idx])
    log(f"{a.shards} input shards written to {out / 'in'} ({len(uniq):,} prompts)")


def rows(a: argparse.Namespace) -> None:
    DEV = a.device
    W = Path(a.world); d = Path(a.dir) if a.dir else W / "span_necessity"; (d / "out").mkdir(exist_ok=True)
    final = d / "out" / f"kl_{a.shard:03d}.npz"
    if final.exists():
        print(f"shard {a.shard}: done already", flush=True); return
    spec = json.loads((d / "spec.json").read_text())
    world = World(resolve_path(spec["manifest_dir"], a.dataset_root), dataset_root=a.dataset_root, segments=spec["datasets"])

    with np.load(d / "in" / f"rows_{a.shard:03d}.npz") as z:
        P, C, A, B = z["prompt"].astype(np.int64), z["cand"].astype(np.int64), z["a"].astype(np.int64), z["b"].astype(np.int64)
    # member layers and members per layer for the communities of this shard
    cinfo: dict[int, tuple[list, dict]] = {}
    for ci in np.unique(C).tolist():
        byl = defaultdict(list)
        for g in spec["communities"][str(ci)]:
            byl[g // config.D_MLP].append(g % config.D_MLP)
        cinfo[ci] = (sorted(byl), {l: torch.as_tensor(v, device=DEV) for l, v in byl.items()})
    bounds = np.flatnonzero(np.diff(np.concatenate([[-1], P, [-1]])))
    print(f"shard {a.shard}: {len(bounds) - 1:,} prompts, {len(P):,} rows, {len(cinfo):,} communities", flush=True)

    trunk = Trunk(a.model, DEV)
    kl_out = np.empty(len(P), np.float32)
    t0 = time.monotonic()
    for n_done, (s0, e0) in enumerate(zip(bounds[:-1], bounds[1:])):
        p = int(P[s0])
        rows_ = [(int(C[k]), int(A[k]), int(B[k])) for k in range(s0, e0)]
        lo = min(cinfo[c][0][x] for c, x, y in rows_)
        tail = trunk.tail(lo)
        tk = world.toks_kept(p)
        with torch.no_grad():
            ctx, rp, rp_t, neurons_base, W_down, base_lp, base_p = trunk.clean(tk, lo)
            clean_base = W_down @ neurons_base
            for c0 in range(0, len(rows_), a.chunk):
                batch = rows_[c0: c0 + a.chunk]; Bn = len(batch)
                hidden = rp_t.unsqueeze(0).repeat(Bn, 1)
                effs: dict[int, torch.Tensor] = {}
                for r, (c, x, y) in enumerate(batch):
                    L, byl = cinfo[c]
                    for l in L[x: y + 1]:
                        e = torch.ones(config.D_MLP, device=DEV); e[byl[l]] = 0.0
                        if l > lo:
                            if l not in effs:
                                effs[l] = torch.ones(Bn, config.D_MLP, device=DEV)
                            effs[l][r] = e
                        else:
                            hidden[r] += W_down @ (neurons_base * e) - clean_base
                lp = torch.log_softmax(tail.step_logits(hidden.to(rp.dtype), ctx, mlp_masks=effs or None).float(), -1)
                kl_out[s0 + c0: s0 + c0 + Bn] = (base_p * (base_lp - lp)).sum(-1).cpu().numpy()
        if (n_done + 1) % 1000 == 0:
            print(f"  {n_done + 1:,}/{len(bounds) - 1:,} prompts ({(time.monotonic() - t0) / 60:.1f} min)", flush=True)
    tmp = d / "out" / f".kl_{a.shard:03d}.tmp.npz"
    np.savez(tmp, prompt=P.astype(np.int32), cand=C.astype(np.int32), a=A.astype(np.int8), b=B.astype(np.int8), kl=kl_out)
    os.replace(tmp, final)
    print(f"shard {a.shard} done: {len(P):,} rows ({(time.monotonic() - t0) / 60:.1f} min)", flush=True)


def aggregate(a: argparse.Namespace) -> None:
    if not a.out:
        raise SystemExit("span-local aggregate: --out <span-local statistics world> is required")
    TAU_N, TAU_S, TAU_L = a.tau_nec, a.tau_suff, a.tau_layer
    W = Path(a.world); D = Path(a.dir) if a.dir else W / "span_necessity"; OUT = Path(a.out)
    for sub in ("subspan", "screen", "coverage"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    for name in ("candidates", "manifest.npz", "dedup_kept.npz", "dedup_drop.npz", "merged"):
        if (W / name).exists() and not (OUT / name).exists():
            os.symlink(os.path.relpath(os.path.realpath(W / name), OUT), OUT / name)
    meta = json.loads((W / "meta.json").read_text())
    meta["span_local_necessity"] = {"source_world": str(W), "scan": str(D), "tau_n": TAU_N, "tau_s": TAU_S,
                                    "note": "necessity ablates only the community members inside the certified sub-span; computed for the pairs the "
                                            "sub-span stage evaluated (whole-community necessary)"}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))

    # scan results -> lookup table
    ks, vs = [], []
    shards = sorted((D / "out").glob("kl_*.npz"))
    for f in shards:
        with np.load(f) as z:
            ks.append(((z["prompt"].astype(np.int64) * 10_000_000 + z["cand"].astype(np.int64)) * 100 + z["a"].astype(np.int64)) * 100 + z["b"].astype(np.int64))
            vs.append(z["kl"].astype(np.float32))
    key = np.concatenate(ks); kl = np.concatenate(vs); order = np.argsort(key, kind="stable"); key, kl = key[order], kl[order]
    log(f"{len(shards)} shards, {len(key):,} scanned rows")

    with np.load(W / "subspan/rows_sorted.npz") as r:
        P = r["prompt"].astype(np.int64); C = r["cand"].astype(np.int64); A = r["a"].astype(np.int64); B = r["b"].astype(np.int64)
        only = r["only"].astype(np.float32); off = r["off"].astype(np.float32)
    q = ((P * 10_000_000 + C) * 100 + A) * 100 + B
    pos = np.minimum(np.searchsorted(key, q), len(key) - 1); hit = key[pos] == q
    nec = np.full(len(q), np.nan, np.float32); nec[hit] = kl[pos[hit]]
    log(f"rows {len(q):,}, matched {int(hit.sum()):,} ({100 * hit.mean():.2f}%)")
    np.savez(OUT / "subspan/nec_span.npz", kl=nec)
    del key, kl, q, pos
    necessary = nec >= TAU_N
    ns = necessary & (only <= TAU_S) & (off >= TAU_L)
    lnec = off >= TAU_L

    cands = load_candidates(W / "candidates/candidates.json.gz")
    layers = [sorted({g // config.D_MLP for g in c["members"]}) for c in cands]

    # certified rows (span-local)
    idx = np.flatnonzero(ns)
    np.savez(OUT / "subspan/rows_sorted.npz", a=A[idx].astype(np.int8), b=B[idx].astype(np.int8), only=only[idx].astype(np.float16), off=off[idx].astype(np.float16),
             nec=nec[idx].astype(np.float16), ns=np.ones(len(idx), bool), prompt=P[idx].astype(np.int32), cand=C[idx].astype(np.int32))
    log(f"certified rows: {len(idx):,}")

    # sub-span table with span-local counts
    gkey = (C * 100 + A) * 100 + B
    ug, inv = np.unique(gkey, return_inverse=True)
    n_in = np.bincount(inv); n_ns = np.bincount(inv, weights=ns).astype(int); n_ln = np.bincount(inv, weights=lnec).astype(int); n_nec = np.bincount(inv, weights=necessary).astype(int)
    order = np.argsort(inv, kind="stable"); bounds = np.flatnonzero(np.diff(np.concatenate([[-1], inv[order], [-1]])))
    table: dict[int, list] = defaultdict(list)
    for gi, g in enumerate(ug.tolist()):
        ci, sa, sb = g // 10_000, (g // 100) % 100, g % 100
        rows_g = order[bounds[gi]:bounds[gi + 1]]
        on = only[rows_g][necessary[rows_g]]
        table[ci].append([layers[ci][sa], layers[ci][sb], int(n_in[gi]), int(n_ns[gi]), int(n_ln[gi]), int(n_nec[gi]), round(float(np.median(on)), 4) if len(on) else None])
    with gzip.open(OUT / "screen/subspan_table.json.gz", "wt") as f:
        json.dump({str(k): v for k, v in table.items()}, f)
    log(f"sub-span table: {len(table):,} communities")

    # certified (prompt, community) pairs -> coverage statistics
    pkey = np.unique(P[idx] * 10_000_000 + C[idx]); cp, cc = pkey // 10_000_000, pkey % 10_000_000
    kept = np.load(W / "dedup_kept.npz")["kept"]; drop = np.load(W / "dedup_drop.npz")["drop"] if (W / "dedup_drop.npz").exists() else np.array([], np.int64)
    live = np.ones(len(kept), bool); live[drop] = False; n_kept = int(live.sum())
    src = prompt_sources(W, kept)
    names = source_names(meta)
    up = np.unique(cp)
    o2 = np.argsort(cc, kind="stable"); cc_s = cc[o2]; cp_s = cp[o2]
    cb = np.flatnonzero(np.diff(np.concatenate([[-1], cc_s, [-1]]))); cis = cc_s[cb[:-1]]; n_obs = np.diff(cb)
    size = np.array([len(cands[ci]["members"]) for ci in cis]); nl = np.array([len(layers[ci]) for ci in cis])
    per_gamma: dict[str, dict] = {}
    gam_of = defaultdict(list)
    for slot, ci in enumerate(cis.tolist()):
        for t in cands[ci]["found"]:
            gam_of[t.split("@")[1]].append(slot)
    for g, slots in sorted(gam_of.items()):
        pr = np.unique(np.concatenate([cp_s[cb[s_]:cb[s_ + 1]] for s_ in slots]))
        per_gamma[g] = {"certified": len(slots), "ge10": int((n_obs[slots] >= 10).sum()), "prompts": int(len(pr)), "frac": len(pr) / n_kept}
    denom = np.bincount(src[live], minlength=len(names)); cov_src = np.bincount(src[up], minlength=len(names))
    stats = {"world": str(OUT), "source_world": str(W), "necessity": "span-local", "n_kept": n_kept,
             "certified_pairs": int(len(pkey)), "certified_rows": int(len(idx)), "union_prompts_observed": int(len(up)), "union_frac_observed": len(up) / n_kept,
             "union_prompts_ht": float(len(up)), "union_frac_ht": len(up) / n_kept, "cap_active": False,
             "per_source": {names[i]: {"kept": int(denom[i]), "covered": int(cov_src[i]), "frac": cov_src[i] / max(denom[i], 1)} for i in range(len(names)) if i in names},
             "certified_communities": int(len(cis)), **{f"certified_ge{k}": int((n_obs >= k).sum()) for k in (10, 50, 100, 500, 1000)},
             "median_size_all": float(np.median(size)) if len(size) else None, "median_size_ge10": float(np.median(size[n_obs >= 10])) if (n_obs >= 10).any() else None,
             "median_layers_ge10": float(np.median(nl[n_obs >= 10])) if (n_obs >= 10).any() else None, "per_gamma": per_gamma}
    (OUT / "coverage/tail_stats.json").write_text(json.dumps(stats, indent=1))
    np.save(OUT / "coverage/certified_prompts.npy", up.astype(np.int32))
    np.savez_compressed(OUT / "coverage/certified_communities.npz", cand=cis.astype(np.int32), size=size.astype(np.int16), n_layers=nl.astype(np.int8),
                        certified_observed=n_obs.astype(np.int32), certified_ht=n_obs.astype(np.float32), weight=np.ones(len(cis), np.float32))
    (OUT / "coverage/summary.json").write_text(json.dumps({"n_prompts": n_kept, "tau_n": TAU_N, "tau_s": TAU_S, "necessity": "span-local",
                                                            "union_frac": len(up) / n_kept, "per_gamma": per_gamma}, indent=1))
    log(f"coverage: {len(up):,} / {n_kept:,} prompts ({100 * len(up) / n_kept:.2f}%), {len(cis):,} certified communities, {int((n_obs >= 10).sum()):,} with >= 10")
    log("done")


def run(a: argparse.Namespace) -> None:
    {"prep": prep, "rows": rows, "aggregate": aggregate}[a.step](a)
