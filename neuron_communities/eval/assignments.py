"""Stage ``assign``: which prompts each community is tested on (the prefilter of the necessity screen).

For every kept prompt, find every community with >= ``--min-present`` members present in the prompt's top-k attribution
sets (sparse boolean matmul: prompts x neurons @ neurons x communities). Communities matching more than their cap get a
uniform random sample of cap prompts with weight = matches/cap recorded (Horvitz-Thompson weights: unbiased coverage
estimates, bounded compute); all others are exhaustive (weight 1). The cap depends on the community size (``--cap`` up to
64 members, ``--cap-large`` up to 256, ``--cap-giant`` above).

Writes under ``--out``:
  pairs.npz          prompt (kept row), cand (index into the candidates), weight (float32), sorted by (prompt, cand)
  match_counts.npz   counts: raw matches per community before the cap
  meta.json          pair counts and the settings
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse

from neuron_communities import config
from neuron_communities.data.io import load_candidates



def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--world", required=True,
                    help="world dir with merged/L*.npz, manifest.npz and dedup_kept.npz whose top-k sets the members are matched against")
    ap.add_argument("--candidates", default=None, help="candidates.json.gz (default <world>/candidates/candidates.json.gz)")
    ap.add_argument("--out", default=None, help="output dir (default <world>/assignments)")
    ap.add_argument("--cap", type=int, default=config.ASSIGN_CAP, help="per-community prompt cap for sizes <= 64")
    ap.add_argument("--cap-large", type=int, default=config.ASSIGN_CAP_LARGE, help="per-community prompt cap for sizes 65-256")
    ap.add_argument("--cap-giant", type=int, default=config.ASSIGN_CAP_GIANT, help="per-community prompt cap for sizes > 256")
    ap.add_argument("--min-present", type=int, default=config.MIN_PRESENT)
    ap.add_argument("--block", type=int, default=4_000, help="prompt rows per sparse-matmul block")
    ap.add_argument("--exclude", default="auto",
                    help="npz with 'drop' = kept-row indices to leave out (short-context exact duplicates); "
                         "'auto' = <world>/dedup_drop.npz when present, '' = none")


def run(args: argparse.Namespace) -> None:
    t0 = time.monotonic()
    MD = Path(args.world)
    out = Path(args.out) if args.out else MD / "assignments"
    out.mkdir(parents=True, exist_ok=True)

    cands = load_candidates(args.candidates or MD / "candidates" / "candidates.json.gz")
    n_cand = len(cands)
    cg = np.concatenate([np.asarray(c["members"], np.int64) for c in cands])
    cc = np.repeat(np.arange(n_cand), [len(c["members"]) for c in cands])
    M_gc = sparse.csr_matrix((np.ones(len(cg), np.int8), (cg, cc)),
                             shape=(config.N_GID, n_cand))
    print(f"{n_cand:,} candidates, {len(cg):,} member slots "
          f"({time.monotonic() - t0:.0f}s)", flush=True)

    kept = np.load(MD / "dedup_kept.npz")["kept"]
    n_rows = len(kept)
    excl_f = (MD / "dedup_drop.npz") if args.exclude == "auto" else (Path(args.exclude) if args.exclude else None)
    excluded = np.load(excl_f)["drop"] if excl_f is not None and excl_f.exists() else np.zeros(0, np.int64)
    print(f"excluding {len(excluded):,} kept rows (short-context duplicates)", flush=True)
    n_prompts = len(np.load(MD / "manifest.npz")["row_index"])
    row_of = np.full(n_prompts, -1, np.int64)
    row_of[kept] = np.arange(n_rows)

    rows_all, gids_all = [], []
    for L in range(config.N_LAYERS):
        with np.load(MD / "merged" / f"L{L:02d}.npz") as z:
            indptr, indices = z["indptr"], z["indices"]
        r = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
        m = row_of[r] >= 0
        if len(excluded):
            m &= ~np.isin(row_of[r], excluded)
        rows_all.append(row_of[r[m]])
        gids_all.append(L * config.D_MLP + indices[m].astype(np.int64))
    rows = np.concatenate(rows_all)
    gids = np.concatenate(gids_all)
    M_pg = sparse.csr_matrix((np.ones(len(rows), np.int8), (rows, gids)),
                             shape=(n_rows, config.N_GID))
    del rows_all, gids_all, rows, gids
    print(f"prompt x gid matrix: {M_pg.nnz:,} nnz ({time.monotonic() - t0:.0f}s)",
          flush=True)

    sizes = np.array([len(c["members"]) for c in cands])
    cap_of = np.where(sizes <= 64, args.cap,
                      np.where(sizes <= 256, args.cap_large, args.cap_giant))

    # pass 1: match counts only (no pair storage — giants match nearly everything)
    counts = np.zeros(n_cand, np.int64)
    for lo in range(0, n_rows, args.block):
        hi = min(lo + args.block, n_rows)
        prod = (M_pg[lo:hi].astype(np.int32) @ M_gc.astype(np.int32)).tocoo()
        m = prod.data >= args.min_present
        counts += np.bincount(prod.col[m], minlength=n_cand)
        if (lo // args.block) % 20 == 0:
            print(f"  count pass rows {hi:,}/{n_rows:,} "
                  f"({(time.monotonic() - t0) / 60:.1f} min)", flush=True)
    p_keep = np.minimum(1.0, cap_of / np.maximum(counts, 1))
    n_capped = int((counts > cap_of).sum())
    exp_pairs = int(np.minimum(counts, cap_of).sum())
    print(f"raw matches {counts.sum():,}; capped candidates {n_capped:,}; "
          f"expected kept pairs {exp_pairs:,}", flush=True)

    # pass 2: Bernoulli(cap/count) per pair, deterministic in (prompt, cand);
    # weight = 1/p (Horvitz-Thompson)
    pp, cc_, ww = [], [], []
    for lo in range(0, n_rows, args.block):
        hi = min(lo + args.block, n_rows)
        prod = (M_pg[lo:hi].astype(np.int32) @ M_gc.astype(np.int32)).tocoo()
        m = prod.data >= args.min_present
        pr = lo + prod.row[m].astype(np.int64)
        cd = prod.col[m].astype(np.int64)
        h = (pr.astype(np.uint64) * np.uint64(2654435761)
             ^ cd.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15))
        h = (h ^ (h >> np.uint64(33))) * np.uint64(0xFF51AFD7ED558CCD)
        u = ((h >> np.uint64(11)).astype(np.float64)) / float(2**53)
        k = u < p_keep[cd]
        pp.append(pr[k]); cc_.append(cd[k])
        ww.append((1.0 / p_keep[cd[k]]).astype(np.float32))
        if (lo // args.block) % 20 == 0:
            print(f"  sample pass rows {hi:,}/{n_rows:,}, kept "
                  f"{sum(len(x) for x in pp):,} ({(time.monotonic() - t0) / 60:.1f} min)",
                  flush=True)
    P = np.concatenate(pp); C = np.concatenate(cc_); weight = np.concatenate(ww)
    del pp, cc_, ww
    o = np.lexsort((C, P))
    np.savez_compressed(out / "pairs.npz", prompt=P[o].astype(np.int32),
                        cand=C[o].astype(np.int32), weight=weight[o])
    np.savez_compressed(out / "match_counts.npz", counts=counts.astype(np.int64))
    meta = {"pairs": int(len(P)), "candidates": n_cand, "min_present": args.min_present,
            "excluded_rows": int(len(excluded)),
            "capped_candidates": int(n_capped), "cap": args.cap,
            "prompts_with_match": int(len(np.unique(P))),
            "pairs_per_prompt_mean": float(len(P) / n_rows)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    print(f"done {(time.monotonic() - t0) / 60:.1f} min")
