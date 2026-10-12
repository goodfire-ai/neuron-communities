"""Community discovery: CPM Leiden on the z-gated co-occurrence graph of one layer window.

The graph of the window [lo, hi] is assembled from the census tiles ``pairs_L{a:02d}_L{b:02d}.npz`` with
lo <= a <= b <= hi: every stored pair with z >= CPM_EDGE_Z is an edge, weighted by its z; node ids are absolute
(layer * D_MLP + neuron). Leiden (``leidenalg.CPMVertexPartition``, seed 0) runs once per resolution gamma of the
grid; a community found at several gammas is one candidate whose provenance lists every discovery.

Candidates keep size 2..SIZE_CAP; larger communities are counted (``*_stats.json``, ``oversize``) but not pooled.

Writes ``<out-dir>/<out-name>.json.gz``:
  [{"members": [gid...], "span": [lo, hi], "n_found": k, "stops": [min_gamma, max_gamma],
    "widths": [w, w], "found": ["lo-hi@cpm<gamma>", ...], "src": "cpm"} ...]
plus ``<out-name>_stats.json`` (per-run node accounting and size histograms, the range check of the gamma grid).

Sharded discovery (one Leiden run per job): ``--n-runs`` prints the number of gammas, ``--run i`` writes
``<out-dir>/runs/run_<i>.json.gz`` ({lo, hi, param, n_runs, diag, comms: [[members, gamma, count, tags], ...]}),
``--merge-runs`` pools the run files into the candidates output. Without ``--run``, all gammas run in-process.

  python -m neuron_communities leiden --census-dir <world>/census --out-dir <world>/cand_cpm --n-runs
  python -m neuron_communities leiden --census-dir <world>/census --out-dir <world>/cand_cpm --run $SLURM_ARRAY_TASK_ID
  python -m neuron_communities leiden --census-dir <world>/census --out-dir <world>/cand_cpm --merge-runs
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np

from neuron_communities import config

t0 = time.monotonic()


def assemble(census_dir: Path, lo: int, hi: int) -> dict[str, np.ndarray]:
    """Edge arrays of the window: absolute node ids i < j, census z as weight."""
    ii, jj, zz = [], [], []
    for a in range(lo, hi + 1):
        for b in range(a, hi + 1):
            f_ = census_dir / f"pairs_L{a:02d}_L{b:02d}.npz"
            with np.load(f_) as f:
                i, j = f["i"].astype(np.int64), f["j"].astype(np.int64)
                z = np.nan_to_num(f["z"].astype(np.float32), posinf=65000.0)
            if a == b:
                gi, gj = a * config.D_MLP + i, a * config.D_MLP + j
            else:
                m = (i < config.D_MLP) & (j >= config.D_MLP)
                gi, gj = a * config.D_MLP + i[m], b * config.D_MLP + (j[m] - config.D_MLP)
                z = z[m]
            ii.append(gi); jj.append(gj); zz.append(z)
    i = np.concatenate(ii); j = np.concatenate(jj); z = np.concatenate(zz)
    print(f"assembled {len(i):,} edges, weight = z (p50 {np.median(z):.2f}, p99 {np.percentile(z, 99):.2f}) ({time.monotonic() - t0:.0f}s)", flush=True)
    return {"i": i, "j": j, "z": z}


def leiden(i, j, z, resolution, min_size=2):
    import igraph as ig
    import leidenalg
    # vectorized node relabelling + numpy edge array: the same graph (and, with seed 0, the same partition) as a
    # list-of-tuples build, at a fraction of the memory on 100M+-edge graphs
    nodes, inv = np.unique(np.concatenate([i, j]), return_inverse=True)
    graph = ig.Graph(n=len(nodes), edges=np.column_stack([inv[:len(i)], inv[len(i):]]))
    graph.es["weight"] = z.astype(float)
    part = leidenalg.find_partition(graph, leidenalg.CPMVertexPartition, weights="weight",
                                    resolution_parameter=resolution, seed=0)
    return [sorted(int(nodes[v]) for v in comm) for comm in part if len(comm) >= min_size]


def one_window(E: dict[str, np.ndarray], lo: int, hi: int, gammas: list[float], cpm_z: float, size_cap: int):
    """Leiden at every gamma of the window's z-gated graph; returns (lo, hi, discoveries, n_runs, diag)."""
    i, j, z = E["i"], E["j"], E["z"]
    found: dict[tuple, tuple[float, int]] = {}   # members -> (gamma of first discovery, count of discoveries)
    tags: dict[tuple, set] = {}                  # members -> {"cpm<gamma>"}
    n_runs = 0
    diag = []                                    # per run: node accounting + size histogram (range check for the gamma grid)
    sm = z >= cpm_z
    n_nodes = len(np.unique(np.concatenate([i[sm], j[sm]])))
    for res in gammas:
        n_runs += 1
        comms = leiden(i[sm], j[sm], z[sm], resolution=res)
        sizes = np.array([len(c) for c in comms]) if comms else np.zeros(0, int)
        kept_n = int(sizes[sizes <= size_cap].sum()); over_n = int(sizes[sizes > size_cap].sum())
        diag.append({"lo": lo, "hi": hi, "gamma": res, "edges": int(sm.sum()), "nodes": n_nodes,
                     "nodes_in_kept": kept_n, "nodes_over_cap": over_n, "singletons": n_nodes - kept_n - over_n,
                     "communities_kept": int((sizes <= size_cap).sum()), "communities_over_cap": int((sizes > size_cap).sum()),
                     "size_hist": {k: int(v) for k, v in zip(["2", "3-5", "6-15", "16-32", "33-64", "65-128", ">128"],
                                                            np.histogram(sizes, bins=[2, 3, 6, 16, 33, 65, 129, 10**9])[0])}})
        print(f"  window {lo}-{hi} cpm gamma {res:g}: {int(sm.sum()):,} edges -> {len(comms):,} communities; nodes kept/over-cap/single "
              f"{kept_n:,}/{over_n:,}/{n_nodes - kept_n - over_n:,} ({(time.monotonic() - t0) / 60:.1f} min)", flush=True)
        for comm in comms:
            key = tuple(comm)
            st, k = found.get(key, (res, 0))
            found[key] = (st, k + 1)
            tags.setdefault(key, set()).add(f"cpm{res:g}")
    return lo, hi, [(list(k), st, c, sorted(tags[k])) for k, (st, c) in found.items()], n_runs, diag


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--census-dir", type=Path, required=True, help="dir holding pairs_L??_L??.npz")
    parser.add_argument("--out-dir", type=Path, required=True, help="output dir (candidates + runs/)")
    parser.add_argument("--out-name", default="candidates")
    parser.add_argument("--window", default=f"{config.LAYER_TILE[0]}-{config.LAYER_TILE[1]}", help="layer window 'lo-hi'")
    parser.add_argument("--gammas", default=",".join(f"{g:g}" for g in config.GAMMAS), help="CPM resolutions (gamma), comma-separated")
    parser.add_argument("--cpm-z", type=float, default=config.CPM_EDGE_Z, help="edge threshold z for the CPM graph")
    parser.add_argument("--size-cap", type=int, default=config.SIZE_CAP)
    parser.add_argument("--run", type=int, default=None,
                        help="sharded discovery: do only gamma #i of the grid and write <out-dir>/runs/run_<i>.json.gz")
    parser.add_argument("--n-runs", action="store_true", help="print the number of sharded runs and exit")
    parser.add_argument("--merge-runs", action="store_true", help="pool the runs/ files of a sharded discovery into the candidates output")


def run(args: argparse.Namespace) -> None:
    lo, hi = (int(x) for x in args.window.split("-"))
    gammas = [float(x) for x in args.gammas.split(",")]
    cpm_z, size_cap = args.cpm_z, args.size_cap
    OUT = args.out_dir
    runs = [(lo, hi, p) for p in gammas]
    if args.n_runs:
        print(len(runs))
        return
    OUT.mkdir(parents=True, exist_ok=True)

    if args.run is not None:                       # sharded: one Leiden run, dumped for --merge-runs
        lo, hi, p = runs[args.run]
        E = assemble(args.census_dir, lo, hi)
        lo, hi, comms, n_runs, diag = one_window(E, lo, hi, [p], cpm_z, size_cap)
        (OUT / "runs").mkdir(exist_ok=True)
        with gzip.open(OUT / "runs" / f"run_{args.run:03d}.json.gz", "wt") as f:
            json.dump({"lo": lo, "hi": hi, "param": p, "n_runs": n_runs, "diag": diag,
                       "comms": [[m, st, c, tg] for m, st, c, tg in comms]}, f)
        print(f"run {args.run} ({lo}-{hi} @ {p:g}) done {(time.monotonic() - t0) / 60:.1f} min")
        return

    if args.merge_runs:                            # pool the shards exactly as one_window would
        files = sorted((OUT / "runs").glob("run_*.json.gz"))
        assert len(files) == len(runs), f"{len(files)} run files, expected {len(runs)}"
        per_win: dict[tuple, dict] = {}
        n_runs_w: dict[tuple, int] = {}
        diag_all: list = []
        for f_ in files:
            with gzip.open(f_, "rt") as f:
                r = json.load(f)
            diag_all += r.get("diag", [])
            key_w = (r["lo"], r["hi"])
            n_runs_w[key_w] = n_runs_w.get(key_w, 0) + r["n_runs"]
            d = per_win.setdefault(key_w, {})
            for m, st, c, tg in r["comms"]:
                key = tuple(m)
                st0, c0, tg0 = d.get(key, (st, 0, set()))
                d[key] = (min(st0, st), c0 + c, tg0 | set(tg))
        results = [(lo, hi, [(list(k), st, c, sorted(tg)) for k, (st, c, tg) in d.items()], n_runs_w[(lo, hi)])
                   for (lo, hi), d in per_win.items()]
        print(f"merged {len(files)} runs over {len(results)} windows", flush=True)
    else:
        E = assemble(args.census_dir, lo, hi)
        lo, hi, comms, n_runs, diag_all = one_window(E, lo, hi, gammas, cpm_z, size_cap)
        results = [(lo, hi, comms, n_runs)]

    # dedup: members -> aggregate provenance
    agg: dict[tuple, dict] = {}
    stats = []
    for lo, hi, comms, n_runs in results:
        w = hi - lo + 1
        stats.append({"lo": lo, "hi": hi, "runs": n_runs, "comms": len(comms)})
        for members, stop, cnt, tg in comms:
            key = tuple(members)
            a = agg.setdefault(key, {"n_found": 0, "min_stop": stop,
                                     "max_stop": stop, "min_w": w, "max_w": w, "found": set()})
            a["n_found"] += cnt
            for t in tg:
                a["found"].add((lo, hi, t))
            a["min_stop"] = min(a["min_stop"], stop)
            a["max_stop"] = max(a["max_stop"], stop)
            a["min_w"] = min(a["min_w"], w)
            a["max_w"] = max(a["max_w"], w)

    out, n_large, oversize = [], 0, []
    for members, a in agg.items():
        span = [min(members) // config.D_MLP, max(members) // config.D_MLP]
        if len(members) > size_cap:
            n_large += 1
            oversize.append([len(members), span[0], span[1], a["min_stop"]])
            continue
        out.append({"members": list(members), "span": span, "n_found": a["n_found"],
                    "stops": [a["min_stop"], a["max_stop"]],
                    "widths": [a["min_w"], a["max_w"]],
                    "found": sorted([f"{lo}-{hi}@{t}" for lo, hi, t in a["found"]]),
                    "src": "cpm"})
    out.sort(key=lambda c: (len(c["members"]), c["members"]))
    with gzip.open(OUT / f"{args.out_name}.json.gz", "wt") as f:
        json.dump(out, f)
    (OUT / f"{args.out_name}_stats.json").write_text(json.dumps(
        {"windows": stats, "unique": len(agg), "pooled": len(out),
         "dropped_gt_cap": n_large, "size_cap": size_cap, "weight": "z",
         "method": "cpm", "params": gammas, "cpm_stop": cpm_z,
         "oversize": oversize, "oversize_columns": ["size", "lo", "hi", "param"], "runs_diag": diag_all}))
    sizes = np.array([len(c["members"]) for c in out])
    spans = np.array([c["span"][1] - c["span"][0] + 1 for c in out])
    print(f"\nunique {len(agg):,}; pooled (size<= {size_cap}) {len(out):,}; "
          f"dropped {n_large:,} oversize")
    if len(out):
        print(f"size p50/p90/max: {np.percentile(sizes, [50, 90]).round(1)}/{sizes.max()}; "
              f"span p50/p90/max: {np.percentile(spans, [50, 90]).round(1)}/{spans.max()}")
    print(f"done {(time.monotonic() - t0) / 60:.1f} min")
