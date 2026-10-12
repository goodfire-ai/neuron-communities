"""Merge the discoveries of one or more ``leiden`` outputs into one candidate pool, deduplicated by member set,
with per-approach provenance.

  found: ["0-29@cpm2", "0-29@cpm4", ...]   (window@approach; approach = cpm<gamma>)
  src:   "cpm"

Writes ``<out>/candidates.json.gz`` with records
  {"members", "span", "n_found", "found", "src", "stops": [0, 0], "widths": [w, w], "n_layers"}
sorted by (size, members), plus ``<out>/candidates_stats.json`` (pool size, discoveries per approach, size and
layer-count percentiles, and the number of communities spanning more than MAX_SPAN_LAYERS member layers, which the
sub-span stage will not test).

  python -m neuron_communities merge --inputs <world>/cand_cpm/candidates.json.gz --out <world>/candidates
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

from neuron_communities import config



def retag(tag: str, method: str) -> str:
    win, z = tag.split("@")
    if method == "cpm":                       # "8/cpm2" or "cpm2" -> "cpm2"
        return f"{win}@{z.split('/')[-1]}"
    return f"{win}@z{z}"


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--inputs", type=Path, nargs="+", required=True, help="candidates.json.gz files written by the leiden stage")
    parser.add_argument("--out", type=Path, required=True, help="output dir (candidates.json.gz + candidates_stats.json)")
    parser.add_argument("--keep-windows", default=None,
                        help="comma-separated windows 'lo-hi' whose discoveries are kept (default all); a community found only in other windows is dropped")


def run(args: argparse.Namespace) -> None:
    pool: dict[tuple, dict] = {}
    keep_windows = set(filter(None, (args.keep_windows or "").split(",")))
    method = "cpm"
    for path in args.inputs:
        with gzip.open(path, "rt") as f:
            cands = json.load(f)
        for c in cands:
            if keep_windows:
                c["found"] = [t for t in c["found"] if t.split("@")[0] in keep_windows]
                if not c["found"]:
                    continue
            key = tuple(c["members"])
            e = pool.setdefault(key, {"members": list(key), "span": c["span"], "n_found": 0, "found": set(), "src": set()})
            e["n_found"] += c["n_found"]
            e["found"].update(retag(t, method) for t in c["found"])
            e["src"].add(method)
        print(f"{path}: {len(cands):,} communities; pool now {len(pool):,}")

    out = []
    for e in pool.values():
        layers = sorted({g // config.D_MLP for g in e["members"]})
        out.append({"members": e["members"], "span": e["span"], "n_found": e["n_found"],
                    "found": sorted(e["found"]), "src": "both" if len(e["src"]) == 2 else next(iter(e["src"])),
                    "stops": [0, 0], "widths": [e["span"][1] - e["span"][0] + 1] * 2, "n_layers": len(layers)})
    out.sort(key=lambda c: (len(c["members"]), c["members"]))
    args.out.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out / "candidates.json.gz", "wt") as f:
        json.dump(out, f)
    appr = Counter(t for c in out for t in c["found"])
    sizes = np.array([len(c["members"]) for c in out]); nl = np.array([c["n_layers"] for c in out])
    stats = {"pooled": len(out), "src": dict(Counter(c["src"] for c in out)), "per_approach": dict(sorted(appr.items())),
             "size_p50_p90_max": [float(np.percentile(sizes, 50)), float(np.percentile(sizes, 90)), int(sizes.max())],
             "layers_p50_p90_max": [float(np.percentile(nl, 50)), float(np.percentile(nl, 90)), int(nl.max())],
             f"gt{config.MAX_SPAN_LAYERS}_layers": int((nl > config.MAX_SPAN_LAYERS).sum())}
    (args.out / "candidates_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
