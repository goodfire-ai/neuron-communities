"""Hub removal: rebuild the candidate dictionary from the raw CPM Leiden runs with the always-on (hub) neurons
stripped from every community. No Leiden rerun: every community of every gamma loses its hub members, and what
remains is kept when MIN_SIZE <= size <= SIZE_CAP. Communities that were over the size cap in Leiden are admitted
when they fit after removal. Identical member sets across gammas are merged (provenance in ``found``), so the output
has the record format of the ``leiden`` / ``merge`` stages plus ``hubs_removed`` (members lost):

  {"members", "span", "n_found", "stops": [min_gamma, max_gamma], "widths": [w, w],
   "found": ["lo-hi@cpm<gamma>", ...], "src": "cpm", "hubs_removed": k}

Writes ``<out>/candidates.json.gz`` and ``<out>/candidates_stats.json``.

  python -m neuron_communities hubs --out <world>/candidates --runs <world>/cand_cpm/runs [--hubs L2:4786,...]
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from neuron_communities import config



def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=Path, required=True, help="output dir (candidates.json.gz + candidates_stats.json)")
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="dirs holding run_*.json.gz of the leiden stage")
    parser.add_argument("--hubs", default=config.hub_string(), help="hub neurons 'L2:4786,L3:6390,...'")
    parser.add_argument("--size-cap", type=int, default=config.SIZE_CAP)
    parser.add_argument("--min-size", type=int, default=config.MIN_SIZE)
    parser.add_argument("--exclude-candidates", type=Path, default=None,
                        help="candidates.json.gz whose member sets are dropped (keep only NEW communities)")


def run(a: argparse.Namespace) -> None:
    hubs = {L * config.D_MLP + i for L, i in config.parse_hubs(a.hubs).items()}
    excl = set()
    if a.exclude_candidates:
        with gzip.open(a.exclude_candidates, "rt") as fh:
            excl = {tuple(sorted(c["members"])) for c in json.load(fh)}
        print(f"excluding {len(excl):,} member sets already in {a.exclude_candidates}")
    files = sorted(f for d in a.runs for f in Path(d).glob("run_*.json.gz"))
    agg: dict[tuple, dict] = {}
    st = Counter(); per_gamma = {}
    for f in files:
        with gzip.open(f, "rt") as fh:
            r = json.load(fh)
        lo, hi, g = r["lo"], r["hi"], r["param"]
        tag = f"{lo}-{hi}@cpm{g:g}"
        pg = per_gamma.setdefault(f"{g:g}", Counter())
        for rec in r["comms"]:
            members = rec[0] if isinstance(rec[0], list) else rec
            n0 = len(members)
            stripped = tuple(sorted(m for m in members if m not in hubs))
            lost = n0 - len(stripped)
            st["communities_in"] += 1; pg["in"] += 1
            if lost:
                st["touched"] += 1; pg["touched"] += 1
            if len(stripped) < a.min_size:
                st["dropped_small"] += 1; pg["dropped_small"] += 1
                continue
            if len(stripped) > a.size_cap:
                st["dropped_oversize"] += 1; pg["oversize"] += 1
                continue
            if n0 > a.size_cap:
                st["admitted_from_oversize"] += 1; pg["admitted_from_oversize"] += 1
            if stripped in excl:
                st["dropped_already_known"] += 1; pg["known"] += 1
                continue
            e = agg.setdefault(stripped, {"n_found": 0, "min_stop": g, "max_stop": g, "found": set(), "lost": lost})
            e["n_found"] += 1; e["found"].add(tag)
            e["min_stop"] = min(e["min_stop"], g); e["max_stop"] = max(e["max_stop"], g)
            pg["kept"] += 1
    out = []
    for members, e in agg.items():
        span = [members[0] // config.D_MLP, members[-1] // config.D_MLP]
        out.append({"members": list(members), "span": span, "n_found": e["n_found"], "stops": [e["min_stop"], e["max_stop"]],
                    "widths": [span[1] - span[0] + 1] * 2, "found": sorted(e["found"]), "src": "cpm", "hubs_removed": e["lost"]})
    out.sort(key=lambda c: (len(c["members"]), c["members"]))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with gzip.open(Path(a.out) / "candidates.json.gz", "wt") as fh:
        json.dump(out, fh)
    stats = {"runs": [str(f) for f in files], "hubs": sorted(hubs), "size_cap": a.size_cap, "min_size": a.min_size,
             "unique_candidates": len(out), "with_hubs_removed": sum(1 for c in out if c["hubs_removed"]),
             **dict(st), "per_gamma": {g: dict(c) for g, c in per_gamma.items()},
             "note": "post-hoc hub removal from raw Leiden runs; communities over the size cap in Leiden are admitted if <= cap after removal"}
    (Path(a.out) / "candidates_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"{len(files)} runs -> {len(out):,} unique candidates ({stats['with_hubs_removed']:,} lost hub members; "
          f"{st['admitted_from_oversize']:,} records admitted from over-cap communities; {st['dropped_small']:,} dropped as < {a.min_size}; "
          f"{st['dropped_oversize']:,} still over cap)")
    for g, c in sorted(per_gamma.items(), key=lambda kv: float(kv[0])):
        print(f"  gamma {g:>6s}: in {c['in']:7,}  kept {c['kept']:7,}  touched {c['touched']:6,}  from over-cap {c.get('admitted_from_oversize', 0):4,}  oversize {c.get('oversize', 0):4,}")
