"""Stage ``summary``: coverage and dictionary statistics of a finished world.

Per world: union over all gamma stops of certified prompts (observed lower bound + Horvitz-Thompson estimate under the
per-community assignment cap), per-source union coverage, certified communities (deduplicated across gammas) with
>= 10 / >= 50 observed prompts, their median size and layer span, and the marginal unique prompts each gamma adds to the
union of the others. A pair is certified when ``screen/ns_results.npz`` kl_drop >= TAU_NEC and ``screen/subspan.npz``
kl_sub <= TAU_SUFF (the best sufficiency over the layer-necessary sub-spans).

Writes ``coverage/tail_stats.json``, ``coverage/certified_prompts.npy`` (for cross-world overlap) and
``coverage/certified_communities.npz``, and prints the JSON. With ``--no-hubs`` communities made mostly of hub neurons
(>= ``--hub-frac`` of their members) are dropped first and the files get a ``_nohub`` suffix: such communities match
every prompt and pass on a few percent of generic text through the hubs alone.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from neuron_communities import config
from neuron_communities.data.io import load_candidates, prompt_sources, source_names



def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--world", required=True, help="world dir with assignments/, screen/, candidates/, manifest.npz, dedup_*.npz")
    ap.add_argument("--tau-nec", type=float, default=config.TAU_NEC)
    ap.add_argument("--tau-suff", type=float, default=config.TAU_SUFF)
    ap.add_argument("--no-hubs", action="store_true", help="drop certified communities with >= --hub-frac hub members")
    ap.add_argument("--hub-frac", type=float, default=0.5)
    ap.add_argument("--hubs", default=config.hub_string(), help="hub neurons, 'L2:4786,L3:6390,...'")


def run(a: argparse.Namespace) -> None:
    TAU_N, TAU_S = a.tau_nec, a.tau_suff
    NO_HUBS, HUB_FRAC = a.no_hubs, a.hub_frac
    HUBS = {l * config.D_MLP + i for l, i in config.parse_hubs(a.hubs).items()}
    SUF = "_nohub" if NO_HUBS else ""
    W = Path(a.world)
    S = W / "screen"
    with np.load(W / "assignments/pairs.npz") as z:
        kd = np.load(S / "ns_results.npz")["kl_drop"]
        nec = np.flatnonzero(np.nan_to_num(kd, nan=0) >= TAU_N); del kd
        ksub = np.load(S / "subspan.npz")["kl_sub"][nec]
        cert_idx = nec[np.nan_to_num(ksub, nan=9) <= TAU_S]; del ksub, nec
        P = z["prompt"][cert_idx].astype(np.int64); C = z["cand"][cert_idx].astype(np.int64)
        WGT = z["weight"][cert_idx].astype(np.float64) if "weight" in z.files else np.ones(len(P))
        n_pairs = len(z["prompt"])
    cands = load_candidates(W / "candidates/candidates.json.gz")
    hub_comms = {ci for ci, c in enumerate(cands) if len(HUBS.intersection(c["members"])) >= HUB_FRAC * len(c["members"])}
    n_hub_cert = int(np.isin(np.unique(C), list(hub_comms)).sum())
    if NO_HUBS:
        keep = ~np.isin(C, list(hub_comms)); P, C, WGT = P[keep], C[keep], WGT[keep]
        print(f"{W.name}: dropped {n_hub_cert} certified hub communities ({(~keep).sum():,} certified pairs)", flush=True)
    meta = json.loads((W / "meta.json").read_text()) if (W / "meta.json").exists() else {}
    names = source_names(meta)
    kept = np.load(W / "dedup_kept.npz")["kept"]                      # prompt id p in pairs -> manifest row kept[p]
    drop = np.load(W / "dedup_drop.npz")["drop"] if (W / "dedup_drop.npz").exists() else np.array([], np.int64)
    psrc = prompt_sources(W, kept)                                    # source per prompt id
    live = np.ones(len(kept), bool); live[drop] = False
    n_kept = int(live.sum()); denom = np.bincount(psrc[live], minlength=len(names))

    # per community: observed prompts, HT weight sum, per-prompt miss prob
    order = np.argsort(C, kind="stable"); P, C, WGT = P[order], C[order], WGT[order]
    bounds = np.flatnonzero(np.diff(np.concatenate([[-1], C, [-1]])))
    cis = C[bounds[:-1]]
    n_obs = np.diff(bounds); w_sum = np.add.reduceat(WGT, bounds[:-1])
    gammas_of = defaultdict(list)
    for ci in cis.tolist():
        for t in cands[ci]["found"]:
            gammas_of[float(t.split("@")[1][3:])].append(ci)
    size = np.array([len(cands[ci]["members"]) for ci in cis])
    span = np.array([len({m // config.D_MLP for m in cands[ci]["members"]}) for ci in cis])
    span_w = np.array([(max(m // config.D_MLP for m in cands[ci]["members"]) - min(m // config.D_MLP for m in cands[ci]["members"]) + 1) for ci in cis])

    # union (all gammas): observed + HT
    up = np.unique(P)
    miss = np.ones(len(up)); pos = np.searchsorted(up, P)
    np.multiply.at(miss, pos, 1.0 - 1.0 / WGT)
    capped = bool((WGT > 1).any())
    ht = float((1.0 / (1.0 - miss)).sum()) if capped else float(len(up))
    per_src = np.bincount(psrc[up], minlength=len(names))
    ht_src = np.bincount(psrc[up], weights=1.0 / (1.0 - miss), minlength=len(names)) if capped else per_src.astype(float)

    # where the HT mass sits: communities by sampling weight band (weight = hits / cap; 1 = exhaustive)
    w_c = np.array([WGT[bounds[k]] for k in range(len(cis))])          # weight is constant within a community
    bands = [("w=1 (uncapped)", 1.0, 1.0), ("1<w<=2", 1.0 + 1e-9, 2.0), ("2<w<=10", 2.0 + 1e-9, 10.0), ("w>10", 10.0 + 1e-9, np.inf)]
    band_stats = {}
    for lab, lo, hi in bands:
        sel = (w_c >= lo) & (w_c <= hi)
        idx = np.concatenate([np.arange(bounds[k], bounds[k + 1]) for k in np.flatnonzero(sel)]) if sel.any() else np.array([], np.int64)
        band_stats[lab] = {"communities": int(sel.sum()), "ge10": int((n_obs[sel] >= 10).sum()), "unique_prompts_observed": int(len(np.unique(P[idx]))),
                           "certified_pairs_weighted": float(w_sum[sel].sum()), "median_size": float(np.median(size[sel])) if sel.any() else None}
    # weight = matches / cap, so matches = cap x weight (cap of the assignment run; the diagnostic uses the base cap)
    cap = config.ASSIGN_CAP
    if (W / "assignments/meta.json").exists():
        cap = int(json.loads((W / "assignments/meta.json").read_text()).get("cap", cap))
    top = np.argsort(-(w_sum - n_obs))[:10]
    top_capped = [{"cand": int(cis[k]), "size": int(size[k]), "weight": float(w_c[k]), "est_hits": int(round(cap * w_c[k])),
                   "certified_observed": int(n_obs[k]), "certified_ht": float(w_sum[k]), "cert_rate": float(n_obs[k] / min(cap * w_c[k], cap))} for k in top if w_c[k] > 1]
    up_unc = np.unique(P[np.concatenate([np.arange(bounds[k], bounds[k + 1]) for k in np.flatnonzero(w_c == 1)])]) if (w_c == 1).any() else np.array([], np.int64)

    # marginal value of each gamma: prompts lost from the union if that gamma is removed
    prompts_by_gamma = {}
    ci_to_slot = {int(ci): k for k, ci in enumerate(cis.tolist())}
    for g, lst in gammas_of.items():
        sl = np.array([ci_to_slot[ci] for ci in lst])
        idx = np.concatenate([np.arange(bounds[k], bounds[k + 1]) for k in sl])
        prompts_by_gamma[g] = np.unique(P[idx])
    cnt = np.zeros(len(up), np.int64)
    for g, ps in prompts_by_gamma.items():
        cnt[np.searchsorted(up, ps)] += 1
    marginal = {g: int((cnt[np.searchsorted(up, ps)] == 1).sum()) for g, ps in prompts_by_gamma.items()}

    out = {
        "world": str(W), "n_kept": int(n_kept), "hub_communities_certified": n_hub_cert, "hubs_excluded": NO_HUBS, "n_pairs_assigned": int(n_pairs), "certified_pairs": int(len(P)),
        "cap_active": capped, "max_weight": float(WGT.max()) if len(WGT) else 1.0,
        "union_prompts_observed": int(len(up)), "union_frac_observed": len(up) / n_kept,
        "union_prompts_ht": ht, "union_frac_ht": ht / n_kept,
        "per_source": {names[i]: {"kept": int(denom[i]), "covered": int(per_src[i]), "frac": per_src[i] / max(denom[i], 1),
                                  "covered_ht": float(ht_src[i]), "frac_ht": ht_src[i] / max(denom[i], 1)} for i in range(len(names))},
        "certified_communities": int(len(cis)),
        "certified_ge10": int((n_obs >= 10).sum()), "certified_ge50": int((n_obs >= 50).sum()), "certified_ge100": int((n_obs >= 100).sum()),
        "certified_ge500": int((n_obs >= 500).sum()), "certified_ge1000": int((n_obs >= 1000).sum()),
        "certified_ge100_ht": int((w_sum >= 100).sum()), "certified_ge500_ht": int((w_sum >= 500).sum()), "certified_ge1000_ht": int((w_sum >= 1000).sum()),
        "certified_ge10_ht": int((w_sum >= 10).sum()), "certified_ge50_ht": int((w_sum >= 50).sum()),
        "certified_capped_communities": int((w_c > 1).sum()),
        "union_prompts_observed_uncapped_only": int(len(up_unc)), "union_frac_observed_uncapped_only": len(up_unc) / n_kept,
        "weight_bands": band_stats, "top_capped_communities": top_capped,
        "median_size_all": float(np.median(size)) if len(size) else None,
        "median_size_ge10": float(np.median(size[n_obs >= 10])) if (n_obs >= 10).any() else None,
        "median_layers_ge10": float(np.median(span[n_obs >= 10])) if (n_obs >= 10).any() else None,
        "median_span_width_ge10": float(np.median(span_w[n_obs >= 10])) if (n_obs >= 10).any() else None,
        "size_hist_ge10": {f"{lo}-{hi}": int(((size >= lo) & (size <= hi) & (n_obs >= 10)).sum()) for lo, hi in [(2, 5), (6, 15), (16, 40), (41, 128)]},
        "per_gamma": {f"{g:g}": {"certified": len(gammas_of[g]), "ge10": int(sum(n_obs[ci_to_slot[ci]] >= 10 for ci in gammas_of[g])),
                                 "prompts": int(len(prompts_by_gamma[g])), "marginal_unique_prompts": marginal[g]} for g in sorted(gammas_of)},
    }
    (W / "coverage").mkdir(exist_ok=True)
    (W / f"coverage/tail_stats{SUF}.json").write_text(json.dumps(out, indent=1))
    np.save(W / f"coverage/certified_prompts{SUF}.npy", up.astype(np.int32))
    np.savez_compressed(W / f"coverage/certified_communities{SUF}.npz", cand=cis.astype(np.int32), size=size.astype(np.int16), n_layers=span.astype(np.int8),
                        span_width=span_w.astype(np.int8), certified_observed=n_obs.astype(np.int32), certified_ht=w_sum.astype(np.float32), weight=w_c.astype(np.float32))
    print(json.dumps(out, indent=1), flush=True)
