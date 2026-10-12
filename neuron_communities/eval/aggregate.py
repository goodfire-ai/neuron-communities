"""Stage ``aggregate``: collect the necessity / sub-span shards of a world and certify its prompts.

  necessity   necessity/nec_shard*.npz -> screen/kl_drop.npy (aligned with assignments/pairs.npz) and
              subspan/pairs.npz (the necessary pairs, kl_drop >= TAU_NEC), the input of the ``subspan`` stage
  subspan     subspan/subspan_shard*.npz -> screen/ns_results.npz (kl_drop, kl_layer_only = full-span
              "only members + hubs", legacy fields NaN), screen/subspan.npz (kl_sub, sub_lo, sub_hi, n_explained),
              layer_necessity/layer_necessity.npz (full-span "hubs only" row = layer necessity of the involved layers),
              screen/subspan_table.json.gz and subspan/rows_sorted.npz (every sub-span row with its certification flag),
              then ``coverage``
  coverage    per-approach (tile x method x parameter) certified pairs -> coverage/summary.json, coverage/curves.npz,
              and the one-span-per-community variant coverage/summary_perspan.json, coverage/curves_perspan.npz

Certified pair: kl_drop >= TAU_NEC and some contiguous sub-span of the involved layers with "hubs only" >= TAU_LAYER
(layer-necessary) and "only members + hubs" <= TAU_SUFF (sufficient). ``kl_sub`` is the best "only" value over the
layer-necessary sub-spans, ``sub_lo``/``sub_hi`` the widest explained sub-span.

File formats:
  screen/kl_drop.npy            float32 per assignments pair, NaN where no shard evaluated it
  screen/subspan.npz            kl_sub (float32), sub_lo / sub_hi (int16 layer indices, -1 = none), n_explained (int16)
  screen/subspan_table.json.gz  {cand: [[lo, hi, n_in_span_members, n_ns, n_layer_nec, n_nec, median_only], ...]}, one row
                                per contiguous sub-span [lo, hi] of the community's involved layers
  subspan/rows_sorted.npz       every (prompt, cand, a, b, only, off, ns) row sorted by pair key (prompt * 1e7 + cand) with
                                key_u / starts for lookup
  coverage/curves.npz           per approach "<tile> <method><param>|neurons" / "|prompts": unique prompts covered against
                                cumulative neurons, communities in density order (certified prompts per neuron, best first)
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from neuron_communities import config
from neuron_communities.data.io import approach_of, load_candidates, n_live_prompts



def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("mode", choices=["necessity", "subspan", "coverage"])
    ap.add_argument("--world", required=True, help="world dir with assignments/, necessity/, subspan/, candidates/")
    ap.add_argument("--tau-nec", type=float, default=config.TAU_NEC)
    ap.add_argument("--tau-suff", type=float, default=config.TAU_SUFF)
    ap.add_argument("--tau-layer", type=float, default=config.TAU_LAYER)


def key_of(p, c):
    return p * 10_000_000 + c


def _pairs(W: Path):
    with np.load(W / "assignments/pairs.npz") as z:
        return z["prompt"].astype(np.int64), z["cand"].astype(np.int64)


def aggregate_necessity(W: Path, TAU_N: float) -> None:
    S = W / "screen"; S.mkdir(exist_ok=True)
    P, C = _pairs(W)
    parts = sorted((W / "necessity").glob("nec_shard*.npz"))
    sp = np.concatenate([np.load(f)["prompt"] for f in parts]).astype(np.int64)
    sc = np.concatenate([np.load(f)["cand"] for f in parts]).astype(np.int64)
    sk = np.concatenate([np.load(f)["kl_drop"] for f in parts]).astype(np.float32)
    k_all = key_of(P, C); k_got = key_of(sp, sc)
    o = np.argsort(k_got); k_got, sk = k_got[o], sk[o]
    pos = np.searchsorted(k_got, k_all); pos = np.minimum(pos, len(k_got) - 1)
    hit = k_got[pos] == k_all
    kd = np.full(len(P), np.nan, np.float32); kd[hit] = sk[pos[hit]]
    np.save(S / "kl_drop.npy", kd)
    nec = hit & (kd >= TAU_N)
    (W / "subspan").mkdir(exist_ok=True)
    np.savez(W / "subspan/pairs.npz", prompt=P[nec].astype(np.int32), cand=C[nec].astype(np.int32))
    print(f"{len(parts)} shards; {hit.sum():,}/{len(P):,} pairs evaluated; necessary (>= {TAU_N}): {nec.sum():,} "
          f"over {len(np.unique(P[nec])):,} prompts and {len(np.unique(C[nec])):,} communities")


def aggregate_subspan(W: Path, TAU_N: float, TAU_S: float, TAU_L: float) -> None:
    S = W / "screen"; S.mkdir(exist_ok=True)
    P, C = _pairs(W)
    cands = load_candidates(W / "candidates/candidates.json.gz")
    layers = [sorted({g // config.D_MLP for g in c["members"]}) for c in cands]
    kd = np.load(S / "kl_drop.npy")
    parts = [np.load(f) for f in sorted((W / "subspan").glob("subspan_shard*.npz"))]
    sp = np.concatenate([p["prompt"] for p in parts]).astype(np.int64); sc = np.concatenate([p["cand"] for p in parts]).astype(np.int64)
    a_ = np.concatenate([p["a"] for p in parts]).astype(int); b_ = np.concatenate([p["b"] for p in parts]).astype(int)
    ko = np.concatenate([p["kl_only"] for p in parts]).astype(np.float32); kf = np.concatenate([p["kl_off"] for p in parts]).astype(np.float32)
    n = len(P)
    kl_sub = np.full(n, np.nan, np.float32); lo = np.full(n, -1, np.int16); hi = np.full(n, -1, np.int16); nexp = np.zeros(n, np.int16)
    k_only_full = np.full(n, np.nan, np.float32); k_off_full = np.full(n, np.nan, np.float32)
    k_all = key_of(P, C); o_all = np.argsort(k_all); k_sorted = k_all[o_all]
    key = key_of(sp, sc)
    order = np.argsort(key, kind="stable"); key, a_, b_, ko, kf, sc = key[order], a_[order], b_[order], ko[order], kf[order], sc[order]
    bounds = np.flatnonzero(np.diff(key)) + 1; starts = np.r_[0, bounds]; ends = np.r_[bounds, len(key)]
    grp_keys = key[starts]
    pos = np.searchsorted(k_sorted, grp_keys); pos = np.minimum(pos, n - 1); ok = k_sorted[pos] == grp_keys
    rows = np.where(ok, o_all[pos], -1)
    for gi, (s0, e0) in enumerate(zip(starts, ends)):
        r = rows[gi]
        if r < 0:
            continue
        L = layers[int(sc[s0])]; k = len(L)
        full = (a_[s0:e0] == 0) & (b_[s0:e0] == k - 1)
        if full.any():
            j = s0 + int(np.argmax(full)); k_only_full[r] = ko[j]; k_off_full[r] = kf[j]
        valid = kf[s0:e0] >= TAU_L
        if valid.any():
            kl_sub[r] = ko[s0:e0][valid].min()
        expl = valid & (ko[s0:e0] <= TAU_S)
        nexp[r] = int(expl.sum())
        if expl.any():
            w = np.where(expl, b_[s0:e0] - a_[s0:e0], -1); j = int(np.argmax(w))
            lo[r] = L[a_[s0:e0][j]]; hi[r] = L[b_[s0:e0][j]]
    nan = np.full(n, np.nan, np.float32)
    np.savez_compressed(S / "ns_results.npz", kl_drop=kd, kl_range_only=nan, kl_layer_only=k_only_full, kl_sub_drop=nan, kl_sub_range=nan)
    np.savez_compressed(S / "subspan.npz", kl_sub=kl_sub, sub_lo=lo, sub_hi=hi, n_explained=nexp)
    (S / "meta.json").write_text(json.dumps({"tau_n": TAU_N, "tau_s": TAU_S, "readout": "sub-span",
                                             "pairs": n, "evaluated_subspan": int(np.isfinite(k_off_full).sum())}))
    # layer necessity file = full-span "hubs only" rows, keyed by (prompt, layer bitmask)
    m = np.isfinite(k_off_full)
    lmask = np.array([sum(1 << l for l in L) for L in layers], np.int64)[C[m]]
    (W / "layer_necessity").mkdir(exist_ok=True)
    np.savez_compressed(W / "layer_necessity/layer_necessity.npz", prompt=P[m], layer_mask=lmask, kl=k_off_full[m],
                        target_lp=np.full(m.sum(), np.nan, np.float32), target_lp_clean=np.full(m.sum(), np.nan, np.float32))
    nec = np.nan_to_num(kd, nan=0) >= TAU_N
    cert = nec & (np.nan_to_num(kl_sub, nan=9) <= TAU_S)
    print(f"{len(parts)} shards, {len(key):,} sub-span rows; necessary pairs {nec.sum():,}, with sub-span rows {m.sum():,}, "
          f"certified {cert.sum():,} over {len(np.unique(P[cert])):,} prompts / {len(np.unique(C[cert])):,} communities")
    del parts, sp, a_, b_, ko, kf, sc, key
    subspan_table(W, TAU_N, TAU_S, TAU_L)


def subspan_table(W: Path, TAU_N: float, TAU_S: float, TAU_L: float) -> None:
    """Per-community sub-span table (one row per contiguous sub-span tried) and the sorted row store.

    screen/subspan_table.json.gz : {cand: [[lo, hi, n_in_span_members, n_ns, n_layer_nec, n_nec, median_only], ...]} at the
        fixed certification thresholds: necessary kl_drop >= TAU_NEC; layer-necessary "hubs only" >= TAU_LAYER; N&S
        additionally "members + hubs only" <= TAU_SUFF. median_only is over the community's necessary prompts.
    subspan/rows_sorted.npz : every (prompt, cand, a, b, only, off) row sorted by pair key (prompt*1e7 + cand) with
        `key_u` / `starts` for lookup, so a browser can show each prompt's values on every sub-span.
    """
    t0 = time.monotonic()
    P, C = _pairs(W)
    kd = np.load(W / "screen/ns_results.npz")["kl_drop"]
    cands = load_candidates(W / "candidates/candidates.json.gz")
    layers = [sorted({g // config.D_MLP for g in c["members"]}) for c in cands]
    parts = [np.load(f) for f in sorted((W / "subspan").glob("subspan_shard*.npz"))]
    sp = np.concatenate([p["prompt"] for p in parts]).astype(np.int64); sc = np.concatenate([p["cand"] for p in parts]).astype(np.int64)
    a_ = np.concatenate([p["a"] for p in parts]).astype(np.int8); b_ = np.concatenate([p["b"] for p in parts]).astype(np.int8)
    ko = np.concatenate([p["kl_only"] for p in parts]).astype(np.float16); kf = np.concatenate([p["kl_off"] for p in parts]).astype(np.float16)
    del parts
    print(f"{len(sp):,} rows loaded ({time.monotonic() - t0:.0f}s)", flush=True)

    # necessity of each row's pair
    k_pairs = P * 10_000_000 + C; o = np.argsort(k_pairs); k_pairs_s = k_pairs[o]; kd_s = kd[o]
    key = sp * 10_000_000 + sc
    pos = np.searchsorted(k_pairs_s, key); pos = np.minimum(pos, len(k_pairs_s) - 1)
    row_kd = np.where(k_pairs_s[pos] == key, kd_s[pos], np.nan).astype(np.float32)
    nec = row_kd >= TAU_N
    ln = kf.astype(np.float32) >= TAU_L
    ns = nec & ln & (ko.astype(np.float32) <= TAU_S)

    # sorted store for the browser
    order = np.argsort(key, kind="stable")
    key_s = key[order]
    bounds = np.flatnonzero(np.diff(key_s)) + 1
    np.savez(W / "subspan/rows_sorted.npz", key_u=key_s[np.r_[0, bounds]], starts=np.r_[0, bounds, len(key_s)].astype(np.int64),
             a=a_[order], b=b_[order], only=ko[order], off=kf[order], ns=ns[order], prompt=sp[order].astype(np.int32), cand=sc[order].astype(np.int32))
    print(f"rows_sorted.npz written ({time.monotonic() - t0:.0f}s)", flush=True)

    # per (cand, a, b) counts
    g = sc * 1024 + a_.astype(np.int64) * 32 + b_.astype(np.int64)
    gu, inv = np.unique(g, return_inverse=True)
    n_ns = np.bincount(inv, weights=ns, minlength=len(gu)).astype(int)
    n_ln = np.bincount(inv, weights=nec & ln, minlength=len(gu)).astype(int)
    n_nec = np.bincount(inv, weights=nec, minlength=len(gu)).astype(int)
    # median "only" over necessary rows per group
    med = np.full(len(gu), np.nan, np.float32)
    sel = np.flatnonzero(nec)
    og = np.lexsort((ko[sel].astype(np.float32), inv[sel])); gs, vs = inv[sel][og], ko[sel][og].astype(np.float32)
    gb = np.flatnonzero(np.diff(gs)) + 1; st = np.r_[0, gb]; en = np.r_[gb, len(gs)]
    med[gs[st]] = vs[(st + en - 1) // 2]
    table: dict[int, list] = {}
    cand_of_g = gu // 1024; a_of_g = (gu % 1024) // 32; b_of_g = gu % 32
    for i in range(len(gu)):
        c = int(cand_of_g[i]); L = layers[c]; lo, hi = L[int(a_of_g[i])], L[int(b_of_g[i])]
        n_in = sum(1 for x in cands[c]["members"] if lo <= x // config.D_MLP <= hi)
        table.setdefault(c, []).append([int(lo), int(hi), int(n_in), int(n_ns[i]), int(n_ln[i]), int(n_nec[i]),
                                        None if np.isnan(med[i]) else round(float(med[i]), 3)])
    for c in table:
        table[c].sort(key=lambda r: (r[0], r[1]))
    with gzip.open(W / "screen/subspan_table.json.gz", "wt") as f:
        json.dump({str(c): rows for c, rows in table.items()}, f)
    n_any = sum(1 for rows in table.values() if any(r[3] > 0 for r in rows))
    print(f"{len(table):,} communities with sub-span rows, {n_any:,} with an N&S sub-span; {len(gu):,} (community, span) rows "
          f"-> {W / 'screen/subspan_table.json.gz'} ({time.monotonic() - t0:.0f}s)")


def _approaches(cands) -> dict[tuple, list]:
    """(tile, method, parameter) -> community indices discovered there (``found`` tags)."""
    appr: dict[tuple, list] = defaultdict(list)
    for ci, c in enumerate(cands):
        for t in c["found"]:
            appr[approach_of(t)].append(ci)
    return appr


def coverage(W: Path, TAU_N: float, TAU_S: float) -> None:
    S = W / "screen"
    P, C = _pairs(W)
    with np.load(W / "assignments/pairs.npz") as z:   # Horvitz-Thompson weights of the per-community prompt cap (1 = exhaustive)
        WGT = z["weight"].astype(np.float64) if "weight" in z.files else np.ones(len(P))
    cands = load_candidates(W / "candidates/candidates.json.gz")
    kd = np.load(S / "ns_results.npz")["kl_drop"]; ksub = np.load(S / "subspan.npz")["kl_sub"]
    cert = (np.nan_to_num(kd, nan=0) >= TAU_N) & (np.nan_to_num(ksub, nan=9) <= TAU_S)
    n_kept = n_live_prompts(W)
    size = np.array([len(c["members"]) for c in cands])
    Pc, Cc, Wc = P[cert], C[cert], WGT[cert]
    prompts_of: dict[int, set] = defaultdict(set)
    wsum_of: dict[int, float] = defaultdict(float)        # weighted certified prompts per community (unbiased under the cap)
    qmiss_of: dict[int, dict] = defaultdict(dict)          # community -> {prompt: 1 - q} with q = 1/weight = sampling prob
    for p, c, w in zip(Pc.tolist(), Cc.tolist(), Wc.tolist()):
        prompts_of[c].add(p); wsum_of[c] += w; qmiss_of[c][p] = 1.0 - 1.0 / w
    capped = bool((WGT > 1.0).any())
    if capped:
        print(f"assignment cap active: {int((WGT > 1).sum()):,} of {len(WGT):,} pairs carry weight > 1 (max {WGT.max():.1f}); "
              f"coverage reports the observed unique prompts (lower bound) and a Horvitz-Thompson estimate", flush=True)
    # approaches: tile x method x parameter
    appr = _approaches(cands)
    summary, curves = [], {}
    for (tile, method, param), cis in sorted(appr.items()):
        used = [ci for ci in cis if ci in prompts_of]
        # density order: (weighted) certified prompts per neuron, best first; unique prompts vs cumulative neurons / communities
        used.sort(key=lambda ci: -wsum_of[ci] / size[ci])
        u: set = set(); ys = []; xs_n = np.cumsum([size[ci] for ci in used])
        miss: dict[int, float] = {}                        # prompt -> prod over its certified sampled communities of (1 - q_c)
        for ci in used:
            u |= prompts_of[ci]; ys.append(len(u))
            for p_, m_ in qmiss_of[ci].items():
                miss[p_] = miss.get(p_, 1.0) * m_
        ht = float(sum(1.0 / (1.0 - m_) for m_ in miss.values())) if capped else float(len(u))
        name = f"{tile} {method}{param:g}"
        curves[name + "|neurons"] = xs_n.astype(np.int64); curves[name + "|prompts"] = np.array(ys, np.int64)
        summary.append({"tile": tile, "method": method, "param": param, "communities": len(cis), "certified_communities": len(used),
                        "prompts_covered": len(u), "prompts_frac": len(u) / n_kept, "neurons": int(xs_n[-1]) if len(used) else 0,
                        "certified_pairs": int(sum(len(prompts_of[ci]) for ci in used)),
                        "certified_pairs_weighted": float(sum(wsum_of[ci] for ci in used)),
                        "prompts_covered_ht": ht, "prompts_frac_ht": ht / n_kept,
                        "neurons_per_prompt": float(xs_n[-1] / max(len(u), 1)) if len(used) else float("nan")})
    (W / "coverage").mkdir(exist_ok=True)
    (W / "coverage/summary.json").write_text(json.dumps({"n_prompts": n_kept, "tau_n": TAU_N, "tau_s": TAU_S, "approaches": summary}, indent=1))
    np.savez_compressed(W / "coverage/curves.npz", **curves)
    print(f"{'tile':8s} {'approach':8s} {'comms':>7s} {'cert':>6s} {'prompts':>8s} {'%':>5s} {'HT est':>9s} {'HT %':>5s} {'neurons':>8s} {'n/prompt':>8s}")
    for s in summary:
        print(f"{s['tile']:8s} {s['method'] + format(s['param'], 'g'):8s} {s['communities']:7,} {s['certified_communities']:6,} "
              f"{s['prompts_covered']:8,} {100 * s['prompts_frac']:5.1f} {s['prompts_covered_ht']:9,.0f} {100 * s['prompts_frac_ht']:5.1f} {s['neurons']:8,} {s['neurons_per_prompt']:8.1f}")
    print(f"wrote {W / 'coverage'}")
    coverage_perspan(W, TAU_N, TAU_S)


def coverage_perspan(W: Path, TAU_N: float, TAU_S: float) -> None:
    """One span per community: the widest sub-span with >= min_prompts N&S prompts; the community's prompts are those
    N&S on that span, its neurons the members inside the span."""
    S = W / "screen"
    cands = load_candidates(W / "candidates/candidates.json.gz")
    with gzip.open(S / "subspan_table.json.gz", "rt") as f:
        table = {int(k): v for k, v in json.load(f).items()}
    rs = np.load(W / "subspan/rows_sorted.npz")
    ns = rs["ns"].astype(bool)
    r_c, r_p, r_a, r_b = rs["cand"][ns].astype(np.int64), rs["prompt"][ns].astype(np.int64), rs["a"][ns].astype(np.int64), rs["b"][ns].astype(np.int64)
    layers = [sorted({g // config.D_MLP for g in c["members"]}) for c in cands]
    lo_of = np.array([layers[c][a] for c, a in zip(r_c, r_a)]); hi_of = np.array([layers[c][b] for c, b in zip(r_c, r_b)])
    gkey = r_c * 1024 + lo_of * 32 + hi_of
    o = np.argsort(gkey, kind="stable"); gkey, r_p = gkey[o], r_p[o]
    gu, st = np.unique(gkey, return_index=True); en = np.r_[st[1:], len(gkey)]
    ns_prompts = {int(k): r_p[s:e] for k, s, e in zip(gu, st, en)}
    n_kept = n_live_prompts(W)
    appr = _approaches(cands)

    def choose(ci, minp):
        best = None
        for r in table.get(ci, []):
            if r[3] >= minp and (best is None or (r[1] - r[0]) > (best[1] - best[0]) or ((r[1] - r[0]) == (best[1] - best[0]) and r[3] > best[3])):
                best = r
        return best

    out = {"n_prompts": n_kept, "tau_n": TAU_N, "tau_s": TAU_S, "rule": "widest sub-span with >= min_prompts N&S prompts; neurons = members inside it", "by_min_prompts": {}}
    curves = {}
    for minp in (1, 5, 20):
        rows = []
        for (tile, method, param), cis in sorted(appr.items()):
            picks = []
            for ci in cis:
                ch = choose(ci, minp)
                if ch is None:
                    continue
                pr = ns_prompts.get(ci * 1024 + ch[0] * 32 + ch[1])
                if pr is None or len(pr) == 0:
                    continue
                picks.append((ch[2], set(pr.tolist()), ch[1] - ch[0] + 1))
            picks.sort(key=lambda t: -len(t[1]) / max(t[0], 1))
            u: set = set(); ys = []; xs = np.cumsum([p[0] for p in picks]) if picks else np.zeros(0, np.int64)
            for _, pr, _ in picks:
                u |= pr; ys.append(len(u))
            name = f"{tile} {method}{param:g}"
            if minp == 5:
                curves[name + "|neurons"] = xs.astype(np.int64); curves[name + "|prompts"] = np.array(ys, np.int64)
            rows.append({"tile": tile, "method": method, "param": param, "communities": len(cis), "certified_communities": len(picks),
                         "prompts_covered": len(u), "prompts_frac": len(u) / n_kept, "neurons": int(xs[-1]) if len(picks) else 0,
                         "median_span_width": float(np.median([p[2] for p in picks])) if picks else float("nan"),
                         "neurons_per_prompt": float(xs[-1] / max(len(u), 1)) if len(picks) else float("nan")})
        out["by_min_prompts"][str(minp)] = rows
    (W / "coverage").mkdir(exist_ok=True)
    (W / "coverage/summary_perspan.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(W / "coverage/curves_perspan.npz", **curves)
    summary = out["by_min_prompts"]["5"]
    print(f"\none span per community, min prompts 5\n{'tile':8s} {'approach':8s} {'comms':>7s} {'cert':>6s} {'prompts':>8s} {'%':>5s} {'neurons':>8s} {'n/prompt':>8s} {'med width':>9s}")
    for s_ in summary:
        print(f"{s_['tile']:8s} {s_['method'] + format(s_['param'], 'g'):8s} {s_['communities']:7,} {s_['certified_communities']:6,} "
              f"{s_['prompts_covered']:8,} {100 * s_['prompts_frac']:5.1f} {s_['neurons']:8,} {s_['neurons_per_prompt']:8.1f} {s_['median_span_width']:9.1f}")
    print(f"wrote per-span coverage to {W / 'coverage'}")


def run(a: argparse.Namespace) -> None:
    W = Path(a.world)
    if a.mode == "necessity":
        aggregate_necessity(W, a.tau_nec)
    elif a.mode == "subspan":
        aggregate_subspan(W, a.tau_nec, a.tau_suff, a.tau_layer)
        coverage(W, a.tau_nec, a.tau_suff)
    else:
        coverage(W, a.tau_nec, a.tau_suff)
