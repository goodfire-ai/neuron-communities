"""``dedup`` stage: near-duplicate removal over a prompt manifest, in two passes.

Pass 1, window-64 leader dedup (``dedup_kept.npz``). Each prompt is represented by the set of 8-token shingle
hashes of its last 64 tokens. Within a document, prompts are scanned in position order and a prompt whose window
has Jaccard similarity >= 0.5 with the current leader joins the leader's cluster, otherwise it starts a new one.
Leaders of different documents are then unioned when they share a shingle (inverted index, shingles occurring in
more than 100 leaders are ignored) and their windows have Jaccard >= 0.5. One prompt per final cluster is kept
(``leader_dedup`` takes an optional priority array; by default the lowest manifest index wins).
Writes <manifest-dir>/dedup_kept.npz: ``kept`` = sorted manifest indices of the survivors.

Pass 2, exact dedup of short contexts (``dedup_drop.npz``). The shingle dedup cannot see contexts shorter than
8 tokens, so among kept prompts whose context has fewer than ``--min-context`` tokens one leader per exact token
context is kept and the rest are listed in <manifest-dir>/dedup_drop.npz (``drop`` = indices into ``kept``,
``min_context``). Kept-row numbering is unchanged, so every index computed over kept rows stays valid;
consumers filter the dropped rows out.

  python -m neuron_communities dedup --manifest-dir wiki250k
  python -m neuron_communities dedup --manifest-dir world_base --short-only   # pass 2 only (after ``combine``)
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from neuron_communities import config
from neuron_communities.data import io

SHINGLE = 8                     # tokens per shingle (contexts shorter than this are handled by pass 2)
SIM_THRESHOLD = 0.5             # Jaccard similarity of shingle sets above which two prompts are duplicates
WINDOW = config.CONTEXT_TOKENS  # tokens of context the shingles are taken from
DF_CAP = 100                    # shingles shared by more leaders than this do not generate candidate pairs
MIX = np.uint64(0x9E3779B97F4A7C15)


def gram_hashes(ids: np.ndarray) -> np.ndarray:
    ids_u = ids.astype(np.uint64)
    n_grams = len(ids_u) - SHINGLE + 1
    if n_grams <= 0:
        return np.empty(0, dtype=np.uint64)
    acc = np.zeros(n_grams, dtype=np.uint64)
    for k in range(SHINGLE):
        acc = (acc ^ (ids_u[k:k + n_grams] + np.uint64(k) + MIX)) * MIX
    return acc


def leader_dedup(meta: dict[str, np.ndarray], doc_tokens: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Window-64 leader dedup; returns (KEPT indices, cluster id per prompt).

    ``meta`` holds ``row_index`` and ``position`` (and optionally ``priority``: the prompt with the highest value is
    kept per cluster; with no priority, or ties, the lowest index wins). ``doc_tokens`` maps row -> token ids.
    """
    n = len(meta["position"])
    doc_hashes = {row: gram_hashes(tokens) for row, tokens in doc_tokens.items()}

    def window(row, pos):
        lo = max(0, pos - WINDOW + 1)
        hi = pos - SHINGLE + 2
        return np.unique(doc_hashes[row][lo:hi]) if hi > lo else np.empty(0, dtype=np.uint64)

    win = [window(int(meta["row_index"][i]), int(meta["position"][i])) for i in range(n)]

    def jaccard(i, j):
        a, b = win[i], win[j]
        if not len(a) or not len(b):
            return 0.0
        inter = len(np.intersect1d(a, b, assume_unique=True))
        return inter / (len(a) + len(b) - inter)

    by_doc = defaultdict(list)
    for i in range(n):
        by_doc[int(meta["row_index"][i])].append(i)
    cluster_of = np.full(n, -1, dtype=np.int64)
    reps: list[int] = []
    for _row, ids in by_doc.items():
        positioned = sorted((int(meta["position"][i]), i) for i in ids)
        rep = None
        for _pos, local in positioned:
            if rep is None or jaccard(rep, local) < SIM_THRESHOLD:
                rep = local
                reps.append(local)
            cluster_of[local] = len(reps) - 1

    parent = list(range(len(reps)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    inverted = defaultdict(list)
    for cid, rep in enumerate(reps):
        for h in win[rep]:
            inverted[int(h)].append(cid)
    candidates = set()
    for members in inverted.values():
        if 1 < len(members) <= DF_CAP:
            for a_i in range(len(members)):
                for b_i in range(a_i + 1, len(members)):
                    a, b = members[a_i], members[b_i]
                    if int(meta["row_index"][reps[a]]) != int(meta["row_index"][reps[b]]):
                        candidates.add((min(a, b), max(a, b)))
    del inverted
    for a, b in candidates:
        if find(a) != find(b) and jaccard(reps[a], reps[b]) >= SIM_THRESHOLD:
            parent[find(a)] = find(b)
    final_of = np.array([find(int(c)) for c in cluster_of])
    abl = meta["priority"] if "priority" in meta else np.zeros(n, dtype=np.float32)
    best: dict[int, int] = {}
    for local in range(n):
        c = int(final_of[local])
        if c not in best or abl[local] > abl[best[c]]:
            best[c] = local
    return np.array(sorted(best.values())), final_of


def dedup_windows(MD: Path, DATASET: Path) -> None:
    """Pass 1: window-64 leader dedup of <MD>/manifest.npz -> <MD>/dedup_kept.npz."""
    row, pos = io.load_manifest(MD)
    locations = io.read_locations(DATASET)
    site = io.tokens_site(DATASET)
    doc_tokens, handles = {}, {}
    for r in sorted(set(row.tolist())):
        c, o, n = locations[r]
        if c not in handles:
            handles[c] = io.open_chunk(DATASET, c, site, framework="pt")
        doc_tokens[r] = handles[c].get_tensor("token_ids")[o:o + n].numpy()
    print(f"{len(doc_tokens):,} docs loaded")
    meta = {"row_index": row, "position": pos}
    kept, _ = leader_dedup(meta, doc_tokens)
    np.savez_compressed(MD / "dedup_kept.npz", kept=kept)
    print(f"dedup: kept {len(kept):,} of {len(row):,}")


def dedup_short_contexts(MD: Path, ds: Path, min_context: int) -> None:
    """Pass 2: exact dedup of kept prompts with context < min_context tokens -> <MD>/dedup_drop.npz."""
    row, pos = io.load_manifest(MD)
    kept = io.load_kept(MD)
    loc = io.read_locations(ds)
    site = io.tokens_site(ds)
    H = {}
    groups = defaultdict(list)
    for k in np.where(pos[kept] < min_context)[0]:
        mi = int(kept[k]); c, o, n = loc[int(row[mi])]
        if c not in H:
            H[c] = io.open_chunk(ds, c, site, framework="pt")
        groups[tuple(H[c].get_tensor("token_ids")[o: o + int(pos[mi]) + 1].tolist())].append(int(k))
    drop = sorted(k for v in groups.values() for k in v[1:])
    np.savez_compressed(MD / "dedup_drop.npz", drop=np.array(drop, np.int64), min_context=np.array([min_context]))
    print(f"{sum(len(v) for v in groups.values()):,} kept prompts with context < {min_context} tokens in {len(groups):,} exact contexts; "
          f"dropping {len(drop):,} redundant copies -> {MD / 'dedup_drop.npz'}")


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--manifest-dir", required=True, help="manifest directory (from the sample stage)")
    ap.add_argument("--dataset", default=None, help="tokens dataset; default: the one recorded in <manifest-dir>/meta.json")
    ap.add_argument("--min-context", type=int, default=config.MIN_CONTEXT_FOR_SHINGLE_DEDUP,
                    help="contexts shorter than this many tokens are deduplicated by exact match (pass 2)")
    ap.add_argument("--short-only", action="store_true",
                    help="run only pass 2 on an existing dedup_kept.npz (e.g. a combined world whose per-corpus pass 1 already ran)")


def run(a: argparse.Namespace) -> None:
    MD = Path(a.manifest_dir)
    DATASET = io.dataset_of(MD, a.dataset)
    if a.short_only:
        assert (MD / "dedup_kept.npz").exists(), f"--short-only needs an existing {MD / 'dedup_kept.npz'}"
    else:
        dedup_windows(MD, DATASET)
    dedup_short_contexts(MD, DATASET, a.min_context)
