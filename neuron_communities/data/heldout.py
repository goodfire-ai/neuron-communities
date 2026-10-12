"""``heldout-dedup`` stage: remove from a NEW manifest every dedup-kept prompt that collides with the OLD (training)
manifest of the same corpus, so the held-out set shares no prompt with the set the dictionary was built on.

Two collision rules, both exact:
  1. same (row, position) — the same prompt sampled twice (only meaningful when both manifests share a token dataset);
  2. same 64-token context (tokens[pos-63 .. pos], zero-padded at document start) — identical prefix text in another
     place (boilerplate, duplicated documents), hashed with blake2b.
Documents are NOT held out (both samples draw from the same corpora), which is stated in the output stats.

Rewrites <new>/dedup_kept.npz (original saved as dedup_kept_before_cross.npz) and writes <new>/cross_dedup.json. When
<new>/dedup_drop.npz exists (indices into kept rows), it is remapped to the new kept numbering (original saved as
dedup_drop_before_cross.npz) so the short-context exclusions stay valid.

  python -m neuron_communities heldout-dedup --new fineweb5m_heldout --old fineweb5m
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open

from neuron_communities import config
from neuron_communities.data import io

WIN = config.CONTEXT_TOKENS


def load(md: Path, dataset: str | None = None):
    row, pos = io.load_manifest(md)
    return row, pos, io.load_kept(md), io.dataset_of(md, dataset)


def context_hashes(ds: Path, rows, poss) -> np.ndarray:
    """blake2b-64 of the last WIN token ids ending at pos (inclusive), per (row, pos); reads chunk tensors once each."""
    loc = np.array(io.read_locations(ds), dtype=np.int64).reshape(-1, 3)
    site = io.tokens_site(ds)
    out = np.zeros(len(rows), dtype=np.uint64)
    chunk_of = loc[rows, 0]
    order = np.argsort(chunk_of, kind="stable")
    t0 = time.monotonic(); done = 0
    for c in np.unique(chunk_of):
        idx = order[np.searchsorted(chunk_of[order], c, "left"):np.searchsorted(chunk_of[order], c, "right")]
        with safe_open(str(io.chunk_path(ds, int(c), site)), framework="np") as f:
            tok = f.get_tensor("token_ids")
        for i in idx:
            o = int(loc[rows[i], 1]); p = int(poss[i])
            w = tok[o + max(0, p - WIN + 1): o + p + 1]
            if len(w) < WIN:
                w = np.concatenate([np.zeros(WIN - len(w), w.dtype), w])
            out[i] = np.frombuffer(hashlib.blake2b(np.ascontiguousarray(w.astype(np.int32)).tobytes(), digest_size=8).digest(), dtype=np.uint64)[0]
        done += len(idx)
        if done % 500_000 < len(idx):
            print(f"  {done:,}/{len(rows):,} hashed ({(time.monotonic() - t0) / 60:.1f} min)", flush=True)
    return out


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--new", required=True, help="held-out manifest dir (its dedup_kept.npz is rewritten)")
    ap.add_argument("--old", required=True, help="training manifest dir of the same corpus")
    ap.add_argument("--new-dataset", default=None, help="tokens dataset of --new; default: the one in its meta.json")
    ap.add_argument("--old-dataset", default=None, help="tokens dataset of --old; default: the one in its meta.json")


def run(a: argparse.Namespace) -> None:
    NEW, OLD = Path(a.new), Path(a.old)
    nr, npos, nk, nds = load(NEW, a.new_dataset); orow, opos, ok, ods = load(OLD, a.old_dataset)
    same_dataset = nds.resolve() == ods.resolve()
    # rule 1: same (row, pos) among kept prompts of both sets
    if same_dataset:
        okey = set((orow[ok] * (1 << 32) + opos[ok]).tolist())
        nkey = nr[nk] * (1 << 32) + npos[nk]
        same_prompt = np.array([k in okey for k in nkey.tolist()])
    else:
        same_prompt = np.zeros(len(nk), bool)
    # rule 2: same 64-token context
    print(f"hashing {len(ok):,} old + {len(nk):,} new contexts", flush=True)
    oh = context_hashes(ods, orow[ok], opos[ok]); nh = context_hashes(nds, nr[nk], npos[nk])
    oset = set(oh.tolist())
    same_ctx = np.array([h in oset for h in nh.tolist()])
    drop = same_prompt | same_ctx
    kept_after = nk[~drop]
    (NEW / "dedup_kept_before_cross.npz").write_bytes((NEW / "dedup_kept.npz").read_bytes())
    np.savez(NEW / "dedup_kept.npz", kept=kept_after)
    n_drop_before = n_drop_after = None
    if (NEW / "dedup_drop.npz").exists():                       # kept-row indices: remap to the surviving rows
        with np.load(NEW / "dedup_drop.npz") as z:
            old_drop, min_context = z["drop"].astype(np.int64), z["min_context"]
        (NEW / "dedup_drop_before_cross.npz").write_bytes((NEW / "dedup_drop.npz").read_bytes())
        new_pos = np.cumsum(~drop) - 1
        surviving = old_drop[~drop[old_drop]]
        np.savez_compressed(NEW / "dedup_drop.npz", drop=new_pos[surviving].astype(np.int64), min_context=min_context)
        n_drop_before, n_drop_after = int(len(old_drop)), int(len(surviving))
    # document overlap for the record
    odocs = set(orow[ok].tolist()); shared_docs = int(np.isin(nr[kept_after], list(odocs)).sum()) if same_dataset else None
    stats = {"new": str(NEW), "old": str(OLD), "new_kept_before": int(len(nk)), "same_prompt": int(same_prompt.sum()),
             "same_context_64": int(same_ctx.sum()), "dropped": int(drop.sum()), "new_kept_after": int(len(kept_after)),
             "short_context_drop_before": n_drop_before, "short_context_drop_after": n_drop_after,
             "new_kept_in_docs_also_used_by_old": shared_docs,
             "note": "prompt- and context-disjoint from the training sample; documents are shared (same corpora)" if same_dataset
                     else "context-disjoint from the training sample (different token datasets; document overlap not tracked)"}
    (NEW / "cross_dedup.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
