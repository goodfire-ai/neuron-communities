"""``combine`` stage: concatenate per-corpus manifests into ONE manifest over one combined tokens dataset (a "world").

Every source is a manifest dir with manifest.npz + dedup_kept.npz + meta.json (per-source dedup already run).
Sources are given as ``--source DIR:TAG`` in the order they should be concatenated; the position of a tag in that
order is its ``source`` id.

  <tokens-out>/   combined tokens dataset: every source's chunk files symlinked under layers/tokens with
                  renumbered chunk ids, and the sequence tables concatenated into one index.sqlite (document rows
                  renumbered by a per-source row offset); metadata.json
  <out>/          manifest.npz (row_index, position, source, pattern, subcat), dedup_kept.npz (per-source kept indices
                  shifted by the source's prompt offset, sorted), meta.json (dataset, n, sources{tag: ...}, ...)

Run the ``dedup`` stage's short-context pass afterwards if needed; dedup is per corpus and is not repeated here.

  python -m neuron_communities combine --source fineweb5m:fineweb --source wiki2500k:wiki \\
      --out world_base --tokens-out world_tokens
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from neuron_communities.data import io


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--source", action="append", required=True, metavar="DIR:TAG",
                    help="manifest dir and its tag, repeatable; the order given is the source id order")
    ap.add_argument("--out", required=True, help="combined manifest directory to create")
    ap.add_argument("--tokens-out", required=True, help="combined tokens dataset directory to create")


def run(a: argparse.Namespace) -> None:
    OUT, TOK = Path(a.out), Path(a.tokens_out)
    SOURCES = [(Path(d), t) for d, t in (x.rsplit(":", 1) for x in a.source)]
    t0 = time.monotonic()
    for md, tag in SOURCES:
        for f in ("manifest.npz", "dedup_kept.npz", "meta.json"):
            assert (md / f).exists(), f"{tag}: {f} missing"
    OUT.mkdir(parents=True, exist_ok=True)
    (TOK / "layers" / "tokens").mkdir(parents=True, exist_ok=True)

    conn = io.create_index_db(TOK)
    row_offset, chunk_offset = 0, 0
    row_offsets, n_rows = {}, {}
    for md, tag in SOURCES:
        ds = io.dataset_of(md)
        site = io.tokens_site(ds)
        rows = io.read_locations(ds)
        chunk_ids = sorted({c for c, _, _ in rows})
        for c in chunk_ids:
            link = TOK / "layers" / "tokens" / f"chunk_{chunk_offset + c:06d}.safetensors"
            target = io.chunk_path(ds, c, site).resolve()
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(target, link)
        conn.executemany("INSERT INTO sequence_locations VALUES (?,?,?,?)",
                         [(f"seq_{row_offset + i}", chunk_offset + c, o, n) for i, (c, o, n) in enumerate(rows)])
        conn.commit()
        row_offsets[tag], n_rows[tag] = row_offset, len(rows)
        print(f"{tag}: {len(rows):,} docs, {len(chunk_ids)} chunks (row offset {row_offset:,}, chunk offset {chunk_offset})", flush=True)
        row_offset += len(rows); chunk_offset += max(chunk_ids) + 1
    conn.close()
    (TOK / "metadata.json").write_text(json.dumps({"mode": "token", "tokens_only": True, "combined_from": [str(m) for m, _ in SOURCES],
                                                  "has_token_ids": True, "layers": [], "d_models": {}}, indent=1))

    tags = [tag for _, tag in SOURCES]
    rows_all, pos_all, kept_all, src_tag, pat_all, sub_all = [], [], [], [], [], []
    prompt_offset = 0
    for md, tag in SOURCES:
        m = np.load(md / "manifest.npz")
        r, p = m["row_index"].astype(np.int64), m["position"].astype(np.int64)
        rows_all.append(r + row_offsets[tag]); pos_all.append(p)
        kept_all.append(np.load(md / "dedup_kept.npz")["kept"].astype(np.int64) + prompt_offset)
        src_tag.append(np.full(len(r), tags.index(tag), np.int8))
        pat_all.append(m["pattern"].astype(str) if "pattern" in m.files else np.full(len(r), "", dtype="<U1"))
        sub_all.append(m["subcat"].astype(str) if "subcat" in m.files else np.full(len(r), "", dtype="<U1"))
        prompt_offset += len(r)
    row_index, position = np.concatenate(rows_all), np.concatenate(pos_all)
    np.savez_compressed(OUT / "manifest.npz", row_index=row_index, position=position, source=np.concatenate(src_tag),
                        pattern=np.concatenate(pat_all).astype(str), subcat=np.concatenate(sub_all).astype(str))
    np.savez_compressed(OUT / "dedup_kept.npz", kept=np.sort(np.concatenate(kept_all)))
    (OUT / "meta.json").write_text(json.dumps({
        "dataset": str(TOK), "n": int(len(row_index)),
        "sources": {tag: {"manifest": str(m), "n_prompts": int(len(np.load(m / "manifest.npz")["row_index"])),
                          "n_kept": int(len(np.load(m / "dedup_kept.npz")["kept"])), "row_offset": row_offsets[tag], "n_docs": n_rows[tag],
                          "source_id": tags.index(tag)} for m, tag in SOURCES},
        "selection": "concatenation of the per-corpus manifests (order: " + " | ".join(tags) + ")",
        "note": "manifest.npz carries 'source' (index into meta.sources order), 'pattern' and 'subcat'; dedup is per corpus; "
                "position = last prefix token"}, indent=1))
    print(f"manifest: {len(row_index):,} prompts, dedup kept {sum(len(k) for k in kept_all):,}; per source "
          + ", ".join(f"{t}={len(r):,}" for t, r in zip(tags, rows_all)) + f" ({time.monotonic() - t0:.0f}s)", flush=True)
