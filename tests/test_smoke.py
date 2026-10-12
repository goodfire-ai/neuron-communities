"""End-to-end smoke test of every pipeline stage on a randomly initialised two-layer Llama (CPU, no network).

The stages run in-process through ``neuron_communities.cli.main`` after ``config.configure(2, 64)`` (the same override
``NEURON_COMMUNITIES_GEOMETRY=2x64`` applies on the command line), in pipeline order, on a synthetic tokens dataset
written here with the package's own index writer. After every stage the test asserts that the documented output files
exist and have the documented keys, shapes and dtypes; it does not check the numbers, which only mean something on the
real model. The certification thresholds are relaxed (tau_nec = tau_layer = -1, tau_suff = 1e9) so that every evaluation
stage has rows to process on a random model. Run with ``pytest -s`` to see the per-stage counts.

The ``tokens`` stage is covered separately on a local JSONL directory with a tiny word-level tokenizer.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import sqlite3
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")          # before ``datasets`` / ``transformers`` are imported
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from neuron_communities import config
from neuron_communities.cli import main
from neuron_communities.data import io

N_LAYERS, D_MLP, VOCAB = 2, 64, 512
N_DOCS, N_TRAIN, N_HELDOUT = 300, 300, 200
BLOCK = 64
K = config.TOP_K
GAMMAS = "0.1,0.5"
RELAXED = ["--tau-nec", "-1", "--tau-suff", "1e9", "--tau-layer", "-1"]   # every evaluated row passes -> every stage has rows
                                                                            # (-1, not 0: the KL of a clean replay can be -1e-7)

T0 = time.monotonic()


def stage(*argv) -> None:
    argv = [str(a) for a in argv]
    print(f"\n### [{time.monotonic() - T0:5.0f}s] python -m neuron_communities {' '.join(argv)}", flush=True)
    main(argv)


def npz_keys(path: Path, keys: set[str]) -> dict[str, np.ndarray]:
    assert path.exists(), f"missing {path}"
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) >= keys, f"{path}: keys {sorted(z.files)} lack {sorted(keys - set(z.files))}"
        return {k: z[k] for k in z.files}


def write_tokens_dataset(out: Path, rng: np.random.Generator, n_docs: int, chunk_tokens: int = 3000) -> list[np.ndarray]:
    """A tokens-only dataset in the ``tokens`` stage layout: BOS-prefixed random documents over a small vocabulary with
    repeated phrases (so dedup, co-occurrence and Leiden have structure) and some exact duplicate documents."""
    (out / "layers" / "tokens").mkdir(parents=True)
    phrases = [rng.integers(2, VOCAB, size=int(rng.integers(4, 9))) for _ in range(24)]
    docs: list[np.ndarray] = []
    for i in range(n_docs):
        if i >= 20 and rng.random() < 0.1:                  # exact duplicate of an earlier document
            docs.append(docs[int(rng.integers(0, i))].copy()); continue
        parts = [np.array([1])]                              # BOS
        n = int(rng.integers(12, 48))
        while sum(len(p) for p in parts) < n:
            parts.append(phrases[int(rng.integers(len(phrases)))] if rng.random() < 0.5 else rng.integers(2, VOCAB, size=3))
        docs.append(np.concatenate(parts)[:n].astype(np.int32))
    conn = io.create_index_db(out)
    rows, buf, off, cid, chunks = [], [], 0, 0, []

    def flush():
        nonlocal buf, off, cid
        ids = torch.from_numpy(np.concatenate(buf).astype(np.int32))
        save_file({"token_ids": ids}, str(out / "layers" / "tokens" / f"chunk_{cid:06d}.safetensors"))
        chunks.append({"chunk_id": cid, "num_items": int(len(ids))}); cid += 1; buf, off = [], 0

    for i, d in enumerate(docs):
        if off + len(d) > chunk_tokens and buf:
            flush()
        rows.append((f"seq_{i}", cid, off, int(len(d)))); buf.append(d); off += len(d)
    flush()
    conn.executemany("INSERT INTO sequence_locations VALUES (?,?,?,?)", rows); conn.commit(); conn.close()
    (out / "metadata.json").write_text(json.dumps({"mode": "token", "tokens_only": True, "num_chunks": cid,
                                                   "total_items": int(sum(len(d) for d in docs)), "chunks": chunks, "has_token_ids": True}))
    return docs


def save_tiny_llama(out: Path) -> None:
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=D_MLP, num_hidden_layers=N_LAYERS, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=256, bos_token_id=1, eos_token_id=2)
    LlamaForCausalLM(cfg).save_pretrained(out)


@pytest.fixture(scope="module")
def st(tmp_path_factory) -> dict:
    """Shared state of the ordered stage tests: paths plus whatever earlier stages learned."""
    config.configure(N_LAYERS, D_MLP)                     # == NEURON_COMMUNITIES_GEOMETRY=2x64 on the command line
    assert (config.N_LAYERS, config.D_MLP, config.LAYER_TILE, config.HUBS) == (2, 64, (0, 1), {})
    R = tmp_path_factory.mktemp("nc_smoke")
    docs = write_tokens_dataset(R / "tok", np.random.default_rng(0), N_DOCS)
    save_tiny_llama(R / "tinyllama")
    s = {"R": R, "docs": docs, "tok": R / "tok", "model": R / "tinyllama", "man": R / "man", "held": R / "held",
         "harv": R / "harv", "W": R / "world", "W2": R / "world_spanlocal"}
    print(f"\nsynthetic dataset: {len(docs)} docs, {sum(len(d) for d in docs)} tokens, "
          f"{len({d.tobytes() for d in docs})} distinct documents; model {s['model']}")
    return s


def test_01_sample(st):
    for name, n, seed in (("man", N_TRAIN, 0), ("held", N_HELDOUT, 1)):
        stage("sample", "--dataset", st["tok"], "--out", st[name], "--n", n, "--seed", seed)
        m = npz_keys(st[name] / "manifest.npz", {"row_index", "position"})
        assert m["row_index"].dtype == np.int64 and m["position"].dtype == np.int64 and m["row_index"].shape == (n,) == m["position"].shape
        counts = io.read_counts(st["tok"])
        assert (m["row_index"] >= 0).all() and (m["row_index"] < len(counts)).all()
        assert (m["position"] >= 1).all() and (m["position"] < counts[m["row_index"]]).all()
        meta = json.loads((st[name] / "meta.json").read_text())
        assert meta["n"] == n and Path(meta["dataset"]) == st["tok"] and meta["seed"] == seed


def test_02_dedup(st):
    for name in ("man", "held"):
        stage("dedup", "--manifest-dir", st[name])
        n = len(np.load(st[name] / "manifest.npz")["row_index"])
        kept = npz_keys(st[name] / "dedup_kept.npz", {"kept"})["kept"]
        assert kept.ndim == 1 and (np.diff(kept) > 0).all() and 0 <= kept[0] and kept[-1] < n and len(kept) <= n
        d = npz_keys(st[name] / "dedup_drop.npz", {"drop", "min_context"})
        assert d["min_context"].tolist() == [config.MIN_CONTEXT_FOR_SHINGLE_DEDUP]
        assert d["drop"].ndim == 1 and d["drop"].dtype == np.int64 and (np.diff(d["drop"]) > 0).all() and (d["drop"] < len(kept)).all()
        _, pos = io.load_manifest(st[name])
        assert (pos[kept[d["drop"]]] < config.MIN_CONTEXT_FOR_SHINGLE_DEDUP).all()
        print(f"{name}: {n} prompts -> {len(kept)} kept by window dedup, {len(d['drop'])} short-context exact duplicates")


def test_03_heldout_dedup(st):
    before = io.load_kept(st["held"]); drop_before = io.load_drop(st["held"])
    stage("heldout-dedup", "--new", st["held"], "--old", st["man"])
    after = io.load_kept(st["held"]); drop_after = io.load_drop(st["held"])
    assert np.array_equal(np.load(st["held"] / "dedup_kept_before_cross.npz")["kept"], before)
    assert np.isin(after, before).all() and (np.diff(after) > 0).all()
    x = json.loads((st["held"] / "cross_dedup.json").read_text())
    for k in ("new_kept_before", "same_prompt", "same_context_64", "dropped", "new_kept_after", "new_kept_in_docs_also_used_by_old",
              "short_context_drop_before", "short_context_drop_after", "note"):
        assert k in x, k
    assert x["new_kept_before"] == len(before) and x["new_kept_after"] == len(after) == len(before) - x["dropped"]
    # dedup_drop.npz indexes kept rows: it must be remapped to the new numbering and point at the same manifest entries
    assert (drop_after < len(after)).all() and x["short_context_drop_before"] == len(drop_before) and x["short_context_drop_after"] == len(drop_after)
    assert set(after[drop_after].tolist()) == set(before[drop_before].tolist()) & set(after.tolist())
    # same tokens dataset: a prompt of the old sample never survives in the new one
    old_keys = set((io.load_manifest(st["man"])[0][io.load_kept(st["man"])] * (1 << 32) + io.load_manifest(st["man"])[1][io.load_kept(st["man"])]).tolist())
    r, p = io.load_manifest(st["held"])
    assert not (set((r[after] * (1 << 32) + p[after]).tolist()) & old_keys)
    print(f"held-out: {len(before)} -> {len(after)} kept ({x['same_prompt']} same prompt, {x['same_context_64']} same 64-token context)")


def test_04_attribution(st):
    for shard in range(2):
        stage("attribution", "--manifest-dir", st["man"], "--out-dir", st["harv"], "--shard", shard, "--shards", 2,
              "--block-size", BLOCK, "--model", st["model"], "--device", "cpu")
    blocks = sorted((st["harv"] / "blocks").glob("block_*.npz"))
    assert [b.name for b in blocks] == [f"block_{i:06d}.npz" for i in range(math.ceil(N_TRAIN / BLOCK))]
    man_row, man_pos = io.load_manifest(st["man"])
    for i, b in enumerate(blocks):
        z = npz_keys(b, {"row_index", "position", "manifest_lo", "k", "variants", "crosscheck", "crosscheck_rel_err", "teacher_valid",
                         "logp_ga", "idx_act_ga", "val_act_ga"})
        n = len(z["row_index"]); lo = int(z["manifest_lo"][0])
        assert lo == i * BLOCK and n == min(BLOCK, N_TRAIN - lo)
        assert np.array_equal(z["row_index"], man_row[lo:lo + n]) and np.array_equal(z["position"], man_pos[lo:lo + n])
        assert z["k"].tolist() == [K] and z["variants"].tolist() == ["act_ga"]
        assert z["idx_act_ga"].shape == (n, N_LAYERS, K) and z["idx_act_ga"].dtype == np.uint16 and (z["idx_act_ga"] < D_MLP).all()
        assert z["val_act_ga"].shape == (n, N_LAYERS, K) and z["val_act_ga"].dtype == np.float32 and np.isfinite(z["val_act_ga"]).all()
        assert (np.diff(z["val_act_ga"], axis=-1) <= 0).all(), "top-k is stored in score order"
        assert all(len(set(r)) == K for r in z["idx_act_ga"].reshape(-1, K)), "top-k sets have no repeats"
        assert z["teacher_valid"].dtype == bool and z["teacher_valid"].shape == (n,) and z["logp_ga"].dtype == np.float32
        assert np.isfinite(z["logp_ga"]).all() and (z["logp_ga"] <= 0).all()
        assert len(z["crosscheck"]) == len(z["crosscheck_rel_err"]) >= 1 and (0 <= z["crosscheck"]).all() and (z["crosscheck"] <= 1).all()
    stage("attribution", "--manifest-dir", st["man"], "--out-dir", st["harv"], "--shard", 0, "--shards", 2, "--block-size", BLOCK,
          "--model", st["model"], "--device", "cpu")                       # skip-if-exists: a no-op
    print(f"attribution: {len(blocks)} blocks; crosscheck overlap {np.mean([np.load(b)['crosscheck'].mean() for b in blocks]):.2f}")


def test_05_world(st):
    W = st["W"]
    stage("world", "--harvest-dir", st["harv"], "--manifest-dir", st["man"], "--out-dir", W, "--copy")
    assert sorted(p.name for p in (W / "merged").iterdir()) == [f"L{L:02d}.npz" for L in range(N_LAYERS)]
    for L in range(N_LAYERS):
        z = npz_keys(W / "merged" / f"L{L:02d}.npz", {"indptr", "indices", "values", "layer"})
        assert z["layer"].tolist() == [L]
        assert z["indptr"].dtype == np.int64 and np.array_equal(z["indptr"], np.arange(N_TRAIN + 1) * K)
        assert z["indices"].dtype == np.uint16 and z["indices"].shape == (N_TRAIN * K,) and (z["indices"] < D_MLP).all()
        assert z["values"].dtype == np.float32 and z["values"].shape == (N_TRAIN * K,)
        blk = np.load(st["harv"] / "blocks" / "block_000000.npz")
        assert np.array_equal(z["indices"][: BLOCK * K], blk["idx_act_ga"][:, L].reshape(-1)), "manifest order is preserved"
    for name in ("manifest.npz", "dedup_kept.npz", "dedup_drop.npz"):
        assert (W / name).exists() and not (W / name).is_symlink()
    meta = json.loads((W / "meta.json").read_text())
    assert meta["attr_source"]["k"] == K and meta["attr_source"]["variant"] == "act_ga" and Path(meta["dataset"]) == st["tok"]
    hs = json.loads((st["harv"] / "harvest_summary.json").read_text())
    assert hs["n"] == N_TRAIN and hs["k"] == K and hs["layers"] == N_LAYERS and hs["blocks"] == math.ceil(N_TRAIN / BLOCK)
    assert io.n_live_prompts(W) == len(io.load_kept(W)) - len(io.load_drop(W))
    print(f"world: {N_TRAIN} prompts, {io.n_live_prompts(W)} live after dedup; harvest summary {hs}")


def test_06_census(st, capsys):
    W = st["W"]
    stage("census", "--world", W, "--n-tiles")
    assert capsys.readouterr().out.strip().splitlines()[-1] == "3"     # (0,0), (0,1), (1,1) over the 0-1 window
    tiles = [(0, 0), (0, 1), (1, 1)]
    n_kept = len(io.load_kept(W))
    for t, (a, b) in enumerate(tiles):
        stage("census", "--world", W, "--tile", t, "--workers", 2, "--replicates", 6, "--c-min", 2, "--z-store", 2)
        z = npz_keys(W / "census" / f"pairs_L{a:02d}_L{b:02d}.npz", {"i", "j", "z", "c", "mean", "freq_i", "freq_j", "n_rows", "layers", "replicates"})
        m = len(z["i"])
        assert z["i"].dtype == z["j"].dtype == np.int32 and z["z"].dtype == z["mean"].dtype == np.float32 and z["c"].dtype == np.int32
        assert all(z[k].shape == (m,) for k in ("i", "j", "z", "c", "mean", "freq_i", "freq_j"))
        assert z["n_rows"].tolist() == [n_kept] and z["layers"].tolist() == [a, b] and z["replicates"].tolist() == [6]
        assert (z["c"] >= 2).all() and (z["z"] >= 2).all() and (z["freq_i"] >= z["c"]).all() and (z["freq_j"] >= z["c"]).all()
        if a == b:
            assert (z["i"] < z["j"]).all() and (z["j"] < D_MLP).all(), "diagonal tile: strict upper triangle"
        else:
            assert (z["i"] < D_MLP).all() and (z["j"] >= D_MLP).all() and (z["j"] < 2 * D_MLP).all(), "cross tile: j carries the D_MLP offset"
        print(f"census L{a}-L{b}: {m} pairs stored")
    stage("census", "--world", W, "--tile", 0, "--workers", 2, "--replicates", 6)   # existing tile: skipped
    assert "present" in capsys.readouterr().out


CAND_KEYS = {"members", "span", "n_found", "stops", "widths", "found", "src"}


def check_candidates(path: Path, extra: set[str] = frozenset()) -> list[dict]:
    with gzip.open(path, "rt") as f:
        cands = json.load(f)
    assert isinstance(cands, list)
    seen = set()
    for c in cands:
        assert set(c) >= CAND_KEYS | extra, sorted(c)
        m = c["members"]
        assert 2 <= len(m) <= config.SIZE_CAP and m == sorted(m) and len(set(m)) == len(m) and 0 <= m[0] and m[-1] < config.N_GID
        assert tuple(m) not in seen, "member sets are unique"; seen.add(tuple(m))
        assert c["span"] == [m[0] // D_MLP, m[-1] // D_MLP] and c["src"] == "cpm" and c["n_found"] >= 1
        assert c["found"] and all(t.split("@")[0] == "0-1" and t.split("@")[1].startswith("cpm") for t in c["found"])
        assert len(c["stops"]) == 2 and c["stops"][0] <= c["stops"][1] and len(c["widths"]) == 2
    return cands


def test_07_leiden(st, capsys):
    W = st["W"]; C = W / "census"; out = W / "cand_cpm"
    stage("leiden", "--census-dir", C, "--out-dir", out, "--gammas", GAMMAS, "--cpm-z", 2, "--n-runs")
    assert capsys.readouterr().out.strip().splitlines()[-1] == "2"
    for r in range(2):
        stage("leiden", "--census-dir", C, "--out-dir", out, "--gammas", GAMMAS, "--cpm-z", 2, "--run", r)
        with gzip.open(out / "runs" / f"run_{r:03d}.json.gz", "rt") as f:
            run = json.load(f)
        assert set(run) >= {"lo", "hi", "param", "n_runs", "diag", "comms"} and (run["lo"], run["hi"], run["n_runs"]) == (0, 1, 1)
        assert run["param"] == float(GAMMAS.split(",")[r]) and len(run["diag"]) == 1
        assert all(len(rec) == 4 and rec[0] == sorted(rec[0]) and rec[3] == [f"cpm{run['param']:g}"] for rec in run["comms"])
    stage("leiden", "--census-dir", C, "--out-dir", out, "--gammas", GAMMAS, "--cpm-z", 2, "--merge-runs")
    cands = check_candidates(out / "candidates.json.gz")
    stats = json.loads((out / "candidates_stats.json").read_text())
    assert set(stats) >= {"windows", "unique", "pooled", "dropped_gt_cap", "size_cap", "method", "params", "runs_diag"}
    assert stats["pooled"] == len(cands) and stats["params"] == [0.1, 0.5]
    # in-process discovery (no --run) gives the same dictionary as the sharded runs
    stage("leiden", "--census-dir", C, "--out-dir", W / "cand_inproc", "--gammas", GAMMAS, "--cpm-z", 2)
    assert check_candidates(W / "cand_inproc" / "candidates.json.gz") == cands
    print(f"leiden: {len(cands)} communities over gammas {GAMMAS} (sizes {sorted(len(c['members']) for c in cands)})")


def test_08_merge(st):
    W = st["W"]
    stage("merge", "--inputs", W / "cand_cpm" / "candidates.json.gz", "--out", W / "candidates_merged")
    cands = check_candidates(W / "candidates_merged" / "candidates.json.gz", {"n_layers"})
    src = check_candidates(W / "cand_cpm" / "candidates.json.gz")
    assert {tuple(c["members"]) for c in cands} == {tuple(c["members"]) for c in src}
    assert all(c["n_layers"] == len({g // D_MLP for g in c["members"]}) for c in cands)
    if cands:
        stats = json.loads((W / "candidates_merged" / "candidates_stats.json").read_text())
        assert stats["pooled"] == len(cands) and set(stats) >= {"src", "per_approach", "size_p50_p90_max", "layers_p50_p90_max"}
    print(f"merge: {len(cands)} communities")


def test_09_hubs(st):
    W = st["W"]
    stage("hubs", "--out", W / "candidates", "--runs", W / "cand_cpm" / "runs")        # no hubs configured for 2x64: a pure rebuild
    cands = check_candidates(W / "candidates" / "candidates.json.gz", {"hubs_removed"})
    assert all(c["hubs_removed"] == 0 for c in cands)
    assert {tuple(c["members"]) for c in cands} == {tuple(c["members"]) for c in check_candidates(W / "cand_cpm" / "candidates.json.gz")}
    stats = json.loads((W / "candidates" / "candidates_stats.json").read_text())
    assert stats["unique_candidates"] == len(cands) and stats["hubs"] == [] and len(stats["runs"]) == 2
    print(f"hubs: {len(cands)} communities after hub removal (hubs configured: {config.HUBS})")
    if len(cands) < 3:
        # A random model need not produce a co-occurrence graph with communities. The discovery outputs above are then
        # empty but well formed; to still exercise the evaluation stages, plant a few communities from the most
        # frequent harvested neurons, in the dictionary's record format.
        f = [np.bincount(np.load(W / "merged" / f"L{L:02d}.npz")["indices"], minlength=D_MLP) for L in range(N_LAYERS)]
        top = [np.argsort(-f[L])[:4].tolist() for L in range(N_LAYERS)]
        sets = [[top[0][0], D_MLP + top[1][0]], [top[0][1], D_MLP + top[1][1]], [top[0][2], top[0][3]], [D_MLP + top[1][2], D_MLP + top[1][3]],
                [top[0][0], top[0][1], D_MLP + top[1][0]]]
        cands = [{"members": sorted(m), "span": [min(m) // D_MLP, max(m) // D_MLP], "n_found": 1, "stops": [0.5, 0.5], "widths": [2, 2],
                  "found": ["0-1@cpm0.5"], "src": "cpm", "hubs_removed": 0} for m in sets]
        with gzip.open(W / "candidates" / "candidates.json.gz", "wt") as fh:
            json.dump(cands, fh)
        print(f"PLANTED {len(cands)} communities from the most frequent neurons so the evaluation stages have work")
    st["cands"] = check_candidates(W / "candidates" / "candidates.json.gz", {"hubs_removed"})


def test_10_assign(st):
    W = st["W"]; cands = st["cands"]
    stage("assign", "--world", W, "--cap", 50)
    z = npz_keys(W / "assignments" / "pairs.npz", {"prompt", "cand", "weight"})
    P, C, Wt = z["prompt"], z["cand"], z["weight"]
    assert P.dtype == C.dtype == np.int32 and Wt.dtype == np.float32 and P.shape == C.shape == Wt.shape
    assert len(P) > 0, "no (prompt, community) pair has >= 2 members present"
    assert (P >= 0).all() and (P < len(io.load_kept(W))).all() and (C >= 0).all() and (C < len(cands)).all()
    assert not np.isin(P, io.load_drop(W)).any(), "short-context duplicates are excluded"
    key = P.astype(np.int64) * len(cands) + C
    assert (np.diff(key) > 0).all(), "sorted by (prompt, cand), no duplicate pairs"
    assert (Wt >= 1).all()
    counts = npz_keys(W / "assignments" / "match_counts.npz", {"counts"})["counts"]
    assert counts.shape == (len(cands),) and counts.dtype == np.int64
    assert (np.bincount(C, minlength=len(cands)) <= counts).all(), "pairs per community never exceed the raw match count"
    # every pair really has >= 2 members in the prompt's top-k sets
    kept = io.load_kept(W); gids = np.zeros((N_TRAIN, N_LAYERS * K), np.int64)
    for L in range(N_LAYERS):
        gids[:, L * K:(L + 1) * K] = L * D_MLP + np.load(W / "merged" / f"L{L:02d}.npz")["indices"].reshape(N_TRAIN, K)
    for p, c in zip(P[:200], C[:200]):
        assert len(set(gids[kept[p]].tolist()) & set(cands[c]["members"])) >= config.MIN_PRESENT
    meta = json.loads((W / "assignments" / "meta.json").read_text())
    assert meta["pairs"] == len(P) and meta["candidates"] == len(cands) and meta["cap"] == 50
    print(f"assign: {len(P)} pairs over {len(np.unique(P))} prompts and {len(np.unique(C))} communities; capped {meta['capped_candidates']}")


def test_11_necessity(st):
    W = st["W"]
    for shard in range(2):
        stage("necessity", "--world", W, "--shard", shard, "--shards", 2, "--model", st["model"], "--device", "cpu")
    pairs = npz_keys(W / "assignments" / "pairs.npz", {"prompt", "cand"})
    got = []
    for shard in range(2):
        z = npz_keys(W / "necessity" / f"nec_shard{shard:03d}.npz", {"prompt", "cand", "kl_drop"})
        assert z["prompt"].dtype == z["cand"].dtype == np.int32 and z["kl_drop"].dtype == np.float32
        assert z["prompt"].shape == z["cand"].shape == z["kl_drop"].shape and np.isfinite(z["kl_drop"]).all() and (z["kl_drop"] > -1e-3).all()
        got.append(set(zip(z["prompt"].tolist(), z["cand"].tolist())))
    assert not (got[0] & got[1]), "shards split the prompt set"
    assert got[0] | got[1] == set(zip(pairs["prompt"].tolist(), pairs["cand"].tolist())), "shards cover every assigned pair once"
    stage("necessity", "--world", W, "--shard", 0, "--shards", 2, "--model", st["model"], "--device", "cpu")   # done already: no-op
    kl = np.concatenate([np.load(W / "necessity" / f"nec_shard{s:03d}.npz")["kl_drop"] for s in range(2)])
    print(f"necessity: {len(kl)} pairs, KL median {np.median(kl):.3f}, >= {config.TAU_NEC}: {(kl >= config.TAU_NEC).sum()}")


def test_12_aggregate_necessity(st):
    W = st["W"]
    stage("aggregate", "necessity", "--world", W, *RELAXED)
    P = np.load(W / "assignments" / "pairs.npz")["prompt"]
    kd = np.load(W / "screen" / "kl_drop.npy")
    assert kd.dtype == np.float32 and kd.shape == P.shape and np.isfinite(kd).all(), "every pair was evaluated by some shard"
    z = npz_keys(W / "subspan" / "pairs.npz", {"prompt", "cand"})
    assert z["prompt"].dtype == z["cand"].dtype == np.int32 and len(z["prompt"]) == len(P), "tau_nec = -1: every pair is necessary"
    print(f"aggregate necessity: {len(z['prompt'])} necessary pairs at tau_nec = -1; at {config.TAU_NEC}: {(kd >= config.TAU_NEC).sum()}")


def test_13_subspan(st):
    W = st["W"]; cands = st["cands"]
    for shard in range(2):
        stage("subspan", "--world", W, "--shard", shard, "--shards", 2, "--model", st["model"], "--device", "cpu")
    pairs = npz_keys(W / "subspan" / "pairs.npz", {"prompt", "cand"})
    n_layers_of = [len({g // D_MLP for g in c["members"]}) for c in cands]
    expected_rows = sum(n_layers_of[c] * (n_layers_of[c] + 1) // 2 for c in pairs["cand"])
    rows = 0; seen = set()
    for shard in range(2):
        z = npz_keys(W / "subspan" / f"subspan_shard{shard:03d}.npz", {"prompt", "cand", "a", "b", "kl_only", "kl_off"})
        assert z["prompt"].dtype == z["cand"].dtype == np.int32 and z["a"].dtype == z["b"].dtype == np.int8
        assert z["kl_only"].dtype == z["kl_off"].dtype == np.float16
        n = len(z["prompt"]); assert all(z[k].shape == (n,) for k in ("cand", "a", "b", "kl_only", "kl_off"))
        assert (z["a"] >= 0).all() and (z["a"] <= z["b"]).all() and all(int(b) < n_layers_of[c] for b, c in zip(z["b"], z["cand"]))
        assert np.isfinite(z["kl_only"].astype(np.float32)).all() and np.isfinite(z["kl_off"].astype(np.float32)).all()
        seen |= set(zip(z["prompt"].tolist(), z["cand"].tolist())); rows += n
    assert rows == expected_rows and seen == set(zip(pairs["prompt"].tolist(), pairs["cand"].tolist()))
    print(f"subspan: {rows} (prompt, community, sub-span) rows over {len(seen)} pairs")


def test_14_aggregate_subspan(st):
    W = st["W"]; cands = st["cands"]
    stage("aggregate", "subspan", "--world", W, *RELAXED)
    P = np.load(W / "assignments" / "pairs.npz")["prompt"]; n = len(P)
    ns = npz_keys(W / "screen" / "ns_results.npz", {"kl_drop", "kl_range_only", "kl_layer_only", "kl_sub_drop", "kl_sub_range"})
    assert all(v.shape == (n,) and v.dtype == np.float32 for v in ns.values()) and np.isfinite(ns["kl_layer_only"]).all()
    sub = npz_keys(W / "screen" / "subspan.npz", {"kl_sub", "sub_lo", "sub_hi", "n_explained"})
    assert sub["kl_sub"].dtype == np.float32 and sub["sub_lo"].dtype == sub["sub_hi"].dtype == sub["n_explained"].dtype == np.int16
    assert all(v.shape == (n,) for v in sub.values()) and (sub["n_explained"] >= 1).all() and (sub["sub_lo"] <= sub["sub_hi"]).all()
    assert (sub["sub_lo"] >= 0).all() and (sub["sub_hi"] < N_LAYERS).all(), "relaxed thresholds: every pair has an explained sub-span"
    ln = npz_keys(W / "layer_necessity" / "layer_necessity.npz", {"prompt", "layer_mask", "kl", "target_lp", "target_lp_clean"})
    assert ln["prompt"].shape == ln["kl"].shape == ln["layer_mask"].shape == (n,) and (ln["layer_mask"] > 0).all()
    with gzip.open(W / "screen" / "subspan_table.json.gz", "rt") as f:
        table = json.load(f)
    assert set(table) == {str(c) for c in np.unique(np.load(W / "assignments" / "pairs.npz")["cand"])}
    for c, rows in table.items():
        nl = len({g // D_MLP for g in cands[int(c)]["members"]})
        assert len(rows) == nl * (nl + 1) // 2 and all(len(r) == 7 and r[0] <= r[1] and r[2] >= 1 and r[3] <= r[4] <= r[5] for r in rows)
    rs = npz_keys(W / "subspan" / "rows_sorted.npz", {"key_u", "starts", "a", "b", "only", "off", "ns", "prompt", "cand"})
    m = len(rs["prompt"])
    assert rs["starts"][0] == 0 and rs["starts"][-1] == m and len(rs["starts"]) == len(rs["key_u"]) + 1 and (np.diff(rs["key_u"]) > 0).all()
    key = rs["prompt"].astype(np.int64) * 10_000_000 + rs["cand"]
    assert (np.diff(key) >= 0).all() and np.array_equal(key[rs["starts"][:-1]], rs["key_u"])
    assert rs["ns"].dtype == bool and rs["ns"].all() and rs["only"].dtype == rs["off"].dtype == np.float16
    s = json.loads((W / "coverage" / "summary.json").read_text())
    assert set(s) >= {"n_prompts", "tau_n", "tau_s", "approaches"} and s["n_prompts"] == io.n_live_prompts(W)
    for a in s["approaches"]:
        assert set(a) >= {"tile", "method", "param", "communities", "certified_communities", "prompts_covered", "prompts_frac", "neurons",
                          "certified_pairs", "certified_pairs_weighted", "prompts_covered_ht", "prompts_frac_ht", "neurons_per_prompt"}
        assert a["tile"] == "L0-L1" and a["method"] == "cpm" and 0 <= a["prompts_frac"] <= 1
    curves = np.load(W / "coverage" / "curves.npz")
    assert {k.split("|")[1] for k in curves.files} <= {"neurons", "prompts"} and len(curves.files) == 2 * len(s["approaches"])
    sp = json.loads((W / "coverage" / "summary_perspan.json").read_text())
    assert set(sp["by_min_prompts"]) == {"1", "5", "20"} and (W / "coverage" / "curves_perspan.npz").exists()
    print(f"aggregate subspan: {m} rows, {int(rs['ns'].sum())} certified; " + "; ".join(f"cpm{a['param']:g}: {a['prompts_covered']} prompts" for a in s["approaches"]))


def test_15_span_local(st):
    W = st["W"]; W2 = st["W2"]; D = W / "span_necessity"
    rs = np.load(W / "subspan" / "rows_sorted.npz"); m = len(rs["prompt"])
    stage("span-local", "prep", "--world", W, "--shards", 2)
    spec = json.loads((D / "spec.json").read_text())
    assert set(spec) >= {"world", "manifest_dir", "datasets", "n_rows", "shards", "communities"} and spec["n_rows"] == m and spec["shards"] == 2
    assert set(spec["communities"]) == {str(c) for c in np.unique(rs["cand"])}
    ins = [npz_keys(D / "in" / f"rows_{k:03d}.npz", {"prompt", "cand", "a", "b"}) for k in range(2)]
    assert sum(len(z["prompt"]) for z in ins) == m and not (set(ins[0]["prompt"].tolist()) & set(ins[1]["prompt"].tolist()))
    assert all((np.diff(z["prompt"]) >= 0).all() for z in ins), "rows of a shard are sorted by prompt"
    for k in range(2):
        stage("span-local", "rows", "--world", W, "--shard", k, "--model", st["model"], "--device", "cpu")
        z = npz_keys(D / "out" / f"kl_{k:03d}.npz", {"prompt", "cand", "a", "b", "kl"})
        assert z["kl"].dtype == np.float32 and np.isfinite(z["kl"]).all() and (z["kl"] > -1e-3).all()
        assert all(np.array_equal(z[f], ins[k][f]) for f in ("prompt", "cand", "a", "b")), "output rows are in input order"
    stage("span-local", "aggregate", "--world", W, "--out", W2, *RELAXED)
    nec = npz_keys(W2 / "subspan" / "nec_span.npz", {"kl"})["kl"]
    assert nec.shape == (m,) and np.isfinite(nec).all(), "every source row got a span-local value"
    r2 = npz_keys(W2 / "subspan" / "rows_sorted.npz", {"a", "b", "only", "off", "nec", "ns", "prompt", "cand"})
    assert len(r2["prompt"]) == m and r2["ns"].all() and r2["nec"].dtype == np.float16
    for name in ("candidates", "manifest.npz", "dedup_kept.npz", "dedup_drop.npz", "merged"):
        assert (W2 / name).is_symlink() and (W2 / name).exists()
    assert "span_local_necessity" in json.loads((W2 / "meta.json").read_text())
    with gzip.open(W2 / "screen" / "subspan_table.json.gz", "rt") as f:
        assert all(len(r) == 7 for rows in json.load(f).values() for r in rows)
    ts = json.loads((W2 / "coverage" / "tail_stats.json").read_text())
    for k in ("necessity", "n_kept", "certified_pairs", "certified_rows", "union_prompts_observed", "union_frac_observed", "union_prompts_ht",
              "per_source", "certified_communities", "certified_ge10", "median_size_all", "per_gamma"):
        assert k in ts, k
    assert ts["necessity"] == "span-local" and ts["n_kept"] == io.n_live_prompts(W) and ts["certified_rows"] == m
    cp = np.load(W2 / "coverage" / "certified_prompts.npy"); assert cp.dtype == np.int32 and len(cp) == ts["union_prompts_observed"]
    cc = npz_keys(W2 / "coverage" / "certified_communities.npz", {"cand", "size", "n_layers", "certified_observed", "certified_ht", "weight"})
    assert len(cc["cand"]) == ts["certified_communities"]
    assert set(json.loads((W2 / "coverage" / "summary.json").read_text())) >= {"n_prompts", "tau_n", "tau_s", "necessity", "union_frac", "per_gamma"}
    print(f"span-local: {m} rows re-certified, union coverage {ts['union_prompts_observed']}/{ts['n_kept']} prompts, {ts['certified_communities']} communities")


def test_16_summary(st):
    W = st["W"]
    stage("summary", "--world", W, "--tau-nec", "-1", "--tau-suff", "1e9")
    ts = json.loads((W / "coverage" / "tail_stats.json").read_text())
    for k in ("world", "n_kept", "n_pairs_assigned", "certified_pairs", "cap_active", "max_weight", "union_prompts_observed", "union_frac_observed",
              "union_prompts_ht", "union_frac_ht", "per_source", "certified_communities", "certified_ge10", "certified_ge50", "weight_bands",
              "top_capped_communities", "median_size_all", "size_hist_ge10", "per_gamma"):
        assert k in ts, k
    n_pairs = len(np.load(W / "assignments" / "pairs.npz")["prompt"])
    assert ts["n_pairs_assigned"] == n_pairs == ts["certified_pairs"] and ts["n_kept"] == io.n_live_prompts(W)
    assert 0 < ts["union_frac_observed"] <= 1 and ts["union_prompts_ht"] >= ts["union_prompts_observed"]
    assert set(ts["per_source"]) == {"all"} and ts["per_source"]["all"]["kept"] == ts["n_kept"]
    cp = np.load(W / "coverage" / "certified_prompts.npy"); assert cp.dtype == np.int32 and len(cp) == ts["union_prompts_observed"]
    cc = npz_keys(W / "coverage" / "certified_communities.npz", {"cand", "size", "n_layers", "span_width", "certified_observed", "certified_ht", "weight"})
    assert len(cc["cand"]) == ts["certified_communities"] and (cc["certified_ht"] >= cc["certified_observed"] - 1e-3).all()
    print(f"summary: union coverage {ts['union_prompts_observed']}/{ts['n_kept']} (HT {ts['union_prompts_ht']:.1f}), "
          f"{ts['certified_communities']} certified communities; per gamma {ts['per_gamma']}\nTOTAL {time.monotonic() - T0:.0f}s")


# ---- tokens stage on a local corpus ---------------------------------------------------------------------------------

def test_tokens_stage(tmp_path):
    """``tokens`` on a local JSONL directory (``datasets`` streams it like a hub dataset) with a tiny word-level tokenizer."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast
    words = [f"w{i}" for i in range(40)]
    vocab = {"<s>": 0, "</s>": 1, "<unk>": 2, **{w: 3 + i for i, w in enumerate(words)}}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 0)])
    PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", unk_token="<unk>").save_pretrained(tmp_path / "tokenizer")
    rng = np.random.default_rng(0)
    data = tmp_path / "corpus"; data.mkdir()
    texts = [" ".join(rng.choice(words, size=int(rng.integers(3, 60))).tolist()) for _ in range(40)]
    with open(data / "docs.jsonl", "w") as f:
        for i, t in enumerate(texts):
            f.write(json.dumps({"text": t, "id": f"doc{i}"}) + "\n")
    out = tmp_path / "tok"
    stage("tokens", "--hf-dataset", data, "--n-docs", 25, "--min-chars", 20, "--max-length", 32, "--chunk-tokens", 300,
          "--shuffle-buffer", 8, "--model", tmp_path / "tokenizer", "--out", out)
    rows = io.read_locations(out)
    assert len(rows) == 25 and io.tokens_site(out) == "tokens"
    conn = sqlite3.connect(str(out / "index.sqlite"))
    assert [r[0] for r in conn.execute("SELECT sequence_id FROM sequence_locations ORDER BY rowid")] == [f"seq_{i}" for i in range(25)]
    assert {t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")} >= {"sequence_locations", "sequence_metadata", "token_positions"}
    conn.close()
    meta = json.loads((out / "metadata.json").read_text())
    chunks = sorted((out / "layers" / "tokens").glob("chunk_*.safetensors"))
    assert meta["has_token_ids"] and meta["num_chunks"] == len(chunks) > 1 and meta["total_items"] == sum(n for _, _, n in rows)
    reader = io.TokensReader(out)
    for r, (c, o, n) in enumerate(rows):
        ids = io.open_chunk(out, c).get_tensor("token_ids")
        assert ids.dtype == torch.int32 and o + n <= len(ids) and 2 <= n <= 32
        doc = reader.doc(r)
        assert doc[0].item() == 0 and len(doc) == n and (doc[1:] >= 3).all(), "BOS first, then word ids"
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["n_docs"] == 25 and prov["max_length"] == 32 and prov["total_tokens"] == meta["total_items"] and Path(prov["hf_dataset"]) == data
    print(f"tokens: 25 docs, {meta['total_items']} tokens in {len(chunks)} chunks")
