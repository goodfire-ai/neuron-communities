"""Activation x gradient attribution sets over a frozen manifest (the forward x backward harvest).

Every manifest target gets, per layer, the top-k neurons by ``|h| x |dL/dh|`` where ``h`` is the SwiGLU neuron
activation at the target position and ``L = log p(top-1 next token)`` (the model's own argmax; the variant the paper
calls ``act_ga``). ``|h * dL/dh|`` is the first-order estimate of the effect of zeroing the neuron on the model's
confidence in its prediction.

Mechanics: a target-position loss depends on a target-position neuron only through the target row of the layers
above it (nothing else attends to the last position), so the backward pass runs through CachedTail's single-row
replay: one clean trunk forward (no grad) captures resid_post@L0 and the layer-0 neurons; CachedTail(split_layer=0)
.precompute caches context K/V; one grad-enabled step_logits over the tail layers (with the layer-0 MLP recomputed
from a leaf) yields the logits, and torch.autograd.grad gives dL/dh for all layers at once. Memory is that of a
single-token row, whatever the prefix length. The layer count comes from the loaded model.

Output per block of ``--block-size`` targets (``<out-dir>/blocks/block_XXXXXX.npz``, atomic, skip-if-exists):
``idx_act_ga`` (n, layers, k) uint16 and ``val_act_ga`` (n, layers, k) float32 (gradient products underflow fp16),
``row_index`` / ``position`` / ``manifest_lo`` / ``k`` / ``variants`` (= ["act_ga"]), the per-row top-1 log-prob
``logp_ga``, a ``teacher_valid`` flag (False when the manifest position is the document's last token, i.e. there is
no corpus next token), and a crosscheck against a naive full-sequence backward on one row per sub-batch
(``crosscheck``: top-k overlap of |h| x |grad| at the check layer, ``crosscheck_rel_err``: max relative gradient
error over layers).

Blocks are assigned to shards contiguously: shard ``i`` of ``n`` owns blocks ``[i * bps, (i + 1) * bps)`` with
``bps = ceil(n_blocks / n)`` (or ``--blocks-per-shard``).

  python -m neuron_communities attribution --manifest-dir WORLD --out-dir HARVEST --k 5 --shard 0 --shards 64
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from neuron_communities import config
from neuron_communities.data.io import TokensReader, load_manifest_with_dataset
from neuron_communities.tail import CachedTail, kv_bytes_per_row_token_layer, load_model

VARIANT = "act_ga"


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--manifest-dir", type=Path, required=True, help="directory with manifest.npz and meta.json")
    ap.add_argument("--tokens", type=Path, default=None, help="tokens dataset (default: meta.json 'dataset')")
    ap.add_argument("--out-dir", type=Path, required=True, help="harvest dir; blocks go to <out-dir>/blocks/")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--blocks-per-shard", type=int, default=0, help="override ceil(n_blocks / shards)")
    ap.add_argument("--block-size", type=int, default=config.BLOCK_SIZE)
    ap.add_argument("--model", default=config.MODEL_ID)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--kv-budget-gb", type=float, default=8.0)
    ap.add_argument("--forward-token-budget", type=int, default=49152)
    ap.add_argument("--max-blocks", type=int, default=0, help="smoke: process at most this many blocks of the shard")


def argmax_logprob(logits: torch.Tensor) -> torch.Tensor:
    """Per-row (B,) log-probability of the model's top-1 next token (graph-connected)."""
    return torch.log_softmax(logits.float(), dim=-1).max(dim=-1).values


def run(args: argparse.Namespace) -> None:
    DEV = args.device
    K = args.k

    man_row, man_pos, dataset = load_manifest_with_dataset(args.manifest_dir, args.tokens)
    n_blocks_total = (len(man_row) + args.block_size - 1) // args.block_size
    bps = args.blocks_per_shard or (n_blocks_total + args.shards - 1) // args.shards
    first = args.shard * bps
    my_blocks = list(range(first, min(first + bps, n_blocks_total)))
    out_dir = args.out_dir / "blocks"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [b for b in my_blocks if not (out_dir / f"block_{b:06d}.npz").exists()]
    if args.max_blocks:
        todo = todo[: args.max_blocks]
    if not my_blocks:
        print(f"shard {args.shard}: no blocks (total {n_blocks_total})", flush=True)
        return
    print(f"shard {args.shard}/{args.shards}: blocks {my_blocks[0]}..{my_blocks[-1]}, {len(todo)} to do", flush=True)
    if not todo:
        return

    tokens = TokensReader(dataset)
    model = load_model(args.model, DEV)
    n_layers = len(model.model.layers)
    layer0 = model.model.layers[0]
    tail = CachedTail(model, split_layer=0)  # layers 1..n_layers-1 single-row replay
    tail_layers = [layer for _idx, layer in tail._layers]
    assert len(tail_layers) == n_layers - 1
    kv_bytes = kv_bytes_per_row_token_layer(model)
    check_layer = min(18, n_layers - 1)          # layer whose top-k sets the crosscheck compares

    def batches(targets):
        """Length-sorted sub-batches under the K/V and forward token budgets."""
        order = np.argsort([p for _, p in targets])
        i = 0
        while i < len(order):
            sub = [order[i]]
            i += 1
            while i < len(order):
                s_max = targets[order[i]][1] + 1
                rows = len(sub) + 1
                if rows * s_max * (n_layers - 1) * kv_bytes > args.kv_budget_gb * 1e9 \
                        or rows * s_max > args.forward_token_budget:
                    break
                sub.append(order[i])
                i += 1
            docs = [tokens.doc_tokens(r, p + 2) for r, p in (targets[r] for r in sub)]  # prefix + the corpus next token when it exists
            toks = [d[: targets[r][1] + 1] for d, r in zip(docs, sub)]
            valid = torch.tensor([len(d) == len(t) + 1 for d, t in zip(docs, toks)], device=DEV)
            s_max = max(len(t) for t in toks)
            input_ids = torch.zeros(len(sub), s_max, dtype=torch.long)
            attn = torch.zeros(len(sub), s_max, dtype=torch.long)
            for j, t in enumerate(toks):
                input_ids[j, : len(t)] = t
                attn[j, : len(t)] = 1
            tpos = torch.tensor([len(t) - 1 for t in toks], device=DEV)
            yield sub, input_ids.to(DEV), attn.to(DEV), tpos, valid

    def clean_capture(input_ids, attn, tpos):
        """Clean trunk forward: resid_post@L0 (B,S,d) and layer-0 neurons at the target (B,d_mlp)."""
        cap = {}
        h1 = layer0.register_forward_hook(lambda _m, _i, out: cap.__setitem__("rp0", out[0] if isinstance(out, tuple) else out))
        h2 = layer0.mlp.down_proj.register_forward_pre_hook(lambda _m, ins: cap.__setitem__("h0", ins[0]))
        try:
            with torch.no_grad():
                model.model(input_ids=input_ids, attention_mask=attn)
        finally:
            h1.remove(); h2.remove()
        rows = torch.arange(input_ids.shape[0], device=DEV)
        return cap["rp0"], cap["h0"][rows, tpos]

    def attribute(input_ids, attn, tpos):
        """Returns h (L, B, d_mlp) float, grad dL/dh (L, B, d_mlp) float, loss (B,) for L = log p(top-1)."""
        rp0, h0c = clean_capture(input_ids, attn, tpos)
        ctx = tail.precompute(rp0, None, attn, tpos)
        rows = torch.arange(input_ids.shape[0], device=DEV)
        with torch.enable_grad():
            h0 = h0c.detach().clone().requires_grad_(True)
            hidden0 = rp0[rows, tpos] - layer0.mlp.down_proj(h0c) + layer0.mlp.down_proj(h0)
            captured: list[torch.Tensor] = []
            handles = [layer.mlp.down_proj.register_forward_pre_hook(lambda _m, ins: captured.append(ins[0]))
                       for layer in tail_layers]
            try:
                logits = tail.step_logits(hidden0, ctx)
            finally:
                for hd in handles:
                    hd.remove()
            assert len(captured) == n_layers - 1, f"captured {len(captured)} tail MLP inputs"
            # differentiate w.r.t. the captured tensors themselves (a squeezed/indexed copy is not on the
            # path to the loss); the tail's MLP inputs are (B, 1, d_mlp)
            hs = [h0] + captured
            loss = argmax_logprob(logits)
            g = torch.autograd.grad(loss.sum(), hs)
            grad = torch.stack([gi.reshape(gi.shape[0], -1).float() for gi in g])  # (L, B, d_mlp)
        h = torch.stack([hi.detach().reshape(hi.shape[0], -1).float() for hi in hs])
        return h, grad, loss.detach()

    def naive_crosscheck(input_ids_row, attn_row, tpos_row, h_ref, g_ref):
        """Full-sequence forward+backward on one row (no CachedTail). Returns (top-k overlap of |h| x |grad|
        at the check layer, max over layers of relative L2 error of dL/dh)."""
        cap: list[torch.Tensor] = []
        handles = [model.model.layers[L].mlp.down_proj.register_forward_pre_hook(lambda _m, ins: cap.append(ins[0]))
                   for L in range(n_layers)]
        try:
            with torch.enable_grad():
                emb = model.model.embed_tokens(input_ids_row)
                emb.requires_grad_(True)  # anchors a graph so intermediate MLP inputs are grad-tracked
                out = model.model(inputs_embeds=emb, attention_mask=attn_row)
                hid = out.last_hidden_state[0, tpos_row]
                logits = model.lm_head(hid)[None]
                loss = argmax_logprob(logits)
                g = torch.autograd.grad(loss.sum(), cap)  # full-sequence grads; pick the target row below
        finally:
            for hd in handles:
                hd.remove()
        h_n = torch.stack([c.detach()[0, tpos_row].float() for c in cap])
        g_n = torch.stack([x[0, tpos_row].float() for x in g])
        rel = ((g_n - g_ref).norm(dim=-1) / (g_n.norm(dim=-1) + 1e-12)).max().item()
        top_ref = set(torch.topk((h_ref[check_layer].abs() * g_ref[check_layer].abs()), K).indices.tolist())
        top_n = set(torch.topk((h_n[check_layer].abs() * g_n[check_layer].abs()), K).indices.tolist())
        return len(top_ref & top_n) / K, rel

    def process_block(b: int) -> None:
        t0 = time.monotonic()
        lo, hi = b * args.block_size, min((b + 1) * args.block_size, len(man_row))
        targets = [(int(man_row[i]), int(man_pos[i])) for i in range(lo, hi)]
        n = len(targets)
        idx_out = np.zeros((n, n_layers, K), np.uint16)
        val_out = np.zeros((n, n_layers, K), np.float32)
        lp_out = np.full(n, np.nan, np.float32)
        valid_out = np.zeros(n, bool)
        checks, rels = [], []
        for sub, input_ids, attn, tpos, valid in batches(targets):
            h, grad, loss = attribute(input_ids, attn, tpos)
            val, idx = torch.topk(h.abs() * grad.abs(), K, dim=-1)  # (L, B, K)
            for jj, r in enumerate(sub):
                idx_out[r] = idx[:, jj].cpu().numpy().astype(np.uint16)
                val_out[r] = val[:, jj].cpu().numpy().astype(np.float32)
                valid_out[r] = bool(valid[jj])
                lp_out[r] = float(loss[jj])
            j = int(np.random.default_rng(b * 131 + int(sub[0])).integers(len(sub)))
            ov, rel = naive_crosscheck(input_ids[j: j + 1, : tpos[j] + 1], attn[j: j + 1, : tpos[j] + 1], tpos[j],
                                       h[:, j], grad[:, j])
            checks.append(ov); rels.append(rel)
            del h, grad, val, idx
            if DEV.startswith("cuda"):
                torch.cuda.empty_cache()
        payload = {"row_index": man_row[lo:hi], "position": man_pos[lo:hi], "manifest_lo": np.array([lo]),
                   "k": np.array([K]), "variants": np.array([VARIANT]), "crosscheck": np.array(checks, np.float32),
                   "crosscheck_rel_err": np.array(rels, np.float32), "teacher_valid": valid_out, "logp_ga": lp_out,
                   f"idx_{VARIANT}": idx_out, f"val_{VARIANT}": val_out}
        tmp = out_dir / f".block_{b:06d}.tmp.npz"
        np.savez_compressed(tmp, **payload)
        os.replace(tmp, out_dir / f"block_{b:06d}.npz")
        print(f"block {b}: crosscheck top{K} overlap {np.mean(checks):.3f}, grad rel err max {np.max(rels):.2e}; "
              f"teacher valid {valid_out.mean():.2f}; mean logp top-1 {np.nanmean(lp_out):.2f} "
              f"({(time.monotonic() - t0) / 60:.1f} min)", flush=True)

    for b in todo:
        process_block(b)
    print(f"shard {args.shard} done", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
