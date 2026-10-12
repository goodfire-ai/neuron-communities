"""Standalone replay of the top of a Llama model, with cached-prefix replay for target-position-only interventions.

Shared by the attribution harvest (the backward pass runs through ``CachedTail.step_logits``) and by every
evaluation stage (``Trunk``).

`ModelTail` runs the frozen layers **after** a split layer (e.g. layers 19-31 + final norm + lm_head) on a given residual
stream — it is **not** a patch into a running forward. It mirrors `LlamaModel.forward` exactly (same `create_causal_mask`
and `rotary_emb` helpers, same per-layer call); no KV cache, a single full forward over the sequence. It is a plain class
(not an `nn.Module`): it owns no parameters of its own, only references into the base model.

`CachedTail` is the cached-context variant used by every intervention in this package. All interventions are
target-token-only: the context positions ``< t`` stay clean, so their layer-``L+1..`` keys/values do not depend on the
intervention. One `ModelTail` forward per prompt captures the clean per-layer K/V (and the clean target logits), after which
every intervened row replays **only the target position** — attention reads the frozen context K/V plus the row's own
key/value at ``t``. Per-row cost is then nearly independent of prefix length, which is what makes thousands of
(community, sub-span) rows per prompt tractable.

The step path re-implements Llama attention for a single query position (GQA via grouped matmuls, softmax in float32
exactly like the eager attention path) and matches `ModelTail` values bit for bit. The layer count and the attention
geometry come from the loaded model, so any Llama-style ``AutoModelForCausalLM`` works.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformers.masking_utils import create_causal_mask

from neuron_communities import config


def load_model(model_id: str = config.MODEL_ID, device: str = "cuda"):
    """HF causal LM in bf16 on ``device``, eval mode, no parameter gradients."""
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    model.eval().requires_grad_(False)
    return model


def kv_bytes_per_row_token_layer(model) -> int:
    """Bytes of cached K+V per (row, token, tail layer) for the K/V budget of batched replays."""
    cfg = model.config
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
    return 2 * int(cfg.num_key_value_heads) * head_dim * 2   # k+v, kv heads, head_dim, bf16


class ModelTail:
    """Replay frozen ``model.layers[split_layer+1:] + norm + lm_head`` on a residual stream."""

    def __init__(self, model, split_layer: int) -> None:
        """``model`` is a (frozen) ``LlamaForCausalLM``; ``split_layer`` is the layer whose output residual is fed in.

        The residual fed to :meth:`forward` is ``resid_post@split_layer`` (the output of that decoder layer), so the tail
        begins at ``split_layer + 1``.
        """
        self.config = model.config
        # Plain references — assigning these as attributes does NOT register submodules because this
        # is not an nn.Module, so the frozen base params never leak into ModelTail.parameters().
        self._layers = list(enumerate(model.model.layers[split_layer + 1 :], start=split_layer + 1))
        self._norm = model.model.norm
        self._rotary_emb = model.model.rotary_emb
        self._lm_head = model.lm_head
        self.split_layer = split_layer

    @torch.no_grad()
    def _position_ids(self, resid: torch.Tensor, position_ids: torch.Tensor | None) -> torch.Tensor:
        if position_ids is not None:
            return position_ids
        seq_len = resid.shape[1]
        return torch.arange(seq_len, device=resid.device).unsqueeze(0)

    def forward(
        self,
        resid_post: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_positions: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``resid_post`` ``(B, S, d_model)`` -> logits ``(B, S, vocab)``.

        ``attention_mask`` is the ``(B, S)`` 1/0 padding mask; the causal mask is built from it with the same helper
        `LlamaModel.forward` uses.
        """
        position_ids = self._position_ids(resid_post, position_ids)
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=resid_post,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )
        position_embeddings = self._rotary_emb(resid_post, position_ids)

        hidden_states = resid_post
        for _layer_idx, layer in self._layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
            )
        hidden_states = self._norm(hidden_states)
        if output_positions is not None:
            if isinstance(output_positions, int):
                hidden_states = hidden_states[:, output_positions]
            else:
                rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)
                hidden_states = hidden_states[rows, output_positions]
        return self._lm_head(hidden_states)

    __call__ = forward


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary embedding for ``x`` of shape (B, heads, S, head_dim); cos/sin of shape (B, S, head_dim)."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (x * cos) + (_rotate_half(x) * sin)


@dataclass
class TailContext:
    """Frozen clean-context state for one batch of targets (everything a step needs)."""

    keys: list[torch.Tensor]  # per tail layer: (B, kv_heads, S, head_dim), RoPE applied
    values: list[torch.Tensor]  # per tail layer: (B, kv_heads, S, head_dim)
    context_valid: torch.Tensor  # (B, S) bool — True at clean context positions (< target, unpadded)
    cos_target: torch.Tensor  # (B, 1, head_dim) rotary at each row's target position
    sin_target: torch.Tensor
    clean_target_logits: torch.Tensor  # (B, vocab) clean logits at the target positions
    target_positions: torch.Tensor  # (B,) long


class CachedTail:
    """Single-position tail replay against a precomputed clean-context K/V cache."""

    def __init__(self, model, split_layer: int) -> None:
        self._tail = ModelTail(model, split_layer=split_layer)
        self._layers = self._tail._layers
        self._norm = self._tail._norm
        self._rotary_emb = self._tail._rotary_emb
        self._lm_head = self._tail._lm_head
        config = model.config
        self._num_kv_heads = int(config.num_key_value_heads)
        self._num_q_heads = int(config.num_attention_heads)
        self._head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        self._groups = self._num_q_heads // self._num_kv_heads

    @property
    def tail(self) -> ModelTail:
        """The plain full-sequence tail (for callers that also need uncached replays)."""
        return self._tail

    @torch.no_grad()
    def precompute(
        self,
        resid_post: torch.Tensor,
        position_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        target_positions: torch.Tensor,
    ) -> TailContext:
        """One clean full-prefix tail forward: capture per-layer K/V, return the step context.

        ``resid_post`` (B, S, d_model) holds each row's clean causal prefix, right-padded; each
        row's target is at ``target_positions[row]`` (its last real position). The captured K/V at
        the target position itself is masked out of ``context_valid`` — the step path recomputes
        the target's own key/value from the intervened hidden state every step.
        """
        batch, seq_len, _ = resid_post.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=resid_post.device).unsqueeze(0).expand(batch, -1)

        captured_k: list[torch.Tensor] = []
        captured_v: list[torch.Tensor] = []
        handles = []
        for _layer_idx, layer in self._layers:
            handles.append(
                layer.self_attn.k_proj.register_forward_hook(
                    lambda _m, _i, out: captured_k.append(out)
                )
            )
            handles.append(
                layer.self_attn.v_proj.register_forward_hook(
                    lambda _m, _i, out: captured_v.append(out)
                )
            )
        try:
            clean_logits = self._tail(
                resid_post, position_ids, attention_mask, output_positions=target_positions
            )
        finally:
            for handle in handles:
                handle.remove()
        if len(captured_k) != len(self._layers) or len(captured_v) != len(self._layers):
            raise RuntimeError("K/V capture saw an unexpected number of projection calls")

        positions = torch.arange(seq_len, device=resid_post.device).unsqueeze(0)
        context_valid = positions < target_positions.unsqueeze(1)
        if attention_mask is not None:
            context_valid = context_valid & attention_mask.bool()

        cos, sin = self._rotary_emb(resid_post, position_ids)
        keep = context_valid[:, None, :, None]
        keys, values = [], []
        for k_flat, v_flat in zip(captured_k, captured_v, strict=True):
            k = k_flat.view(batch, seq_len, self._num_kv_heads, self._head_dim).transpose(1, 2)
            v = v_flat.view(batch, seq_len, self._num_kv_heads, self._head_dim).transpose(1, 2)
            # Zero invalid slots: padded rows can be NaN after earlier tail layers (fully-masked
            # attention queries), and NaN survives an additive -inf bias in the step softmax.
            keys.append(torch.where(keep, _apply_rope(k, cos, sin), 0.0).contiguous())
            values.append(torch.where(keep, v, 0.0).contiguous())

        rows = torch.arange(batch, device=resid_post.device)
        return TailContext(
            keys=keys,
            values=values,
            context_valid=context_valid,
            cos_target=cos[rows, target_positions].unsqueeze(1),
            sin_target=sin[rows, target_positions].unsqueeze(1),
            clean_target_logits=clean_logits,
            target_positions=target_positions,
        )

    def step_logits(
        self,
        hidden_target: torch.Tensor,
        context: TailContext,
        mlp_masks: dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Logits at the target positions from intervened ``resid_post`` rows.

        ``hidden_target`` (B, d_model) is the intervened residual at each row's target position.
        Attention per tail layer reads the frozen context K/V plus this step's own key/value at the
        target (a position always attends to itself), so the intervention touches only the target
        position — exactly the target-token-only intervention.

        ``mlp_masks`` optionally maps an absolute tail layer index to a per-row effective
        membership ``(B, d_mlp)``: that layer's MLP output at the target position becomes
        ``down_proj(mask * act(gate(x)) * up(x))`` — computed live, so it composes correctly with
        interventions below it (a mask in one layer changing the next layer's input). Context
        positions are untouched (they live in the frozen K/V).
        """
        batch, d_model = hidden_target.shape
        heads_q, heads_kv, groups, head_dim = self._num_q_heads, self._num_kv_heads, self._groups, self._head_dim
        neg_inf = torch.finfo(hidden_target.dtype).min
        # (B, kv_heads, groups, S) mask addend over context slots; the self slot is always valid.
        context_bias = torch.where(
            context.context_valid[:, None, None, :], 0.0, neg_inf
        ).to(hidden_target.dtype)
        scale = head_dim**-0.5

        hidden = hidden_target.unsqueeze(1)
        for index, (layer_idx, layer) in enumerate(self._layers):
            attn_input = layer.input_layernorm(hidden)
            q = layer.self_attn.q_proj(attn_input).view(batch, 1, heads_q, head_dim).transpose(1, 2)
            k = layer.self_attn.k_proj(attn_input).view(batch, 1, heads_kv, head_dim).transpose(1, 2)
            v = layer.self_attn.v_proj(attn_input).view(batch, 1, heads_kv, head_dim).transpose(1, 2)
            q = _apply_rope(q, context.cos_target, context.sin_target)
            k = _apply_rope(k, context.cos_target, context.sin_target)

            # GQA without expanding the cache: group the query heads under their kv head.
            q = q.reshape(batch, heads_kv, groups, head_dim)
            scores_context = (
                torch.matmul(q, context.keys[index].transpose(-1, -2)) * scale + context_bias
            )
            scores_self = (q * k.reshape(batch, heads_kv, 1, head_dim)).sum(-1, keepdim=True) * scale
            weights = torch.softmax(
                torch.cat([scores_context, scores_self], dim=-1).float(), dim=-1
            ).to(hidden.dtype)
            attn_out = torch.matmul(weights[..., :-1], context.values[index]) + weights[
                ..., -1:
            ] * v.reshape(batch, heads_kv, 1, head_dim)
            attn_out = attn_out.reshape(batch, heads_q, head_dim).reshape(batch, 1, heads_q * head_dim)
            hidden = hidden + layer.self_attn.o_proj(attn_out)
            mlp_input = layer.post_attention_layernorm(hidden)
            if mlp_masks is not None and layer_idx in mlp_masks:
                neurons = layer.mlp.act_fn(layer.mlp.gate_proj(mlp_input)) * layer.mlp.up_proj(mlp_input)
                effective = mlp_masks[layer_idx].to(neurons.dtype).unsqueeze(1)
                hidden = hidden + layer.mlp.down_proj(neurons * effective)
            else:
                hidden = hidden + layer.mlp(mlp_input)
        return self._lm_head(self._norm(hidden)).squeeze(1)


class Trunk:
    """One model, a `CachedTail` per split layer, and the per-prompt clean pass at a split layer.

    The clean pass runs layers ``0..lo`` once on the prompt's prefix, captures the residual after layer ``lo`` and the
    input of layer ``lo``'s MLP, precomputes the tail's clean-context K/V and returns everything an intervention row needs:
    the target residual, the split layer's neuron activations and ``W_down`` (so members *of the split layer itself* can be
    removed by subtracting their writes from the residual), and the clean log-probabilities / probabilities at the target.
    """

    def __init__(self, model_name: str = config.MODEL_ID, device: str = "cuda"):
        self.device = device
        self.model = load_model(model_name, device)
        self.tails: dict[int, CachedTail] = {}

    @property
    def n_layers(self) -> int:
        return len(self.model.model.layers)

    def tail(self, lo: int) -> CachedTail:
        if lo not in self.tails:
            self.tails[lo] = CachedTail(self.model, split_layer=lo)
        return self.tails[lo]

    def clean(self, tk: torch.Tensor, lo: int):
        """Run the trunk; return (ctx, rp, residual at target, split-layer neuron acts, W_down, base_lp, base_p)."""
        DEV = self.device
        base_layer = self.model.model.layers[lo]; base_mlp = base_layer.mlp
        input_ids = tk.unsqueeze(0).to(DEV); attn = torch.ones_like(input_ids)
        tpos = torch.tensor([len(tk) - 1], device=DEV)
        cap = {}
        h1 = base_layer.register_forward_hook(lambda _m, _i, o_: cap.__setitem__("rp", o_[0] if isinstance(o_, tuple) else o_))
        h2 = base_mlp.register_forward_hook(lambda _m, ins, _o: cap.__setitem__("mx", ins[0]))
        try:
            self.model.model(input_ids=input_ids, attention_mask=attn)
        finally:
            h1.remove(); h2.remove()
        rp = cap["rp"]; ctx = self.tail(lo).precompute(rp, None, attn, tpos)
        rp_t = rp[0, tpos[0]].float(); mx = cap["mx"][0, tpos[0]]
        neurons_base = (base_mlp.act_fn(base_mlp.gate_proj(mx)) * base_mlp.up_proj(mx)).float()
        base_lp = torch.log_softmax(ctx.clean_target_logits.float(), -1)[0]
        return ctx, rp, rp_t, neurons_base, base_mlp.down_proj.weight.float(), base_lp, base_lp.exp()


def kl_from_logits(base_p: torch.Tensor, base_lp: torch.Tensor, logits: torch.Tensor) -> np.ndarray:
    """KL(clean || intervened) per row from the intervened logits ``(B, vocab)``."""
    lp = torch.log_softmax(logits.float(), -1)
    return (base_p * (base_lp - lp)).sum(-1).cpu().numpy()
