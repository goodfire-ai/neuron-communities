#!/bin/bash
# The whole pipeline on one machine, in stage order, with the paper's settings. Every stage is skipped when its output
# already exists, so the script can be re-run after an interruption; the sharded GPU stages loop over SHARDS shards and
# skip the shards whose output file exists (the same shard numbering as scripts/slurm/stage_array.slurm, so a world
# started here can be continued on a job array and vice versa).
#
#   MODEL=meta-llama/Llama-3.1-8B WORLD=worlds/llama31_8b SHARDS=512 scripts/run_pipeline.sh
#
# Settings (environment variables; defaults are the paper's):
#   MODEL             Hugging Face model id or local path (default meta-llama/Llama-3.1-8B)
#   WORLD             output root; everything below is written under it (default worlds/llama31_8b)
#   FINEWEB_TOKENS, WIKI_TOKENS, FINEMATH_TOKENS
#                     tokens datasets of the three corpora (default $WORLD/tokens/<corpus>, built by the tokens stage when
#                     missing from HuggingFaceFW/fineweb-edu sample-10BT, wikimedia/wikipedia 20231101.en and
#                     HuggingFaceTB/finemath finemath-4plus)
#   FINEWEB_DOCS, WIKI_DOCS, FINEMATH_DOCS
#                     documents tokenised per corpus (default 700000 / 400000 / 400000)
#   FINEWEB_PROMPTS, WIKI_PROMPTS, FINEMATH_PROMPTS
#                     prompts sampled per corpus for the training world (default 5000000 / 2500000 / 2500000); the held-out
#                     world samples the same counts with HELDOUT_SEED
#   SEED, HELDOUT_SEED
#                     sampling seeds of the training and held-out worlds (default 2 and 3)
#   SHARDS            shards of the attribution / necessity / subspan / span-local stages (default 512)
#   EVAL_SPLITS       worlds the dictionary is evaluated on: "heldout" (the paper), "train" or "train heldout"
#   DEVICE            torch device of the GPU stages (default cuda)
#   CENSUS_WORKERS    worker processes of a census tile (default 10)
#   PYTHON            interpreter (default python)
set -euo pipefail

MODEL=${MODEL:-meta-llama/Llama-3.1-8B}
WORLD=${WORLD:-worlds/llama31_8b}
FINEWEB_TOKENS=${FINEWEB_TOKENS:-$WORLD/tokens/fineweb}
WIKI_TOKENS=${WIKI_TOKENS:-$WORLD/tokens/wiki}
FINEMATH_TOKENS=${FINEMATH_TOKENS:-$WORLD/tokens/finemath}
FINEWEB_DOCS=${FINEWEB_DOCS:-700000}
WIKI_DOCS=${WIKI_DOCS:-400000}
FINEMATH_DOCS=${FINEMATH_DOCS:-400000}
FINEWEB_PROMPTS=${FINEWEB_PROMPTS:-5000000}
WIKI_PROMPTS=${WIKI_PROMPTS:-2500000}
FINEMATH_PROMPTS=${FINEMATH_PROMPTS:-2500000}
SEED=${SEED:-2}
HELDOUT_SEED=${HELDOUT_SEED:-3}
SHARDS=${SHARDS:-512}
EVAL_SPLITS=${EVAL_SPLITS:-heldout}
DEVICE=${DEVICE:-cuda}
CENSUS_WORKERS=${CENSUS_WORKERS:-10}
PYTHON=${PYTHON:-python}

CORPORA="fineweb wiki finemath"
run() { echo "+ $*"; "$@"; }
nc() { run "$PYTHON" -m neuron_communities "$@"; }
tokens_dir() { case $1 in fineweb) echo "$FINEWEB_TOKENS";; wiki) echo "$WIKI_TOKENS";; finemath) echo "$FINEMATH_TOKENS";; esac; }
n_prompts() { case $1 in fineweb) echo "$FINEWEB_PROMPTS";; wiki) echo "$WIKI_PROMPTS";; finemath) echo "$FINEMATH_PROMPTS";; esac; }
mkdir -p "$WORLD"
echo "model $MODEL, world $WORLD, $SHARDS shards, evaluation on: $EVAL_SPLITS"

# ---- 1. tokens: one tokens dataset per corpus (<tokens>/index.sqlite, layers/tokens/chunk_*.safetensors, metadata.json,
#         provenance.json), streamed from the Hugging Face hub, Llama-3.1-8B tokenizer, documents truncated to 4096 tokens
[ -f "$FINEWEB_TOKENS/index.sqlite" ] || nc tokens --hf-dataset HuggingFaceFW/fineweb-edu --hf-config sample-10BT --text-field text \
    --n-docs "$FINEWEB_DOCS" --seed "$SEED" --model "$MODEL" --out "$FINEWEB_TOKENS"
[ -f "$WIKI_TOKENS/index.sqlite" ] || nc tokens --hf-dataset wikimedia/wikipedia --hf-config 20231101.en --text-field text \
    --n-docs "$WIKI_DOCS" --seed "$SEED" --model "$MODEL" --out "$WIKI_TOKENS"
[ -f "$FINEMATH_TOKENS/index.sqlite" ] || nc tokens --hf-dataset HuggingFaceTB/finemath --hf-config finemath-4plus --text-field text \
    --n-docs "$FINEMATH_DOCS" --seed "$SEED" --model "$MODEL" --out "$FINEMATH_TOKENS"

# ---- 2. sample + dedup per corpus and split: <manifest>/manifest.npz, meta.json (sample); dedup_kept.npz (window-64
#         leader dedup) and dedup_drop.npz (exact dedup of contexts shorter than 8 tokens)
for split in train heldout; do
  seed=$SEED; [ "$split" = heldout ] && seed=$HELDOUT_SEED
  for c in $CORPORA; do
    M=$WORLD/manifests/${c}_$split
    [ -f "$M/manifest.npz" ] || nc sample --dataset "$(tokens_dir "$c")" --out "$M" --n "$(n_prompts "$c")" --seed "$seed"
    [ -f "$M/dedup_kept.npz" ] || nc dedup --manifest-dir "$M"
  done
done

# ---- 3. heldout-dedup per corpus: drops held-out prompts that are the same prompt as, or share the 64-token context of,
#         a training prompt; rewrites <heldout manifest>/dedup_kept.npz (+ remapped dedup_drop.npz), writes cross_dedup.json
for c in $CORPORA; do
  M=$WORLD/manifests/${c}_heldout
  [ -f "$M/cross_dedup.json" ] || nc heldout-dedup --new "$M" --old "$WORLD/manifests/${c}_train"
done

# ---- 4. combine: the three per-corpus manifests laid end to end -> <split>/base/manifest.npz (with source ids),
#         dedup_kept.npz, meta.json and <split>/tokens (one index over the per-corpus chunks); then the short-context pass
#         of dedup over the combined manifest -> <split>/base/dedup_drop.npz
for split in train heldout; do
  B=$WORLD/$split/base
  [ -f "$B/manifest.npz" ] || nc combine --source "$WORLD/manifests/fineweb_$split:fineweb" --source "$WORLD/manifests/wiki_$split:wiki" \
      --source "$WORLD/manifests/finemath_$split:finemath" --out "$B" --tokens-out "$WORLD/$split/tokens"
  [ -f "$B/dedup_drop.npz" ] || nc dedup --manifest-dir "$B" --short-only
done

# ---- 5. attribution (GPU, sharded): top-5 neurons per layer by |activation| x |gradient of the top-1 log-prob| per
#         prompt -> <split>/harvest/blocks/block_*.npz (64 prompts per block; the stage itself skips blocks that exist)
#         then world: the blocks as per-layer CSR matrices -> <split>/world/merged/L??.npz (+ manifest files, meta.json)
for split in train heldout; do
  B=$WORLD/$split/base; H=$WORLD/$split/harvest; W=$WORLD/$split/world
  for s in $(seq 0 $((SHARDS - 1))); do
    nc attribution --manifest-dir "$B" --out-dir "$H" --shard "$s" --shards "$SHARDS" --model "$MODEL" --device "$DEVICE"
  done
  [ -d "$W/merged" ] || nc world --harvest-dir "$H" --manifest-dir "$B" --out-dir "$W"
done

# ---- 6. census (CPU, one tile per layer pair of the 0-29 window = 465 tiles): co-occurrence counts against curveball
#         nulls -> train/world/census/pairs_L??_L??.npz (an existing tile is skipped by the stage)
W=$WORLD/train/world
N_TILES=$(nc census --world "$W" --n-tiles | tail -1)
for t in $(seq 0 $((N_TILES - 1))); do
  nc census --world "$W" --tile "$t" --workers "$CENSUS_WORKERS"
done

# ---- 7. leiden (CPU, one run per gamma of the grid): CPM Leiden on the z >= 8 graph -> train/world/cand_cpm/runs/run_*.json.gz,
#         pooled by --merge-runs into cand_cpm/candidates.json.gz (+ candidates_stats.json)
CD=$W/cand_cpm
N_RUNS=$(nc leiden --census-dir "$W/census" --out-dir "$CD" --n-runs | tail -1)
for r in $(seq 0 $((N_RUNS - 1))); do
  [ -f "$CD/runs/$(printf 'run_%03d.json.gz' "$r")" ] || nc leiden --census-dir "$W/census" --out-dir "$CD" --run "$r"
done
[ -f "$CD/candidates.json.gz" ] || nc leiden --census-dir "$W/census" --out-dir "$CD" --merge-runs

# ---- 8. merge: the discoveries as one dictionary deduplicated by member set, with per-gamma provenance
#         -> train/world/candidates_with_hubs/candidates.json.gz (the dictionary before hub removal, kept for reference)
#         hubs: the same rebuilt from the raw runs with the 17 always-on hub neurons stripped from every community
#         -> train/world/candidates/candidates.json.gz (the dictionary every evaluation stage reads)
[ -f "$W/candidates_with_hubs/candidates.json.gz" ] || nc merge --inputs "$CD/candidates.json.gz" --out "$W/candidates_with_hubs"
[ -f "$W/candidates/candidates.json.gz" ] || nc hubs --out "$W/candidates" --runs "$CD/runs"

# ---- 9. evaluation of the dictionary on each requested world (the held-out world gets the training dictionary by symlink)
for split in $EVAL_SPLITS; do
  W=$WORLD/$split/world
  if [ "$split" != train ] && [ ! -e "$W/candidates" ]; then run ln -s ../../train/world/candidates "$W/candidates"; fi

  # assign: prompts with >= 2 members present per community, capped with Horvitz-Thompson weights -> assignments/pairs.npz,
  #         match_counts.npz, meta.json
  [ -f "$W/assignments/pairs.npz" ] || nc assign --world "$W"

  # necessity (GPU, sharded): KL(clean || community zero-ablated) per pair -> necessity/nec_shard???.npz
  for s in $(seq 0 $((SHARDS - 1))); do
    [ -f "$W/necessity/$(printf 'nec_shard%03d.npz' "$s")" ] || nc necessity --world "$W" --shard "$s" --shards "$SHARDS" --model "$MODEL" --device "$DEVICE"
  done
  # aggregate necessity: screen/kl_drop.npy (aligned with pairs.npz) and subspan/pairs.npz (the pairs with KL >= 0.7)
  [ -f "$W/subspan/pairs.npz" ] || nc aggregate necessity --world "$W"

  # subspan (GPU, sharded): for every contiguous sub-span of a necessary pair's member layers, "only members + hubs"
  #         (sufficiency) and "hubs only" (layer necessity) -> subspan/subspan_shard???.npz
  for s in $(seq 0 $((SHARDS - 1))); do
    [ -f "$W/subspan/$(printf 'subspan_shard%03d.npz' "$s")" ] || nc subspan --world "$W" --shard "$s" --shards "$SHARDS" --model "$MODEL" --device "$DEVICE"
  done
  # aggregate subspan: certification (nec >= 0.7, suff <= 0.2, layer >= 0.7) -> screen/ns_results.npz, screen/subspan.npz,
  #         screen/subspan_table.json.gz, layer_necessity/, subspan/rows_sorted.npz, coverage/summary*.json, curves*.npz
  [ -f "$W/subspan/rows_sorted.npz" ] || nc aggregate subspan --world "$W"

  # span-local necessity: prep -> span_necessity/spec.json, in/rows_???.npz; rows (GPU, sharded) -> out/kl_???.npz;
  #         aggregate -> the statistics world <split>/world_spanlocal (subspan/nec_span.npz, re-certified rows_sorted.npz,
  #         screen/subspan_table.json.gz, coverage/tail_stats.json = the paper's coverage numbers)
  SN=$W/span_necessity
  [ -f "$SN/spec.json" ] || nc span-local prep --world "$W" --shards "$SHARDS"
  for s in $(seq 0 $((SHARDS - 1))); do
    [ -f "$SN/out/$(printf 'kl_%03d.npz' "$s")" ] || nc span-local rows --world "$W" --shard "$s" --model "$MODEL" --device "$DEVICE"
  done
  [ -f "${W}_spanlocal/coverage/tail_stats.json" ] || nc span-local aggregate --world "$W" --out "${W}_spanlocal"

  # summary: union coverage (observed + Horvitz-Thompson), per-gamma dictionary sizes, community size statistics under
  #         whole-community necessity -> coverage/tail_stats.json, certified_prompts.npy, certified_communities.npz (printed)
  nc summary --world "$W"
done
echo "done: $WORLD"
