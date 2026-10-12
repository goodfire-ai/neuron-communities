#!/bin/bash
#SBATCH --job-name=neuron-communities
#SBATCH --gres=gpu:1
#SBATCH --array=0-0
#SBATCH --output=slurm-%x-%A_%a.log
# One template for every sharded stage: STAGE and ARGS come from the environment, the shard from the array index, e.g.
#   sbatch --array=0-511 --export=ALL,STAGE=necessity,ARGS="--world worlds/llama31_8b/heldout/world" scripts/slurm/stage_array.slurm
# (same for attribution, subspan and STAGE=span-local ARGS="rows --world ..."; census and leiden take SHARD_FLAG=--tile / --run
# with no --shards, and need no GPU: add --gres=gpu:0). Shards whose output exists are skipped by the stage itself.
set -euo pipefail
: "${STAGE:?set STAGE (attribution | necessity | subspan | span-local | census | leiden)}"
SHARD_FLAG=${SHARD_FLAG:---shard}
SHARDS_FLAG=""; [ "$SHARD_FLAG" = --shard ] && SHARDS_FLAG="--shards $SLURM_ARRAY_TASK_COUNT"
set -x
python -m neuron_communities "$STAGE" ${ARGS:-} "$SHARD_FLAG" "$SLURM_ARRAY_TASK_ID" $SHARDS_FLAG
