# Neuron communities

Code for *Neuron-sparse behaviors in an LLM: small neuron communities causally necessary and sufficient for many
computations*. It finds small groups of MLP neurons in Llama-3.1-8B that act together, and tests on held-out prompts
whether each group is necessary (zero-ablating it changes the next-token prediction) and sufficient (keeping only it,
inside its layers, preserves the prediction).

Explore the result: **https://goodfire-ai.github.io/neuron-communities/** (1,941 labelled communities with their
prompts, interventions and statistics).

Everything here was run on `meta-llama/Llama-3.1-8B`. The code is model-agnostic for Llama-style architectures but the
constants in `neuron_communities/config.py` (layer tile, hub neurons, thresholds) are the ones used for that model.

## Method in one page

1. **Prompts.** Sample target positions uniformly from tokenised FineWeb-Edu, Wikipedia and FineMath documents; a prompt is
   the target position plus its 63 preceding tokens. Deduplicate by 64-token window (and by exact match for contexts
   shorter than 8 tokens). A held-out set is sampled with a different seed and cross-deduplicated against the training
   prompts.
2. **Attribution sets.** For every prompt and layer, keep the top-5 neurons by |activation| × |gradient| of the
   log-probability of the model's top-1 next token at the target position. This gives a sparse prompt × neuron
   matrix per layer.
3. **Co-occurrence graph.** For every pair of layers, count on how many prompts two neurons co-occur in the top-5 sets,
   compare with a popularity-preserving (curveball) null, and keep pairs with z ≥ 6. Partition the z-gated graph over
   layers 0–29 with CPM Leiden at a grid of resolutions γ; communities larger than 128 neurons are dropped, and the
   17 always-on hub neurons are removed from every community afterwards.
4. **Necessity and sufficiency.** For each community, take the prompts on which at least two members are in the top-5
   sets (capped per community with Horvitz-Thompson weights). Necessity: KL(clean ‖ community zero-ablated) at the
   target position ≥ 0.7. Sufficiency: for every contiguous sub-span of the community's layers, keep only the members
   (plus the hubs) inside that span and leave everything else clean; the prompt passes if that KL ≤ 0.2 while keeping
   only the hubs gives KL ≥ 0.7. Span-local necessity then re-certifies each (prompt, community, span) by ablating only
   the members inside the span. A community explains a prompt when necessity, sufficiency and layer necessity all pass
   on the same span.
5. **Statistics.** Union coverage of the held-out prompts over all γ stops (observed and Horvitz-Thompson), per-γ
   dictionary sizes, and coverage as a function of a community size cap.

## Pipeline

`python -m neuron_communities <stage> --help` lists the arguments of each stage. `scripts/run_pipeline.sh` runs them in
order on one machine with the paper's settings: `HuggingFaceFW/fineweb-edu` (`sample-10BT`), `wikimedia/wikipedia`
(`20231101.en`) and `HuggingFaceTB/finemath` (`finemath-4plus`) through the `tokens` stage, 5M + 2.5M + 2.5M prompts
for the training world and as many for the held-out world (`MODEL`, `WORLD`, the per-corpus tokens datasets, document
and prompt counts, seeds, `SHARDS` and `EVAL_SPLITS` are environment variables; every stage is skipped when its output
exists, so the script resumes). Stages that scale with the number of prompts take `--shard i --shards n` and write one output per
shard, skipping shards whose output exists, so they map directly onto a job array:
`scripts/slurm/stage_array.slurm` is one generic `#SBATCH --array` template for all of them (`STAGE` and `ARGS` in the
environment, the shard from the array index; `--tile` / `--run` for `census` / `leiden`).

| stage | what it does | reads | writes | GPU |
|---|---|---|---|---|
| `tokens` | tokenise a Hugging Face corpus (streaming, shard-shuffled, Llama-3.1-8B tokenizer) | HF dataset | `<tokens>/index.sqlite`, `layers/tokens/chunk_*.safetensors` | no |
| `sample` | freeze a uniform random prompt manifest over a tokens dataset | tokens | `<manifest>/manifest.npz`, `meta.json` | no |
| `dedup` | window-64 leader dedup, then exact dedup of contexts shorter than 8 tokens (`--short-only`: the second pass alone, e.g. on a combined world) | manifest | `dedup_kept.npz`, `dedup_drop.npz` | no |
| `combine` | concatenate per-corpus manifests into one world (shared tokens index) | manifests | `<world>/manifest.npz`, `dedup_kept.npz`, `meta.json`, combined tokens | no |
| `heldout-dedup` | drop held-out prompts that are a training prompt or share its 64-token context | two manifests | rewritten `dedup_kept.npz` (and `dedup_drop.npz`), `cross_dedup.json` | no |
| `attribution` | top-5 neurons per layer by activation × gradient of the target log-prob, per prompt | world, model | `<harvest>/blocks/block_*.npz` (sharded) | yes |
| `world` | assemble the blocks into per-layer CSR matrices in manifest order | harvest | `<world>/merged/L??.npz` | no |
| `census` | co-occurrence counts for one layer pair with curveball nulls; z ≥ 6 pairs kept | world | `<world>/census/pairs_L??_L??.npz` (one tile per (a, b)) | no |
| `leiden` | CPM Leiden at every γ on the z-gated graph of layers 0–29, size cap 128 (`--run i` per γ, `--merge-runs`) | census | `<out-dir>/runs/run_*.json.gz`, `candidates.json.gz` | no |
| `merge` | merge discoveries into one dictionary deduplicated by member set | candidates | `candidates.json.gz` with provenance | no |
| `hubs` | strip the 17 always-on hub neurons from every community | Leiden runs | `candidates.json.gz` | no |
| `assign` | prompts to test per community (≥ 2 members present; capped with HT weights) | world, candidates | `assignments/pairs.npz` | no |
| `necessity` | KL(clean ‖ community zero-ablated) per pair, cached-prefix replay | world, pairs, model | `necessity/nec_shard*.npz` (sharded) | yes |
| `aggregate necessity` | collect shards, keep pairs with KL ≥ 0.7 | shards | `screen/kl_drop.npy`, `subspan/pairs.npz` | no |
| `subspan` | for every contiguous sub-span: keep only members + hubs (sufficiency), only hubs (layer necessity) | world, pairs, model | `subspan/subspan_shard*.npz` (sharded) | yes |
| `aggregate subspan` | certify prompts (nec ≥ 0.7, suff ≤ 0.2, layer ≥ 0.7), coverage cells | shards | `screen/*.npz`, `subspan/rows_sorted.npz`, `coverage/` | no |
| `span-local prep / rows / aggregate` | re-certify every (prompt, community, span) with span-local necessity | world, model | `span_necessity/in/rows_*.npz`, `out/kl_*.npz` (sharded); a statistics world with `subspan/nec_span.npz`, `coverage/tail_stats.json` | rows: yes |
| `summary` | union coverage (observed + HT), per-γ dictionary sizes, community size statistics | world | `coverage/tail_stats.json` (printed), `certified_prompts.npy`, `certified_communities.npz` | no |

Measured on the paper's 10.5M-prompt world with the model in bfloat16 on 80 GB GPUs: the attribution harvest takes
about 6 s per block of 64 prompts (roughly 2.5 GPU-hours per 100k prompts, embarrassingly parallel over shards); one
census tile (a layer pair) takes about 70 min on 10 CPU cores, and the 0–29 window has 465 tiles; one CPM Leiden run on
the full graph takes about an hour per γ; necessity evaluates roughly 250k (prompt, community) pairs per GPU-minute and
the sub-span stage roughly 80k sub-span rows per GPU-minute, both sharded.

## Install

```
pip install -e .
```

`torch`, `transformers`, `safetensors`, `datasets`, `numpy`, `scipy`, `python-igraph` and `leidenalg` are the only
dependencies (`pytest` for the test). The GPU stages were run on single 80 GB GPUs with the model in bfloat16.

## Test

`pytest -s tests` (about two minutes on 8 CPU cores, no network, no GPU) runs every stage of the pipeline in order on a
randomly initialised two-layer Llama (hidden 64, 64 neurons per layer, saved to a temporary directory) and a synthetic
tokens dataset of 300 documents over a 512-token vocabulary with repeated phrases and duplicate documents, written with
the package's own index writer. The stages run in-process through the command-line entry point with the geometry
override `config.configure(2, 64)` (`NEURON_COMMUNITIES_GEOMETRY=2x64` on the command line): `sample` and `dedup` of a
training and a held-out sample, `heldout-dedup`, `attribution` in two shards, `world`, the three `census` tiles of the
0–1 window, `leiden` at two γ (sharded runs, `--merge-runs`, and in-process, which must agree), `merge`, `hubs`,
`assign`, `necessity` in two shards, `aggregate necessity`, `subspan` in two shards, `aggregate subspan`, `span-local`
prep / rows / aggregate and `summary`. After each stage it asserts that the documented files exist with the documented
keys, shapes, dtypes and invariants (manifest and dedup index ranges, block and CSR layout in manifest order, census tile
triangles and offsets, candidate record format and uniqueness, pair sorting and shard coverage, sub-span row counts,
`rows_sorted` lookup structure, the statistics JSON keys) and prints the per-stage counts; it does not check the numbers,
which only mean something on the real model, and relaxes the certification thresholds so that every evaluation stage has
rows to process. If Leiden finds fewer than three communities on the random model, the discovery outputs are still
checked and a few communities are planted from the most frequent neurons so the evaluation stages run. A separate test
runs the `tokens` stage on a local JSONL directory with a tiny word-level tokenizer.

## Citation

<!-- CITATION -->

## License

MIT.
