"""Every constant of the method in one place (values are the ones used for the paper and the public site).

The model geometry (``N_LAYERS``, ``D_MLP``, the derived ``N_GID``, the discovery window ``LAYER_TILE`` and the hub
set ``HUBS``) can be overridden for tests and small models only: call :func:`configure` before building the CLI, or
set ``NEURON_COMMUNITIES_GEOMETRY="<n_layers>x<d_mlp>"`` in the environment (read once, when this module is
imported). Every stage reads these names from this module at call time, so an override reaches the whole package.
"""

from __future__ import annotations

import os

# ---- model ---------------------------------------------------------------------------------------------------------
MODEL_ID = "meta-llama/Llama-3.1-8B"
N_LAYERS = 32
D_MLP = 14336                      # neurons per MLP layer; a neuron's global id is layer * D_MLP + index
N_GID = N_LAYERS * D_MLP           # number of global neuron ids

# ---- prompts -------------------------------------------------------------------------------------------------------
CONTEXT_TOKENS = 64                # a prompt = the target position plus the 63 tokens before it
MIN_CONTEXT_FOR_SHINGLE_DEDUP = 8  # contexts shorter than this are deduplicated by exact match instead
BLOCK_SIZE = 64                    # prompts per harvest block (one block = one output file)

# ---- attribution harvest -------------------------------------------------------------------------------------------
TOP_K = 5                          # neurons kept per layer per prompt (|activation| x |gradient of the top-1 log-prob|)

# ---- co-occurrence census ------------------------------------------------------------------------------------------
C_MIN = 5                          # a neuron pair is stored when it co-occurs on >= C_MIN prompts ...
Z_STORE = 6.0                      # ... and its z-score against the popularity-preserving null is >= Z_STORE
N_REPLICATES = 20                  # curveball null replicates per layer pair

# ---- community discovery (CPM Leiden on the z-gated graph) ---------------------------------------------------------
LAYER_TILE = (0, 29)               # layers whose neurons enter the graph (the last two layers are left out)
CPM_EDGE_Z = 8.0                   # edge threshold z for the CPM graph
GAMMAS = (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)   # CPM resolution grid
SIZE_CAP = 128                     # communities above this size are dropped
MIN_SIZE = 2

# always-on "hub" neurons (in the top-k sets of nearly every prompt, one per layer where present) removed from every
# community after discovery; found by their frequency in the attribution sets of the 10M-prompt sample
HUBS = {2: 4786, 3: 6390, 5: 7012, 6: 5866, 7: 6673, 9: 4255, 10: 11570, 11: 11321,
        12: 13860, 13: 4208, 16: 1241, 18: 7417, 20: 3972, 23: 306, 24: 5326, 27: 8140, 29: 12010}

# ---- necessity and sufficiency -------------------------------------------------------------------------------------
MIN_PRESENT = 2                    # a prompt is assigned to a community when >= MIN_PRESENT members are in its top-k sets
ASSIGN_CAP = 20_000                # prompts tested per community (uniform sample above the cap, Horvitz-Thompson weights)
ASSIGN_CAP_LARGE = 2_000           # ... for communities with many members
ASSIGN_CAP_GIANT = 500
TAU_NEC = 0.7                      # KL(clean || community zero-ablated) >= TAU_NEC          -> necessary
TAU_SUFF = 0.2                     # KL(clean || only community + hubs kept) <= TAU_SUFF    -> sufficient
TAU_LAYER = 0.7                    # KL(clean || only hubs kept in the span) >= TAU_LAYER   -> the span's layers matter
MAX_SPAN_LAYERS = 24               # communities spanning more member layers are not sub-span tested


def hub_string(hubs: dict[int, int] | None = None) -> str:
    """'L2:4786,L3:6390,...' form used on the command line (empty string for no hubs)."""
    return ",".join(f"L{l}:{i}" for l, i in sorted((HUBS if hubs is None else hubs).items()))


def parse_hubs(s: str) -> dict[int, int]:
    return {int(t.split(":")[0].lstrip("L")): int(t.split(":")[1]) for t in s.split(",") if t.strip()}


# ---- geometry override (tests and small models only) ---------------------------------------------------------------

GEOMETRY_ENV = "NEURON_COMMUNITIES_GEOMETRY"


def configure(n_layers: int, d_mlp: int, layer_tile: tuple[int, int] | None = None, hubs: dict[int, int] | None = None) -> None:
    """Rewrite the model geometry for a model other than Llama-3.1-8B (tests and small models only).

    Sets ``N_LAYERS``, ``D_MLP`` and the derived ``N_GID``; ``LAYER_TILE`` becomes ``layer_tile`` or, when not given,
    the whole model ``(0, n_layers - 1)``; ``HUBS`` becomes ``hubs`` or, when not given, ``{}`` (no always-on neurons
    are known for an arbitrary model). Call it before building the command-line parser: argument defaults such as
    ``--layers`` / ``--hubs`` are taken from these names when the parser is built.
    """
    global N_LAYERS, D_MLP, N_GID, LAYER_TILE, HUBS
    n_layers, d_mlp = int(n_layers), int(d_mlp)
    if n_layers < 1 or d_mlp < 1:
        raise ValueError(f"configure: need n_layers >= 1 and d_mlp >= 1, got {n_layers}x{d_mlp}")
    N_LAYERS, D_MLP, N_GID = n_layers, d_mlp, n_layers * d_mlp
    LAYER_TILE = (0, n_layers - 1) if layer_tile is None else (int(layer_tile[0]), int(layer_tile[1]))
    HUBS = {} if hubs is None else {int(l): int(i) for l, i in hubs.items()}
    if not 0 <= LAYER_TILE[0] <= LAYER_TILE[1] < n_layers:
        raise ValueError(f"configure: layer_tile {LAYER_TILE} outside 0..{n_layers - 1}")
    if any(not 0 <= l < n_layers or not 0 <= i < d_mlp for l, i in HUBS.items()):
        raise ValueError(f"configure: hubs {HUBS} outside the {n_layers}x{d_mlp} geometry")


def _configure_from_env() -> None:
    """``NEURON_COMMUNITIES_GEOMETRY="<n_layers>x<d_mlp>"`` -> :func:`configure` (read once, at import)."""
    spec = os.environ.get(GEOMETRY_ENV, "").strip()
    if not spec:
        return
    try:
        n_layers, d_mlp = (int(x) for x in spec.lower().split("x"))
    except ValueError:
        raise ValueError(f"{GEOMETRY_ENV}={spec!r}: expected '<n_layers>x<d_mlp>', e.g. '2x64'") from None
    configure(n_layers, d_mlp)


_configure_from_env()
