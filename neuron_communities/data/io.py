"""Readers (and the index-schema writer) for the on-disk layouts every stage shares: tokens datasets, manifest dirs,
worlds and candidate dictionaries.

Tokens dataset (written by the ``tokens`` and ``combine`` stages)::

  <dataset>/index.sqlite                           sequence_locations(sequence_id, chunk_id, offset, count), one row per
                                                   document in insertion order (rowid is the document's row index);
                                                   empty sequence_metadata / token_positions tables for schema parity
  <dataset>/layers/<site>/chunk_XXXXXX.safetensors  tensor ``token_ids`` (int32): the documents of that chunk concatenated,
                                                   so document ``r`` is ``token_ids[offset:offset + count]``
  <dataset>/metadata.json                          mode, num_chunks, total_items, chunks [...], has_token_ids, ...
  <dataset>/provenance.json                        how the corpus was sampled (``tokens`` stage only)

A tokens-only dataset has a single site, ``tokens``; ``tokens_site`` returns the first site found so datasets that
also store activations can be read the same way.

Manifest dir (a frozen prompt sample over one tokens dataset)::

  <manifest>/manifest.npz           row_index (int64, document row), position (int64, target token; the prompt is the
                                    document prefix up to and including it); ``combine`` adds source (int8, index into
                                    meta.sources order), pattern and subcat (str, empty when the source had none)
  <manifest>/dedup_kept.npz         kept: indices into manifest.npz surviving the window-64 leader dedup (sorted)
  <manifest>/dedup_drop.npz         drop: indices into ``kept`` (not into the manifest) whose context is shorter than
                                    min_context tokens and duplicates an earlier kept prompt exactly; min_context: [int]
  <manifest>/meta.json              dataset (path of the tokens dataset, as given on the command line), n, ...

Kept-row numbering never changes once dedup_kept.npz is written: later stages index by kept row and filter
``dedup_drop`` rows out themselves.

World dir (a manifest dir plus everything the harvest and evaluation stages derive from it)::

  <world>/merged/L<LL>.npz          per-layer CSR matrices (indptr, indices, values, layer) of the top-k neuron sets per
                                    manifest entry (the ``world`` stage)
  <world>/candidates/candidates.json.gz
                                    the community dictionary: a list of ``{"members": [global neuron ids],
                                    "found": ["<lo>-<hi>@cpm<gamma>", ...], ...}``
  <world>/meta.json                 ``dataset`` as above, or (``combine`` stage) ``sources`` with a ``dataset`` and
                                    ``dataset_row_offset`` per corpus when several tokens datasets were laid end to end

A *kept row* ``p`` refers to manifest entry ``kept[p]`` and is the prompt id used by every evaluation file (pairs.npz,
shards, screen/). Dataset paths recorded in meta.json are used as written when absolute; relative paths resolve against
``dataset_root`` when a stage is given one (default: the working directory), with a fallback to the manifest dir.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

INDEX_SCHEMA = """
    DROP TABLE IF EXISTS sequence_locations; DROP TABLE IF EXISTS sequence_metadata; DROP TABLE IF EXISTS token_positions;
    CREATE TABLE sequence_locations (sequence_id TEXT NOT NULL, chunk_id INTEGER NOT NULL, offset INTEGER NOT NULL, count INTEGER NOT NULL);
    CREATE TABLE sequence_metadata (sequence_id TEXT PRIMARY KEY, label INTEGER, source TEXT, metadata TEXT);
    CREATE TABLE token_positions (sequence_id TEXT NOT NULL, position INTEGER NOT NULL, chunk_id INTEGER NOT NULL, chunk_offset INTEGER NOT NULL, PRIMARY KEY (sequence_id, position));
    CREATE INDEX idx_sequence_id ON sequence_locations(sequence_id);
    CREATE INDEX idx_chunk_id ON sequence_locations(chunk_id);
"""


# ---- tokens dataset -------------------------------------------------------------------------------------------------

def create_index_db(dataset_dir: str | Path) -> sqlite3.Connection:
    """Create (or reset) ``<dataset>/index.sqlite`` with the chunked-dataset schema and return the open connection."""
    conn = sqlite3.connect(str(Path(dataset_dir) / "index.sqlite"))
    conn.executescript(INDEX_SCHEMA)
    return conn


def read_locations(dataset_dir: str | Path) -> list[tuple[int, int, int]]:
    """``[(chunk_id, offset, count), ...]`` for every document, in row order."""
    conn = sqlite3.connect(str(Path(dataset_dir) / "index.sqlite"))
    rows = [(int(c), int(o), int(n)) for c, o, n in conn.execute(
        "SELECT chunk_id, offset, count FROM sequence_locations ORDER BY rowid")]
    conn.close()
    return rows


def read_counts(dataset_dir: str | Path) -> np.ndarray:
    """Token count per document, in row order (int64)."""
    conn = sqlite3.connect(str(Path(dataset_dir) / "index.sqlite"))
    counts = np.array([n for (n,) in conn.execute("SELECT count FROM sequence_locations ORDER BY rowid")], dtype=np.int64)
    conn.close()
    return counts


def tokens_site(dataset_dir: str | Path) -> str:
    """Name of the site directory under ``<dataset>/layers`` holding the token chunks (``tokens`` for tokens-only data)."""
    return sorted(p.name for p in (Path(dataset_dir) / "layers").iterdir() if p.is_dir())[0]


def chunk_path(dataset_dir: str | Path, chunk_id: int, site: str | None = None) -> Path:
    return Path(dataset_dir) / "layers" / (site or tokens_site(dataset_dir)) / f"chunk_{int(chunk_id):06d}.safetensors"


def open_chunk(dataset_dir: str | Path, chunk_id: int, site: str | None = None, framework: str = "pt"):
    """``safe_open`` handle of one chunk file; ``get_tensor("token_ids")`` gives the concatenated documents."""
    return safe_open(str(chunk_path(dataset_dir, chunk_id, site)), framework=framework)


# ---- manifest dir ---------------------------------------------------------------------------------------------------

def load_meta(manifest_dir: str | Path) -> dict:
    return json.loads((Path(manifest_dir) / "meta.json").read_text())


def resolve_path(p: str | Path, root: str | Path | None) -> Path:
    """Paths recorded in meta.json / spec.json are taken as-is when absolute, else relative to ``root`` (default cwd)."""
    p = Path(p)
    if p.is_absolute() or root is None:
        return p
    return Path(root) / p


def dataset_of(manifest_dir: str | Path, override: str | Path | None = None, root: str | Path | None = None) -> Path:
    """Tokens dataset a manifest was sampled from: ``override`` when given, else ``meta.json["dataset"]`` as recorded
    (relative to ``root`` when given; otherwise, when it does not exist as written, relative to the manifest dir)."""
    if override is not None:
        return Path(override)
    recorded = load_meta(manifest_dir)["dataset"]
    p = resolve_path(recorded, root)
    if root is None and not p.exists() and (Path(manifest_dir) / recorded).exists():
        p = Path(manifest_dir) / recorded
    return p


def load_manifest(manifest_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """``(row_index, position)`` of ``manifest.npz``."""
    with np.load(Path(manifest_dir) / "manifest.npz") as m:
        return m["row_index"], m["position"]


def load_manifest_with_dataset(manifest_dir: str | Path, dataset: str | Path | None = None) -> tuple[np.ndarray, np.ndarray, Path]:
    """``(row_index, position, tokens dataset dir)``; the dataset is ``dataset`` when given, else :func:`dataset_of`."""
    row, pos = load_manifest(manifest_dir)
    ds = dataset_of(manifest_dir, dataset)
    assert ds.exists(), f"tokens dataset not found: {ds}"
    return row, pos, ds


def load_kept(manifest_dir: str | Path) -> np.ndarray:
    """Manifest indices surviving the leader dedup (``dedup_kept.npz``)."""
    return np.load(Path(manifest_dir) / "dedup_kept.npz")["kept"]


def load_drop(manifest_dir: str | Path) -> np.ndarray:
    """Kept-row indices excluded by the exact short-context dedup (``dedup_drop.npz``); empty when the file is absent."""
    p = Path(manifest_dir) / "dedup_drop.npz"
    return np.load(p)["drop"] if p.exists() else np.zeros(0, dtype=np.int64)


def dataset_segments(meta: dict) -> list[tuple[int, str]]:
    """``[(row_offset, dataset path)]`` of a world: one segment for a single-dataset world, one per source otherwise."""
    if meta.get("dataset"):
        return [(0, meta["dataset"])]
    segs = sorted({(int(s["dataset_row_offset"]), s["dataset"]) for s in meta["sources"].values()})
    return [(o, p) for o, p in segs]


# ---- token lookup ---------------------------------------------------------------------------------------------------

class TokensReader:
    """Token lookup by document row over one tokens dataset, or several laid end to end (``[(row_offset, path)]``),
    with a small open-chunk cache. Returns ``torch`` int32 tensors."""

    def __init__(self, dataset: str | Path | list[tuple[int, str | Path]]):
        segments = [(0, dataset)] if isinstance(dataset, (str, Path)) else list(dataset)
        self._segs: list[tuple[int, Path, list[tuple[int, int, int]]]] = []
        for off, p in sorted(segments, key=lambda t: t[0]):
            ds = Path(p)
            self._segs.append((int(off), ds / "layers" / tokens_site(ds), read_locations(ds)))
        self._chunks: dict[tuple[str, int], object] = {}

    def _locate(self, row: int) -> tuple[Path, int, int, int]:
        off, site_dir, loc = max((s for s in self._segs if s[0] <= row), key=lambda s: s[0])
        cid, o, n_ = loc[row - off]
        return site_dir, cid, o, n_

    def _chunk(self, site_dir: Path, cid: int):
        key = (str(site_dir), cid)
        if key not in self._chunks:
            if len(self._chunks) > 64:
                self._chunks.clear()
            self._chunks[key] = safe_open(str(site_dir / f"chunk_{cid:06d}.safetensors"), framework="pt")
        return self._chunks[key]

    def doc(self, row: int) -> torch.Tensor:
        """All token ids of document ``row``."""
        site_dir, cid, o, n_ = self._locate(row)
        return self._chunk(site_dir, cid).get_tensor("token_ids")[o: o + n_]

    def doc_tokens(self, row: int, upto: int) -> torch.Tensor:
        """The first ``upto`` tokens of document ``row`` (fewer when the document is shorter)."""
        site_dir, cid, o, n_ = self._locate(row)
        return self._chunk(site_dir, cid).get_tensor("token_ids")[o: o + min(upto, n_)]

    def prefix_tokens(self, row: int, pos: int) -> torch.Tensor:
        """Tokens ``0..pos`` of document ``row`` (the prompt whose target is ``pos``)."""
        return self.doc_tokens(row, pos + 1)


class World:
    """Manifest + tokens dataset of a world: kept rows, (row, pos) per manifest index, token lookup."""

    def __init__(self, manifest_dir: str | Path, dataset: str | Path | None = None, dataset_root: str | Path | None = None,
                 segments: list[tuple[int, str]] | None = None):
        md = Path(manifest_dir)
        self.dir = md
        self.kept = load_kept(md)
        self.man_row, self.man_pos = load_manifest(md)
        self.meta = load_meta(md) if (md / "meta.json").exists() else {}
        if segments is None:
            segments = [(0, dataset)] if dataset else dataset_segments(self.meta)
        self.segments = [(int(o), str(resolve_path(p, dataset_root))) for o, p in segments]
        self.reader = TokensReader(self.segments)

    def toks(self, ki: int) -> torch.Tensor:
        """Tokens of manifest entry ``ki`` up to and including its target position (the prompt's causal prefix)."""
        return self.reader.prefix_tokens(int(self.man_row[ki]), int(self.man_pos[ki]))

    def toks_kept(self, p: int) -> torch.Tensor:
        """Tokens of kept row ``p``."""
        return self.toks(int(self.kept[p]))

    def live_mask(self) -> np.ndarray:
        """Kept rows minus the ``dedup_drop.npz`` exclusions."""
        live = np.ones(len(self.kept), bool)
        live[load_drop(self.dir)] = False
        return live


def n_live_prompts(world_dir: str | Path) -> int:
    """Number of evaluated prompts of a world: kept rows minus ``dedup_drop.npz`` exclusions (the coverage denominator)."""
    return len(load_kept(world_dir)) - len(load_drop(world_dir))


# ---- world sources and the candidate dictionary ---------------------------------------------------------------------

def source_names(meta: dict) -> dict[int, str]:
    """``{source id: corpus name}`` of a world; a single-corpus world without ``sources`` is ``{0: "all"}``."""
    sources = meta.get("sources")
    if isinstance(sources, dict) and all("source_id" in v for v in sources.values()):
        return {int(v["source_id"]): k for k, v in sources.items()}
    return {0: "all"}


def prompt_sources(world_dir: str | Path, kept: np.ndarray) -> np.ndarray:
    """Corpus id of every kept row (zeros when the manifest carries no ``source``)."""
    with np.load(Path(world_dir) / "manifest.npz") as m:
        src = m["source"].astype(np.int64) if "source" in m.files else np.zeros(len(m["row_index"]), np.int64)
    return src[kept]


def load_candidates(path: str | Path) -> list[dict]:
    """The community dictionary (``candidates.json.gz``): a list of records with ``members`` (global neuron ids)."""
    with gzip.open(path, "rt") as f:
        return json.load(f)


def approach_of(found_tag: str) -> tuple[str, str, float]:
    """``'<lo>-<hi>@cpm<gamma>'`` -> ``('L<lo>-L<hi>', method, parameter)``."""
    win, tag = found_tag.split("@")
    lo, hi = win.split("-")
    method, param = ("cpm", float(tag[3:])) if tag.startswith("cpm") else ("z", float(tag[1:]))
    return f"L{lo}-L{hi}", method, param
