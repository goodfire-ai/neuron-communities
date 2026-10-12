"""Neuron communities: small groups of MLP neurons that are jointly necessary and sufficient for recurring behaviours.

Pipeline (one subcommand per stage, ``python -m neuron_communities <stage> --help``):

  data     tokens -> manifest -> heldout      corpus tokens, prompt sample with dedup, held-out cross-dedup
  harvest  attribution -> world               top-k activation x gradient neuron sets per prompt and layer, CSR worlds
  graph    census -> leiden -> merge -> hubs  cross-layer co-occurrence statistics, CPM Leiden communities, hub removal
  eval     assign -> necessity -> subspan -> span-local -> stats
                                              which prompts each community explains, and how well
"""

__version__ = "0.1.0"
