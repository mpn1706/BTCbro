"""Slice-1 shared config (single source of truth for seed/params)."""

SEED = 42

FLAT_THRESHOLD_DEFAULT = 0.005
FLAT_THRESHOLD_CHOICES = frozenset({0.003, 0.005, 0.01})

EMBARGO_DEFAULT = 1
EMBARGO_CHOICES = frozenset({1, 2})

PAIR = "XXBTZEUR"

TRAIN_END = "2022-12-31"
VAL_END = "2023-12-31"
