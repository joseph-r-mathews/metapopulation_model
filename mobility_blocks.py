"""The fixed, certified-optimal mobility-only two-block partition."""

import numpy as np


MOBILITY_K2_BLOCK_STATES = (
    (
        "WY", "AK", "ND", "RI", "NH", "NM", "UT", "IA", "OK", "LA",
        "AL", "SC", "WI", "MD", "MO", "IN", "TN", "WA", "VA", "OH",
        "PA", "NY", "FL", "CA",
    ),
    (
        "VT", "DC", "SD", "DE", "MT", "ME", "HI", "ID", "WV", "NE",
        "KS", "NV", "MS", "AR", "CT", "OR", "KY", "MN", "CO", "MA",
        "AZ", "NJ", "MI", "NC", "GA", "IL", "TX",
    ),
)

# Zero-based indices in the ordering of geodata_2019.csv.
MOBILITY_K2_BLOCK_INDICES = (
    (0, 3, 4, 8, 10, 15, 20, 21, 23, 26, 27, 28, 31, 32, 33, 34, 35, 38,
     39, 44, 46, 47, 48, 50),
    (1, 2, 5, 6, 7, 9, 11, 12, 13, 14, 16, 17, 18, 19, 22, 24, 25, 29, 30,
     36, 37, 40, 41, 42, 43, 45, 49),
)


def fixed_mobility_blocks(labels):
    """Return the fixed index blocks after validating the supplied ordering."""
    labels = np.asarray(labels, dtype=str)
    if len(MOBILITY_K2_BLOCK_STATES) != 2:
        raise RuntimeError("The fixed mobility partition must contain two blocks")
    flat_states = sum(MOBILITY_K2_BLOCK_STATES, ())
    flat_indices = sum(MOBILITY_K2_BLOCK_INDICES, ())
    if tuple(map(len, MOBILITY_K2_BLOCK_STATES)) != (24, 27):
        raise RuntimeError("The fixed mobility block sizes must be 24 and 27")
    if len(flat_states) != 51 or len(set(flat_states)) != 51:
        raise RuntimeError("The fixed mobility state partition is not one-to-one")
    if sorted(flat_indices) != list(range(51)):
        raise RuntimeError("The fixed mobility index partition is not one-to-one")
    for states, indices in zip(
        MOBILITY_K2_BLOCK_STATES, MOBILITY_K2_BLOCK_INDICES
    ):
        if tuple(labels[list(indices)]) != states:
            raise ValueError("Location ordering does not match the fixed partition")
    return tuple(np.asarray(block, dtype=int) for block in MOBILITY_K2_BLOCK_INDICES)
