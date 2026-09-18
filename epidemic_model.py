"""Read the original population and mobility inputs without rescaling them.

The active scientific equations and constants are in stratified_model.py.
"""

import csv
from pathlib import Path

import numpy as np

from stratified_model import N_LOCATIONS


def load_population_and_mobility(directory):
    directory = Path(directory)
    with (directory / "geodata_2019.csv").open() as stream:
        states = list(csv.DictReader(stream))
    labels = np.array([row["USPS"] for row in states])
    subpop = np.array([row["subpop"] for row in states])
    populations = np.array([int(row["population"]) for row in states], dtype=np.int64)
    index = {code: i for i, code in enumerate(subpop)}
    mobility = np.zeros((len(populations), len(populations)), dtype=np.int64)
    with (directory / "mobility_2011-2015_statelevel.csv").open() as stream:
        for row in csv.DictReader(stream):
            mobility[index[row["ori"]], index[row["dest"]]] += int(row["amount"])
    np.fill_diagonal(mobility, 0)
    if len(populations) != N_LOCATIONS:
        raise ValueError(f"Expected {N_LOCATIONS} locations, found {len(populations)}")
    return labels, subpop, populations, mobility
