"""Synthetic data for SOFTWARE tests only. Never used as evidence of model performance."""
import numpy as np
import pandas as pd


def synthetic_df(spec, n=300, seed=0):
    rng = np.random.default_rng(seed)
    data = {}
    for name, (lo, hi, *_rest) in spec["inputs"].items():
        data[name] = rng.uniform(lo, hi, n)
    for c in spec["categorical"]:
        lo, hi = spec["inputs"][c][:2]
        data[c] = rng.integers(int(lo), int(hi) + 1, n).astype(float)
    first = spec["numeric"][1]
    y = (data[first] + rng.normal(0, 0.3 * (spec["inputs"][first][1] - spec["inputs"][first][0]), n)
         > np.median(data[first])).astype(int)
    df = pd.DataFrame(data)
    df[spec["target"]] = y
    return df[spec["columns"]]
