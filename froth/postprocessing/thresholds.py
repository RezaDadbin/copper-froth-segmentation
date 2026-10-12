"""Bound the historical dynamic threshold for numeric probability maps."""

import numpy as np


def dynamic_threshold(probability, k_factor):
    if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
        raise ValueError("Postprocessing input must contain finite probabilities in [0, 1]")
    if np.ptp(probability) <= 1e-7:
        return 0.5
    # A negative threshold would turn zero-valued background into foreground.
    return float(np.clip(probability.mean() - k_factor * probability.std(), 0, np.nextafter(1.0, 0.0)))
