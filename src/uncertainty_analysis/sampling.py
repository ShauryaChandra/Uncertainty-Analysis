"""Latin Hypercube Sampling for uncertain power profiles."""
import numpy as np
from scipy.stats import norm, qmc

def latin_hypercube(base_profile: np.ndarray, n_scenarios: int = 1000,
                    uncertainty_fraction: float = 0.15,
                    random_state: int | None = 42) -> np.ndarray:
    """Sample each time-step marginal using Latin Hypercube Sampling.

    Each time step is treated as an independent normal marginal around its base
    power value. Negative power is clipped to zero.
    """
    base = np.asarray(base_profile, dtype=float).reshape(-1)
    if base.size == 0 or np.any(base < 0):
        raise ValueError("base_profile must be a non-empty, non-negative array.")
    if n_scenarios < 1 or uncertainty_fraction < 0:
        raise ValueError("n_scenarios must be positive and uncertainty_fraction non-negative.")
    sampler = qmc.LatinHypercube(d=base.size, seed=random_state)
    uniform_samples = sampler.random(n=n_scenarios)
    # Avoid infinite inverse-CDF values if an implementation returns endpoints.
    uniform_samples = np.clip(uniform_samples, 1e-10, 1 - 1e-10)
    z = norm.ppf(uniform_samples)
    scenarios = base + z * (uncertainty_fraction * base)
    return np.clip(scenarios, 0.0, None)
