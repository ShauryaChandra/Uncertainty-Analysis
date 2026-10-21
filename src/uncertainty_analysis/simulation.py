"""Monte Carlo uncertainty simulation."""
import numpy as np

def monte_carlo(base_profile: np.ndarray, n_scenarios: int = 1000,
                uncertainty_fraction: float = 0.15,
                random_state: int | None = 42) -> np.ndarray:
    """Generate scenarios with independent Gaussian noise at each time step.

    Standard deviation at each time step is uncertainty_fraction * base power.
    Negative simulated power is clipped to zero.
    Returns an array shaped (n_scenarios, n_time_steps).
    """
    base = np.asarray(base_profile, dtype=float).reshape(-1)
    if base.size == 0 or np.any(base < 0):
        raise ValueError("base_profile must be a non-empty, non-negative array.")
    if n_scenarios < 1 or uncertainty_fraction < 0:
        raise ValueError("n_scenarios must be positive and uncertainty_fraction non-negative.")
    rng = np.random.default_rng(random_state)
    noise = rng.normal(0.0, uncertainty_fraction * base, size=(n_scenarios, base.size))
    return np.clip(base + noise, 0.0, None)
