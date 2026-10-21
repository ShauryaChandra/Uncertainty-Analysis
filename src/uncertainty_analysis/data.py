"""Input data loading and example profile generation."""
from pathlib import Path
import numpy as np
import pandas as pd

def load_wind_power(path: str | Path) -> np.ndarray:
    """Load a CSV containing a numeric `wind_power` column."""
    frame = pd.read_csv(path)
    if "wind_power" not in frame.columns:
        raise ValueError("Input CSV must contain a 'wind_power' column.")
    values = pd.to_numeric(frame["wind_power"], errors="coerce").dropna().to_numpy(dtype=float)
    if values.size < 2:
        raise ValueError("At least two valid wind_power observations are required.")
    if np.any(values < 0):
        raise ValueError("Wind power values must be non-negative.")
    return values
