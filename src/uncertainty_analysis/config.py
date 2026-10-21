"""Project configuration and reproducibility settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

RANDOM_SEED = 42
N_SCENARIOS = 1000
UNCERTAINTY_FRACTION = 0.15
N_CLUSTERS = 8
N_PCA_COMPONENTS = 2
