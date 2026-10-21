"""Run the complete renewable-energy uncertainty-analysis pipeline."""
import json
import numpy as np
import pandas as pd

from .config import (
    DATA_DIR, OUTPUT_DIR, FIGURE_DIR, RANDOM_SEED, N_SCENARIOS,
    UNCERTAINTY_FRACTION, N_CLUSTERS, N_PCA_COMPONENTS,
)
from .data import load_wind_power
from .simulation import monte_carlo
from .sampling import latin_hypercube
from .distributions import fit_normal_pdf_cdf, fit_weibull, fit_beta_normalized
from .reduction import reduce_scenarios, pca_projection
from .visualization import (
    plot_scenarios, plot_pdf, plot_cdf, plot_centroids, plot_pca,
)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    base = load_wind_power(DATA_DIR / "wind_power.csv")

    mc = monte_carlo(base, N_SCENARIOS, UNCERTAINTY_FRACTION, RANDOM_SEED)
    lhs = latin_hypercube(base, N_SCENARIOS, UNCERTAINTY_FRACTION, RANDOM_SEED)

    plot_scenarios(base, mc, "Monte Carlo Simulation with 15% Uncertainty",
                   FIGURE_DIR / "monte_carlo.png")
    plot_scenarios(base, lhs, "Latin Hypercube Sampling with 15% Uncertainty",
                   FIGURE_DIR / "latin_hypercube.png")

    # Distribution fitting is applied to all simulated values, matching the report's workflow.
    flat = mc.ravel()
    x, pdf, cdf, mu, sigma = fit_normal_pdf_cdf(flat)
    plot_pdf(x, pdf, flat, "Probability Density Function (Normal Fit)",
             FIGURE_DIR / "normal_pdf.png")
    plot_cdf(x, cdf, flat, "Cumulative Distribution Function (Normal Fit)",
             FIGURE_DIR / "normal_cdf.png")

    wx, wpdf, wcdf, shape, scale = fit_weibull(flat)
    plot_pdf(wx, wpdf, flat, "Weibull Distribution PDF",
             FIGURE_DIR / "weibull_pdf.png", "Fitted Weibull PDF")
    plot_cdf(wx, wcdf, flat, "Weibull Distribution CDF",
             FIGURE_DIR / "weibull_cdf.png", "Fitted Weibull CDF")

    bx, bpdf, bcdf, alpha, beta, min_value, max_value = fit_beta_normalized(flat)
    normalized = np.clip((flat - min_value) / (max_value - min_value), 1e-6, 1 - 1e-6)
    plot_pdf(bx, bpdf, normalized, "Beta Distribution PDF (Normalized Data)",
             FIGURE_DIR / "beta_pdf.png", "Fitted Beta PDF")
    plot_cdf(bx, bcdf, normalized, "Beta Distribution CDF (Normalized Data)",
             FIGURE_DIR / "beta_cdf.png", "Fitted Beta CDF")

    centroids = reduce_scenarios(mc, N_CLUSTERS, RANDOM_SEED)
    plot_centroids(centroids, FIGURE_DIR / "kmeans_centroids.png")
    projected, explained, _ = pca_projection(mc, N_PCA_COMPONENTS)
    plot_pca(projected, FIGURE_DIR / "pca_projection.png")

    pd.DataFrame(mc).to_csv(OUTPUT_DIR / "monte_carlo_scenarios.csv", index=False)
    pd.DataFrame(lhs).to_csv(OUTPUT_DIR / "lhs_scenarios.csv", index=False)
    pd.DataFrame(centroids).to_csv(OUTPUT_DIR / "kmeans_centroids.csv", index=False)

    summary = {
        "input_time_steps": int(base.size),
        "n_scenarios": int(N_SCENARIOS),
        "uncertainty_fraction": UNCERTAINTY_FRACTION,
        "normal_fit": {"mean": float(mu), "std_dev": float(sigma)},
        "weibull_fit": {"shape": float(shape), "scale": float(scale)},
        "beta_fit_normalized": {"alpha": float(alpha), "beta": float(beta)},
        "pca_explained_variance_ratio": [float(v) for v in explained],
        "pca_total_explained_variance": float(np.sum(explained)),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Plots saved to: {FIGURE_DIR}")
    print(f"CSV outputs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
