"""Plotting helpers for the uncertainty-analysis workflow."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_scenarios(base, scenarios, title, output_path):
    base = np.asarray(base)
    scenarios = np.asarray(scenarios)
    fig, ax = plt.subplots(figsize=(10, 5))
    for row in scenarios[:min(100, len(scenarios))]:
        ax.plot(np.arange(base.size), row, alpha=0.12, linewidth=0.8)
    mean = scenarios.mean(axis=0)
    lower, upper = np.percentile(scenarios, [5, 95], axis=0)
    ax.plot(base, label="Original profile", color="black", linewidth=2)
    ax.plot(mean, label="Scenario mean", color="red", linestyle="--")
    ax.fill_between(np.arange(base.size), lower, upper, alpha=0.2, label="5–95% interval")
    ax.set(title=title, xlabel="Time step", ylabel="Wind power")
    ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(output_path, dpi=160); plt.close(fig)

def plot_pdf(x, pdf, values, title, output_path, label="Fitted PDF"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.asarray(values).ravel(), bins=35, density=True, alpha=0.45, label="Simulated data")
    ax.plot(x, pdf, linewidth=2, label=label)
    ax.set(title=title, xlabel="Wind power", ylabel="Probability density")
    ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(output_path, dpi=160); plt.close(fig)

def plot_cdf(x, cdf, values, title, output_path, label="Fitted CDF"):
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = np.sort(np.asarray(values).ravel())
    empirical = np.arange(1, len(vals) + 1) / len(vals)
    ax.plot(vals, empirical, label="Empirical CDF")
    ax.plot(x, cdf, linestyle="--", linewidth=2, label=label)
    ax.set(title=title, xlabel="Wind power", ylabel="Cumulative probability")
    ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(output_path, dpi=160); plt.close(fig)

def plot_centroids(centroids, output_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, row in enumerate(centroids):
        ax.plot(row, marker="o", markersize=2, label=f"Cluster {i + 1}")
    ax.set(title="K-means Scenario Reduction", xlabel="Time step", ylabel="Wind power")
    ax.grid(True, alpha=0.3); ax.legend(ncol=2); fig.tight_layout()
    fig.savefig(output_path, dpi=160); plt.close(fig)

def plot_pca(projected, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(projected[:, 0], projected[:, 1], alpha=0.45, s=12)
    ax.set(title="PCA Projection", xlabel="Principal Component 1", ylabel="Principal Component 2")
    ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(output_path, dpi=160); plt.close(fig)
