# Uncertainty Analysis in Renewable Energy Power Systems

A data-driven study of uncertainty in renewable energy generation using probabilistic modelling, Monte Carlo simulation, sampling techniques, and scenario reduction.

## Overview

Renewable energy sources such as wind and solar power are inherently variable because their output depends on weather and environmental conditions. This variability introduces uncertainty into power systems, potentially contributing to power imbalances, grid instability, and inefficient system planning.

This project explores probabilistic methods for modelling renewable energy uncertainty, generating possible power-output scenarios, analysing their distributions, and reducing large scenario sets while retaining useful information.

The work was developed as part of a **Project-Based Learning (PBL) project during the 4th semester**, with an emphasis on uncertainty analysis in power systems.

## Objectives

- Model uncertainty in wind power generation using Monte Carlo simulation.
- Estimate probability density functions (PDFs) and cumulative distribution functions (CDFs).
- Fit statistical distributions to simulated power data.
- Apply Latin Hypercube Sampling (LHS) for improved sample-space coverage.
- Reduce large numbers of scenarios using K-Means clustering.
- Explore dimensionality reduction using Principal Component Analysis (PCA).

## Methodology

### 1. Wind Power Uncertainty Modelling

Monte Carlo simulation generates multiple possible wind-power profiles by introducing random variation into a baseline power profile.

The simulated power is modelled as:

$$
P_{\mathrm{sim}} = \max(P_{\mathrm{base}} + \epsilon, 0)
$$

where:

- $P_{\mathrm{sim}}$ is the simulated power output.
- $P_{\mathrm{base}}$ is the baseline power profile.
- $\epsilon$ is random noise drawn from a normal distribution.

The noise standard deviation is proportional to the baseline power:

$$
\sigma = \alpha P_{\mathrm{base}}
$$

where $\alpha = 0.15$, representing a 15% uncertainty level in the current implementation.

Repeated simulations generate a collection of possible power-output scenarios, allowing the variability and range of potential outcomes to be studied.

### 2. Probability Density and Cumulative Distribution Functions

The simulated power values are analysed using two probability functions.

**Probability Density Function (PDF)**

The PDF describes the distribution of simulated values and identifies the regions where values are more concentrated.

**Cumulative Distribution Function (CDF)**

The CDF gives the probability that a random variable is less than or equal to a specified value:

$$
F(x) = P(X \leq x)
$$

Together, the PDF and CDF help characterise the distribution of power output and understand the probability of different outcomes.

### 3. Statistical Distribution Fitting

The project explores three statistical distribution models:

- **Normal distribution:** Used to characterise the overall distribution of simulated power values through its mean and standard deviation.
- **Weibull distribution:** Commonly used in wind-energy analysis to model wind-related variability through shape and scale parameters.
- **Beta distribution:** Suitable for bounded variables, such as normalized renewable-energy output or proportions, through its shape parameters.

For Beta fitting, the power data is normalized to a bounded interval before estimating the distribution parameters.

### 4. Latin Hypercube Sampling (LHS)

Latin Hypercube Sampling is used to improve the coverage of the sampling space.

Instead of relying solely on unrestricted random draws, LHS divides the sampling distribution into equal-probability intervals and selects samples across those intervals.

This approach can reduce sample clustering and provide better coverage of the input space with a limited number of samples.

### 5. Scenario Reduction Using K-Means

Monte Carlo simulation can generate a large number of scenarios, making subsequent power-system analysis computationally expensive.

K-Means clustering groups similar scenarios into clusters. Each cluster is represented by its centroid, which serves as a representative profile.

The process involves:

1. Selecting the number of clusters.
2. Assigning scenarios to their nearest centroids.
3. Updating centroids based on cluster membership.
4. Repeating until the clusters stabilize.

Scenario reduction aims to decrease computational complexity while retaining important patterns in the generated profiles.

### 6. Principal Component Analysis (PCA)

PCA is used to transform high-dimensional scenario data into a smaller set of principal components.

The principal components capture directions of variation in the data, allowing the scenario set to be examined in a lower-dimensional space.

A two-component projection provides a visual representation of scenario variation, while the explained variance ratio indicates how much of the total variation is retained by those components.

## Workflow

The overall workflow is:

1. Load the baseline wind-power profile.
2. Generate uncertain power scenarios using Monte Carlo simulation.
3. Estimate probability distributions and fit statistical models.
4. Apply Latin Hypercube Sampling.
5. Reduce scenarios using K-Means clustering.
6. Apply PCA to examine lower-dimensional representations.
7. Export numerical results and visualizations.

## Technology Stack

- **Python 3.12** — Programming language
- **NumPy** — Numerical computation and array operations
- **Pandas** — Data loading and manipulation
- **SciPy** — Statistical distributions and fitting
- **Matplotlib** — Data visualization
- **Scikit-learn** — K-Means clustering and PCA

## Project Structure

```text
uncertainty_analysis_github/
├── data/
│   └── wind_power.csv
├── outputs/
│   ├── figures/
│   └── ...
├── src/
│   └── uncertainty_analysis/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── simulation.py
│       ├── distributions.py
│       ├── sampling.py
│       ├── reduction.py
│       ├── visualization.py
│       └── main.py
├── .gitignore
├── .python-version
├── requirements.txt
└── README.md
```

## Installation and Usage

### Prerequisites

- Python 3.12
- Git (optional, for cloning the repository)

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Renewable-Energy-Uncertainty-Analysis.git
cd Renewable-Energy-Uncertainty-Analysis
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Run the analysis

```bash
python -m src.uncertainty_analysis.main
```

The program prints a summary of the fitted distributions and PCA results. Generated figures are saved in `outputs/figures/`, and tabular outputs are saved in `outputs/`.

## Current Implementation Results

The current Python implementation completed successfully with the following configuration and results:

| Metric | Result |
|---|---:|
| Input time steps | 24 |
| Monte Carlo scenarios | 1,000 |
| Uncertainty fraction | 15% |
| Normal distribution mean | 276.54 |
| Normal distribution standard deviation | 49.81 |
| Weibull shape parameter | 5.94 |
| Weibull scale parameter | 297.38 |
| Beta shape parameter α | 6.30 |
| Beta shape parameter β | 7.81 |
| Variance explained by first two PCA components | 10.48% |

These numerical values are from the current implementation and its illustrative input data. They should not be interpreted as validated measurements from the original report's dataset.

## Limitations

- The current repository uses an illustrative 24-point wind-power profile reconstructed approximately from a graph in the project material; it is not the original experimental dataset.
- Results depend on the input profile, simulation assumptions, uncertainty level, and sampling configuration.
- Distribution fitting describes the supplied data and does not establish that a particular distribution is universally appropriate for renewable-energy generation.
- PCA results depend on the characteristics and preprocessing of the scenario data.
- Further validation using the original dataset would be needed before drawing conclusions about real-world power-system performance.

## Conclusion

This project demonstrates a computational workflow for analysing uncertainty in renewable energy generation. Monte Carlo simulation generates possible power-output scenarios, statistical distributions characterize their variability, Latin Hypercube Sampling improves sampling coverage, K-Means reduces the scenario set, and PCA provides a lower-dimensional representation of the data.

Together, these techniques provide a foundation for studying uncertainty and preparing representative renewable-energy scenarios for further power-system analysis.

**Project:** Uncertainty Analysis Based on Power Systems with Emphasis on Renewable Energy