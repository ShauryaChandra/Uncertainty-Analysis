"""PDF/CDF estimation and probability distribution fitting."""
import numpy as np
from scipy import stats

def fit_normal_pdf_cdf(values: np.ndarray, points: int = 400):
    """Fit a normal distribution and return x, PDF, CDF, mean, standard deviation."""
    xdata = np.asarray(values, dtype=float).ravel()
    xdata = xdata[np.isfinite(xdata)]
    if xdata.size < 2:
        raise ValueError("At least two finite observations are required.")
    mu, sigma = stats.norm.fit(xdata)
    if sigma <= 0:
        raise ValueError("Cannot fit a normal distribution to constant data.")
    x = np.linspace(float(xdata.min()), float(xdata.max()), points)
    return x, stats.norm.pdf(x, mu, sigma), stats.norm.cdf(x, mu, sigma), mu, sigma

def fit_weibull(values: np.ndarray, points: int = 400):
    """Fit a two-parameter Weibull distribution to non-negative observations.

    Location is fixed at zero, which is a common assumption for non-negative power.
    Returns x, PDF, CDF, shape, scale.
    """
    data = np.asarray(values, dtype=float).ravel()
    data = data[np.isfinite(data)]
    data = data[data >= 0]
    if data.size < 2 or np.allclose(data, data[0]):
        raise ValueError("Need at least two non-constant non-negative observations.")
    shape, loc, scale = stats.weibull_min.fit(data, floc=0)
    x = np.linspace(0, max(float(data.max()), float(scale) * 1.1), points)
    return x, stats.weibull_min.pdf(x, shape, loc=0, scale=scale), stats.weibull_min.cdf(x, shape, loc=0, scale=scale), shape, scale

def fit_beta_normalized(values: np.ndarray, points: int = 400, eps: float = 1e-6):
    """Fit a Beta distribution after min-max scaling observations to (0, 1)."""
    data = np.asarray(values, dtype=float).ravel()
    data = data[np.isfinite(data)]
    if data.size < 3:
        raise ValueError("At least three finite observations are required.")
    minimum, maximum = float(data.min()), float(data.max())
    if np.isclose(minimum, maximum):
        raise ValueError("Cannot normalize constant data for a Beta fit.")
    normalized = (data - minimum) / (maximum - minimum)
    normalized = np.clip(normalized, eps, 1 - eps)
    alpha, beta, loc, scale = stats.beta.fit(normalized, floc=0, fscale=1)
    x = np.linspace(0, 1, points)
    return x, stats.beta.pdf(x, alpha, beta), stats.beta.cdf(x, alpha, beta), alpha, beta, minimum, maximum
