import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def baseline_gifts(y, lam=1e6, p=0.01, iterations=10):
    """
    Perform baseline correction using Asymmetric Least Squares (ALS) method.

    Parameters:
        y (array): Input spectrum (intensity values).
        lam (float): Smoothness parameter (higher = smoother baseline).
        p (float): Asymmetry parameter (0 < p < 1).
        number_iterations (int): Number of iterations.

    Returns:
        baseline (array): Estimated baseline.
    """
    if lam is None:
        lam = 1e6  # Default smoothness parameter
    if p is None:
        p = 0.01  # Default asymmetry parameter
    if iterations is None:
        iterations = 10  # Default number of iterations

    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.T)  # Precompute the smoothness matrix
    w = np.ones(L)

    for _ in range(iterations):
        W = diags(w, 0)
        Z = W + D
        baseline = spsolve(Z, w * y)
        w = p * (y > baseline) + (1 - p) * (y <= baseline)

    return baseline
