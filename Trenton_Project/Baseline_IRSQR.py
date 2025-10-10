import numpy as np
import matplotlib.pyplot as plt
from pybaselines.utils import relative_difference
from pybaselines import _weighting, Baseline
import pandas as pd
import ast


def baseline_irsqr(
    data,
    lam=1e6,
    quantile=0.05,
    num_knots=100,
    spline_degree=3,
    diff_order=3,
    max_iterations=100,
    tolerance=1e-6,
    weights=None,
    eps=None,
    x_axis=None,
):
    """
    Iterative Reweighted Spline Quantile Regression (IRSQR).

    Fits the baseline using quantile regression with penalized splines.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points. Must not
        contain missing data (NaN) or Inf.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e6.
    quantile : float, optional
        The quantile at which to fit the baseline. Default is 0.05.
    num_knots : int, optional
        The number of knots for the spline. Default is 100.
    spline_degree : int, optional
        The degree of the spline. Default is 3, which is a cubic spline.
    diff_order : int, optional
        The order of the differential matrix. Must be greater than 0. Default is 3
        (third order differential matrix). Typical values are 3, 2, or 1.
    max_iterations : int, optional
        The max number of fit iterations. Default is 100.
    tolerance : float, optional
        The exit criteria. Default is 1e-6.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then the initial weights
        will be an array with size equal to N and all values set to 1.
    eps : float, optional
        A small value added to the square of the residual to prevent dividing by 0.
        Default is None, which uses the square of the maximum-absolute-value of the
        fit each iteration multiplied by 1e-6.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'weights': numpy.ndarray, shape (N,)
            The weight array used for fitting the data.
        * 'tol_history': numpy.ndarray
            An array containing the calculated tolerance values for
            each iteration. The length of the array is the number of iterations
            completed. If the last value in the array is greater than the input
            `tol` value, then the function did not converge.

    Raises
    ------
    ValueError
        Raised if quantile is not between 0 and 1.

    References
    ----------
    Han, Q., et al. Iterative Reweighted Quantile Regression Using Augmented Lagrangian
    Optimization for Baseline Correction. 2018 5th International Conference on 
    Information Science and Control Engineering (ICISCE), 2018, 280-284.

    """

    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")

    if x_axis is None:
        x_axis = np.arange(len(data))
    baseline_obj = Baseline(x_axis)
    # print('DEBUG: data type:', type(data), 'shape:', getattr(data, 'shape', None), 
    # 'len:', len(data) if hasattr(data, '__len__') else None)
    # print('DEBUG: weights:', weights, 'type:', type(weights))
    y, weight_array, pspline = baseline_obj._setup_spline(
        data, weights, spline_degree, num_knots, True, diff_order, lam
    )
    old_coef = np.zeros(baseline_obj._spline_basis._num_bases)
    tol_history = np.empty(max_iterations + 1)
    for i in range(max_iterations + 1):
        baseline = pspline.solve_pspline(y, weight_array)
        calc_difference = relative_difference(old_coef, pspline.coef)
        tol_history[i] = calc_difference
        if calc_difference < tolerance:
            break
        old_coef = pspline.coef
        weight_array = _weighting._quantile(y, baseline, quantile, eps)

    parameters = {"weights": weight_array, "tol_history": tol_history[: i + 1]}
    return baseline, parameters
