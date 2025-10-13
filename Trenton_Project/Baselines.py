# Created: 10-13-2025
# Author: Trenton Wells
# Organization: NREL
# NREL Contact: trenton.wells@nrel.gov
# Personal Contact: trentonwells73@gmail.com

import numpy as np
import ast
import warnings
import plotly.graph_objs as go
import ipywidgets as widgets
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.signal import convolve
from math import ceil
from scipy.interpolate import CubicSpline
from pybaselines.smooth import arpls
from pybaselines.spline import irsqr
from pybaselines.classification import fabc
    
from pybaselines.utils import (
    relative_difference,
    optimize_window,
    pad_edges,
    ParameterWarning
)
from pybaselines import (
    _weighting,
)
from scipy.ndimage import (
    binary_erosion,
    binary_opening
)
from IPython.display import (
    display,
    clear_output
)

### GIFTS BASELINE ### (Documentation needs to be added)
### Consider replacing with pybaselines.Baseline.arpls (best general AsLS method)
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

### IRSQR BASELINE ###
# from pybaselines.spline
# https://pybaselines.readthedocs.io/en/latest/_modules/pybaselines/spline.html#Baseline.irsqr
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
    Fit the baseline using quantile regression with penalized splines.
    
    Iterative Reweighted Spline Quantile Regression (IRSQR).

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

### FABC BASELINE ###
# from pybaselines.classification
# https://pybaselines.readthedocs.io/en/latest/_modules/pybaselines/classification.html#Baseline.fabc
def _haar(num_points, scale=2):
    """
    Create a Haar wavelet.

    Helps baseline_fabc() identify the baseline and peak regions.

    Parameters
    ----------
    num_points : int
        The number of points for the wavelet. Note that if `num_points` is odd
        and `scale` is even, or if `num_points` is even and `scale` is odd, then
        the length of the output wavelet will be `num_points` + 1 to ensure the
        symmetry of the wavelet.
    scale : int, optional
        The scale at which the wavelet is evaluated. Default is 2.

    Returns
    -------
    wavelet : numpy.ndarray
        The Haar wavelet.

    Notes
    -----
    This implementation is only designed to work for integer scales.

    Matches pywavelets's Haar implementation after applying patches from pywavelets
    issue #365 and pywavelets pull request #580.

    Raises
    ------
    TypeError
        Raised if `scale` is not an integer.

    References
    ----------
    https://wikipedia.org/wiki/Haar_wavelet

    """
    if not isinstance(scale, int):
        raise TypeError("scale must be an integer for the Haar wavelet")
    # to maintain symmetry, even scales should have even windows and odd
    # scales have odd windows
    odd_scale = scale % 2
    odd_window = num_points % 2
    if (odd_scale and not odd_window) or (not odd_scale and odd_window):
        num_points += 1
    # center at 0 rather than 1/2 to make calculation easier
    # from [-scale/2 to 0), wavelet = 1; [0, scale/2), wavelet = -1
    x_vals = np.arange(num_points) - (num_points - 1) / 2
    wavelet = np.zeros(num_points)
    if not odd_scale:
        wavelet[(x_vals >= -scale / 2) & (x_vals < 0)] = 1
        wavelet[(x_vals < scale / 2) & (x_vals >= 0)] = -1
    else:
        # set such that wavelet[x_vals == 0] = 0
        wavelet[(x_vals > -scale / 2) & (x_vals < 0)] = 1
        wavelet[(x_vals < scale / 2) & (x_vals > 0)] = -1

    # the 1/sqrt(scale) is a normalization
    return wavelet / (np.sqrt(scale))


def _iteration_threshold(power, num_std=3.0):
    """
    Iteratively threshold a power spectrum based on the mean and standard deviation.

    Any values greater than the mean of the power spectrum plus a multiple of the
    standard deviation are masked out to create a new power spectrum. The process
    is performed iteratively until no further points are masked out. Helps 
    baseline_fabc() identify baseline regions.

    Parameters
    ----------
    power : numpy.ndarray, shape (N,)
        The power spectrum to threshold.
    num_std : float, optional
        The number of standard deviations to include when thresholding. Default is 3.0.

    Returns
    -------
    mask : numpy.ndarray, shape (N,)
        The boolean mask with True values where any point in the input power spectrum
        was less than

    References
    ----------
    Dietrich, W., et al. Fast and Precise Automatic Baseline Correction of One- and
    Two-Dimensional NMR Spectra. Journal of Magnetic Resonance. 1991, 91, 1-11.

    """
    mask = power < np.mean(power) + num_std * np.std(power, ddof=1)
    old_mask = np.ones_like(mask)
    while not np.array_equal(mask, old_mask):
        old_mask = mask
        masked_power = power[mask]
        if masked_power.size < 2:  # need at least 2 points for std calculation
            warnings.warn(
                'not enough baseline points found; "num_std" is likely too low',
                ParameterWarning,
                stacklevel=2,
            )
            break
        mask = power < np.mean(masked_power) + num_std * np.std(masked_power, ddof=1)

    return mask


def _refine_mask(mask, min_length=2):
    """
    Remove small consecutive True values and lone False values from a boolean mask.

    Helps baseline_fabc() identify baseline regions without small interruptions.

    Parameters
    ----------
    mask : numpy.ndarray
        The boolean array designating baseline points as True and peak points as False.
    min_length : int, optional
        The minimum consecutive length of True values needed for a section to remain True.
        Lengths of True values less than `min_length` are converted to False. Default is
        2, which removes all lone True values.

    Returns
    -------
    numpy.ndarray
        The input mask after removing lone True and False values.

    Notes
    -----
    Removes the lone True values first since True values designate the baseline.
    That way, the approach is more conservative with assigning baseline points.

    Examples
    --------
    >>> mask = np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1])
    >>> _refine_mask(mask, 3).astype(int)
    array([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    >>> _refine_mask(mask, 5).astype(int)
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    """
    min_length = max(min_length, 1)  # has to be at least 1 for the binary opening
    half_length = min_length // 2
    # do not use border_value=1 since that automatically makes the borders True and
    # extends the True section by half_window on each side
    output = binary_opening(
        np.pad(mask, half_length, "constant", constant_values=True),
        np.ones(min_length, bool),
    )[half_length : len(mask) + half_length]

    # convert lone False values to True
    np.logical_or(output, binary_erosion(output, [1, 0, 1]), out=output)
    # TODO should there be an erosion step here, using another parameter (erode_hw)?
    # that way, can control both the minimum length and then remove edges of baselines
    # independently, allowing more control over the output mask
    return output


def _cwt(data, wavelet, widths, dtype=None, **kwargs):
    """
    Perform a continuous wavelet transform.

    Performs a continuous wavelet transform on `data`,
    using the `wavelet` function. A CWT performs a convolution
    with `data` using the `wavelet` function, which is characterized
    by a width parameter and length parameter. The `wavelet` function
    is allowed to be complex. Helps baseline_fabc() identify the baseline regions.

    Parameters
    ----------
    data : array-like, shape (N,)
        Data on which to perform the transform.
    wavelet : Callable
        Wavelet function, which should take 2 arguments.
        The first argument is the number of points that the returned vector
        will have (len(wavelet(length,width)) == length).
        The second is a width parameter, defining the size of the wavelet
        (e.g. standard deviation of a gaussian). See `ricker`, which
        satisfies these requirements.
    widths : Sequence[scalar, ...]
        Widths to use for transform with length `M`.
    dtype : type or numpy.dtype, optional
        The desired data type of output. Defaults to ``float64`` if the
        output of `wavelet` is real and ``complex128`` if it is complex.
    **kwargs
        Keyword arguments passed to wavelet function.

    Returns
    -------
    cwt: numpy.ndarray, shape (M, N)
        Will have shape of (len(widths), len(data)).

    Notes
    -----
    This function was deprecated from scipy.signal in version 1.12.

    References
    ----------
    S. Mallat, "A Wavelet Tour of Signal Processing (3rd Edition)", Academic Press, 
    2009.

    """
    # Determine output type
    if dtype is None:
        if np.asarray(wavelet(1, widths[0], **kwargs)).dtype.char in "FDG":
            dtype = np.complex128
        else:
            dtype = np.float64

    output = np.empty((len(widths), len(data)), dtype=dtype)
    for ind, width in enumerate(widths):
        N = np.min([10 * width, len(data)])
        wavelet_data = np.conj(wavelet(N, width, **kwargs)[::-1])
        output[ind] = convolve(data, wavelet_data, mode="same")
    return output


def baseline_fabc(
    self,
    data,
    lam=1e6,
    scale=None,
    num_std=3.0,
    diff_order=2,
    min_length=2,
    weights=None,
    weights_as_mask=False,
    pad_kwargs=None,
    **kwargs,
):
    """
    Perform a fully automatic baseline correction (fabc).

    Similar to Dietrich's method, except that the derivative is estimated using a
    continuous wavelet transform and the baseline is calculated using Whittaker
    smoothing through the identified baseline points.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e6.
    scale : int, optional
        The scale at which to calculate the continuous wavelet transform. Should be
        approximately equal to the index-based full-width-at-half-maximum of the peaks
        or features in the data. Default is None, which will use half of the value from
        :func:`.optimize_window`, which is not always a good value, but at least scales
        with the number of data points and gives a starting point for tuning the 
        parameter.
    num_std : float, optional
        The number of standard deviations to include when thresholding. Higher values
        will assign more points as baseline. Default is 3.0.
    diff_order : int, optional
        The order of the differential matrix. Must be greater than 0. Default is 2
        (second order differential matrix). Typical values are 2 or 1.
    min_length : int, optional
        Any region of consecutive baseline points less than `min_length` is considered
        to be a false positive and all points in the region are converted to peak 
        points.
        A higher `min_length` ensures fewer points are falsely assigned as baseline 
        points.
        Default is 2, which only removes lone baseline points.
    weights : array-like, shape (N,), optional
        The weighting array, used to override the function's baseline identification
        to designate peak points. Only elements with 0 or False values will have
        an effect; all non-zero values are considered baseline points. If None
        (default), then will be an array with size equal to N and all values set to 1.
    weights_as_mask : bool, optional
        If True, signifies that the input `weights` is the mask to use for fitting,
        which skips the continuous wavelet calculation and just smooths the input data.
        Default is False.
    pad_kwargs : dict, optional
        A dictionary of keyword arguments to pass to :func:`.pad_edges` for padding
        the edges of the data to prevent edge effects from convolution for the
        continuous wavelet transform. Default is None.
    **kwargs

        .. deprecated:: 1.2.0
            Passing additional keyword arguments is deprecated and will be removed in 
            version 1.4.0. Pass keyword arguments using `pad_kwargs`.
    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    parameters : dict
        A dictionary with the following items:

        * 'mask': numpy.ndarray, shape (N,)
            The boolean array designating baseline points as True and peak points
            as False.
        * 'weights': numpy.ndarray, shape (N,)
            The weight array used for fitting the data.

    Notes
    -----
    The classification of baseline points is similar to :meth:`~.Baseline.dietrich`, 
    except that this method approximates the first derivative using a continuous wavelet
    transform with the Haar wavelet, which is more robust than the numerical derivative 
    in Dietrich's method.

    References
    ----------
    Cobas, J., et al. A new general-purpose fully automatic baseline-correction
    procedure for 1D and 2D NMR data. Journal of Magnetic Resonance, 2006, 183(1),
    145-151.

    """
    self._deprecate_pad_kwargs(**kwargs)

    if weights_as_mask:
        y, whittaker_weights, whittaker_system = self._setup_whittaker(
            data, lam, diff_order, weights
        )
        mask = whittaker_weights.astype(bool)
    else:
        y, weight_array = self._setup_classification(data, weights)
        if scale is None:
            # optimize_window(y) / 2 gives an "okay" estimate that at least scales
            # with data size
            scale = ceil(optimize_window(y) / 2)
        half_window = scale * 2
        pad_kwargs = pad_kwargs if pad_kwargs is not None else {}
        wavelet_cwt = _cwt(
            pad_edges(y, half_window, **pad_kwargs, **kwargs), _haar, [scale]
        )
        power = wavelet_cwt[0, half_window:-half_window] ** 2

        mask = _refine_mask(_iteration_threshold(power, num_std), min_length)
        np.logical_and(mask, weight_array, out=mask)

        _, whittaker_weights, whittaker_system = self._setup_whittaker(
            y, lam, diff_order, mask
        )
        if self._sort_order is not None:
            whittaker_weights = whittaker_weights[self._inverted_order]

    whittaker_weights = whittaker_weights.astype(float)
    baseline = whittaker_system.solve(
        whittaker_system.add_diagonal(whittaker_weights),
        whittaker_weights * y,
        overwrite_b=True,
        overwrite_ab=True,
    )
    parameters = {"mask": mask, "weights": whittaker_weights}

    return baseline, parameters

### MANUAL BASELINE ###
def select_anchor_points(
    FTIR_dataframe, material=None, filepath=None, try_it_out=True, dataframe_path=None
):
    """
    Interactively select anchor points for FTIR baseline correction.

    Lets user select anchor points from a spectrum in the DataFrame for baseline 
    correction. Anchor points are selected by clicking on the plot, and will apply to 
    each file of that material. After selection, a cubic spline baseline is fit and 
    previewed, and the user can accept or redo the selection.

    Parameters
    ----------
    FTIR_dataframe : pd.DataFrame
        The DataFrame containing the spectral data.
    material : str, optional
        Material name to analyze (ignored if filepath is provided).
    filepath : str, optional
        If provided, only process this file (by 'File Location' + 'File Name').
    try_it_out : bool, optional
        If True, only prints the anchor points (default). If False, saves anchor points 
        to 'Baseline Parameters' column for all rows with the same material.
    dataframe_path : str, optional
        Path to save the DataFrame as CSV if anchor points are saved (used only if 
        try_it_out is False).

    Returns
    -------
    None
        The selected anchor points are stored in the dataframe under 'Baseline 
        Parameters'.
    """
    SELECTED_ANCHOR_POINTS = []
    clear_output(wait=True)  # Forcibly reset output area before anything else

    # --- Data selection logic ---
    if filepath is not None:
        import os

        if os.path.sep in filepath:
            folder, fname = os.path.split(filepath)
            filtered = FTIR_dataframe[
                (FTIR_dataframe["File Location"] == folder)
                & (FTIR_dataframe["File Name"] == fname)
            ]
        else:
            filtered = FTIR_dataframe[FTIR_dataframe["File Name"] == filepath]
        if filtered.empty:
            raise ValueError(f"No entry found for file '{filepath}'.")
        row = filtered.iloc[0]
    else:
        if material is None:
            raise ValueError("Material must be specified if filepath is not provided.")
        filtered = FTIR_dataframe[
            (FTIR_dataframe["Material"] == material) & (FTIR_dataframe["Time"] == 0)
        ]
        if filtered.empty:
            raise ValueError(
                f"No entry found for material '{material}' with time == 0."
            )
        row = filtered.iloc[0]

    x_data = (
        ast.literal_eval(row["X-Axis"])
        if isinstance(row["X-Axis"], str)
        else row["X-Axis"]
    )
    y_data = (
        ast.literal_eval(row["Raw Data"])
        if isinstance(row["Raw Data"], str)
        else row["Raw Data"]
    )

    # --- Widget and button setup (define early for scope) ---
    accept_button = widgets.Button(description="Continue", button_style="success")
    redo_button = widgets.Button(description="Redo", button_style="warning")
    button_box = widgets.HBox([accept_button, redo_button])
    button_box_out = widgets.Output()
    done = widgets.Output()
    output = widgets.Output()
    anchor_points = []
    anchor_markers = go.Scatter(
        x=[],
        y=[],
        mode="markers",
        marker=dict(color="red", size=10),
        name="Anchor Points",
    )

    # --- Plot setup ---
    fig = go.FigureWidget(
        data=[
            go.Scatter(x=x_data, y=y_data, mode="lines", name="Raw Data"),
            anchor_markers,
        ]
    )
    fig.update_layout(
        title="Click to select anchor points for baseline correction",
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="Absorbance (AU)",
    )

    # --- Click handler for anchor selection ---
    def _on_click(trace, points, selector):
        """
        Handle click events on the plot to select anchor points.

        Helper function for select_anchor_points().
        """
        if points.xs:
            x_val = points.xs[0]
            anchor_points.append(x_val)
            # Add a vertical line for the anchor point
            vline = dict(
                type="line",
                x0=x_val,
                x1=x_val,
                y0=min(y_data),
                y1=max(y_data),
                line=dict(color="red", dash="dash"),
            )
            fig.add_shape(vline)
            # Add a marker at the selected point on the line
            idx = np.argmin(np.abs(np.array(x_data) - x_val))
            y_val = y_data[idx]
            with fig.batch_update():
                anchor_markers.x = list(anchor_markers.x) + [x_val]
                anchor_markers.y = list(anchor_markers.y) + [y_val]
            with output:
                clear_output(wait=True)
                print(f"Anchor point selected: {x_val}")

    scatter = fig.data[0]
    scatter.on_click(_on_click)

    # --- Accept/Redo logic, defined once and reused ---
    def show_baseline_preview():
        """
        Show a preview of the cubic spline baseline and baseline-corrected spectrum 
        using the selected anchor points.

        Helper function for select_anchor_points().
        Lets the user accept or redo the selection after visualizing the correction.
        """
        # Fit cubic spline to anchor points with zero slope at endpoints
        anchor_x = np.array(sorted(anchor_points))
        anchor_y = np.array(
            [y_data[np.argmin(np.abs(np.array(x_data) - x))] for x in anchor_x]
        )
        spline = CubicSpline(anchor_x, anchor_y, bc_type=((1, 0.0), (1, 0.0)))
        x_dense = np.linspace(min(x_data), max(x_data), 1000)
        spline_y = spline(x_dense)
        y_interp = np.interp(x_data, x_dense, spline_y)
        baseline_corrected = np.array(y_data) - y_interp
        from plotly.subplots import make_subplots

        fig2 = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=("Original with Baseline", "Baseline Corrected"),
        )
        fig2.add_trace(
            go.Scatter(x=x_data, y=y_data, mode="lines", name="Raw Data"), row=1, col=1
        )
        fig2.add_trace(
            go.Scatter(
                x=x_dense,
                y=spline_y,
                mode="lines",
                name="Baseline",
                line=dict(color="green"),
            ),
            row=1,
            col=1,
        )
        fig2.add_trace(
            go.Scatter(
                x=anchor_x,
                y=anchor_y,
                mode="markers",
                marker=dict(color="red", size=10),
                name="Anchor Points",
            ),
            row=1,
            col=1,
        )
        fig2.add_trace(
            go.Scatter(
                x=x_data,
                y=baseline_corrected,
                mode="lines",
                name="Baseline Corrected",
                line=dict(color="purple"),
            ),
            row=2,
            col=1,
        )
        fig2.update_layout(
            height=800,
            title_text="Baseline Correction Preview",
            xaxis_title="Wavenumber (cm⁻¹)",
            yaxis_title="Absorbance (AU)",
        )
        # Accept/Redo for preview
        accept2 = widgets.Button(description="Accept", button_style="success")
        redo2 = widgets.Button(description="Redo", button_style="warning")
        button_box2 = widgets.HBox([accept2, redo2])
        button_box2_out = widgets.Output()
        done2 = widgets.Output()

        def accept2_callback(b):
            global SELECTED_ANCHOR_POINTS
            SELECTED_ANCHOR_POINTS = sorted(anchor_points)
            with done2:
                clear_output()
                # If in the "try baselines" section, print the results. If not, save to 
                # dataframe.
                if try_it_out:
                    print(
                        f"DONE--Final selected anchor points: {SELECTED_ANCHOR_POINTS}"
                    )
                else:
                    # Save anchor points to Baseline Parameters for all rows with the 
                    # same material
                    mat = row["Material"]
                    for idx, r in FTIR_dataframe.iterrows():
                        if r["Material"] == mat:
                            FTIR_dataframe.at[idx, "Baseline Parameters"] = str(
                                SELECTED_ANCHOR_POINTS
                            )
                            FTIR_dataframe.at[idx, "Baseline Function"] = "Manual"
                    message = (f"Anchor points saved to Baseline Parameters and "
                        f"Baseline Function set to 'Manual' for material '{mat}'."
                    )
                    print(message)
                    # --- Save DataFrame to CSV if dataframe_path is provided ---
                    if dataframe_path:
                        try:
                            FTIR_dataframe.to_csv(dataframe_path, index=False)
                            print(f"DataFrame saved to {dataframe_path}.")
                        except Exception as e:
                            message = (f"Warning: Could not save DataFrame to "
                                f"{dataframe_path}: {e}"
                            )
                            print(message)
                    else:
                        print(
                            "Warning: dataframe_path not provided. DataFrame not saved."
                        )
            button_box2_out.clear_output()

        def redo2_callback(b):
            with done2:
                clear_output()
            reset_selection()

        accept2.on_click(accept2_callback)
        redo2.on_click(redo2_callback)
        clear_output(wait=True)
        display(fig2, button_box2_out, done2)
        with button_box2_out:
            display(button_box2)

    def reset_selection():
        """
        Clear current selection and reset the plot for new anchor point selection.

        Helper function for select_anchor_points(). Clears anchor points and resets the 
        interactive plot and widgets for a new selection.
        """
        # Clear anchor points and reset plot
        anchor_points.clear()
        fig.layout.shapes = ()
        with fig.batch_update():
            anchor_markers.x = []
            anchor_markers.y = []
        with output:
            clear_output()
            print("Anchor points cleared. Please select again.")
        with done:
            clear_output()
        button_box_out.clear_output()
        display(fig, output, button_box_out, done)
        with button_box_out:
            display(button_box)

    def continue_callback(b):
        """
        Handle click on Continue button.
        
        Helper function for select_anchor_points(). Handles the Continue button for the
        initial anchor selection. If enough points are selected, shows the baseline 
        preview; otherwise, prompts the user to select more points.
        """
        button_box_out.clear_output()
        with done:
            clear_output()
        if len(anchor_points) < 2:
            with done:
                print("Select at least two anchor points for spline.")
            reset_selection()
            return
        show_baseline_preview()

    def redo_callback(b):
        """
        Helper function:
        Handles the Redo button for the initial anchor selection, resetting the 
        selection process.
        """
        reset_selection()

    accept_button.on_click(continue_callback)
    redo_button.on_click(redo_callback)

    # --- Initial display ---
    clear_output(wait=True)  # Ensure output area is reset to avoid UID errors
    display(fig, output, button_box_out, done)
    with button_box_out:
        display(button_box)
    return None