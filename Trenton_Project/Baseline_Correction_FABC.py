import matplotlib.pyplot as plt
import numpy as np
from pybaselines import Baseline
from math import ceil
import warnings
import pandas as pd
import ast

from scipy.ndimage import (
    binary_erosion, binary_opening
)
from scipy.signal import convolve
from pybaselines.utils import ( 
    optimize_window, pad_edges, ParameterWarning
)

def _haar(num_points, scale=2):
    """
    Creates a Haar wavelet.

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
        raise TypeError('scale must be an integer for the Haar wavelet')
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

def _iter_threshold(power, num_std=3.0):
    """
    Iteratively thresholds a power spectrum based on the mean and standard deviation.

    Any values greater than the mean of the power spectrum plus a multiple of the
    standard deviation are masked out to create a new power spectrum. The process
    is performed iteratively until no further points are masked out.

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
                ParameterWarning, stacklevel=2
            )
            break
        mask = power < np.mean(masked_power) + num_std * np.std(masked_power, ddof=1)

    return mask

def _refine_mask(mask, min_length=2):
    """
    Removes small consecutive True values and lone False values from a boolean mask.

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
        np.pad(mask, half_length, 'constant', constant_values=True), np.ones(min_length, bool)
    )[half_length:len(mask) + half_length]

    # convert lone False values to True
    np.logical_or(output, binary_erosion(output, [1, 0, 1]), out=output)
    # TODO should there be an erosion step here, using another parameter (erode_hw)?
    # that way, can control both the minimum length and then remove edges of baselines
    # independently, allowing more control over the output mask
    return output

def _cwt(data, wavelet, widths, dtype=None, **kwargs):
    """
    Continuous wavelet transform.

    Performs a continuous wavelet transform on `data`,
    using the `wavelet` function. A CWT performs a convolution
    with `data` using the `wavelet` function, which is characterized
    by a width parameter and length parameter. The `wavelet` function
    is allowed to be complex.

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
    S. Mallat, "A Wavelet Tour of Signal Processing (3rd Edition)", Academic Press, 2009.

    """
    # Determine output type
    if dtype is None:
        if np.asarray(wavelet(1, widths[0], **kwargs)).dtype.char in 'FDG':
            dtype = np.complex128
        else:
            dtype = np.float64

    output = np.empty((len(widths), len(data)), dtype=dtype)
    for ind, width in enumerate(widths):
        N = np.min([10 * width, len(data)])
        wavelet_data = np.conj(wavelet(N, width, **kwargs)[::-1])
        output[ind] = convolve(data, wavelet_data, mode='same')
    return output

def fabc(self, data, lam=1e6, scale=None, num_std=3.0, diff_order=2, min_length=2,
             weights=None, weights_as_mask=False, pad_kwargs=None, **kwargs):
        """
        Fully automatic baseline correction (fabc).

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
            with the number of data points and gives a starting point for tuning the parameter.
        num_std : float, optional
            The number of standard deviations to include when thresholding. Higher values
            will assign more points as baseline. Default is 3.0.
        diff_order : int, optional
            The order of the differential matrix. Must be greater than 0. Default is 2
            (second order differential matrix). Typical values are 2 or 1.
        min_length : int, optional
            Any region of consecutive baseline points less than `min_length` is considered
            to be a false positive and all points in the region are converted to peak points.
            A higher `min_length` ensures less points are falsely assigned as baseline points.
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
                Passing additional keyword arguments is deprecated and will be removed in version
                1.4.0. Pass keyword arguments using `pad_kwargs`.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'mask': numpy.ndarray, shape (N,)
                The boolean array designating baseline points as True and peak points
                as False.
            * 'weights': numpy.ndarray, shape (N,)
                The weight array used for fitting the data.

        Notes
        -----
        The classification of baseline points is similar to :meth:`~.Baseline.dietrich`, except that
        this method approximates the first derivative using a continous wavelet transform
        with the Haar wavelet, which is more robust than the numerical derivative in
        Dietrich's method.

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
            wavelet_cwt = _cwt(pad_edges(y, half_window, **pad_kwargs, **kwargs), _haar, [scale])
            power = wavelet_cwt[0, half_window:-half_window]**2

            mask = _refine_mask(_iter_threshold(power, num_std), min_length)
            np.logical_and(mask, weight_array, out=mask)

            _, whittaker_weights, whittaker_system = self._setup_whittaker(y, lam, diff_order, mask)
            if self._sort_order is not None:
                whittaker_weights = whittaker_weights[self._inverted_order]

        whittaker_weights = whittaker_weights.astype(float)
        baseline = whittaker_system.solve(
            whittaker_system.add_diagonal(whittaker_weights), whittaker_weights * y,
            overwrite_b=True, overwrite_ab=True
        )
        params = {'mask': mask, 'weights': whittaker_weights}

        return baseline, params

if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\twells\Documents\GitHub\FTIR-data-analysis-PV\Trenton_Project\dataframe.csv")
    for idx, row in df.iterrows():
        # Get x-axis data from column before data_list (column 6, index 6)
        x_data = ast.literal_eval(row.iloc[6])
        # Convert string list to actual list for y-axis
        data_list = ast.literal_eval(row.iloc[7])
        baseline_obj = Baseline()
        baseline, params = baseline_obj.fabc(data_list)
        baseline_corrected = np.array(data_list) - baseline
        # Plot using x_data for x-axis
        plt.plot(x_data, data_list, label='Original Data')
        plt.plot(x_data, baseline, label='Baseline', linestyle='--')
        #plt.plot(x_data, baseline_corrected, label='Baseline_Corrected', linestyle=(0, (1, 1)))
        plt.xlabel('Wavenumber (cm$^{-1}$)')
        plt.ylabel('Absorbance')
        # Use filename from second column (index 1)
        plt.title(str(row.iloc[1]))
        plt.legend()
        plt.show()