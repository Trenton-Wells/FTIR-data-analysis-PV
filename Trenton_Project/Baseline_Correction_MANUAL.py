def manual_baseline_correction(y, anchor_points):
    """
    Perform manual baseline correction on a given spectrum using specified anchor points.
    Creates polynomial splines between anchor points.

    Parameters
    ----------
    y : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    anchor_points : list
        A list of x values representing points on the baseline.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    """
    import numpy as np
    from scipy.interpolate import CubicSpline

    # Extract x and y coordinates from baseline points
    x_points, y_points = zip(*baseline_points)
    
    # Create an interpolation function for the baseline
    interp_func = interp1d(x_points, y_points, kind='linear', fill_value='extrapolate')
    
    # Generate x values for the entire spectrum
    x_values = np.arange(len(spectrum))
    
    # Calculate the baseline using the interpolation function
    baseline = interp_func(x_values)
    
    # Perform baseline correction
    corrected_spectrum = spectrum - baseline
    
    return corrected_spectrum, baseline