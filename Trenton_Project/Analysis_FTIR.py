#Created: 9-23-2025
#Author: Trenton Wells
#Organization: NREL
#NREL Contact: trenton.wells@nrel.gov
#Personal Contact: trentonwells73@gmail.com
import pandas as pd
import sys
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from Baseline_GIFTS import baseline_gifts
from Baseline_IRSQR import baseline_irsqr
from pybaselines import Baseline
import ast

def parse_parameters(parameter_str):
    """
    Parse a parameter string into a dictionary.
    Example input: "lam=100, quantile=0.05"
    Example output: {'lam': 100, 'quantile': 0.05}
    
    Parameters
    ----------
    param_str : str
        A string containing key=value pairs separated by commas.
    
    Returns
    -------
    parameter_dictionary : dict
        A dictionary with parameter names as keys and their corresponding values.
    """
    # Converts 'lam=100, quantile=0.05' to a dictionary
    parameter_dictionary = {}
    for item in parameter_str.split(','):
        if '=' in item:
            key, value = item.split('=', 1)
            parameter_dictionary[key.strip()] = value.strip()
    return parameter_dictionary

def get_default_parameters(function_name):
    """
    Input the name of a baseline function and return its default parameters as a dictionary.
    Example: get_default_parameters('GIFTS') returns {'lam': 1e6, 'p': 0.01, 'n_iter': 10}
    Parameters
    ----------
    function_name : str
        The name of the baseline function.

    Returns
    -------
    BASELINE_DEFAULTS.get(function_name.upper(), {}) : dict
        A dictionary of default parameters for the given function.
    """
    BASELINE_DEFAULTS = {
    'GIFTS': {'lam': 1e6, 'p': 0.01, 'iterations': 10},
    'IRSQR': {'lam': 1e6, 'quantile': 0.05, 'num_knots': 100, 'spline_degree': 3, 'diff_order': 3, 'max_iterations': 100, 'tolerance': 1e-6, 'weights': None, 'eps': None},
    'FABC': {'lam': 1e6, 'scale': None, 'num_std': 3.0, 'diff_order': 2, 'min_length': 2},
    'MANUAL': {}
    }
    return BASELINE_DEFAULTS.get(function_name.upper(), {})

def cast_parameter_types(function_name, parameters):
    """
    Cast parameter types for each function based on known parameter types.

    Parameters
    ----------
    function_name : str
        The name of the baseline function.
    parameters : dict
        A dictionary of parameters to cast.

    Returns
    -------
    parameters : dict
        The dictionary with casted parameter types.
    """
    function = function_name.upper()
    if function == 'GIFTS':
        # lam: float, p: float, n_iter: int
        if 'lam' in parameters: parameters['lam'] = float(parameters['lam'])
        if 'p' in parameters: parameters['p'] = float(parameters['p'])
        if 'iterations' in parameters: parameters['iterations'] = int(parameters['iterations'])
    elif function == 'IRSQR':
        # lam: float, quantile: float, num_knots: int, spline_degree: int, diff_order: int, max_iter: int, tol: float
        if 'lam' in parameters: parameters['lam'] = float(parameters['lam'])
        if 'quantile' in parameters: parameters['quantile'] = float(parameters['quantile'])
        if 'num_knots' in parameters: parameters['num_knots'] = int(parameters['num_knots'])
        if 'spline_degree' in parameters: parameters['spline_degree'] = int(parameters['spline_degree'])
        if 'diff_order' in parameters: parameters['diff_order'] = int(parameters['diff_order'])
        if 'max_iterations' in parameters: parameters['max_iterations'] = int(parameters['max_iterations'])
        if 'tolerance' in parameters: parameters['tolerance'] = float(parameters['tolerance'])
    elif function == 'FABC':
        # lam: float, scale: int or None, num_std: float, diff_order: int, min_length: int
        if 'lam' in parameters: parameters['lam'] = float(parameters['lam'])
        if 'scale' in parameters:
            if parameters['scale'] is None or str(parameters['scale']).lower() == 'none' or parameters['scale'] == '':
                parameters['scale'] = None
            else:
                parameters['scale'] = int(parameters['scale'])
        if 'num_std' in parameters: parameters['num_std'] = float(parameters['num_std'])
        if 'diff_order' in parameters: parameters['diff_order'] = int(parameters['diff_order'])
        if 'min_length' in parameters: parameters['min_length'] = int(parameters['min_length'])
    return parameters

def baseline_selection(FTIR_dataframe, materials_to_use=None, baseline_function=None, parameter_dictionary=None):
    """
    Set the baseline function for specified materials in the DataFrame.

    Parameters
    ----------
    FTIR_dataframe : pd.DataFrame
        The DataFrame containing the spectral data.
    materials_to_use : list, optional
        List of material names to apply the baseline function to.
    baseline_function : str, optional
        Baseline function to use.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame.
    """
    # Prompt for materials if not provided
    if materials_to_use is None:
        materials_input = input("Enter materials to apply baseline function to (comma-separated): ").strip()
        materials_to_use = [mat.strip() for mat in materials_input.split(',')]

    # Prompt for baseline function if not provided
    if baseline_function is None:
        baseline_function = input("Enter baseline function (GIFTS, IRSQR, FABC, Manual): ").strip()

    for material in materials_to_use:
        mask = FTIR_dataframe['Material'] == material
        FTIR_dataframe.loc[mask, 'Baseline Function'] = baseline_function
        print(f"Applied baseline function {baseline_function} to material: {material}")

    return FTIR_dataframe

def parameter_selection(FTIR_dataframe, materials_to_use=None, parameters_list=None):
    """
    Set the baseline parameters for specified materials in the DataFrame.
    
    Parameters
    ----------
    FTIR_dataframe : pd.DataFrame
        The DataFrame containing the spectral data.
    materials_to_use : list, optional
        List of material names to apply the parameters to.
    parameters_list : list, optional
        List of parameter values (as dict or string) to assign to each material (must match order of materials_to_use).
    
    Returns
    -------
    pd.DataFrame
        The updated DataFrame.
    """
    # Prompt for materials if not provided
    if materials_to_use is None:
        materials_input = input("Enter materials to apply parameters to (comma-separated): ").strip()
        materials_to_use = [mat.strip() for mat in materials_input.split(',')]

    # Prompt for parameters if not provided
    if parameters_list is None:
        params_input = input("Enter parameters for each material (comma-separated, as dict or string): ").strip()
        parameters_list = [param.strip() for param in params_input.split(',')]

    if len(materials_to_use) != len(parameters_list):
        raise ValueError("materials_to_use and parameters_list must have the same length.")

    for material, params in zip(materials_to_use, parameters_list):
        mask = FTIR_dataframe['Material'] == material
        FTIR_dataframe.loc[mask, 'Baseline Parameters'] = str(params)
        print(f"Applied parameters {params} to material: {material}")

    return FTIR_dataframe

def baseline_selection_quick(FTIR_dataframe, baseline_function=None, parameter_dictionary=None):
    """
    Modify the DataFrame by adding baseline function and parameters based on user input. Save the results back to the CSV file. Quick version that uses the same function and parameters for all rows.
    
    Parameters
    ----------
    dataframe_path : str
        The path to the CSV file to modify.

    Returns
    -------
    None
    """
    # Always use the same baseline function and parameters for all rows
    if baseline_function is None:
        baseline_function = input("Enter the baseline function to use for all materials (e.g., 'IRSQR', 'GIFTS', 'FABC'): ").strip().upper()
    for idx in FTIR_dataframe.index:
        FTIR_dataframe.at[idx, 'Baseline Function'] = baseline_function

    if parameter_dictionary is None:
        parameter_dictionary = get_default_parameters(baseline_function)
    for idx in FTIR_dataframe.index:
        FTIR_dataframe.at[idx, 'Baseline Parameters'] = str(parameter_dictionary)

    return FTIR_dataframe

def baseline_correction(dataframe_path):
    """
    Baseline correction function accesses the dataframe and applies baseline correction based on user-specified functions and parameters.
    Then saves the baseline and baseline-corrected spectra back to the CSV file.

    Parameters
    ----------
    dataframe_path : str
        The path to the CSV file to modify.

    Returns
    -------
    None
    """
    import ast
    from Trenton_Project.Baseline_GIFTS import baseline_correction as gifts_baseline
    from Trenton_Project.Baseline_IRSQR import irsqr as irsqr_baseline
    from pybaselines import Baseline

    dataframe = pd.read_csv(dataframe_path)

    # Add new columns for Baseline Function and Parameters if they don't exist
    if 'Baseline' not in dataframe.columns:
        dataframe['Baseline'] = None
    if 'Corrected' not in dataframe.columns:
        dataframe['Corrected'] = None

    for idx, row in dataframe.iterrows():
        baseline_name = row['Baseline Function']
        parameter_dictionary = ast.literal_eval(row['Baseline Parameters']) if row['Baseline Parameters'] else {}
        # Example: get y-data (Raw Data) and x-data (X-Axis)
        try:
            y_data = ast.literal_eval(row['Raw Data'])
        except Exception:
            y_data = row['Raw Data']

        baseline = None
        baseline_corrected = None

        if baseline_name == 'GIFTS':
            # Call GIFTS baseline correction
            baseline = gifts_baseline(y_data, **parameter_dictionary)
            baseline_corrected = [y - b for y, b in zip(y_data, baseline)]
        elif baseline_name == 'IRSQR':
            # Call IRSQR baseline correction
            baseline, _ = irsqr_baseline(None, y_data, **parameter_dictionary)  # Pass None for self if using as standalone
            baseline_corrected = [y - b for y, b in zip(y_data, baseline)]
        elif baseline_name == 'FABC':
            # Call FABC baseline correction
            baseline_obj = Baseline()
            baseline, _ = baseline_obj.fabc(y_data, **parameter_dictionary)
            baseline_corrected = [y - b for y, b in zip(y_data, baseline)]
        elif baseline_name == 'MANUAL':
            # Call Manual baseline correction
            from scipy.interpolate import CubicSpline
            anchor_points = parameter_dictionary.get('anchor_points', [])
            x_axis = ast.literal_eval(row['X-Axis']) if isinstance(row['X-Axis'], str) else row['X-Axis']
            # For each anchor point, find the index of the closest value in X-Axis
            # This ensures that the anchor points correspond to actual data points in each spectrum
            anchor_indices = [min(range(len(x_axis)), key=lambda i: abs(x_axis[i] - ap)) for ap in anchor_points]
            y_anchor = [y_data[i] for i in anchor_indices]
            baseline = CubicSpline(x=anchor_points, y=y_anchor, extrapolate=True)(x_axis)
        else:
            print(f"Unknown baseline function: {baseline_name} for row {idx}")
            continue
        # Save results back to DataFrame
        # Cast all baseline values to float before saving
        if baseline is not None:
            baseline_floats = [float(val) for val in baseline]
            dataframe.at[idx, 'Baseline'] = (baseline_floats)
        else:
           dataframe.at[idx, 'Baseline'] = None
        # Cast all corrected values to float before saving
        if baseline_corrected is not None:
           corrected_floats = [float(val) for val in baseline_corrected]
           dataframe.at[idx, 'Corrected'] = (corrected_floats)
        else:
           dataframe.at[idx, 'Corrected'] = None


    # Save updated DataFrame
    dataframe.to_csv(dataframe_path, index=False)

def plot_grouped_spectra(FTIR_dataframe, materials, conditions, times, raw_data=True, baseline=False, baseline_corrected=False, separate_plots=False, include_replicates=True, zoom=None):
    """
    Plot grouped spectra based on material, condition, and time. Accepts lists or 'any' for each category.

    Parameters
    ----------
    FTIR_dataframe : pd.DataFrame
        The DataFrame containing the spectral data.
    material : str, list, or 'any'
        The material(s) to filter by, or 'any' to include all.
    condition : str, list, or 'any'
        The condition(s) to filter by, or 'any' to include all.
    time : str, int, list, or 'any'
        The time(s) to filter by, or 'any' to include all.
    raw_data : bool, optional
        Whether to plot the raw data (default is True).
    baseline : bool, optional
        Whether to plot the baseline (default is False).
    baseline_corrected : bool, optional
        Whether to plot the baseline-corrected data (default is False).
    separate_plots : bool, optional
        Whether to create separate plots for each spectrum (default is False).
    include_replicates : bool, optional
        Whether to include all replicates or just the first of each group (default is True).
    zoom : str, optional
        A string specifying the x-axis zoom range in the format "min-max" (e.g., "400-4000").

    Returns
    -------
    None
    """
    
    # Parse comma-separated strings into lists, handle 'any' (case-insensitive)
    mask = pd.Series([True] * len(FTIR_dataframe))
    if isinstance(materials, str) and materials.strip().lower() != "any":
        material_list = [m.strip() for m in materials.split(',') if m.strip()]
        mask &= FTIR_dataframe['Material'].isin(material_list)
    if isinstance(conditions, str) and conditions.strip().lower() != "any":
        condition_list = [c.strip() for c in conditions.split(',') if c.strip()]
        mask &= FTIR_dataframe['Conditions'].isin(condition_list)
    if isinstance(times, str) and times.strip().lower() != "any":
        # Try to convert to int if possible, else keep as string
        time_list = []
        for t in times.split(','):
            t = t.strip()
            if t:
                try:
                    time_list.append(int(t))
                except ValueError:
                    time_list.append(t)
        mask &= FTIR_dataframe['Time'].isin(time_list)
    filtered_data = FTIR_dataframe[mask]

    # If not including replicates, keep only the first member of each (Material, Conditions, Time) group
    if not include_replicates:
        filtered_data = filtered_data.sort_values(by=['Material', 'Conditions', 'Time'])
        filtered_data = filtered_data.drop_duplicates(subset=['Material', 'Conditions', 'Time'], keep='first')

    # Sort by time once for both legend and plotting (assume all times are integers)
    filtered_data_sorted = filtered_data.sort_values(by='Time')
    x_axis_col = 'X-Axis' if 'X-Axis' in filtered_data_sorted.columns else 'Wavelength'

    # Plot all together (legend in time order) and record colors for each spectrum (including replicates)
    plt.figure(figsize=(10, 6))
    legend_entries = []
    color_map = {}  # Map from (data type, DataFrame index) to color
    legend_filepaths = []  # List of filepaths in legend order
    for idx, spectrum_row in filtered_data_sorted.iterrows():
        material_val = spectrum_row.get('Material', '')
        condition_val = spectrum_row.get('Conditions', spectrum_row.get('Condition', ''))
        time_val = spectrum_row.get('Time', '')
        spectrum_label = f"{material_val}, {condition_val}, {time_val}"
        # Use ast.literal_eval for x_axis and data columns if they are strings
        x_axis = spectrum_row.get(x_axis_col)
        if isinstance(x_axis, str):
            try:
                x_axis = ast.literal_eval(x_axis)
            except Exception:
                pass
        file_path = os.path.join(spectrum_row['File Location'], spectrum_row['File Name'])
        if raw_data and 'Raw Data' in spectrum_row:
            y_data = spectrum_row['Raw Data']
            if isinstance(y_data, str):
                try:
                    y_data = ast.literal_eval(y_data)
                except Exception:
                    pass
            line_handle, = plt.plot(x_axis, y_data, label=f"Raw: {spectrum_label}")
            legend_entries.append((line_handle, f"Raw: {spectrum_label}"))
            color_map[("Raw", idx)] = line_handle.get_color()
            legend_filepaths.append(file_path)
        if baseline and 'Baseline' in spectrum_row and spectrum_row['Baseline'] is not None:
            y_data = spectrum_row['Baseline']
            if isinstance(y_data, str):
                try:
                    y_data = ast.literal_eval(y_data)
                except Exception:
                    pass
            line_handle, = plt.plot(x_axis, y_data, '--', label=f"Baseline: {spectrum_label}")
            legend_entries.append((line_handle, f"Baseline: {spectrum_label}"))
            color_map[("Baseline", idx)] = line_handle.get_color()
            legend_filepaths.append(file_path)
        if baseline_corrected and 'Corrected' in spectrum_row and spectrum_row['Corrected'] is not None:
            y_data = spectrum_row['Corrected']
            if isinstance(y_data, str):
                try:
                    y_data = ast.literal_eval(y_data)
                except Exception:
                    pass
            line_handle, = plt.plot(x_axis, y_data, ':', label=f"Corrected: {spectrum_label}")
            legend_entries.append((line_handle, f"Corrected: {spectrum_label}"))
            color_map[("Corrected", idx)] = line_handle.get_color()
            legend_filepaths.append(file_path)
    handles = [entry[0] for entry in legend_entries]
    labels = [entry[1] for entry in legend_entries]
    # Print filepaths in legend order
    for fp in legend_filepaths:
        print(f"Plotting: {fp}")
    plt.title(f"Spectra for Material: {materials} | Condition: {conditions} | Time: {times}")
    plt.xlabel('Wavelength (cm¯¹)')
    plt.ylabel('Absorbance (AU)')
    plt.legend(handles, labels)
    # Set zoom if provided
    if zoom is not None and isinstance(zoom, str):
        try:
            zoom_range = zoom.replace(' ', '').split('-')
            if len(zoom_range) == 2:
                x_min, x_max = float(zoom_range[0]), float(zoom_range[1])
                plt.xlim(x_min, x_max)
        except Exception as e:
            print(f"Warning: Could not parse zoom argument '{zoom}': {e}")
    plt.show()

    # Plot each file individually if requested, in sequential order by time
    if separate_plots:
        for idx, row in filtered_data_sorted.iterrows():
            # Print the file path for this individual plot only
            file_path = os.path.join(row.get('File Location', ''), row.get('File Name', ''))
            print(f"Plotting: {file_path}")
            material_val = row.get('Material', '')
            condition_val = row.get('Conditions', row.get('Condition', ''))
            time_val = row.get('Time', '')
            spectrum_label = f"{material_val}, {condition_val}, {time_val}"
            x_axis = row.get(x_axis_col)
            if isinstance(x_axis, str):
                try:
                    x_axis = ast.literal_eval(x_axis)
                except Exception:
                    pass
            plt.figure(figsize=(8, 5))
            if raw_data and 'Raw Data' in row:
                y_data = row['Raw Data']
                if isinstance(y_data, str):
                    try:
                        y_data = ast.literal_eval(y_data)
                    except Exception:
                        pass
                color = color_map.get(("Raw", idx), None)
                plt.plot(x_axis, y_data, label="Raw", color=color)
            if baseline and 'Baseline' in row and row['Baseline'] is not None:
                y_data = row['Baseline']
                if isinstance(y_data, str):
                    try:
                        y_data = ast.literal_eval(y_data)
                    except Exception:
                        pass
                color = color_map.get(("Baseline", idx), None)
                plt.plot(x_axis, y_data, '--', label="Baseline", color=color)
            if baseline_corrected and 'Corrected' in row and row['Corrected'] is not None:
                y_data = row['Corrected']
                if isinstance(y_data, str):
                    try:
                        y_data = ast.literal_eval(y_data)
                    except Exception:
                        pass
                color = color_map.get(("Corrected", idx), None)
                plt.plot(x_axis, y_data, ':', label="Corrected", color=color)
            plt.title(f"Spectrum: {spectrum_label}")
            plt.xlabel('Wavelength (cm¯¹)')
            plt.ylabel('Absorbance (AU)')
            plt.legend()
            # Set zoom if provided
            if zoom is not None and isinstance(zoom, str):
                try:
                    zoom_range = zoom.replace(' ', '').split('-')
                    if len(zoom_range) == 2:
                        x_min, x_max = float(zoom_range[0]), float(zoom_range[1])
                        plt.xlim(x_min, x_max)
                except Exception as e:
                    print(f"Warning: Could not parse zoom argument '{zoom}': {e}")
            plt.show()

def try_baseline(FTIR_dataframe, material=None, baseline_function=None, parameter_string=None, filepath=None):
    """
    Apply a baseline correction to the first file of a given material and plot the result, or to a specific file if filepath is provided.

    Args:
        FTIR_dataframe (pd.DataFrame): The in-memory DataFrame containing all spectra.
        material (str, optional): Material name to analyze (ignored if filepath is provided).
        baseline_function (str): Baseline function to use ('GIFTS', 'IRSQR', 'FABC').
        parameter_string (str, optional): Baseline parameters as key=value pairs, comma-separated.
        filepath (str, optional): If provided, only process this file (by 'File Location' + 'File Name').
    """

    if baseline_function is None:
        raise ValueError("Baseline function must be specified.")

    if filepath is not None:
        # Find the row matching the given file path (must match both File Location and File Name)
        # filepath can be full path or just file name; try both
        if os.path.sep in filepath:
            # Full path: split into folder and file
            folder, fname = os.path.split(filepath)
            filtered = FTIR_dataframe[(FTIR_dataframe['File Location'] == folder) & (FTIR_dataframe['File Name'] == fname)]
        else:
            # Just file name: match any file with that name
            filtered = FTIR_dataframe[FTIR_dataframe['File Name'] == filepath]
        if filtered.empty:
            raise ValueError(f"No entry found for file '{filepath}'.")
        row = filtered.iloc[0]
        material = row.get('Material', 'Unknown')
    else:
        if material is None:
            raise ValueError("Material must be specified if filepath is not provided.")
        # Select the first row for the specified material where time == 0
        filtered = FTIR_dataframe[(FTIR_dataframe['Material'] == material) & (FTIR_dataframe['Time'] == 0)]
        if filtered.empty:
            raise ValueError(f"No entry found for material '{material}' with time == 0.")
        row = filtered.iloc[0]

    x = ast.literal_eval(row['X-Axis']) if isinstance(row['X-Axis'], str) else row['X-Axis']
    y = ast.literal_eval(row['Raw Data']) if isinstance(row['Raw Data'], str) else row['Raw Data']
    y = np.array(y, dtype=float)
    if parameter_string:
        parameters = parse_parameters(parameter_string)
    else:
        parameters = get_default_parameters(baseline_function)
    parameters = cast_parameter_types(baseline_function, parameters)

    # Print the full file path before plotting
    file_path = os.path.join(row['File Location'], row['File Name'])
    print(f"Plotting: {file_path}")

    if baseline_function.upper() == 'GIFTS':
        baseline = baseline_gifts(y, **parameters)
    elif baseline_function.upper() == 'IRSQR':
        baseline, _ = baseline_irsqr(y, **parameters, x_axis=x)
    elif baseline_function.upper() == 'FABC':
        baseline_obj = Baseline(x)
        baseline, _ = baseline_obj.fabc(y, **parameters)
    else:
        raise ValueError(f"Unknown baseline function: {baseline_function}")

    baseline_corrected = [a - b for a, b in zip(y, baseline)]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    # Top plot: original and baseline
    ax1.plot(x, y, label='Raw Data')
    ax1.plot(x, baseline, label=f'{baseline_function} Baseline', linestyle='--')
    ax1.set_ylabel('Absorbance (AU)')
    ax1.set_title(f'{material}: Raw Data and {baseline_function} Baseline')
    ax1.legend()
    # Bottom plot: baseline-corrected
    ax2.plot(x, baseline_corrected, label='Baseline-Corrected', color='tab:green')
    ax2.set_xlabel('Wavenumber (cm¯¹)')
    ax2.set_ylabel('Absorbance (AU)')
    ax2.set_title('Baseline-Corrected')
    ax2.legend()
    plt.tight_layout()
    plt.show()

def bring_in_dataframe(dataframe_path=None):
    """
    Load the CSV file into a pandas DataFrame.

    Parameters
    ----------
    dataframe_path : str
        The path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.
    """
    if dataframe_path is None:
        dataframe_path = "FTIR_dataframe.csv" # Default path if none is provided (will be in active directory)
    if os.path.exists(dataframe_path):
        FTIR_dataframe = pd.read_csv(dataframe_path) # Load the DataFrame from the specified path
    else:
        FTIR_dataframe = pd.DataFrame() # Create a new empty DataFrame if it doesn't exist
    return FTIR_dataframe
