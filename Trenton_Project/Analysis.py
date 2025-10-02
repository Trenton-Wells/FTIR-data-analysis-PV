#Created: 9-23-2025
#Author: Trenton Wells
#Organization: NREL
#NREL Contact: trenton.wells@nrel.gov
#Personal Contact: trentonwells73@gmail.com
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(r"C:\Users\twells\Documents\GitHub\FTIR-data-analysis-PV")

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

def baseline_selection(dataframe_path, materials_to_use=None, baseline_function=None, parameter_dictionary=None):
    """
    Add baseline function and parameters to a DataFrame for specified materials.
    
    Parameters
    ----------
    dataframe_path : str
        Path to the CSV file.
    materials_to_use : list of str or dict, optional
        Materials to apply baseline function to. Can be:
        - List of material names (same parameters for all)
        - Dict mapping material names to parameter dictionaries
    baseline_function : str, optional
        Baseline function to use.
    parameter_dictionary : dict, optional
        Parameters for baseline function (used when materials_to_use is a list).
    
    Returns
    -------
    None
    """

    # Read the existing CSV file into a DataFrame
    dataframe = pd.read_csv(dataframe_path)

    # Add new columns for Baseline Function and Parameters if they don't exist
    if 'Baseline Function' not in dataframe.columns:
        dataframe['Baseline Function'] = None
    if 'Baseline Parameters' not in dataframe.columns:
        dataframe['Baseline Parameters'] = None

    # User provides a list of materials to use
    if materials_to_use is None:
        materials_input = input("Enter materials to apply baseline function to (comma-separated): ").strip()
        materials_to_use = [mat.strip() for mat in materials_input.split(',')]
    
    # Get baseline function
    if baseline_function is None:
        baseline_function = input("Enter baseline function (GIFTS, IRSQR, FABC, Manual): ").strip()
    
    # Handle different parameter scenarios
    if isinstance(materials_to_use, dict):
        # Materials_to_use is a dictionary mapping materials to parameters
        for material, material_parameters in materials_to_use.items():
            if material_parameters is None:
                material_parameters = get_default_parameters(baseline_function)
            
            # Apply to rows with this material
            mask = dataframe['material'] == material
            dataframe.loc[mask, 'Baseline Function'] = baseline_function
            dataframe.loc[mask, 'Baseline Parameters'] = str(material_parameters)
            print(f"Applied {baseline_function} with parameters {material_parameters} to material: {material}")
    
    else:
        # Materials_to_use is a list - same parameters for all materials
        if parameter_dictionary is None:
            parameter_dictionary = get_default_parameters(baseline_function)
        
        for material in materials_to_use:
            mask = dataframe['material'] == material
            dataframe.loc[mask, 'Baseline Function'] = baseline_function
            dataframe.loc[mask, 'Baseline Parameters'] = str(parameter_dictionary)
            print(f"Applied {baseline_function} with parameters {parameter_dictionary} to material: {material}")
    
    # Save the modified DataFrame
    dataframe.to_csv(dataframe_path, index=False)
    print(f"Updated DataFrame saved to {dataframe_path}")

def baseline_selection_quick(dataframe_path, baseline_function=None, parameter_dictionary=None):
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
    # Read the existing CSV file into a DataFrame
    dataframe = pd.read_csv(dataframe_path)

    # Add new columns for Baseline Function and Parameters if they don't exist
    if 'Baseline Function' not in dataframe.columns:
        dataframe['Baseline Function'] = None
    if 'Baseline Parameters' not in dataframe.columns:
        dataframe['Baseline Parameters'] = None

    # Always use the same baseline function and parameters for all rows
    if baseline_function is None:
        baseline_function = input("Enter the baseline function to use for all materials (e.g., 'IRSQR', 'GIFTS', 'FABC'): ").strip().upper()
    for idx in dataframe.index:
        dataframe.at[idx, 'Baseline Function'] = baseline_function

    if parameter_dictionary is None:
        parameter_dictionary = get_default_parameters(baseline_function)
    for idx in dataframe.index:
        dataframe.at[idx, 'Baseline Parameters'] = str(parameter_dictionary)

    # Save the modified DataFrame back to a CSV file
    dataframe.to_csv(dataframe_path, index=False)

def delete_columns(dataframe_path, columns_to_delete=None):
    """
    Delete specified columns from the DataFrame and save the results back to the CSV file.

    Parameters
    ----------
    dataframe_path : str
        The path to the CSV file to modify.
    columns_to_delete : list of str
        A list of column names to delete from the DataFrame.

    Returns
    -------
    None
    """
    # Read the existing CSV file into a DataFrame
    dataframe = pd.read_csv(dataframe_path)
    if columns_to_delete is None:
        # Get Columns to delete from user
        columns_to_delete = input("Enter the column names to delete, separated by commas: ").strip().split(',')
    # State which columns will be deleted
    print("Columns requested for deletion:", columns_to_delete)
    # Delete specified columns if they exist
    for col in columns_to_delete:
        if col in dataframe.columns:
            dataframe.drop(col, axis=1, inplace=True)
            print(f"Deleted column: {col}")
        else:
            print(f"Column not found, cannot delete: {col}")

    # Save the modified DataFrame back to a CSV file
    dataframe.to_csv(dataframe_path, index=False)

def object_class_changer(dataframe_path):
    """
    Detect and print the class types of objects in each column of the DataFrame.
    Give options to recast columns to specific types.

    Parameters
    ----------
    dataframe_path : str
        The path to the CSV file to analyze.
    
    Returns
    -------
    None
    """
    import ast
    import numpy as np
    dataframe = pd.read_csv(dataframe_path)
    for column in dataframe.columns:
        print(f"Column: {column}")
        value = dataframe[column].iloc[0]
        print(f" Row 0: Type: {type(value)}, Value: {value}")
        print("\n")
        recast = input(f"Would you like to recast column '{column}' as a different type? (y/n): ").strip().lower()
        if recast == 'y':
            print("Options: string, float, integer, list, array, dictionary")
            dtype = input("Enter type to recast to: ").strip().lower()
            if dtype == 'string':
                dataframe[column] = dataframe[column].astype(str)
            elif dtype == 'float':
                dataframe[column] = pd.to_numeric(dataframe[column], errors='coerce')
            elif dtype == 'integer':
                dataframe[column] = pd.to_numeric(dataframe[column], errors='coerce').astype('Int64')
            elif dtype == 'list':
                dataframe[column] = dataframe[column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else list(x) if hasattr(x, '__iter__') and not isinstance(x, str) else [x])
            elif dtype == 'array':
                dataframe[column] = dataframe[column].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else np.array(x) if hasattr(x, '__iter__') and not isinstance(x, str) else np.array([x]))
            elif dtype == 'dictionary':
                dataframe[column] = dataframe[column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else dict(x) if isinstance(x, dict) else {})
            else:
                print("Unknown type. Skipping recast.")
            print(f"Column '{column}' recast to {dtype}.")
    # Save the modified DataFrame back to a CSV file
    dataframe.to_csv(dataframe_path, index=False)

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

def plot_grouped_spectra(FTIR_dataframe, material, condition, time, raw_data=True, baseline=False, baseline_corrected=False):
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

    Returns
    -------
    None
    """
    # Parse comma-separated strings into lists, handle 'any' (case-insensitive)
    mask = pd.Series([True] * len(FTIR_dataframe))
    if isinstance(material, str) and material.strip().lower() != "any":
        material_list = [m.strip() for m in material.split(',') if m.strip()]
        mask &= FTIR_dataframe['Material'].isin(material_list)
    if isinstance(condition, str) and condition.strip().lower() != "any":
        condition_list = [c.strip() for c in condition.split(',') if c.strip()]
        mask &= FTIR_dataframe['Conditions'].isin(condition_list)
    if isinstance(time, str) and time.strip().lower() != "any":
        # Try to convert to int if possible, else keep as string
        time_list = []
        for t in time.split(','):
            t = t.strip()
            if t:
                try:
                    time_list.append(int(t))
                except ValueError:
                    time_list.append(t)
        mask &= FTIR_dataframe['Time'].isin(time_list)
    filtered_data = FTIR_dataframe[mask]

    plt.figure(figsize=(10, 6))
    for index, row in filtered_data.iterrows():
        # Build label for this line
        mat = row.get('Material', '')
        cond = row.get('Conditions', row.get('Condition', ''))
        t = row.get('Time', '')
        base_label = f"{mat}, {cond}, {t}"
        # Choose which data to plot
        if raw_data and 'Raw Data' in row:
            plt.plot(row.get('X-Axis', row.get('Wavelength')), row['Raw Data'], label=f"Raw: {base_label}")
        if baseline and 'Baseline' in row and row['Baseline'] is not None:
            plt.plot(row.get('X-Axis', row.get('Wavelength')), row['Baseline'], '--', label=f"Baseline: {base_label}")
        if baseline_corrected and 'Corrected' in row and row['Corrected'] is not None:
            plt.plot(row.get('X-Axis', row.get('Wavelength')), row['Corrected'], ':', label=f"Corrected: {base_label}")
    plt.title(f"Spectra for Material: {material} | Condition: {condition} | Time: {time}")
    plt.xlabel('Wavelength (cm^-1)')
    plt.ylabel('Absorbance')
    plt.legend()
    plt.show()