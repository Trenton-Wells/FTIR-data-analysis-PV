#Temporary file for the purpose of modifying the existing CSV Dataframe
import pandas as pd
import sys
import os
sys.path.append(r"C:\Users\twells\Documents\GitHub\FTIR-data-analysis-PV")

def parse_parameters(param_str):
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
    for item in param_str.split(','):
        if '=' in item:
            key, value = item.split('=', 1)
            parameter_dictionary[key.strip()] = value.strip()
    return parameter_dictionary

def get_default_params(function_name):
    """
    Input the name of a baseline function and return its default parameters as a dictionary.
    Example: get_default_params('GIFTS') returns {'lam': 1e6, 'p': 0.01, 'n_iter': 10}
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
    'GIFTS': {'lam': 1e6, 'p': 0.01, 'n_iter': 10},
    'IRSQR': {'lam': 100, 'quantile': 0.05, 'num_knots': 100, 'spline_degree': 3, 'diff_order': 3, 'max_iter': 100, 'tol': 1e-6},
    'FABC': {'lam': 1e6, 'scale': None, 'num_std': 3.0, 'diff_order': 2, 'min_length': 2},
    'MANUAL': {}
    }
    return BASELINE_DEFAULTS.get(function_name.upper(), {})

    # Cast parameter types for each function
def cast_param_types(function_name, params):
    """
    Cast parameter types for each function based on known parameter types.

    Parameters
    ----------
    function_name : str
        The name of the baseline function.
    params : dict
        A dictionary of parameters to cast.

    Returns
    -------
    params : dict
        The dictionary with casted parameter types.
    """
    fn = function_name.upper()
    if fn == 'GIFTS':
        # lam: float, p: float, n_iter: int
        if 'lam' in params: params['lam'] = float(params['lam'])
        if 'p' in params: params['p'] = float(params['p'])
        if 'n_iter' in params: params['n_iter'] = int(params['n_iter'])
    elif fn == 'IRSQR':
        # lam: float, quantile: float, num_knots: int, spline_degree: int, diff_order: int, max_iter: int, tol: float
        if 'lam' in params: params['lam'] = float(params['lam'])
        if 'quantile' in params: params['quantile'] = float(params['quantile'])
        if 'num_knots' in params: params['num_knots'] = int(params['num_knots'])
        if 'spline_degree' in params: params['spline_degree'] = int(params['spline_degree'])
        if 'diff_order' in params: params['diff_order'] = int(params['diff_order'])
        if 'max_iter' in params: params['max_iter'] = int(params['max_iter'])
        if 'tol' in params: params['tol'] = float(params['tol'])
    elif fn == 'FABC':
        # lam: float, scale: int or None, num_std: float, diff_order: int, min_length: int
        if 'lam' in params: params['lam'] = float(params['lam'])
        if 'scale' in params:
            if params['scale'] is None or str(params['scale']).lower() == 'none' or params['scale'] == '':
                params['scale'] = None
            else:
                params['scale'] = int(params['scale'])
        if 'num_std' in params: params['num_std'] = float(params['num_std'])
        if 'diff_order' in params: params['diff_order'] = int(params['diff_order'])
        if 'min_length' in params: params['min_length'] = int(params['min_length'])
    return params

def prompt_parameters(function_name, material=None):
    """
    Prompt the user to input baseline parameters for a given function and material.

    Parameters
    ----------
    function_name : str
        The name of the baseline function.
    material : str, optional
        The material name for context (default is None).

    Returns
    -------
    params : dict
        A dictionary of parameters.
    """
    defaults = get_default_params(function_name)
    prompt_str = f"Enter the baseline parameters for {function_name}"
    if material:
        prompt_str += f" (material '{material}')"
    prompt_str += f" as key=value pairs separated by commas.\nPress Enter to use defaults: {defaults}\n> "
    user_input = input(prompt_str).strip()
    if not user_input:
        return defaults
    params = parse_parameters(user_input)
    # Fill in missing defaults
    for k, v in defaults.items():
        if k not in params:
            params[k] = v
    params = cast_param_types(function_name, params)
    return params

def baseline_selection(file_path):
    """
    Modify the DataFrame by adding baseline function and parameters based on user input. Save the results back to the CSV file.
    
    Parameters
    ----------
    file_path : str
        The path to the CSV file to modify.

    Returns
    -------
    None
    """
    # Read the existing CSV file into a DataFrame
    dataframe = pd.read_csv(file_path)

    # Add new columns for Baseline Function and Parameters if they don't exist
    if 'Baseline Function' not in dataframe.columns:
        dataframe['Baseline Function'] = None
    if 'Baseline Parameters' not in dataframe.columns:
        dataframe['Baseline Parameters'] = None

    # Ask user if they want to use the same baseline function for every material
    use_same_baseline = input("Would you like to use the same baseline function for every material? (y/n): ").strip().lower()
    print(f"User selected: {'Yes' if use_same_baseline == 'y' else 'No'}")
    if use_same_baseline == 'y':
        baseline_function = input("Enter the baseline function to use (e.g., 'IRSQR', 'GIFTS', 'FABC', 'MANUAL'): ").strip().upper()
        # Set the same baseline function for all rows
        for idx in dataframe.index:
            dataframe.at[idx, 'Baseline Function'] = baseline_function
    else:
        # For each unique material, ask user for baseline function and set for all rows with that material
        unique_materials = dataframe['material'].unique()
        for material in unique_materials:
            print(f"\nMaterial: {material}")
            baseline_function = input(f"Enter the baseline function to use for material '{material}' (e.g., 'IRSQR', 'GIFTS', 'FABC', 'MANUAL'): ").strip().upper()
            dataframe.loc[dataframe['material'] == material, 'Baseline Function'] = baseline_function

    use_same_parameters = input("Would you like to use the same baseline parameters for every material? (y/n): ").strip().lower()
    print(f"User selected: {'Yes' if use_same_parameters == 'y' else 'No'}")

    if use_same_parameters == 'y':
        # Use the same parameters for all rows, based on the selected baseline function
        if use_same_baseline == 'y':
            function_name = baseline_function
        else:
            # If not all the same, just use the first material's function
            function_name = dataframe['Baseline Function'].iloc[0]
        parameter_dictionary = prompt_parameters(function_name)
        for idx in dataframe.index:
            dataframe.at[idx, 'Baseline Parameters'] = str(parameter_dictionary)
    else:
        # For each unique material, ask user for parameters based on its baseline function
        for material in unique_materials:
            function_name = dataframe.loc[dataframe['material'] == material, 'Baseline Function'].iloc[0]
            parameter_dictionary = prompt_parameters(function_name, material)
            dataframe.loc[dataframe['material'] == material, 'Baseline Parameters'] = str(parameter_dictionary)
    # Save the modified DataFrame back to a CSV file
    dataframe.to_csv(file_path, index=False)

def delete_columns(file_path, columns_to_delete=None):
    """
    Delete specified columns from the DataFrame and save the results back to the CSV file.

    Parameters
    ----------
    file_path : str
        The path to the CSV file to modify.
    columns_to_delete : list of str
        A list of column names to delete from the DataFrame.

    Returns
    -------
    None
    """
    # Read the existing CSV file into a DataFrame
    dataframe = pd.read_csv(file_path)
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
    dataframe.to_csv(file_path, index=False)

def object_class_changer(file_path):
    """
    Detect and print the class types of objects in each column of the DataFrame.
    Give options to recast columns to specific types.

    Parameters
    ----------
    file_path : str
        The path to the CSV file to analyze.
    
    Returns
    -------
    None
    """
    import ast
    import numpy as np
    dataframe = pd.read_csv(file_path)
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
    dataframe.to_csv(file_path, index=False)

def baseline_correction(file_path):
    """
    Baseline correction function accesses the dataframe and applies baseline correction based on user-specified functions and parameters.
    Then saves the baseline and baseline-corrected spectra back to the CSV file.

    Parameters
    ----------
    file_path : str
        The path to the CSV file to modify.

    Returns
    -------
    None
    """
    import ast
    from Trenton_Project.Baseline_GIFTS import baseline_correction as gifts_baseline
    from Trenton_Project.Baseline_IRSQR import irsqr as irsqr_baseline
    from pybaselines import Baseline

    dataframe = pd.read_csv(file_path)

    # Add new columns for Baseline Function and Parameters if they don't exist
    if 'Baseline' not in dataframe.columns:
        dataframe['Baseline'] = None
    if 'Corrected' not in dataframe.columns:
        dataframe['Corrected'] = None

    for idx, row in dataframe.iterrows():
        baseline_name = row['Baseline Function']
        param_dict = ast.literal_eval(row['Baseline Parameters']) if row['Baseline Parameters'] else {}
        # Example: get y-data (Raw Data) and x-data (X-Axis)
        try:
            y_data = ast.literal_eval(row['Raw Data'])
        except Exception:
            y_data = row['Raw Data']

        baseline = None
        baseline_corrected = None

        if baseline_name == 'GIFTS':
            # Call GIFTS baseline correction
            baseline = gifts_baseline(y_data, **param_dict)
            baseline_corrected = [y - b for y, b in zip(y_data, baseline)]
        elif baseline_name == 'IRSQR':
            # Call IRSQR baseline correction
            baseline, _ = irsqr_baseline(None, y_data, **param_dict)  # Pass None for self if using as standalone
            baseline_corrected = [y - b for y, b in zip(y_data, baseline)]
        elif baseline_name == 'FABC':
            # Call FABC baseline correction
            baseline_obj = Baseline()
            baseline, _ = baseline_obj.fabc(y_data, **param_dict)
            baseline_corrected = [y - b for y, b in zip(y_data, baseline)]
        elif baseline_name == 'MANUAL':
            # Call Manual baseline correction
            from scipy.interpolate import CubicSpline
            anchor_points = param_dict.get('anchor_points', [])
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
    dataframe.to_csv(file_path, index=False)
if __name__ == "__main__":
    #baseline_selection(file_path=r"C:\Users\twells\Documents\GitHub\FTIR-data-analysis-PV\Trenton_Project\dataframe.csv")
    baseline_correction(file_path=r"C:\Users\twells\Documents\GitHub\FTIR-data-analysis-PV\Trenton_Project\dataframe.csv")