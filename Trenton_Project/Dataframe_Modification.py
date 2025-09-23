#Temporary file for the purpose of modifying the existing CSV Dataframe
import pandas as pd

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

def modify_dataframe(file_path):
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

if __name__ == "__main__":
    file_path = r"C:\Users\twells\Documents\GitHub\FTIR-data-analysis-PV\Trenton_Project\dataframe.csv"
    modify_dataframe(file_path)