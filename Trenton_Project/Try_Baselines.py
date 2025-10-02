#Created: 9-30-2025
#Author: Trenton Wells
#Organization: NREL
#NREL Contact: trenton.wells@nrel.gov
#Personal Contact: trentonwells73@gmail.com
import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np

from Baseline_GIFTS import baseline_gifts
from Baseline_IRSQR import baseline_irsqr
from pybaselines import Baseline
from Trenton_Project.Analysis import get_default_parameters, parse_parameters, cast_parameter_types

def try_baseline(dataframe_path, material, baseline_function, parameter_string=None):
    """
    Apply a baseline correction to the first file of a given material and plot the result.

    Args:
        dataframe_path (str): Path to the dataframe CSV file.
        material (str): Material name to analyze.
        baseline_function (str): Baseline function to use ('GIFTS', 'IRSQR', 'FABC').
        parameter_string (str, optional): Baseline parameters as key=value pairs, comma-separated.
    """

    if dataframe_path is None:
        dataframe_path = r"dataframe.csv"
    if material is None:
        raise ValueError("Material must be specified.")
    if baseline_function is None:
        raise ValueError("Baseline function must be specified.")
    dataframe = pd.read_csv(dataframe_path)
    # Select the first row for the specified material where time == 0
    filtered = dataframe[(dataframe['material'] == material) & (dataframe['time'] == 0)]
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
    ax1.plot(x, y, label='Original Data')
    ax1.plot(x, baseline, label=f'{baseline_function} Baseline', linestyle='--')
    ax1.set_ylabel('absorbance')
    ax1.set_title(f'{material}: Data and {baseline_function} Baseline')
    ax1.legend()
    # Bottom plot: baseline-corrected
    ax2.plot(x, baseline_corrected, label='Baseline-Corrected', color='tab:green')
    ax2.set_xlabel('Wavenumber (cm^-1)')
    ax2.set_ylabel('absorbance')
    ax2.set_title('Baseline-Corrected')
    ax2.legend()
    plt.tight_layout()
    plt.show()

# Example usage (uncomment to use interactively):
#try_baseline(
#    dataframe_path=r"dataframe.csv",
#    material="PPE",
#    baseline_function="FABC",
#    parameter_string=None
#)
