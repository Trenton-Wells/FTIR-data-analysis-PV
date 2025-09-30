import pandas as pd
import ast
import matplotlib.pyplot as plt
from Baseline_GIFTS import baseline_correction as gifts_baseline
from Baseline_IRSQR import irsqr as irsqr_baseline
from pybaselines import Baseline

# --- User input ---
dataframe_path = r"c:\Users\twells\Documents\GitHub\FTIR-data-analysis-PV\Trenton_Project\dataframe.csv"
material = input("Enter the material to analyze: ").strip()
baseline_function = input("Enter the baseline function (GIFTS, IRSQR, FABC): ").strip().upper()

# Optional: user can specify parameters as a string, or leave blank for defaults
param_str = input("Enter baseline parameters as key=value pairs (or press Enter for defaults): ").strip()
def parse_parameters(param_str):
    param_dict = {}
    for item in param_str.split(','):
        if '=' in item:
            key, value = item.split('=', 1)
            param_dict[key.strip()] = float(value.strip()) if value.replace('.', '', 1).isdigit() else value.strip()
    return param_dict

# --- Load DataFrame and select file ---
df = pd.read_csv(dataframe_path)
row = df[df['material'] == material].iloc[0]
x = ast.literal_eval(row['X-Axis']) if isinstance(row['X-Axis'], str) else row['X-Axis']
y = ast.literal_eval(row['Raw Data']) if isinstance(row['Raw Data'], str) else row['Raw Data']

# --- Get parameters ---
def get_default_parameters(function_name):
    BASELINE_DEFAULTS = {
        'GIFTS': {'lam': 1e6, 'p': 0.01, 'n_iter': 10},
        'IRSQR': {'lam': 1e6, 'quantile': 0.05, 'num_knots': 100, 'spline_degree': 3, 'diff_order': 3, 'max_iter': 100, 'tol': 1e-6},
        'FABC': {'lam': 1e6, 'scale': None, 'num_std': 3.0, 'diff_order': 2, 'min_length': 2},
    }
    return BASELINE_DEFAULTS.get(function_name.upper(), {})

if param_str:
    params = parse_parameters(param_str)
else:
    params = get_default_parameters(baseline_function)

# --- Apply baseline ---
if baseline_function == 'GIFTS':
    baseline = gifts_baseline(y, **params)
elif baseline_function == 'IRSQR':
    baseline, _ = irsqr_baseline(None, y, **params)
elif baseline_function == 'FABC':
    baseline_obj = Baseline()
    baseline, _ = baseline_obj.fabc(y, **params)
else:
    raise ValueError(f"Unknown baseline function: {baseline_function}")

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Original Data')
plt.plot(x, baseline, label=f'{baseline_function} Baseline', linestyle='--')
plt.plot(x, [a - b for a, b in zip(y, baseline)], label='Baseline-Corrected', linestyle=':')
plt.xlabel('Wavenumber (cm^-1)')
plt.ylabel('absorbance')
plt.title(f'{material}: Data, {baseline_function} Baseline, and Corrected')
plt.legend()
plt.tight_layout()
plt.show()
