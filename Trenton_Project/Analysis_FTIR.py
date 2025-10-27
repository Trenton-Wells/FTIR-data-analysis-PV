# Created: 9-23-2025
# Author: Trenton Wells
# Organization: NREL
# NREL Contact: trenton.wells@nrel.gov
# Personal Contact: trentonwells73@gmail.com
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from pybaselines.whittaker import arpls
from pybaselines.spline import irsqr
from pybaselines.classification import fabc
from scipy.interpolate import CubicSpline
from pybaselines.utils import optimize_window
from scipy.signal import find_peaks
import ast
import ipywidgets as widgets
import plotly.graph_objs as go
from IPython.display import display, clear_output
from math import ceil
import importlib
from plotly.subplots import make_subplots
import threading


# ---- Validation helpers for clearer, user-friendly errors ---- #
def _require_columns(df, columns, context="DataFrame"):
    """
    Ensure the DataFrame contains the given columns, else raise a KeyError.

    Helper function.
    """
    if df is None:
        raise ValueError(f"{context} is None. A valid pandas DataFrame is required.")
    try:
        cols = list(df.columns)
    except Exception:
        raise TypeError(
            f"{context} must be a pandas DataFrame with a 'columns' " f"attribute."
        )
    missing = [c for c in columns if c not in cols]
    if missing:
        raise KeyError(
            f"Missing required column(s) in {context}: {missing}. Available columns: "
            f"{cols}"
        )


def _safe_literal_eval(val, value_name="value"):
    """
    Safely parse string representations of Python literals, with descriptive errors.

    Helper function.
    """
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception as e:
            raise ValueError(
                f"Could not parse {value_name} from string: {val!r}. " f"Error: {e}"
            )
    return val


def _ensure_1d_numeric_array(name, seq):
    """Coerce a sequence into a 1D float numpy array; raise clear error if invalid."""
    seq = _safe_literal_eval(seq, value_name=name)
    try:
        arr = np.asarray(seq, dtype=float)
    except Exception as e:
        raise ValueError(
            f"{name} must be a sequence of numbers. Got: "
            f"{type(seq).__name__}. Error: {e}"
        )
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D. Got array with shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} is empty. A non-empty sequence is required.")
    return arr


def _parse_parameters(parameter_str):
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
    if parameter_str is None:
        return {}
    if not isinstance(parameter_str, str):
        raise TypeError(
            f"parameter_str must be a string of key=value pairs, got "
            f"{type(parameter_str).__name__}."
        )
    tokens = [tok.strip() for tok in parameter_str.split(",") if tok.strip()]
    if not tokens:
        return {}

    def _coerce_scalar(s):
        low = s.strip().lower()
        if low in {"none", "null", "nan"}:
            return None
        if low in {"true", "false"}:
            return low == "true"
        try:
            if any(ch in s for ch in ".eE"):
                return float(s)
            return int(s)
        except Exception:
            try:
                return ast.literal_eval(s)
            except Exception:
                return s

    parameter_dictionary = {}
    for item in tokens:
        if "=" not in item:
            raise ValueError(
                f"Invalid parameter token {item!r}. Expected format 'key=value'. Full "
                f"string: {parameter_str!r}"
            )
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Found empty parameter name in token {item!r}.")
        parameter_dictionary[key] = _coerce_scalar(value)
    return parameter_dictionary


def _get_default_parameters(function_name):
    """
    Input the name of a baseline function and return its default parameters as a
    dictionary.

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
        "ARPLS": {
            "lam": 1e5,
            "diff_order": 2,
            "max_iter": 50,
            "tol": 1e-3,
            "weights": None,
        },
        "IRSQR": {
            "lam": 1e6,
            "quantile": 0.05,
            "num_knots": 100,
            "spline_degree": 3,
            "diff_order": 3,
            "max_iter": 100,
            "tol": 1e-6,
            "weights": None,
            "eps": None,
        },
        "FABC": {
            "lam": 1e6,
            "scale": None,
            "num_std": 3.0,
            "diff_order": 2,
            "min_length": 2,
            "weights": None,
            "weights_as_mask": False,
            "pad_kwargs": None,
        },
        "MANUAL": {},
    }
    return BASELINE_DEFAULTS.get(function_name.upper(), {})


def _cast_parameter_types(function_name, parameters):
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
    if function == "ARPLS":
        if "lam" in parameters:
            parameters["lam"] = float(parameters["lam"])
        if "diff_order" in parameters:
            parameters["diff_order"] = int(parameters["diff_order"])
        if "max_iter" in parameters:
            parameters["max_iter"] = int(parameters["max_iter"])
        if "tol" in parameters:
            parameters["tol"] = float(parameters["tol"])
        if "weights" in parameters:
            if str(parameters["weights"]).lower() not in ["none", "null", ""]:
                try:
                    parameters["weights"] = ast.literal_eval(parameters["weights"])
                except Exception:
                    pass
    elif function == "IRSQR":
        if "lam" in parameters:
            parameters["lam"] = float(parameters["lam"])
        if "quantile" in parameters:
            parameters["quantile"] = float(parameters["quantile"])
        if "num_knots" in parameters:
            parameters["num_knots"] = int(parameters["num_knots"])
        if "spline_degree" in parameters:
            parameters["spline_degree"] = int(parameters["spline_degree"])
        if "diff_order" in parameters:
            parameters["diff_order"] = int(parameters["diff_order"])
        if "max_iter" in parameters:
            parameters["max_iter"] = int(parameters["max_iter"])
        if "tol" in parameters:
            parameters["tol"] = float(parameters["tol"])
        if "weights" in parameters:
            if str(parameters["weights"]).lower() not in ["none", "null", ""]:
                try:
                    parameters["weights"] = ast.literal_eval(parameters["weights"])
                except Exception:
                    pass
        if "eps" in parameters:
            if str(parameters["eps"]).lower() not in ["none", "null", ""]:
                try:
                    parameters["eps"] = float(parameters["eps"])
                except Exception:
                    pass
    elif function == "FABC":
        if "lam" in parameters:
            parameters["lam"] = float(parameters["lam"])
        if str(parameters["scale"]).lower() not in ["none", "null", ""]:
            try:
                parameters["scale"] = ast.literal_eval(parameters["scale"])
            except Exception:
                pass
        if "num_std" in parameters:
            parameters["num_std"] = float(parameters["num_std"])
        if "diff_order" in parameters:
            parameters["diff_order"] = int(parameters["diff_order"])
        if "min_length" in parameters:
            parameters["min_length"] = int(parameters["min_length"])
        if "weights" in parameters:
            if str(parameters["weights"]).lower() not in ["none", "null", ""]:
                try:
                    parameters["weights"] = ast.literal_eval(parameters["weights"])
                except Exception:
                    pass
        if "weights_as_mask" in parameters:
            if str(parameters["weights_as_mask"]).lower() in ["true"]:
                parameters["weights_as_mask"] = True
            else:
                parameters["weights_as_mask"] = False
        if "pad_kwargs" in parameters:
            if parameters["pad_kwargs"] is not None:
                try:
                    parameters["pad_kwargs"] = ast.literal_eval(
                        parameters["pad_kwargs"]
                    )
                except Exception:
                    pass
    return parameters


def baseline_correction(FTIR_DataFrame, materials="any"):
    """
    Apply baseline correction to spectra in the DataFrame for selected materials or all.

    Parameters
    ----------
    FTIR_DataFrame : pd.DataFrame
        The DataFrame to update in-place.
    materials : str or list
        Material(s) to filter and process. Use 'any' (case-insensitive) to process all
        rows.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame.
    """
    # Validate required columns early for clearer errors
    _require_columns(
        FTIR_DataFrame,
        ["Material", "X-Axis", "Raw Data"],
        context="FTIR_DataFrame (baseline_correction)",
    )

    # Ensure destination columns exist and are object dtype (for per-row lists)
    for col in ("Baseline", "Baseline-Corrected Data"):
        if col not in FTIR_DataFrame.columns:
            FTIR_DataFrame[col] = None
    # Coerce sequence-holding columns to object dtype to avoid shape/broadcast issues
    try:
        FTIR_DataFrame[["Baseline", "Baseline-Corrected Data"]] = FTIR_DataFrame[
            ["Baseline", "Baseline-Corrected Data"]
        ].astype(object)
    except Exception:
        # Fall back to individual-coercion if slice fails (e.g., missing one column)
        for _col in ("Baseline", "Baseline-Corrected Data"):
            if _col in FTIR_DataFrame.columns:
                try:
                    FTIR_DataFrame[_col] = FTIR_DataFrame[_col].astype(object)
                except Exception:
                    pass

    # Build mask for materials
    if isinstance(materials, str):
        if materials.strip().lower() == "any":
            mask = pd.Series([True] * len(FTIR_DataFrame), index=FTIR_DataFrame.index)
        else:
            material_list = [m.strip() for m in materials.split(",") if m.strip()]
            mask = FTIR_DataFrame["Material"].astype(str).isin(material_list)
    elif isinstance(materials, (list, tuple)):
        material_list = [str(m).strip() for m in materials if str(m).strip()]
        mask = FTIR_DataFrame["Material"].astype(str).isin(material_list)
    else:
        mask = pd.Series([True] * len(FTIR_DataFrame), index=FTIR_DataFrame.index)

    updated = 0
    skipped = 0
    for idx in FTIR_DataFrame.index[mask]:
        row = FTIR_DataFrame.loc[idx]
        baseline_name = row.get("Baseline Function", None)
        if baseline_name is None or str(baseline_name).strip() == "":
            print(f"Row {idx}: Missing 'Baseline Function'; skipping.")
            skipped += 1
            continue

        # Robustly parse parameters (dict or string) and merge with defaults
        raw_params = row.get("Baseline Parameters", {})
        if isinstance(raw_params, dict):
            params = raw_params.copy()
        elif isinstance(raw_params, str) and raw_params.strip():
            try:
                maybe = ast.literal_eval(raw_params)
                params = (
                    maybe if isinstance(maybe, dict) else _parse_parameters(raw_params)
                )
            except Exception:
                params = _parse_parameters(raw_params)
        else:
            params = {}

        func_name = str(baseline_name).strip().upper()
        defaults = _get_default_parameters(func_name)
        params = {**defaults, **params}
        params = _cast_parameter_types(func_name, params)

        # Parse data arrays
        try:
            y_data = (
                ast.literal_eval(row.get("Raw Data"))
                if isinstance(row.get("Raw Data"), str)
                else row.get("Raw Data")
            )
        except Exception:
            y_data = row.get("Raw Data")
        try:
            x_axis = (
                ast.literal_eval(row.get("X-Axis"))
                if isinstance(row.get("X-Axis"), str)
                else row.get("X-Axis")
            )
        except Exception:
            x_axis = row.get("X-Axis")
        if y_data is None or x_axis is None:
            print(f"Row {idx}: Missing X-Axis or Raw Data; skipping.")
            skipped += 1
            continue

        baseline = None
        baseline_corrected = None
        try:
            if func_name == "ARPLS":
                result = arpls(y_data, **params)
            elif func_name == "IRSQR":
                result = (
                    irsqr(y_data, **params, x_data=x_axis)
                    if "x_data" not in params
                    else irsqr(y_data, **params)
                )
            elif func_name == "FABC":
                result = fabc(y_data, **params)
            elif func_name == "MANUAL":
                anchor_points = params.get("anchor_points", [])
                if not anchor_points:
                    raise ValueError("MANUAL baseline requires 'anchor_points'.")
                # indices of closest x to each anchor point
                anchor_indices = [
                    min(range(len(x_axis)), key=lambda i: abs(x_axis[i] - ap))
                    for ap in anchor_points
                ]
                y_anchor = [y_data[i] for i in anchor_indices]
                baseline = CubicSpline(x=anchor_points, y=y_anchor, extrapolate=True)(
                    x_axis
                )
            else:
                raise ValueError(f"Unknown baseline function: {baseline_name}")

            if baseline is None:
                # Normalize return type from pybaselines
                if isinstance(result, tuple):
                    baseline = result[0]
                elif isinstance(result, dict):
                    baseline = result.get("baseline", None)
                else:
                    baseline = result
            baseline = np.asarray(baseline, dtype=float)
            y_arr = np.asarray(y_data, dtype=float)
            if baseline.shape != y_arr.shape:
                raise ValueError(
                    f"Baseline shape {baseline.shape} does not match data shape "
                    f"{y_arr.shape}."
                )
            baseline_corrected = (y_arr - baseline).astype(float)
        except Exception as e:
            print(f"Row {idx}: Error computing baseline: {e}")
            print(f" - Baseline Function: {baseline_name}")
            print(f" - Baseline Parameters: {params}")
            skipped += 1
            continue

        # Save results back to DataFrame
        baseline = np.asarray(baseline, dtype=float)
        baseline_corrected = np.asarray(baseline_corrected, dtype=float)
        if baseline.ndim > 1:
            baseline = baseline.flatten()
        if baseline_corrected.ndim > 1:
            baseline_corrected = baseline_corrected.flatten()
        # Store as plain Python lists for portability/CSV round-trip
        FTIR_DataFrame.at[idx, "Baseline"] = baseline.tolist()
        FTIR_DataFrame.at[idx, "Baseline-Corrected Data"] = baseline_corrected.tolist()
        updated += 1

    return FTIR_DataFrame


def plot_grouped_spectra(
    FTIR_DataFrame,
    materials,
    conditions,
    times,
    raw_data=True,
    baseline=False,
    baseline_corrected=False,
    normalized=False,
    separate_plots=False,
    include_replicates=True,
    zoom=None,
):
    """
    Plot grouped spectra based on material, condition, and time.

    Accepts lists or 'any' for each category.

    Parameters
    ----------
    FTIR_DataFrame : pd.DataFrame
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
    normalized : bool, optional
        Whether to plot the normalized-and-corrected data from the column
        'Normalized and Corrected Data' (default is False).
    separate_plots : bool, optional
        Whether to create separate plots for each spectrum (default is False).
    include_replicates : bool, optional
        Whether to include all replicates or just the first of each group (default is
        True).
    zoom : str, optional
        A string specifying the x-axis zoom range in the format "min-max" (e.g.,
        "400-4000").

    Returns
    -------
    None
    """

    # Parse comma-separated strings into lists, handle 'any' (case-insensitive)
    mask = pd.Series([True] * len(FTIR_DataFrame))
    if isinstance(materials, str) and materials.strip().lower() != "any":
        material_list = [m.strip() for m in materials.split(",") if m.strip()]
        mask &= FTIR_DataFrame["Material"].isin(material_list)
    if isinstance(conditions, str) and conditions.strip().lower() != "any":
        condition_list = [c.strip() for c in conditions.split(",") if c.strip()]
        mask &= FTIR_DataFrame["Conditions"].isin(condition_list)
    if isinstance(times, str) and times.strip().lower() != "any":
        # Try to convert to int if possible, else keep as string
        time_list = []
        for t in times.split(","):
            t = t.strip()
            if t:
                try:
                    time_list.append(int(t))
                except ValueError:
                    time_list.append(t)
        mask &= FTIR_DataFrame["Time"].isin(time_list)
    filtered_data = FTIR_DataFrame[mask]

    # If not including replicates, keep only the first member of each (Material,
    # Conditions, Time) group
    if not include_replicates:
        filtered_data = filtered_data.sort_values(by=["Material", "Conditions", "Time"])
        filtered_data = filtered_data.drop_duplicates(
            subset=["Material", "Conditions", "Time"], keep="first"
        )

    # Sort by time once for both legend and plotting (assume all times are integers)
    filtered_data_sorted = filtered_data.sort_values(by="Time")
    x_axis_col = "X-Axis" if "X-Axis" in filtered_data_sorted.columns else "Wavelength"

    # Plot all together (legend in time order) and record colors for each spectrum
    # (including replicates)
    plt.figure(figsize=(10, 6))
    legend_entries = []
    color_map = {}  # Map from (data type, DataFrame index) to color
    legend_filepaths = []  # List of filepaths in legend order
    for idx, spectrum_row in filtered_data_sorted.iterrows():
        material_val = spectrum_row.get("Material", "")
        condition_val = spectrum_row.get(
            "Conditions", spectrum_row.get("Condition", "")
        )
        time_val = spectrum_row.get("Time", "")
        spectrum_label = f"{material_val}, {condition_val}, {time_val}"
        # Use ast.literal_eval for x_axis and data columns if they are strings
        x_axis = spectrum_row.get(x_axis_col)
        if isinstance(x_axis, str):
            try:
                x_axis = ast.literal_eval(x_axis)
            except Exception:
                pass
        file_path = os.path.join(
            spectrum_row["File Location"], spectrum_row["File Name"]
        )
        if raw_data and "Raw Data" in spectrum_row:
            y_data = spectrum_row["Raw Data"]
            if isinstance(y_data, str):
                try:
                    y_data = ast.literal_eval(y_data)
                except Exception:
                    pass
            (line_handle,) = plt.plot(x_axis, y_data, label=f"Raw: {spectrum_label}")
            legend_entries.append((line_handle, f"Raw: {spectrum_label}"))
            color_map[("Raw", idx)] = line_handle.get_color()
            legend_filepaths.append(file_path)
        if (
            baseline
            and "Baseline" in spectrum_row
            and spectrum_row["Baseline"] is not None
        ):
            y_data = spectrum_row["Baseline"]
            if isinstance(y_data, str):
                try:
                    y_data = ast.literal_eval(y_data)
                except Exception:
                    pass
            if raw_data or baseline_corrected or normalized:
                (line_handle,) = plt.plot(
                    x_axis, y_data, "--", label=f"Baseline: {spectrum_label}"
                )
            else:
                (line_handle,) = plt.plot(
                    x_axis, y_data, label=f"Baseline: {spectrum_label}"
                )
            legend_entries.append((line_handle, f"Baseline: {spectrum_label}"))
            color_map[("Baseline", idx)] = line_handle.get_color()
            legend_filepaths.append(file_path)
        if (
            baseline_corrected
            and "Baseline-Corrected Data" in spectrum_row
            and spectrum_row["Baseline-Corrected Data"] is not None
        ):
            y_data = spectrum_row["Baseline-Corrected Data"]
            if isinstance(y_data, str):
                try:
                    y_data = ast.literal_eval(y_data)
                except Exception:
                    pass
            if raw_data or baseline or normalized:
                (line_handle,) = plt.plot(
                    x_axis,
                    y_data,
                    ":",
                    label=f"Baseline-Corrected: {spectrum_label}",
                )
            else:
                (line_handle,) = plt.plot(
                    x_axis, y_data, label=f"Baseline-Corrected: {spectrum_label}"
                )
            legend_entries.append(
                (line_handle, f"Baseline-Corrected: {spectrum_label}")
            )
            color_map[("Baseline-Corrected", idx)] = line_handle.get_color()
            legend_filepaths.append(file_path)
        if (
            normalized
            and "Normalized and Corrected Data" in spectrum_row
            and spectrum_row["Normalized and Corrected Data"] is not None
        ):
            y_data = spectrum_row["Normalized and Corrected Data"]
            if isinstance(y_data, str):
                try:
                    y_data = ast.literal_eval(y_data)
                except Exception:
                    pass
            if raw_data or baseline or baseline_corrected:
                (line_handle,) = plt.plot(
                    x_axis,
                    y_data,
                    "-.",
                    label=f"Normalized and Corrected: {spectrum_label}",
                )
            else:
                (line_handle,) = plt.plot(
                    x_axis, y_data, label=f"Normalized and Corrected: {spectrum_label}"
                )
            legend_entries.append(
                (line_handle, f"Normalized and Corrected: {spectrum_label}")
            )
            color_map[("Normalized and Corrected", idx)] = line_handle.get_color()
            legend_filepaths.append(file_path)
    handles = [entry[0] for entry in legend_entries]
    labels = [entry[1] for entry in legend_entries]
    # Print filepaths in legend order
    for fp in legend_filepaths:
        print(f"Plotting: {fp}")
    plt.title(
        f"Spectra for Material: {materials} | Condition: {conditions} | Time: {times}"
    )
    plt.xlabel("Wavelength (cm¯¹)")
    plt.ylabel("Absorbance (AU)")
    plt.legend(handles, labels)
    # Set zoom if provided
    if zoom is not None and isinstance(zoom, str):
        try:
            zoom_range = zoom.replace(" ", "").split("-")
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
            file_path = os.path.join(
                row.get("File Location", ""), row.get("File Name", "")
            )
            print(f"Plotting: {file_path}")
            material_val = row.get("Material", "")
            condition_val = row.get("Conditions", row.get("Condition", ""))
            time_val = row.get("Time", "")
            spectrum_label = f"{material_val}, {condition_val}, {time_val}"
            x_axis = row.get(x_axis_col)
            if isinstance(x_axis, str):
                try:
                    x_axis = ast.literal_eval(x_axis)
                except Exception:
                    pass
            plt.figure(figsize=(8, 5))
            if raw_data and "Raw Data" in row:
                y_data = row["Raw Data"]
                if isinstance(y_data, str):
                    try:
                        y_data = ast.literal_eval(y_data)
                    except Exception:
                        pass
                color = color_map.get(("Raw", idx), None)
                plt.plot(x_axis, y_data, label="Raw", color=color)
            if baseline and "Baseline" in row and row["Baseline"] is not None:
                y_data = row["Baseline"]
                if isinstance(y_data, str):
                    try:
                        y_data = ast.literal_eval(y_data)
                    except Exception:
                        pass
                color = color_map.get(("Baseline", idx), None)
                if raw_data or baseline_corrected or normalized:
                    plt.plot(x_axis, y_data, "--", label="Baseline", color=color)
                else:
                    plt.plot(x_axis, y_data, label="Baseline", color=color)
            if (
                baseline_corrected
                and "Baseline-Corrected Data" in row
                and row["Baseline-Corrected Data"] is not None
            ):
                y_data = row["Baseline-Corrected Data"]
                if isinstance(y_data, str):
                    try:
                        y_data = ast.literal_eval(y_data)
                    except Exception:
                        pass
                color = color_map.get(("Baseline-Corrected", idx), None)
                if raw_data or baseline or normalized:
                    plt.plot(
                        x_axis,
                        y_data,
                        ":",
                        label="Baseline-Corrected",
                        color=color,
                    )
                else:
                    plt.plot(x_axis, y_data, label="Baseline-Corrected", color=color)
            if (
                normalized
                and "Normalized and Corrected Data" in row
                and row["Normalized and Corrected Data"] is not None
            ):
                y_data = row["Normalized and Corrected Data"]
                if isinstance(y_data, str):
                    try:
                        y_data = ast.literal_eval(y_data)
                    except Exception:
                        pass
                color = color_map.get(("Normalized and Corrected", idx), None)
                if raw_data or baseline or baseline_corrected:
                    plt.plot(
                        x_axis,
                        y_data,
                        "-.",
                        label="Normalized and Corrected",
                        color=color,
                    )
                else:
                    plt.plot(
                        x_axis, y_data, label="Normalized and Corrected", color=color
                    )
            plt.title(f"Spectrum: {spectrum_label}")
            plt.xlabel("Wavelength (cm¯¹)")
            plt.ylabel("Absorbance (AU)")
            plt.legend()
            # Set zoom if provided
            if zoom is not None and isinstance(zoom, str):
                try:
                    zoom_range = zoom.replace(" ", "").split("-")
                    if len(zoom_range) == 2:
                        x_min, x_max = float(zoom_range[0]), float(zoom_range[1])
                        plt.xlim(x_min, x_max)
                except Exception as e:
                    print(f"Warning: Could not parse zoom argument '{zoom}': {e}")
            plt.show()


def try_baseline(
    FTIR_DataFrame,
    material=None,
    baseline_function=None,
    filepath=None,
):
    """
    Apply a modifiable baseline to a single spectrum from the DataFrame.

    Allows for on-the-fly parameter adjustments via interactive widgets and
    experimentation with different baseline functions.

    Parameters
    ----------
    FTIR_DataFrame (pd.DataFrame): The in-memory DataFrame containing all spectra.
    material (str, optional): Material name to analyze (ignored if filepath is
        provided).
    baseline_function (str): Baseline function to use ('ARPLS', 'IRSQR', 'FABC').
    filepath (str, optional): If provided, only process this file (by 'File Location'
        + 'File Name').

    Returns
    -------
    FTIR_DataFrame : pd.DataFrame
        The updated DataFrame with baseline corrections applied, if user chooses to save
        choices. Otherwise, the DataFrame remains unchanged.
    """
    if baseline_function is None:
        raise ValueError("Baseline function must be specified.")

    if filepath is not None:
        if os.path.sep in filepath:
            folder, fname = os.path.split(filepath)
            filtered = FTIR_DataFrame[
                (FTIR_DataFrame["File Location"] == folder)
                & (FTIR_DataFrame["File Name"] == fname)
            ]
        else:
            filtered = FTIR_DataFrame[FTIR_DataFrame["File Name"] == filepath]
        if filtered.empty:
            raise ValueError(f"No entry found for file '{filepath}'.")
        row = filtered.iloc[0]
        material = row.get("Material", "Unknown")
    else:
        if material is None:
            raise ValueError("Material must be specified if filepath is not provided.")
        filtered = FTIR_DataFrame[
            (FTIR_DataFrame["Material"] == material) & (FTIR_DataFrame["Time"] == 0)
        ]
        if filtered.empty:
            raise ValueError(
                f"No entry found for material '{material}' with time == 0."
            )
        row = filtered.iloc[0]

    x = (
        ast.literal_eval(row["X-Axis"])
        if isinstance(row["X-Axis"], str)
        else row["X-Axis"]
    )
    y = (
        ast.literal_eval(row["Raw Data"])
        if isinstance(row["Raw Data"], str)
        else row["Raw Data"]
    )
    y = np.array(y, dtype=float)

    parameters = _get_default_parameters(baseline_function)
    parameters = _cast_parameter_types(baseline_function, parameters)

    file_path = os.path.join(row["File Location"], row["File Name"])
    print(f"Plotting: {file_path}")

    # Widget setup for live parameter editing
    param_widgets = {}
    # Explicitly define widgets for each baseline function and parameter
    if baseline_function.upper() == "ARPLS":
        # lam: float, p: float, iterations: int
        param_widgets["lam"] = widgets.FloatSlider(
            value=parameters.get("lam", 1e5),
            min=1e4,
            max=1e6,
            step=1e4,
            description="Smoothness (lam)",
            readout_format=".1e",
            style={"description_width": "auto"},
        )
        param_widgets["diff_order"] = widgets.IntSlider(
            value=parameters.get("diff_order", 2),
            min=1,
            max=2,
            step=1,
            description="Differential Order",
            style={"description_width": "auto"},
        )
        param_widgets["max_iter"] = widgets.IntSlider(
            value=parameters.get("max_iter", 50),
            min=1,
            max=200,
            step=1,
            description="Max Iterations",
            style={"description_width": "auto"},
        )
        param_widgets["tol"] = widgets.FloatSlider(
            value=parameters.get("tol", 1e-3),
            min=1e-6,
            max=1e-1,
            step=1e-4,
            description="Tolerance",
            readout_format=".1e",
            style={"description_width": "auto"},
        )
    elif baseline_function.upper() == "IRSQR":
        # lam: float, quantile: float, num_knots: int, spline_degree: int, diff_order:
        # int, max_iterations: int, tolerance: float, eps: float
        param_widgets["lam"] = widgets.FloatSlider(
            value=parameters.get("lam", 1e6),
            min=1e5,
            max=1e7,
            step=1e5,
            description="Smoothness (lam)",
            readout_format=".1e",
            style={"description_width": "auto"},
        )
        param_widgets["quantile"] = widgets.FloatSlider(
            value=parameters.get("quantile", 0.05),
            min=0.001,
            max=0.5,
            step=0.001,
            description="Quantile",
            readout_format=".3f",
            style={"description_width": "auto"},
        )
        param_widgets["num_knots"] = widgets.IntSlider(
            value=parameters.get("num_knots", 100),
            min=5,
            max=500,
            step=5,
            description="Knots",
            style={"description_width": "auto"},
        )
        param_widgets["spline_degree"] = widgets.IntSlider(
            value=parameters.get("spline_degree", 3),
            min=1,
            max=5,
            step=1,
            description="Spline Degree",
            style={"description_width": "auto"},
        )
        param_widgets["diff_order"] = widgets.IntSlider(
            value=parameters.get("diff_order", 3),
            min=1,
            max=3,
            step=1,
            description="Differential Order",
            style={"description_width": "auto"},
        )
        param_widgets["max_iter"] = widgets.IntSlider(
            value=parameters.get("max_iter", 100),
            min=1,
            max=1000,
            step=1,
            description="Max Iterations",
            style={"description_width": "auto"},
        )
        param_widgets["tol"] = widgets.FloatSlider(
            value=parameters.get("tol", 1e-6),
            min=1e-10,
            max=1e-2,
            step=1e-7,
            description="Tolerance",
            readout_format=".1e",
            style={"description_width": "auto"},
        )
    elif baseline_function.upper() == "FABC":
        # lam: float, scale: int or None, num_std: float, diff_order: int, min_length:
        # int
        param_widgets["lam"] = widgets.FloatSlider(
            value=parameters.get("lam", 1e6),
            min=1e4,
            max=1e7,
            step=1e5,
            description="Smoothness (lam)",
            readout_format=".1e",
            style={"description_width": "auto"},
        )
        raw_data = (
            ast.literal_eval(row["Raw Data"])
            if isinstance(row["Raw Data"], str)
            else row["Raw Data"]
        )
        scale_default = ceil(optimize_window(raw_data) / 2)
        scale_val = parameters.get("scale", None)
        if scale_val is None:
            scale_val = scale_default
        param_widgets["scale"] = widgets.IntSlider(
            value=int(scale_val),
            min=2,
            max=500,
            step=1,
            description="Scale",
            style={"description_width": "auto"},
        )
        param_widgets["num_std"] = widgets.FloatSlider(
            value=parameters.get("num_std", 3.0),
            min=1.5,
            max=4.5,
            step=0.1,
            description="Standard Deviations",
            readout_format=".2f",
            style={"description_width": "auto"},
        )
        param_widgets["diff_order"] = widgets.IntSlider(
            value=parameters.get("diff_order", 2),
            min=1,
            max=3,
            step=1,
            description="Differential Order",
            style={"description_width": "auto"},
        )
        param_widgets["min_length"] = widgets.IntSlider(
            value=parameters.get("min_length", 2),
            min=1,
            max=6,
            step=1,
            description="Min Baseline Span Length",
            style={"description_width": "auto"},
        )

    output = widgets.Output()

    def _plot_baseline(**widget_params):
        # Merge widget params with any non-widget params
        param_vals = parameters.copy()
        param_vals.update(widget_params)
        param_vals = _cast_parameter_types(baseline_function, param_vals)
        with output:
            clear_output(wait=True)
            if baseline_function.upper() == "ARPLS":
                baseline_result = arpls(y, **param_vals)
            elif baseline_function.upper() == "IRSQR":
                baseline_result = irsqr(y, **param_vals, x_data=x)
            elif baseline_function.upper() == "FABC":
                baseline_result = fabc(y, **param_vals)
            else:
                print(f"Unknown baseline function: {baseline_function}")
                return

            # Handle return type: pybaselines returns (baseline, details_dict) or just
            # baseline
            if isinstance(baseline_result, tuple):
                baseline = baseline_result[0]
            elif isinstance(baseline_result, dict):
                # If a dict is returned, try to get 'baseline' key
                baseline = baseline_result.get("baseline", None)
                if baseline is None:
                    print("Error: Baseline function did not return a baseline array.")
                    return
            else:
                baseline = baseline_result

            # Ensure baseline and y are numpy arrays for subtraction
            baseline = np.asarray(baseline)
            y_arr = np.asarray(y)
            if baseline.shape != y_arr.shape:
                message = (
                    f"Error: Baseline shape {baseline.shape} does not match data"
                    f" shape {y_arr.shape}."
                )
                print(message)
                return
            baseline_corrected = y_arr - baseline

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
            ax1.plot(x, y, label="Raw Data")
            ax1.plot(x, baseline, label=f"{baseline_function} Baseline", linestyle="--")
            ax1.set_ylabel("Absorbance (AU)")
            ax1.set_title(f"{material}: Raw Data and {baseline_function} Baseline")
            ax1.legend()
            ax2.plot(
                x, baseline_corrected, label="Baseline-Corrected", color="tab:green"
            )
            ax2.set_xlabel("Wavenumber (cm¯¹)")
            ax2.set_ylabel("Absorbance (AU)")
            ax2.set_title("Baseline-Corrected")
            ax2.legend()
            plt.tight_layout()
            plt.show()

    # Create interactive widget
    if param_widgets:
        # For each parameter, create a row with the widget and its own reset button
        defaults = _get_default_parameters(baseline_function)
        widget_rows = []
        for key, widget in param_widgets.items():
            reset_btn = widgets.Button(
                description=f"Reset",
                button_style="info",
                layout=widgets.Layout(width="70px", margin="0 0 0 10px"),
            )
            # For 'scale', use the initial scale_val as the reset value
            if key == "scale":
                reset_value = int(scale_val)
            else:
                reset_value = defaults.get(key, widget.value)

            def make_reset_func(w, default_val):
                return lambda b: setattr(w, "value", default_val)

            reset_btn.on_click(make_reset_func(widget, reset_value))
            widget_row = widgets.HBox([widget, reset_btn])
            widget_rows.append(widget_row)

        # Add a 'Reset All' button beneath the individual reset buttons
        reset_all_btn = widgets.Button(
            description="Reset All",
            button_style="warning",
            layout=widgets.Layout(width="100px", margin="10px 10px 0 0"),
        )

        def reset_all_callback(b):
            for key, widget in param_widgets.items():
                if key == "scale":
                    widget.value = int(scale_val)
                elif key in defaults:
                    widget.value = defaults[key]

        reset_all_btn.on_click(reset_all_callback)

        # Helper to gather current parameter values (widgets + any non-widget ones)
        def _current_param_values():
            current = parameters.copy()
            for k, w in param_widgets.items():
                current[k] = w.value
            return _cast_parameter_types(baseline_function, current)

        # Save buttons to persist choices
        save_file_btn = widgets.Button(
            description="Save for file",
            button_style="success",
            layout=widgets.Layout(margin="10px 10px 0 0"),
        )
        save_material_btn = widgets.Button(
            description="Save for material",
            button_style="info",
            layout=widgets.Layout(margin="10px 10px 0 0"),
        )

        def _serialize_params(d):
            # Ensure plain Python types for safe storage/round-trip via ast.literal_eval
            def to_plain(v):
                try:

                    if isinstance(v, (np.integer,)):
                        return int(v)
                    if isinstance(v, (np.floating,)):
                        return float(v)
                    if isinstance(v, np.ndarray):
                        return v.tolist()
                except Exception:
                    pass
                return v

            return {k: to_plain(v) for k, v in d.items()}

        # Finalize routine: disable/close all widgets and clear outputs to avoid UID
        # errors
        def _finalize_and_close(container_widget=None):
            try:
                for w in param_widgets.values():
                    try:
                        w.disabled = True
                    except Exception:
                        pass
                for btn in (save_file_btn, save_material_btn, reset_all_btn):
                    try:
                        btn.disabled = True
                    except Exception:
                        pass
                # Close widgets to free comms (prevents duplicate UID issues)
                try:
                    for w in param_widgets.values():
                        w.close()
                except Exception:
                    pass
                try:
                    save_file_btn.close()
                    save_material_btn.close()
                    reset_all_btn.close()
                except Exception:
                    pass
                # Close interactive binding and its output widget if present
                try:
                    widget_func.close()
                except Exception:
                    pass
                try:
                    output.clear_output()
                    output.close()
                except Exception:
                    pass
                try:
                    plt.close("all")
                except Exception:
                    pass
                if container_widget is not None:
                    try:
                        container_widget.close()
                    except Exception:
                        pass
            finally:
                pass

        def on_save_for_file(b):
            param_vals = _serialize_params(_current_param_values())
            # Persist to the accessed row
            FTIR_DataFrame.at[row.name, "Baseline Function"] = str(
                baseline_function
            ).upper()
            FTIR_DataFrame.at[row.name, "Baseline Parameters"] = str(param_vals)
            # Print a summary of the action (do not auto-close)
            summary_out = widgets.Output()
            display(summary_out)
            with summary_out:
                try:
                    _file_path = os.path.join(
                        row.get("File Location", ""), row.get("File Name", "")
                    )
                except Exception:
                    _file_path = row.get("File Name", "")
                print("Saved baseline settings for this file:")
                print(f" - File: {_file_path} (row index: {row.name})")
                print(f" - Baseline Function: {str(baseline_function).upper()}")
                print(f" - Parameters: {param_vals}")

        def on_save_for_material(b):
            param_vals = _serialize_params(_current_param_values())
            mat_val = row.get("Material", material)
            if mat_val is None:
                mat_val = material
            mask = FTIR_DataFrame["Material"] == mat_val
            FTIR_DataFrame.loc[mask, "Baseline Function"] = str(
                baseline_function
            ).upper()
            FTIR_DataFrame.loc[mask, "Baseline Parameters"] = str(param_vals)
            # Print a summary of the action (do not auto-close)
            summary_out = widgets.Output()
            display(summary_out)
            with summary_out:
                try:
                    _count = int(mask.sum())
                except Exception:
                    _count = None
                print("Saved baseline settings for this material:")
                print(f" - Material: {mat_val}")
                if _count is not None:
                    print(f" - Rows updated: {_count}")
                print(f" - Baseline Function: {str(baseline_function).upper()}")
                print(f" - Parameters: {param_vals}")

        save_file_btn.on_click(on_save_for_file)
        save_material_btn.on_click(on_save_for_material)

        close_btn = widgets.Button(
            description="Close",
            button_style="danger",
            layout=widgets.Layout(margin="10px 0 0 0"),
        )
        controls_footer = widgets.HBox(
            [save_file_btn, save_material_btn, reset_all_btn, close_btn]
        )
        ui = widgets.VBox(widget_rows + [controls_footer])

        widget_func = widgets.interactive_output(
            _plot_baseline, {k: w for k, w in param_widgets.items()}
        )
        # Keep a handle to the displayed container so we can close it later
        container = widgets.HBox([ui, output])
        display(container)

        # Wire the Close button after container is created so it can be properly closed
        def on_close(b):
            try:
                plt.close("all")
            except Exception:
                pass
            _finalize_and_close(container_widget=container)

        close_btn.on_click(on_close)
    else:
        # No parameters to edit, just plot once
        _plot_baseline()
        # Provide a simple Close button to clear the plot output in this mode
        close_btn = widgets.Button(description="Close", button_style="danger")

        def _close_simple(b):
            try:
                plt.close("all")
            except Exception:
                pass

        close_btn.on_click(_close_simple)
        display(close_btn)


def test_baseline_choices(FTIR_DataFrame, material=None):
    """
    Plot three random spectra for a given material, showing baseline results.

    Plots raw data, baseline, and baseline-corrected data. The baseline function and
    parameters are taken from the DataFrame columns. Assumes user has already filled
    those columns earlier in the workflow.

    Parameters
    ----------
    FTIR_DataFrame : pd.DataFrame
        The DataFrame containing the spectral data.
    material : str
        The material to filter and plot.

    Returns
    -------
    None
    """
    if material is None:
        material = input(
            "Enter the material to test baseline and parameter choices for: "
        ).strip()
    # Filter for the specified material
    filtered = FTIR_DataFrame[FTIR_DataFrame["Material"] == material]
    if len(filtered) < 1:
        print(f"No rows found for material '{material}'.")
        return
    # Pick up to 3 random rows
    n = min(3, len(filtered))
    random_rows = filtered.sample(n=n, random_state=None)

    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]  # Make iterable for single row

    for i, (idx, row) in enumerate(random_rows.iterrows()):
        # Parse x and y data
        x = (
            ast.literal_eval(row["X-Axis"])
            if isinstance(row["X-Axis"], str)
            else row["X-Axis"]
        )
        y = (
            ast.literal_eval(row["Raw Data"])
            if isinstance(row["Raw Data"], str)
            else row["Raw Data"]
        )
        y = np.array(y, dtype=float)
        baseline_func = row.get("Baseline Function", None)
        # Robustly coerce parameters
        raw_params = row.get("Baseline Parameters", {})
        if isinstance(raw_params, dict):
            params = raw_params.copy()
        elif isinstance(raw_params, str) and raw_params.strip():
            try:
                maybe = ast.literal_eval(raw_params)
                params = (
                    maybe if isinstance(maybe, dict) else _parse_parameters(raw_params)
                )
            except Exception:
                params = _parse_parameters(raw_params)
        else:
            params = {}

        # Compute baseline
        baseline = None
        baseline_corrected = None
        try:
            if baseline_func is None:
                raise ValueError("No baseline function specified.")
            func = baseline_func.strip().upper()
            # Merge with defaults and cast types
            defaults = _get_default_parameters(func)
            params = {**defaults, **params}
            params = _cast_parameter_types(func, params)
            if func == "ARPLS":
                result = arpls(y, **params)
            elif func == "IRSQR":
                if "x_data" in params:
                    result = irsqr(y, **params)
                else:
                    result = irsqr(y, **params, x_data=x)
            elif func == "FABC":
                result = fabc(y, **params)
            elif func == "MANUAL":
                anchor_points = params.get("anchor_points", [])
                if not anchor_points:
                    raise ValueError("No anchor_points for MANUAL baseline.")
                anchor_indices = [
                    min(range(len(x)), key=lambda i: abs(x[i] - ap))
                    for ap in anchor_points
                ]
                y_anchor = [y[i] for i in anchor_indices]
                result = CubicSpline(x=anchor_points, y=y_anchor, extrapolate=True)(x)
            else:
                raise ValueError(f"Unknown baseline function: {baseline_func}")
            # Normalize return type to baseline array
            if isinstance(result, tuple):
                baseline = result[0]
            elif isinstance(result, dict):
                baseline = result.get("baseline", None)
                if baseline is None:
                    raise ValueError(
                        "Baseline function did not return a baseline array."
                    )
            else:
                baseline = result
            baseline = np.asarray(baseline, dtype=float)
            baseline_corrected = y - baseline
        except Exception as e:
            print(f"Error computing baseline for row {idx}: {e}")
            print(f" - Baseline Function: {baseline_func}")
            print(f" - Baseline Parameters: {params}")
            print(f" - X-Axis shape: {np.shape(x)}, Raw Data shape: {np.shape(y)}")
            baseline = np.full_like(y, np.nan)
            baseline_corrected = np.full_like(y, np.nan)

        # Plot raw and baseline
        ax0 = axes[i][0] if n > 1 else axes[0]
        ax0.plot(x, y, label="Raw Data")
        if baseline is not None:
            ax0.plot(x, baseline, "--", label="Baseline")
        ax0.set_title(f"{material} | File: {row['File Name']}")
        ax0.set_ylabel("Absorbance (AU)")
        ax0.legend()

        # Plot baseline-corrected
        ax1 = axes[i][1] if n > 1 else axes[1]
        if baseline_corrected is not None:
            ax1.plot(
                x, baseline_corrected, color="tab:green", label="Baseline-Corrected"
            )
        ax1.set_title("Baseline-Corrected")
        ax1.set_xlabel("Wavenumber (cm¯¹)")
        ax1.set_ylabel("Absorbance (AU)")
        ax1.legend()

    plt.tight_layout()
    plt.show()


def bring_in_DataFrame(DataFrame_path=None):
    """
    Load the CSV file into a pandas DataFrame.

    Allows for easy DataFrame manipulation in memory over the course of the analysis.

    Parameters
    ----------
    DataFrame_path : str
        The path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.
    """
    if DataFrame_path is None:
        DataFrame_path = "FTIR_DataFrame.csv"  # Default path if none is provided (will
        # be in active directory)
    else:
        pass
    if os.path.exists(DataFrame_path):
        FTIR_DataFrame = pd.read_csv(
            DataFrame_path
        )  # Load the DataFrame from the specified path
    else:
        FTIR_DataFrame = (
            pd.DataFrame()
        )  # Create a new empty DataFrame if it doesn't exist
    return FTIR_DataFrame, DataFrame_path


def anchor_points_selection(
    FTIR_DataFrame, material=None, filepath=None, try_it_out=True
):
    """
    Interactively select anchor points for FTIR baseline correction.

    Lets user select anchor points from a spectrum in the DataFrame for baseline
    correction. Anchor points are selected by clicking on the plot, and will apply to
    each file of that material. After selection, a cubic spline baseline is fit and
    previewed, and the user can accept or redo the selection.

    Parameters
    ----------
    FTIR_DataFrame : pd.DataFrame
        The DataFrame containing the spectral data.
    material : str, optional
        Material name to analyze (ignored if filepath is provided).
    filepath : str, optional
        If provided, only process this file (by 'File Location' + 'File Name').
    try_it_out : bool, optional
        If True, only prints the anchor points (default). If False, saves anchor points
        to 'Baseline Parameters' column for all rows with the same material.
    DataFrame_path : str, optional
        Path to save the DataFrame as CSV if anchor points are saved (used only if
        try_it_out is False).

    Returns
    -------
    None
        The selected anchor points are stored in the DataFrame under 'Baseline
        Parameters'.
    """
    SELECTED_ANCHOR_POINTS = []
    clear_output(wait=True)  # Forcibly reset output area before anything else

    # --- Data selection logic ---
    if filepath is not None:

        if os.path.sep in filepath:
            folder, fname = os.path.split(filepath)
            filtered = FTIR_DataFrame[
                (FTIR_DataFrame["File Location"] == folder)
                & (FTIR_DataFrame["File Name"] == fname)
            ]
        else:
            filtered = FTIR_DataFrame[FTIR_DataFrame["File Name"] == filepath]
        if filtered.empty:
            raise ValueError(f"No entry found for file '{filepath}'.")
        row = filtered.iloc[0]
    else:
        if material is None:
            raise ValueError("Material must be specified if filepath is not provided.")
        filtered = FTIR_DataFrame[
            (FTIR_DataFrame["Material"] == material) & (FTIR_DataFrame["Time"] == 0)
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
    close_button = widgets.Button(description="Close", button_style="danger")
    button_box = widgets.HBox([accept_button, redo_button, close_button])
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
        close2 = widgets.Button(description="Close", button_style="danger")
        button_box2 = widgets.HBox([accept2, redo2, close2])
        button_box2_out = widgets.Output()
        done2 = widgets.Output()

        def accept2_callback(b):
            global SELECTED_ANCHOR_POINTS
            SELECTED_ANCHOR_POINTS = sorted(anchor_points)
            with done2:
                clear_output()
                if try_it_out:
                    print(
                        f"DONE--Final selected anchor points: {SELECTED_ANCHOR_POINTS}"
                    )
                else:
                    mat = row["Material"]
                    for idx, r in FTIR_DataFrame.iterrows():
                        if r["Material"] == mat:
                            FTIR_DataFrame.at[idx, "Baseline Parameters"] = str(
                                SELECTED_ANCHOR_POINTS
                            )
                            FTIR_DataFrame.at[idx, "Baseline Function"] = "Manual"
                    message = (
                        f"Anchor points saved to Baseline Parameters and "
                        f"Baseline Function set to 'Manual' for material '{mat}'."
                    )
                    print(message)
            button_box2_out.clear_output()

        def redo2_callback(b):
            with done2:
                clear_output()
            reset_selection()

        accept2.on_click(accept2_callback)
        redo2.on_click(redo2_callback)

        def _close_preview(b):
            try:
                fig2.close()
            except Exception:
                pass

        close2.on_click(_close_preview)
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

    # Close handler for initial interactive view
    def _close_initial(b):
        try:
            scatter.on_click(None)
        except Exception:
            pass
        try:
            fig.close()
        except Exception:
            pass
        try:
            close_button.disabled = True
            accept_button.disabled = True
            redo_button.disabled = True
        except Exception:
            pass

    close_button.on_click(_close_initial)

    # --- Initial display ---
    clear_output(wait=True)  # Ensure output area is reset to avoid UID errors
    display(fig, output, button_box_out, done)
    with button_box_out:
        display(button_box)
    return None


def normalization_peak_selection(FTIR_DataFrame, material=None, filepath=None):
    """
    Interactively select and save a normalization peak range for FTIR spectra.

    Plots either a predefined specific file (via filepath) or the first time-zero
    file for a specified material. The user selects two points on the plot to define
    an x-range (wavenumber window) for normalization. The selected range is printed
    and can be saved to the DataFrame column 'Normalization Peak Wavenumber'.

    Parameters
    ----------
    FTIR_DataFrame : pd.DataFrame
        The DataFrame containing the spectral data.
    material : str, optional
        Material to visualize when filepath is not provided. The first Time == 0
        entry for this material will be shown.
    filepath : str, optional
        A specific file to visualize, provided as a full path or just filename.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame with the selected normalization peak range saved.
    """

    clear_output(wait=True)

    # Ensure destination column exists
    target_col = "Normalization Peak Wavenumber"
    if target_col not in FTIR_DataFrame.columns:
        FTIR_DataFrame[target_col] = None

    # --- Select a single row by filepath or by (material, Time == 0) ---
    if filepath is not None:
        if os.path.sep in filepath:
            folder, fname = os.path.split(filepath)
            filtered = FTIR_DataFrame[
                (FTIR_DataFrame["File Location"] == folder)
                & (FTIR_DataFrame["File Name"] == fname)
            ]
        else:
            filtered = FTIR_DataFrame[FTIR_DataFrame["File Name"] == filepath]
        if filtered.empty:
            raise ValueError(f"No entry found for file '{filepath}'.")
        row = filtered.iloc[0]
        material_name = row.get("Material", "Unknown")
    else:
        if material is None:
            raise ValueError("Material must be specified if filepath is not provided.")
        filtered = FTIR_DataFrame[
            (FTIR_DataFrame["Material"] == material) & (FTIR_DataFrame["Time"] == 0)
        ]
        if filtered.empty:
            raise ValueError(
                f"No entry found for material '{material}' with time == 0."
            )
        row = filtered.iloc[0]
        material_name = material

    # --- Extract data ---
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

    # --- Widgets and outputs ---
    save_spec_btn = widgets.Button(
        description="Save for this file", button_style="success"
    )
    save_mat_btn = widgets.Button(
        description="Save for this material", button_style="info"
    )
    redo_btn = widgets.Button(description="Redo", button_style="warning")
    cancel_btn = widgets.Button(description="Close", button_style="danger")
    btn_box = widgets.HBox([save_spec_btn, save_mat_btn, redo_btn, cancel_btn])
    info_out = widgets.Output()
    msg_out = widgets.Output()

    selected_points = []  # store up to two x positions
    shaded_shape_id = "norm_range_rect"
    first_vline_name = "norm_vline_first"  # visual indicator for first click

    file_path = os.path.join(row.get("File Location", ""), row.get("File Name", ""))
    with info_out:
        print(f"Plotting: {file_path}")
        print("Click two points to define the normalization range.")

    # --- Build interactive figure ---
    fig = go.FigureWidget(
        data=[
            go.Scatter(x=x_data, y=y_data, mode="lines", name="Raw Data"),
        ]
    )
    fig.update_layout(
        title=f"Select Normalization Range | Material: {material_name}",
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="Absorbance (AU)",
    )

    def _clear_selection_visuals():
        # remove all our shapes (simplest and robust)
        fig.layout.shapes = ()

    def _draw_first_click(x0: float):
        # dotted vertical line to show first click registered
        vline = dict(
            type="line",
            x0=x0,
            x1=x0,
            y0=min(y_data),
            y1=max(y_data),
            line=dict(color="red", dash="dot"),
            name=first_vline_name,
        )
        fig.add_shape(vline)

    def _draw_selection_visuals(x0, x1):
        # vertical lines
        vline1 = dict(
            type="line",
            x0=x0,
            x1=x0,
            y0=min(y_data),
            y1=max(y_data),
            line=dict(color="red", dash="dash"),
            name="norm_vline_1",
        )
        vline2 = dict(
            type="line",
            x0=x1,
            x1=x1,
            y0=min(y_data),
            y1=max(y_data),
            line=dict(color="red", dash="dash"),
            name="norm_vline_2",
        )
        # rectangle shading
        rect = dict(
            type="rect",
            x0=min(x0, x1),
            x1=max(x0, x1),
            y0=min(y_data),
            y1=max(y_data),
            fillcolor="rgba(0,128,0,0.15)",
            line=dict(width=0),
            layer="below",
            name=shaded_shape_id,
        )
        fig.add_shape(vline1)
        fig.add_shape(vline2)
        fig.add_shape(rect)

    def _on_click(trace, points, selector):
        if not points.xs:
            return
        x_val = float(points.xs[0])
        # If already have two, start over with new first point
        if len(selected_points) >= 2:
            selected_points.clear()
            _clear_selection_visuals()

        selected_points.append(x_val)

        if len(selected_points) == 1:
            # show immediate visual feedback for first click
            _clear_selection_visuals()
            _draw_first_click(x_val)
            with msg_out:
                clear_output(wait=True)
                print(f"First point set at x = {x_val:.3f} cm⁻¹. Click second point…")
        elif len(selected_points) == 2:
            x0, x1 = selected_points
            _clear_selection_visuals()
            _draw_selection_visuals(x0, x1)
            with msg_out:
                clear_output(wait=True)
                lo, hi = (min(x0, x1), max(x0, x1))
                print(f"Selected normalization range: [{lo:.3f}, {hi:.3f}] cm⁻¹")
        else:
            with msg_out:
                clear_output(wait=True)
                print(f"First point set at x = {x_val:.3f} cm⁻¹. Click second point…")

    fig.data[0].on_click(_on_click)

    def _current_range():
        if len(selected_points) != 2:
            return None
        x0, x1 = selected_points
        return [float(min(x0, x1)), float(max(x0, x1))]

    def _finalize_and_clear():
        # Detach callbacks and close interactive widgets so UI disappears
        try:
            fig.data[0].on_click(None)
        except Exception:
            pass
        # Close the plot first
        try:
            fig.close()
        except Exception:
            pass
        # Close the entire button box so no greyed-out buttons remain
        try:
            btn_box.close()
        except Exception:
            pass
        # Also close individual buttons defensively (in case they were displayed 
        # separately)
        for b in (save_spec_btn, save_mat_btn, redo_btn, cancel_btn):
            try:
                b.close()
            except Exception:
                pass

    def _save_for_this_spectrum(b):
        rng = _current_range()
        if rng is None:
            with msg_out:
                clear_output(wait=True)
                print("Please select two points before saving.")
            return
        FTIR_DataFrame.at[row.name, target_col] = str(rng)
        with msg_out:
            clear_output(wait=True)
            print(f"Saved normalization peak range {rng} for this spectrum.")

    def _save_for_this_material(b):
        rng = _current_range()
        if rng is None:
            with msg_out:
                clear_output(wait=True)
                print("Please select two points before saving.")
            return
        mat = row.get("Material", None)
        if mat is None:
            with msg_out:
                clear_output(wait=True)
                print("Row has no 'Material' value; cannot save for material.")
            return
        mask = FTIR_DataFrame["Material"] == mat
        FTIR_DataFrame.loc[mask, target_col] = str(rng)
        with msg_out:
            clear_output(wait=True)
            print(f"Saved normalization peak range {rng} for material '{mat}'.")

    def _redo(b):
        selected_points.clear()
        _clear_selection_visuals()
        with msg_out:
            clear_output()
            print("Selection cleared. Click two points to select a range.")

    def _close(b):
        _finalize_and_clear()

    save_spec_btn.on_click(_save_for_this_spectrum)
    save_mat_btn.on_click(_save_for_this_material)
    redo_btn.on_click(_redo)
    cancel_btn.on_click(_close)

    # --- Initial display ---
    display(fig, info_out, msg_out, btn_box)
    return FTIR_DataFrame


def spectrum_normalization(
    FTIR_DataFrame,
    material,
):
    """
    Normalize baseline-corrected spectra.

    For each spectrum of the given material, finds the maximum value within the
    selected normalization range and divides the entire spectrum by that value,
    making the local maximum equal to 1.

    Parameters
    ----------
    FTIR_DataFrame : pd.DataFrame
        DataFrame containing spectra.
    material : str
        Material to normalize.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with normalized values written to 'Normalized and Corrected
        Data'.
    """
    if FTIR_DataFrame is None:
        raise ValueError("FTIR_DataFrame must be loaded in.")
    if material is None or str(material).strip() == "":
        raise ValueError("material is required.")

    source_column = "Baseline-Corrected Data"
    dest_column = "Normalized and Corrected Data"
    range_column = "Normalization Peak Wavenumber"

    subset = FTIR_DataFrame[FTIR_DataFrame["Material"] == material]
    if subset.empty:
        raise ValueError(f"No rows found for material '{material}'.")

    # Determine x-axis column
    x_axis_column = "X-Axis" if "X-Axis" in FTIR_DataFrame.columns else None
    if x_axis_column is None:
        raise ValueError(
            f"No 'X-Axis' column found in FTIR_DataFrame. Ensure the DataFrame is "
            f"loaded correctly."
        )
    # Validate required columns for normalization
    _require_columns(
        FTIR_DataFrame,
        ["Material", source_column, range_column, x_axis_column],
        context="FTIR_DataFrame (spectrum_normalization)",
    )

    # Ensure source/destination columns are object dtype (hold per-row lists)
    if source_column in FTIR_DataFrame.columns:
        try:
            FTIR_DataFrame[source_column] = FTIR_DataFrame[source_column].astype(object)
        except Exception:
            pass
    # Ensure destination column exists and is object dtype
    if dest_column not in FTIR_DataFrame.columns:
        FTIR_DataFrame[dest_column] = None
    try:
        FTIR_DataFrame[dest_column] = FTIR_DataFrame[dest_column].astype(object)
    except Exception:
        pass

    # Normalize each spectrum by its own max within the normalization window
    updated = 0
    skipped = 0
    errors = []
    for idx, row in subset.iterrows():
        norm_range = row.get(range_column, None)
        if norm_range is None:
            skipped += 1
            errors.append((idx, "Missing normalization range"))
            continue
        # Parse normalization range
        if isinstance(norm_range, str):
            try:
                norm_range = ast.literal_eval(norm_range)
            except Exception:
                skipped += 1
                errors.append(
                    (idx, f"Could not parse normalization range string: {norm_range!r}")
                )
                continue
        if not isinstance(norm_range, (list, tuple)) or len(norm_range) != 2:
            skipped += 1
            errors.append(
                (
                    idx,
                    f"Normalization range must be a list/tuple of length 2. "
                    f"Got: {type(norm_range).__name__}",
                )
            )
            continue
        try:
            lo, hi = float(norm_range[0]), float(norm_range[1])
        except Exception:
            skipped += 1
            errors.append(
                (idx, f"Normalization range values must be numeric. Got: {norm_range}")
            )
            continue

        x = row.get(x_axis_column, None)
        y = row.get(source_column, None)
        if x is None or y is None:
            skipped += 1
            errors.append((idx, "Missing x-axis or baseline-corrected data"))
            continue
        if isinstance(x, str):
            try:
                x = ast.literal_eval(x)
            except Exception:
                skipped += 1
                errors.append((idx, "Could not parse x-axis string to sequence"))
                continue
        if isinstance(y, str):
            try:
                y = ast.literal_eval(y)
            except Exception:
                skipped += 1
                errors.append(
                    (idx, "Could not parse baseline-corrected data string to sequence")
                )
                continue
        try:
            x_arr = np.asarray(x, dtype=float)
            y_arr = np.asarray(y, dtype=float)
        except Exception:
            skipped += 1
            errors.append((idx, "x or y could not be coerced to numeric arrays"))
            continue
        if x_arr.shape[0] != y_arr.shape[0] or x_arr.ndim != 1:
            skipped += 1
            errors.append(
                (
                    idx,
                    f"Shape mismatch or non-1D arrays: x.shape={x_arr.shape}, "
                    f"y.shape={y_arr.shape}",
                )
            )
            continue

        lo_, hi_ = min(lo, hi), max(lo, hi)
        mask = (x_arr >= lo_) & (x_arr <= hi_)
        if not np.any(mask):
            skipped += 1
            errors.append(
                (idx, f"No x points within normalization range [{lo_}, {hi_}]")
            )
            continue

        local_peak = np.nanmax(y_arr[mask])
        if not np.isfinite(local_peak) or local_peak <= 0:
            skipped += 1
            errors.append(
                (idx, f"Local peak is not finite/positive within range: {local_peak}")
            )
            continue

        # Scale this spectrum so its max in the range becomes 1 and write to DataFrame
        y_scaled = (y_arr / local_peak).astype(float).tolist()
        FTIR_DataFrame.at[idx, dest_column] = y_scaled
        updated += 1

    print(
        f"Normalized material '{material}': updated {updated} spectra; skipped "
        f"{skipped} (missing/invalid range or data). "
        f"Each spectrum scaled by its own peak within the selected range."
    )
    if updated == 0 and skipped > 0 and errors:
        examples = "; ".join([f"row {i}: {reason}" for i, reason in errors[:5]])
        more = "" if len(errors) <= 5 else f" (and {len(errors) - 5} more)"
        raise ValueError(
            f"Normalization failed for material '{material}': no spectra updated. "
            f"Examples: {examples}{more}"
        )
    return FTIR_DataFrame


def find_peak_info(FTIR_DataFrame, materials=None, filepath=None):
    """
    Interactive peak finder for normalized and baseline-corrected spectra.

    - Uses scipy.signal.find_peaks on 'Normalized and Corrected Data'.
    - Checkboxes enable up to 3 independent X-range sliders; peaks are found in the
    union of enabled ranges.
    - Displays a live-updating plot with user-adjustable parameters.
    - Saves results (lists) to 'Peak Wavenumbers' and 'Peak Absorbances' columns.

    Parameters
    ----------
    FTIR_DataFrame : pd.DataFrame
        The DataFrame containing FTIR spectral data.
    materials : list[str] | str | None
        Materials to include; if str, comma-separated is accepted. Ignored if filepath
        is provided.
    filepath : str | None
        Specific file path to filter by (exact match). If provided, overrides materials.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame (in-place modifications also applied).
    """
    if FTIR_DataFrame is None or len(FTIR_DataFrame) == 0:
        raise ValueError("FTIR_DataFrame must be loaded and non-empty.")

    # Build filtered set
    if filepath is not None:
        filtered = FTIR_DataFrame[FTIR_DataFrame["File Path"] == filepath]
        if filtered.empty:
            raise ValueError(f"No rows found for filepath '{filepath}'.")
    elif materials is not None:
        if isinstance(materials, str):
            mats = [m.strip() for m in materials.split(",") if m.strip()]
        else:
            mats = [str(m).strip() for m in materials]
        filtered = FTIR_DataFrame[FTIR_DataFrame["Material"].isin(mats)]
        if filtered.empty:
            raise ValueError(f"No rows found for materials: {mats}.")
    else:
        raise ValueError("Either 'materials' or 'filename' must be provided.")

    # Ensure destination columns exist and are object dtype
    for col in ("Peak Wavenumbers", "Peak Absorbances"):
        if col not in FTIR_DataFrame.columns:
            FTIR_DataFrame[col] = None
        try:
            FTIR_DataFrame[col] = FTIR_DataFrame[col].astype(object)
        except Exception:
            pass

    def _parse_seq(val):
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except Exception:
                return None
        return val

    # Build spectrum options (index -> label)
    options = []
    for idx, r in filtered.iterrows():
        label = (
            f"{r.get('Material','')} | {r.get('Conditions', r.get('Condition',''))}"
            f" | T={r.get('Time','')} | {r.get('File Name','')}"
        )
        options.append((label, idx))
    if not options:
        raise ValueError("No spectra available after filtering.")

    # Seed from first spectrum
    first_idx = options[0][1]
    x0 = _parse_seq(FTIR_DataFrame.loc[first_idx].get("X-Axis"))
    y0 = _parse_seq(FTIR_DataFrame.loc[first_idx].get("Normalized and Corrected Data"))
    if x0 is None or y0 is None:
        raise ValueError(
            "Selected spectrum is missing 'X-Axis' or 'Normalized and Corrected Data'."
        )
    x0 = np.asarray(x0, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    xmin, xmax = (float(np.nanmin(x0)), float(np.nanmax(x0)))

    # Widgets
    spectrum_sel = widgets.Dropdown(
        options=options,
        value=first_idx,
        description="Spectrum",
        layout=widgets.Layout(width="70%"),
    )
    # Up to three optional X-range selectors, each gated by a checkbox
    step_val = (xmax - xmin) / 1000 or 1.0
    use_r1 = widgets.Checkbox(value=True, description="Use range 1")
    x_range1 = widgets.FloatRangeSlider(
        value=[xmin, xmax],
        min=xmin,
        max=xmax,
        step=step_val,
        description="X-range 1",
        continuous_update=False,
        readout_format=".1f",
        layout=widgets.Layout(width="90%"),
        disabled=not use_r1.value,
    )
    use_r2 = widgets.Checkbox(value=False, description="Use range 2")
    x_range2 = widgets.FloatRangeSlider(
        value=[xmin, xmax],
        min=xmin,
        max=xmax,
        step=step_val,
        description="X-range 2",
        continuous_update=False,
        readout_format=".1f",
        layout=widgets.Layout(width="90%"),
        disabled=not use_r2.value,
    )
    use_r3 = widgets.Checkbox(value=False, description="Use range 3")
    x_range3 = widgets.FloatRangeSlider(
        value=[xmin, xmax],
        min=xmin,
        max=xmax,
        step=step_val,
        description="X-range 3",
        continuous_update=False,
        readout_format=".1f",
        layout=widgets.Layout(width="90%"),
        disabled=not use_r3.value,
    )
    prominence = widgets.FloatSlider(
        value=0.05,
        min=0.0,
        max=1.0,
        step=0.005,
        description="Prominence",
        readout_format=".3f",
        continuous_update=False,
        style={"description_width": "auto"},
    )
    min_height = widgets.FloatSlider(
        value=0.0,
        min=0.01,
        max=1.0,
        step=0.01,
        description="Min height",
        readout_format=".2f",
        continuous_update=False,
        style={"description_width": "auto"},
    )
    distance = widgets.IntSlider(
        value=5,
        min=1,
        max=250,
        step=1,
        description="Min separation",
        continuous_update=False,
        style={"description_width": "auto"},
    )
    width = widgets.IntSlider(
        value=1,
        min=1,
        max=50,
        step=1,
        description="Min width",
        continuous_update=False,
        style={"description_width": "auto"},
    )
    max_peaks = widgets.IntSlider(
        value=10,
        min=1,
        max=25,
        step=1,
        description="Max peaks",
        continuous_update=False,
        style={"description_width": "auto"},
    )

    save_file_btn = widgets.Button(description="Save for file", button_style="success")
    save_all_btn = widgets.Button(description="Save for filtered", button_style="info")
    close_btn = widgets.Button(description="Close", button_style="danger")
    msg_out = widgets.Output()

    # Plotly figure
    fig = go.FigureWidget()
    fig.add_scatter(x=x0.tolist(), y=y0.tolist(), mode="lines", name="Normalized and Corrected")
    fig.add_scatter(
        x=[],
        y=[],
        mode="markers",
        name="Peaks",
        marker=dict(color="red", size=9, symbol="x"),
    )
    fig.update_layout(
        title="Peak Selection (live)",
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="Absorbance (AU)",
    )

    def _get_xy(row_idx):
        r = FTIR_DataFrame.loc[row_idx]
        x = _parse_seq(r.get("X-Axis"))
        y = _parse_seq(r.get("Normalized and Corrected Data"))
        if x is None or y is None:
            return None, None
        try:
            x_arr = np.asarray(x, dtype=float)
            y_arr = np.asarray(y, dtype=float)
        except Exception:
            return None, None
        if x_arr.ndim != 1 or y_arr.ndim != 1 or x_arr.shape[0] != y_arr.shape[0]:
            return None, None
        return x_arr, y_arr

    def _compute_peaks_for_ranges(x_arr, y_arr, ranges):
        # Build a combined mask for all enabled ranges
        if not ranges:
            return np.array([], dtype=int), np.array([], dtype=float)
        mask = np.zeros(x_arr.shape[0], dtype=bool)
        for x_min, x_max in ranges:
            if x_min is None or x_max is None:
                continue
            lo, hi = (float(min(x_min, x_max)), float(max(x_min, x_max)))
            mask |= (x_arr >= lo) & (x_arr <= hi)
        if not np.any(mask):
            return np.array([], dtype=int), np.array([], dtype=float)
        y_sub = y_arr[mask]
        idx_sub = np.where(mask)[0]
        kwargs = {
            "prominence": (
                float(prominence.value) if prominence.value is not None else None
            ),
            "distance": int(distance.value) if distance.value is not None else None,
            "width": int(width.value) if width.value is not None else None,
        }
        if min_height.value and float(min_height.value) > 0:
            kwargs["height"] = float(min_height.value)
        peaks_local, _props = find_peaks(
            y_sub, **{k: v for k, v in kwargs.items() if v is not None}
        )
        if peaks_local.size == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        peaks_global = idx_sub[peaks_local]
        # limit to top-N by height if requested
        if (
            max_peaks.value
            and int(max_peaks.value) > 0
            and peaks_global.size > int(max_peaks.value)
        ):
            heights = y_arr[peaks_global]
            order = np.argsort(heights)[::-1][: int(max_peaks.value)]
            peaks_global = peaks_global[order]
        return peaks_global, y_arr[peaks_global]

    def _current_ranges():
        rs = []
        if use_r1.value:
            rs.append((x_range1.value[0], x_range1.value[1]))
        if use_r2.value:
            rs.append((x_range2.value[0], x_range2.value[1]))
        if use_r3.value:
            rs.append((x_range3.value[0], x_range3.value[1]))
        return rs

    def _update_plot(*args):
        idx = spectrum_sel.value
        x_arr, y_arr = _get_xy(idx)
        if x_arr is None:
            with msg_out:
                msg_out.clear_output()
                print("Selected spectrum missing or invalid normalized data.")
            return
        # Update traces
        with fig.batch_update():
            fig.data[0].x = x_arr.tolist()
            fig.data[0].y = y_arr.tolist()
        # Update bounds for each slider and enable/disable based on checkboxes
        x_min, x_max = float(np.nanmin(x_arr)), float(np.nanmax(x_arr))
        for cb, sl in ((use_r1, x_range1), (use_r2, x_range2), (use_r3, x_range3)):
            sl.min = x_min
            sl.max = x_max
            try:
                lo, hi = sl.value
            except Exception:
                lo, hi = x_min, x_max
            lo = max(x_min, min(lo, x_max))
            hi = max(lo, min(hi, x_max))
            sl.value = [lo, hi]
            sl.disabled = not cb.value
        # Peaks and shading across all enabled ranges
        ranges = _current_ranges()
        peaks_idx, peaks_y = _compute_peaks_for_ranges(x_arr, y_arr, ranges)
        with fig.batch_update():
            fig.data[1].x = x_arr[peaks_idx].tolist() if peaks_idx.size else []
            fig.data[1].y = peaks_y.tolist() if peaks_idx.size else []
            # Rebuild all range shapes deterministically for reliability with multiple 
            # ranges
            y0_min = float(np.nanmin(y_arr))
            y0_max = float(np.nanmax(y_arr))
            shapes_list = []
            for i_r, rng in enumerate(ranges):
                if rng is None:
                    continue
                lo_i, hi_i = float(min(rng)), float(max(rng))
                shapes_list.append(
                    dict(
                        type="rect",
                        x0=lo_i,
                        x1=hi_i,
                        y0=y0_min,
                        y1=y0_max,
                        fillcolor="rgba(0,128,0,0.12)",
                        line=dict(width=0),
                        layer="below",
                        name=f"range_rect_{i_r}",
                    )
                )
            # Assign shapes in one shot to avoid inconsistent state when multiple are 
            # active
            fig.layout.shapes = tuple(shapes_list)
        with msg_out:
            msg_out.clear_output()
            if not ranges:
                print("Enable at least one X-range to find peaks.")
            else:
                print(f"Peaks found: {len(peaks_idx)}")

    def _save_for_file(b):
        idx = spectrum_sel.value
        x_arr, y_arr = _get_xy(idx)
        if x_arr is None:
            with msg_out:
                msg_out.clear_output()
                print("Cannot save: selected spectrum missing normalized data.")
            return
        ranges = _current_ranges()
        if not ranges:
            with msg_out:
                msg_out.clear_output()
                print("Please enable at least one X-range before saving.")
            return
        peaks_idx, peaks_y = _compute_peaks_for_ranges(x_arr, y_arr, ranges)
        FTIR_DataFrame.at[idx, "Peak Wavenumbers"] = (
            x_arr[peaks_idx].astype(float).tolist()
        )
        FTIR_DataFrame.at[idx, "Peak Absorbances"] = peaks_y.astype(float).tolist()
        with msg_out:
            msg_out.clear_output()
            print(
                f"Saved {len(peaks_idx)} peaks for file "
                f"'{FTIR_DataFrame.loc[idx, 'File Name']}'."
            )

    def _save_for_filtered(b):
        ranges = _current_ranges()
        if not ranges:
            with msg_out:
                msg_out.clear_output()
                print("Please enable at least one X-range before saving.")
            return
        updated, skipped = 0, 0
        for idx, _row in filtered.iterrows():
            x_arr, y_arr = _get_xy(idx)
            if x_arr is None:
                skipped += 1
                continue
            peaks_idx, peaks_y = _compute_peaks_for_ranges(x_arr, y_arr, ranges)
            FTIR_DataFrame.at[idx, "Peak Wavenumbers"] = (
                x_arr[peaks_idx].astype(float).tolist()
            )
            FTIR_DataFrame.at[idx, "Peak Absorbances"] = peaks_y.astype(float).tolist()
            updated += 1
        with msg_out:
            msg_out.clear_output()
            print(
                f"Updated {updated} spectra; skipped {skipped} (missing/invalid data)."
            )

    def _close_ui(b):
        try:
            spectrum_sel.close()
            x_range1.close()
            x_range2.close()
            x_range3.close()
            use_r1.close()
            use_r2.close()
            use_r3.close()
            prominence.close()
            min_height.close()
            distance.close()
            width.close()
            max_peaks.close()
            save_file_btn.close()
            save_all_btn.close()
            close_btn.close()
            # Leave msg_out displayed so the last messages remain visible
            fig.close()
        except Exception:
            pass

    # Wire events
    spectrum_sel.observe(_update_plot, names="value")
    for w in (x_range1, x_range2, x_range3, use_r1, use_r2, use_r3):
        w.observe(_update_plot, names="value")
    prominence.observe(_update_plot, names="value")
    min_height.observe(_update_plot, names="value")
    distance.observe(_update_plot, names="value")
    width.observe(_update_plot, names="value")
    max_peaks.observe(_update_plot, names="value")
    save_file_btn.on_click(_save_for_file)
    save_all_btn.on_click(_save_for_filtered)
    close_btn.on_click(_close_ui)

    controls_row1 = widgets.HBox([spectrum_sel])
    controls_row2 = widgets.HBox([use_r1, x_range1])
    controls_row3 = widgets.HBox([use_r2, x_range2])
    controls_row4 = widgets.HBox([use_r3, x_range3])
    controls_row5 = widgets.HBox([prominence, min_height, distance])
    controls_row6 = widgets.HBox(
        [width, max_peaks, save_file_btn, save_all_btn, close_btn]
    )
    ui = widgets.VBox(
        [
            controls_row1,
            controls_row2,
            controls_row3,
            controls_row4,
            controls_row5,
            controls_row6,
        ]
    )

    display(ui, fig, msg_out)
    _update_plot()

    return FTIR_DataFrame


def peak_deconvolution(FTIR_DataFrame, filepath=None):
    """
    Interactively deconvolute found peaks for area analysis.

    Takes the peak info from find_peak_info and utilizes a Pseudo-Voigt model to
    approximately model the peak components as a linear combination of Gaussian and
    Lorentzian distributions. Allows for live changing of the Gaussian-Lorentzian
    fraction parameter for each peak.

    Parameters
    ----------
    FTIR_DataFrame : pd.DataFrame
        The DataFrame containing FTIR spectral data.
    filepath : str | None
        Specific file path to filter by (exact match). If provided, limits the list to
        that file.

    Returns
    -------
    None (update later for Json filling)
    """

    try:
        _lmfit_models = importlib.import_module("lmfit.models")
        PseudoVoigtModel = getattr(_lmfit_models, "PseudoVoigtModel")
    except Exception as e:
        raise ImportError(
            "lmfit is required for peak_deconvolution. Please install it (e.g., pip "
            "install lmfit)."
        ) from e

    if FTIR_DataFrame is None or len(FTIR_DataFrame) == 0:
        raise ValueError("FTIR_DataFrame must be loaded and non-empty.")

    # Initial selection
    if filepath is not None:
        # Match by full path (File Location + File Name) or by just filename
        if os.path.sep in str(filepath):
            file_dir, file_name = os.path.split(str(filepath))
            filtered = FTIR_DataFrame[
                (FTIR_DataFrame.get("File Location", "") == file_dir)
                & (FTIR_DataFrame.get("File Name", "") == file_name)
            ]
        else:
            filtered = FTIR_DataFrame[
                FTIR_DataFrame.get("File Name", "") == str(filepath)
            ]
        if filtered.empty:
            raise ValueError(f"No rows found for filepath '{filepath}'.")
    else:
        # Start with full DataFrame; Material/Conditions dropdowns will constrain
        filtered = FTIR_DataFrame.copy()

    # Ensure destination column exists for saving results
    results_col = "Deconvolution Results"
    if results_col not in FTIR_DataFrame.columns:
        FTIR_DataFrame[results_col] = None
    try:
        FTIR_DataFrame[results_col] = FTIR_DataFrame[results_col].astype(object)
    except Exception:
        pass

    def _parse_seq(val):
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except Exception:
                return None
        return val

    # Build spectrum options (only include rows with normalized data available)
    options = []
    for idx, r in filtered.iterrows():
        try:
            norm_val = r.get("Normalized and Corrected Data", None)
            # Skip rows without normalized data
            if pd.isna(norm_val) if "pd" in globals() else (norm_val is None):
                continue
        except Exception:
            if r.get("Normalized and Corrected Data", None) is None:
                continue
        label = (
            f"{r.get('Material','')} | {r.get('Conditions', r.get('Condition',''))}"
            f" | T={r.get('Time','')} | {r.get('File Name','')}"
        )
        options.append((label, idx))
    if not options:
        raise ValueError("No spectra available after filtering.")

    # Seed from first spectrum
    first_idx = options[0][1]
    x0 = _parse_seq(FTIR_DataFrame.loc[first_idx].get("X-Axis"))
    y0 = _parse_seq(FTIR_DataFrame.loc[first_idx].get("Normalized and Corrected Data"))
    if x0 is None or y0 is None:
        raise ValueError(
            "Selected spectrum is missing 'X-Axis' or 'Normalized and Corrected Data'."
        )
    x0 = np.asarray(x0, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    xmin, xmax = (float(np.nanmin(x0)), float(np.nanmax(x0)))

    # Widgets (spectrum and global fit controls; full x-range is always used)
    spectrum_sel = widgets.Dropdown(
        options=options,
        value=first_idx,
        description="Spectrum",
        layout=widgets.Layout(width="70%"),
    )
    # Build initial material/conditions dropdowns for interactive filtering
    try:
        norm_mask_init = filtered["Normalized and Corrected Data"].notna()
        filterable_df = filtered[norm_mask_init]
    except Exception:
        filterable_df = filtered
    unique_materials = (
        sorted(
            {
                str(v)
                for v in filterable_df.get("Material", pd.Series([], dtype=object))
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            }
        )
        if "Material" in filterable_df.columns
        else []
    )
    # conditions might be under 'Conditions' or 'Condition'
    cond_series = (
        filterable_df["Conditions"]
        if "Conditions" in filterable_df.columns
        else (
            filterable_df["Condition"]
            if "Condition" in filterable_df.columns
            else pd.Series([], dtype=object)
        )
    )
    # Build conditions list excluding 'unexposed' (case-insensitive)
    _all_conditions = [
        str(v) for v in cond_series.dropna().astype(str).unique().tolist()
    ]
    unique_conditions = sorted(
        [c for c in _all_conditions if c.strip().lower() != "unexposed"]
    )
    material_dd = widgets.Dropdown(
        options=["any"] + unique_materials,
        value="any",
        description="Material",
        layout=widgets.Layout(width="40%"),
    )
    conditions_dd = widgets.Dropdown(
        options=["any"] + unique_conditions,
        value="any",
        description="Conditions",
        layout=widgets.Layout(width="40%"),
    )
    center_window = widgets.FloatSlider(
        value=15.0,
        min=1.0,
        max=50.0,
        step=1.0,
        description="Center ±window (cm⁻¹)",
        continuous_update=False,
        style={"description_width": "auto"},
        readout_format=".0f",
    )
    init_sigma = widgets.FloatSlider(
        value=10.0,
        min=1.0,
        max=100.0,
        step=0.5,
        description="Init sigma (cm⁻¹)",
        continuous_update=False,
        style={"description_width": "auto"},
        readout_format=".1f",
    )
    # Defaults for reset operations
    DEFAULT_ALPHA = 0.5
    DEFAULT_INCLUDE = True
    default_center_window_value = float(center_window.value)
    default_init_sigma_value = float(init_sigma.value)
    # Add peaks workflow controls
    add_peaks_btn = widgets.Button(
        description="Add peaks",
        tooltip="Click, then click on the plot to add one or more peaks",
        button_style="primary",
        layout=widgets.Layout(width="110px"),
    )
    accept_new_peaks_btn = widgets.Button(
        description="Accept new peaks",
        button_style="success",
        layout=widgets.Layout(width="150px"),
    )
    # Iterative correction button to minimize reduced chi-square via coordinate descent
    iter_btn = widgets.Button(
        description="Iteratively correct",
        button_style="info",
        layout=widgets.Layout(width="175px"),
        tooltip=(
            "Adjust α, center window, and init σ to reduce reduced chi-square."
        ),
    )
    cancel_fit_btn = widgets.Button(
        description="Cancel Fit",
        button_style="danger",
        layout=widgets.Layout(width="120px"),
        tooltip="Interrupt the current fit or iterative correction",
    )
    redo_new_peaks_btn = widgets.Button(
        description="Redo new peaks",
        button_style="warning",
        layout=widgets.Layout(width="140px"),
    )
    cancel_new_peaks_btn = widgets.Button(
        description="Cancel peak addition",
        button_style="danger",
        layout=widgets.Layout(width="190px"),
    )
    save_btn = widgets.Button(description="Save for file", button_style="success")
    close_btn = widgets.Button(description="Close", button_style="danger")
    # Dedicated status label to avoid Output-widget buffering issues
    status_html = widgets.HTML(value="")
    msg_out = widgets.Output()
    # Reset buttons for globals and all
    reset_center_btn = widgets.Button(
        description="Reset",
        tooltip="Reset Center ±window to default",
        layout=widgets.Layout(width="80px"),
    )
    reset_sigma_btn = widgets.Button(
        description="Reset",
        tooltip="Reset Init sigma to default",
        layout=widgets.Layout(width="80px"),
    )
    reset_all_btn = widgets.Button(
        description="Reset all",
        button_style="warning",
        tooltip="Reset all sliders and selections to defaults",
        layout=widgets.Layout(width="120px"),
    )

    # Cancellation/interrupt support for long-running fits
    # Global cancellation token for long-running workflows (iteration).
    # Note: Do NOT replace this Event with a new instance elsewhere; it must remain
    # stable so a single set() call cancels the entire workflow.
    cancel_event = threading.Event()
    fit_thread = None
    iter_thread = None
    fit_cancel_token = None  # points to the active fit's cancel event, if any
    iterating_in_progress = False  # suppress per-fit status updates while iterating
    # Snapshot of reduced chi-square at the moment the user clicks Iterate
    iter_start_redchi = None
    # Final reduced chi-square once iteration completes and a flag to show a summary
    iter_final_redchi = None
    iter_summary_pending = False
    # Helper to marshal UI updates back onto the notebook's main IOLoop
    try:
        from tornado.ioloop import IOLoop as _IOLoop
    except Exception:
        _IOLoop = None
    # Capture the main thread's IOLoop now; using current() inside a worker thread
    # can create a new, non-running loop which would drop callbacks.
    try:
        _MAIN_IOLOOP = _IOLoop.current() if _IOLoop is not None else None
    except Exception:
        _MAIN_IOLOOP = None

    def _on_main_thread(fn, *args, **kwargs):
        try:
            if _MAIN_IOLOOP is not None:
                _MAIN_IOLOOP.add_callback(lambda: fn(*args, **kwargs))
                return
        except Exception:
            pass
        # Fallback: call directly (may work in some environments)
        try:
            fn(*args, **kwargs)
        except Exception:
            pass

    # Dynamic per-peak controls: include checkbox + alpha slider per peak
    alpha_sliders = []  # list[widgets.FloatSlider]
    include_checkboxes = []  # list[widgets.Checkbox]
    peak_controls_box = widgets.VBox([])

    # Persisted per-spectrum settings so switching spectra preserves choices
    per_spec_alpha = {}     # idx -> list[float]
    per_spec_include = {}   # idx -> list[bool]
    per_spec_globals = {}   # idx -> { 'center_window': float, 'init_sigma': float }
    # Guard to suppress redundant fits during bulk programmatic updates
    bulk_update_in_progress = False

    def _snapshot_current_controls():
        """Persist current UI control values for the active spectrum."""
        try:
            idx = spectrum_sel.value
        except Exception:
            return
        try:
            per_spec_alpha[idx] = [float(s.value) for s in alpha_sliders]
        except Exception:
            per_spec_alpha[idx] = []
        try:
            per_spec_include[idx] = [bool(cb.value) for cb in include_checkboxes]
        except Exception:
            per_spec_include[idx] = []
        try:
            per_spec_globals[idx] = {
                "center_window": float(center_window.value),
                "init_sigma": float(init_sigma.value),
            }
        except Exception:
            pass

    def _on_control_change(*_):
        """Snapshot controls and refit, unless in a bulk update."""
        _snapshot_current_controls()
        if bulk_update_in_progress:
            return
        _fit_and_update_plot()

    # Track last reduced chi-square per spectrum to report refit deltas
    last_redchi_by_idx = {}
    # Store last successful fit result per spectrum for Save action
    last_result_by_idx = {}

    # Plot figure: data, fit, components (dynamic)
    fig = go.FigureWidget()
    fig.add_scatter(x=x0.tolist(), y=y0.tolist(), mode="lines", name="Data (Norm+Corr)")
    fig.add_scatter(
        x=[], y=[], mode="lines", name="Composite Fit", line=dict(color="red")
    )
    fig.update_layout(
        title="Peak Deconvolution (Pseudo-Voigt)",
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="Absorbance (AU)",
    )

    # --- Add-peaks mode state ---
    adding_mode = False
    new_peak_xs = []  # temporary stash of user-clicked x positions (snapped to grid)

    def _hide(w):
        try:
            w.layout.display = "none"
        except Exception:
            pass

    def _show(w):
        try:
            w.layout.display = ""
        except Exception:
            pass

    # Show/Hide and enablement for the Cancel Fit button based on active work
    def _update_cancel_fit_visibility():
        active = False
        try:
            active = (
                (fit_thread is not None and fit_thread.is_alive())
                or (iter_thread is not None and iter_thread.is_alive())
            )
        except Exception:
            pass
        try:
            cancel_fit_btn.disabled = not active
        except Exception:
            pass
        if active:
            _show(cancel_fit_btn)
        else:
            _hide(cancel_fit_btn)

    # hide action buttons initially
    _hide(accept_new_peaks_btn)
    _hide(redo_new_peaks_btn)
    _hide(cancel_new_peaks_btn)
    _hide(cancel_fit_btn)

    def _clear_add_peak_shapes():
        # Remove only our temporary marker shapes
        try:
            shapes = list(getattr(fig.layout, "shapes", ()))
            shapes = [
                s for s in shapes if getattr(s, "name", None) != "add_peak_marker"
            ]
            fig.layout.shapes = tuple(shapes)
        except Exception:
            try:
                # When shapes are plain dicts
                shapes = list(getattr(fig.layout, "shapes", ()))
                new_shapes = []
                for s in shapes:
                    try:
                        if s.get("name") != "add_peak_marker":
                            new_shapes.append(s)
                    except Exception:
                        new_shapes.append(s)
                fig.layout.shapes = tuple(new_shapes)
            except Exception:
                pass

    # Click handler for adding peaks when in adding mode
    def _on_data_click(trace, points, selector):
        # Only respond when adding mode is ON and a valid click occurred
        try:
            nonlocal adding_mode
            if not adding_mode:
                return
            if not points or not getattr(points, "xs", None):
                return
            x_clicked = float(points.xs[0])
        except Exception:
            return

        # Get current spectrum arrays
        idx = spectrum_sel.value
        x_arr, y_arr = _get_xy(idx)
        if x_arr is None or y_arr is None or x_arr.size == 0:
            with msg_out:
                msg_out.clear_output()
                print("Cannot add peak: current spectrum has no normalized data.")
            return

        # Snap to nearest x
        try:
            nearest_i = int(np.argmin(np.abs(x_arr - x_clicked)))
        except Exception:
            with msg_out:
                msg_out.clear_output()
                print("Could not determine nearest x for the clicked location.")
            return
        x_new = float(x_arr[nearest_i])
        # Enforce proximity using Center ±window as minimum separation (customizable)
        try:
            min_sep = float(center_window.value)
        except Exception:
            min_sep = 0.0
        # Too close to a previously selected (session) peak?
        for existing_x in new_peak_xs:
            if abs(existing_x - x_new) <= min_sep:
                try:
                    status_html.value = (
                        f"<span style='color:#a00;'>Rejected: {x_new:.3f} cm⁻¹ is "
                        f"within ±{min_sep:.2f} cm⁻¹ of another selected peak "
                        f"({existing_x:.3f}). "
                        f"Tip: reduce the Center ±window to fit peaks in small "
                        f"spaces.</span>"
                    )
                except Exception:
                    with msg_out:
                        msg_out.clear_output()
                        print(
                            f"Rejected: {x_new:.3f} cm⁻¹ is within ±{min_sep:.2f} cm⁻¹ "
                            f"of another selected peak ({existing_x:.3f}). "
                            f"Tip: reduce the Center ±window to fit peaks in small "
                            f"spaces."
                        )
                return
        # Too close to an existing (committed) peak for this spectrum?
        xs_existing, _ys_existing = _get_peaks(idx)
        for xe in xs_existing:
            try:
                if abs(float(xe) - x_new) <= min_sep:
                    try:
                        status_html.value = (
                            f"<span style='color:#a00;'>Rejected: {x_new:.3f} cm⁻¹ is "
                            f"within ±{min_sep:.2f} cm⁻¹ of existing peak "
                            f"{float(xe):.3f}. "
                            f"Tip: reduce the Center ±window to fit peaks in small "
                            f"spaces.</span>"
                        )
                    except Exception:
                        with msg_out:
                            msg_out.clear_output()
                            print(
                                f"Rejected: {x_new:.3f} cm⁻¹ is within ±{min_sep:.2f} "
                                f"cm⁻¹ of existing peak {float(xe):.3f}. "
                                f"Tip: reduce the Center ±window to fit peaks in small"
                                f" spaces."
                            )
                    return
            except Exception:
                continue
        # Record and draw a vertical line marker
        new_peak_xs.append(x_new)
        y_min = float(np.nanmin(y_arr))
        y_max = float(np.nanmax(y_arr))
        try:
            fig.add_shape(
                dict(
                    type="line",
                    x0=x_new,
                    x1=x_new,
                    y0=y_min,
                    y1=y_max,
                    line=dict(color="#ff7f0e", dash="dot", width=1.5),
                    name="add_peak_marker",
                )
            )
        except Exception:
            pass
        with msg_out:
            msg_out.clear_output()
            print(
                f"Selected new peak at x = {x_new:.3f} cm⁻¹. Click more points, or "
                f"'Accept new peaks'."
            )

    # Always register the click handler; it checks the toggle state internally
    try:
        fig.data[0].on_click(_on_data_click)
    except Exception:
        pass

    def _get_xy(row_idx):
        r = FTIR_DataFrame.loc[row_idx]
        x = _parse_seq(r.get("X-Axis"))
        y = _parse_seq(r.get("Normalized and Corrected Data"))
        if x is None or y is None:
            return None, None
        try:
            x_arr = np.asarray(x, dtype=float)
            y_arr = np.asarray(y, dtype=float)
        except Exception:
            return None, None
        if x_arr.ndim != 1 or y_arr.ndim != 1 or x_arr.shape[0] != y_arr.shape[0]:
            return None, None
        return x_arr, y_arr

    def _get_peaks(row_idx):
        r = FTIR_DataFrame.loc[row_idx]
        xs = _parse_seq(r.get("Peak Wavenumbers"))
        ys = _parse_seq(r.get("Peak Absorbances"))
        if xs is None or ys is None:
            return [], []
        try:
            xs = list(xs)
            ys = list(ys)
        except Exception:
            return [], []
        if len(xs) != len(ys):
            return [], []
        # Sort peaks by wavenumber (ascending)
        try:
            pairs = sorted(zip(xs, ys), key=lambda t: float(t[0]))
            xs_sorted, ys_sorted = [list(t) for t in zip(*pairs)] if pairs else ([], [])
            return xs_sorted, ys_sorted
        except Exception:
            return xs, ys

    def _rebuild_alpha_sliders(row_idx):
        nonlocal alpha_sliders, include_checkboxes
        xs, ys = _get_peaks(row_idx)
        alpha_sliders = []
        include_checkboxes = []
        children = []
        if not xs:
            peak_controls_box.children = [
                widgets.HTML(
                    "<b>No peaks found for this spectrum.</b> Run find_peak_info first."
                )
            ]
            return
        saved_alphas = per_spec_alpha.get(row_idx)
        saved_includes = per_spec_include.get(row_idx)
        for i, (cx, cy) in enumerate(zip(xs, ys)):
            label = widgets.Label(
                value=f"Peak {i+1} @ {cx:.1f} cm⁻¹",
                layout=widgets.Layout(width="220px"),
            )
            cb = widgets.Checkbox(
                value=(
                    (saved_includes[i] if saved_includes is not None and i < len(saved_includes) else DEFAULT_INCLUDE)
                ),
                description="Include",
                indent=False,
                layout=widgets.Layout(width="100px"),
            )
            s = widgets.FloatSlider(
                value=(
                    float(saved_alphas[i]) if saved_alphas is not None and i < len(saved_alphas) else DEFAULT_ALPHA
                ),
                min=0.0,
                max=1.0,
                step=0.01,
                description="α",
                continuous_update=False,
                readout_format=".2f",
                style={"description_width": "auto"},
                layout=widgets.Layout(width="300px"),
            )
            # Per-slider reset button
            rb = widgets.Button(
                description="Reset",
                tooltip="Reset α to default (0.5)",
                layout=widgets.Layout(width="70px"),
            )

            def _make_reset_one(slider_ref):
                def _reset_one(_b=None):
                    nonlocal bulk_update_in_progress
                    bulk_update_in_progress = True
                    try:
                        slider_ref.value = DEFAULT_ALPHA
                    except Exception:
                        pass
                    _snapshot_current_controls()
                    bulk_update_in_progress = False
                    _fit_and_update_plot()
                return _reset_one

            rb.on_click(_make_reset_one(s))
            # Observe changes to snapshot + refit
            cb.observe(_on_control_change, names="value")
            s.observe(_on_control_change, names="value")
            include_checkboxes.append(cb)
            alpha_sliders.append(s)
            children.append(widgets.HBox([label, cb, s, rb]))
        peak_controls_box.children = children

    # --- Filtering helpers for material/conditions -> spectrum options ---
    def _row_condition_value(row):
        try:
            return row.get("Conditions", row.get("Condition", ""))
        except Exception:
            return ""

    def _rebuild_spectrum_options(*_):
        # Build candidate set from initial 'filtered' and drop rows without normalized
        # data
        try:
            cand = filtered.copy()
            cand = cand[cand["Normalized and Corrected Data"].notna()]
        except Exception:
            cand = filtered
        # Apply material filter
        sel_mat = material_dd.value if hasattr(material_dd, "value") else "any"
        if sel_mat != "any" and "Material" in cand.columns:
            cand = cand[cand["Material"].astype(str) == str(sel_mat)]
        # Apply conditions/condition filter; always include 'unexposed' rows as they
        # represent time-zero for every condition.
        sel_cond = conditions_dd.value if hasattr(conditions_dd, "value") else "any"
        if sel_cond != "any":

            def _cond_includes_unexposed(r):
                val = str(_row_condition_value(r))
                return (val == str(sel_cond)) or (val.strip().lower() == "unexposed")

            cand = cand[cand.apply(_cond_includes_unexposed, axis=1)]
        # Build options
        new_options = []
        for idx2, r2 in cand.iterrows():
            label = (
                f"{r2.get('Material','')} | "
                f"{r2.get('Conditions', r2.get('Condition',''))}"
                f" | T={r2.get('Time','')} | {r2.get('File Name','')}"
            )
            new_options.append((label, idx2))
        # Update dropdown
        prev_value = spectrum_sel.value
        spectrum_sel.options = new_options
        # Choose value: keep previous if still present; else first option
        valid_values = [v for (_lbl, v) in new_options]
        if prev_value in valid_values:
            spectrum_sel.value = prev_value
        elif valid_values:
            spectrum_sel.value = valid_values[0]
        else:
            # No spectra available, clear plot and controls
            with fig.batch_update():
                fig.data[0].x = []
                fig.data[0].y = []
                fig.data[1].x = []
                fig.data[1].y = []
            peak_controls_box.children = [
                widgets.HTML("<b>No spectra available after filtering.</b>")
            ]
            with msg_out:
                msg_out.clear_output()
                print("No spectra available after filtering selections.")
            return
        # Trigger downstream updates
        _on_spectrum_change()

    def _fit_and_update_plot(*_):
        nonlocal fit_thread, cancel_event, fit_cancel_token, iterating_in_progress

        idx = spectrum_sel.value
        x_arr, y_arr = _get_xy(idx)
        if x_arr is None:
            with msg_out:
                msg_out.clear_output()
                print("Selected spectrum missing normalized data.")
            return None

        peaks_x, peaks_y = _get_peaks(idx)
        if not peaks_x:
            with msg_out:
                msg_out.clear_output()
                print("No peaks found for this spectrum.")
            with fig.batch_update():
                fig.data[0].x = x_arr.tolist()
                fig.data[0].y = y_arr.tolist()
                fig.data[1].x = []
                fig.data[1].y = []
                fig.layout.shapes = ()
                while len(fig.data) > 2:
                    fig.data = tuple(fig.data[:2])
            return None

        # Determine which peaks are included
        included = [i for i, cb in enumerate(include_checkboxes) if cb.value]
        if len(included) == 0:
            with msg_out:
                msg_out.clear_output()
                print("No peaks selected. Enable one or more to fit.")
            with fig.batch_update():
                fig.data[0].x = x_arr.tolist()
                fig.data[0].y = y_arr.tolist()
                fig.data[1].x = []
                fig.data[1].y = []
                while len(fig.data) > 2:
                    fig.data = tuple(fig.data[:2])
            return None

        # Prepare component traces count on main thread for consistent layout
        comp_traces_needed = len(included)
        with fig.batch_update():
            current_components = max(0, len(fig.data) - 2)
            if current_components > comp_traces_needed:
                fig.data = tuple(list(fig.data)[: 2 + comp_traces_needed])
            elif current_components < comp_traces_needed:
                for _k in range(comp_traces_needed - current_components):
                    fig.add_scatter(
                        x=[],
                        y=[],
                        mode="lines",
                        line=dict(dash="dot"),
                        name=f"Component {_k+1}",
                    )

        # Cancel any running fit and start a new one in the background
        try:
            if fit_thread is not None and fit_thread.is_alive():
                # Signal the currently running worker to stop (per-fit only)
                try:
                    if fit_cancel_token is not None:
                        fit_cancel_token.set()
                except Exception:
                    pass
        except Exception:
            pass
        # Create a fresh cancel token for this new worker and capture it locally
        local_cancel = threading.Event()
        # Keep the global cancel_event stable; only update the active per-fit token
        fit_cancel_token = local_cancel

        old_redchi = last_redchi_by_idx.get(idx, None)
        # Update status label immediately on main thread
        if not iterating_in_progress:
            try:
                status_html.value = (
                    "<span style='color:#555;'>Refitting...</span>"
                    if old_redchi is not None
                    else "<span style='color:#555;'>Fitting...</span>"
                )
            except Exception:
                with msg_out:
                    msg_out.clear_output()
                    print("Refitting..." if old_redchi is not None else "Fitting...")

        def _worker(local_cancel_token=local_cancel):
            try:
                # Use full x-range for fitting inside worker
                x_sub = x_arr
                y_sub = y_arr

                # Build composite model
                comp_model = None
                params = None
                for i in included:
                    cx = peaks_x[i]
                    m = PseudoVoigtModel(prefix=f"p{i}_")
                    p = m.make_params()
                    p[f"p{i}_center"].set(
                        value=float(cx),
                        min=float(cx) - center_window.value,
                        max=float(cx) + center_window.value,
                    )
                    p[f"p{i}_sigma"].set(
                        value=float(init_sigma.value), min=1e-3, max=1e3
                    )
                    alpha_val = (
                        float(alpha_sliders[i].value) if i < len(alpha_sliders) else 0.5
                    )
                    p[f"p{i}_fraction"].set(
                        value=alpha_val, min=0.0, max=1.0, vary=False
                    )
                    amp0 = abs(float(peaks_y[i])) * max(1.0, float(init_sigma.value))
                    p[f"p{i}_amplitude"].set(value=amp0, min=0.0)

                    if comp_model is None:
                        comp_model = m
                        params = p
                    else:
                        comp_model = comp_model + m
                        params.update(p)

                # iter_cb to allow cooperative cancellation
                def _iter_cb(params_, iter_, resid_, *args, **kws):
                    if local_cancel_token.is_set():
                        raise KeyboardInterrupt("Fit cancelled by user")

                result = comp_model.fit(y_sub, params, x=x_sub, iter_cb=_iter_cb)
                if local_cancel_token.is_set():
                    return
                y_fit = result.eval(x=x_arr)
                comps = result.eval_components(x=x_arr)

                # Persist last successful result for this spectrum
                try:
                    last_result_by_idx[idx] = result
                except Exception:
                    pass

                def _apply_results_on_ui():
                    nonlocal iter_summary_pending, iter_start_redchi, iter_final_redchi
                    # Always try to update the plot, but don't let plotting failures
                    # prevent status messages from updating.
                    plot_ok = True
                    try:
                        comp_traces_needed = len(included)
                        with fig.batch_update():
                            # Ensure we have exactly 2 + comp_traces_needed traces
                            current_components = max(0, len(fig.data) - 2)
                            if current_components > comp_traces_needed:
                                fig.data = tuple(
                                    list(fig.data)[: 2 + comp_traces_needed]
                                )
                            elif current_components < comp_traces_needed:
                                for _k in range(
                                    comp_traces_needed - current_components
                                ):
                                    fig.add_scatter(
                                        x=[],
                                        y=[],
                                        mode="lines",
                                        line=dict(dash="dot"),
                                        name=f"Component {_k+1}",
                                    )
                            # Update data and fit traces
                            fig.data[0].x = x_arr.tolist()
                            fig.data[0].y = y_arr.tolist()
                            fig.data[1].x = x_arr.tolist()
                            fig.data[1].y = (y_fit.tolist() if hasattr(y_fit, "tolist") else list(y_fit))
                            # Update component traces safely
                            for comp_idx, i in enumerate(included):
                                key = f"p{i}_"
                                y_comp = comps.get(key, np.zeros_like(x_arr))
                                fig.data[2 + comp_idx].x = x_arr.tolist()
                                fig.data[2 + comp_idx].y = (y_comp.tolist() if hasattr(y_comp, "tolist") else list(y_comp))
                    except Exception:
                        plot_ok = False
                    # Update message regardless of plot success
                    # Update status label text (suppressed during iterative correction)
                    new_redchi = getattr(result, "redchi", np.nan)
                    try:
                        if not iterating_in_progress:
                            # If an iteration just completed, prefer the pre- vs post-iteration summary
                            if iter_summary_pending and iter_start_redchi is not None and iter_final_redchi is not None:
                                try:
                                    old_str = f"{float(iter_start_redchi):.4g}"
                                except Exception:
                                    old_str = str(iter_start_redchi)
                                try:
                                    new_str = f"{float(iter_final_redchi):.4g}"
                                except Exception:
                                    new_str = str(iter_final_redchi)
                                status_html.value = (
                                    f"<span style='color:#000;'>Iterative correction complete. Reduced chi-square: ("
                                    f"{old_str}) ---&gt; ({new_str})</span>"
                                )
                                # Clear the pending summary after showing it once
                                iter_summary_pending = False
                            else:
                                # Format values consistently, showing change when refitting
                                try:
                                    old_str = (
                                        f"{old_redchi:.4g}" if old_redchi is not None else None
                                    )
                                except Exception:
                                    old_str = (
                                        str(old_redchi) if old_redchi is not None else None
                                    )
                                try:
                                    new_str = f"{new_redchi:.4g}"
                                except Exception:
                                    new_str = str(new_redchi)
                                if old_redchi is not None:
                                    status_html.value = (
                                        f"<span style='color:#000;'>Refit complete. Reduced "
                                        f"chi-square: ("
                                        f"{old_str}) ---&gt; ({new_str})</span>"
                                    )
                                else:
                                    status_html.value = (
                                        f"<span style='color:#000;'>Fit complete. Reduced "
                                        f"chi-square: ("
                                        f"{new_str})</span>"
                                    )
                        # Persist the new redchi so subsequent runs are treated as 
                        # refits
                        try:
                            if np.isfinite(new_redchi):
                                last_redchi_by_idx[idx] = float(new_redchi)
                        except Exception:
                            # If np.isfinite is not available or new_redchi is not 
                            # numeric, store raw
                            try:
                                last_redchi_by_idx[idx] = float(new_redchi)
                            except Exception:
                                last_redchi_by_idx[idx] = new_redchi
                        if not plot_ok:
                            # Also echo a note in the log output
                            with msg_out:
                                msg_out.clear_output(wait=True)
                                print(
                                    f"(Note: Plot update partially failed; re-run to "
                                    f"refresh components.)"
                                )
                    except Exception:
                        if not iterating_in_progress:
                            status_html.value = (
                                "<span style='color:#000;'>Fit complete.</span>"
                            )
                        try:
                            if np.isfinite(new_redchi):
                                last_redchi_by_idx[idx] = float(new_redchi)
                            else:
                                last_redchi_by_idx[idx] = new_redchi
                        except Exception:
                            last_redchi_by_idx[idx] = new_redchi
                    # Update Cancel Fit button visibility after this fit completes
                    try:
                        _update_cancel_fit_visibility()
                    except Exception:
                        pass

                _on_main_thread(_apply_results_on_ui)
            except KeyboardInterrupt:
                # Cancellation requested
                def _notify_cancel():
                    if not iterating_in_progress:
                        try:
                            status_html.value = (
                                "<span style='color:#a00;'>Fit cancelled.</span>"
                            )
                        except Exception:
                            with msg_out:
                                msg_out.clear_output()
                                print("Fit cancelled.")
                    try:
                        _update_cancel_fit_visibility()
                    except Exception:
                        pass

                _on_main_thread(_notify_cancel)
            except Exception as e:

                def _notify_error():
                    if not iterating_in_progress:
                        try:
                            status_html.value = (
                                f"<span style='color:#a00;'>Fit failed: {e}</span>"
                            )
                        except Exception:
                            with msg_out:
                                msg_out.clear_output()
                                print(f"Fit failed: {e}")
                    try:
                        _update_cancel_fit_visibility()
                    except Exception:
                        pass

                _on_main_thread(_notify_error)

        fit_thread = threading.Thread(target=_worker, daemon=True)
        fit_thread.start()
        try:
            _update_cancel_fit_visibility()
        except Exception:
            pass
        return None

    def _on_spectrum_change(*_):
        nonlocal bulk_update_in_progress
        # Snapshot previous spectrum's controls before switching
        _snapshot_current_controls()
        idx = spectrum_sel.value
        # Restore per-spectrum globals and per-peak controls without triggering many fits
        bulk_update_in_progress = True
        try:
            g = per_spec_globals.get(idx)
            if g is not None:
                try:
                    center_window.value = float(g.get("center_window", center_window.value))
                except Exception:
                    pass
                try:
                    init_sigma.value = float(g.get("init_sigma", init_sigma.value))
                except Exception:
                    pass
        except Exception:
            pass
        _rebuild_alpha_sliders(idx)
        bulk_update_in_progress = False
        _fit_and_update_plot()

    # No range checkboxes to manage

    def _save_for_file(b):
        idx = spectrum_sel.value
        # Use the last completed result if available; if not, trigger a fit and inform
        # the user to wait for completion.
        res = last_result_by_idx.get(idx)
        if res is None:
            _fit_and_update_plot()
            with msg_out:
                msg_out.clear_output()
                print("Fitting... Please click 'Save' again once the fit completes.")
            return
        peaks_x, _ = _get_peaks(idx)
        included = [i for i, cb in enumerate(include_checkboxes) if cb.value]
        out = []
        for i in included:
            cx = peaks_x[i]
            d = {}
            for name in ("amplitude", "center", "sigma", "fraction"):
                p = res.params.get(f"p{i}_{name}")
                if p is not None:
                    d[name] = float(p.value)
            out.append(d)
        FTIR_DataFrame.at[idx, results_col] = out
        with msg_out:
            msg_out.clear_output()
            print(
                f"Saved deconvolution for file "
                f"'{FTIR_DataFrame.loc[idx, 'File Name']}'."
            )

    def _close_ui(b):
        # Signal cancellation and close widgets promptly
        try:
            cancel_event.set()
        except Exception:
            pass
        try:
            spectrum_sel.close()
            material_dd.close()
            conditions_dd.close()
            center_window.close()
            init_sigma.close()
            add_peaks_btn.close()
            accept_new_peaks_btn.close()
            redo_new_peaks_btn.close()
            cancel_new_peaks_btn.close()
            # Ensure iterative/cancel controls disappear on close
            iter_btn.close()
            cancel_fit_btn.close()
            save_btn.close()
            close_btn.close()
            for s in alpha_sliders:
                s.close()
            peak_controls_box.close()
            # Close the main UI container and status label so no stray widgets remain
            try:
                status_html.close()
            except Exception:
                pass
            try:
                ui.close()
            except Exception:
                pass
            # Leave msg_out displayed so the last messages remain visible
            fig.close()
        except Exception:
            pass

    # Wire events
    spectrum_sel.observe(_on_spectrum_change, names="value")
    material_dd.observe(_rebuild_spectrum_options, names="value")
    conditions_dd.observe(_rebuild_spectrum_options, names="value")
    for w in (center_window, init_sigma):
        w.observe(_on_control_change, names="value")
    # Wire reset buttons
    def _reset_center(_b=None):
        nonlocal bulk_update_in_progress
        bulk_update_in_progress = True
        try:
            center_window.value = default_center_window_value
        except Exception:
            pass
        _snapshot_current_controls()
        bulk_update_in_progress = False
        _fit_and_update_plot()

    def _reset_sigma(_b=None):
        nonlocal bulk_update_in_progress
        bulk_update_in_progress = True
        try:
            init_sigma.value = default_init_sigma_value
        except Exception:
            pass
        _snapshot_current_controls()
        bulk_update_in_progress = False
        _fit_and_update_plot()

    def _reset_all(_b=None):
        nonlocal bulk_update_in_progress
        bulk_update_in_progress = True
        # Reset globals
        try:
            center_window.value = default_center_window_value
        except Exception:
            pass
        try:
            init_sigma.value = default_init_sigma_value
        except Exception:
            pass
        # Reset per-peak controls
        try:
            for cb in include_checkboxes:
                cb.value = DEFAULT_INCLUDE
        except Exception:
            pass
        try:
            for s in alpha_sliders:
                s.value = DEFAULT_ALPHA
        except Exception:
            pass
        _snapshot_current_controls()
        bulk_update_in_progress = False
        _fit_and_update_plot()

    reset_center_btn.on_click(_reset_center)
    reset_sigma_btn.on_click(_reset_sigma)
    reset_all_btn.on_click(_reset_all)
    save_btn.on_click(_save_for_file)
    close_btn.on_click(_close_ui)

    def _iteratively_correct_worker():
        """Coordinate-descent style tuning of parameters to reduce reduced chi-square.

        Iterates over per-peak α sliders (for included peaks), the center window, and
        the init sigma, trying +/- step changes one parameter at a time. Keeps a
        change only if reduced chi-square improves. Stops when a full sweep makes no
        improvements, or after a safety cap of sweeps.
        """
        import time
        nonlocal iterating_in_progress, iter_start_redchi, iter_final_redchi, iter_summary_pending

        idx = spectrum_sel.value

        # Helper: run fit synchronously and return last redchi
        def _run_fit_and_wait():
            # If cancellation is already requested, don't start a new fit
            try:
                if cancel_event.is_set():
                    return np.inf
            except Exception:
                pass
            _fit_and_update_plot()
            # Wait for background fit to complete (with a timeout guard)
            try:
                for _ in range(400):  # up to ~40s total at 0.1s intervals
                    th = fit_thread
                    if th is None or not th.is_alive():
                        break
                    # Allow cooperative cancellation
                    try:
                        if cancel_event.is_set():
                            break
                    except Exception:
                        pass
                    time.sleep(0.1)
            except Exception:
                pass
            rc = last_redchi_by_idx.get(idx, np.inf)
            try:
                return float(rc)
            except Exception:
                return np.inf

        # Establish baseline reduced chi-square from snapshot taken at click time.
        # Fall back to one quick fit if no prior value exists.
        start_rc = iter_start_redchi
        try:
            start_rc = float(start_rc)
        except Exception:
            start_rc = np.inf
        if not np.isfinite(start_rc):
            # No prior redchi recorded; compute once.
            try:
                status_html.value = (
                    "<span style='color:#555;'>Iteratively correcting (pre-fit)...</span>"
                )
            except Exception:
                pass
            start_rc = _run_fit_and_wait()
        base_rc = start_rc
        # Iteration change counter: increments whenever a parameter change is kept
        iteration_changes = 0
        # Minimal running status; will update the counter on each kept change
        try:
            status_html.value = (
                "<span style='color:#555;'>iterating... (iterations so far: 0)</span>"
            )
        except Exception:
            pass
        try:
            if cancel_event.is_set():
                # Show old -> current comparison on cancel
                try:
                    old_str = f"{start_rc:.4g}"
                except Exception:
                    old_str = str(start_rc)
                try:
                    new_str = f"{base_rc:.4g}"
                except Exception:
                    new_str = str(base_rc)
                try:
                    status_html.value = (
                        f"<span style='color:#a00;'>Iterative correction cancelled. "
                        f"Reduced chi-square: ({old_str}) ---&gt; ({new_str})</span>"
                    )
                except Exception:
                    pass
                try:
                    _on_main_thread(_update_cancel_fit_visibility)
                except Exception:
                    pass
                # Allow normal fit status updates again
                iterating_in_progress = False
                return
        except Exception:
            pass
        if not np.isfinite(base_rc):
            base_rc = _run_fit_and_wait()

        # Build the list of adjustable parameters
        included_idxs = [i for i, cb in enumerate(include_checkboxes) if cb.value]
        if len(included_idxs) == 0:
            try:
                status_html.value = (
                    "<span style='color:#a00;'>Cannot iterate: no peaks selected.</span>"
                )
            except Exception:
                with msg_out:
                    msg_out.clear_output()
                    print("Cannot iterate: no peaks selected.")
            try:
                _on_main_thread(_update_cancel_fit_visibility)
            except Exception:
                pass
            return

        alpha_steps = {i: 0.1 for i in included_idxs}
        center_step = 2.0
        sigma_step = 1.0

        def _clamp(val, lo, hi):
            try:
                return max(lo, min(hi, val))
            except Exception:
                return val

        # Try a single parameter perturbation, returning (improved_rc, kept_change)
        def _try_adjust(getter, setter, decrementer, restore, label):
            nonlocal base_rc
            # If cancel requested, do not proceed
            try:
                if cancel_event.is_set():
                    return base_rc, False
            except Exception:
                pass
            # Try +step
            setter()
            rc_plus = _run_fit_and_wait()
            if rc_plus < base_rc:
                base_rc = rc_plus
                return rc_plus, True
            # Try -step
            decrementer()
            rc_minus = _run_fit_and_wait()
            # Early exit on cancel
            try:
                if cancel_event.is_set():
                    restore()
                    return base_rc, False
            except Exception:
                pass
            if rc_minus < base_rc:
                base_rc = rc_minus
                return rc_minus, True
            # Revert if neither improved
            restore()
            return base_rc, False

        sweeps = 0
        max_sweeps = 10
        improved_any = True
        while improved_any and sweeps < max_sweeps:
            # Check for cancellation at the start of each sweep
            try:
                if cancel_event.is_set():
                    # Show running total and break to final message below
                    try:
                        status_html.value = (
                            f"<span style='color:#a00;'>Iterative correction cancelled.</span>"
                        )
                    except Exception:
                        pass
                    break
            except Exception:
                pass
            improved_any = False
            sweeps += 1
            # No per-sweep status update; counter increments only on kept changes

            # 1) Per-peak α sliders (only included peaks)
            for i in included_idxs:
                try:
                    if cancel_event.is_set():
                        break
                except Exception:
                    pass
                if i >= len(alpha_sliders):
                    continue
                sld = alpha_sliders[i]
                if sld is None:
                    continue
                v0 = float(sld.value)
                step = float(alpha_steps.get(i, 0.1))
                # Define actions
                def set_plus(v0=v0, sld=sld, step=step):
                    sld.value = _clamp(v0 + step, 0.0, 1.0)

                def set_minus(v0=v0, sld=sld, step=step):
                    sld.value = _clamp(v0 - step, 0.0, 1.0)

                def restore(v0=v0, sld=sld):
                    sld.value = v0

                # Use getter label for debugging (not printed)
                _, kept = _try_adjust(
                    getter=lambda: sld.value,
                    setter=set_plus,
                    decrementer=set_minus,
                    restore=restore,
                    label=f"alpha[{i}]",
                )
                if kept:
                    improved_any = True
                    iteration_changes += 1
                    try:
                        status_html.value = (
                            f"<span style='color:#555;'>iterating... (iterations so far: {iteration_changes})</span>"
                        )
                    except Exception:
                        pass

            # 2) Center window (affects allowed center bounds)
            try:
                if cancel_event.is_set():
                    break
            except Exception:
                pass
            cw0 = float(center_window.value)

            def cw_plus(cw0=cw0):
                center_window.value = _clamp(
                    cw0 + center_step, float(center_window.min), float(center_window.max)
                )

            def cw_minus(cw0=cw0):
                center_window.value = _clamp(
                    cw0 - center_step, float(center_window.min), float(center_window.max)
                )

            def cw_restore(cw0=cw0):
                center_window.value = cw0

            _, kept = _try_adjust(
                getter=lambda: center_window.value,
                setter=cw_plus,
                decrementer=cw_minus,
                restore=cw_restore,
                label="center_window",
            )
            if kept:
                improved_any = True
                iteration_changes += 1
                try:
                    status_html.value = (
                        f"<span style='color:#555;'>iterating... (iterations so far: {iteration_changes})</span>"
                    )
                except Exception:
                    pass

            # 3) Init sigma
            try:
                if cancel_event.is_set():
                    break
            except Exception:
                pass
            sg0 = float(init_sigma.value)

            def sg_plus(sg0=sg0):
                init_sigma.value = _clamp(
                    sg0 + sigma_step, float(init_sigma.min), float(init_sigma.max)
                )

            def sg_minus(sg0=sg0):
                init_sigma.value = _clamp(
                    sg0 - sigma_step, float(init_sigma.min), float(init_sigma.max)
                )

            def sg_restore(sg0=sg0):
                init_sigma.value = sg0

            _, kept = _try_adjust(
                getter=lambda: init_sigma.value,
                setter=sg_plus,
                decrementer=sg_minus,
                restore=sg_restore,
                label="init_sigma",
            )
            if kept:
                improved_any = True
                iteration_changes += 1
                try:
                    status_html.value = (
                        f"<span style='color:#555;'>iterating... (iterations so far: {iteration_changes})</span>"
                    )
                except Exception:
                    pass

        # Final status: show old -> new comparison
        try:
            old_str = f"{start_rc:.4g}"
        except Exception:
            old_str = str(start_rc)
        try:
            new_str = f"{base_rc:.4g}"
        except Exception:
            new_str = str(base_rc)
        try:
            if cancel_event.is_set():
                status_html.value = (
                    f"<span style='color:#a00;'>Iterative correction cancelled. "
                    f"Reduced chi-square: ({old_str}) ---&gt; ({new_str})</span>"
                )
            else:
                status_html.value = (
                    f"<span style='color:#000;'>Iterative correction complete. "
                    f"Reduced chi-square: ({old_str}) ---&gt; ({new_str})</span>"
                )
        except Exception:
            with msg_out:
                msg_out.clear_output()
                if cancel_event.is_set():
                    print(
                        f"Iterative correction cancelled. Reduced chi-square: ({old_str}) -> ({new_str})"
                    )
                else:
                    print(
                        f"Iterative correction complete. Reduced chi-square: ({old_str}) -> ({new_str})"
                    )
        # Remember iteration summary for the next refit message
        iter_final_redchi = base_rc
        iter_summary_pending = True
        # Allow normal fit status updates again
        iterating_in_progress = False
        # Hide the Cancel Fit button when iteration ends (success or cancel)
        try:
            _on_main_thread(_update_cancel_fit_visibility)
        except Exception:
            pass

    # Wire the iterative correct to run in background and support cancellation
    def _on_iteratively_correct_click(b):
        nonlocal iter_thread, iterating_in_progress, cancel_event, iter_start_redchi
        try:
            if iter_thread is not None and iter_thread.is_alive():
                try:
                    status_html.value = (
                        "<span style='color:#a60;'>Iterative correction is already running.</span>"
                    )
                except Exception:
                    pass
                return
        except Exception:
            pass
        # Snapshot current redchi for this spectrum at the moment of click
        try:
            idx_snapshot = spectrum_sel.value
            iter_start_redchi = last_redchi_by_idx.get(idx_snapshot, np.inf)
        except Exception:
            iter_start_redchi = np.inf
        # Reset cancel flag and mark iteration as active
        try:
            cancel_event.clear()
        except Exception:
            pass
        iterating_in_progress = True
        try:
            status_html.value = (
                "<span style='color:#555;'>Starting iterative correction...</span>"
            )
        except Exception:
            pass
        iter_thread = threading.Thread(target=_iteratively_correct_worker, daemon=True)
        iter_thread.start()
        try:
            _update_cancel_fit_visibility()
        except Exception:
            pass

    iter_btn.on_click(_on_iteratively_correct_click)

    def _on_cancel_fit(b):
        # Signal cancellation for any active fit/iteration
        try:
            cancel_event.set()
        except Exception:
            pass
        try:
            if fit_cancel_token is not None:
                fit_cancel_token.set()
        except Exception:
            pass
        try:
            status_html.value = (
                "<span style='color:#a00;'>Cancellation requested...</span>"
            )
        except Exception:
            pass
        # Keep visibility consistent; it will auto-hide when work stops
        try:
            _update_cancel_fit_visibility()
        except Exception:
            pass
    cancel_fit_btn.on_click(_on_cancel_fit)

    # Layout
    controls_row_filters = widgets.HBox([material_dd, conditions_dd])
    controls_row_spectrum = widgets.HBox([spectrum_sel])
    globals_column = widgets.VBox(
        [
            widgets.HBox([center_window, reset_center_btn]),
            widgets.HBox([init_sigma, reset_sigma_btn]),
        ]
    )
    reset_all_row = widgets.HBox([reset_all_btn])
    buttons_row = widgets.HBox(
        [
            add_peaks_btn,
            accept_new_peaks_btn,
            redo_new_peaks_btn,
            cancel_new_peaks_btn,
            iter_btn,
            cancel_fit_btn,
            save_btn,
            close_btn,
        ]
    )
    status_row = widgets.HBox([status_html])
    ui = widgets.VBox(
        [
            controls_row_filters,
            controls_row_spectrum,
            peak_controls_box,
            globals_column,
            reset_all_row,
            buttons_row,
            status_row,
        ]
    )

    # --- Add-peaks workflow callbacks ---
    def _enter_add_mode(b=None):
        nonlocal adding_mode
        adding_mode = True
        new_peak_xs.clear()
        _clear_add_peak_shapes()
        _hide(add_peaks_btn)
        _show(accept_new_peaks_btn)
        _show(redo_new_peaks_btn)
        _show(cancel_new_peaks_btn)
        # Hide parameter modifiers during add-peaks mode; keep Center ±window visible
        _hide(init_sigma)
        _hide(reset_sigma_btn)
        _hide(iter_btn)
        _hide(cancel_fit_btn)
        _hide(save_btn)
        _hide(close_btn)
        _hide(peak_controls_box)
        # Disable selection widgets to avoid changing spectrum/filters mid-selection
        try:
            spectrum_sel.disabled = True
            material_dd.disabled = True
            conditions_dd.disabled = True
        except Exception:
            pass
        with msg_out:
            msg_out.clear_output()
            print(
                f"Add-peaks mode: click one or more x-locations on the plot. Then "
                f"accept/redo/cancel."
            )

    def _accept_new_peaks(b=None):
        nonlocal adding_mode
        idx = spectrum_sel.value
        x_arr, y_arr = _get_xy(idx)
        if x_arr is None:
            with msg_out:
                msg_out.clear_output()
                print("Cannot accept: current spectrum has no normalized data.")
            return
        if len(new_peak_xs) == 0:
            # Nothing selected; just exit mode
            _cancel_new_peaks()
            return
        # Merge with existing peaks and sort; reject new peaks too close to existing
        xs_existing, ys_existing = _get_peaks(idx)
        xs_list = [float(v) for v in (xs_existing or [])]
        ys_list = [float(v) for v in (ys_existing or [])]

        try:
            min_sep = float(center_window.value)
        except Exception:
            min_sep = 0.0

        rejected_close = []
        for x_new in new_peak_xs:
            # Reject if too close to any existing (committed) peak
            too_close = False
            for xe in xs_existing or []:
                try:
                    if abs(float(xe) - float(x_new)) <= min_sep:
                        too_close = True
                        break
                except Exception:
                    continue
            if too_close:
                rejected_close.append(float(x_new))
                continue
            # find nearest y
            try:
                i = int(np.argmin(np.abs(x_arr - x_new)))
                y_new = float(y_arr[i])
            except Exception:
                y_new = float("nan")
            xs_list.append(float(x_new))
            ys_list.append(y_new)

        # Sort by x
        try:
            pairs = sorted(zip(xs_list, ys_list), key=lambda t: float(t[0]))
            xs_sorted, ys_sorted = [list(t) for t in zip(*pairs)] if pairs else ([], [])
        except Exception:
            xs_sorted, ys_sorted = xs_list, ys_list

        # Persist back to DataFrame for this spectrum
        FTIR_DataFrame.at[idx, "Peak Wavenumbers"] = [float(v) for v in xs_sorted]
        FTIR_DataFrame.at[idx, "Peak Absorbances"] = [float(v) for v in ys_sorted]

        # Peaks changed; clear any saved per-peak settings for this spectrum
        try:
            per_spec_alpha.pop(idx, None)
            per_spec_include.pop(idx, None)
        except Exception:
            pass

        # Rebuild UI and refit with the updated peaks
        _rebuild_alpha_sliders(idx)
        _fit_and_update_plot()

        # Exit add mode and clean up visuals
        _clear_add_peak_shapes()
        new_peak_xs.clear()
        adding_mode = False
        _show(add_peaks_btn)
        _hide(accept_new_peaks_btn)
        _hide(redo_new_peaks_btn)
        _hide(cancel_new_peaks_btn)
        # Restore previously hidden/disabled controls after exiting add-peaks mode
        _show(init_sigma)
        _show(reset_sigma_btn)
        _show(iter_btn)
        _update_cancel_fit_visibility()
        _show(save_btn)
        _show(close_btn)
        _show(peak_controls_box)
        try:
            spectrum_sel.disabled = False
            material_dd.disabled = False
            conditions_dd.disabled = False
        except Exception:
            pass
        # Status: show any rejections in red
        if rejected_close:
            joined = ", ".join(f"{v:.3f}" for v in rejected_close)
            msg = (
                "Rejected due to proximity (±{:.2f} cm⁻¹): {}. "
                "Tip: reduce the Center ±window to fit peaks in small spaces."
            ).format(min_sep, joined)
            try:
                status_html.value = f"<span style='color:#a00;'>{msg}</span>"
            except Exception:
                with msg_out:
                    msg_out.clear_output()
                    print(msg)
        else:
            with msg_out:
                msg_out.clear_output()
                print("New peaks accepted and added to the current spectrum.")

    def _redo_new_peaks(b=None):
        new_peak_xs.clear()
        _clear_add_peak_shapes()
        with msg_out:
            msg_out.clear_output()
            print("Selection cleared. Click on the plot to select peaks again.")

    def _cancel_new_peaks(b=None):
        nonlocal adding_mode
        adding_mode = False
        new_peak_xs.clear()
        _clear_add_peak_shapes()
        _show(add_peaks_btn)
        _hide(accept_new_peaks_btn)
        _hide(redo_new_peaks_btn)
        _hide(cancel_new_peaks_btn)
        # Restore previously hidden/disabled controls after cancelling add-peaks mode
        _show(init_sigma)
        _show(reset_sigma_btn)
        _show(iter_btn)
        _update_cancel_fit_visibility()
        _show(save_btn)
        _show(close_btn)
        _show(peak_controls_box)
        try:
            spectrum_sel.disabled = False
            material_dd.disabled = False
            conditions_dd.disabled = False
        except Exception:
            pass
        with msg_out:
            msg_out.clear_output()
            print("Peak addition cancelled. No changes were made.")

    add_peaks_btn.on_click(_enter_add_mode)
    accept_new_peaks_btn.on_click(_accept_new_peaks)
    redo_new_peaks_btn.on_click(_redo_new_peaks)
    cancel_new_peaks_btn.on_click(_cancel_new_peaks)

    display(ui, fig, msg_out)
    # Seed options with current filters (defaults 'any') and trigger initial updates
    _rebuild_spectrum_options()

    return FTIR_DataFrame
