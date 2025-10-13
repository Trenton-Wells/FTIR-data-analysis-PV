# Created: 9-23-2025
# Author: Trenton Wells
# Organization: NREL
# NREL Contact: trenton.wells@nrel.gov
# Personal Contact: trentonwells73@gmail.com
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from pybaselines.smooth import arpls
from pybaselines.spline import irsqr
from pybaselines.classification import fabc
import ast
import random
import ipywidgets as widgets
from IPython.display import display, clear_output


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
    parameter_dictionary = {}
    for item in parameter_str.split(","):
        if "=" in item:
            key, value = item.split("=", 1)
            parameter_dictionary[key.strip()] = value.strip()
    return parameter_dictionary


def _get_default_parameters(function_name):
    """
    Input the name of a baseline function and return its default parameters as a 
    dictionary.

    Example: _get_default_parameters('ARPLS') returns {'lam': 1e6, 'p': 0.01, 
    'iterations': 10}
    
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
        "ARPLS": {"lam": 1e6, "p": 0.01, "iterations": 10},
        "IRSQR": {
            "lam": 1e6,
            "quantile": 0.05,
            "num_knots": 100,
            "spline_degree": 3,
            "diff_order": 3,
            "max_iterations": 100,
            "tolerance": 1e-6,
            "weights": None,
            "eps": None,
        },
        "FABC": {
            "lam": 1e6,
            "scale": None,
            "num_std": 3.0,
            "diff_order": 2,
            "min_length": 2,
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
        # lam: float, p: float, iterations: int
        if "lam" in parameters:
            parameters["lam"] = float(parameters["lam"])
        if "p" in parameters:
            parameters["p"] = float(parameters["p"])
        if "iterations" in parameters:
            parameters["iterations"] = int(parameters["iterations"])
    elif function == "IRSQR":
        # lam: float, quantile: float, num_knots: int, spline_degree: int, 
        # diff_order: int, max_iterations: int, tolerance: float
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
        if "max_iterations" in parameters:
            parameters["max_iterations"] = int(parameters["max_iterations"])
        if "tolerance" in parameters:
            parameters["tolerance"] = float(parameters["tolerance"])
    elif function == "FABC":
        # lam: float, scale: int or None, num_std: float, diff_order: int, 
        # min_length: int
        if "lam" in parameters:
            parameters["lam"] = float(parameters["lam"])
        if "scale" in parameters:
            if (
                parameters["scale"] is None
                or str(parameters["scale"]).lower() == "none"
                or parameters["scale"] == ""
            ):
                parameters["scale"] = None
            else:
                parameters["scale"] = int(parameters["scale"])
        if "num_std" in parameters:
            parameters["num_std"] = float(parameters["num_std"])
        if "diff_order" in parameters:
            parameters["diff_order"] = int(parameters["diff_order"])
        if "min_length" in parameters:
            parameters["min_length"] = int(parameters["min_length"])
    return parameters


def baseline_selection(FTIR_dataframe, materials=None, baseline_function=None):
    """
    Set the baseline function for specified materials in the DataFrame.

    Parameters
    ----------
    FTIR_dataframe : pd.DataFrame
        The DataFrame containing the spectral data.
    materials : list, optional
        List of material names to apply the baseline function to.
    baseline_function : str, optional
        Baseline function to use.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame.
    """

    # Accept either a string (comma-separated) or a list for materials
    if materials is None:
        materials_input = input(
            "Enter materials to apply baseline function to (comma-separated): "
        ).strip()
        materials = [mat.strip() for mat in materials_input.split(",")]
    elif isinstance(materials, str):
        materials = [mat.strip() for mat in materials.split(",")]

    # Prompt for baseline function if not provided
    if baseline_function is None:
        baseline_function = input(
            "Enter baseline function (ARPLS, IRSQR, FABC, Manual): "
        ).strip()

    for material in materials:
        mask = FTIR_dataframe["Material"] == material
        FTIR_dataframe.loc[mask, "Baseline Function"] = baseline_function
        print(f"Applied baseline function {baseline_function} to material: {material}")

    return FTIR_dataframe

def parameter_selection(FTIR_dataframe, materials=None, parameters=None):
    """
    Set the baseline parameters for specified materials in the DataFrame.

    Parameters
    ----------
    FTIR_dataframe : pd.DataFrame
        The DataFrame containing the spectral data.
    materials : string, optional
        String of material names to apply the parameters to.
    parameters : str or dict, optional
        The parameter values (as dict or string) to assign to all specified materials.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame.
    """

    # Accept either a string (comma-separated) or a list for materials
    if materials is None:
        materials_input = input(
            "Enter materials to apply baseline parameters to (comma-separated): "
        ).strip()
        materials = [mat.strip() for mat in materials_input.split(",")]
    elif isinstance(materials, str):
        materials = [mat.strip() for mat in materials.split(",")]

    # Prompt for parameters if not provided
    if parameters is None:
        message = (f"Enter baseline parameters (as dict or string) to apply to all "
                   f"selected materials: ")
        parameters = input(message).strip()
    for material in materials:
        mask = FTIR_dataframe["Material"] == material
        FTIR_dataframe.loc[mask, "Baseline Parameters"] = str(parameters)
        print(f"Applied parameters {parameters} to material: {material}")

    return FTIR_dataframe


def baseline_correction(FTIR_dataframe):
    """
    Apply baseline correction to the dataframe.

    Parameters
    ----------
    dataframe_path : str
        The path to the CSV file to modify.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame.
    """
    # Add new columns for Baseline Function and Parameters if they don't exist
    if "Baseline" not in FTIR_dataframe.columns:
        FTIR_dataframe["Baseline"] = None
    if "Corrected" not in FTIR_dataframe.columns:
        FTIR_dataframe["Corrected"] = None

    for idx, row in FTIR_dataframe.iterrows():
        baseline_name = row["Baseline Function"]
        parameter_dictionary = (
            ast.literal_eval(row["Baseline Parameters"])
            if row["Baseline Parameters"]
            else {}
        )
        # Example: get y-data (Raw Data) and x-data (X-Axis)
        try:
            y_data = ast.literal_eval(row["Raw Data"])
        except Exception:
            y_data = row["Raw Data"]

        baseline = None
        baseline_corrected = None

        if baseline_name == "ARPLS":
            # Call ARPLS baseline correction
            baseline = arpls(y_data, **parameter_dictionary)
            baseline_corrected = [y - b for y, b in zip(y_data, baseline)]
        elif baseline_name == "IRSQR":
            # Call IRSQR baseline correction
            baseline, _ = irsqr_baseline(
                None, y_data, **parameter_dictionary
            )  # Pass None for self if using as standalone
            baseline_corrected = [y - b for y, b in zip(y_data, baseline)]
        elif baseline_name == "FABC":
            # Call FABC baseline correction
            baseline_obj = Baseline()
            baseline, _ = baseline_obj.fabc(y_data, **parameter_dictionary)
            baseline_corrected = [y - b for y, b in zip(y_data, baseline)]
        elif baseline_name == "MANUAL":
            # Call Manual baseline correction
            from scipy.interpolate import CubicSpline

            anchor_points = parameter_dictionary.get("anchor_points", [])
            x_axis = (
                ast.literal_eval(row["X-Axis"])
                if isinstance(row["X-Axis"], str)
                else row["X-Axis"]
            )
            # For each anchor point, find the index of the closest value in X-Axis
            # This ensures that the anchor points correspond to actual data points in 
            # each spectrum
            anchor_indices = [
                min(range(len(x_axis)), key=lambda i: abs(x_axis[i] - ap))
                for ap in anchor_points
            ]
            y_anchor = [y_data[i] for i in anchor_indices]
            baseline = CubicSpline(x=anchor_points, y=y_anchor, extrapolate=True)(
                x_axis
            )
        else:
            print(f"Unknown baseline function: {baseline_name} for row {idx}")
            continue
        # Save results back to DataFrame
        # Cast all baseline values to float before saving
        if baseline is not None:
            baseline_floats = [float(val) for val in baseline]
            dataframe.at[idx, "Baseline"] = baseline_floats
        else:
            dataframe.at[idx, "Baseline"] = None
        # Cast all corrected values to float before saving
        if baseline_corrected is not None:
            corrected_floats = [float(val) for val in baseline_corrected]
            dataframe.at[idx, "Corrected"] = corrected_floats
        else:
            dataframe.at[idx, "Corrected"] = None

    # Save updated DataFrame
    dataframe.to_csv(dataframe_path, index=False)


def plot_grouped_spectra(
    FTIR_dataframe,
    materials,
    conditions,
    times,
    raw_data=True,
    baseline=False,
    baseline_corrected=False,
    separate_plots=False,
    include_replicates=True,
    zoom=None,
):
    """
    Plot grouped spectra based on material, condition, and time. 
    
    Accepts lists or 'any' for each category.

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
    mask = pd.Series([True] * len(FTIR_dataframe))
    if isinstance(materials, str) and materials.strip().lower() != "any":
        material_list = [m.strip() for m in materials.split(",") if m.strip()]
        mask &= FTIR_dataframe["Material"].isin(material_list)
    if isinstance(conditions, str) and conditions.strip().lower() != "any":
        condition_list = [c.strip() for c in conditions.split(",") if c.strip()]
        mask &= FTIR_dataframe["Conditions"].isin(condition_list)
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
        mask &= FTIR_dataframe["Time"].isin(time_list)
    filtered_data = FTIR_dataframe[mask]

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
            (line_handle,) = plt.plot(
                x_axis, y_data, "--", label=f"Baseline: {spectrum_label}"
            )
            legend_entries.append((line_handle, f"Baseline: {spectrum_label}"))
            color_map[("Baseline", idx)] = line_handle.get_color()
            legend_filepaths.append(file_path)
        if (
            baseline_corrected
            and "Corrected" in spectrum_row
            and spectrum_row["Corrected"] is not None
        ):
            y_data = spectrum_row["Corrected"]
            if isinstance(y_data, str):
                try:
                    y_data = ast.literal_eval(y_data)
                except Exception:
                    pass
            (line_handle,) = plt.plot(
                x_axis, y_data, ":", label=f"Corrected: {spectrum_label}"
            )
            legend_entries.append((line_handle, f"Corrected: {spectrum_label}"))
            color_map[("Corrected", idx)] = line_handle.get_color()
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
                plt.plot(x_axis, y_data, "--", label="Baseline", color=color)
            if (
                baseline_corrected
                and "Corrected" in row
                and row["Corrected"] is not None
            ):
                y_data = row["Corrected"]
                if isinstance(y_data, str):
                    try:
                        y_data = ast.literal_eval(y_data)
                    except Exception:
                        pass
                color = color_map.get(("Corrected", idx), None)
                plt.plot(x_axis, y_data, ":", label="Corrected", color=color)
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
    FTIR_dataframe,
    material=None,
    baseline_function=None,
    parameter_string=None,
    filepath=None,
):
    """
    Apply a baseline correction to the first file of a given material and plot the 
    result.

    Parameters
    ----------
    FTIR_dataframe (pd.DataFrame): The in-memory DataFrame containing all spectra.
    material (str, optional): Material name to analyze (ignored if filepath is 
        provided).
    baseline_function (str): Baseline function to use ('ARPLS', 'IRSQR', 'FABC').
    parameter_string (str, optional): Baseline parameters as key=value pairs, 
        comma-separated.
    filepath (str, optional): If provided, only process this file (by 'File Location' 
        + 'File Name').

    Returns
    -------
    None
    """
    if baseline_function is None:
        raise ValueError("Baseline function must be specified.")

    if filepath is not None:
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
        material = row.get("Material", "Unknown")
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
    if parameter_string:
        parameters = _parse_parameters(parameter_string)
    else:
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
            value=parameters.get("lam", 1e6),
            min=1e3,
            max=1e9,
            step=1e3,
            description="lam",
            readout_format=".1e",
        )
        param_widgets["p"] = widgets.FloatSlider(
            value=parameters.get("p", 0.01),
            min=0.001,
            max=0.999,
            step=0.001,
            description="p",
            readout_format=".3f",
        )
        param_widgets["iterations"] = widgets.IntSlider(
            value=parameters.get("iterations", 10),
            min=1,
            max=100,
            step=1,
            description="iterations",
        )
    elif baseline_function.upper() == "IRSQR":
        # lam: float, quantile: float, num_knots: int, spline_degree: int, diff_order: 
        # int, max_iterations: int, tolerance: float, eps: float
        param_widgets["lam"] = widgets.FloatSlider(
            value=parameters.get("lam", 1e6),
            min=1e3,
            max=1e9,
            step=1e3,
            description="lam",
            readout_format=".1e",
        )
        param_widgets["quantile"] = widgets.FloatSlider(
            value=parameters.get("quantile", 0.05),
            min=0.001,
            max=0.5,
            step=0.001,
            description="quantile",
            readout_format=".3f",
        )
        param_widgets["num_knots"] = widgets.IntSlider(
            value=parameters.get("num_knots", 100),
            min=5,
            max=500,
            step=5,
            description="num_knots",
        )
        param_widgets["spline_degree"] = widgets.IntSlider(
            value=parameters.get("spline_degree", 3),
            min=1,
            max=5,
            step=1,
            description="spline_degree",
        )
        param_widgets["diff_order"] = widgets.IntSlider(
            value=parameters.get("diff_order", 1),
            min=1,
            max=5,
            step=1,
            description="diff_order",
        )
        param_widgets["max_iterations"] = widgets.IntSlider(
            value=parameters.get("max_iterations", 100),
            min=1,
            max=1000,
            step=1,
            description="max_iterations",
        )
        param_widgets["tolerance"] = widgets.FloatSlider(
            value=parameters.get("tolerance", 1e-6),
            min=1e-10,
            max=1e-2,
            step=1e-7,
            description="tolerance",
            readout_format=".1e",
        )
        param_widgets["eps"] = widgets.FloatSlider(
            value=(
                parameters.get("eps", 1e-6)
                if parameters.get("eps", None) is not None
                else 1e-6
            ),
            min=1e-10,
            max=1e-2,
            step=1e-7,
            description="eps",
            readout_format=".1e",
        )
    elif baseline_function.upper() == "FABC":
        # lam: float, scale: int or None, num_std: float, diff_order: int, min_length: 
        # int
        param_widgets["lam"] = widgets.FloatSlider(
            value=parameters.get("lam", 1e6),
            min=1e3,
            max=1e9,
            step=1e3,
            description="lam",
            readout_format=".1e",
        )
        param_widgets["scale"] = widgets.IntSlider(
            value=(
                parameters.get("scale", 50)
                if parameters.get("scale", None) is not None
                else 50
            ),
            min=1,
            max=500,
            step=1,
            description="scale",
        )
        param_widgets["num_std"] = widgets.FloatSlider(
            value=parameters.get("num_std", 3.0),
            min=0.1,
            max=10.0,
            step=0.1,
            description="num_std",
            readout_format=".2f",
        )
        param_widgets["diff_order"] = widgets.IntSlider(
            value=parameters.get("diff_order", 2),
            min=1,
            max=5,
            step=1,
            description="diff_order",
        )
        param_widgets["min_length"] = widgets.IntSlider(
            value=parameters.get("min_length", 2),
            min=1,
            max=20,
            step=1,
            description="min_length",
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
                baseline = arpls(y, **param_vals)
            elif baseline_function.upper() == "IRSQR":
                baseline, _ = irsqr(y, **param_vals, x_axis=x)
            elif baseline_function.upper() == "FABC":
                baseline, _ = fabc(y, **param_vals)
            else:
                print(f"Unknown baseline function: {baseline_function}")
                return
            baseline_corrected = [a - b for a, b in zip(y, baseline)]
            import matplotlib.pyplot as plt

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
        ui = widgets.VBox([w for w in param_widgets.values()])
        from functools import partial

        widget_func = widgets.interactive_output(
            _plot_baseline, {k: w for k, w in param_widgets.items()}
        )
        display(ui, output)
        widget_func
    else:
        # No parameters to edit, just plot once
        _plot_baseline()


def test_baseline_choices(FTIR_dataframe, material=None):
    """
    Plot three random spectra for a given material, showing baseline results. 
    
    Plots raw data, baseline, and baseline-corrected data. The baseline function and 
    parameters are taken from the DataFrame columns. Assumes user has already filled 
    those columns earlier in the workflow.

    Parameters
    ----------
    FTIR_dataframe : pd.DataFrame
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
    filtered = FTIR_dataframe[FTIR_dataframe["Material"] == material]
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
        param_str = row.get("Baseline Parameters", None)
        if param_str:
            try:
                if isinstance(param_str, dict):
                    params = param_str
                else:
                    params = ast.literal_eval(param_str)
            except Exception:
                params = {}
        else:
            params = {}

        # Compute baseline
        baseline = None
        baseline_corrected = None
        try:
            if baseline_func is None:
                raise ValueError("No baseline function specified.")
            func = baseline_func.strip().upper()
            if func == "ARPLS":
                baseline = arpls(y, **params)
            elif func == "IRSQR":
                baseline, _ = irsqr(y, **params)
            elif func == "FABC":
                baseline, _ = fabc(y, **params)
            elif func == "MANUAL":
                anchor_points = params.get("anchor_points", [])
                if not anchor_points:
                    raise ValueError("No anchor_points for MANUAL baseline.")
                anchor_indices = [
                    min(range(len(x)), key=lambda i: abs(x[i] - ap))
                    for ap in anchor_points
                ]
                y_anchor = [y[i] for i in anchor_indices]
                baseline = CubicSpline(x=anchor_points, y=y_anchor, extrapolate=True)(x)
            else:
                raise ValueError(f"Unknown baseline function: {baseline_func}")
            baseline = np.array(baseline, dtype=float)
            baseline_corrected = y - baseline
        except Exception as e:
            print(f"Error computing baseline for row {idx}: {e}")
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


def bring_in_dataframe(dataframe_path=None):
    """
    Load the CSV file into a pandas DataFrame.

    Allows for easy dataframe manipulation in memory over the course of the analysis.

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
        dataframe_path = "FTIR_dataframe.csv"  # Default path if none is provided (will 
        # be in active directory)
    if os.path.exists(dataframe_path):
        FTIR_dataframe = pd.read_csv(
            dataframe_path
        )  # Load the DataFrame from the specified path
    else:
        FTIR_dataframe = (
            pd.DataFrame()
        )  # Create a new empty DataFrame if it doesn't exist
    return FTIR_dataframe

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