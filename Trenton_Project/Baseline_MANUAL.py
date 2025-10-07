#Created: 9-24-2025
#Author: Trenton Wells
#Organization: NREL
#NREL Contact: trenton.wells@nrel.gov
#Personal Contact: trentonwells73@gmail.com

## Description: While baseline generation is simply done with the CubicSplines function from scipy, this script defines the function which allows 
## the user to click on a plot of the raw data in order to select anchor points for a particular material's spectra.
## The user can then save these anchor points in the "anchor_points" section of the materials.json file for future use.
## The Baseline Parameters section of dataframe.csv will show the relevant anchor points for each material, which will be used to generate the baseline.


# Interactive anchor point selection for baseline correction
import ast
import numpy as np
import plotly.graph_objs as go
import ipywidgets as widgets
from IPython.display import display, clear_output
from scipy.interpolate import CubicSpline

SELECTED_ANCHOR_POINTS = []  # Global variable to store selected anchor points

# --- Clean up old widgets to avoid Plotly/ipywidgets state errors ---
def _cleanup_widgets():
    """
    Cleans up any lingering ipywidgets or Plotly FigureWidget objects from previous runs.
    This prevents state errors and memory leaks when running interactive selection multiple times in Jupyter/VS Code.
    """
    try:
        import gc
        for obj in gc.get_objects():
            if hasattr(obj, 'close') and callable(obj.close):
                try:
                    obj.close()
                except Exception:
                    pass
    except Exception:
        pass

def select_anchor_points(FTIR_dataframe, material=None, filepath=None, try_it_out=True):
    """
    Interactive anchor point selection for FTIR baseline correction.

    Lets user select anchor points from a spectrum in the DataFrame for baseline correction. Anchor points are selected by clicking on the plot, and will apply to each file of that material. After selection, a cubic spline baseline is fit and previewed, and the user can accept or redo the selection.

    Parameters
    ----------
    FTIR_dataframe : pd.DataFrame
        The DataFrame containing the spectral data.
    material : str, optional
        Material name to analyze (ignored if filepath is provided).
    filepath : str, optional
        If provided, only process this file (by 'File Location' + 'File Name').
    try_it_out : bool, optional
        If True, only prints the anchor points (default). If False, saves anchor points to 'Baseline Parameters' column for all rows with the same material.

    Returns
    -------
    None
        The selected anchor points are stored in the global variable SELECTED_ANCHOR_POINTS.
    """
    _cleanup_widgets()  # Clean up widgets from previous runs

    # --- Data selection logic ---
    if filepath is not None:
        import os
        if os.path.sep in filepath:
            folder, fname = os.path.split(filepath)
            filtered = FTIR_dataframe[(FTIR_dataframe['File Location'] == folder) & (FTIR_dataframe['File Name'] == fname)]
        else:
            filtered = FTIR_dataframe[FTIR_dataframe['File Name'] == filepath]
        if filtered.empty:
            raise ValueError(f"No entry found for file '{filepath}'.")
        row = filtered.iloc[0]
    else:
        if material is None:
            raise ValueError("Material must be specified if filepath is not provided.")
        filtered = FTIR_dataframe[(FTIR_dataframe['Material'] == material) & (FTIR_dataframe['Time'] == 0)]
        if filtered.empty:
            raise ValueError(f"No entry found for material '{material}' with time == 0.")
        row = filtered.iloc[0]

    x_data = ast.literal_eval(row['X-Axis']) if isinstance(row['X-Axis'], str) else row['X-Axis']
    y_data = ast.literal_eval(row['Raw Data']) if isinstance(row['Raw Data'], str) else row['Raw Data']

    # --- Widget and button setup (define early for scope) ---
    accept_button = widgets.Button(description='Accept', button_style='success')
    redo_button = widgets.Button(description='Redo', button_style='warning')
    button_box = widgets.HBox([accept_button, redo_button])
    button_box_out = widgets.Output()
    done = widgets.Output()
    output = widgets.Output()
    anchor_points = []
    anchor_markers = go.Scatter(x=[], y=[], mode='markers', marker=dict(color='red', size=10), name='Anchor Points')

    # --- Plot setup ---
    fig = go.FigureWidget(data=[go.Scatter(x=x_data, y=y_data, mode='lines', name='Raw Data'), anchor_markers])
    fig.update_layout(title='Click to select anchor points for baseline correction',
                     xaxis_title='Wavenumber (cm⁻¹)',
                     yaxis_title='Absorbance (AU)')

    # --- Click handler for anchor selection ---
    def on_click(trace, points, selector):
        if points.xs:
            x_val = points.xs[0]
            anchor_points.append(x_val)
            # Add a vertical line for the anchor point
            vline = dict(type='line', x0=x_val, x1=x_val, y0=min(y_data), y1=max(y_data),
                         line=dict(color='red', dash='dash'))
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
    scatter.on_click(on_click)

    # --- Accept/Redo logic, defined once and reused ---
    def show_baseline_preview():
        """
        Helper function:
        Shows a preview of the cubic spline baseline and baseline-corrected spectrum using the selected anchor points.
        Lets the user accept or redo the selection after visualizing the correction.
        """
        # Fit cubic spline to anchor points with zero slope at endpoints
        anchor_x = np.array(sorted(anchor_points))
        anchor_y = np.array([y_data[np.argmin(np.abs(np.array(x_data) - x))] for x in anchor_x])
        spline = CubicSpline(anchor_x, anchor_y, bc_type=((1, 0.0), (1, 0.0)))
        x_dense = np.linspace(min(x_data), max(x_data), 1000)
        spline_y = spline(x_dense)
        y_interp = np.interp(x_data, x_dense, spline_y)
        baseline_corrected = np.array(y_data) - y_interp
        from plotly.subplots import make_subplots
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=("Original with Baseline", "Baseline Corrected"))
        fig2.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name='Raw Data'), row=1, col=1)
        fig2.add_trace(go.Scatter(x=x_dense, y=spline_y, mode='lines', name='Baseline', line=dict(color='green')), row=1, col=1)
        fig2.add_trace(go.Scatter(x=anchor_x, y=anchor_y, mode='markers', marker=dict(color='red', size=10), name='Anchor Points'), row=1, col=1)
        fig2.add_trace(go.Scatter(x=x_data, y=baseline_corrected, mode='lines', name='Baseline Corrected', line=dict(color='purple')), row=2, col=1)
        fig2.update_layout(height=800, title_text="Baseline Correction Preview",
                          xaxis_title='Wavenumber (cm⁻¹)', yaxis_title='Absorbance (AU)')
        # Accept/Redo for preview
        accept2 = widgets.Button(description='Accept', button_style='success')
        redo2 = widgets.Button(description='Redo', button_style='warning')
        button_box2 = widgets.HBox([accept2, redo2])
        button_box2_out = widgets.Output()
        done2 = widgets.Output()
        def accept2_callback(b):
            global SELECTED_ANCHOR_POINTS
            SELECTED_ANCHOR_POINTS = sorted(anchor_points)
            with done2:
                clear_output()
                # If in the "try baselines" section, print the results. If not, save to dataframe.
                if try_it_out:
                    print(f"DONE--Final selected anchor points: {SELECTED_ANCHOR_POINTS}")
                else:
                    # Save anchor points to Baseline Parameters for all rows with the same material
                    mat = row['Material']
                    for idx, r in FTIR_dataframe.iterrows():
                        if r['Material'] == mat:
                            FTIR_dataframe.at[idx, 'Baseline Parameters'] = str(SELECTED_ANCHOR_POINTS)
                    print(f"Anchor points saved to Baseline Parameters for material '{mat}'.")
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
        Helper function:
        Clears anchor points and resets the interactive plot and widgets for a new selection.
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

    def accept_callback(b):
        """
        Helper function:
        Handles the Accept button for the initial anchor selection.
        If enough points are selected, shows the baseline preview; otherwise, prompts the user to select more points.
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
        Handles the Redo button for the initial anchor selection, resetting the selection process.
        """
        reset_selection()

    accept_button.on_click(accept_callback)
    redo_button.on_click(redo_callback)

    # --- Initial display ---
    display(fig, output, button_box_out, done)
    with button_box_out:
        display(button_box)
    return None
