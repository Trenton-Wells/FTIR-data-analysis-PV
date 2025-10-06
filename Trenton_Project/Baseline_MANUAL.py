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
import matplotlib
matplotlib.use("TkAgg")  # Use TkAgg backend for interactive windows on Windows
import matplotlib.pyplot as plt
import numpy as np
import ast

def select_anchor_points(FTIR_dataframe, material=None, filepath=None):
	"""
	Lets user select anchor points from a spectrum in the DataFrame for baseline correction. Anchor points are selected by clicking on the plot, and will apply to each file of that material.

	Parameters
	----------
	FTIR_dataframe : pd.DataFrame
		The DataFrame containing the spectral data.
	material : str, optional
		Material name to analyze (ignored if filepath is provided).
	filepath : str, optional
		If provided, only process this file (by 'File Location' + 'File Name').

	Returns
	-------
	list
		List of selected anchor points (x-coordinates).
	"""

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

	import matplotlib.widgets as mwidgets
	figure, ax_plot = plt.subplots(figsize=(8, 7))
	ax_plot.plot(x_data, y_data, label='Raw Data')
	ax_plot.set_xlabel('Wavenumber (X)')
	ax_plot.set_ylabel('Absorbance (Y)')
	ax_plot.set_title('Click to select anchor points for baseline correction')
	anchor_points_x = []
	selection_complete = [False]

	def handle_plot_click(event):
		# Only register clicks inside the plot
		if event.inaxes == ax_plot and not selection_complete[0]:
			anchor_points_x.append(event.xdata)
			ax_plot.axvline(event.xdata, color='r', linestyle='--')
			plt.draw()
			print(f"Anchor point selected: {event.xdata}")

	click_connection_id = figure.canvas.mpl_connect('button_press_event', handle_plot_click)

	# Show the plot and let the user select anchor points
	plt.show()
	figure.canvas.mpl_disconnect(click_connection_id)
	print(f"Selected anchor points: {anchor_points_x}")

	# Interpolate using CubicSpline and plot
	from scipy.interpolate import CubicSpline
	anchor_points_x_sorted = sorted(anchor_points_x)
	# Find the y values at the exact anchor x positions from the original data
	anchor_points_y = []
	for x_anchor in anchor_points_x_sorted:
		# Find the closest x_data value to x_anchor
		idx = np.argmin(np.abs(np.array(x_data) - x_anchor))
		anchor_points_y.append(y_data[idx])
	spline = CubicSpline(anchor_points_x_sorted, anchor_points_y)
	x_spline = np.linspace(min(x_data), max(x_data), 1000)
	y_spline = spline(x_spline)

	# Plot original data and spline
	interp_fig, interp_ax = plt.subplots(figsize=(8, 7))
	interp_ax.plot(x_data, y_data, label='Raw Data')
	interp_ax.plot(x_spline, y_spline, color='g', label='Cubic Spline Baseline')
	interp_ax.scatter(anchor_points_x_sorted, anchor_points_y, color='r', label='Anchor Points')
	interp_ax.set_xlabel('Wavenumber (X)')
	interp_ax.set_ylabel('Absorbance (Y)')
	interp_ax.set_title('Baseline Interpolation Preview')
	interp_ax.legend()

	# Show the interpolation plot
	interp_fig.show()
	plt.show()  # This will block until the plot window is closed

	# Ask user to accept or redo via y/n input
	while True:
		user_response = input("Accept baseline interpolation? (y/n): ").strip().lower()
		if user_response == 'y':
			anchor_points = anchor_points_x
			print(f"Final selected anchor points: {anchor_points}")
			return anchor_points
		elif user_response == 'n':
			return select_anchor_points(FTIR_dataframe, material=material, filepath=filepath)
		else:
			print("Please enter 'y' to accept or 'n' to redo.")

if __name__ == "__main__":
    print("Starting anchor point selection...")
    anchor_points = select_anchor_points()
    print(f"Final selected anchor points: {anchor_points}")


