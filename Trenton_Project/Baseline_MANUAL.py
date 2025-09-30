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

def select_anchor_points_from_file():
	data_file_path = input("Enter the path to the data file to plot: ").strip()
	x_data = []
	y_data = []
	# Assumes two columns: x and y, whitespace or comma separated
	with open(data_file_path, 'r') as raw_data_file:
		for line in raw_data_file:
			data_columns = line.strip().replace(',', ' ').split()
			if len(data_columns) >= 2:
				try:
					x_data.append(float(data_columns[0]))
					y_data.append(float(data_columns[1]))
				except ValueError:
					continue

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
			return anchor_points_x
		elif user_response == 'n':
			return select_anchor_points_from_file()
		else:
			print("Please enter 'y' to accept or 'n' to redo.")

if __name__ == "__main__":
    print("Starting anchor point selection...")
    anchor_points = select_anchor_points_from_file()
    print(f"Final selected anchor points: {anchor_points}")


