import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import pandas as pd
import ast

def baseline_correction(y, lam=1e6, p=0.01, n_iter=10):
    """
    Perform baseline correction using Asymmetric Least Squares (ALS) method.

    Parameters:
        y (array): Input spectrum (intensity values).
        lam (float): Smoothness parameter (higher = smoother baseline).
        p (float): Asymmetry parameter (0 < p < 1).
        n_iter (int): Number of iterations.

    Returns:
        baseline (array): Estimated baseline.
    """
    if lam is None:
        lam = 1e6  # Default smoothness parameter
    if p is None:
        p = 0.01  # Default asymmetry parameter
    if n_iter is None:
        n_iter = 10  # Default number of iterations

    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.T)  # Precompute the smoothness matrix
    w = np.ones(L)

    for _ in range(n_iter):
        W = diags(w, 0)
        Z = W + D
        baseline = spsolve(Z, w * y)
        w = p * (y > baseline) + (1 - p) * (y <= baseline)

    return baseline

if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\twells\Documents\GitHub\FTIR-data-analysis-PV\Trenton_Project\dataframe.csv")
    for idx, row in df.iterrows():
        # Get x-axis data from column before data_list (column 6, index 6)
        x_data = ast.literal_eval(row.iloc[6])
        # Convert string list to actual list for y-axis
        data_list = ast.literal_eval(row.iloc[7])
        baseline = baseline_correction(data_list)
        # Calculate baseline-corrected data
        baseline_corrected = np.array(data_list) - baseline
        # Plot using x_data for x-axis
        plt.plot(x_data, data_list, label='Original Data')
        plt.plot(x_data, baseline, label='Baseline', linestyle='--')
        #plt.plot(x_data, baseline_corrected, label='Baseline_Corrected', linestyle=(0, (1, 1)))
        plt.xlabel('Wavenumber (cm$^{-1}$)')
        plt.ylabel('Absorbance')
        # Use filename from second column (index 1)
        plt.title(str(row.iloc[1]))
        plt.legend()
        plt.show()
