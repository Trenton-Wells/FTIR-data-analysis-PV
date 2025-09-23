import pandas as pd
import matplotlib.pyplot as plt


def plot_ftir_spectrum(file_path=None):
    if file_path is None:
        ## Prompt user for file path
        file_path = input("Enter the path to your data file (CSV or TXT): ")

    ## Get parent folder and file name for title
    import os
    parent_folder = os.path.basename(os.path.dirname(file_path))
    file_name = os.path.basename(file_path)
    file_name_no_ext = os.path.splitext(file_name)[0]
    file_title = f"{parent_folder}/{file_name_no_ext}"

    ## Read the data file
    data = pd.read_csv(file_path, delim_whitespace=True, header=None)
    ## Define columns to plot (0-indexed)
    x_col = int(0)
    y_col = int(1)

    ## Plot
    plt.plot(data.iloc[:, x_col], data.iloc[:, y_col])
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Absorbance")
    plt.title(f"FTIR Spectrum {file_title}")
    plt.show()

if __name__ == "__main__":
    plot_ftir_spectrum()