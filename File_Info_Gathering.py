"""
created: 9-15-2025
author: Trenton Wells
organization: NREL
NREL_contact: trenton.wells@nrel.gov
personal_contact: trentonwells73@gmail.com
"""

import os
import re
import pandas as pd

print(os.path.exists(r"C:\Script Testing"))
print(os.listdir(r"C:\Script Testing"))
# Set your root directory here
#root_dir = r"Y:\5200\Packaging Reliability\Durability Tool\Ray Tracing and Activation Spectrum\ATR-FTIR Data" TRUE RELEVANT PATH
root_dir = r"C:\Script Testing"

data = []

## Note that file organization should be as follows, with \ denoting file paths:
## date of scan \ details of spectrum
## ex: 05-25-2024 \ A4_PPE_500hr.csv
## Order of details is unimportant, spacing between details can be an underscore or a blank space
## Time detail must be followed with "h" or "hr" (not case-sensitive) to be recognized
for dirpath, _, filenames in os.walk(root_dir):
    print(f"Scanning directory: {dirpath}")
    print(f"Filenames found: {filenames}")
    parent_folder = os.path.basename(dirpath)
    # Extract date from parent folder name (adjust regex as needed)
    # If date not found by program, entire folder name will be inputted instead
    date_match = re.search(r'\d{2}-\d{2}-\d{4}', parent_folder)
    date = date_match.group(0) if date_match else parent_folder

    for fname in filenames:
        if fname.startswith('.'): #this will skip any files whose name starts with ".", since these are usually system files that are not relevant
            print(f"Skipping hidden/system file: {fname}")
            continue
        print(f"Processing file: {fname}")
        # Search for preset terms in filename
        # Put the names of the weathering conditions here so that they can be parsed out of the filename
        conditions_terms = ["A3", "A4", "A5", "ARC", "OPN", "KKCE", "0.5X", "1X", "2.5X", "5X"]
        # Put the names of the materials here so that they can be parsed out of the filename
        material_terms = ["CPC", "t-PVDF", "t-PVF", "o-PVF", "PPE", "J-BOX#1", "J-BOX#2", "PO"]
        # Find conditions
        conditions = next((term for term in conditions_terms if term in fname), "")
        # Find material
        material = next((term for term in material_terms if term in fname), "")
        # Find time
        time_match = re.search(r'(\d+)(?:H|hr)', fname, re.IGNORECASE)
        time = int(time_match.group(1)) if time_match else ""
        print(f"Parsed values -> Conditions: {conditions}, Material: {material}, Time: {time}, Date: {date}, Location: {dirpath}")
        # Add parsed data to dataframe
        data_row = {
            "file location": dirpath,
            "file name": fname,
            "date": date,
            "conditions": conditions,
            "material": material,
            "time": time
        }
        print(f"Appending row: {data_row}")
        data.append(data_row)
df = pd.DataFrame(data)
print(df)
