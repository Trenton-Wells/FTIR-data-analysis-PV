#Created: 9-15-2025
#Author: Trenton Wells
#Organization: NREL
#NREL Contact: trenton.wells@nrel.gov
#Personal Contact: trentonwells73@gmail.com

## This script gathers basic spectra file information from a specified root directory and its subdirectories,
## extracting details from filenames and parent folder names to create a structured dataframe.
## The dataframe includes columns for file location, file name, date of scan, conditions, material, and time(duration).

# single comments are optional debugging lines that can be uncommented as needed.
## double comments contain important information about how to use or modify the script.

import os
import re
import pandas as pd
## User input: specify file types to scan (extensions, e.g. .csv,.0,.dpt)
file_types_input = input("Enter file types to scan, separated by commas (e.g. .csv,.0,.dpt): ").strip()
file_types = [ft.strip() for ft in file_types_input.split(',') if ft.strip()]

## Debugging note: if file extension name is part of material or condition terms, (e.g. extension .0 for the condition .005X),
## then the script may not correctly identify those terms. Consider adjusting the terms accordingly.

## User input: specify the separator(s) used in filenames and folder names
print("Do not pick separators that are part of material or condition terms.")
separator_input = input("Enter separator(s) used in filenames and folder names (e.g. _ or space or -): ").strip()
if separator_input.lower() == 'space':
    separators = [' ']
else:
    separators = [sep.strip() for sep in separator_input.split(',') if sep.strip()]

## User input: list the material terms to search for
material_terms = input("Enter material terms separated by commas (e.g. CPC,t-PVDF,t-PVF,o-PVF,PPE,J-BOX#1,J-BOX#2,PO): ").split(',')
material_terms = [term.strip() for term in material_terms]

## User input: list the condition terms to search for
conditions_terms = input("Enter condition terms separated by commas (e.g. A3,A4,A5,ARC,OPN,KKCE,0.5X,1X,2.5X,5X): ").split(',')
conditions_terms = [term.strip() for term in conditions_terms]

## User input: specify the root directory to scan
root_dir = input("Enter the path to the folder to be scanned: ").strip()
# print(os.path.exists(root_dir))
# print(os.listdir(root_dir))
data = []

def find_term(term, text):
    ## Add spaces around text to catch terms at the start/end
    return re.search(rf'(?<!\S){re.escape(term)}(?!\S)', f' {text} ', re.IGNORECASE) is not None

## FTIR Spectra may be organized in subfolders, but the relevant spectra file or its parent folder must contain the following details:
## Date of scan, conditions, material, time(duration). Otherwise, the dataframe will have empty entries for those details.
## If any of these details are found in the folder name, they will be taken from there; otherwise, the script will look for them in the filename.
## Material and conditions must be exactly as listed in the terms lists below to be recognized, so modify them to fit your needs (not case-sensitive).
## Time(duration) must be followed with "h" or "hr" (not case-sensitive) to be recognized.
## All date formats are recognized as long as days and months are two digits each and years are four digits, separated by hyphens (e.g., 09-15-2025 or 2025-09-15).
for dirpath, _, filenames in os.walk(root_dir):
    # print(f"Scanning directory: {dirpath}")
    # print(f"Filenames found: {filenames}") 
    parent_folder = os.path.basename(dirpath)

    ## This will ignore files that start with a dot, usually these are hidden/system files.
    for fname in filenames:
        if fname.startswith('.'):
            # print(f"Skipping hidden/system file: {fname}")
            continue
        ## Only scan files with the specified extensions
        if not any(fname.lower().endswith(ft.lower()) for ft in file_types):
            continue
        ## Remove file extension from fname for material and conditions term matching
        fname_no_ext = fname
        for ft in file_types:
            if fname_no_ext.lower().endswith(ft.lower()):
                fname_no_ext = fname_no_ext[:-(len(ft))]
        ## Normalize separators in parent_folder and fname for term matching
        norm_fname = fname_no_ext
        norm_parent_folder = parent_folder
        for sep in separators:
            norm_parent_folder = norm_parent_folder.replace(sep, ' ')
            norm_fname = norm_fname.replace(sep, ' ')

        # print(f"Processing file: {fname}")
        ## Search for date, time, and preset terms in parent folder name first, then filename if not found.
        ## These terms will be parsed out of filename or parent folder name to fill in dataframe.
        ## Find date
        date_match = re.search(r'(\d{2}-\d{2}-\d{4}|\d{4}-\d{2}-\d{2})', parent_folder)
        date = date_match.group(0) if date_match else None
        if not date:
            date_match_fname = re.search(r'(\d{2}-\d{2}-\d{4}|\d{4}-\d{2}-\d{2})', fname)
            date = date_match_fname.group(0) if date_match_fname else None
        ## Find conditions
        conditions = next((term for term in conditions_terms if find_term(term, norm_parent_folder)), None)
        if not conditions:
            conditions = next((term for term in conditions_terms if find_term(term, norm_fname)), None)
        ## Find material
        material = next((term for term in material_terms if find_term(term, norm_parent_folder)), None)
        if not material:
            material = next((term for term in material_terms if find_term(term, norm_fname)), None)
        ## Find time
        time_match = re.search(r'(\d+)(?:H|hr)', parent_folder, re.IGNORECASE)
        if time_match:
            time = int(time_match.group(1))
        else:
            time_match = re.search(r'(\d+)(?:H|hr)', fname, re.IGNORECASE)
            time = int(time_match.group(1)) if time_match else None
        ## Error printout if any value is missing
        if date is None or conditions is None or material is None or time is None:
            print(f"ValueError: Missing value for file '{fname}'. Results: date={date}, conditions={conditions}, material={material}, time={time}")
        # print(f"Parsed values -> Date: {date}, Conditions: {conditions}, Material: {material}, Time: {time}, Location: {dirpath}")
        ## Add parsed data to dataframe
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


