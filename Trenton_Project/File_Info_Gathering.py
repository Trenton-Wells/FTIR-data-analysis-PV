#Created: 9-15-2025
#Author: Trenton Wells
#Organization: NREL
#NREL Contact: trenton.wells@nrel.gov
#Personal Contact: trentonwells73@gmail.com

## This script gathers basic spectra file information from a specified root directory and its subdirectories,
## extracting details from filenames and parent folder names to create a structured dataframe.
## The dataframe includes columns for file location, file name, date of scan, conditions, material, and time(duration).

## General usage note: Avoid using separators that are part of material or condition terms, and avoid using extensions that are part of material or condition terms.
## Word recognition is case-insensitive and looks for whole words only (e.g., "PPE" will be found, but not "PPE1" or "XPPE").

# single comments are optional debugging lines that can be uncommented as needed.
## double comments contain important information about how to use or modify the script.

import os
import re
import pandas as pd

def find_term(term, text):
    """
    Helper function to find whole word matches of a term in text, case-insensitive.
    Adds spaces around text to catch terms at the start/end, since term-finding uses spaces on either side to detect whole words.

    Parameters:
    -----------
    term : string
        The term to search for.
    text : string
        The text to search within.

    Returns:
    -----------
    Boolean
        True if the term is found as a whole word in the text, False otherwise.
    """
    return re.search(rf'(?<!\S){re.escape(term)}(?!\S)', f' {text} ', re.IGNORECASE) is not None

def gather_file_info(file_types, separators, material_terms, conditions_terms, root_dir, append_missing, missing_txt_path=None):
   
    """
    Helper function that gathers file information from a specified root directory and its subdirectories.
    Extracts details from filenames and parent folder names to create a structured dataframe.
    The dataframe includes columns for file location, file name, date of scan, conditions, material, and time(duration).

    Parameters:
    -----------
    file_types : list of str
        List of file extensions to consider (e.g., ['.csv', '.0', '.dpt']).
    separators : list of str
        List of separator characters used in filenames and folder names (e.g., ['_', ' ', '-']).
    material_terms : list of str
        List of material terms to search for in filenames and folder names.
    conditions_terms : list of str
        List of condition terms to search for in filenames and folder names.
    root_dir : str
        The root directory to scan.
    """
    ## Info is first derived from parent folder names, then filenames if not found

    data = []
    grouped_files = {}
    ## Load existing dataframe and build set of processed files
    ## This prevents re-processing files that are already in the dataframe
    processed_files = set()
    csv_path = "dataframe.csv"
    if os.path.exists(csv_path):
        try:
            existing_dataframe = pd.read_csv(csv_path)
            for _, row in existing_dataframe.iterrows():
                processed_files.add((row["file location"], row["file name"]))
        except Exception:
            pass
    ## Track files already written to missing_txt_path to avoid duplicates
    already_missing = set()
    if missing_txt_path and os.path.exists(missing_txt_path):
        with open(missing_txt_path, 'r') as file:
            for line in file:
                try:
                    entry = eval(line.strip())
                    key = (entry.get("file location"), entry.get("file name"))
                    already_missing.add(key)
                except Exception:
                    continue
    for dirpath, _, filenames in os.walk(root_dir):
        parent_folder = os.path.basename(dirpath)
        for filename in filenames:
            ## Skip files already in dataframe
            if (dirpath, filename) in processed_files:
                continue
            first_col_list = []
            second_col_list = []
            ## Skip hidden files, system files, and files with 'ignore' in the name
            if filename.startswith('.'):
                continue
            if 'ignore' in filename.lower():
                continue
            ## Skip files that do not match the specified file types
            if not any(filename.lower().endswith(file_type.lower()) for file_type in file_types):
                continue
            ## Read first and second columns from the file, save as lists of floats
            file_path = os.path.join(dirpath, filename)
            try:
                with open(file_path, 'r') as data_file:
                    for line in data_file:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            try:
                                first_col_list.append(float(parts[0]))
                                second_col_list.append(float(parts[1]))
                            except ValueError:
                                continue
            except Exception:
                pass
            ## Group files by base name (filename without trailing _1, _2, etc. and without extension), case insensitive
            ## This finds replicate measurements and places their names in a dictionary
            base_name = filename
            for file_type in file_types:
                pattern = rf"_(\d)\{file_type}$"
                if re.search(pattern, filename):
                    base_name = re.sub(pattern, file_type, filename)
                    break
            base_name_lower = base_name.lower()
            group_key = base_name_lower
            grouped_files.setdefault(group_key, []).append(filename)
            ## Normalize filename and parent folder by removing file extension and replacing separators with spaces
            ## Makes for easier term-finding
            filename_no_ext = filename
            for file_type in file_types:
                if filename_no_ext.lower().endswith(file_type.lower()):
                    filename_no_ext = filename_no_ext[:-(len(file_type))]
            normalized_filename = filename_no_ext
            normalized_parent_folder = parent_folder
            for sep in separators:
                normalized_parent_folder = normalized_parent_folder.replace(sep, ' ')
                normalized_filename = normalized_filename.replace(sep, ' ')
            ## Extract date from parent folder or filename
            ## All date formats accepted, as long as they have 2 digits for month and day, and 4 digits for year, separated by hyphens
            date_match = re.search(r'(\d{2}-\d{2}-\d{4}|\d{4}-\d{2}-\d{2})', parent_folder)
            date = date_match.group(0) if date_match else None
            if not date:
                date_match_filename = re.search(r'(\d{2}-\d{2}-\d{4}|\d{4}-\d{2}-\d{2})', filename)
                date = date_match_filename.group(0) if date_match_filename else None
            conditions = next((term for term in conditions_terms if find_term(term, normalized_parent_folder)), None)
            ## Extract conditions from parent folder or filename
            if not conditions:
                conditions = next((term for term in conditions_terms if find_term(term, normalized_filename)), None)
            material = next((term for term in material_terms if find_term(term, normalized_parent_folder)), None)
            ## Extract material from parent folder or filename
            if not material:
                material = next((term for term in material_terms if find_term(term, normalized_filename)), None)
            time_match = re.search(r'(\d+)(?:H|hr)', parent_folder, re.IGNORECASE)
            ## Extract time(duration) from parent folder or filename
            if time_match:
                time = int(time_match.group(1))
            else:
                time_match = re.search(r'(\d+)(?:H|hr)', filename, re.IGNORECASE)
                time = int(time_match.group(1)) if time_match else None
            ## If condition is 'unexposed', set time to 0
            if conditions is not None and conditions.lower() == 'unexposed':
                time = 0
            missing_any = (date is None or conditions is None or material is None or time is None)
            ## Print a warning if any value is missing
            if missing_any:
                print(f"ValueError: Missing value for file '{filename}'. Results: date={date}, conditions={conditions}, material={material}, time={time}")
                ## Optionally write missing row to text file for later review-- default filename is missing_data.txt
                if missing_txt_path:
                    key = (dirpath, filename)
                    if key not in already_missing:
                        with open(missing_txt_path, 'a') as f:
                            f.write(str({
                                "file location": dirpath,
                                "file name": filename,
                                "date": date,
                                "conditions": conditions,
                                "material": material,
                                "time": time
                            }) + '\n')
                        already_missing.add(key)
            data_row = {
                "file location": dirpath,
                "file name": filename,
                "date": date,
                "conditions": conditions,
                "material": material,
                "time": time,
                "first column list": first_col_list,
                "second column list": second_col_list
            }
            if append_missing:
                #print(f"Appending row: {data_row}")
                data.append(data_row)
            else:
                if not missing_any:
                    #print(f"Appending row: {data_row}")
                    data.append(data_row)
    return data, grouped_files

def print_grouped_files(grouped_files):
    """
    Helper function that prints grouped files that share the same base name in the same directory, AKA repeated measurements.

    Parameters:
    -----------
    grouped_files : dict
        Dictionary with keys as (directory path, base filename) and values as lists of filenames.
    
    Returns:
    -----------
    print to console : string
        Prints grouped files with the same base name, indicating repeated measurements (replicates).
    """
    for base_name, file_list in grouped_files.items():
        if len(file_list) > 1:
            print(f"Replicate Files with base name '{base_name}': {file_list}")

def file_info_extractor(file_types=None, separators=None, material_terms=None, conditions_terms=None, root_dir=None, append_missing=None, save_missing_txt=None):
    """
    Main function to gather file information and print the resulting dataframe and grouped replicates. Handles user input for all parameters.

    Parameters:
    -----------
    file_types : str or None
        Comma-separated string of file extensions to consider (e.g., '.csv,.0,.dpt'). If None, prompts user for input.
    separators : str or None
        Comma-separated string of separator characters used in filenames and folder names (e.g., '_ , space , -'). If None, prompts user for input.
    material_terms : str or None
        Comma-separated string of material terms to search for in filenames and folder names. If None, prompts user for input.
    conditions_terms : str or None
        Comma-separated string of condition terms to search for in filenames and folder names. If None, prompts user for input.
    root_dir : str or None
        The root directory to scan. If None, prompts user for input.

    Returns:
    -----------
    DataFrame : pd.DataFrame
        DataFrame containing the gathered file information.
        Also prints the dataframe to console.
    Columns:
        file location : str or None
            Full path to the file.
        file name : str or None
            Name of the file.
        date : str or None
            Date of the scan extracted from filename or parent folder (format: MM-DD-YYYY or
        conditions : str or None
            Condition term extracted from filename or parent folder.
        material : str or None
            Material term extracted from filename or parent folder.
        time : float or None
            Duration in hours extracted from filename or parent folder.
    Grouped files : dict or None
        For purposes of identifying repeated measurements.
        Also prints grouped files to console.
    """

    ## User input: appending missing data to dataframe y/n
    if append_missing is None:
        append_missing = input("Do you want to append rows with missing values into the dataframe? (y/n): ").strip().lower()
        append_missing = True if append_missing == 'y' else False

    ## User input: saving missing data and the associated filename and path to a text file y/n
    if save_missing_txt is None:
        save_missing_txt = input("Do you want to save missing data rows to a text file for later review? (y/n): ").strip().lower()
        save_missing_txt = True if save_missing_txt == 'y' else False
    missing_txt_path = None
    if save_missing_txt:
        missing_txt_path = input("Enter the path for the missing data text file (default: missing_data.txt): ").strip()
        if not missing_txt_path:
            missing_txt_path = "missing_data.txt"

    ## User input: specify file types to scan (extensions, e.g. .csv,.0,.dpt)a
    if file_types is None:
        file_types = input("Enter file types to scan, separated by commas (e.g. .csv,.0,.dpt): ").strip()
    file_types = [ft.strip() for ft in file_types.split(',') if ft.strip()]
    #print("File types to scan:", file_types)

    ## User input: specify separators used in filenames and folder names (e.g. _ or space)       
    if separators is None:
        #print("Do not pick separators that are part of material or condition terms.")
        separators = input("Enter separator(s) used in filenames and folder names (e.g. _ or space): ").strip()
    #print("Separators:", separators)
    if separators.lower() == 'space':
        separators = [' ']
    separators = [sep.strip() for sep in separators.split(',') if sep.strip()]

    ## User input: specify material terms to find in filenames and folder names
    if material_terms is None:
        material_terms = input("Enter material terms to find, separated by commas (e.g. CPC,PPE,PO): ").strip().lower()
    material_terms = [term.strip() for term in material_terms.split(',') if term.strip()]
    #print("Material terms:", material_terms)

    ## User input: specify condition terms to find in filenames and folder names
    if conditions_terms is None:
        conditions_terms = input("Enter condition terms to find, separated by commas (e.g. A3,A4,A5): ").strip().lower()
    conditions_terms = [term.strip() for term in conditions_terms.split(',') if term.strip()]
    #print("Condition terms:", conditions_terms)

    if root_dir is None:
        root_dir = input("Enter the path to the folder to be scanned: ").strip()
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Directory not found: {root_dir}")
    #print("Root directory to scan:", root_dir)

    data, grouped_files = gather_file_info(file_types, separators, material_terms, conditions_terms, root_dir, append_missing, missing_txt_path)
    #print_grouped_files(grouped_files)
    ## Save or append to CSV file named dataframe.csv in the current working directory
    new_dataframe = pd.DataFrame(data)
    csv_path = "dataframe.csv"
    if os.path.exists(csv_path):
        ## Only append new rows, since gather_file_info already skips existing ones
        updated_dataframe = pd.concat([pd.read_csv(csv_path), new_dataframe], ignore_index=True)
        updated_dataframe.drop_duplicates(subset=["file location", "file name"], inplace=True)
        updated_dataframe.to_csv(csv_path, index=False)
    else:
        new_dataframe.to_csv(csv_path, index=False)
    #print("Finished gathering file info.")

## General Use:
## if __name__ == "__main__":
##    file_info_extractor()

## Specific Project Example Use:
if __name__ == "__main__":
    file_info_extractor(file_types=".dpt", separators="_", material_terms="CPC,PPE,PO,J-BOX#1,J-BOX#2,t-PVDF,t-PVF,o-PVF,PMMA", conditions_terms="A3,A4,A5,ARC,OPN,KKCE,0.5X,1X,2.5X,5X,unexposed", root_dir=r"Y:\5200\Packaging Reliability\Durability Tool\Ray Tracing and Activation Spectrum\ATR-FTIR Data", append_missing=False, save_missing_txt=True)