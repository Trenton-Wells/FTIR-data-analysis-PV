# Created: 9-15-2025
# Author: Trenton Wells
# Organization: NREL
# NREL Contact: trenton.wells@nrel.gov
# Personal Contact: trentonwells73@gmail.com

# This script gathers basic spectra file information from a specified root directory 
# and its subdirectories, extracting details from filenames and parent folder names to 
# create a structured DataFrame.
# The DataFrame includes columns for file location, file name, date of scan, 
# conditions, material, and time(duration).
# General usage note: Avoid using separators that are part of material or condition 
# terms, and avoid using extensions that are part of material or condition terms.
# Word recognition is case-insensitive and looks for whole words only (e.g., "PPE" will
#  be found, but not "PPE1" or "XPPE").

import os
import re
import pandas as pd
import numpy as np
import ast


def find_term(term, text):
    """
    Find whole word matches of a term in text, case-insensitive.

    Helper function to find whole word matches of a term in text, case-insensitive.
    Adds spaces around text to catch terms at the start/end, since term-finding uses 
    spaces on either side to detect whole words.

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
    return (
        re.search(rf"(?<!\S){re.escape(term)}(?!\S)", f" {text} ", re.IGNORECASE)
        is not None
    )


def _gather_file_info(
    FTIR_DataFrame,
    file_types,
    separators,
    material_terms,
    conditions_terms,
    directory,
    append_missing,
    access_subdirectories=True,
    track_replicates=False,
):
    """
    Gather file information from a specified root directory and its subdirectories.
    
    Helps file_info_extractor() create a structured DataFrame by extracting details
    from filenames and parent folder names.
    If "ignore" is in the filename, the file will be skipped.

    Parameters:
    -----------
    FTIR_DataFrame : pd.DataFrame
        The existing DataFrame to append new data to.
    file_types : list of str
        List of file extensions to consider (e.g., ['.csv', '.0', '.dpt']).
    separators : list of str
        List of separator characters used in filenames and folder names (e.g., ['_',
         ' ', '-']).
    material_terms : list of str
        List of material terms to search for in filenames and folder names.
    conditions_terms : list of str
        List of condition terms to search for in filenames and folder names.
    directory : str
        The root directory to scan.
    access_subdirectories : bool, optional
        When False, only descend into immediate subfolders of 'directory' whose
        names contain a date label (MM-DD-YYYY or YYYY-MM-DD). Files in the root
        directory are still scanned. Default is True (descend into all subfolders).
    track_replicates : bool, optional
        Whether to print groups of replicate files. Default is False.
    """
    # Info is first derived from parent folder names, then filenames if not found

    data = []
    grouped_files = {}
    # Build set of processed files
    # This prevents re-processing files that are already in the DataFrame
    processed_files = set()
    for _, row in FTIR_DataFrame.iterrows():
        processed_files.add((row["File Location"], row["File Name"]))
    for file_path, dirnames, filenames in os.walk(directory):
        # Optionally restrict traversal to only date-labeled immediate subfolders
        if not access_subdirectories:
            # Only filter when we are at the root directory level
            if os.path.normpath(file_path) == os.path.normpath(directory):
                date_regex = re.compile(r"(\d{2}-\d{2}-\d{4}|\d{4}-\d{2}-\d{2})")
                dirnames[:] = [d for d in dirnames if date_regex.search(d)]
        parent_folder = os.path.basename(file_path)
        for filename in filenames:
            # Skip files already in DataFrame
            if (file_path, filename) in processed_files:
                continue
            first_column_list = []
            second_column_list = []
            # Skip hidden files, system files, and files with 'ignore' in the name
            if filename.startswith("."):
                continue
            if "ignore" in filename.lower():
                continue
            # Skip files that do not match the specified file types
            if not any(
                filename.lower().endswith(file_type.lower()) for file_type in file_types
            ):
                continue
            # Read first and second columns from the file, save as list of floats
            full_file_path = os.path.join(file_path, filename)
            try:
                with open(full_file_path, "r") as data_file:
                    for line in data_file:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            try:
                                first_column_list.append(float(parts[0]))
                                second_column_list.append(float(parts[1]))
                            except ValueError:
                                continue
            except Exception:
                pass

            # Normalize filename and parent folder by removing file extension and 
            # replacing separators with spaces
            # Makes for easier term-finding
            filename_no_ext = filename
            for file_type in file_types:
                if filename_no_ext.lower().endswith(file_type.lower()):
                    filename_no_ext = filename_no_ext[: -(len(file_type))]
            normalized_filename = filename_no_ext
            normalized_parent_folder = parent_folder
            for sep in separators:
                normalized_parent_folder = normalized_parent_folder.replace(sep, " ")
                normalized_filename = normalized_filename.replace(sep, " ")
            # Extract date from parent folder or filename
            # All date formats accepted, as long as they have 2 digits for month and 
            # day, and 4 digits for year, separated by hyphens
            date_match = re.search(
                r"(\d{2}-\d{2}-\d{4}|\d{4}-\d{2}-\d{2})", parent_folder
            )
            date = date_match.group(0) if date_match else None
            if not date:
                date_match_filename = re.search(
                    r"(\d{2}-\d{2}-\d{4}|\d{4}-\d{2}-\d{2})", filename
                )
                date = date_match_filename.group(0) if date_match_filename else None
            conditions = next(
                (
                    term
                    for term in conditions_terms
                    if find_term(term, normalized_parent_folder)
                ),
                None,
            )
            # Extract conditions from parent folder or filename
            if not conditions:
                conditions = next(
                    (
                        term
                        for term in conditions_terms
                        if find_term(term, normalized_filename)
                    ),
                    None,
                )
            material = next(
                (
                    term
                    for term in material_terms
                    if find_term(term, normalized_parent_folder)
                ),
                None,
            )
            # Extract material from parent folder or filename
            if not material:
                material = next(
                    (
                        term
                        for term in material_terms
                        if find_term(term, normalized_filename)
                    ),
                    None,
                )
            time_match = re.search(r"(\d+)(?:H|hr)", parent_folder, re.IGNORECASE)
            # Extract time(duration) from parent folder or filename
            if time_match:
                time = int(time_match.group(1))
            else:
                time_match = re.search(r"(\d+)(?:H|hr)", filename, re.IGNORECASE)
                time = int(time_match.group(1)) if time_match else None
            # If condition is 'unexposed', set time to 0
            if conditions is not None and conditions.lower() == "unexposed":
                time = 0
            missing_any = (
                date is None or conditions is None or material is None or time is None
            )
            # Print a warning if any value is missing
            if missing_any:
                message = (f"ValueError: Missing value for file '{filename}'. Results: "
                f"date={date}, conditions={conditions}, material={material}, "
                f"time={time}"
                )
                print(message)

            data_row = {
                "File Location": file_path,
                "File Name": filename,
                "Date": date,
                "Conditions": conditions,
                "Material": material,
                "Time": time,
                "X-Axis": first_column_list,
                "Raw Data": second_column_list,
            }
            if append_missing:
                # print(f"Appending row: {data_row}")
                data.append(data_row)
            else:
                if not missing_any:
                    # print(f"Appending row: {data_row}")
                    data.append(data_row)

    # Group files by (material, conditions, time) after all files are processed

    # Optionally print replicate groups to the console
    if track_replicates is None:
        track_replicates = (
            input("Do you want to print groups of replicate files? (y/n): ")
            .strip()
            .lower()
        )
        track_replicates = True if track_replicates == "y" else False
    if track_replicates:
        replicate_groups = {}
        for row in data:
            mat = row.get("Material", None)
            cond = row.get("Conditions", None)
            t = row.get("Time", None)
            group_key = (mat, cond, t)
            # Store both file name and parent folder
            replicate_groups.setdefault(group_key, []).append(
                (row["File Name"], os.path.basename(row["File Location"]))
            )
        print("Replicate groups (groups with more than one file):")
        for group_key, file_list in replicate_groups.items():
            if len(file_list) > 1:
                # Format: [(file, parent_folder), ...]
                formatted = [
                    f"{fname} (parent folder: {pfolder})"
                    for fname, pfolder in file_list
                ]
                print(f"Replicate group {group_key}: {formatted}")

    return data, grouped_files


def file_info_extractor(
    FTIR_DataFrame,
    file_types=None,
    separators=None,
    material_terms=None,
    conditions_terms=None,
    directory=None,
    append_missing=None,
    access_subdirectories=True,
    track_replicates=False,
):
    """
    Use file info to create or update a structured DataFrame of scan details.
    
    Main function to gather file information and update the provided FTIR_DataFrame in 
    memory.

    Parameters:
    -----------
    FTIR_DataFrame : pd.DataFrame
        The existing DataFrame to append new data to (will be updated in memory).
    file_types : str or None
        Comma-separated string of file extensions to consider (e.g. '.csv,.0,.dpt'). If 
        None, prompts user for input.
    separators : str or None
        Comma-separated string of separator characters used in filenames and folder 
        names (e.g. '_ , space , -'). If None, prompts user for input.
    material_terms : str or None
        Comma-separated string of material terms to search for in filenames and folder 
        names. If None, prompts user for input.
    conditions_terms : str or None
        Comma-separated string of condition terms to search for in filenames and folder 
        names. If None, prompts user for input.
    directory : str or None
        The root directory to scan. If None, prompts user for input.
    append_missing : bool or None
        Whether to append rows with missing values. If None, prompts user for input.
    access_subdirectories : bool or None
        If False, only descend into immediate subfolders of 'directory' whose names
        contain a date label (MM-DD-YYYY or YYYY-MM-DD). If None, prompts user for
        input. Default is True.
    track_replicates : bool or None
        Whether to print groups of replicate files. If None, prompts user for input.

    Returns:
    --------
    FTIR_DataFrame : pd.DataFrame
        The updated DataFrame with new file info appended.
    """
    # Ensure required columns exist
    required_columns = [
        "File Location",
        "File Name",
        "Date",
        "Conditions",
        "Material",
        "Time",
        "X-Axis",
        "Raw Data",
        "Baseline Function",
        "Baseline Parameters",
        "Baseline",
        "Baseline-Corrected Data",
        "Normalization Peak Wavenumber",
        "Normalized and Corrected Data",
    ]
    for column in required_columns:
        if column not in FTIR_DataFrame.columns:
            FTIR_DataFrame[column] = None

    # Cast columns to correct dtype
    # String columns
    string_cols = [
        "File Location",
        "File Name",
        "Date",
        "Conditions",
        "Material",
        "Baseline Function",
        "Baseline Parameters",
    ]
    for col in string_cols:
        if col in FTIR_DataFrame.columns:
            FTIR_DataFrame[col] = FTIR_DataFrame[col].astype("string")

    # Integer columns
    if "Time" in FTIR_DataFrame.columns:
        FTIR_DataFrame["Time"] = pd.to_numeric(
            FTIR_DataFrame["Time"], errors="coerce"
        ).astype("Int64")

    # Dictionary columns
    if "Baseline Parameters" in FTIR_DataFrame.columns:
        def _to_dict(val):
            if isinstance(val, dict) or pd.isnull(val):
                return val
            if isinstance(val, str):
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
            return val
        FTIR_DataFrame["Baseline Parameters"] = FTIR_DataFrame["Baseline Parameters"].apply(_to_dict)
    # Float columns
    if "Normalization Peak Wavenumber" in FTIR_DataFrame.columns:
        FTIR_DataFrame["Normalization Peak Wavenumber"] = pd.to_numeric(
            FTIR_DataFrame["Normalization Peak Wavenumber"], errors="coerce"
        ).astype("float")

    # Columns that are lists of floats (leave as object, but ensure lists of floats)
    list_float_cols = [
        "X-Axis",
        "Raw Data",
        "Baseline",
        "Baseline-Corrected Data",
        "Normalized and Corrected Data",
    ]
    for col in list_float_cols:
        if col in FTIR_DataFrame.columns:

            def to_float_list(val):
                if isinstance(val, list):
                    return [float(x) for x in val]
                elif pd.isnull(val):
                    return val
                try:
                    import ast

                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        return [float(x) for x in parsed]
                except Exception:
                    pass

                return val

            FTIR_DataFrame[col] = FTIR_DataFrame[col].apply(to_float_list)

    # Option for if DataFrame should append rows with missing values or not
    if append_missing is None:
        message = (f"Do you want to append rows with missing values into the DataFrame?"
                   f" (y/n): "
                   )
        append_missing = (
            input(message)
            .strip()
            .lower()
        )
        append_missing = True if append_missing == "y" else False

    # Option for whether to access non-date-labeled subdirectories
    if access_subdirectories is None:
        message = (
            "Limit scan to only subfolders with date labels (MM-DD-YYYY or YYYY-MM-DD)? (y/n): "
        )
        resp = input(message).strip().lower()
        access_subdirectories = False if resp == "y" else True

    # Get file types
    if file_types is None:
        file_types = input(
            "Enter file types to scan, separated by commas (e.g. .csv,.0,.dpt): "
        ).strip()
    file_types = [ft.strip() for ft in file_types.split(",") if ft.strip()]

    # Get separators
    if separators is None:
        separators = input(
            "Enter separator(s) used in filenames and folder names (e.g. _ or space): "
        ).strip()
    if separators.lower() == "space":
        separators = [" "]
    separators = [sep.strip() for sep in separators.split(",") if sep.strip()]

    # Get material terms
    if material_terms is None:
        material_terms = (
            input(
                "Enter material terms to find, separated by commas (e.g. CPC,PPE,PO): "
            )
            .strip()
            .lower()
        )
    material_terms = [
        term.strip() for term in material_terms.split(",") if term.strip()
    ]

    # Get condition terms
    if conditions_terms is None:
        conditions_terms = (
            input(
                "Enter condition terms to find, separated by commas (e.g. A3,A4,A5): "
            )
            .strip()
            .lower()
        )
    conditions_terms = [
        term.strip() for term in conditions_terms.split(",") if term.strip()
    ]

    if directory is None:
        directory = input("Enter the path to the folder to be scanned: ").strip()
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

    # Gather new file info
    data, grouped_files = _gather_file_info(
        FTIR_DataFrame=FTIR_DataFrame,
        file_types=file_types,
        separators=separators,
        material_terms=material_terms,
        conditions_terms=conditions_terms,
        directory=directory,
        append_missing=append_missing,
        access_subdirectories=access_subdirectories,
        track_replicates=track_replicates,
    )

    # Append new data to FTIR_DataFrame
    if data:
        new_data = pd.DataFrame(data)
        FTIR_DataFrame = pd.concat([FTIR_DataFrame, new_data], ignore_index=True)
        FTIR_DataFrame.drop_duplicates(
            subset=["File Location", "File Name"], inplace=True
        )
        FTIR_DataFrame.reset_index(drop=True, inplace=True)

    return FTIR_DataFrame


# General Use:
# if __name__ == "__main__":
#    file_info_extractor()
