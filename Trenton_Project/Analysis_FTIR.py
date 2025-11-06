# Created: 9-23-2025
# Author: Trenton Wells
# Organization: NREL
# NREL Contact: trenton.wells@nrel.gov
# Personal Contact: trentonwells73@gmail.com
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from pybaselines.whittaker import arpls
from pybaselines.spline import irsqr
from pybaselines.classification import fabc
from scipy.interpolate import CubicSpline
from pybaselines.utils import optimize_window
from scipy.signal import find_peaks
import ast
import ipywidgets as widgets
import plotly.graph_objs as go
from IPython.display import display, clear_output
from math import ceil
import importlib
from plotly.subplots import make_subplots
import threading
from lmfit.models import PseudoVoigtModel
import time
import html
import json
import contextlib


def batch_rename_files(
    directory=None,
    replace_spaces=None,
    iso_date_rename=None,
    file_rename=None,
    character_to_use=None,
    pairs_input=None,
):
    """
    Change file & folder names in a directory by replacing spaces, dates, and words.

    Scans a directory and its subdirectories to rename files by replacing spaces and/or
    specified words in filenames. Folder names will not be changed for space/word
    replacement. If ISO date renaming is enabled, dates in both filenames and folder
    names will be updated. Recommended to use this tool if file names have inconsistent
    naming conventions that may cause issues.

    Parameters:
    -----------
        directory (str): Directory to scan. If None, prompts user for input.

    Returns:
    -----------
        Renamed files in place; prints changes to console.
    """

    def _date_change_ISO(directory):
        """
        Rename dates in filenames and folder names in the given directory to ISO format.

        ISO format is YYYY-MM-DD. This is an international standard date format and has
        the added benefit of sorting chronologically when sorted alphabetically.

        Parameters:
        -----------
        directory (str): Directory to scan. Must be a valid directory path.

        Returns:
        -----------
        Renamed files and folders in place; prints changes to console.
        """

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    print(f"Scanning directory: {directory}")

    # Option to replace spaces in filenames with different separator character
    if replace_spaces is None:
        replace_spaces_input = (
            input("Do you want to replace spaces in filenames? (y/n): ").strip().lower()
        )
        if replace_spaces_input == "y":
            replace_spaces = True
    if replace_spaces:
        if character_to_use is None:
            character_to_use = input(
                "Enter the separator to use instead of spaces (e.g. _): "
            ).strip()
        print("Replacing spaces now...")
        for root, dirs, files in os.walk(directory):
            for current_filename in files:
                if " " in current_filename:
                    old_filepath = os.path.join(root, current_filename)
                    new_filename = current_filename.replace(" ", character_to_use)
                    new_filepath = os.path.join(root, new_filename)
                    print(f"Renaming: {old_filepath} to {new_filepath}")
                    os.rename(old_filepath, new_filepath)
        print("Space replacement complete.")
    else:
        print("No spaces will be replaced in filenames.")

    # Option to batch rename dates to ISO format (YYYY-MM-DD)
    if iso_date_rename is None:
        message = (
            f"Do you want to convert all dates in filenames to ISO format"
            f" (YYYY-MM-DD)? (y/n): "
        )
        iso_date_input = input(message).strip().lower()
        if iso_date_input == "y":
            iso_date_rename = True
    if iso_date_rename:
        _date_change_ISO(directory)
    else:
        print("No dates will be changed in filenames.")

    if file_rename:
        if pairs_input is None:
            message = (
                f"Enter words to find and their replacements as comma-separated"
                f" pairs (e.g. old1:new1,old2:new2): "
            )
            pairs_input = input(message).strip()
        word_pairs = [pair.split(":") for pair in pairs_input.split(",") if ":" in pair]
        print("Renaming files by replacing specified words...")
        for root, dirs, files in os.walk(directory):
            for current_filename in files:
                new_filename = current_filename
                for word_to_find, word_to_replace in word_pairs:
                    new_filename = new_filename.replace(word_to_find, word_to_replace)
                if new_filename != current_filename:
                    old_filepath = os.path.join(root, current_filename)
                    new_filepath = os.path.join(root, new_filename)
                    print(f"Renaming: {old_filepath} to {new_filepath}")
                    os.rename(old_filepath, new_filepath)
        print("Batch word replacement complete.")
    else:
        print("No words will be replaced in filenames.")


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

    # --- Helper functions (scoped to file_info_extractor) --- #
    def _find_term(term, text):
        """
        Find whole word matches of a term in text, case-insensitive.

        Adds spaces around text to catch terms at the start/end, since term-finding uses
        spaces on either side to detect whole words.
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
        """
        # Info is first derived from parent folder names, then filenames if not found

        data = []
        grouped_files = {}
        print("Scanning directory for spectral files...")
        # Build set of processed files
        # This prevents re-processing files that are already in the DataFrame
        processed_files = set()
        for _, row in FTIR_DataFrame.iterrows():
            processed_files.add((row["File Location"], row["File Name"]))
        # Pre-count candidate files for progress tracking
        total_candidates = 0
        for _file_path, _dirnames, _filenames in os.walk(directory):
            if not access_subdirectories:
                if os.path.normpath(_file_path) == os.path.normpath(directory):
                    date_regex = re.compile(r"(\d{2}-\d{2}-\d{4}|\d{4}-\d{2}-\d{2})")
                    _dirnames[:] = [d for d in _dirnames if date_regex.search(d)]
            for _filename in _filenames:
                # Apply the same core file filters as the main loop
                if (_file_path, _filename) in processed_files:
                    continue
                if _filename.startswith("."):
                    continue
                if "ignore" in _filename.lower():
                    continue
                if not any(
                    _filename.lower().endswith(file_type.lower())
                    for file_type in file_types
                ):
                    continue
                total_candidates += 1
        print(f"Found {total_candidates} spectral files to parse…")

        parsed_count = 0

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
                    filename.lower().endswith(file_type.lower())
                    for file_type in file_types
                ):
                    continue
                # Update progress for each candidate file encountered
                parsed_count += 1
                # Print a single-line progress indicator
                try:
                    pct = (
                        (parsed_count / total_candidates * 100)
                        if total_candidates
                        else 0
                    )
                    print(
                        f"\rParsed {parsed_count}/{total_candidates} files ({pct:.1f}%)",
                        end="",
                        flush=True,
                    )
                except Exception:
                    pass
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
                    normalized_parent_folder = normalized_parent_folder.replace(
                        sep, " "
                    )
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
                        if _find_term(term, normalized_parent_folder)
                    ),
                    None,
                )
                # Extract conditions from parent folder or filename
                if not conditions:
                    conditions = next(
                        (
                            term
                            for term in conditions_terms
                            if _find_term(term, normalized_filename)
                        ),
                        None,
                    )
                material = next(
                    (
                        term
                        for term in material_terms
                        if _find_term(term, normalized_parent_folder)
                    ),
                    None,
                )
                # Extract material from parent folder or filename
                if not material:
                    material = next(
                        (
                            term
                            for term in material_terms
                            if _find_term(term, normalized_filename)
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
                    date is None
                    or conditions is None
                    or material is None
                    or time is None
                )
                # Print a warning if any value is missing
                if missing_any:
                    message = (
                        f"ValueError: Missing value for file '{filename}'. Results: "
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
                    data.append(data_row)
                else:
                    if not missing_any:
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

        # Finish progress line with newline for clean output
        try:
            print("\nBasic file information extraction complete.")
        except Exception:
            pass

        return data, grouped_files

    # Ensure required columns exist
    required_columns = [
        "File Location",
        "File Name",
        "Date",
        "Quality",
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
        "Peak Wavenumbers",
        "Peak Absorbances",
        "Deconvolution Results",
        "Time-Series Fit Results",
    ]
    for column in required_columns:
        if column not in FTIR_DataFrame.columns:
            FTIR_DataFrame[column] = None
    # Ensure 'Quality' column exists and defaults to 'good'; migrate legacy 'quality' if present
    try:
        if (
            "Quality" not in FTIR_DataFrame.columns
            and "quality" in FTIR_DataFrame.columns
        ):
            try:
                FTIR_DataFrame.rename(columns={"quality": "Quality"}, inplace=True)
            except Exception:
                pass
        # If both exist, coalesce values into 'Quality' and drop legacy 'quality'
        if "Quality" in FTIR_DataFrame.columns and "quality" in FTIR_DataFrame.columns:
            try:
                q_legacy = FTIR_DataFrame["quality"].astype("string")
                q_new = FTIR_DataFrame["Quality"].astype("string")
                FTIR_DataFrame["Quality"] = q_new.fillna(q_legacy).fillna("good")
                with contextlib.suppress(Exception):
                    FTIR_DataFrame.drop(columns=["quality"], inplace=True)
            except Exception:
                pass
        if "Quality" not in FTIR_DataFrame.columns:
            FTIR_DataFrame["Quality"] = "good"
        else:
            FTIR_DataFrame["Quality"] = (
                FTIR_DataFrame["Quality"].astype("string").fillna("good")
            )
    except Exception:
        try:
            FTIR_DataFrame["Quality"] = "good"
        except Exception:
            pass

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

        FTIR_DataFrame["Baseline Parameters"] = FTIR_DataFrame[
            "Baseline Parameters"
        ].apply(_to_dict)
    if "Deconvolution Results" in FTIR_DataFrame.columns:

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

        FTIR_DataFrame["Deconvolution Results"] = FTIR_DataFrame[
            "Deconvolution Results"
        ].apply(_to_dict)
    if "Time-Series Fit Results" in FTIR_DataFrame.columns:

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

        FTIR_DataFrame["Time-Series Fit Results"] = FTIR_DataFrame[
            "Time-Series Fit Results"
        ].apply(_to_dict)
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
        "Peak Wavenumbers",
        "Peak Absorbances",
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

    # Ensure list-like columns are stored with object dtype (per-row lists)
    try:
        existing_list_cols = [c for c in list_float_cols if c in FTIR_DataFrame.columns]
        if existing_list_cols:
            FTIR_DataFrame[existing_list_cols] = FTIR_DataFrame[
                existing_list_cols
            ].astype(object)
    except Exception:
        # Fallback: coerce individually if bulk coercion fails
        for c in list_float_cols:
            if c in FTIR_DataFrame.columns:
                try:
                    FTIR_DataFrame[c] = FTIR_DataFrame[c].astype(object)
                except Exception:
                    pass

    # Option for if DataFrame should append rows with missing values or not
    if append_missing is None:
        message = (
            f"Do you want to append rows with missing values into the DataFrame?"
            f" (y/n): "
        )
        append_missing = input(message).strip().lower()
        append_missing = True if append_missing == "y" else False

    # Option for whether to access non-date-labeled subdirectories
    if access_subdirectories is None:
        message = (
            f"Limit scan to only subfolders with date labels"
            f"(MM-DD-YYYY or YYYY-MM-DD)? (y/n): "
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
        # Ensure 'Quality' exists for new rows; migrate legacy 'quality' if present
        try:
            if "Quality" not in new_data.columns and "quality" in new_data.columns:
                try:
                    new_data.rename(columns={"quality": "Quality"}, inplace=True)
                except Exception:
                    pass
            # If both exist, coalesce and drop legacy 'quality'
            if "Quality" in new_data.columns and "quality" in new_data.columns:
                try:
                    q_legacy = new_data["quality"].astype("string")
                    q_new = new_data["Quality"].astype("string")
                    new_data["Quality"] = q_new.fillna(q_legacy).fillna("good")
                    with contextlib.suppress(Exception):
                        new_data.drop(columns=["quality"], inplace=True)
                except Exception:
                    pass
            if "Quality" not in new_data.columns:
                new_data["Quality"] = "good"
            else:
                new_data["Quality"] = (
                    new_data["Quality"].astype("string").fillna("good")
                )
        except Exception:
            try:
                new_data["Quality"] = "good"
            except Exception:
                pass
        FTIR_DataFrame = pd.concat([FTIR_DataFrame, new_data], ignore_index=True)
        FTIR_DataFrame.drop_duplicates(
            subset=["File Location", "File Name"], inplace=True
        )
        FTIR_DataFrame.reset_index(drop=True, inplace=True)
        # Reorder columns to place 'Quality' between 'Date' and 'Conditions' if possible
        try:
            cols = list(FTIR_DataFrame.columns)
            if (
                "Date" in cols
                and "Quality" in cols
                and ("Conditions" in cols or "Condition" in cols)
            ):
                cols.remove("Quality")
                insert_pos = cols.index("Date") + 1
                cols.insert(insert_pos, "Quality")
                FTIR_DataFrame = FTIR_DataFrame[cols]
        except Exception:
            pass

    return FTIR_DataFrame


# ---- Validation helpers for clearer, user-friendly errors ---- #
def _require_columns(df, columns, context="DataFrame"):
    """
    Ensure the DataFrame contains the given columns, else raise a KeyError.

    Helper function.
    """
    if df is None:
        raise ValueError(f"{context} is None. A valid pandas DataFrame is required.")
    try:
        cols = list(df.columns)
    except Exception:
        raise TypeError(
            f"{context} must be a pandas DataFrame with a 'columns' " f"attribute."
        )
    missing = [c for c in columns if c not in cols]
    if missing:
        raise KeyError(
            f"Missing required column(s) in {context}: {missing}. Available columns: "
            f"{cols}"
        )


def _safe_literal_eval(val, value_name="value"):
    """
    Safely parse string representations of Python literals, with descriptive errors.

    Helper function.
    """
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception as e:
            raise ValueError(
                f"Could not parse {value_name} from string: {val!r}. " f"Error: {e}"
            )
    return val


# --------------------------- Quality helpers ---------------------------- #
def _quality_column_name(df):
    try:
        # Prefer canonical 'Quality'; migrate legacy 'quality' to 'Quality' when found
        if "Quality" in df.columns:
            return "Quality"
        if "quality" in df.columns:
            try:
                df.rename(columns={"quality": "Quality"}, inplace=True)
            except Exception:
                pass
            return "Quality"
    except Exception:
        pass
    # Default canonical name
    return "Quality"


def _quality_series(df):
    try:
        col = _quality_column_name(df)
        if col in df.columns:
            return df[col].astype("string").str.lower().fillna("good")
    except Exception:
        pass
    # Default to all 'good' if column missing
    try:
        return pd.Series(["good"] * len(df), index=df.index, dtype="string")
    except Exception:
        return pd.Series(["good"] * len(df), index=df.index)


def _quality_good_mask(df):
    try:
        qs = _quality_series(df)
        return qs != "bad"
    except Exception:
        return pd.Series([True] * len(df), index=getattr(df, "index", None))


# ----------------------- Session selection persistence ----------------------- #
# Persist last-used selections across interactive tools within this module.
_SESSION_SELECTIONS = {"material": "any", "conditions": "any", "time": "any"}

# Track active widgets/figures created by try_baseline to ensure clean re-entry
_TB_WIDGETS = []


def _get_session_defaults():
    """Return a shallow copy of the last-used material/conditions/time selections."""
    try:
        return dict(_SESSION_SELECTIONS)
    except Exception:
        return {"material": "any", "conditions": "any", "time": "any"}


def _set_session_selection(material=None, conditions=None, time=None):
    """Update persisted selections; ignores None. Accepts 'any' or concrete values.

    - material, conditions: coerced to str.
    - time: left as-is if numeric; otherwise coerced to str (including 'any').
    """
    try:
        if material is not None:
            _SESSION_SELECTIONS["material"] = str(material)
        if conditions is not None:
            _SESSION_SELECTIONS["conditions"] = str(conditions)
        if time is not None:
            # keep numeric times numeric when possible for easier matching
            try:
                if isinstance(time, str) and time.strip().lower() == "any":
                    _SESSION_SELECTIONS["time"] = "any"
                else:
                    _SESSION_SELECTIONS["time"] = int(time)
            except Exception:
                _SESSION_SELECTIONS["time"] = str(time)
    except Exception:
        # Best-effort; do not raise in UX path
        pass

def _ensure_1d_numeric_array(name, seq):
    """Coerce a sequence into a 1D float numpy array; raise clear error if invalid."""
    seq = _safe_literal_eval(seq, value_name=name)
    try:
        arr = np.asarray(seq, dtype=float)
    except Exception as e:
        raise ValueError(
            f"{name} must be a sequence of numbers. Got: "
            f"{type(seq).__name__}. Error: {e}"
        )
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D. Got array with shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} is empty. A non-empty sequence is required.")
    return arr


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
    if parameter_str is None:
        return {}
    if not isinstance(parameter_str, str):
        raise TypeError(
            f"parameter_str must be a string of key=value pairs, got "
            f"{type(parameter_str).__name__}."
        )
    tokens = [tok.strip() for tok in parameter_str.split(",") if tok.strip()]
    if not tokens:
        return {}

    def _coerce_scalar(s):
        low = s.strip().lower()
        if low in {"none", "null", "nan"}:
            return None
        if low in {"true", "false"}:
            return low == "true"
        try:
            if any(ch in s for ch in ".eE"):
                return float(s)
            return int(s)
        except Exception:
            try:
                return ast.literal_eval(s)
            except Exception:
                return s

    parameter_dictionary = {}
    for item in tokens:
        if "=" not in item:
            raise ValueError(
                f"Invalid parameter token {item!r}. Expected format 'key=value'. Full "
                f"string: {parameter_str!r}"
            )
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Found empty parameter name in token {item!r}.")
        parameter_dictionary[key] = _coerce_scalar(value)
    return parameter_dictionary


def _get_default_parameters(function_name):
    """
    Input the name of a baseline function and return its default parameters as a
    dictionary.

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
        "ARPLS": {
            "lam": 1e5,
            "diff_order": 2,
            "max_iter": 50,
            "tol": 1e-3,
            "weights": None,
        },
        "IRSQR": {
            "lam": 1e6,
            "quantile": 0.05,
            "num_knots": 100,
            "spline_degree": 3,
            "diff_order": 3,
            "max_iter": 100,
            "tol": 1e-6,
            "weights": None,
            "eps": None,
        },
        "FABC": {
            "lam": 1e6,
            "scale": None,
            "num_std": 3.0,
            "diff_order": 2,
            "min_length": 2,
            "weights": None,
            "weights_as_mask": False,
            "pad_kwargs": None,
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
        if "lam" in parameters:
            parameters["lam"] = float(parameters["lam"])
        if "diff_order" in parameters:
            parameters["diff_order"] = int(parameters["diff_order"])
        if "max_iter" in parameters:
            parameters["max_iter"] = int(parameters["max_iter"])
        if "tol" in parameters:
            parameters["tol"] = float(parameters["tol"])
        if "weights" in parameters:
            if str(parameters["weights"]).lower() not in ["none", "null", ""]:
                try:
                    parameters["weights"] = ast.literal_eval(parameters["weights"])
                except Exception:
                    pass
    elif function == "IRSQR":
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
        if "max_iter" in parameters:
            parameters["max_iter"] = int(parameters["max_iter"])
        if "tol" in parameters:
            parameters["tol"] = float(parameters["tol"])
        if "weights" in parameters:
            if str(parameters["weights"]).lower() not in ["none", "null", ""]:
                try:
                    parameters["weights"] = ast.literal_eval(parameters["weights"])
                except Exception:
                    pass
        if "eps" in parameters:
            if str(parameters["eps"]).lower() not in ["none", "null", ""]:
                try:
                    parameters["eps"] = float(parameters["eps"])
                except Exception:
                    pass
    elif function == "FABC":
        if "lam" in parameters:
            parameters["lam"] = float(parameters["lam"])
        if str(parameters["scale"]).lower() not in ["none", "null", ""]:
            try:
                parameters["scale"] = ast.literal_eval(parameters["scale"])
            except Exception:
                pass
        if "num_std" in parameters:
            parameters["num_std"] = float(parameters["num_std"])
        if "diff_order" in parameters:
            parameters["diff_order"] = int(parameters["diff_order"])
        if "min_length" in parameters:
            parameters["min_length"] = int(parameters["min_length"])
        if "weights" in parameters:
            if str(parameters["weights"]).lower() not in ["none", "null", ""]:
                try:
                    parameters["weights"] = ast.literal_eval(parameters["weights"])
                except Exception:
                    pass
        if "weights_as_mask" in parameters:
            if str(parameters["weights_as_mask"]).lower() in ["true"]:
                parameters["weights_as_mask"] = True
            else:
                parameters["weights_as_mask"] = False
        if "pad_kwargs" in parameters:
            if parameters["pad_kwargs"] is not None:
                try:
                    parameters["pad_kwargs"] = ast.literal_eval(
                        parameters["pad_kwargs"]
                    )
                except Exception:
                    pass
    return parameters


def baseline_correction(FTIR_DataFrame, materials="any"):
    """
    Apply baseline correction to spectra in the DataFrame for selected materials or all.

    Parameters
    ----------
    FTIR_DataFrame : pd.DataFrame
        The DataFrame to update in-place.
    materials : str or list
        Material(s) to filter and process. Use 'any' (case-insensitive) to process all
        rows.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame.
    """
    # Validate required columns early for clearer errors
    _require_columns(
        FTIR_DataFrame,
        ["Material", "X-Axis", "Raw Data"],
        context="FTIR_DataFrame (baseline_correction)",
    )

    # Ensure destination columns exist and are object dtype (for per-row lists)
    for col in ("Baseline", "Baseline-Corrected Data"):
        if col not in FTIR_DataFrame.columns:
            FTIR_DataFrame[col] = None
    # Coerce sequence-holding columns to object dtype to avoid shape/broadcast issues
    try:
        FTIR_DataFrame[["Baseline", "Baseline-Corrected Data"]] = FTIR_DataFrame[
            ["Baseline", "Baseline-Corrected Data"]
        ].astype(object)
    except Exception:
        # Fall back to individual-coercion if slice fails (e.g., missing one column)
        for _col in ("Baseline", "Baseline-Corrected Data"):
            if _col in FTIR_DataFrame.columns:
                try:
                    FTIR_DataFrame[_col] = FTIR_DataFrame[_col].astype(object)
                except Exception:
                    pass

    # Build mask for materials
    if isinstance(materials, str):
        if materials.strip().lower() == "any":
            mask = pd.Series([True] * len(FTIR_DataFrame), index=FTIR_DataFrame.index)
        else:
            material_list = [m.strip() for m in materials.split(",") if m.strip()]
            mask = FTIR_DataFrame["Material"].astype(str).isin(material_list)
    elif isinstance(materials, (list, tuple)):
        material_list = [str(m).strip() for m in materials if str(m).strip()]
        mask = FTIR_DataFrame["Material"].astype(str).isin(material_list)
    else:
        mask = pd.Series([True] * len(FTIR_DataFrame), index=FTIR_DataFrame.index)

    # Exclude rows marked as bad quality
    try:
        mask = mask & _quality_good_mask(FTIR_DataFrame)
    except Exception:
        pass

    updated = 0
    skipped = 0
    for idx in FTIR_DataFrame.index[mask]:
        row = FTIR_DataFrame.loc[idx]
        baseline_name = row.get("Baseline Function", None)
        if baseline_name is None or str(baseline_name).strip() == "":
            print(f"Row {idx}: Missing 'Baseline Function'; skipping.")
            skipped += 1
            continue

        # Robustly parse parameters (dict or string) and merge with defaults
        raw_params = row.get("Baseline Parameters", {})
        if isinstance(raw_params, dict):
            params = raw_params.copy()
        elif isinstance(raw_params, str) and raw_params.strip():
            try:
                maybe = ast.literal_eval(raw_params)
                params = (
                    maybe if isinstance(maybe, dict) else _parse_parameters(raw_params)
                )
            except Exception:
                params = _parse_parameters(raw_params)
        else:
            params = {}

        func_name = str(baseline_name).strip().upper()
        defaults = _get_default_parameters(func_name)
        params = {**defaults, **params}
        params = _cast_parameter_types(func_name, params)

        # Parse data arrays
        try:
            y_data = (
                ast.literal_eval(row.get("Raw Data"))
                if isinstance(row.get("Raw Data"), str)
                else row.get("Raw Data")
            )
        except Exception:
            y_data = row.get("Raw Data")
        try:
            x_axis = (
                ast.literal_eval(row.get("X-Axis"))
                if isinstance(row.get("X-Axis"), str)
                else row.get("X-Axis")
            )
        except Exception:
            x_axis = row.get("X-Axis")
        if y_data is None or x_axis is None:
            print(f"Row {idx}: Missing X-Axis or Raw Data; skipping.")
            skipped += 1
            continue

        baseline = None
        baseline_corrected = None
        try:
            if func_name == "ARPLS":
                result = arpls(y_data, **params)
            elif func_name == "IRSQR":
                result = (
                    irsqr(y_data, **params, x_data=x_axis)
                    if "x_data" not in params
                    else irsqr(y_data, **params)
                )
            elif func_name == "FABC":
                result = fabc(y_data, **params)
            elif func_name == "MANUAL":
                anchor_points = params.get("anchor_points", [])
                if not anchor_points:
                    raise ValueError("MANUAL baseline requires 'anchor_points'.")
                # indices of closest x to each anchor point
                anchor_indices = [
                    min(range(len(x_axis)), key=lambda i: abs(x_axis[i] - ap))
                    for ap in anchor_points
                ]
                y_anchor = [y_data[i] for i in anchor_indices]
                baseline = CubicSpline(x=anchor_points, y=y_anchor, extrapolate=True)(
                    x_axis
                )
            else:
                raise ValueError(f"Unknown baseline function: {baseline_name}")

            if baseline is None:
                # Normalize return type from pybaselines
                if isinstance(result, tuple):
                    baseline = result[0]
                elif isinstance(result, dict):
                    baseline = result.get("baseline", None)
                else:
                    baseline = result
            baseline = np.asarray(baseline, dtype=float)
            y_arr = np.asarray(y_data, dtype=float)
            if baseline.shape != y_arr.shape:
                raise ValueError(
                    f"Baseline shape {baseline.shape} does not match data shape "
                    f"{y_arr.shape}."
                )
            baseline_corrected = (y_arr - baseline).astype(float)
        except Exception as e:
            print(f"Row {idx}: Error computing baseline: {e}")
            print(f" - Baseline Function: {baseline_name}")
            print(f" - Baseline Parameters: {params}")
            skipped += 1
            continue

        # Save results back to DataFrame
        baseline = np.asarray(baseline, dtype=float)
        baseline_corrected = np.asarray(baseline_corrected, dtype=float)
        if baseline.ndim > 1:
            baseline = baseline.flatten()
        if baseline_corrected.ndim > 1:
            baseline_corrected = baseline_corrected.flatten()
        # Store as plain Python lists for portability/CSV round-trip
        FTIR_DataFrame.at[idx, "Baseline"] = baseline.tolist()
        FTIR_DataFrame.at[idx, "Baseline-Corrected Data"] = baseline_corrected.tolist()
        updated += 1

    return FTIR_DataFrame


def plot_grouped_spectra(
    FTIR_DataFrame,
    materials=None,
    conditions=None,
    times=None,
    raw_data=True,
    baseline=False,
    baseline_corrected=False,
    normalized=False,
    separate_plots=False,
    include_replicates=True,
    mark_bad=None,
    mark_good=None,
    show_bad=False,
    interactive=True,
):
    """
    Plot grouped spectra based on material, condition, and time.

    Accepts lists or 'any' for each category.

    Parameters
    ----------
    FTIR_DataFrame : pd.DataFrame
        The DataFrame containing the spectral data.
    material : str, list, or 'any'
        The material(s) to filter by, or 'any' to include all.
    condition : str, list, or 'any'
        The condition(s) to filter by, or 'any' to include all.
    time : str, int, list, or 'any'
            try:
                filtered = filtered[_quality_good_mask(filtered)]
            except Exception:
                pass
            if filtered.empty:
    raw_data : bool, optional
        Whether to plot the raw data (default is True).
    baseline : bool, optional
        Whether to plot the baseline (default is False).
            try:
                filtered = filtered[_quality_good_mask(filtered)]
            except Exception:
                pass
    baseline_corrected : bool, optional
        Whether to plot the baseline-corrected data (default is False).
    normalized : bool, optional
        Whether to plot the normalized-and-corrected data from the column
        'Normalized and Corrected Data' (default is False).
    separate_plots : bool, optional
        Whether to create separate plots for each spectrum (default is False).
    include_replicates : bool, optional
        Whether to include all replicates or just the first of each group (default is
        True).

    Returns
    -------
    None

    mark_bad (optional)
        If provided, mark rows as bad quality:
        - 'all': mark all plotted rows as bad
        - list of DataFrame indices
        - list of file names (matches 'File Name')
    mark_good (optional)
        If provided, mark rows as good quality (same accepted value forms as mark_bad).
    show_bad : bool, optional
        When True, include spectra marked as bad in the plots; when False (default),
        bad spectra are excluded.
    """
    # Interactive widget UI: if interactive=True (default) or no primary filters provided, show controls
    if interactive or (materials is None and conditions is None and times is None):
        try:
            # Build options from DataFrame
            try:
                materials_opts = sorted(
                    set(
                        str(x)
                        for x in FTIR_DataFrame.get("Material", []).dropna().astype(str)
                    )
                )
            except Exception:
                materials_opts = []
            try:
                conditions_col = (
                    "Conditions"
                    if "Conditions" in FTIR_DataFrame.columns
                    else (
                        "Condition" if "Condition" in FTIR_DataFrame.columns else None
                    )
                )
                if conditions_col is not None:
                    conditions_opts = sorted(
                        set(
                            str(x)
                            for x in FTIR_DataFrame[conditions_col].dropna().astype(str)
                        )
                    )
                else:
                    conditions_opts = []
            except Exception:
                conditions_opts = []
            try:
                times_raw = FTIR_DataFrame.get("Time", [])
                times_set = set()
                seq = list(
                    getattr(times_raw, "dropna", lambda: [])().tolist()
                    if hasattr(times_raw, "dropna")
                    else times_raw
                )
                for t in seq:
                    try:
                        times_set.add(int(t))
                    except Exception:
                        try:
                            times_set.add(str(t))
                        except Exception:
                            pass
                times_opts = sorted(times_set, key=lambda v: (isinstance(v, str), v))
            except Exception:
                times_opts = []

            # Widgets
            materials_dd = widgets.Dropdown(
                options=["any"] + materials_opts,
                value="any",
                description="Material",
                layout=widgets.Layout(width="40%"),
            )

            conditions_dd = widgets.Dropdown(
                options=["any"] + conditions_opts,
                value="any",
                description="Conditions",
                layout=widgets.Layout(width="40%"),
            )

            # For Time, allow 'any' or a single specific time
            # Keep underlying values numeric when possible
            time_options = [("any", "any")] + [
                (str(v), v) for v in times_opts
            ]
            times_dd = widgets.Dropdown(
                options=time_options,
                value="any",
                description="Time",
                layout=widgets.Layout(width="30%"),
            )

            # Apply persisted defaults if available
            try:
                _sess = _get_session_defaults()
                if _sess.get("material") in materials_dd.options:
                    materials_dd.value = _sess.get("material")
                if _sess.get("conditions") in conditions_dd.options:
                    conditions_dd.value = _sess.get("conditions")
                saved_time = _sess.get("time", "any")
                time_values = [v for (_lab, v) in time_options]
                if saved_time in time_values:
                    times_dd.value = saved_time
            except Exception:
                pass

            # Compact multi-select checkboxes for trace types
            raw_cb = widgets.Checkbox(
                value=True if materials is None else bool(raw_data),
                description="Raw",
                layout=widgets.Layout(width="auto"),
            )
            base_cb = widgets.Checkbox(
                value=False if materials is None else bool(baseline),
                description="Baseline",
                layout=widgets.Layout(width="auto"),
            )
            blc_cb = widgets.Checkbox(
                value=False if materials is None else bool(baseline_corrected),
                description="Baseline-corrected",
                layout=widgets.Layout(width="auto"),
            )
            norm_cb = widgets.Checkbox(
                value=False if materials is None else bool(normalized),
                description="Normalized",
                layout=widgets.Layout(width="auto"),
            )
            traces_row = widgets.HBox(
                [
                    widgets.Label(value="Traces:"),
                    raw_cb,
                    base_cb,
                    blc_cb,
                    norm_cb,
                ]
            )
            separate_plots_chk = widgets.Checkbox(
                value=True if materials is None else bool(separate_plots),
                description="Separate plots",
            )
            include_replicates_chk = widgets.Checkbox(
                value=True if materials is None else bool(include_replicates),
                description="Include replicates",
            )
            show_bad_chk = widgets.Checkbox(
                value=False if materials is None else bool(show_bad),
                description="Include bad spectra",
            )


            plot_button = widgets.Button(description="Plot", button_style="primary")
            close_button = widgets.Button(description="Close", button_style="danger")
            out = widgets.Output()

            def _parse_mark(text):
                s = (text or "").strip()
                if not s:
                    return None
                if s.lower() == "all":
                    return "all"
                parts = [p.strip() for p in s.split(",") if p.strip()]
                parsed = []
                for p in parts:
                    try:
                        parsed.append(int(p))
                    except Exception:
                        parsed.append(p)
                return parsed if parsed else None

            def _materials_value():
                v = materials_dd.value
                return str(v) if v not in (None, "", "any") else "any"

            def _conditions_value():
                v = conditions_dd.value
                return str(v) if v not in (None, "", "any") else "any"

            def _times_value():
                v = times_dd.value
                return str(v) if v not in (None, "", "any") else "any"

            def _on_plot(_b):
                with out:
                    clear_output(wait=True)
                    try:
                        # Map trace checkboxes to boolean flags
                        tr_raw = bool(raw_cb.value)
                        tr_base = bool(base_cb.value)
                        tr_blc = bool(blc_cb.value)
                        tr_norm = bool(norm_cb.value)
                        plot_grouped_spectra(
                            FTIR_DataFrame=FTIR_DataFrame,
                            materials=_materials_value(),
                            conditions=_conditions_value(),
                            times=_times_value(),
                            raw_data=tr_raw,
                            baseline=tr_base,
                            baseline_corrected=tr_blc,
                            normalized=tr_norm,
                            separate_plots=separate_plots_chk.value,
                            include_replicates=include_replicates_chk.value,
                            show_bad=show_bad_chk.value,
                            interactive=False,
                        )
                    except Exception as e:
                        print(f"Error while plotting: {e}")

            plot_button.on_click(_on_plot)

            # Persist selections on Plot and on changes
            def _persist(_=None):
                try:
                    _set_session_selection(
                        material=materials_dd.value,
                        conditions=conditions_dd.value,
                        time=times_dd.value,
                    )
                except Exception:
                    pass

            plot_button.on_click(lambda _b: _persist())
            materials_dd.observe(lambda ch: _persist(), names="value")
            conditions_dd.observe(lambda ch: _persist(), names="value")
            times_dd.observe(lambda ch: _persist(), names="value")

            def _on_close(_b):
                """Close the interactive UI and clear outputs, similar to other modules."""
                try:
                    # Clear the visible area first
                    clear_output(wait=True)
                except Exception:
                    pass
                # Attempt to close widgets to free comms and avoid duplicate UIDs
                for w in [
                    # Individual selector widgets
                    materials_dd,
                    conditions_dd,
                    times_dd,
                    # Trace checkboxes and their row container
                    raw_cb,
                    base_cb,
                    blc_cb,
                    norm_cb,
                    traces_row,
                    # Toggle checkboxes and their container rows
                    separate_plots_chk,
                    include_replicates_chk,
                    show_bad_chk,
                    # Top-level layout containers
                    None,  # placeholder if we later add more dynamic containers
                    # Action buttons & output
                    plot_button,
                    close_button,
                    out,
                ] + [
                    # Higher-level composite containers (selectors/toggles/controls)
                    'selectors_placeholder',  # will be swapped below if defined
                    'toggles_placeholder',
                    'controls_placeholder',
                ]:
                    # Replace placeholder strings with actual widget objects if they exist
                    if w == 'selectors_placeholder':
                        w = selectors
                    elif w == 'toggles_placeholder':
                        w = toggles
                    elif w == 'controls_placeholder':
                        w = controls
                    if w is None:
                        continue
                    try:
                        w.close()
                    except Exception:
                        pass

            close_button.on_click(_on_close)

            # Layout and display
            selectors = widgets.HBox(
                [materials_dd, conditions_dd, times_dd]
            )
            toggles = widgets.HBox(
                [traces_row, widgets.VBox([separate_plots_chk, include_replicates_chk, show_bad_chk])]
            )
            controls = widgets.VBox([selectors, toggles, widgets.HBox([plot_button, close_button])])
            display(controls, out)
            return
        except Exception as e:
            # If widgets are unavailable or something fails, fall back to non-interactive path with a note
            try:
                print(f"Interactive controls unavailable, falling back to static plot: {e}")
            except Exception:
                pass

    # Non-interactive path: coalesce None to 'any'
    if materials is None:
        materials = "any"
    if conditions is None:
        conditions = "any"
    if times is None:
        times = "any"

    # Parse comma-separated strings into lists, handle 'any' (case-insensitive)
    mask = pd.Series([True] * len(FTIR_DataFrame))
    # Optionally exclude rows marked as bad quality
    try:
        if not show_bad:
            mask &= _quality_good_mask(FTIR_DataFrame).values
    except Exception:
        pass
    if isinstance(materials, str) and materials.strip().lower() != "any":
        material_list = [m.strip() for m in materials.split(",") if m.strip()]
        mask &= FTIR_DataFrame["Material"].isin(material_list)
    if isinstance(conditions, str) and conditions.strip().lower() != "any":
        condition_list = [c.strip() for c in conditions.split(",") if c.strip()]
        # Base condition mask for selected conditions
        cond_mask = FTIR_DataFrame["Conditions"].isin(condition_list)
        # If Time == 'any', always include 'unexposed' spectra for the selected material(s),
        # regardless of the chosen condition(s) (applies across conditions)
        if isinstance(times, str) and times.strip().lower() == "any":
            try:
                cond_series = FTIR_DataFrame["Conditions"].astype(str).str.lower()
                cond_mask = cond_mask | (cond_series == "unexposed")
            except Exception:
                # Fallback without case normalization
                cond_mask = cond_mask | (FTIR_DataFrame["Conditions"] == "unexposed")
        mask &= cond_mask
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
        mask &= FTIR_DataFrame["Time"].isin(time_list)
    filtered_data = FTIR_DataFrame[mask]

    # If nothing matches, explain why and bail early instead of showing a blank plot
    if filtered_data.empty:
        mats = sorted(
            set(
                map(
                    str, FTIR_DataFrame.get("Material", pd.Series([])).dropna().unique()
                )
            )
        )
        conds_col = (
            "Conditions"
            if "Conditions" in FTIR_DataFrame.columns
            else ("Condition" if "Condition" in FTIR_DataFrame.columns else None)
        )
        conds = (
            sorted(
                set(
                    map(
                        str,
                        FTIR_DataFrame.get(conds_col, pd.Series([])).dropna().unique(),
                    )
                )
            )
            if conds_col
            else []
        )
        times_avail = sorted(
            set(FTIR_DataFrame.get("Time", pd.Series([])).dropna().unique())
        )
        print(
            "No spectra matched the current filters.\n"
            f" - materials={materials!r}, conditions={conditions!r}, times={times!r}\n"
            "Try relaxing one or more filters (e.g., set to 'any').\n"
            f"Available Materials: {mats}\n"
            f"Available Conditions: {conds}\n"
            f"Available Times: {times_avail}"
        )
        return

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

    # Plot all together (legend in time order) with Plotly
    fig_group = go.FigureWidget()
    for idx, spectrum_row in filtered_data_sorted.iterrows():
        material_val = spectrum_row.get("Material", "")
        condition_val = spectrum_row.get(
            "Conditions", spectrum_row.get("Condition", "")
        )
        time_val = spectrum_row.get("Time", "")
        spectrum_label = f"{material_val}, {condition_val}, {time_val}"
        # Parse x-axis
        x_axis = spectrum_row.get(x_axis_col)
        if isinstance(x_axis, str):
            try:
                x_axis = ast.literal_eval(x_axis)
            except Exception:
                x_axis = None
        if x_axis is None:
            print(f"Skipping index {idx}: missing X-axis ('{x_axis_col}').")
            continue

        # Plot selected series
        def _add_series(y, name_suffix):
            if isinstance(y, str):
                try:
                    y_v = ast.literal_eval(y)
                except Exception:
                    y_v = None
            else:
                y_v = y
            if y_v is not None:
                fig_group.add_scatter(
                    x=list(x_axis),
                    y=list(y_v),
                    mode="lines",
                    name=f"{name_suffix}: {spectrum_label}",
                )

        if raw_data and ("Raw Data" in spectrum_row):
            _add_series(spectrum_row.get("Raw Data"), "Raw")
        if baseline and (spectrum_row.get("Baseline") is not None):
            _add_series(spectrum_row.get("Baseline"), "Baseline")
        if baseline_corrected and (
            spectrum_row.get("Baseline-Corrected Data") is not None
        ):
            _add_series(
                spectrum_row.get("Baseline-Corrected Data"), "Baseline-Corrected"
            )
        if normalized and (
            spectrum_row.get("Normalized and Corrected Data") is not None
        ):
            _add_series(
                spectrum_row.get("Normalized and Corrected Data"),
                "Normalized and Corrected",
            )
    fig_group.update_layout(
        title=f"Spectra for Material: {materials} | Condition: {conditions} | Time: {times}",
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="Absorbance (AU)",
        legend=dict(orientation="h", y=-0.2),
    )
    display(fig_group)

    # Optional: mark selected rows as good/bad after plotting
    try:
        qcol = _quality_column_name(FTIR_DataFrame)
        # Mark good first
        if mark_good is not None and not filtered_data.empty:
            if isinstance(mark_good, str) and mark_good.strip().lower() == "all":
                FTIR_DataFrame.loc[filtered_data.index, qcol] = "good"
            elif isinstance(mark_good, (list, tuple, set)):
                to_mark_idx = []
                for item in mark_good:
                    try:
                        if isinstance(item, (int, np.integer)):
                            to_mark_idx.append(int(item))
                        else:
                            matches = filtered_data.index[
                                filtered_data.get("File Name", "").astype(str)
                                == str(item)
                            ].tolist()
                            to_mark_idx.extend(matches)
                    except Exception:
                        pass
                if to_mark_idx:
                    FTIR_DataFrame.loc[list(set(to_mark_idx)), qcol] = "good"
        # Then mark bad
        if mark_bad is not None and not filtered_data.empty:
            if isinstance(mark_bad, str) and mark_bad.strip().lower() == "all":
                FTIR_DataFrame.loc[filtered_data.index, qcol] = "bad"
            elif isinstance(mark_bad, (list, tuple, set)):
                to_mark_idx = []
                for item in mark_bad:
                    try:
                        if isinstance(item, (int, np.integer)):
                            to_mark_idx.append(int(item))
                        else:
                            matches = filtered_data.index[
                                filtered_data.get("File Name", "").astype(str)
                                == str(item)
                            ].tolist()
                            to_mark_idx.extend(matches)
                    except Exception:
                        pass
                if to_mark_idx:
                    FTIR_DataFrame.loc[list(set(to_mark_idx)), qcol] = "bad"
    except Exception:
        pass

    # Plot each file individually if requested, in sequential order by time
    if separate_plots:
        for idx, row in filtered_data_sorted.iterrows():
            # Create individual Plotly figure
            fig_i = go.FigureWidget()
            x_axis = row.get(x_axis_col)
            if isinstance(x_axis, str):
                try:
                    x_axis = ast.literal_eval(x_axis)
                except Exception:
                    x_axis = None
            if x_axis is None:
                continue

            def _add_series_i(y, name_suffix):
                if isinstance(y, str):
                    try:
                        y_v = ast.literal_eval(y)
                    except Exception:
                        y_v = None
                else:
                    y_v = y
                if y_v is not None:
                    fig_i.add_scatter(
                        x=list(x_axis), y=list(y_v), mode="lines", name=name_suffix
                    )

            if raw_data:
                _add_series_i(row.get("Raw Data"), "Raw")
            if baseline and (row.get("Baseline") is not None):
                _add_series_i(row.get("Baseline"), "Baseline")
            if baseline_corrected and (row.get("Baseline-Corrected Data") is not None):
                _add_series_i(row.get("Baseline-Corrected Data"), "Baseline-Corrected")
            if normalized and (row.get("Normalized and Corrected Data") is not None):
                _add_series_i(
                    row.get("Normalized and Corrected Data"), "Normalized and Corrected"
                )
            material_val = row.get("Material", "")
            condition_val = row.get("Conditions", row.get("Condition", ""))
            time_val = row.get("Time", "")
            fig_i.update_layout(
                title=f"Spectrum: {material_val}, {condition_val}, {time_val}",
                xaxis_title="Wavenumber (cm⁻¹)",
                yaxis_title="Absorbance (AU)",
                legend=dict(orientation="h", y=-0.2),
            )

            # Mark buttons next to each individual plot
            mark_bad_btn = widgets.Button(
                description="Mark as bad", button_style="danger"
            )
            mark_good_btn = widgets.Button(
                description="Mark as good", button_style="success"
            )

            # IMPORTANT: capture per-iteration button instances in default args to avoid late binding
            def _on_mark_bad_local(
                _b=None, i=idx, bad_btn=mark_bad_btn, good_btn=mark_good_btn
            ):
                try:
                    qcol = _quality_column_name(FTIR_DataFrame)
                    FTIR_DataFrame.at[i, qcol] = "bad"
                except Exception:
                    pass
                # Toggle button visibility for this row's buttons only
                try:
                    bad_btn.layout.display = "none"
                    good_btn.layout.display = ""
                except Exception:
                    pass

            def _on_mark_good_local(
                _b=None, i=idx, bad_btn=mark_bad_btn, good_btn=mark_good_btn
            ):
                try:
                    qcol = _quality_column_name(FTIR_DataFrame)
                    FTIR_DataFrame.at[i, qcol] = "good"
                except Exception:
                    pass
                # Toggle button visibility for this row's buttons only
                try:
                    bad_btn.layout.display = ""
                    good_btn.layout.display = "none"
                except Exception:
                    pass

            mark_bad_btn.on_click(_on_mark_bad_local)
            mark_good_btn.on_click(_on_mark_good_local)
            # Initialize button visibility for this row
            try:
                qcol = _quality_column_name(FTIR_DataFrame)
                status = None
                try:
                    status = FTIR_DataFrame.at[idx, qcol]
                except Exception:
                    status = None
                is_bad = str(status).strip().lower() == "bad"
                mark_bad_btn.layout.display = "none" if is_bad else ""
                mark_good_btn.layout.display = "" if is_bad else "none"
            except Exception:
                pass
            display(widgets.HBox([fig_i, widgets.VBox([mark_bad_btn, mark_good_btn])]))


def try_baseline(
    FTIR_DataFrame,
    material=None,
    baseline_function=None,
    filepath=None,
):
    """
    Apply a modifiable baseline to a single spectrum from the DataFrame.

    Allows for on-the-fly parameter adjustments via interactive widgets and
    experimentation with different baseline functions.

    Parameters
    ----------
    FTIR_DataFrame (pd.DataFrame): The in-memory DataFrame containing all spectra.
    material (str, optional): Material name to analyze (ignored if filepath is
        provided).
    baseline_function (str): Baseline function to use ('ARPLS', 'IRSQR', 'FABC').
    filepath (str, optional): If provided, only process this file (by 'File Location'
        + 'File Name').

    Returns
    -------
    FTIR_DataFrame : pd.DataFrame
        The updated DataFrame with baseline corrections applied, if user chooses to save
        choices. Otherwise, the DataFrame remains unchanged.
    """
    # Proactively close any leftover widgets from a prior session to ensure fresh UI renders
    try:
        for _w in list(_TB_WIDGETS):
            try:
                _w.close()
            except Exception:
                pass
        _TB_WIDGETS.clear()
    except Exception:
        pass
    try:
        clear_output(wait=True)
    except Exception:
        pass

    if baseline_function is None:
        # Default to ARPLS when not specified; user can change via dropdown below
        baseline_function = "ARPLS"
    # Do NOT auto-launch manual baseline; user must still select a spectrum first (minimal mode preserved)
    # Initialize selection placeholders; user will pick a spectrum via dropdowns
    row = None
    x = np.array([])
    y = np.array([])
    # If neither material nor filepath provided, try session defaults for material
    if material is None and filepath is None:
        try:
            _sess = _get_session_defaults()
            sess_mat = _sess.get("material")
            if isinstance(sess_mat, str) and sess_mat.strip().lower() != "any":
                material = sess_mat
        except Exception:
            pass

    if filepath is not None:
        if os.path.sep in filepath:
            folder, fname = os.path.split(filepath)
            filtered = FTIR_DataFrame[
                (FTIR_DataFrame["File Location"] == folder)
                & (FTIR_DataFrame["File Name"] == fname)
            ]
        else:
            filtered = FTIR_DataFrame[FTIR_DataFrame["File Name"] == filepath]
        # Exclude rows marked as bad quality
        try:
            filtered = filtered[_quality_good_mask(filtered)]
        except Exception:
            pass
        if filtered.empty:
            raise ValueError(f"No entry found for file '{filepath}'.")
        row = filtered.iloc[0]
        material = row.get("Material", "Unknown")
        # Persist this selection to session state
        try:
            _set_session_selection(
                material=row.get("Material"),
                conditions=row.get("Conditions"),
                time=row.get("Time"),
            )
        except Exception:
            pass
    # If a specific file is selected, compute x/y; otherwise wait for user selection
    if row is not None:
        x = (
            ast.literal_eval(row["X-Axis"]) if isinstance(row["X-Axis"], str) else row["X-Axis"]
        )
        y = (
            ast.literal_eval(row["Raw Data"]) if isinstance(row["Raw Data"], str) else row["Raw Data"]
        )
        y = np.array(y, dtype=float)

    parameters = _get_default_parameters(baseline_function)
    parameters = _cast_parameter_types(baseline_function, parameters)

    # Print selected file path only after a specific file is chosen
    if row is not None:
        file_path = os.path.join(row.get("File Location", ""), row.get("File Name", ""))
        try:
            print(f"Plotting: {file_path}")
        except Exception:
            pass

    # Widget setup for live parameter editing
    param_widgets = {}
    # Explicitly define widgets for each baseline function and parameter
    if baseline_function.upper() == "ARPLS":
        # lam: float, iterations: int (diff_order fixed internally; not user-editable)
        param_widgets["lam"] = widgets.FloatSlider(
            value=parameters.get("lam", 1e5),
            min=1e4,
            max=1e6,
            step=1e4,
            description="Smoothness (lam)",
            readout_format=".1e",
            style={"description_width": "auto"},
        )
        param_widgets["max_iter"] = widgets.IntSlider(
            value=parameters.get("max_iter", 50),
            min=1,
            max=200,
            step=1,
            description="Max Iterations",
            style={"description_width": "auto"},
        )
        param_widgets["tol"] = widgets.FloatSlider(
            value=parameters.get("tol", 1e-3),
            min=1e-6,
            max=1e-1,
            step=1e-4,
            description="Tolerance",
            readout_format=".1e",
            style={"description_width": "auto"},
        )
    elif baseline_function.upper() == "IRSQR":
        # lam: float, quantile: float, num_knots: int, spline_degree: int, diff_order:
        # int, max_iterations: int, tolerance: float, eps: float
        param_widgets["lam"] = widgets.FloatSlider(
            value=parameters.get("lam", 1e6),
            min=1e5,
            max=1e7,
            step=1e5,
            description="Smoothness (lam)",
            readout_format=".1e",
            style={"description_width": "auto"},
        )
        param_widgets["quantile"] = widgets.FloatSlider(
            value=parameters.get("quantile", 0.05),
            min=0.001,
            max=0.5,
            step=0.001,
            description="Quantile",
            readout_format=".3f",
            style={"description_width": "auto"},
        )
        param_widgets["num_knots"] = widgets.IntSlider(
            value=parameters.get("num_knots", 100),
            min=5,
            max=500,
            step=5,
            description="Knots",
            style={"description_width": "auto"},
        )
        param_widgets["spline_degree"] = widgets.IntSlider(
            value=parameters.get("spline_degree", 3),
            min=1,
            max=5,
            step=1,
            description="Spline Degree",
            style={"description_width": "auto"},
        )
        param_widgets["diff_order"] = widgets.IntSlider(
            value=parameters.get("diff_order", 3),
            min=1,
            max=3,
            step=1,
            description="Differential Order",
            style={"description_width": "auto"},
        )
        param_widgets["max_iter"] = widgets.IntSlider(
            value=parameters.get("max_iter", 100),
            min=1,
            max=1000,
            step=1,
            description="Max Iterations",
            style={"description_width": "auto"},
        )
        param_widgets["tol"] = widgets.FloatSlider(
            value=parameters.get("tol", 1e-6),
            min=1e-10,
            max=1e-2,
            step=1e-7,
            description="Tolerance",
            readout_format=".1e",
            style={"description_width": "auto"},
        )
    elif baseline_function.upper() == "FABC":
        # lam: float, scale: int or None, num_std: float, diff_order: int, min_length:
        # int
        param_widgets["lam"] = widgets.FloatSlider(
            value=parameters.get("lam", 1e6),
            min=1e4,
            max=1e7,
            step=1e5,
            description="Smoothness (lam)",
            readout_format=".1e",
            style={"description_width": "auto"},
        )
        # If no spectrum is selected yet, use a generic default for scale; recomputed on selection
        if row is not None:
            _raw_data = (
                ast.literal_eval(row["Raw Data"]) if isinstance(row["Raw Data"], str) else row["Raw Data"]
            )
            scale_default = ceil(optimize_window(_raw_data) / 2)
        else:
            scale_default = 50
        scale_val = parameters.get("scale", None)
        if scale_val is None:
            scale_val = scale_default
        param_widgets["scale"] = widgets.IntSlider(
            value=int(scale_val),
            min=2,
            max=500,
            step=1,
            description="Scale",
            style={"description_width": "auto"},
        )
        param_widgets["num_std"] = widgets.FloatSlider(
            value=parameters.get("num_std", 3.0),
            min=1.5,
            max=4.5,
            step=0.1,
            description="Standard Deviations",
            readout_format=".2f",
            style={"description_width": "auto"},
        )
        param_widgets["diff_order"] = widgets.IntSlider(
            value=parameters.get("diff_order", 2),
            min=1,
            max=3,
            step=1,
            description="Differential Order",
            style={"description_width": "auto"},
        )
        param_widgets["min_length"] = widgets.IntSlider(
            value=parameters.get("min_length", 2),
            min=1,
            max=6,
            step=1,
            description="Min Baseline Span Length",
            style={"description_width": "auto"},
        )

    # -------------------
    # Filtering and spectrum selection controls (Material/Conditions/Spectrum)
    # -------------------
    include_bad_cb = widgets.Checkbox(value=False, description="Include bad spectra")
    # Build base DataFrame for options: restrict to filepath if provided
    if filepath is not None:
        base_df = filtered.copy()
    else:
        base_df = FTIR_DataFrame.copy()
        try:
            base_df = base_df[_quality_good_mask(base_df)]
        except Exception:
            pass
    # Unique materials
    try:
        unique_materials = (
            sorted({str(v) for v in base_df.get("Material", pd.Series([], dtype=object)).dropna().astype(str).unique().tolist()})
            if "Material" in base_df.columns
            else []
        )
    except Exception:
        unique_materials = []
    material_dd = widgets.Dropdown(
        options=["any"] + unique_materials,
        value="any",
        description="Material",
        layout=widgets.Layout(width="40%"),
    )
    # Conditions list, exclude 'unexposed'
    try:
        cond_series = (
            base_df["Conditions"]
            if "Conditions" in base_df.columns
            else (base_df["Condition"] if "Condition" in base_df.columns else pd.Series([], dtype=object))
        )
        _all_conditions = [str(v) for v in cond_series.dropna().astype(str).unique().tolist()]
        unique_conditions = sorted([c for c in _all_conditions if c.strip().lower() != "unexposed"])
    except Exception:
        unique_conditions = []
    conditions_dd = widgets.Dropdown(
        options=["any"] + unique_conditions,
        value="any",
        description="Conditions",
        layout=widgets.Layout(width="40%"),
    )
    # Apply session defaults to filters
    try:
        _sess = _get_session_defaults()
        sess_mat = _sess.get("material")
        if isinstance(sess_mat, str) and sess_mat in unique_materials:
            material_dd.value = sess_mat
        sess_cond = _sess.get("conditions")
        if isinstance(sess_cond, str) and sess_cond in unique_conditions:
            conditions_dd.value = sess_cond
    except Exception:
        pass
    # Baseline function dropdown (user can switch between methods)
    baseline_dd = widgets.Dropdown(
        options=["ARPLS", "IRSQR", "FABC", "MANUAL"],
        value=str(baseline_function).upper() if str(baseline_function).upper() in ["ARPLS", "IRSQR", "FABC", "MANUAL"] else "ARPLS",
        description="Baseline",
        layout=widgets.Layout(width="30%"),
    )
    # Spectrum dropdown (built via helper)
    spectrum_sel = widgets.Dropdown(options=[("Select a spectrum…", None)], value=None, description="Spectrum", layout=widgets.Layout(width="70%"))

    def _rebuild_conditions_options():
        try:
            if material_dd.value == "any":
                dfm = base_df
            else:
                dfm = base_df[base_df.get("Material", "").astype(str) == str(material_dd.value)]
            cs = (
                dfm["Conditions"]
                if "Conditions" in dfm.columns
                else (dfm["Condition"] if "Condition" in dfm.columns else pd.Series([], dtype=object))
            )
            cvals = [str(v) for v in cs.dropna().astype(str).unique().tolist()]
            cvals = sorted([c for c in cvals if c.strip().lower() != "unexposed"])
            curr = conditions_dd.value if conditions_dd.value in (["any"] + cvals) else "any"
            conditions_dd.options = ["any"] + cvals
            conditions_dd.value = curr
        except Exception:
            pass

    def _build_spectrum_options():
        # nonlocal row, x, y to update current selection
        nonlocal row, x, y, material
        try:
            df = base_df.copy()
            if not include_bad_cb.value:
                try:
                    df = df[_quality_good_mask(df)]
                except Exception:
                    pass
            if material_dd.value != "any":
                df = df[df.get("Material", "").astype(str) == str(material_dd.value)]
            # Filter by conditions if chosen
            if conditions_dd.value != "any":
                cond_col = "Conditions" if "Conditions" in df.columns else ("Condition" if "Condition" in df.columns else None)
                if cond_col is not None:
                    # Always include 'unexposed' spectra in addition to the selected condition
                    sel_val = str(conditions_dd.value)
                    cond_series = df.get(cond_col, pd.Series([], dtype=object)).astype(str)
                    unexp_mask = cond_series.str.strip().str.lower() == "unexposed"
                    cond_mask = cond_series == sel_val
                    df = df[cond_mask | unexp_mask]
            # Sort by time if present
            if "Time" in df.columns:
                df["_sort_time"] = pd.to_numeric(df["Time"], errors="coerce").fillna(float("inf"))
                df = df.sort_values(by=["_sort_time"], kind="mergesort")
            options = []
            for idx2, r2 in df.iterrows():
                label = (
                    f"{r2.get('Material','')} | {r2.get('Conditions', r2.get('Condition',''))}"
                    f" | T={r2.get('Time','')} | {r2.get('File Name','')}"
                )
                options.append((label, idx2))
            if not options:
                spectrum_sel.options = [("<no spectra>", None)]
                spectrum_sel.value = None
                return
            # Prefer previously selected idx if still present; else prefer session time
            prev = spectrum_sel.value
            values = [v for (_l, v) in options]
            chosen = None
            if prev in values:
                chosen = prev
            else:
                # try session time
                try:
                    _sess2 = _get_session_defaults()
                    stime = _sess2.get("time")
                except Exception:
                    stime = None
                if stime is not None:
                    for (_l, v) in options:
                        try:
                            if float(FTIR_DataFrame.loc[v].get("Time")) == float(stime):
                                chosen = v
                                break
                        except Exception:
                            continue
            # Present a placeholder + options; do not auto-select if no match
            spectrum_sel.options = [("Select a spectrum…", None)] + options
            spectrum_sel.value = chosen if chosen in values else None
            if chosen is not None:
                # Update current selection variables only when a specific spectrum is chosen
                rsel = FTIR_DataFrame.loc[chosen]
                row = rsel
                material = rsel.get("Material", material)
                x = (ast.literal_eval(rsel["X-Axis"]) if isinstance(rsel["X-Axis"], str) else rsel["X-Axis"]) 
                y = (ast.literal_eval(rsel["Raw Data"]) if isinstance(rsel["Raw Data"], str) else rsel["Raw Data"]) 
                y = np.array(y, dtype=float)
                # Persist session
                try:
                    _set_session_selection(material=row.get("Material"), conditions=row.get("Conditions"), time=row.get("Time"))
                except Exception:
                    pass
        except Exception:
            pass

    # Seed initial lists based on current state
    _rebuild_conditions_options()
    _build_spectrum_options()

    # Force minimal mode on entry: avoid auto-selected spectrum from session state
    # so the UI always renders and the user explicitly confirms the selection.
    try:
        if spectrum_sel.value is not None:
            spectrum_sel.value = None
            row = None
    except Exception:
        pass

    output = widgets.Output()
    # Persist a single Plotly FigureWidget and update its traces for low flicker
    fig_widget = None

    def _plot_baseline(**widget_params):
        nonlocal fig_widget
        # Merge and cast widget parameters
        param_vals = parameters.copy()
        param_vals.update(widget_params)
        param_vals = _cast_parameter_types(baseline_function, param_vals)
        with output:
            # If no spectrum has been selected yet, prompt once
            if row is None or spectrum_sel.value is None:
                try:
                    clear_output(wait=True)
                except Exception:
                    pass
                print("Select a spectrum to preview the baseline.")
                return

            # Compute baseline safely
            try:
                if baseline_function.upper() == "ARPLS":
                    baseline_result = arpls(y, **param_vals)
                elif baseline_function.upper() == "IRSQR":
                    baseline_result = irsqr(y, **param_vals, x_data=x)
                elif baseline_function.upper() == "FABC":
                    baseline_result = fabc(y, **param_vals)
                else:
                    try:
                        clear_output(wait=True)
                    except Exception:
                        pass
                    print(f"Unknown baseline function: {baseline_function}")
                    return
            except Exception as e:
                try:
                    clear_output(wait=True)
                except Exception:
                    pass
                print(f"Baseline computation error: {e}")
                return

            # Normalize baseline output
            if isinstance(baseline_result, tuple):
                baseline = baseline_result[0]
            elif isinstance(baseline_result, dict):
                baseline = baseline_result.get("baseline")
                if baseline is None:
                    try:
                        clear_output(wait=True)
                    except Exception:
                        pass
                    print("Error: Baseline function did not return a baseline array.")
                    return
            else:
                baseline = baseline_result

            try:
                # Prepare arrays (parameter summary logic removed per simplification request)
                x_arr = np.asarray(x)
                y_arr = np.asarray(y)
                baseline_arr = np.asarray(baseline)
                residual = y_arr - baseline_arr

                # Build or update Plotly FigureWidget
                title_top = "Raw Data and Baseline"
                if fig_widget is None or len(getattr(fig_widget, "data", [])) < 3:
                    try:
                        clear_output(wait=True)
                    except Exception:
                        pass
                    # Create subplots: top (raw + baseline), bottom (baseline-corrected)
                    base_fig = make_subplots(
                        rows=2,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.12,
                        subplot_titles=(title_top, "Baseline-Corrected"),
                    )
                    fig_widget = go.FigureWidget(base_fig)
                    # Raw spectrum
                    fig_widget.add_scatter(x=x_arr, y=y_arr, mode="lines", name="Spectrum", line=dict(color="black"), row=1, col=1)
                    # Baseline
                    fig_widget.add_scatter(x=x_arr, y=baseline_arr, mode="lines", name="Baseline", line=dict(color="red", width=1.5, dash="dash"), row=1, col=1)
                    # Baseline-corrected (spectrum - baseline)
                    fig_widget.add_scatter(x=x_arr, y=residual, mode="lines", name="Baseline-Corrected", line=dict(color="blue"), row=2, col=1)
                    # Axes labels and layout
                    fig_widget.update_yaxes(title_text="Absorbance (AU)", row=1, col=1)
                    fig_widget.update_yaxes(title_text="", row=2, col=1)
                    fig_widget.update_xaxes(title_text="Wavenumber (cm⁻¹)", row=2, col=1)
                    fig_widget.update_layout(legend=dict(orientation="h", y=-0.2), height=800)
                    display(fig_widget)
                else:
                    # Update data traces in-place (no redraw flicker)
                    try:
                        fig_widget.data[0].x = x_arr
                        fig_widget.data[0].y = y_arr
                        fig_widget.data[1].x = x_arr
                        fig_widget.data[1].y = baseline_arr
                        fig_widget.data[2].x = x_arr
                        fig_widget.data[2].y = residual
                    except Exception:
                        # Fall back to rebuild if trace shapes changed unexpectedly
                        fig_widget = None
                        _plot_baseline(**widget_params)
                        return
                    # Update subplot titles
                    try:
                        if hasattr(fig_widget.layout, "annotations") and len(fig_widget.layout.annotations) >= 2:
                            fig_widget.layout.annotations[0].text = title_top
                            fig_widget.layout.annotations[1].text = "Baseline-Corrected"
                    except Exception:
                        pass
            except Exception as e:
                try:
                    clear_output(wait=True)
                except Exception:
                    pass
                print(f"Plot error: {e}")

    # Minimal UI when no spectrum is selected: show only filters, spectrum dropdown, and Close button
    try:
        _no_selection = (row is None) or (spectrum_sel.value is None)
    except Exception:
        _no_selection = True
    if _no_selection:
        # Simple message prompting selection
        with output:
            try:
                clear_output(wait=True)
                print("Select a spectrum to preview the baseline.")
            except Exception:
                pass

        close_btn = widgets.Button(
            description="Close",
            button_style="danger",
            layout=widgets.Layout(margin="10px 0 0 0"),
        )

        # Filters and spectrum rows (no sliders or other buttons)
        filters_row = widgets.HBox([material_dd, conditions_dd, include_bad_cb, baseline_dd])
        spectrum_row = widgets.HBox([spectrum_sel])
        ui = widgets.VBox([filters_row, spectrum_row, close_btn])

        container = widgets.VBox([ui, output])
        display(container)
        try:
            _TB_WIDGETS.extend([container])
        except Exception:
            pass

        # Wire minimal interactions: rebuild options on filter change
        def _on_mat_min(change):
            if change.get("name") == "value":
                _rebuild_conditions_options()
                _build_spectrum_options()
        def _on_cond_min(change):
            if change.get("name") == "value":
                _build_spectrum_options()
        def _on_inc_min(change):
            if change.get("name") == "value":
                _build_spectrum_options()
        def _on_base_min(change):
            if change.get("name") == "value":
                nonlocal baseline_function
                try:
                    baseline_function = str(change.get("new")).upper()
                except Exception:
                    baseline_function = "ARPLS"
                # For MANUAL, defer anchor point UI until a spectrum is chosen.
        def _on_spec_min(change):
            if change.get("name") == "value" and change.get("new") is not None:
                # A spectrum has been chosen; build full UI in-place without recursive re-entry
                try:
                    sel_idx = change.get("new")
                    nonlocal row, x, y, material
                    row = FTIR_DataFrame.loc[sel_idx]
                    material = row.get("Material", material)
                    x = (ast.literal_eval(row["X-Axis"]) if isinstance(row["X-Axis"], str) else row["X-Axis"])
                    y = (ast.literal_eval(row["Raw Data"]) if isinstance(row["Raw Data"], str) else row["Raw Data"])
                    y = np.array(y, dtype=float)
                    try:
                        _set_session_selection(material=row.get("Material"), conditions=row.get("Conditions"), time=row.get("Time"))
                    except Exception:
                        pass
                except Exception:
                    return
                # Close minimal container
                try:
                    container.close()
                except Exception:
                    pass
                # Build integrated MANUAL mode or parameter UI depending on selection
                def _build_manual_ui():
                    # Close any prior full UI/figure
                    try:
                        plt.close("all")
                    except Exception:
                        pass
                    # Manual state
                    manual_out = widgets.Output()
                    anchor_points = []
                    # Flag controlling whether baseline preview is active (after Continue clicked)
                    baseline_active = False
                    # Buttons
                    continue_btn = widgets.Button(description="Continue", button_style="success")
                    redo_btn = widgets.Button(description="Redo All", button_style="warning")
                    undo_btn = widgets.Button(description="Undo", button_style="")
                    save_file_btn_m = widgets.Button(description="Save for file", button_style="success")
                    save_mat_btn_m = widgets.Button(description="Save for material", button_style="info")
                    close_btn_m = widgets.Button(description="Close", button_style="danger")
                    mark_bad_btn_m = widgets.Button(description="Mark as bad", button_style="danger")
                    mark_good_btn_m = widgets.Button(description="Mark as good", button_style="success")
                    # Figures:
                    #  - fig_m: raw + anchor markers + baseline (preview)
                    #  - fig_corr: baseline-corrected in a separate plot
                    fig_m = go.FigureWidget()
                    fig_m.add_scatter(x=np.asarray(x, dtype=float), y=np.asarray(y, dtype=float), mode="lines", name="Spectrum", line=dict(color="black"))
                    fig_m.add_scatter(x=[], y=[], mode="markers", name="Anchor Points", marker=dict(color="red", size=10))
                    fig_m.update_layout(title="Manual Baseline: click to add anchor points", xaxis_title="Wavenumber (cm⁻¹)", yaxis_title="Absorbance (AU)", height=450)

                    fig_corr = go.FigureWidget()
                    fig_corr.add_scatter(x=[], y=[], mode="lines", name="Baseline-Corrected", line=dict(color="blue"))
                    fig_corr.update_layout(title="Baseline-Corrected", xaxis_title="Wavenumber (cm⁻¹)", yaxis_title="Absorbance (AU)", height=350)

                    # Mark buttons visibility sync
                    def _refresh_mark_btns_m():
                        # Default assume 'good' (show Mark as bad, hide Mark as good)
                        is_bad = False
                        try:
                            qcol = _quality_column_name(FTIR_DataFrame)
                            st = FTIR_DataFrame.at[row.name, qcol]
                            is_bad = str(st).strip().lower() == "bad"
                        except Exception:
                            pass
                        try:
                            mark_bad_btn_m.layout.display = "none" if is_bad else ""
                            mark_good_btn_m.layout.display = "" if is_bad else "none"
                        except Exception:
                            pass

                    def _mark_bad_m(_b=None):
                        try:
                            qcol = _quality_column_name(FTIR_DataFrame)
                            FTIR_DataFrame.at[row.name, qcol] = "bad"
                        except Exception:
                            pass
                        _refresh_mark_btns_m()

                    def _mark_good_m(_b=None):
                        try:
                            qcol = _quality_column_name(FTIR_DataFrame)
                            FTIR_DataFrame.at[row.name, qcol] = "good"
                        except Exception:
                            pass
                        _refresh_mark_btns_m()

                    mark_bad_btn_m.on_click(_mark_bad_m)
                    mark_good_btn_m.on_click(_mark_good_m)

                    # Click handler to add anchor point at nearest x
                    def _on_click(trace, points, selector):
                        try:
                            if not points.xs:
                                return
                            x_click = float(points.xs[0])
                            xs = np.asarray(x, dtype=float)
                            ys = np.asarray(y, dtype=float)
                            idx_near = int(np.nanargmin(np.abs(xs - x_click)))
                            apx = float(xs[idx_near])
                            # Avoid duplicates
                            if apx not in anchor_points:
                                anchor_points.append(apx)
                                ap_sorted = sorted(anchor_points)
                                fig_m.data[1].x = ap_sorted
                                fig_m.data[1].y = [float(ys[int(np.nanargmin(np.abs(xs - ax)))]) for ax in ap_sorted]
                                # If baseline already active, recompute immediately for live update
                                if baseline_active and len(anchor_points) >= 2:
                                    _preview_baseline()
                        except Exception:
                            pass

                    # Attach click to raw trace
                    try:
                        fig_m.data[0].on_click(_on_click)
                    except Exception:
                        pass

                    # Helper: compute preview baseline and show
                    def _preview_baseline():
                        with manual_out:
                            try:
                                clear_output(wait=True)
                            except Exception:
                                pass
                        # Capture current axis ranges to preserve user zoom
                        x_range_main = y_range_main = x_range_corr = y_range_corr = None
                        try:
                            if fig_m.layout.xaxis.autorange is not True and fig_m.layout.xaxis.range:
                                x_range_main = list(fig_m.layout.xaxis.range)
                            if fig_m.layout.yaxis.autorange is not True and fig_m.layout.yaxis.range:
                                y_range_main = list(fig_m.layout.yaxis.range)
                        except Exception:
                            pass
                        try:
                            if fig_corr.layout.xaxis.autorange is not True and fig_corr.layout.xaxis.range:
                                x_range_corr = list(fig_corr.layout.xaxis.range)
                            if fig_corr.layout.yaxis.autorange is not True and fig_corr.layout.yaxis.range:
                                y_range_corr = list(fig_corr.layout.yaxis.range)
                        except Exception:
                            pass
                        xs = np.asarray(x, dtype=float)
                        ys = np.asarray(y, dtype=float)
                        if len(anchor_points) < 2:
                            with manual_out:
                                print("Select at least two anchor points to preview.")
                            return
                        ap_sorted = np.array(sorted(anchor_points), dtype=float)
                        y_anchor = np.array([ys[int(np.nanargmin(np.abs(xs - ap)))] for ap in ap_sorted], dtype=float)
                        try:
                            spline = CubicSpline(ap_sorted, y_anchor, bc_type=((1, 0.0), (1, 0.0)))
                        except Exception:
                            spline = CubicSpline(ap_sorted, y_anchor)
                        baseline_vals = spline(xs)
                        corrected = ys - baseline_vals
                        # Update raw/baseline figure (keep 0: raw, 1: anchors)
                        while len(fig_m.data) > 2:
                            fig_m.data = tuple(fig_m.data[:2])
                        fig_m.add_scatter(x=xs, y=baseline_vals, mode="lines", name="Baseline", line=dict(color="red", width=1.5, dash="dash"))
                        # Update corrected figure (single trace)
                        try:
                            fig_corr.data[0].x = xs
                            fig_corr.data[0].y = corrected
                        except Exception:
                            pass
                        # Reapply previous axis ranges to avoid automatic rescaling
                        try:
                            if x_range_main:
                                fig_m.update_xaxes(range=x_range_main, autorange=False)
                            if y_range_main:
                                fig_m.update_yaxes(range=y_range_main, autorange=False)
                        except Exception:
                            pass
                        try:
                            if x_range_corr:
                                fig_corr.update_xaxes(range=x_range_corr, autorange=False)
                            if y_range_corr:
                                fig_corr.update_yaxes(range=y_range_corr, autorange=False)
                        except Exception:
                            pass

                    def _continue(_b=None):
                        nonlocal baseline_active
                        baseline_active = True
                        # Hide Continue after first activation
                        try:
                            continue_btn.layout.display = "none"
                        except Exception:
                            pass
                        _preview_baseline()

                    def _redo(_b=None):
                        # Close current manual UI and rebuild fresh to ensure clean handlers and controls
                        try:
                            fig_m.close()
                        except Exception:
                            pass
                        try:
                            fig_corr.close()
                        except Exception:
                            pass
                        try:
                            manual_container.close()
                        except Exception:
                            pass
                        _build_manual_ui()

                    def _undo(_b=None):
                        # Remove the most recently added anchor point and update plots
                        try:
                            if not anchor_points:
                                return
                            # Capture axis ranges to preserve user zoom
                            x_range_main = y_range_main = x_range_corr = y_range_corr = None
                            try:
                                if fig_m.layout.xaxis.autorange is not True and fig_m.layout.xaxis.range:
                                    x_range_main = list(fig_m.layout.xaxis.range)
                                if fig_m.layout.yaxis.autorange is not True and fig_m.layout.yaxis.range:
                                    y_range_main = list(fig_m.layout.yaxis.range)
                            except Exception:
                                pass
                            try:
                                if fig_corr.layout.xaxis.autorange is not True and fig_corr.layout.xaxis.range:
                                    x_range_corr = list(fig_corr.layout.xaxis.range)
                                if fig_corr.layout.yaxis.autorange is not True and fig_corr.layout.yaxis.range:
                                    y_range_corr = list(fig_corr.layout.yaxis.range)
                            except Exception:
                                pass
                            # Pop last added point (reverse chronological)
                            last = anchor_points.pop()
                            xs = np.asarray(x, dtype=float)
                            ys = np.asarray(y, dtype=float)
                            ap_sorted = sorted(anchor_points)
                            # Update anchor markers
                            try:
                                fig_m.data[1].x = ap_sorted
                                fig_m.data[1].y = [float(ys[int(np.nanargmin(np.abs(xs - ax)))]) for ax in ap_sorted]
                            except Exception:
                                pass
                            # Recompute or clear baseline/corrected
                            if baseline_active and len(anchor_points) >= 2:
                                _preview_baseline()
                            else:
                                # Clear baseline preview and corrected plot if insufficient points
                                try:
                                    while len(fig_m.data) > 2:
                                        fig_m.data = tuple(fig_m.data[:2])
                                    fig_corr.data[0].x = []
                                    fig_corr.data[0].y = []
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        # Reapply stored ranges
                        try:
                            if x_range_main:
                                fig_m.update_xaxes(range=x_range_main, autorange=False)
                            if y_range_main:
                                fig_m.update_yaxes(range=y_range_main, autorange=False)
                        except Exception:
                            pass
                        try:
                            if x_range_corr:
                                fig_corr.update_xaxes(range=x_range_corr, autorange=False)
                            if y_range_corr:
                                fig_corr.update_yaxes(range=y_range_corr, autorange=False)
                        except Exception:
                            pass

                    def _save_file(_b=None):
                        if row is None or len(anchor_points) < 2:
                            with manual_out:
                                print("Need at least two anchor points to save.")
                            return
                        xs = np.asarray(x, dtype=float)
                        ys = np.asarray(y, dtype=float)
                        ap_sorted = np.array(sorted(anchor_points), dtype=float)
                        y_anchor = np.array([ys[int(np.nanargmin(np.abs(xs - ap)))] for ap in ap_sorted], dtype=float)
                        try:
                            spline = CubicSpline(ap_sorted, y_anchor, bc_type=((1, 0.0), (1, 0.0)))
                        except Exception:
                            spline = CubicSpline(ap_sorted, y_anchor)
                        baseline_vals = spline(xs).astype(float)
                        corrected = (ys - baseline_vals).astype(float)
                        # Persist to DataFrame for this file
                        try:
                            FTIR_DataFrame.at[row.name, "Baseline Function"] = "Manual"
                            FTIR_DataFrame.at[row.name, "Baseline Parameters"] = str({"anchor_points": [float(v) for v in ap_sorted.tolist()]})
                            FTIR_DataFrame.at[row.name, "Baseline"] = baseline_vals.tolist()
                            FTIR_DataFrame.at[row.name, "Baseline-Corrected Data"] = corrected.tolist()
                        except Exception:
                            pass
                        with manual_out:
                            print("Saved manual baseline for this file.")

                    def _save_material(_b=None):
                        if row is None or len(anchor_points) < 2:
                            with manual_out:
                                print("Need at least two anchor points to save.")
                            return
                        mat_val = row.get("Material", material)
                        ap_sorted = sorted(anchor_points)
                        try:
                            msk = FTIR_DataFrame["Material"] == mat_val
                            FTIR_DataFrame.loc[msk, "Baseline Function"] = "Manual"
                            FTIR_DataFrame.loc[msk, "Baseline Parameters"] = str({"anchor_points": [float(v) for v in ap_sorted]})
                        except Exception:
                            pass
                        with manual_out:
                            print(f"Saved manual anchor points for material '{mat_val}'.")

                    def _close_m(_b=None):
                        try:
                            fig_m.close()
                        except Exception:
                            pass
                        try:
                            fig_corr.close()
                        except Exception:
                            pass
                        try:
                            manual_container.close()
                        except Exception:
                            pass
                        try:
                            for _w in (manual_container, fig_m, fig_corr):
                                if _w in _TB_WIDGETS:
                                    _TB_WIDGETS.remove(_w)
                        except Exception:
                            pass

                    continue_btn.on_click(_continue)
                    redo_btn.on_click(_redo)
                    undo_btn.on_click(_undo)
                    save_file_btn_m.on_click(_save_file)
                    save_mat_btn_m.on_click(_save_material)
                    close_btn_m.on_click(_close_m)

                    # Filter/spectrum observers for manual mode
                    def _on_mat_m(change):
                        if change.get("name") == "value":
                            _rebuild_conditions_options(); _build_spectrum_options()
                            # Reset selection on filter change
                            try:
                                anchor_points.clear()
                                fig_m.data[1].x = []
                                fig_m.data[1].y = []
                                while len(fig_m.data) > 2:
                                    fig_m.data = tuple(fig_m.data[:2])
                                # clear corrected plot
                                fig_corr.data[0].x = []
                                fig_corr.data[0].y = []
                                # Reset preview state and show Continue again
                                nonlocal baseline_active
                                baseline_active = False
                                try:
                                    continue_btn.layout.display = ""
                                except Exception:
                                    pass
                                _refresh_mark_btns_m()
                            except Exception:
                                pass
                    def _on_cond_m(change):
                        if change.get("name") == "value":
                            _build_spectrum_options()
                            try:
                                anchor_points.clear()
                                fig_m.data[1].x = []
                                fig_m.data[1].y = []
                                while len(fig_m.data) > 2:
                                    fig_m.data = tuple(fig_m.data[:2])
                                fig_corr.data[0].x = []
                                fig_corr.data[0].y = []
                                nonlocal baseline_active
                                baseline_active = False
                                try:
                                    continue_btn.layout.display = ""
                                except Exception:
                                    pass
                                _refresh_mark_btns_m()
                            except Exception:
                                pass
                    def _on_inc_m(change):
                        if change.get("name") == "value":
                            _build_spectrum_options()
                    def _on_spec_m(change):
                        if change.get("name") == "value" and change.get("new") is not None:
                            try:
                                sel_idx3 = change.get("new")
                                r3 = FTIR_DataFrame.loc[sel_idx3]
                                nonlocal row, x, y, material
                                row = r3
                                material = r3.get("Material", material)
                                x = (ast.literal_eval(r3["X-Axis"]) if isinstance(r3["X-Axis"], str) else r3["X-Axis"])
                                y = (ast.literal_eval(r3["Raw Data"]) if isinstance(r3["Raw Data"], str) else r3["Raw Data"])
                                y_arr = np.asarray(y, dtype=float)
                                x_arr = np.asarray(x, dtype=float)
                                fig_m.data[0].x = x_arr; fig_m.data[0].y = y_arr
                                # reset anchors and preview
                                anchor_points.clear(); fig_m.data[1].x = []; fig_m.data[1].y = []
                                while len(fig_m.data) > 2:
                                    fig_m.data = tuple(fig_m.data[:2])
                                fig_corr.data[0].x = []
                                fig_corr.data[0].y = []
                                nonlocal baseline_active
                                baseline_active = False
                                try:
                                    continue_btn.layout.display = ""
                                except Exception:
                                    pass
                                _refresh_mark_btns_m()
                                _set_session_selection(material=row.get("Material"), conditions=row.get("Conditions"), time=row.get("Time"))
                            except Exception:
                                pass
                    def _on_base_m(change):
                        if change.get("name") == "value":
                            new_b = str(change.get("new")).upper()
                            if new_b != "MANUAL":
                                # Switch away: close manual UI and rebuild param UI for new baseline
                                try:
                                    fig_m.close(); fig_corr.close(); manual_container.close()
                                except Exception:
                                    pass
                                try_baseline(FTIR_DataFrame, material=material, baseline_function=new_b, filepath=filepath)
                                return

                    material_dd.observe(_on_mat_m, names="value")
                    conditions_dd.observe(_on_cond_m, names="value")
                    include_bad_cb.observe(_on_inc_m, names="value")
                    spectrum_sel.observe(_on_spec_m, names="value")
                    baseline_dd.observe(_on_base_m, names="value")

                    # Compose UI
                    controls_row_top = widgets.HBox([material_dd, conditions_dd, include_bad_cb, baseline_dd])
                    spec_row = widgets.HBox([spectrum_sel])
                    mark_row_m = widgets.HBox([mark_bad_btn_m, mark_good_btn_m])
                    btn_row_m = widgets.HBox([continue_btn, redo_btn, undo_btn, save_file_btn_m, save_mat_btn_m, close_btn_m])
                    manual_container = widgets.VBox([controls_row_top, spec_row, fig_m, fig_corr, manual_out, mark_row_m, btn_row_m])
                    display(manual_container)
                    try:
                        _TB_WIDGETS.extend([manual_container, fig_m, fig_corr])
                    except Exception:
                        pass
                    _refresh_mark_btns_m()

                # If MANUAL, build manual UI and return
                if baseline_function.upper() == "MANUAL":
                    _build_manual_ui()
                    return

                # Otherwise, proceed to rebuild parameter widgets (refresh defaults for selected baseline)
                nonlocal param_widgets
                param_widgets = {}
                parameters_local = _get_default_parameters(baseline_function)
                parameters_local = _cast_parameter_types(baseline_function, parameters_local)
                if baseline_function.upper() == "ARPLS":
                    param_widgets["lam"] = widgets.FloatSlider(value=parameters_local.get("lam", 1e5), min=1e4, max=1e6, step=1e4, description="Smoothness (lam)", readout_format=".1e", continuous_update=False, style={"description_width": "auto"})
                    param_widgets["max_iter"] = widgets.IntSlider(value=parameters_local.get("max_iter", 50), min=1, max=200, step=1, description="Max Iterations", continuous_update=False, style={"description_width": "auto"})
                    param_widgets["tol"] = widgets.FloatSlider(value=parameters_local.get("tol", 1e-3), min=1e-6, max=1e-1, step=1e-4, description="Tolerance", readout_format=".1e", continuous_update=False, style={"description_width": "auto"})
                elif baseline_function.upper() == "IRSQR":
                    param_widgets["lam"] = widgets.FloatSlider(value=parameters_local.get("lam", 1e6), min=1e5, max=1e7, step=1e5, description="Smoothness (lam)", readout_format=".1e", continuous_update=False, style={"description_width": "auto"})
                    param_widgets["quantile"] = widgets.FloatSlider(value=parameters_local.get("quantile", 0.05), min=0.001, max=0.5, step=0.001, description="Quantile", readout_format=".3f", continuous_update=False, style={"description_width": "auto"})
                    param_widgets["num_knots"] = widgets.IntSlider(value=parameters_local.get("num_knots", 100), min=5, max=500, step=5, description="Knots", continuous_update=False, style={"description_width": "auto"})
                    param_widgets["spline_degree"] = widgets.IntSlider(value=parameters_local.get("spline_degree", 3), min=1, max=5, step=1, description="Spline Degree", continuous_update=False, style={"description_width": "auto"})
                    param_widgets["diff_order"] = widgets.IntSlider(value=parameters_local.get("diff_order", 3), min=1, max=3, step=1, description="Differential Order", continuous_update=False, style={"description_width": "auto"})
                    param_widgets["max_iter"] = widgets.IntSlider(value=parameters_local.get("max_iter", 100), min=1, max=1000, step=1, description="Max Iterations", continuous_update=False, style={"description_width": "auto"})
                    param_widgets["tol"] = widgets.FloatSlider(value=parameters_local.get("tol", 1e-6), min=1e-10, max=1e-2, step=1e-7, description="Tolerance", readout_format=".1e", continuous_update=False, style={"description_width": "auto"})
                elif baseline_function.upper() == "FABC":
                    param_widgets["lam"] = widgets.FloatSlider(value=parameters_local.get("lam", 1e6), min=1e4, max=1e7, step=1e5, description="Smoothness (lam)", readout_format=".1e", continuous_update=False, style={"description_width": "auto"})
                    # scale recompute from selected spectrum
                    try:
                        scale_default2 = int(np.clip(ceil(optimize_window(y) / 2), 2, 500))
                    except Exception:
                        scale_default2 = 50
                    scale_val2 = parameters_local.get("scale") or scale_default2
                    param_widgets["scale"] = widgets.IntSlider(value=int(scale_val2), min=2, max=500, step=1, description="Scale", continuous_update=False, style={"description_width": "auto"})
                    param_widgets["num_std"] = widgets.FloatSlider(value=parameters_local.get("num_std", 3.0), min=1.5, max=4.5, step=0.1, description="Standard Deviations", readout_format=".2f", continuous_update=False, style={"description_width": "auto"})
                    param_widgets["diff_order"] = widgets.IntSlider(value=parameters_local.get("diff_order", 2), min=1, max=3, step=1, description="Differential Order", continuous_update=False, style={"description_width": "auto"})
                    param_widgets["min_length"] = widgets.IntSlider(value=parameters_local.get("min_length", 2), min=1, max=6, step=1, description="Min Baseline Span Length", continuous_update=False, style={"description_width": "auto"})
                # Build full UI identical to main branch
                defaults_full = _get_default_parameters(baseline_function)
                widget_rows_full = []
                for k, w in param_widgets.items():
                    rb = widgets.Button(description="Reset", button_style="info", layout=widgets.Layout(width="70px", margin="0 0 6px 8px"))
                    if k == "scale":
                        def _reset_scale2(_b=None, w=w):
                            try:
                                new_def = int(np.clip(ceil(optimize_window(y) / 2), 2, 500))
                                w.value = new_def
                            except Exception:
                                pass
                        rb.on_click(_reset_scale2)
                    else:
                        rv = defaults_full.get(k, w.value)
                        def make_reset_f(w2, val2):
                            return lambda _b=None: setattr(w2, "value", val2)
                        rb.on_click(make_reset_f(w, rv))
                    widget_rows_full.append(widgets.HBox([w, rb]))
                reset_all_btn2 = widgets.Button(description="Reset All", button_style="warning", layout=widgets.Layout(width="90px", margin="10px 10px 0 0"))
                def _reset_all2(_b=None):
                    for kk, ww in param_widgets.items():
                        if kk == "scale":
                            try:
                                ww.value = int(np.clip(ceil(optimize_window(y) / 2), 2, 500))
                            except Exception:
                                pass
                        elif kk in defaults_full:
                            ww.value = defaults_full[kk]
                reset_all_btn2.on_click(_reset_all2)
                save_file_btn2 = widgets.Button(description="Save for file", button_style="success", layout=widgets.Layout(margin="10px 10px 0 0"))
                save_material_btn2 = widgets.Button(description="Save for material", button_style="info", layout=widgets.Layout(margin="10px 10px 0 0"))
                mark_bad_btn2 = widgets.Button(description="Mark as bad", button_style="danger", layout=widgets.Layout(margin="10px 10px 0 0"))
                mark_good_btn2 = widgets.Button(description="Mark as good", button_style="success", layout=widgets.Layout(margin="10px 10px 0 0"))
                close_btn2 = widgets.Button(description="Close", button_style="danger", layout=widgets.Layout(margin="10px 0 0 0"))
                def _refresh_mark_btns2():
                    try:
                        qcol = _quality_column_name(FTIR_DataFrame)
                        st = FTIR_DataFrame.at[row.name, qcol]
                        is_bad = str(st).strip().lower() == "bad"
                        mark_bad_btn2.layout.display = "none" if is_bad else ""
                        mark_good_btn2.layout.display = "" if is_bad else "none"
                    except Exception:
                        pass
                # Click handlers to mirror plot_grouped_spectra behavior: update Quality and toggle visibility
                def _mark_bad2(_b=None):
                    try:
                        qcol = _quality_column_name(FTIR_DataFrame)
                        FTIR_DataFrame.at[row.name, qcol] = "bad"
                    except Exception:
                        pass
                    try:
                        mark_bad_btn2.layout.display = "none"
                        mark_good_btn2.layout.display = ""
                    except Exception:
                        pass

                def _mark_good2(_b=None):
                    try:
                        qcol = _quality_column_name(FTIR_DataFrame)
                        FTIR_DataFrame.at[row.name, qcol] = "good"
                    except Exception:
                        pass
                    try:
                        mark_bad_btn2.layout.display = ""
                        mark_good_btn2.layout.display = "none"
                    except Exception:
                        pass
                mark_bad_btn2.on_click(_mark_bad2)
                mark_good_btn2.on_click(_mark_good2)
                def _current_params2():
                    cur = parameters_local.copy()
                    for kk, ww in param_widgets.items():
                        cur[kk] = ww.value
                    return _cast_parameter_types(baseline_function, cur)
                def _serialize2(d):
                    def to_plain(v):
                        try:
                            if isinstance(v, (np.integer,)): return int(v)
                            if isinstance(v, (np.floating,)): return float(v)
                            if isinstance(v, np.ndarray): return v.tolist()
                        except Exception: pass
                        return v
                    return {kk: to_plain(vv) for kk, vv in d.items()}
                def _save_file2(_b=None):
                    if row is None: return
                    pv = _serialize2(_current_params2())
                    FTIR_DataFrame.at[row.name, "Baseline Function"] = baseline_function.upper()
                    FTIR_DataFrame.at[row.name, "Baseline Parameters"] = str(pv)
                    note = widgets.Output(); display(note)
                    with note:
                        print("Saved baseline settings for this file.")
                def _save_material2(_b=None):
                    if row is None: return
                    pv = _serialize2(_current_params2())
                    mat_val = row.get("Material", material)
                    msk = FTIR_DataFrame["Material"] == mat_val
                    FTIR_DataFrame.loc[msk, "Baseline Function"] = baseline_function.upper()
                    FTIR_DataFrame.loc[msk, "Baseline Parameters"] = str(pv)
                    note = widgets.Output(); display(note)
                    with note:
                        print(f"Saved baseline settings for material '{mat_val}'.")
                save_file_btn2.on_click(_save_file2)
                save_material_btn2.on_click(_save_material2)
                def _close_full2(_b=None):
                    try: plt.close("all")
                    except Exception: pass
                    try: full_container.close()
                    except Exception: pass
                    try:
                        if full_container in _TB_WIDGETS:
                            _TB_WIDGETS.remove(full_container)
                    except Exception:
                        pass
                close_btn2.on_click(_close_full2)
                # Observers for filters in full mode
                def _on_mat_full(change):
                    if change.get("name") == "value":
                        _rebuild_conditions_options(); _build_spectrum_options(); _plot_baseline(**{kk: ww.value for kk, ww in param_widgets.items()}); _refresh_mark_btns2()
                def _on_cond_full(change):
                    if change.get("name") == "value":
                        _build_spectrum_options(); _plot_baseline(**{kk: ww.value for kk, ww in param_widgets.items()}); _refresh_mark_btns2()
                def _on_inc_full(change):
                    if change.get("name") == "value":
                        _build_spectrum_options(); _plot_baseline(**{kk: ww.value for kk, ww in param_widgets.items()}); _refresh_mark_btns2()
                def _on_base_full(change):
                    if change.get("name") == "value":
                        new_val = str(change.get("new")).upper()
                        if new_val == "MANUAL":
                            # Switch to manual mode inline
                            try:
                                full_container.close()
                            except Exception:
                                pass
                            nonlocal baseline_function
                            baseline_function = new_val
                            _build_manual_ui()
                            return
                        # Switch to non-manual: rebuild by re-entering function
                        try_baseline(FTIR_DataFrame, material=material, baseline_function=new_val, filepath=filepath)
                material_dd.observe(_on_mat_full, names="value")
                conditions_dd.observe(_on_cond_full, names="value")
                include_bad_cb.observe(_on_inc_full, names="value")
                baseline_dd.observe(_on_base_full, names="value")
                def _on_spec_full(change):
                    if change.get("name") == "value" and change.get("new") is not None:
                        try:
                            sel_idx2 = change.get("new")
                            rsel2 = FTIR_DataFrame.loc[sel_idx2]
                            nonlocal row, x, y, material
                            row = rsel2
                            material = rsel2.get("Material", material)
                            x = (ast.literal_eval(rsel2["X-Axis"]) if isinstance(rsel2["X-Axis"], str) else rsel2["X-Axis"])
                            y = (ast.literal_eval(rsel2["Raw Data"]) if isinstance(rsel2["Raw Data"], str) else rsel2["Raw Data"])
                            y = np.array(y, dtype=float)
                            # If currently using MANUAL baseline, switch into the integrated inline manual UI
                            if baseline_function.upper() == "MANUAL":
                                try:
                                    full_container.close()
                                except Exception:
                                    pass
                                _build_manual_ui()
                                return
                            if baseline_function.upper() == "FABC" and "scale" in param_widgets:
                                try:
                                    new_scale3 = int(np.clip(ceil(optimize_window(y) / 2), 2, 500))
                                    param_widgets["scale"].value = new_scale3
                                except Exception: pass
                            _plot_baseline(**{kk: ww.value for kk, ww in param_widgets.items()})
                            _refresh_mark_btns2()
                            _set_session_selection(material=row.get("Material"), conditions=row.get("Conditions"), time=row.get("Time"))
                        except Exception:
                            pass
                spectrum_sel.observe(_on_spec_full, names="value")
                # Build final UI
                if baseline_function.upper() != "MANUAL":
                    mark_row2 = widgets.HBox([mark_bad_btn2, mark_good_btn2])
                    controls_footer2 = widgets.HBox([save_file_btn2, save_material_btn2, reset_all_btn2, close_btn2])
                    ui_full = widgets.VBox([widgets.HBox([material_dd, conditions_dd, include_bad_cb, baseline_dd]), widgets.HBox([spectrum_sel])] + widget_rows_full + [mark_row2, controls_footer2])
                    full_container = widgets.VBox([ui_full, output])
                    display(full_container)
                    try:
                        _TB_WIDGETS.extend([full_container])
                    except Exception:
                        pass
                    _refresh_mark_btns2()
                    # Parameter slider -> live plot updates
                    def _on_param_change(change):
                        if change.get("name") == "value":
                            _plot_baseline(**{kk: ww.value for kk, ww in param_widgets.items()})
                    for _pw in param_widgets.values():
                        try:
                            _pw.observe(_on_param_change, names="value")
                        except Exception:
                            pass
                    # Initial plot
                    _plot_baseline(**{kk: ww.value for kk, ww in param_widgets.items()})
                else:
                    # If MANUAL, delegate to manual UI builder
                    _build_manual_ui()

        def _on_close_min(_b=None):
            try:
                plt.close("all")
            except Exception:
                pass
            try:
                container.close()
            except Exception:
                pass
            try:
                if container in _TB_WIDGETS:
                    _TB_WIDGETS.remove(container)
            except Exception:
                pass

        material_dd.observe(_on_mat_min, names="value")
        conditions_dd.observe(_on_cond_min, names="value")
        include_bad_cb.observe(_on_inc_min, names="value")
        baseline_dd.observe(_on_base_min, names="value")
        spectrum_sel.observe(_on_spec_min, names="value")
        close_btn.on_click(_on_close_min)

    # Stay in minimal mode until selection; skip legacy matplotlib-based UI below
    return FTIR_DataFrame


def JSON_population(
    FTIR_DataFrame,
    materials_json_path=None,
):
    """
    Populate materials.json with materials, conditions, and time values from FTIR_DataFrame.

    Parameters
    ------
    FTIR_DataFrame : pd.DataFrame
        In-memory FTIR dataset containing at least the columns 'Material',
        'Conditions', and 'Time'.
    materials_json_path : str | None
        Path to the materials.json file. Defaults to Trenton_Project/materials.json
        alongside this module.

    Behavior
    --------
    - Preserves existing M000 metadata block as-is.
    - Adds or updates entries (M001, M002, ...) for each unique material in the
      DataFrame, creating a minimal structure compatible with materials_backup.json:
        {
          "name": material,
          "alias": material,
          "peaks": {
            "1": {
              "name": "",
              "center_wavenumber": 0,
              "sg": 0,
              "sl": 0,
              "f": 0,
              "conditions": {
                 <condition>: {"time": [...], "A": []}, ...
              }
            }
          }
        }
    """
    # Validate required columns early for clearer errors
    _require_columns(
        FTIR_DataFrame,
        ["Material", "Conditions", "Time"],
        context="FTIR_DataFrame (JSON_population)",
    )

    # Resolve default path relative to this file
    if materials_json_path is None:
        base_dir = os.path.dirname(__file__)
        materials_json_path = os.path.join(base_dir, "materials.json")

    # Load existing JSON; expect a top-level list containing a single object
    try:
        with open(materials_json_path, "r", encoding="utf-8") as f:
            content = json.load(f)
    except FileNotFoundError:
        # Initialize with an empty shell if missing (will add M000 if present later)
        content = [{}]
    if not isinstance(content, list) or not content:
        # Normalize to expected shape
        content = [content if isinstance(content, dict) else {}]
    top = content[0]

    # Build reverse index from existing entries by alias/name to code keys (Mxxx)
    def _material_key_lookup(material_str):
        for code_key, payload in top.items():
            if not isinstance(payload, dict):
                continue
            alias = payload.get("alias")
            name = payload.get("name")
            if alias == material_str or name == material_str:
                return code_key
        return None

    # Compute next available M### index
    def _next_material_code():
        nums = []
        for k in top.keys():
            if (
                isinstance(k, str)
                and len(k) == 4
                and k.startswith("M")
                and k[1:].isdigit()
            ):
                nums.append(int(k[1:]))
        nxt = max(nums) + 1 if nums else 0
        return f"M{nxt:03d}"

    # Extract materials, conditions, times from DataFrame
    # Normalize data: coerce to strings/ints where appropriate, drop missing
    df = FTIR_DataFrame.copy()
    # Drop rows without Material or Conditions
    df = df[~df["Material"].isna() & ~df["Conditions"].isna()]

    # Ensure Time is numeric (nullable ints), ignore NaN times for the time list
    try:
        df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    except Exception:
        pass

    # Iterate materials
    for material in sorted(df["Material"].dropna().astype(str).unique()):
        mat_df = df[df["Material"].astype(str) == material]

        # Build condition -> sorted unique times mapping
        cond_map = {}
        for condition, cdf in mat_df.groupby("Conditions"):
            if pd.isna(condition):
                continue
            cond_str = str(condition)
            times = (
                cdf["Time"]
                .dropna()
                .astype(float)
                .astype(int)
                .sort_values()
                .unique()
                .tolist()
            )
            # Always include conditions, even when no valid times were found (time: [])
            cond_map[cond_str] = {"time": times, "A": []}

        # If no conditions found for this material, skip writing this material
        if not cond_map:
            continue

        # Find existing code or allocate a new one
        code = _material_key_lookup(material)
        if code is None:
            code = _next_material_code()

        # Prepare minimal peaks structure (one peak only: "1")
        peaks = {
            "1": {
                "name": "",
                "center_wavenumber": 0,
                "sg": 0,
                "sl": 0,
                "f": 0,
                "conditions": cond_map,
            }
        }

        payload = top.get(code, {}) if isinstance(top.get(code, {}), dict) else {}
        # Do not overwrite existing name/alias; only set if missing. If different values
        # are already present, report that overwrite was blocked.
        if "name" not in payload:
            payload["name"] = material
        else:
            if str(payload.get("name")) != str(material):
                print(
                    f"[JSON_population] Overwrite blocked for {code}.name: keeping existing '{payload.get('name')}', observed '{material}'."
                )
        if "alias" not in payload:
            payload["alias"] = material
        else:
            if str(payload.get("alias")) != str(material):
                print(
                    f"[JSON_population] Overwrite blocked for {code}.alias: keeping existing '{payload.get('alias')}', observed '{material}'."
                )

        # Merge peaks non-destructively; keep existing peaks and fields
        existing_peaks = payload.get("peaks", {})
        if not isinstance(existing_peaks, dict):
            existing_peaks = {}

        # Ensure peak "1" exists; if it does, don't overwrite numeric fields
        peak1 = existing_peaks.get("1", {})
        if not isinstance(peak1, dict):
            peak1 = {}
        # Set defaults only if missing
        peak1.setdefault("name", "")
        peak1.setdefault("center_wavenumber", 0)
        peak1.setdefault("sg", 0)
        peak1.setdefault("sl", 0)
        peak1.setdefault("f", 0)

        # Merge conditions: union times; keep existing A arrays intact. If an existing
        # condition is found, we do not overwrite its values—report actions taken.
        existing_conditions = peak1.get("conditions", {})
        if not isinstance(existing_conditions, dict):
            existing_conditions = {}
        for cond_str, new_payload in cond_map.items():
            if cond_str in existing_conditions and isinstance(
                existing_conditions[cond_str], dict
            ):
                # Merge times
                old_times = existing_conditions[cond_str].get("time", [])
                try:
                    old_times_list = (
                        list(old_times) if isinstance(old_times, (list, tuple)) else []
                    )
                except Exception:
                    old_times_list = []
                new_times_list = list(new_payload.get("time", []))
                merged = sorted(
                    {int(t) for t in old_times_list if pd.notna(t)}
                    | {int(t) for t in new_times_list}
                )
                # Preserve existing A array (or default [])
                A_list = existing_conditions[cond_str].get("A", [])
                if not isinstance(A_list, list):
                    A_list = []
                # Logging: explicitly note non-overwrite behavior
                if old_times_list or A_list:
                    print(
                        "[JSON_population] Existing entry preserved for "
                        f"{code}.peaks['1'].conditions['{cond_str}'] — "
                        f"merged times (old {len(old_times_list)} + new {len(new_times_list)} -> {len(merged)}); "
                        f"kept existing A (len {len(A_list)})."
                    )
                existing_conditions[cond_str] = {"time": merged, "A": A_list}
            else:
                # New condition: add as-is with empty A list (already provided)
                times_copy = list(new_payload.get("time", []))
                existing_conditions[cond_str] = {"time": times_copy, "A": []}

        peak1["conditions"] = existing_conditions
        existing_peaks["1"] = peak1
        payload["peaks"] = existing_peaks
        top[code] = payload

    # Write back to file with pretty formatting
    with open(materials_json_path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=4, ensure_ascii=False)

    print(f"materials.json updated at: {materials_json_path}")


def test_baseline_choices(FTIR_DataFrame, material=None):
    """
    Plot three random spectra for a given material, showing baseline results.

    Plots raw data, baseline, and baseline-corrected data. The baseline function and
    parameters are taken from the DataFrame columns. Assumes user has already filled
    those columns earlier in the workflow.

    Parameters
    ----------
    FTIR_DataFrame : pd.DataFrame
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
    filtered = FTIR_DataFrame[FTIR_DataFrame["Material"] == material]
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
        # Robustly coerce parameters
        raw_params = row.get("Baseline Parameters", {})
        if isinstance(raw_params, dict):
            params = raw_params.copy()
        elif isinstance(raw_params, str) and raw_params.strip():
            try:
                maybe = ast.literal_eval(raw_params)
                params = (
                    maybe if isinstance(maybe, dict) else _parse_parameters(raw_params)
                )
            except Exception:
                params = _parse_parameters(raw_params)
        else:
            params = {}

        # Compute baseline
        baseline = None
        baseline_corrected = None
        try:
            if baseline_func is None:
                raise ValueError("No baseline function specified.")
            func = baseline_func.strip().upper()
            # Merge with defaults and cast types
            defaults = _get_default_parameters(func)
            params = {**defaults, **params}
            params = _cast_parameter_types(func, params)
            if func == "ARPLS":
                result = arpls(y, **params)
            elif func == "IRSQR":
                if "x_data" in params:
                    result = irsqr(y, **params)
                else:
                    result = irsqr(y, **params, x_data=x)
            elif func == "FABC":
                result = fabc(y, **params)
            elif func == "MANUAL":
                anchor_points = params.get("anchor_points", [])
                if not anchor_points:
                    raise ValueError("No anchor_points for MANUAL baseline.")
                anchor_indices = [
                    min(range(len(x)), key=lambda i: abs(x[i] - ap))
                    for ap in anchor_points
                ]
                y_anchor = [y[i] for i in anchor_indices]
                result = CubicSpline(x=anchor_points, y=y_anchor, extrapolate=True)(x)
            else:
                raise ValueError(f"Unknown baseline function: {baseline_func}")
            # Normalize return type to baseline array
            if isinstance(result, tuple):
                baseline = result[0]
            elif isinstance(result, dict):
                baseline = result.get("baseline", None)
                if baseline is None:
                    raise ValueError(
                        "Baseline function did not return a baseline array."
                    )
            else:
                baseline = result
            baseline = np.asarray(baseline, dtype=float)
            baseline_corrected = y - baseline
        except Exception as e:
            print(f"Error computing baseline for row {idx}: {e}")
            print(f" - Baseline Function: {baseline_func}")
            print(f" - Baseline Parameters: {params}")
            print(f" - X-Axis shape: {np.shape(x)}, Raw Data shape: {np.shape(y)}")
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


def bring_in_DataFrame(DataFrame_path=None):
    """
    Load the CSV file into a pandas DataFrame.

    Allows for easy DataFrame manipulation in memory over the course of the analysis.

    Parameters
    ----------
    DataFrame_path : str
        The path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.
    """
    if DataFrame_path is None:
        DataFrame_path = "FTIR_DataFrame.csv"  # Default path if none is provided (will
        # be in active directory)
    else:
        pass
    if os.path.exists(DataFrame_path):
        FTIR_DataFrame = pd.read_csv(
            DataFrame_path
        )  # Load the DataFrame from the specified path
    else:
        FTIR_DataFrame = (
            pd.DataFrame()
        )  # Create a new empty DataFrame if it doesn't exist
    return FTIR_DataFrame, DataFrame_path


def spectral_normalization(FTIR_DataFrame, filepath=None):
    """
    Interactively select and save a normalization peak range for FTIR spectra.

    Plots either a predefined specific file (via filepath) or the first time-zero
    file for a specified material. The user selects two points on the plot to define
    an x-range (wavenumber window) for normalization. The selected range is printed
    and can be saved to the DataFrame column 'Normalization Peak Wavenumber'.

    Parameters
    ----------
    FTIR_DataFrame : pd.DataFrame
        The DataFrame containing the spectral data.
    filepath : str, optional
        If provided, preselect this file (full path or just filename).

    Returns
    -------
    pd.DataFrame
        The updated DataFrame with the selected normalization peak range saved.
    """

    clear_output(wait=True)

    # Ensure destination column exists
    target_col = "Normalization Peak Wavenumber"
    if target_col not in FTIR_DataFrame.columns:
        FTIR_DataFrame[target_col] = None

    # Identify condition column name
    cond_col = (
        "Conditions"
        if "Conditions" in FTIR_DataFrame.columns
        else ("Condition" if "Condition" in FTIR_DataFrame.columns else None)
    )

    # Build dropdown options
    try:
        df_all = FTIR_DataFrame.copy()
        # Exclude rows marked as bad quality
        try:
            df_all = df_all[_quality_good_mask(df_all)]
        except Exception:
            pass
    except Exception:
        df_all = FTIR_DataFrame

    materials = (
        sorted(
            {
                str(v)
                for v in df_all.get("Material", pd.Series([], dtype=object))
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            }
        )
        if "Material" in df_all.columns
        else []
    )
    cond_series = (
        df_all[cond_col]
        if cond_col and (cond_col in df_all.columns)
        else pd.Series([], dtype=object)
    )
    conditions = (
        sorted(
            [
                s
                for s in [
                    str(v) for v in cond_series.dropna().astype(str).unique().tolist()
                ]
                if s.strip().lower() != "unexposed"
            ]
        )
        if cond_col
        else []
    )

    # Preselect by filepath/material if provided
    preselect_idx = None
    preselect_material = None
    preselect_condition = None
    if filepath is not None:
        if os.path.sep in filepath:
            folder, fname = os.path.split(filepath)
            flt = df_all[
                (df_all.get("File Location", "") == folder)
                & (df_all.get("File Name", "") == fname)
            ]
        else:
            flt = df_all[df_all.get("File Name", "") == filepath]
        if not flt.empty:
            preselect_idx = flt.index[0]
            preselect_material = str(flt.iloc[0].get("Material", "any"))
            if cond_col is not None:
                preselect_condition = str(flt.iloc[0].get(cond_col, "any"))
    # No material argument; selection is made via the dropdowns

    # Helper: perform normalization for a material (formerly spectrum_normalization)
    def _normalize(material_name: str):
        if FTIR_DataFrame is None:
            raise ValueError("FTIR_DataFrame is None.")
        if material_name is None or str(material_name).strip() == "":
            raise ValueError("Material must be provided for normalization.")

        source_column = "Baseline-Corrected Data"
        dest_column = "Normalized and Corrected Data"
        range_column = "Normalization Peak Wavenumber"

        subset = FTIR_DataFrame[FTIR_DataFrame["Material"] == material_name]
        if subset.empty:
            raise ValueError(f"No rows found for material '{material_name}'.")

        # Determine x-axis column
        x_axis_column = "X-Axis" if "X-Axis" in FTIR_DataFrame.columns else None
        if x_axis_column is None:
            raise KeyError(
                "Could not find an x-axis column ('X-Axis') in the DataFrame."
            )
        # Validate required columns for normalization
        _require_columns(
            FTIR_DataFrame,
            ["Material", source_column, range_column, x_axis_column],
            context="FTIR_DataFrame (_normalize)",
        )

        # Ensure source/destination columns are object dtype (hold per-row lists)
        if source_column in FTIR_DataFrame.columns:
            try:
                FTIR_DataFrame[source_column] = FTIR_DataFrame[source_column].astype(
                    object
                )
            except Exception:
                pass
        # Ensure destination column exists and is object dtype
        if dest_column not in FTIR_DataFrame.columns:
            FTIR_DataFrame[dest_column] = None
        try:
            FTIR_DataFrame[dest_column] = FTIR_DataFrame[dest_column].astype(object)
        except Exception:
            pass

        # Normalize each spectrum by its own max within the normalization window
        updated = 0
        skipped = 0
        errors = []
        for idx, row in subset.iterrows():
            try:
                x = row.get(x_axis_column)
                y = row.get(source_column)
                rng = row.get(range_column)
                # parse potential string-literals
                if isinstance(x, str):
                    x = ast.literal_eval(x)
                if isinstance(y, str):
                    y = ast.literal_eval(y)
                if isinstance(rng, str):
                    rng = ast.literal_eval(rng)
                if not isinstance(rng, (list, tuple)) or len(rng) != 2:
                    skipped += 1
                    continue
                x = np.asarray(x, dtype=float)
                y = np.asarray(y, dtype=float)
                lo, hi = float(min(rng)), float(max(rng))
                if x.size == 0 or y.size == 0 or x.size != y.size:
                    skipped += 1
                    continue
                # window mask
                mask = (x >= lo) & (x <= hi)
                if not np.any(mask):
                    skipped += 1
                    continue
                local_max = np.nanmax(y[mask])
                if not np.isfinite(local_max) or local_max == 0:
                    skipped += 1
                    continue
                FTIR_DataFrame.at[idx, dest_column] = (y / local_max).tolist()
                updated += 1
            except Exception as e:
                skipped += 1
                errors.append((idx, str(e)))

        print(
            f"Normalized material '{material_name}': updated {updated} spectra; skipped "
            f"{skipped} (missing/invalid range or data). Each spectrum scaled by its own peak."
        )
        if updated == 0 and skipped > 0 and errors:
            # Provide a small hint in the console; keep UI uncluttered
            print(
                "Note: Some rows lacked valid ranges or data for normalization. "
                f"First error: {errors[0][1]}"
            )

    # Widgets: Material/Conditions/Spectrum
    material_dd = widgets.Dropdown(
        options=["any"] + materials,
        value=(preselect_material if preselect_material in materials else "any"),
        description="Material",
        layout=widgets.Layout(width="40%"),
    )
    conditions_dd = widgets.Dropdown(
        options=(["any"] + conditions if cond_col else ["any"]),
        value=(preselect_condition if (preselect_condition in conditions) else "any"),
        description=("Conditions" if cond_col else "Conditions"),
        layout=widgets.Layout(width="40%"),
    )
    include_bad_cb = widgets.Checkbox(value=False, description="Include bad spectra")
    spectrum_sel = widgets.Dropdown(
        options=[], description="Spectrum", layout=widgets.Layout(width="70%")
    )

    # Apply session defaults for material/conditions dropdowns before building spectra list
    try:
        _sess = _get_session_defaults()
        if _sess.get("material") in material_dd.options:
            material_dd.value = _sess.get("material")
        if _sess.get("conditions") in conditions_dd.options:
            conditions_dd.value = _sess.get("conditions")
    except Exception:
        pass

    info_out = widgets.Output()
    msg_out = widgets.Output()

    # State shared with callbacks
    selected_points = []  # up to two x positions
    x_data = []
    y_data = []

    # --- Plot ---
    fig = go.FigureWidget(data=[go.Scatter(x=[], y=[], mode="lines", name="Raw Data")])
    fig.update_layout(
        title="Select Normalization Range",
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="Absorbance (AU)",
    )

    def _clear_selection_visuals():
        fig.layout.shapes = ()

    def _draw_first_click(x0: float):
        vline = dict(
            type="line",
            x0=x0,
            x1=x0,
            y0=(float(np.nanmin(y_data)) if len(y_data) else 0.0),
            y1=(float(np.nanmax(y_data)) if len(y_data) else 1.0),
            line=dict(color="red", dash="dot"),
            name="norm_vline_first",
        )
        fig.add_shape(vline)

    def _draw_selection_visuals(x0, x1):
        y0min = float(np.nanmin(y_data)) if len(y_data) else 0.0
        y0max = float(np.nanmax(y_data)) if len(y_data) else 1.0
        vline1 = dict(
            type="line",
            x0=x0,
            x1=x0,
            y0=y0min,
            y1=y0max,
            line=dict(color="red", dash="dash"),
        )
        vline2 = dict(
            type="line",
            x0=x1,
            x1=x1,
            y0=y0min,
            y1=y0max,
            line=dict(color="red", dash="dash"),
        )
        rect = dict(
            type="rect",
            x0=min(x0, x1),
            x1=max(x0, x1),
            y0=y0min,
            y1=y0max,
            fillcolor="rgba(0,128,0,0.15)",
            line=dict(width=0),
            layer="below",
        )
        fig.add_shape(vline1)
        fig.add_shape(vline2)
        fig.add_shape(rect)

    def _on_click(trace, points, selector):
        if not points.xs:
            return
        x_val = float(points.xs[0])
        if len(selected_points) >= 2:
            selected_points.clear()
            _clear_selection_visuals()
        selected_points.append(x_val)
        if len(selected_points) == 1:
            _clear_selection_visuals()
            _draw_first_click(x_val)
            with msg_out:
                clear_output(wait=True)
                print(f"First point set at x = {x_val:.3f} cm⁻¹. Click second point…")
        elif len(selected_points) == 2:
            a, b = selected_points
            _clear_selection_visuals()
            _draw_selection_visuals(a, b)
            with msg_out:
                clear_output(wait=True)
                lo, hi = (min(a, b), max(a, b))
                print(f"Selected normalization range: [{lo:.3f}, {hi:.3f}] cm⁻¹")

    fig.data[0].on_click(_on_click)

    def _current_range():
        if len(selected_points) != 2:
            return None
        a, b = selected_points
        return [float(min(a, b)), float(max(a, b))]

    def _get_xy(idx):
        row = FTIR_DataFrame.loc[idx]
        x = row.get("X-Axis")
        y = row.get("Raw Data")
        if isinstance(x, str):
            try:
                x = ast.literal_eval(x)
            except Exception:
                pass
        if isinstance(y, str):
            try:
                y = ast.literal_eval(y)
            except Exception:
                pass
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    def _row_filepath(idx):
        r = FTIR_DataFrame.loc[idx]
        return os.path.join(
            str(r.get("File Location", "")), str(r.get("File Name", ""))
        )

    def _row_label(idx):
        r = FTIR_DataFrame.loc[idx]
        t = r.get("Time", "?")
        mat = r.get("Material", "?")
        cond = r.get(cond_col, "?") if cond_col else "?"
        fname = r.get("File Name", "?")
        return f"{mat} | {cond} | t={t} | {fname}"

    def _rebuild_spectrum_options(*_):
        mask = pd.Series([True] * len(FTIR_DataFrame))
        if material_dd.value != "any":
            mask &= FTIR_DataFrame.get(
                "Material", pd.Series([None] * len(FTIR_DataFrame))
            ).astype(str) == str(material_dd.value)
        if cond_col and conditions_dd.value != "any":
            cond_vals = FTIR_DataFrame.get(
                cond_col, pd.Series([None] * len(FTIR_DataFrame))
            ).astype(str)
            # Include rows matching the selected condition OR marked as 'unexposed'
            mask &= (cond_vals == str(conditions_dd.value)) | (
                cond_vals.str.strip().str.lower() == "unexposed"
            )
        filtered = FTIR_DataFrame[mask]
        # Optionally exclude rows marked as bad quality
        try:
            if not include_bad_cb.value:
                filtered = filtered[_quality_good_mask(filtered)]
        except Exception:
            pass
        # sort by Time if present
        if "Time" in filtered.columns:
            try:
                filtered = filtered.sort_values(by="Time")
            except Exception:
                pass
        opts = [(_row_label(i), i) for i in filtered.index]
        if not opts:
            spectrum_sel.options = [("<no spectra>", None)]
            spectrum_sel.value = None
            with info_out:
                clear_output(wait=True)
                print("No spectra match the current filters.")
            return
        spectrum_sel.options = opts
        # choose preselect or first
        if spectrum_sel.value not in [v for _, v in opts]:
            spectrum_sel.value = opts[0][1]

    def _update_plot_for_selection(*_):
        idx = spectrum_sel.value
        if idx is None:
            return
        nonlocal x_data, y_data
        x_data, y_data = _get_xy(idx)
        # update trace
        fig.data[0].x = x_data.tolist()
        fig.data[0].y = y_data.tolist()
        _clear_selection_visuals()
        selected_points.clear()
        # update title and info
        mat_val = FTIR_DataFrame.loc[idx].get("Material", "?")
        fig.update_layout(title=f"Select Normalization Range | Material: {mat_val}")
        with info_out:
            clear_output(wait=True)
            print(f"Plotting: {_row_filepath(idx)}")
            print("Click two points to define the normalization range.")
        # if this row already has a saved range, visualize it
        try:
            rng = FTIR_DataFrame.loc[idx].get(target_col, None)
            if isinstance(rng, str):
                rng = ast.literal_eval(rng)
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                a, b = float(rng[0]), float(rng[1])
                selected_points[:] = [a, b]
                _clear_selection_visuals()
                _draw_selection_visuals(a, b)
                with msg_out:
                    clear_output(wait=True)
                    lo, hi = (min(a, b), max(a, b))
                    print(
                        f"Existing normalization range: [{lo:.3f}, {hi:.3f}] cm⁻¹ (not yet re-saved)"
                    )
        except Exception:
            pass
        try:
            _refresh_mark_buttons()
        except Exception:
            pass
        # Persist time selection for session
        try:
            if idx is not None and "Time" in FTIR_DataFrame.columns:
                _set_session_selection(time=FTIR_DataFrame.loc[idx].get("Time"))
        except Exception:
            pass

    # Build initial options and selection
    _rebuild_spectrum_options()
    # If filepath/material preselection points to a specific row, set it now
    if preselect_idx is not None:
        # adjust dropdowns if needed to include this row context
        try:
            r = FTIR_DataFrame.loc[preselect_idx]
            if str(r.get("Material", "any")) in material_dd.options:
                material_dd.value = str(r.get("Material", "any"))
            if cond_col and (str(r.get(cond_col, "any")) in conditions_dd.options):
                conditions_dd.value = str(r.get(cond_col, "any"))
        except Exception:
            pass
        _rebuild_spectrum_options()
        try:
            spectrum_sel.value = preselect_idx
        except Exception:
            pass

    # Now update the plot for the current selection
    _update_plot_for_selection()

    # --- Buttons ---
    save_spec_btn = widgets.Button(
        description="Save for this file", button_style="success"
    )
    save_mat_btn = widgets.Button(
        description="Save for this material", button_style="info"
    )
    normalize_btn = widgets.Button(
        description="Normalize material", button_style="primary"
    )
    redo_btn = widgets.Button(description="Redo", button_style="warning")
    cancel_btn = widgets.Button(description="Close", button_style="danger")
    mark_bad_btn = widgets.Button(description="Mark as bad", button_style="danger")
    mark_good_btn = widgets.Button(description="Mark as good", button_style="success")
    # Main control row excludes mark buttons; they go on their own row as a pair
    btn_box = widgets.HBox(
        [save_spec_btn, save_mat_btn, redo_btn, normalize_btn, cancel_btn]
    )
    mark_row = widgets.HBox([mark_bad_btn, mark_good_btn])

    def _refresh_mark_buttons():
        try:
            idx = spectrum_sel.value
            qcol = _quality_column_name(FTIR_DataFrame)
            status = None
            try:
                if idx is not None:
                    status = FTIR_DataFrame.at[idx, qcol]
            except Exception:
                status = None
            is_bad = str(status).strip().lower() == "bad"
            mark_bad_btn.layout.display = "none" if is_bad else ""
            mark_good_btn.layout.display = "" if is_bad else "none"
        except Exception:
            pass

    def _finalize_and_clear():
        try:
            fig.data[0].on_click(None)
        except Exception:
            pass
        try:
            fig.close()
        except Exception:
            pass
        for w in (
            save_spec_btn,
            save_mat_btn,
            normalize_btn,
            redo_btn,
            cancel_btn,
            btn_box,
            # Controls
            material_dd,
            conditions_dd,
            include_bad_cb,
            spectrum_sel,
            # Containers
            # These may fail to close in some front-ends; wrap in try/except
            'controls_row',  # placeholder marker; handled below
            'spectrum_row',   # placeholder marker; handled below
            'mark_row',       # placeholder marker; handled below
            # Outputs
            info_out,
            msg_out,
            # Mark buttons
            mark_bad_btn,
            mark_good_btn,
        ):
            try:
                # Skip placeholder strings; close real widget objects below
                if isinstance(w, str):
                    continue
                w.close()
            except Exception:
                pass
        # Best-effort: close row containers if available in this scope
        try:
            controls_row.close()
        except Exception:
            pass
        try:
            spectrum_row.close()
        except Exception:
            pass
        try:
            mark_row.close()
        except Exception:
            pass

    def _save_for_this_spectrum(_b=None):
        idx = spectrum_sel.value
        if idx is None:
            with msg_out:
                clear_output(wait=True)
                print("No spectrum selected.")
            return
        rng = _current_range()
        if rng is None:
            with msg_out:
                clear_output(wait=True)
                print("Please select two points before saving.")
            return
        FTIR_DataFrame.at[idx, target_col] = str(rng)
        with msg_out:
            clear_output(wait=True)
            print(f"Saved normalization peak range {rng} for this spectrum.")

    def _save_for_this_material(_b=None):
        idx = spectrum_sel.value
        if idx is None:
            with msg_out:
                clear_output(wait=True)
                print("No spectrum selected.")
            return
        rng = _current_range()
        if rng is None:
            with msg_out:
                clear_output(wait=True)
                print("Please select two points before saving.")
            return
        mat = FTIR_DataFrame.loc[idx].get("Material", None)
        if mat is None:
            with msg_out:
                clear_output(wait=True)
                print("Row has no 'Material' value; cannot save for material.")
            return
        mask = FTIR_DataFrame["Material"] == mat
        FTIR_DataFrame.loc[mask, target_col] = str(rng)
        with msg_out:
            clear_output(wait=True)
            print(f"Saved normalization peak range {rng} for material '{mat}'.")

    def _normalize_material(_b=None):
        idx = spectrum_sel.value
        if idx is None:
            with msg_out:
                clear_output(wait=True)
                print("No spectrum selected.")
            return
        mat = FTIR_DataFrame.loc[idx].get("Material", None)
        if mat is None:
            with msg_out:
                clear_output(wait=True)
                print("Selected row has no 'Material'.")
            return
        # Run normalization; catch and display any errors cleanly
        try:
            _normalize(mat)
            with msg_out:
                clear_output(wait=True)
                print(f"Normalization complete for material '{mat}'.")
        except Exception as e:
            with msg_out:
                clear_output(wait=True)
                print(f"Normalization failed for material '{mat}': {e}")

    def _redo(_b=None):
        selected_points.clear()
        _clear_selection_visuals()
        with msg_out:
            clear_output(wait=True)
            print("Selection cleared. Click two points to select a range.")

    def _close(_b=None):
        _finalize_and_clear()

    def _mark_bad(_b=None):
        try:
            idx = spectrum_sel.value
            if idx is None:
                return
            qcol = _quality_column_name(FTIR_DataFrame)
            FTIR_DataFrame.at[idx, qcol] = "bad"
            with msg_out:
                clear_output(wait=True)
                print(f"Marked row {idx} as bad quality.")
            # Rebuild options so the bad row no longer appears
            _rebuild_spectrum_options()
            _update_plot_for_selection()
        except Exception:
            pass
        try:
            _refresh_mark_buttons()
        except Exception:
            pass

    def _mark_good(_b=None):
        try:
            idx = spectrum_sel.value
            if idx is None:
                return
            qcol = _quality_column_name(FTIR_DataFrame)
            FTIR_DataFrame.at[idx, qcol] = "good"
            with msg_out:
                clear_output(wait=True)
                print(f"Marked row {idx} as good quality.")
            _rebuild_spectrum_options()
            _update_plot_for_selection()
        except Exception:
            pass
        try:
            _refresh_mark_buttons()
        except Exception:
            pass

    # Wire events
    material_dd.observe(_rebuild_spectrum_options, names="value")
    conditions_dd.observe(_rebuild_spectrum_options, names="value")
    spectrum_sel.observe(_update_plot_for_selection, names="value")
    include_bad_cb.observe(_rebuild_spectrum_options, names="value")
    save_spec_btn.on_click(_save_for_this_spectrum)
    save_mat_btn.on_click(_save_for_this_material)
    normalize_btn.on_click(_normalize_material)
    redo_btn.on_click(_redo)
    mark_bad_btn.on_click(_mark_bad)
    mark_good_btn.on_click(_mark_good)
    cancel_btn.on_click(_close)

    # Layout: controls on top, then plot, then info and messages, then buttons
    controls_row = widgets.HBox([material_dd, conditions_dd, include_bad_cb])
    spectrum_row = widgets.HBox([spectrum_sel])
    display(controls_row, spectrum_row, fig, info_out, msg_out, mark_row, btn_box)
    try:
        _refresh_mark_buttons()
    except Exception:
        pass
    return FTIR_DataFrame


def spectrum_normalization(
    FTIR_DataFrame,
    material,
):
    """
    Normalize baseline-corrected spectra.

    For each spectrum of the given material, finds the maximum value within the
    selected normalization range and divides the entire spectrum by that value,
    making the local maximum equal to 1.

    Parameters
    ----------
    FTIR_DataFrame : pd.DataFrame
        DataFrame containing spectra.
    material : str
        Material to normalize.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with normalized values written to 'Normalized and Corrected
        Data'.
    """
    if FTIR_DataFrame is None:
        raise ValueError("FTIR_DataFrame must be loaded in.")
    if material is None or str(material).strip() == "":
        raise ValueError("material is required.")

    source_column = "Baseline-Corrected Data"
    dest_column = "Normalized and Corrected Data"
    range_column = "Normalization Peak Wavenumber"

    subset = FTIR_DataFrame[FTIR_DataFrame["Material"] == material]
    try:
        subset = subset[_quality_good_mask(subset)]
    except Exception:
        pass
    if subset.empty:
        raise ValueError(f"No rows found for material '{material}'.")

    # Determine x-axis column
    x_axis_column = "X-Axis" if "X-Axis" in FTIR_DataFrame.columns else None
    if x_axis_column is None:
        raise ValueError(
            f"No 'X-Axis' column found in FTIR_DataFrame. Ensure the DataFrame is "
            f"loaded correctly."
        )
    # Validate required columns for normalization
    _require_columns(
        FTIR_DataFrame,
        ["Material", source_column, range_column, x_axis_column],
        context="FTIR_DataFrame (spectrum_normalization)",
    )

    # Ensure source/destination columns are object dtype (hold per-row lists)
    if source_column in FTIR_DataFrame.columns:
        try:
            FTIR_DataFrame[source_column] = FTIR_DataFrame[source_column].astype(object)
        except Exception:
            pass
    # Ensure destination column exists and is object dtype
    if dest_column not in FTIR_DataFrame.columns:
        FTIR_DataFrame[dest_column] = None
    try:
        FTIR_DataFrame[dest_column] = FTIR_DataFrame[dest_column].astype(object)
    except Exception:
        pass

    # Normalize each spectrum by its own max within the normalization window
    updated = 0
    skipped = 0
    errors = []
    for idx, row in subset.iterrows():
        norm_range = row.get(range_column, None)
        if norm_range is None:
            skipped += 1
            errors.append((idx, "Missing normalization range"))
            continue
        # Parse normalization range
        if isinstance(norm_range, str):
            try:
                norm_range = ast.literal_eval(norm_range)
            except Exception:
                skipped += 1
                errors.append(
                    (idx, f"Could not parse normalization range string: {norm_range!r}")
                )
                continue
        if not isinstance(norm_range, (list, tuple)) or len(norm_range) != 2:
            skipped += 1
            errors.append(
                (
                    idx,
                    f"Normalization range must be a list/tuple of length 2. "
                    f"Got: {type(norm_range).__name__}",
                )
            )
            continue
        try:
            lo, hi = float(norm_range[0]), float(norm_range[1])
        except Exception:
            skipped += 1
            errors.append(
                (idx, f"Normalization range values must be numeric. Got: {norm_range}")
            )
            continue

        x = row.get(x_axis_column, None)
        y = row.get(source_column, None)
        if x is None or y is None:
            skipped += 1
            errors.append((idx, "Missing x-axis or baseline-corrected data"))
            continue
        if isinstance(x, str):
            try:
                x = ast.literal_eval(x)
            except Exception:
                skipped += 1
                errors.append((idx, "Could not parse x-axis string to sequence"))
                continue
        if isinstance(y, str):
            try:
                y = ast.literal_eval(y)
            except Exception:
                skipped += 1
                errors.append(
                    (idx, "Could not parse baseline-corrected data string to sequence")
                )
                continue
        try:
            x_arr = np.asarray(x, dtype=float)
            y_arr = np.asarray(y, dtype=float)
        except Exception:
            skipped += 1
            errors.append((idx, "x or y could not be coerced to numeric arrays"))
            continue
        if x_arr.shape[0] != y_arr.shape[0] or x_arr.ndim != 1:
            skipped += 1
            errors.append(
                (
                    idx,
                    f"Shape mismatch or non-1D arrays: x.shape={x_arr.shape}, "
                    f"y.shape={y_arr.shape}",
                )
            )
            continue

        lo_, hi_ = min(lo, hi), max(lo, hi)
        mask = (x_arr >= lo_) & (x_arr <= hi_)
        if not np.any(mask):
            skipped += 1
            errors.append(
                (idx, f"No x points within normalization range [{lo_}, {hi_}]")
            )
            continue

        local_peak = np.nanmax(y_arr[mask])
        if not np.isfinite(local_peak) or local_peak <= 0:
            skipped += 1
            errors.append(
                (idx, f"Local peak is not finite/positive within range: {local_peak}")
            )
            continue

        # Scale this spectrum so its max in the range becomes 1 and write to DataFrame
        y_scaled = (y_arr / local_peak).astype(float).tolist()
        FTIR_DataFrame.at[idx, dest_column] = y_scaled
        updated += 1

    print(
        f"Normalized material '{material}': updated {updated} spectra; skipped "
        f"{skipped} (missing/invalid range or data). "
        f"Each spectrum scaled by its own peak within the selected range."
    )
    if updated == 0 and skipped > 0 and errors:
        examples = "; ".join([f"row {i}: {reason}" for i, reason in errors[:5]])
        more = "" if len(errors) <= 5 else f" (and {len(errors) - 5} more)"
        raise ValueError(
            f"Normalization failed for material '{material}': no spectra updated. "
            f"Examples: {examples}{more}"
        )
    return FTIR_DataFrame


def find_peak_info(FTIR_DataFrame, filepath=None):
    """
    Interactive peak finder for normalized and baseline-corrected spectra.

    - Uses scipy.signal.find_peaks on 'Normalized and Corrected Data'.
    - Checkboxes enable up to 3 independent X-range sliders; peaks are found in the
    union of enabled ranges.
    - Displays a live-updating plot with user-adjustable parameters.
    - Saves results (lists) to 'Peak Wavenumbers' and 'Peak Absorbances' columns.

    Parameters
    ----------
    FTIR_DataFrame : pd.DataFrame
        The DataFrame containing FTIR spectral data.
    filepath : str | None
        Optional. If provided, the UI will be pre-filtered to this file (by matching
        full path or just filename). Otherwise, all spectra will be available and you
        can select Material/Conditions/Spectrum via dropdowns.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame (in-place modifications also applied).
    """
    if FTIR_DataFrame is None or len(FTIR_DataFrame) == 0:
        raise ValueError("FTIR_DataFrame must be loaded and non-empty.")

    # Start with the full DataFrame and optionally pre-filter by filepath
    if filepath is not None:
        # Match by full path (File Location + File Name) or by just filename
        if os.path.sep in str(filepath):
            file_dir, file_name = os.path.split(str(filepath))
            filtered = FTIR_DataFrame[
                (FTIR_DataFrame.get("File Location", "") == file_dir)
                & (FTIR_DataFrame.get("File Name", "") == file_name)
            ]
        else:
            filtered = FTIR_DataFrame[
                FTIR_DataFrame.get("File Name", "") == str(filepath)
            ]
        # Do not exclude bad rows here; a UI checkbox will control inclusion below
        if filtered.empty:
            raise ValueError(
                f"No rows found for filepath '{filepath}'. Ensure 'File Location' and 'File Name' are populated."
            )
    else:
        filtered = FTIR_DataFrame.copy()

    # Ensure destination columns exist and are object dtype
    for col in ("Peak Wavenumbers", "Peak Absorbances"):
        if col not in FTIR_DataFrame.columns:
            FTIR_DataFrame[col] = None
        try:
            FTIR_DataFrame[col] = FTIR_DataFrame[col].astype(object)
        except Exception:
            pass

    def _parse_seq(val):
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except Exception:
                return None
        return val

    # Determine the conditions column
    cond_col = (
        "Conditions"
        if "Conditions" in filtered.columns
        else ("Condition" if "Condition" in filtered.columns else None)
    )

    # Material and Conditions dropdowns (Conditions excludes 'unexposed')
    try:
        unique_materials = (
            sorted(
                {
                    str(v)
                    for v in filtered.get("Material", pd.Series([], dtype=object))
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                }
            )
            if "Material" in filtered.columns
            else []
        )
    except Exception:
        unique_materials = []
    cond_series = (
        filtered[cond_col]
        if cond_col and (cond_col in filtered.columns)
        else pd.Series([], dtype=object)
    )
    _all_conditions = [
        str(v) for v in cond_series.dropna().astype(str).unique().tolist()
    ]
    unique_conditions = sorted(
        [c for c in _all_conditions if c.strip().lower() != "unexposed"]
    )

    material_dd = widgets.Dropdown(
        options=["any"] + unique_materials,
        value="any",
        description="Material",
        layout=widgets.Layout(width="40%"),
    )
    conditions_dd = widgets.Dropdown(
        options=["any"] + unique_conditions,
        value="any",
        description="Conditions",
        layout=widgets.Layout(width="40%"),
    )
    # Seed Material/Conditions from session defaults if available
    try:
        _sess_defaults = _get_session_defaults()
        _sess_mat = str(_sess_defaults.get("material", "any"))
        _sess_cond = str(_sess_defaults.get("conditions", "any"))
        if _sess_mat in list(material_dd.options):
            material_dd.value = _sess_mat
        if _sess_cond in list(conditions_dd.options):
            conditions_dd.value = _sess_cond
    except Exception:
        pass
    # Apply session defaults if present
    try:
        _sess = _get_session_defaults()
        if _sess.get("material") in material_dd.options:
            material_dd.value = _sess.get("material")
        if _sess.get("conditions") in conditions_dd.options:
            conditions_dd.value = _sess.get("conditions")
    except Exception:
        pass

    # Apply session defaults if present before building options
    try:
        _sess = _get_session_defaults()
        if _sess.get("material") in material_dd.options:
            material_dd.value = _sess.get("material")
        if _sess.get("conditions") in conditions_dd.options:
            conditions_dd.value = _sess.get("conditions")
    except Exception:
        pass

    # Build spectrum options using current filters; include 'unexposed' spectra always
    def _current_filtered_df():
        df = filtered.copy()
        # Optionally exclude rows marked as bad quality
        try:
            if not include_bad_cb.value:
                df = df[_quality_good_mask(df)]
        except Exception:
            pass
        # Filter by material
        if material_dd.value != "any":
            try:
                df = df[df.get("Material", "").astype(str) == str(material_dd.value)]
            except Exception:
                df = df[df.get("Material", "") == material_dd.value]
        # Filter by conditions but always include 'unexposed'
        if cond_col and conditions_dd.value != "any":
            cond_vals = df.get(cond_col, pd.Series([None] * len(df))).astype(str)
            mask = (cond_vals == str(conditions_dd.value)) | (
                cond_vals.str.strip().str.lower() == "unexposed"
            )
            df = df[mask]
        return df

    def _build_options():
        df = _current_filtered_df()
        # Only include rows with normalized data available
        try:
            df = df[df["Normalized and Corrected Data"].notna()]
        except Exception:
            # If column missing or not Series, this will result in empty options
            df = df.iloc[0:0]
        # Sort by Time ascending if present
        try:
            if "Time" in df.columns:
                df = df.copy()
                df["_sort_time"] = pd.to_numeric(df["Time"], errors="coerce").fillna(
                    float("inf")
                )
                df = df.sort_values(by=["_sort_time"], kind="mergesort")
        except Exception:
            pass
        opts = []
        for idx, r in df.iterrows():
            label = (
                f"{r.get('Material','')} | {r.get('Conditions', r.get('Condition',''))}"
                f" | T={r.get('Time','')} | {r.get('File Name','')}"
            )
            opts.append((label, idx))
        return opts

    options = _build_options()
    if not options:
        raise ValueError(
            "No spectra available to analyze. Ensure 'Normalized and Corrected Data' is populated (run baseline and normalization)."
        )

    # Seed from first spectrum, prefer session 'time' if available
    try:
        _sess = _get_session_defaults()
        saved_time = _sess.get("time", "any")
        # find first option whose Time matches saved_time
        def _matches_time(idx):
            try:
                t = FTIR_DataFrame.loc[idx].get("Time")
                if isinstance(saved_time, str) and saved_time.strip().lower() == "any":
                    return False
                try:
                    return int(t) == int(saved_time)
                except Exception:
                    return str(t) == str(saved_time)
            except Exception:
                return False
        match = next((idx for (_lab, idx) in options if _matches_time(idx)), None)
        first_idx = match if match is not None else options[0][1]
    except Exception:
        # Fallback to the first available option
        first_idx = options[0][1]
    x0 = _parse_seq(FTIR_DataFrame.loc[first_idx].get("X-Axis"))
    y0 = _parse_seq(FTIR_DataFrame.loc[first_idx].get("Normalized and Corrected Data"))
    if x0 is None or y0 is None:
        raise ValueError(
            "Selected spectrum is missing 'X-Axis' or 'Normalized and Corrected Data'."
        )
    x0 = np.asarray(x0, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    xmin, xmax = (float(np.nanmin(x0)), float(np.nanmax(x0)))

    # Build spectrum options using current filters; include 'unexposed' spectra always
    def _current_filtered_df():
        df = filtered.copy()
        # Optionally exclude rows marked as bad quality
        try:
            if not include_bad_cb.value:
                df = df[_quality_good_mask(df)]
        except Exception:
            pass
        # Filter by material
        if material_dd.value != "any":
            try:
                df = df[df.get("Material", "").astype(str) == str(material_dd.value)]
            except Exception:
                df = df[df.get("Material", "") == material_dd.value]
        # Filter by conditions but always include 'unexposed'
        if cond_col and conditions_dd.value != "any":
            cond_vals = df.get(cond_col, pd.Series([None] * len(df))).astype(str)
            mask = (cond_vals == str(conditions_dd.value)) | (
                cond_vals.str.strip().str.lower() == "unexposed"
            )
            df = df[mask]
        return df

    def _build_options():
        df = _current_filtered_df()
        # Only include rows with normalized data available
        try:
            df = df[df["Normalized and Corrected Data"].notna()]
        except Exception:
            df = df[
                (
                    df.get("Normalized and Corrected Data", None).notna()
                    if hasattr(df.get("Normalized and Corrected Data", None), "notna")
                    else []
                )
            ]
        # Sort by Time ascending if present
        try:
            if "Time" in df.columns:
                df = df.copy()
                df["_sort_time"] = pd.to_numeric(df["Time"], errors="coerce").fillna(
                    float("inf")
                )
                df = df.sort_values(by=["_sort_time"], kind="mergesort")
        except Exception:
            pass
        opts = []
        for idx, r in df.iterrows():
            label = (
                f"{r.get('Material','')} | {r.get('Conditions', r.get('Condition',''))}"
                f" | T={r.get('Time','')} | {r.get('File Name','')}"
            )
            opts.append((label, idx))
        return opts

    options = _build_options()
    if not options:
        raise ValueError("No spectra available after filtering.")

    # Seed from first spectrum; prefer session 'time' match if available
    try:
        _sess = _get_session_defaults()
        saved_time = _sess.get("time", "any")
        def _time_matches(idx):
            try:
                tval = FTIR_DataFrame.loc[idx].get("Time")
                if isinstance(saved_time, str) and saved_time.strip().lower() == "any":
                    return False
                try:
                    return int(tval) == int(saved_time)
                except Exception:
                    return str(tval) == str(saved_time)
            except Exception:
                return False
        _match_idx = next((idx for (_lab, idx) in options if _time_matches(idx)), None)
        first_idx = _match_idx if _match_idx is not None else options[0][1]
    except Exception:
        first_idx = options[0][1]
    x0 = _parse_seq(FTIR_DataFrame.loc[first_idx].get("X-Axis"))
    y0 = _parse_seq(FTIR_DataFrame.loc[first_idx].get("Normalized and Corrected Data"))
    if x0 is None or y0 is None:
        raise ValueError(
            "Selected spectrum is missing 'X-Axis' or 'Normalized and Corrected Data'."
        )
    x0 = np.asarray(x0, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    xmin, xmax = (float(np.nanmin(x0)), float(np.nanmax(x0)))

    # Widgets
    spectrum_sel = widgets.Dropdown(
        options=options,
        value=first_idx,
        description="Spectrum",
        layout=widgets.Layout(width="70%"),
    )

    # Persist changes in dropdowns to session state
    def _persist_material_cond(_=None):
        try:
            _set_session_selection(material=material_dd.value, conditions=conditions_dd.value)
        except Exception:
            pass
    material_dd.observe(_persist_material_cond, names="value")
    conditions_dd.observe(_persist_material_cond, names="value")
    # Up to three optional X-range selectors, each gated by a checkbox
    step_val = (xmax - xmin) / 1000 or 1.0
    use_r1 = widgets.Checkbox(value=True, description="Use range 1")
    x_range1 = widgets.FloatRangeSlider(
        value=[xmin, xmax],
        min=xmin,
        max=xmax,
        step=step_val,
        description="X-range 1",
        continuous_update=False,
        readout_format=".1f",
        layout=widgets.Layout(width="90%"),
        disabled=not use_r1.value,
    )
    use_r2 = widgets.Checkbox(value=False, description="Use range 2")
    x_range2 = widgets.FloatRangeSlider(
        value=[xmin, xmax],
        min=xmin,
        max=xmax,
        step=step_val,
        description="X-range 2",
        continuous_update=False,
        readout_format=".1f",
        layout=widgets.Layout(width="90%"),
        disabled=not use_r2.value,
    )
    use_r3 = widgets.Checkbox(value=False, description="Use range 3")
    x_range3 = widgets.FloatRangeSlider(
        value=[xmin, xmax],
        min=xmin,
        max=xmax,
        step=step_val,
        description="X-range 3",
        continuous_update=False,
        readout_format=".1f",
        layout=widgets.Layout(width="90%"),
        disabled=not use_r3.value,
    )
    prominence = widgets.FloatSlider(
        value=0.05,
        min=0.0,
        max=1.0,
        step=0.005,
        description="Prominence",
        readout_format=".3f",
        continuous_update=False,
        style={"description_width": "auto"},
    )
    min_height = widgets.FloatSlider(
        value=0.0,
        min=0.01,
        max=1.0,
        step=0.01,
        description="Min height",
        readout_format=".2f",
        continuous_update=False,
        style={"description_width": "auto"},
    )
    distance = widgets.IntSlider(
        value=5,
        min=1,
        max=250,
        step=1,
        description="Min separation",
        continuous_update=False,
        style={"description_width": "auto"},
    )
    width = widgets.IntSlider(
        value=1,
        min=1,
        max=50,
        step=1,
        description="Min width",
        continuous_update=False,
        style={"description_width": "auto"},
    )
    max_peaks = widgets.IntSlider(
        value=10,
        min=1,
        max=25,
        step=1,
        description="Max peaks",
        continuous_update=False,
        style={"description_width": "auto"},
    )

    save_file_btn = widgets.Button(description="Save for file", button_style="success")
    save_all_btn = widgets.Button(description="Save for filtered", button_style="info")
    mark_bad_btn = widgets.Button(description="Mark as bad", button_style="danger")
    mark_good_btn = widgets.Button(description="Mark as good", button_style="success")
    include_bad_cb = widgets.Checkbox(value=False, description="Include bad spectra")
    close_btn = widgets.Button(description="Close", button_style="danger")
    msg_out = widgets.Output()

    def _refresh_mark_buttons():
        try:
            idx = spectrum_sel.value
            qcol = _quality_column_name(FTIR_DataFrame)
            status = None
            try:
                if idx is not None:
                    status = FTIR_DataFrame.at[idx, qcol]
            except Exception:
                status = None
            is_bad = str(status).strip().lower() == "bad"
            mark_bad_btn.layout.display = "none" if is_bad else ""
            mark_good_btn.layout.display = "" if is_bad else "none"
        except Exception:
            pass

    # Plotly figure
    fig = go.FigureWidget()
    fig.add_scatter(
        x=x0.tolist(), y=y0.tolist(), mode="lines", name="Normalized and Corrected"
    )
    fig.add_scatter(
        x=[],
        y=[],
        mode="markers",
        name="Peaks",
        marker=dict(color="red", size=9, symbol="x"),
    )
    fig.update_layout(
        title="Peak Selection (live)",
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="Absorbance (AU)",
    )

    def _get_xy(row_idx):
        r = FTIR_DataFrame.loc[row_idx]
        x = _parse_seq(r.get("X-Axis"))
        y = _parse_seq(r.get("Normalized and Corrected Data"))
        if x is None or y is None:
            return None, None
        try:
            x_arr = np.asarray(x, dtype=float)
            y_arr = np.asarray(y, dtype=float)
        except Exception:
            return None, None
        if x_arr.ndim != 1 or y_arr.ndim != 1 or x_arr.shape[0] != y_arr.shape[0]:
            return None, None
        return x_arr, y_arr

    def _compute_peaks_for_ranges(x_arr, y_arr, ranges):
        # Build a combined mask for all enabled ranges
        if not ranges:
            return np.array([], dtype=int), np.array([], dtype=float)
        mask = np.zeros(x_arr.shape[0], dtype=bool)
        for x_min, x_max in ranges:
            if x_min is None or x_max is None:
                continue
            lo, hi = (float(min(x_min, x_max)), float(max(x_min, x_max)))
            mask |= (x_arr >= lo) & (x_arr <= hi)
        if not np.any(mask):
            return np.array([], dtype=int), np.array([], dtype=float)
        y_sub = y_arr[mask]
        idx_sub = np.where(mask)[0]
        kwargs = {
            "prominence": (
                float(prominence.value) if prominence.value is not None else None
            ),
            "distance": int(distance.value) if distance.value is not None else None,
            "width": int(width.value) if width.value is not None else None,
        }
        if min_height.value and float(min_height.value) > 0:
            kwargs["height"] = float(min_height.value)
        peaks_local, _props = find_peaks(
            y_sub, **{k: v for k, v in kwargs.items() if v is not None}
        )
        if peaks_local.size == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        peaks_global = idx_sub[peaks_local]
        # limit to top-N by height if requested
        if (
            max_peaks.value
            and int(max_peaks.value) > 0
            and peaks_global.size > int(max_peaks.value)
        ):
            heights = y_arr[peaks_global]
            order = np.argsort(heights)[::-1][: int(max_peaks.value)]
            peaks_global = peaks_global[order]
        return peaks_global, y_arr[peaks_global]

    def _current_ranges():
        rs = []
        if use_r1.value:
            rs.append((x_range1.value[0], x_range1.value[1]))
        if use_r2.value:
            rs.append((x_range2.value[0], x_range2.value[1]))
        if use_r3.value:
            rs.append((x_range3.value[0], x_range3.value[1]))
        return rs

    def _update_plot(*args):
        idx = spectrum_sel.value
        x_arr, y_arr = _get_xy(idx)
        if x_arr is None:
            with msg_out:
                msg_out.clear_output()
                print("Selected spectrum missing or invalid normalized data.")
            return
        # Update traces
        with fig.batch_update():
            fig.data[0].x = x_arr.tolist()
            fig.data[0].y = y_arr.tolist()
        # Update bounds for each slider and enable/disable based on checkboxes
        x_min, x_max = float(np.nanmin(x_arr)), float(np.nanmax(x_arr))
        for cb, sl in ((use_r1, x_range1), (use_r2, x_range2), (use_r3, x_range3)):
            sl.min = x_min
            sl.max = x_max
            try:
                lo, hi = sl.value
            except Exception:
                lo, hi = x_min, x_max
            lo = max(x_min, min(lo, x_max))
            hi = max(lo, min(hi, x_max))
            sl.value = [lo, hi]
            sl.disabled = not cb.value
        # Peaks and shading across all enabled ranges
        ranges = _current_ranges()
        peaks_idx, peaks_y = _compute_peaks_for_ranges(x_arr, y_arr, ranges)
        with fig.batch_update():
            fig.data[1].x = x_arr[peaks_idx].tolist() if peaks_idx.size else []
            fig.data[1].y = peaks_y.tolist() if peaks_idx.size else []
            # Rebuild all range shapes deterministically for reliability with multiple
            # ranges
            y0_min = float(np.nanmin(y_arr))
            y0_max = float(np.nanmax(y_arr))
            shapes_list = []
            for i_r, rng in enumerate(ranges):
                if rng is None:
                    continue
                lo_i, hi_i = float(min(rng)), float(max(rng))
                shapes_list.append(
                    dict(
                        type="rect",
                        x0=lo_i,
                        x1=hi_i,
                        y0=y0_min,
                        y1=y0_max,
                        fillcolor="rgba(0,128,0,0.12)",
                        line=dict(width=0),
                        layer="below",
                        name=f"range_rect_{i_r}",
                    )
                )
            # Assign shapes in one shot to avoid inconsistent state when multiple are
            # active
            fig.layout.shapes = tuple(shapes_list)
        with msg_out:
            msg_out.clear_output()
            if not ranges:
                print("Enable at least one X-range to find peaks.")
            else:
                print(f"Peaks found: {len(peaks_idx)}")
        # Update mark buttons visibility for current selection
        _refresh_mark_buttons()

    def _save_for_file(b):
        idx = spectrum_sel.value
        x_arr, y_arr = _get_xy(idx)
        if x_arr is None:
            with msg_out:
                msg_out.clear_output()
                print("Cannot save: selected spectrum missing normalized data.")
            return
        ranges = _current_ranges()
        if not ranges:
            with msg_out:
                msg_out.clear_output()
                print("Please enable at least one X-range before saving.")
            return
        peaks_idx, peaks_y = _compute_peaks_for_ranges(x_arr, y_arr, ranges)
        FTIR_DataFrame.at[idx, "Peak Wavenumbers"] = (
            x_arr[peaks_idx].astype(float).tolist()
        )
        FTIR_DataFrame.at[idx, "Peak Absorbances"] = peaks_y.astype(float).tolist()
        with msg_out:
            msg_out.clear_output()
            print(
                f"Saved {len(peaks_idx)} peaks for file "
                f"'{FTIR_DataFrame.loc[idx, 'File Name']}'."
            )

    def _save_for_filtered(b):
        ranges = _current_ranges()
        if not ranges:
            with msg_out:
                msg_out.clear_output()
                print("Please enable at least one X-range before saving.")
            return
        updated, skipped = 0, 0
        for idx, _row in _current_filtered_df().iterrows():
            x_arr, y_arr = _get_xy(idx)
            if x_arr is None:
                skipped += 1
                continue
            peaks_idx, peaks_y = _compute_peaks_for_ranges(x_arr, y_arr, ranges)
            FTIR_DataFrame.at[idx, "Peak Wavenumbers"] = (
                x_arr[peaks_idx].astype(float).tolist()
            )
            FTIR_DataFrame.at[idx, "Peak Absorbances"] = peaks_y.astype(float).tolist()
            updated += 1
        with msg_out:
            msg_out.clear_output()
            print(
                f"Updated {updated} spectra; skipped {skipped} (missing/invalid data)."
            )

    def _mark_bad(_b=None):
        try:
            idx = spectrum_sel.value
            if idx is None:
                return
            qcol = _quality_column_name(FTIR_DataFrame)
            FTIR_DataFrame.at[idx, qcol] = "bad"
        except Exception:
            pass
        try:
            _refresh_mark_buttons()
        except Exception:
            pass
        # If excluding bad spectra, refresh options to hide newly marked row
        try:
            if not include_bad_cb.value:
                _on_filters_change()
        except Exception:
            pass

    def _mark_good(_b=None):
        try:
            idx = spectrum_sel.value
            if idx is None:
                return
            qcol = _quality_column_name(FTIR_DataFrame)
            FTIR_DataFrame.at[idx, qcol] = "good"
        except Exception:
            pass
        try:
            _refresh_mark_buttons()
        except Exception:
            pass

        _refresh_mark_buttons()

    def _close_ui(b):
        try:
            # Close dropdowns/filters first
            try:
                material_dd.close()
            except Exception:
                pass
            try:
                conditions_dd.close()
            except Exception:
                pass
            spectrum_sel.close()
            x_range1.close()
            x_range2.close()
            x_range3.close()
            use_r1.close()
            use_r2.close()
            use_r3.close()
            prominence.close()
            min_height.close()
            distance.close()
            width.close()
            max_peaks.close()
            save_file_btn.close()
            save_all_btn.close()
            try:
                mark_bad_btn.close()
            except Exception:
                pass
            try:
                mark_good_btn.close()
            except Exception:
                pass
            try:
                include_bad_cb.close()
            except Exception:
                pass
            close_btn.close()
            # Leave msg_out displayed so the last messages remain visible
            fig.close()
            try:
                # Close container widgets if available
                filters_row.close()
            except Exception:
                pass
            try:
                ui.close()
            except Exception:
                pass
        except Exception:
            pass

    # Wire events
    def _on_filters_change(*_):
        new_opts = _build_options()
        if not new_opts:
            spectrum_sel.options = [("<no spectra>", None)]
            try:
                spectrum_sel.value = None
            except Exception:
                pass
            with msg_out:
                msg_out.clear_output()
                print("No spectra match the current filters.")
            return
        spectrum_sel.options = new_opts
        if spectrum_sel.value not in [v for _, v in new_opts]:
            spectrum_sel.value = new_opts[0][1]
        _update_plot()
        _refresh_mark_buttons()

    spectrum_sel.observe(_update_plot, names="value")
    # Keep mark buttons visibility in sync with selection
    spectrum_sel.observe(lambda *_: _refresh_mark_buttons(), names="value")
    material_dd.observe(_on_filters_change, names="value")
    conditions_dd.observe(_on_filters_change, names="value")
    for w in (x_range1, x_range2, x_range3, use_r1, use_r2, use_r3):
        w.observe(_update_plot, names="value")
    prominence.observe(_update_plot, names="value")
    min_height.observe(_update_plot, names="value")
    distance.observe(_update_plot, names="value")
    width.observe(_update_plot, names="value")
    max_peaks.observe(_update_plot, names="value")
    save_file_btn.on_click(_save_for_file)
    save_all_btn.on_click(_save_for_filtered)
    mark_bad_btn.on_click(_mark_bad)
    mark_good_btn.on_click(_mark_good)
    include_bad_cb.observe(_on_filters_change, names="value")
    close_btn.on_click(_close_ui)

    controls_row1 = widgets.HBox([spectrum_sel])
    controls_row2 = widgets.HBox([use_r1, x_range1])
    controls_row3 = widgets.HBox([use_r2, x_range2])
    controls_row4 = widgets.HBox([use_r3, x_range3])
    controls_row5 = widgets.HBox([prominence, min_height, distance])
    mark_row = widgets.HBox([mark_bad_btn, mark_good_btn])
    # Keep sliders separate from buttons per requirements
    controls_row6 = widgets.HBox([width, max_peaks])
    buttons_row = widgets.HBox([save_file_btn, save_all_btn, close_btn])
    # Prepend filter controls row
    filters_row = widgets.HBox([material_dd, conditions_dd, include_bad_cb])
    ui = widgets.VBox(
        [
            filters_row,
            controls_row1,
            controls_row2,
            controls_row3,
            controls_row4,
            controls_row5,
            controls_row6,
            buttons_row,
            mark_row,
        ]
    )

    display(ui, fig, msg_out)
    _update_plot()
    _refresh_mark_buttons()

    return FTIR_DataFrame


def peak_deconvolution(FTIR_DataFrame, filepath=None):
    """
    Interactively deconvolute found peaks for area analysis.

    Takes the peak info from find_peak_info and utilizes a Pseudo-Voigt model to
    approximately model the peak components as a linear combination of Gaussian and
    Lorentzian distributions. Allows for live changing of the Gaussian-Lorentzian
    fraction parameter for each peak.

    Parameters
    ----------
    FTIR_DataFrame : pd.DataFrame
        The DataFrame containing FTIR spectral data.
    filepath : str | None
        Specific file path to filter by (exact match). If provided, limits the list to
        that file.

    Returns
    -------
    None (update later for Json filling)
    """

    try:
        _lmfit_models = importlib.import_module("lmfit.models")
        PseudoVoigtModel = getattr(_lmfit_models, "PseudoVoigtModel")
    except Exception as e:
        raise ImportError(
            "lmfit is required for peak_deconvolution. Please install it (e.g., pip "
            "install lmfit)."
        ) from e

    if FTIR_DataFrame is None or len(FTIR_DataFrame) == 0:
        raise ValueError("FTIR_DataFrame must be loaded and non-empty.")

    # Initial selection
    if filepath is not None:
        # Match by full path (File Location + File Name) or by just filename
        if os.path.sep in str(filepath):
            file_dir, file_name = os.path.split(str(filepath))
            filtered = FTIR_DataFrame[
                (FTIR_DataFrame.get("File Location", "") == file_dir)
                & (FTIR_DataFrame.get("File Name", "") == file_name)
            ]
        else:
            filtered = FTIR_DataFrame[
                FTIR_DataFrame.get("File Name", "") == str(filepath)
            ]
        # Exclude bad-quality rows
        try:
            filtered = filtered[_quality_good_mask(filtered)]
        except Exception:
            pass
        if filtered.empty:
            raise ValueError(f"No rows found for filepath '{filepath}'.")
    else:
        # Start with full DataFrame; Material/Conditions dropdowns will constrain
        filtered = FTIR_DataFrame.copy()
        try:
            filtered = filtered[_quality_good_mask(filtered)]
        except Exception:
            pass

    # Ensure destination column exists for saving results
    results_col = "Deconvolution Results"
    if results_col not in FTIR_DataFrame.columns:
        FTIR_DataFrame[results_col] = None
    try:
        FTIR_DataFrame[results_col] = FTIR_DataFrame[results_col].astype(object)
    except Exception:
        pass

    def _parse_seq(val):
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except Exception:
                return None
        return val

    # Build spectrum options (only include rows with normalized data available),
    # sorted by Time ascending so earliest (lowest) time appears first in dropdown.
    # Include-bad toggle for UI
    include_bad_cb = widgets.Checkbox(value=False, description="Include bad spectra")
    try:
        filtered_sorted = filtered.copy()
        try:
            if not include_bad_cb.value:
                filtered_sorted = filtered_sorted[_quality_good_mask(filtered_sorted)]
        except Exception:
            pass
        if "Time" in filtered_sorted.columns:
            filtered_sorted["_sort_time"] = pd.to_numeric(
                filtered_sorted["Time"], errors="coerce"
            )
            filtered_sorted["_sort_time"] = filtered_sorted["_sort_time"].fillna(
                float("inf")
            )
            filtered_sorted = filtered_sorted.sort_values(
                by=["_sort_time"], kind="mergesort"
            )
        else:
            filtered_sorted = filtered
    except Exception:
        filtered_sorted = filtered
    options = []
    for idx, r in filtered_sorted.iterrows():
        try:
            norm_val = r.get("Normalized and Corrected Data", None)
            # Skip rows without normalized data
            if pd.isna(norm_val) if "pd" in globals() else (norm_val is None):
                continue
        except Exception:
            if r.get("Normalized and Corrected Data", None) is None:
                continue
        label = (
            f"{r.get('Material','')} | {r.get('Conditions', r.get('Condition',''))}"
            f" | T={r.get('Time','')} | {r.get('File Name','')}"
        )
        options.append((label, idx))
    if not options:
        raise ValueError("No spectra available after filtering.")

    # Seed from first spectrum
    first_idx = options[0][1]
    x0 = _parse_seq(FTIR_DataFrame.loc[first_idx].get("X-Axis"))
    y0 = _parse_seq(FTIR_DataFrame.loc[first_idx].get("Normalized and Corrected Data"))
    if x0 is None or y0 is None:
        raise ValueError(
            "Selected spectrum is missing 'X-Axis' or 'Normalized and Corrected Data'."
        )
    x0 = np.asarray(x0, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    xmin, xmax = (float(np.nanmin(x0)), float(np.nanmax(x0)))

    # Widgets (spectrum and global fit controls; full x-range is always used)
    spectrum_sel = widgets.Dropdown(
        options=options,
        value=first_idx,
        description="Spectrum",
        layout=widgets.Layout(width="70%"),
    )
    # Build initial material/conditions dropdowns for interactive filtering
    try:
        norm_mask_init = filtered["Normalized and Corrected Data"].notna()
        filterable_df = filtered[norm_mask_init]
    except Exception:
        filterable_df = filtered
    unique_materials = (
        sorted(
            {
                str(v)
                for v in filterable_df.get("Material", pd.Series([], dtype=object))
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            }
        )
        if "Material" in filterable_df.columns
        else []
    )
    # conditions might be under 'Conditions' or 'Condition'
    cond_series = (
        filterable_df["Conditions"]
        if "Conditions" in filterable_df.columns
        else (
            filterable_df["Condition"]
            if "Condition" in filterable_df.columns
            else pd.Series([], dtype=object)
        )
    )
    # Build conditions list excluding 'unexposed' (case-insensitive)
    _all_conditions = [
        str(v) for v in cond_series.dropna().astype(str).unique().tolist()
    ]
    unique_conditions = sorted(
        [c for c in _all_conditions if c.strip().lower() != "unexposed"]
    )
    material_dd = widgets.Dropdown(
        options=["any"] + unique_materials,
        value="any",
        description="Material",
        layout=widgets.Layout(width="40%"),
    )
    conditions_dd = widgets.Dropdown(
        options=["any"] + unique_conditions,
        value="any",
        description="Conditions",
        layout=widgets.Layout(width="40%"),
    )
    center_window = widgets.FloatSlider(
        value=15.0,
        min=1.0,
        max=50.0,
        step=1.0,
        description="Center ±window (cm⁻¹)",
        continuous_update=False,
        style={"description_width": "auto"},
        readout_format=".0f",
    )
    # Fit range selector: only peaks within this range will be modeled and shown
    fit_range = widgets.FloatRangeSlider(
        value=[xmin, xmax],
        min=xmin,
        max=xmax,
        step=(xmax - xmin) / 1000 or 1.0,
        description="Fit X-range",
        continuous_update=False,
        readout_format=".1f",
        layout=widgets.Layout(width="90%"),
    )
    init_sigma = widgets.FloatSlider(
        value=10.0,
        min=1.0,
        max=100.0,
        step=0.5,
        description="Initial σ (cm⁻¹)",
        continuous_update=False,
        style={"description_width": "auto"},
        readout_format=".1f",
    )
    # Defaults for reset operations
    DEFAULT_ALPHA = 0.5
    DEFAULT_INCLUDE = True
    default_center_window_value = float(center_window.value)
    default_init_sigma_value = float(init_sigma.value)
    default_fit_range_value = (float(fit_range.value[0]), float(fit_range.value[1]))
    # Add peaks workflow controls
    # Fit button (manual trigger for fitting)
    fit_btn = widgets.Button(
        description="Fit",
        tooltip="Run fit using current settings",
        button_style="primary",
        layout=widgets.Layout(width="80px"),
    )
    add_peaks_btn = widgets.Button(
        description="Add peaks",
        tooltip="Click, then click on the plot to add one or more peaks",
        button_style="info",
        layout=widgets.Layout(width="110px"),
    )
    accept_new_peaks_btn = widgets.Button(
        description="Accept new peaks",
        button_style="success",
        layout=widgets.Layout(width="150px"),
    )
    # Iterative correction button to minimize reduced chi-square via coordinate descent
    iter_btn = widgets.Button(
        description="Optimize",
        button_style="info",
        layout=widgets.Layout(width="175px"),
        tooltip=(
            "Adjust α, center window, and initial σ to reduce reduced chi-square."
        ),
    )
    cancel_fit_btn = widgets.Button(
        description="Cancel Fit",
        button_style="danger",
        layout=widgets.Layout(width="120px"),
        tooltip="Interrupt the current fit or iterative correction",
    )
    redo_new_peaks_btn = widgets.Button(
        description="Redo new peaks",
        button_style="warning",
        layout=widgets.Layout(width="140px"),
    )
    cancel_new_peaks_btn = widgets.Button(
        description="Cancel peak addition",
        button_style="danger",
        layout=widgets.Layout(width="190px"),
    )
    save_btn = widgets.Button(description="Save for file", button_style="success")
    close_btn = widgets.Button(description="Close", button_style="danger")
    # Dedicated status label to avoid Output-widget buffering issues
    status_html = widgets.HTML(value="")
    optimize_status_html = widgets.HTML(value="")
    # Replace Output-based logging with a single HTML widget to avoid renderer
    # double-echo issues in some notebook front-ends.
    log_html = widgets.HTML(value="")
    # De-duplicate log messages: avoid echoing the same text twice in quick succession
    # Make this thread-safe since some logs originate from worker threads.
    last_msg_text = ""
    last_msg_ts = 0.0
    log_lock = threading.Lock()

    def _log_once(message: str, *, wait: bool = True, clear: bool = True):
        nonlocal last_msg_text, last_msg_ts
        # Normalize message to minimize false negatives due to stray whitespace
        msg_norm = str(message).rstrip()
        try:
            now = time.time()
        except Exception:
            now = 0.0
        # Use a small lock to serialize dedup checks across threads
        with log_lock:
            # Suppress duplicates of identical text within ~1.0s window
            if msg_norm == last_msg_text and (now - last_msg_ts) < 1.0:
                return
            # Update the HTML log widget directly to avoid Output buffering quirks.
            try:
                # Escape to prevent unintended HTML rendering; keep it simple text.
                safe = html.escape(msg_norm)
                log_html.value = f"<div style='font-family:monospace; white-space:pre-wrap;'>{safe}</div>"
            except Exception:
                # As a last resort, try printing to the notebook output
                try:
                    print(msg_norm)
                except Exception:
                    pass
            last_msg_text = msg_norm
            last_msg_ts = now

    # Reset buttons for globals and all
    reset_center_btn = widgets.Button(
        description="Reset",
        tooltip="Reset Center ±window to default",
        layout=widgets.Layout(width="80px"),
    )
    reset_sigma_btn = widgets.Button(
        description="Reset",
        tooltip="Reset Initial σ to default",
        layout=widgets.Layout(width="80px"),
    )
    reset_all_btn = widgets.Button(
        description="Reset all",
        button_style="warning",
        tooltip="Reset all sliders and selections to defaults",
        layout=widgets.Layout(width="120px"),
    )

    # Cancellation/interrupt support for long-running fits
    # Global cancellation token for long-running workflows (iteration).
    # Note: Do NOT replace this Event with a new instance elsewhere; it must remain
    # stable so a single set() call cancels the entire workflow.
    cancel_event = threading.Event()
    fit_thread = None
    iter_thread = None
    fit_cancel_token = None  # points to the active fit's cancel event, if any
    iterating_in_progress = False  # suppress per-fit status updates while iterating
    # Snapshot of reduced chi-square at the moment the user clicks Iterate
    iter_start_redchi = None
    # Final reduced chi-square once iteration completes and a flag to show a summary
    iter_final_redchi = None
    iter_summary_pending = False
    # Helper to marshal UI updates back onto the notebook's main IOLoop
    try:
        from tornado.ioloop import IOLoop as _IOLoop
    except Exception:
        _IOLoop = None
    # Capture the main thread's IOLoop now; using current() inside a worker thread
    # can create a new, non-running loop which would drop callbacks.
    try:
        _MAIN_IOLOOP = _IOLoop.current() if _IOLoop is not None else None
    except Exception:
        _MAIN_IOLOOP = None

    def _on_main_thread(fn, *args, **kwargs):
        try:
            if _MAIN_IOLOOP is not None:
                _MAIN_IOLOOP.add_callback(lambda: fn(*args, **kwargs))
                return
        except Exception:
            pass
        # Fallback: call directly (may work in some environments)
        try:
            fn(*args, **kwargs)
        except Exception:
            pass

    # Dynamic per-peak controls: include checkbox + alpha slider per peak
    alpha_sliders = []  # list[widgets.FloatSlider]
    include_checkboxes = []  # list[widgets.Checkbox]
    peak_controls_box = widgets.VBox([])

    # Persisted per-spectrum settings so switching spectra preserves choices
    per_spec_alpha = {}  # idx -> list[float]
    per_spec_include = {}  # idx -> list[bool]
    per_spec_globals = {}  # idx -> { 'center_window': float, 'init_sigma': float,
    # 'fit_range': (lo,hi) }
    # Track the last active (Material, Conditions) filter to scope the above caches
    current_filter_key = (None, None)
    # Shared, group-scoped manual/template peaks (x positions) that should carry over
    # when switching spectra within the same Material/Conditions selection.
    shared_peaks_x = None  # list[float] | None
    # Guard to suppress redundant fits during bulk programmatic updates
    bulk_update_in_progress = False
    # Lightweight reentrancy/debounce guards to avoid duplicate callbacks
    on_spectrum_change_inflight = False
    last_on_spectrum_change_ts = 0.0
    fit_update_inflight = False
    last_fit_update_ts = 0.0

    # Lightweight per-button click de-bounce to avoid double-firing handlers
    last_click_ts = {}

    def _recent_click(key: str, min_interval: float = 0.35) -> bool:
        try:
            now = time.time()
        except Exception:
            now = 0.0
        last = last_click_ts.get(key, 0.0)
        if (now - last) < min_interval:
            return True
        last_click_ts[key] = now
        return False

    def _snapshot_current_controls():
        """Persist current UI control values for the active spectrum."""
        try:
            idx = spectrum_sel.value
        except Exception:
            return
        try:
            per_spec_alpha[idx] = [float(s.value) for s in alpha_sliders]
        except Exception:
            per_spec_alpha[idx] = []
        try:
            per_spec_include[idx] = [bool(cb.value) for cb in include_checkboxes]
        except Exception:
            per_spec_include[idx] = []
        try:
            lo, hi = _current_fit_range()
            per_spec_globals[idx] = {
                "center_window": float(center_window.value),
                "init_sigma": float(init_sigma.value),
                "fit_range": (float(lo), float(hi)),
            }
        except Exception:
            pass

    def _on_control_change(*_):
        """Generic control-change handler (kept for backward compatibility)."""
        # Delegate to fit-range-specific handler by default
        _on_fit_range_change()

    def _on_include_toggle(*_):
        """Handle include checkbox toggles without rebuilding the per-peak UI."""
        try:
            if iterating_in_progress or bulk_update_in_progress:
                return
        except Exception:
            pass
        _snapshot_current_controls()
        try:
            status_html.value = (
                "<span style='color:#555;'>Include toggled. Click Fit to update.</span>"
            )
        except Exception:
            pass

    def _on_alpha_change(*_):
        """Handle alpha slider changes without rebuilding the per-peak UI."""
        try:
            if iterating_in_progress or bulk_update_in_progress:
                return
        except Exception:
            pass
        _snapshot_current_controls()
        try:
            status_html.value = (
                "<span style='color:#555;'>α changed. Click Fit to update.</span>"
            )
        except Exception:
            pass

    def _on_center_sigma_change(*_):
        """Handle center window/initial sigma changes without rebuilding per-peak UI."""
        try:
            if iterating_in_progress or bulk_update_in_progress:
                return
        except Exception:
            pass
        _snapshot_current_controls()
        try:
            status_html.value = (
                "<span style='color:#555;'>Parameters changed. Click Fit to "
                "update.</span>"
            )
        except Exception:
            pass

    def _on_fit_range_change(*_):
        """Only Fit X-range changes should rebuild the per-peak controls."""
        try:
            if iterating_in_progress or bulk_update_in_progress:
                return
        except Exception:
            pass
        # Snapshot first so states persist across rebuild
        _snapshot_current_controls()
        try:
            _update_fit_range_indicator()
        except Exception:
            pass
        try:
            idx = spectrum_sel.value
            _rebuild_alpha_sliders(idx)
        except Exception:
            pass
        try:
            status_html.value = (
                "<span style='color:#555;'>Fit range changed. Click Fit to "
                "update.</span>"
            )
        except Exception:
            pass

    # Track last reduced chi-square per spectrum to report refit deltas
    last_redchi_by_idx = {}
    # Store last successful fit result per spectrum for Save action
    last_result_by_idx = {}

    # Plot figure: data, fit, components (dynamic)
    fig = go.FigureWidget()
    fig.add_scatter(x=x0.tolist(), y=y0.tolist(), mode="lines", name="Data (Norm+Corr)")
    fig.add_scatter(
        x=[], y=[], mode="lines", name="Composite Fit", line=dict(color="red")
    )
    fig.update_layout(
        title="Peak Deconvolution (Pseudo-Voigt)",
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="Absorbance (AU)",
    )

    # --- Add-peaks mode state ---
    adding_mode = False
    new_peak_xs = []  # temporary stash of user-clicked x positions (snapped to grid)

    def _hide(w):
        try:
            w.layout.display = "none"
        except Exception:
            pass

    def _show(w):
        try:
            w.layout.display = ""
        except Exception:
            pass

    # Show/Hide and enablement for the Cancel Fit button based on active work
    def _update_cancel_fit_visibility():
        active = False
        try:
            active = (fit_thread is not None and fit_thread.is_alive()) or (
                iter_thread is not None and iter_thread.is_alive()
            )
        except Exception:
            pass
        try:
            cancel_fit_btn.disabled = not active
        except Exception:
            pass
        if active:
            _show(cancel_fit_btn)
        else:
            _hide(cancel_fit_btn)

    # hide action buttons initially
    _hide(accept_new_peaks_btn)
    _hide(redo_new_peaks_btn)
    _hide(cancel_new_peaks_btn)
    _hide(cancel_fit_btn)

    def _clear_add_peak_shapes():
        # Remove only our temporary marker shapes
        try:
            shapes = list(getattr(fig.layout, "shapes", ()))
            shapes = [
                s for s in shapes if getattr(s, "name", None) != "add_peak_marker"
            ]
            fig.layout.shapes = tuple(shapes)
        except Exception:
            try:
                # When shapes are plain dicts
                shapes = list(getattr(fig.layout, "shapes", ()))
                new_shapes = []
                for s in shapes:
                    try:
                        if s.get("name") != "add_peak_marker":
                            new_shapes.append(s)
                    except Exception:
                        new_shapes.append(s)
                fig.layout.shapes = tuple(new_shapes)
            except Exception:
                pass

    # Click handler for adding peaks when in adding mode
    def _on_data_click(trace, points, selector):
        # Only respond when adding mode is ON and a valid click occurred
        try:
            nonlocal adding_mode
            if not adding_mode:
                return
            if not points or not getattr(points, "xs", None):
                return
            x_clicked = float(points.xs[0])
        except Exception:
            return

        # Get current spectrum arrays
        idx = spectrum_sel.value
        x_arr, y_arr = _get_xy(idx)
        if x_arr is None or y_arr is None or x_arr.size == 0:
            _log_once("Cannot add peak: current spectrum has no normalized data.")
            return

        # Snap to nearest x
        try:
            nearest_i = int(np.argmin(np.abs(x_arr - x_clicked)))
        except Exception:
            _log_once("Could not determine nearest x for the clicked location.")
            return
        x_new = float(x_arr[nearest_i])
        # Enforce proximity using Center ±window as minimum separation (customizable)
        try:
            min_sep = float(center_window.value)
        except Exception:
            min_sep = 0.0
        # Too close to a previously selected (session) peak?
        for existing_x in new_peak_xs:
            if abs(existing_x - x_new) <= min_sep:
                try:
                    status_html.value = (
                        f"<span style='color:#a00;'>Rejected: {x_new:.3f} cm⁻¹ is "
                        f"within ±{min_sep:.2f} cm⁻¹ of another selected peak "
                        f"({existing_x:.3f}). "
                        f"Tip: reduce the Center ±window to fit peaks in small "
                        f"spaces.</span>"
                    )
                except Exception:
                    _log_once(
                        f"Rejected: {x_new:.3f} cm⁻¹ is within ±{min_sep:.2f} cm⁻¹ of "
                        f"another selected peak ({existing_x:.3f}). Tip: reduce the "
                        f"Center ±window to fit peaks in small spaces."
                    )
                return
        # Too close to an existing (committed) peak for this spectrum?
        xs_existing, _ys_existing = _get_peaks(idx)
        for xe in xs_existing:
            try:
                if abs(float(xe) - x_new) <= min_sep:
                    try:
                        status_html.value = (
                            f"<span style='color:#a00;'>Rejected: {x_new:.3f} cm⁻¹ is "
                            f"within ±{min_sep:.2f} cm⁻¹ of existing peak "
                            f"{float(xe):.3f}. "
                            f"Tip: reduce the Center ±window to fit peaks in small "
                            f"spaces.</span>"
                        )
                    except Exception:
                        _log_once(
                            f"Rejected: {x_new:.3f} cm⁻¹ is within ±{min_sep:.2f} cm⁻¹ "
                            f"of existing peak {float(xe):.3f}. Tip: reduce the Center "
                            f"±window to fit peaks in small spaces."
                        )
                    return
            except Exception:
                continue
        # Record and draw a vertical line marker
        new_peak_xs.append(x_new)
        y_min = float(np.nanmin(y_arr))
        y_max = float(np.nanmax(y_arr))
        try:
            fig.add_shape(
                dict(
                    type="line",
                    x0=x_new,
                    x1=x_new,
                    y0=y_min,
                    y1=y_max,
                    line=dict(color="#ff7f0e", dash="dot", width=1.5),
                    name="add_peak_marker",
                )
            )
        except Exception:
            pass
        _log_once(
            f"Selected new peak at x = {x_new:.3f} cm⁻¹. Click more points, or 'Accept "
            f"new peaks'."
        )

    # Always register the click handler; it checks the toggle state internally
    try:
        fig.data[0].on_click(_on_data_click)
    except Exception:
        pass

    def _get_xy(row_idx):
        r = FTIR_DataFrame.loc[row_idx]
        x = _parse_seq(r.get("X-Axis"))
        y = _parse_seq(r.get("Normalized and Corrected Data"))
        if x is None or y is None:
            return None, None
        try:
            x_arr = np.asarray(x, dtype=float)
            y_arr = np.asarray(y, dtype=float)
        except Exception:
            return None, None
        if x_arr.ndim != 1 or y_arr.ndim != 1 or x_arr.shape[0] != y_arr.shape[0]:
            return None, None
        return x_arr, y_arr

    def _get_peaks(row_idx):
        r = FTIR_DataFrame.loc[row_idx]
        xs = _parse_seq(r.get("Peak Wavenumbers"))
        ys = _parse_seq(r.get("Peak Absorbances"))
        if xs is None or ys is None:
            return [], []
        try:
            xs = list(xs)
            ys = list(ys)
        except Exception:
            return [], []
        if len(xs) != len(ys):
            return [], []
        # Sort peaks by wavenumber (ascending)
        try:
            pairs = sorted(zip(xs, ys), key=lambda t: float(t[0]))
            xs_sorted, ys_sorted = [list(t) for t in zip(*pairs)] if pairs else ([], [])
            return xs_sorted, ys_sorted
        except Exception:
            return xs, ys

    def _current_fit_range():
        try:
            lo, hi = fit_range.value
            return float(min(lo, hi)), float(max(lo, hi))
        except Exception:
            return xmin, xmax

    def _get_visible_peaks(row_idx):
        xs, ys = _get_peaks(row_idx)
        if not xs:
            return [], []
        lo, hi = _current_fit_range()
        xs_f = []
        ys_f = []
        for cx, cy in zip(xs, ys):
            try:
                cxv = float(cx)
            except Exception:
                continue
            if lo <= cxv <= hi:
                xs_f.append(cxv)
                try:
                    ys_f.append(float(cy))
                except Exception:
                    ys_f.append(float("nan"))
        return xs_f, ys_f

    # Visual indicator of the current Fit X-range on the plot
    def _update_fit_range_indicator():
        try:
            idx = spectrum_sel.value
        except Exception:
            return
        x_arr, y_arr = _get_xy(idx)
        if x_arr is None or y_arr is None or x_arr.size == 0:
            return
        try:
            lo, hi = _current_fit_range()
            y0_min = float(np.nanmin(y_arr))
            y0_max = float(np.nanmax(y_arr))
        except Exception:
            return
        rect = dict(
            type="rect",
            x0=float(min(lo, hi)),
            x1=float(max(lo, hi)),
            y0=y0_min,
            y1=y0_max,
            fillcolor="rgba(0,120,215,0.12)",  # subtle blue
            line=dict(color="rgba(0,120,215,0.6)", width=1),
            layer="below",
            name="fit_range_rect",
        )
        try:
            shapes = list(getattr(fig.layout, "shapes", ()))
            new_shapes = []
            for s in shapes:
                try:
                    # Skip any prior fit range shape; keep others
                    # (e.g., add_peak_marker)
                    nm = getattr(s, "name", None)
                except Exception:
                    nm = None
                if nm is None:
                    try:
                        nm = s.get("name")
                    except Exception:
                        nm = None
                if nm == "fit_range_rect":
                    continue
                new_shapes.append(s)
            new_shapes.append(rect)
            fig.layout.shapes = tuple(new_shapes)
        except Exception:
            # Best-effort; ignore if shapes unavailable
            pass

    def _rebuild_alpha_sliders(row_idx):
        nonlocal alpha_sliders, include_checkboxes
        xs, ys = _get_visible_peaks(row_idx)
        alpha_sliders = []
        include_checkboxes = []
        children = []
        if not xs:
            peak_controls_box.children = [
                widgets.HTML(
                    "<b>No peaks in selected range.</b> Adjust 'Fit X-range' or run "
                    "find_peak_info first."
                )
            ]
            return
        saved_alphas = per_spec_alpha.get(row_idx)
        saved_includes = per_spec_include.get(row_idx)
        for i, (cx, cy) in enumerate(zip(xs, ys)):
            label = widgets.Label(
                value=f"Peak {i+1} @ {cx:.1f} cm⁻¹",
                layout=widgets.Layout(width="220px"),
            )
            cb = widgets.Checkbox(
                value=(
                    (
                        saved_includes[i]
                        if saved_includes is not None and i < len(saved_includes)
                        else DEFAULT_INCLUDE
                    )
                ),
                description="Include",
                indent=False,
                layout=widgets.Layout(width="100px"),
            )
            s = widgets.FloatSlider(
                value=(
                    float(saved_alphas[i])
                    if saved_alphas is not None and i < len(saved_alphas)
                    else DEFAULT_ALPHA
                ),
                min=0.0,
                max=1.0,
                step=0.01,
                description="α",
                continuous_update=False,
                readout_format=".2f",
                style={"description_width": "auto"},
                layout=widgets.Layout(width="300px"),
            )
            # Per-slider reset button
            rb = widgets.Button(
                description="Reset",
                tooltip="Reset α to default (0.5)",
                layout=widgets.Layout(width="70px"),
            )

            def _make_reset_one(slider_ref):
                def _reset_one(_b=None):
                    nonlocal bulk_update_in_progress
                    bulk_update_in_progress = True
                    try:
                        slider_ref.value = DEFAULT_ALPHA
                    except Exception:
                        pass
                    _snapshot_current_controls()
                    bulk_update_in_progress = False
                    _fit_and_update_plot()

                return _reset_one

            rb.on_click(_make_reset_one(s))
            # Observe changes to snapshot + refit
            cb.observe(_on_include_toggle, names="value")
            s.observe(_on_alpha_change, names="value")
            include_checkboxes.append(cb)
            alpha_sliders.append(s)
            children.append(widgets.HBox([label, cb, s, rb]))
        peak_controls_box.children = children

    # --- Filtering helpers for material/conditions -> spectrum options ---
    def _row_condition_value(row):
        try:
            return row.get("Conditions", row.get("Condition", ""))
        except Exception:
            return ""

    def _rebuild_spectrum_options(*_):
        nonlocal current_filter_key, bulk_update_in_progress, shared_peaks_x
        # Build candidate set from initial 'filtered' and drop rows without normalized
        # data
        try:
            cand = filtered.copy()
            try:
                if not include_bad_cb.value:
                    cand = cand[_quality_good_mask(cand)]
            except Exception:
                pass
            cand = cand[cand["Normalized and Corrected Data"].notna()]
        except Exception:
            cand = filtered
        # Read current filter selections
        sel_mat = material_dd.value if hasattr(material_dd, "value") else "any"
        sel_cond = conditions_dd.value if hasattr(conditions_dd, "value") else "any"

        # If the (Material, Conditions) filter changed, clear any preserved per-spectrum
        # state so that edits/added peaks do not leak into unrelated spectra groups.
        new_filter_key = (sel_mat, sel_cond)
        if new_filter_key != current_filter_key:
            try:
                per_spec_alpha.clear()
                per_spec_include.clear()
                per_spec_globals.clear()
                last_redchi_by_idx.clear()
                last_result_by_idx.clear()
                # Reset shared peaks template when filter changes
                shared_peaks_x = None
                # Reset global sliders to defaults without triggering refits
                bulk_update_in_progress = True
                try:
                    center_window.value = default_center_window_value
                    init_sigma.value = default_init_sigma_value
                except Exception:
                    pass
                bulk_update_in_progress = False
            except Exception:
                pass
            current_filter_key = new_filter_key

        # Apply material filter

        if sel_mat != "any" and "Material" in cand.columns:
            cand = cand[cand["Material"].astype(str) == str(sel_mat)]
        # Apply conditions/condition filter; always include 'unexposed' rows as they
        # represent time-zero for every condition.
        if sel_cond != "any":

            def _cond_includes_unexposed(r):
                val = str(_row_condition_value(r))
                return (val == str(sel_cond)) or (val.strip().lower() == "unexposed")

            cand = cand[cand.apply(_cond_includes_unexposed, axis=1)]
        # Sort by Time ascending so earliest times appear first
        try:
            if "Time" in cand.columns:
                cand = cand.copy()
                cand["_sort_time"] = pd.to_numeric(cand["Time"], errors="coerce")
                cand["_sort_time"] = cand["_sort_time"].fillna(float("inf"))
                cand = cand.sort_values(by=["_sort_time"], kind="mergesort")
                try:
                    cand = cand.drop(columns=["_sort_time"])
                except Exception:
                    pass
        except Exception:
            pass
        # Build options
        new_options = []
        for idx2, r2 in cand.iterrows():
            label = (
                f"{r2.get('Material','')} | "
                f"{r2.get('Conditions', r2.get('Condition',''))}"
                f" | T={r2.get('Time','')} | {r2.get('File Name','')}"
            )
            new_options.append((label, idx2))
        # Update dropdown (avoid double-calling the value observer)
        prev_value = spectrum_sel.value
        try:
            spectrum_sel.unobserve(_on_spectrum_change, names="value")
        except Exception:
            pass
        spectrum_sel.options = new_options
        # Choose value: prefer session Time match; else keep previous if present; else first option
        valid_values = [v for (_lbl, v) in new_options]
        value_changed = False
        # Try to select a row matching the saved session 'time' (within current filters)
        preferred_idx = None
        try:
            _sess = _get_session_defaults()
            _sess_time = _sess.get("time", "any")
            if _sess_time is not None and str(_sess_time).strip().lower() != "any":
                # Compare numerically when possible, else fallback to string compare
                for (_lbl, _v) in new_options:
                    try:
                        _tval = FTIR_DataFrame.loc[_v].get("Time")
                        # Numeric equality if both castable
                        try:
                            if float(_tval) == float(_sess_time):
                                preferred_idx = _v
                                break
                        except Exception:
                            if str(_tval) == str(_sess_time):
                                preferred_idx = _v
                                break
                    except Exception:
                        continue
        except Exception:
            preferred_idx = None
        if preferred_idx in valid_values:
            spectrum_sel.value = preferred_idx
            value_changed = True
        elif prev_value in valid_values:
            # Only assign if the value actually needs to change
            if spectrum_sel.value != prev_value:
                spectrum_sel.value = prev_value
                value_changed = True
        elif valid_values:
            spectrum_sel.value = valid_values[0]
            value_changed = True
        else:
            # No spectra available, clear plot and controls
            with fig.batch_update():
                fig.data[0].x = []
                fig.data[0].y = []
                fig.data[1].x = []
                fig.data[1].y = []
            peak_controls_box.children = [
                widgets.HTML("<b>No spectra available after filtering.</b>")
            ]
            _log_once("No spectra available after filtering selections.")
            return
        # Re-attach observer and trigger a single downstream update
        try:
            spectrum_sel.observe(_on_spectrum_change, names="value")
        except Exception:
            pass
        # If we did not change the value (e.g., filters adjusted but selection stable),
        # explicitly trigger one update to refresh controls/plot exactly once.
        if not value_changed:
            _on_spectrum_change()

    def _fit_and_update_plot(*_):
        nonlocal fit_thread, cancel_event, fit_cancel_token, iterating_in_progress
        nonlocal fit_update_inflight, last_fit_update_ts
        # Debounce/guard: prevent rapid double invocation
        try:
            now_ts = time.time()
        except Exception:
            now_ts = 0.0
        if fit_update_inflight:
            return None
        if (now_ts - last_fit_update_ts) < 0.03:
            return None
        fit_update_inflight = True

        def _finish_fit_guard():
            nonlocal fit_update_inflight, last_fit_update_ts
            fit_update_inflight = False
            try:
                last_fit_update_ts = time.time()
            except Exception:
                last_fit_update_ts = now_ts

        idx = spectrum_sel.value
        x_arr, y_arr = _get_xy(idx)
        if x_arr is None:
            _log_once("Selected spectrum missing normalized data.")
            _finish_fit_guard()
            return None

        peaks_x, peaks_y = _get_visible_peaks(idx)
        if not peaks_x:
            _log_once("No peaks found in selected fit range.")
            with fig.batch_update():
                fig.data[0].x = x_arr.tolist()
                fig.data[0].y = y_arr.tolist()
                fig.data[1].x = []
                fig.data[1].y = []
                fig.layout.shapes = ()
                while len(fig.data) > 2:
                    fig.data = tuple(fig.data[:2])
            _finish_fit_guard()
            return None

        # Determine which peaks are included
        included = [i for i, cb in enumerate(include_checkboxes) if cb.value]
        if len(included) == 0:
            _log_once("No peaks selected in current range. Enable one or more to fit.")
            with fig.batch_update():
                fig.data[0].x = x_arr.tolist()
                fig.data[0].y = y_arr.tolist()
                fig.data[1].x = []
                fig.data[1].y = []
                while len(fig.data) > 2:
                    fig.data = tuple(fig.data[:2])
            _finish_fit_guard()
            return None

        # Prepare component traces count on main thread for consistent layout
        comp_traces_needed = len(included)
        with fig.batch_update():
            current_components = max(0, len(fig.data) - 2)
            if current_components > comp_traces_needed:
                fig.data = tuple(list(fig.data)[: 2 + comp_traces_needed])
            elif current_components < comp_traces_needed:
                for _k in range(comp_traces_needed - current_components):
                    fig.add_scatter(
                        x=[],
                        y=[],
                        mode="lines",
                        line=dict(dash="dot"),
                        name=f"Component {_k+1}",
                    )

        # Cancel any running fit and start a new one in the background
        try:
            if fit_thread is not None and fit_thread.is_alive():
                # Signal the currently running worker to stop (per-fit only)
                try:
                    if fit_cancel_token is not None:
                        fit_cancel_token.set()
                except Exception:
                    pass
        except Exception:
            pass
        # Create a fresh cancel token for this new worker and capture it locally
        local_cancel = threading.Event()
        # Keep the global cancel_event stable; only update the active per-fit token
        fit_cancel_token = local_cancel

        old_redchi = last_redchi_by_idx.get(idx, None)
        # Update status label immediately on main thread
        if not iterating_in_progress:
            try:
                status_html.value = (
                    "<span style='color:#555;'>Refitting...</span>"
                    if old_redchi is not None
                    else "<span style='color:#555;'>Fitting...</span>"
                )
            except Exception:
                _log_once("Refitting..." if old_redchi is not None else "Fitting...")

        def _worker(local_cancel_token=local_cancel):
            nonlocal fit_thread
            try:
                # Use only the selected Fit X-range for fitting to prevent components
                # going nearly flat when focusing on a small region.
                try:
                    lo, hi = _current_fit_range()
                except Exception:
                    lo, hi = float(np.nanmin(x_arr)), float(np.nanmax(x_arr))
                # Create mask robust to descending x-arrays
                lo_v = float(min(lo, hi))
                hi_v = float(max(lo, hi))
                try:
                    msk = (x_arr >= lo_v) & (x_arr <= hi_v)
                except Exception:
                    msk = np.ones_like(x_arr, dtype=bool)
                x_sub = x_arr[msk]
                y_sub = y_arr[msk]
                # If too few points in range, bail gracefully
                if x_sub.size < max(10, 3 * max(1, len(included))):
                    _on_main_thread(
                        lambda: _log_once(
                            "Selected Fit X-range has too few points for a stable fit. "
                            "Expand the range or include more peaks."
                        )
                    )
                    _finish_fit_guard()
                    return

                # Build composite model
                comp_model = None
                params = None
                for i in included:
                    cx = peaks_x[i]
                    m = PseudoVoigtModel(prefix=f"p{i}_")
                    p = m.make_params()
                    p[f"p{i}_center"].set(
                        value=float(cx),
                        min=float(cx) - center_window.value,
                        max=float(cx) + center_window.value,
                    )
                    p[f"p{i}_sigma"].set(
                        value=float(init_sigma.value), min=1e-3, max=1e3
                    )
                    alpha_val = (
                        float(alpha_sliders[i].value) if i < len(alpha_sliders) else 0.5
                    )
                    p[f"p{i}_fraction"].set(
                        value=alpha_val, min=0.0, max=1.0, vary=False
                    )
                    amp0 = abs(float(peaks_y[i])) * max(1.0, float(init_sigma.value))
                    p[f"p{i}_amplitude"].set(value=amp0, min=0.0)

                    if comp_model is None:
                        comp_model = m
                        params = p
                    else:
                        comp_model = comp_model + m
                        params.update(p)

                # iter_cb to allow cooperative cancellation
                def _iter_cb(params_, iter_, resid_, *args, **kws):
                    if local_cancel_token.is_set():
                        raise KeyboardInterrupt("Fit cancelled by user")

                result = comp_model.fit(y_sub, params, x=x_sub, iter_cb=_iter_cb)
                if local_cancel_token.is_set():
                    return
                # Evaluate results for full x-array for plotting
                y_fit = result.eval(x=x_arr)
                comps = result.eval_components(x=x_arr)

                # Persist last successful result for this spectrum
                try:
                    last_result_by_idx[idx] = result
                except Exception:
                    pass

                def _apply_results_on_ui():
                    nonlocal iter_summary_pending, iter_start_redchi, iter_final_redchi
                    # Always try to update the plot, but don't let plotting failures
                    # prevent status messages from updating.
                    plot_ok = True
                    try:
                        comp_traces_needed = len(included)
                        with fig.batch_update():
                            # Ensure we have exactly 2 + comp_traces_needed traces
                            current_components = max(0, len(fig.data) - 2)
                            if current_components > comp_traces_needed:
                                fig.data = tuple(
                                    list(fig.data)[: 2 + comp_traces_needed]
                                )
                            elif current_components < comp_traces_needed:
                                for _k in range(
                                    comp_traces_needed - current_components
                                ):
                                    fig.add_scatter(
                                        x=[],
                                        y=[],
                                        mode="lines",
                                        line=dict(dash="dot"),
                                        name=f"Component {_k+1}",
                                    )
                            # Update data and fit traces
                            fig.data[0].x = x_arr.tolist()
                            fig.data[0].y = y_arr.tolist()
                            fig.data[1].x = x_arr.tolist()
                            fig.data[1].y = (
                                y_fit.tolist()
                                if hasattr(y_fit, "tolist")
                                else list(y_fit)
                            )
                            # Update component traces safely
                            for comp_idx, i in enumerate(included):
                                key = f"p{i}_"
                                y_comp = comps.get(key, np.zeros_like(x_arr))
                                fig.data[2 + comp_idx].x = x_arr.tolist()
                                fig.data[2 + comp_idx].y = (
                                    y_comp.tolist()
                                    if hasattr(y_comp, "tolist")
                                    else list(y_comp)
                                )
                                # Ensure component names reflect the corresponding peak number
                                try:
                                    fig.data[2 + comp_idx].name = f"Peak {i+1}"
                                except Exception:
                                    pass
                    except Exception:
                        plot_ok = False
                    # Update message regardless of plot success
                    # Update status label text (suppressed during iterative correction)
                    new_redchi = getattr(result, "redchi", np.nan)
                    try:
                        if not iterating_in_progress:
                            # If an iteration just completed, prefer the pre- vs
                            # post-iteration summary
                            if (
                                iter_summary_pending
                                and iter_start_redchi is not None
                                and iter_final_redchi is not None
                            ):
                                try:
                                    old_str = f"{float(iter_start_redchi):.4g}"
                                except Exception:
                                    old_str = str(iter_start_redchi)
                                try:
                                    new_str = f"{float(iter_final_redchi):.4g}"
                                except Exception:
                                    new_str = str(iter_final_redchi)
                                status_html.value = (
                                    f"<span style='color:#000;'>Iterative correction "
                                    f"complete. Reduced chi-square: ("
                                    f"{old_str}) ---&gt; ({new_str})</span>"
                                )
                                # Clear the pending summary after showing it once
                                iter_summary_pending = False
                            else:
                                # Format values consistently, showing change when refitting
                                try:
                                    old_str = (
                                        f"{old_redchi:.4g}"
                                        if old_redchi is not None
                                        else None
                                    )
                                except Exception:
                                    old_str = (
                                        str(old_redchi)
                                        if old_redchi is not None
                                        else None
                                    )
                                try:
                                    new_str = f"{new_redchi:.4g}"
                                except Exception:
                                    new_str = str(new_redchi)
                                if old_redchi is not None:
                                    status_html.value = (
                                        f"<span style='color:#000;'>Refit complete. "
                                        f"Reduced chi-square: ("
                                        f"{old_str}) ---&gt; ({new_str})</span>"
                                    )
                                else:
                                    status_html.value = (
                                        f"<span style='color:#000;'>Fit complete. "
                                        f"Reduced chi-square: ("
                                        f"{new_str})</span>"
                                    )
                        # Persist the new redchi so subsequent runs are treated as
                        # refits
                        try:
                            if np.isfinite(new_redchi):
                                last_redchi_by_idx[idx] = float(new_redchi)
                        except Exception:
                            # If np.isfinite is not available or new_redchi is not
                            # numeric, store raw
                            try:
                                last_redchi_by_idx[idx] = float(new_redchi)
                            except Exception:
                                last_redchi_by_idx[idx] = new_redchi
                        if not plot_ok:
                            # Also echo a note in the log output
                            _log_once(
                                "(Note: Plot update partially failed; re-run to refresh components.)"
                            )
                    except Exception:
                        if not iterating_in_progress:
                            status_html.value = (
                                "<span style='color:#000;'>Fit complete.</span>"
                            )
                        try:
                            if np.isfinite(new_redchi):
                                last_redchi_by_idx[idx] = float(new_redchi)
                            else:
                                last_redchi_by_idx[idx] = new_redchi
                        except Exception:
                            last_redchi_by_idx[idx] = new_redchi
                    # Update Cancel Fit button visibility after this fit completes
                    try:
                        _update_cancel_fit_visibility()
                    except Exception:
                        pass

                _on_main_thread(_apply_results_on_ui)
                # Ensure the Cancel Fit button hides after the worker fully ends
                # by clearing the thread reference and scheduling a final UI update.
                try:
                    fit_thread = None
                except Exception:
                    pass
                try:
                    _on_main_thread(_update_cancel_fit_visibility)
                except Exception:
                    pass
            except KeyboardInterrupt:
                # Cancellation requested
                def _notify_cancel():
                    if not iterating_in_progress:
                        try:
                            status_html.value = (
                                "<span style='color:#a00;'>Fit cancelled.</span>"
                            )
                        except Exception:
                            _log_once("Fit cancelled.")
                    try:
                        _update_cancel_fit_visibility()
                    except Exception:
                        pass

                _on_main_thread(_notify_cancel)
                # Clear thread reference and trigger a final visibility update
                try:
                    fit_thread = None
                except Exception:
                    pass
                try:
                    _on_main_thread(_update_cancel_fit_visibility)
                except Exception:
                    pass
            except Exception as e:

                def _notify_error():
                    if not iterating_in_progress:
                        try:
                            status_html.value = (
                                f"<span style='color:#a00;'>Fit failed: {e}</span>"
                            )
                        except Exception:
                            _log_once(f"Fit failed: {e}")
                    try:
                        _update_cancel_fit_visibility()
                    except Exception:
                        pass

                _on_main_thread(_notify_error)
                # Clear thread reference and trigger a final visibility update
                try:
                    fit_thread = None
                except Exception:
                    pass
                try:
                    _on_main_thread(_update_cancel_fit_visibility)
                except Exception:
                    pass

        fit_thread = threading.Thread(target=_worker, daemon=True)
        fit_thread.start()
        try:
            _update_cancel_fit_visibility()
        except Exception:
            pass
        _finish_fit_guard()
        return None

    def _on_spectrum_change(*_):
        nonlocal bulk_update_in_progress, shared_peaks_x
        nonlocal on_spectrum_change_inflight, last_on_spectrum_change_ts
        # Debounce/guard: prevent rapid double invocation
        try:
            now_ts = time.time()
        except Exception:
            now_ts = 0.0
        if on_spectrum_change_inflight:
            return
        if (now_ts - last_on_spectrum_change_ts) < 0.03:
            return
        on_spectrum_change_inflight = True
        try:
            # Snapshot previous spectrum's controls before switching
            _snapshot_current_controls()
            idx = spectrum_sel.value
            # If a shared peaks template exists for this filter group, apply it to the
            # newly selected spectrum (recompute Y from its data), so user-added peaks
            # carry over when switching spectra within the same Material/Conditions.
            try:
                if shared_peaks_x is not None and isinstance(
                    shared_peaks_x, (list, tuple)
                ):
                    x_arr, y_arr = _get_xy(idx)
                    if x_arr is not None and y_arr is not None:
                        xs = [float(v) for v in shared_peaks_x]
                        ys = []
                        try:
                            for xv in xs:
                                i = int(np.argmin(np.abs(x_arr - xv)))
                                ys.append(float(y_arr[i]))
                        except Exception:
                            ys = [float("nan") for _ in xs]
                        FTIR_DataFrame.at[idx, "Peak Wavenumbers"] = xs
                        FTIR_DataFrame.at[idx, "Peak Absorbances"] = ys
                        # Peak definitions changed for this spectrum; clear per-peak UI
                        try:
                            per_spec_alpha.pop(idx, None)
                            per_spec_include.pop(idx, None)
                        except Exception:
                            pass
            except Exception:
                pass
            # Restore per-spectrum globals and per-peak controls without triggering many
            # fits
            bulk_update_in_progress = True
            try:
                g = per_spec_globals.get(idx)
                if g is not None:
                    try:
                        center_window.value = float(
                            g.get("center_window", center_window.value)
                        )
                    except Exception:
                        pass
                    try:
                        init_sigma.value = float(g.get("init_sigma", init_sigma.value))
                    except Exception:
                        pass
                    try:
                        fr = g.get("fit_range")
                        if isinstance(fr, (list, tuple)) and len(fr) == 2:
                            lo, hi = float(fr[0]), float(fr[1])
                            # clamp to bounds
                            lo = max(
                                float(fit_range.min), min(lo, float(fit_range.max))
                            )
                            hi = max(lo, min(hi, float(fit_range.max)))
                            fit_range.value = [lo, hi]
                    except Exception:
                        pass
            except Exception:
                pass
            _rebuild_alpha_sliders(idx)
            bulk_update_in_progress = False
            # Refresh the fit range overlay for the new spectrum
            try:
                _update_fit_range_indicator()
            except Exception:
                pass
            # Freeze y-axis to this spectrum's data range to reduce flicker on updates
            try:
                _fix_y_range(idx)
            except Exception:
                pass
            # Clear any prior fit from plot; wait for user to click Fit
            try:
                with fig.batch_update():
                    fig.data[1].x = []
                    fig.data[1].y = []
                    while len(fig.data) > 2:
                        fig.data = tuple(fig.data[:2])
            except Exception:
                pass
            try:
                status_html.value = "<span style='color:#555;'>Spectrum changed. Click Fit to compute.</span>"
            except Exception:
                pass
            # Persist time selection to session state
            try:
                if idx is not None and "Time" in FTIR_DataFrame.columns:
                    _set_session_selection(time=FTIR_DataFrame.loc[idx].get("Time"))
            except Exception:
                pass
        finally:
            on_spectrum_change_inflight = False
            try:
                last_on_spectrum_change_ts = time.time()
            except Exception:
                last_on_spectrum_change_ts = now_ts

    # No range checkboxes to manage

    def _save_for_file(b):
        if _recent_click("save_for_file"):
            return
        idx = spectrum_sel.value
        # Use the last completed result if available; if not, trigger a fit and inform
        # the user to wait for completion.
        res = last_result_by_idx.get(idx)
        if res is None:
            _log_once(
                "No current fit. Click 'Fit' to compute, then click 'Save' again."
            )
            return
        # Only save parameters for peaks in the currently selected fit range
        peaks_x, _ = _get_visible_peaks(idx)
        included = [i for i, cb in enumerate(include_checkboxes) if cb.value]
        out = []
        for i in included:
            cx = peaks_x[i]
            d = {}
            for name in ("amplitude", "center", "sigma", "fraction"):
                p = res.params.get(f"p{i}_{name}")
                if p is not None:
                    d[name] = float(p.value)
            out.append(d)
        FTIR_DataFrame.at[idx, results_col] = out
        _log_once(
            f"Saved deconvolution for file '{FTIR_DataFrame.loc[idx, 'File Name']}'."
        )

    def _close_ui(b):
        # Signal cancellation and close widgets promptly
        try:
            cancel_event.set()
        except Exception:
            pass
        try:
            spectrum_sel.close()
            material_dd.close()
            conditions_dd.close()
            center_window.close()
            init_sigma.close()
            add_peaks_btn.close()
            accept_new_peaks_btn.close()
            redo_new_peaks_btn.close()
            cancel_new_peaks_btn.close()
            # Ensure iterative/cancel controls disappear on close
            iter_btn.close()
            cancel_fit_btn.close()
            save_btn.close()
            close_btn.close()
            for s in alpha_sliders:
                s.close()
            peak_controls_box.close()
            # Close the main UI container and status label so no stray widgets remain
            try:
                status_html.close()
            except Exception:
                pass
            try:
                ui.close()
            except Exception:
                pass
            # Leave the log widget displayed so the last message remains visible
            fig.close()
        except Exception:
            pass

    # Wire events
    spectrum_sel.observe(_on_spectrum_change, names="value")
    material_dd.observe(_rebuild_spectrum_options, names="value")
    conditions_dd.observe(_rebuild_spectrum_options, names="value")
    # Persist material/conditions selections
    def _persist_pd_filters(_=None):
        try:
            _set_session_selection(material=material_dd.value, conditions=conditions_dd.value)
        except Exception:
            pass
    material_dd.observe(_persist_pd_filters, names="value")
    conditions_dd.observe(_persist_pd_filters, names="value")
    # Global control observers: split to avoid unnecessary per-peak UI rebuilds
    center_window.observe(_on_center_sigma_change, names="value")
    init_sigma.observe(_on_center_sigma_change, names="value")
    fit_range.observe(_on_fit_range_change, names="value")

    # Wire reset buttons
    def _reset_center(_b=None):
        nonlocal bulk_update_in_progress
        bulk_update_in_progress = True
        try:
            center_window.value = default_center_window_value
        except Exception:
            pass
        _snapshot_current_controls()
        bulk_update_in_progress = False
        try:
            status_html.value = "<span style='color:#555;'>Center window reset. Click Fit to update.</span>"
        except Exception:
            pass

    def _reset_sigma(_b=None):
        nonlocal bulk_update_in_progress
        bulk_update_in_progress = True
        try:
            init_sigma.value = default_init_sigma_value
        except Exception:
            pass
        _snapshot_current_controls()
        bulk_update_in_progress = False
        try:
            status_html.value = (
                "<span style='color:#555;'>Initial σ reset. Click Fit to update.</span>"
            )
        except Exception:
            pass

    def _reset_all(_b=None):
        nonlocal bulk_update_in_progress
        bulk_update_in_progress = True
        # Reset globals
        try:
            center_window.value = default_center_window_value
        except Exception:
            pass
        try:
            init_sigma.value = default_init_sigma_value
        except Exception:
            pass
        try:
            lo, hi = default_fit_range_value
            fit_range.value = [float(lo), float(hi)]
        except Exception:
            pass
        # Reset per-peak controls
        try:
            for cb in include_checkboxes:
                cb.value = DEFAULT_INCLUDE
        except Exception:
            pass
        try:
            for s in alpha_sliders:
                s.value = DEFAULT_ALPHA
        except Exception:
            pass
        _snapshot_current_controls()
        bulk_update_in_progress = False
        try:
            status_html.value = "<span style='color:#555;'>All controls reset. Click Fit to update.</span>"
        except Exception:
            pass

    reset_center_btn.on_click(_reset_center)
    reset_sigma_btn.on_click(_reset_sigma)
    reset_all_btn.on_click(_reset_all)

    def _on_fit_click(_b=None):
        if _recent_click("fit"):
            return
        _fit_and_update_plot()

    fit_btn.on_click(_on_fit_click)
    save_btn.on_click(_save_for_file)
    close_btn.on_click(_close_ui)

    # Helper: quietly set a widget attribute without triggering control-change side effects
    def _set_quiet(widget, attr, value):
        """Set a widget attribute while suppressing on_change observers."""
        nonlocal bulk_update_in_progress
        bulk_update_in_progress = True
        try:
            setattr(widget, attr, value)
        except Exception:
            pass
        finally:
            bulk_update_in_progress = False

    # Fix y-axis range to the current data to prevent autoscale flicker during updates
    def _fix_y_range(idx):
        try:
            x_arr, y_arr = _get_xy(idx)
        except Exception:
            x_arr, y_arr = None, None
        if y_arr is None:
            return
        try:
            y0_min = float(np.nanmin(y_arr))
            y0_max = float(np.nanmax(y_arr))
        except Exception:
            return
        try:
            with fig.batch_update():
                fig.update_yaxes(autorange=False, range=[y0_min, y0_max])
        except Exception:
            pass

    def _iteratively_correct_worker():
        """Coordinate-descent style tuning of parameters to reduce reduced chi-square.

        Iterates over per-peak α sliders (for included peaks), the center window, and
        the initial sigma, trying +/- step changes one parameter at a time. Keeps a
        change only if reduced chi-square improves. Stops when a full sweep makes no
        improvements, or after a safety cap of sweeps.
        """

        nonlocal iterating_in_progress, iter_start_redchi
        nonlocal iter_final_redchi, iter_summary_pending

        idx = spectrum_sel.value

        # Helper: run fit synchronously and return last redchi
        def _run_fit_and_wait():
            # If cancellation is already requested, don't start a new fit
            try:
                if cancel_event.is_set():
                    return np.inf
            except Exception:
                pass
            _fit_and_update_plot()
            # Wait for background fit to complete (with a timeout guard)
            try:
                for _ in range(400):  # up to ~40s total at 0.1s intervals
                    th = fit_thread
                    if th is None or not th.is_alive():
                        break
                    # Allow cooperative cancellation
                    try:
                        if cancel_event.is_set():
                            break
                    except Exception:
                        pass
                    time.sleep(0.1)
            except Exception:
                pass
            rc = last_redchi_by_idx.get(idx, np.inf)
            try:
                return float(rc)
            except Exception:
                return np.inf

        # Establish baseline reduced chi-square from snapshot taken at click time.
        # Fall back to one quick fit if no prior value exists.
        start_rc = iter_start_redchi
        try:
            start_rc = float(start_rc)
        except Exception:
            start_rc = np.inf
        if not np.isfinite(start_rc):
            # No prior redchi recorded; compute once.
            try:
                status_html.value = "<span style='color:#555;'>Iteratively correcting (pre-fit)...</span>"
            except Exception:
                pass
            start_rc = _run_fit_and_wait()
        base_rc = start_rc
        # Iteration change counter: increments whenever a parameter change is kept
        iteration_changes = 0
        # Minimal running status; will update the counter on each kept change
        try:
            status_html.value = (
                "<span style='color:#555;'>iterating... (iterations so far: 0)</span>"
            )
        except Exception:
            pass
        try:
            if cancel_event.is_set():
                # Show old -> current comparison on cancel
                try:
                    old_str = f"{start_rc:.4g}"
                except Exception:
                    old_str = str(start_rc)
                try:
                    new_str = f"{base_rc:.4g}"
                except Exception:
                    new_str = str(base_rc)
                try:
                    status_html.value = (
                        f"<span style='color:#a00;'>Iterative correction cancelled. "
                        f"Reduced chi-square: ({old_str}) ---&gt; ({new_str})</span>"
                    )
                except Exception:
                    pass
                try:
                    _on_main_thread(_update_cancel_fit_visibility)
                except Exception:
                    pass
                # Allow normal fit status updates again
                iterating_in_progress = False
                return
        except Exception:
            pass
        if not np.isfinite(base_rc):
            base_rc = _run_fit_and_wait()

        # Build the list of adjustable parameters
        included_idxs = [i for i, cb in enumerate(include_checkboxes) if cb.value]
        if len(included_idxs) == 0:
            try:
                status_html.value = (
                    f"<span style='color:#a00;'>Cannot iterate: no peaks "
                    f"selected.</span>"
                )
            except Exception:
                _log_once("Cannot iterate: no peaks selected.")
            try:
                _on_main_thread(_update_cancel_fit_visibility)
            except Exception:
                pass
            return

        alpha_steps = {i: 0.1 for i in included_idxs}
        center_step = 2.0
        sigma_step = 1.0

        def _clamp(val, lo, hi):
            try:
                return max(lo, min(hi, val))
            except Exception:
                return val

        # Try a single parameter perturbation, returning (improved_rc, kept_change)
        def _try_adjust(getter, setter, decrementer, restore, label):
            nonlocal base_rc
            # If cancel requested, do not proceed
            try:
                if cancel_event.is_set():
                    return base_rc, False
            except Exception:
                pass
            # Try +step
            setter()
            rc_plus = _run_fit_and_wait()
            if rc_plus < base_rc:
                base_rc = rc_plus
                return rc_plus, True
            # Try -step
            decrementer()
            rc_minus = _run_fit_and_wait()
            # Early exit on cancel
            try:
                if cancel_event.is_set():
                    restore()
                    return base_rc, False
            except Exception:
                pass
            if rc_minus < base_rc:
                base_rc = rc_minus
                return rc_minus, True
            # Revert if neither improved
            restore()
            return base_rc, False

        sweeps = 0
        max_sweeps = 10
        improved_any = True
        while improved_any and sweeps < max_sweeps:
            # Check for cancellation at the start of each sweep
            try:
                if cancel_event.is_set():
                    # Show running total and break to final message below
                    try:
                        status_html.value = (
                            f"<span style='color:#a00;'>Iterative correction "
                            f"cancelled.</span>"
                        )
                    except Exception:
                        pass
                    break
            except Exception:
                pass
            improved_any = False
            sweeps += 1
            # No per-sweep status update; counter increments only on kept changes

            # 1) Per-peak α sliders (only included peaks)
            for i in included_idxs:
                try:
                    if cancel_event.is_set():
                        break
                except Exception:
                    pass
                if i >= len(alpha_sliders):
                    continue
                sld = alpha_sliders[i]
                if sld is None:
                    continue
                v0 = float(sld.value)
                step = float(alpha_steps.get(i, 0.1))

                # Define actions
                def set_plus(v0=v0, sld=sld, step=step):
                    _set_quiet(sld, "value", _clamp(v0 + step, 0.0, 1.0))

                def set_minus(v0=v0, sld=sld, step=step):
                    _set_quiet(sld, "value", _clamp(v0 - step, 0.0, 1.0))

                def restore(v0=v0, sld=sld):
                    _set_quiet(sld, "value", v0)

                # Use getter label for debugging (not printed)
                _, kept = _try_adjust(
                    getter=lambda: sld.value,
                    setter=set_plus,
                    decrementer=set_minus,
                    restore=restore,
                    label=f"alpha[{i}]",
                )
                if kept:
                    improved_any = True
                    iteration_changes += 1
                    try:
                        status_html.value = (
                            f"<span style='color:#555;'>iterating... "
                            f"(iterations so far: {iteration_changes})</span>"
                        )
                    except Exception:
                        pass

            # 2) Center window (affects allowed center bounds)
            try:
                if cancel_event.is_set():
                    break
            except Exception:
                pass
            cw0 = float(center_window.value)

            def cw_plus(cw0=cw0):
                _set_quiet(
                    center_window,
                    "value",
                    _clamp(
                        cw0 + center_step,
                        float(center_window.min),
                        float(center_window.max),
                    ),
                )

            def cw_minus(cw0=cw0):
                _set_quiet(
                    center_window,
                    "value",
                    _clamp(
                        cw0 - center_step,
                        float(center_window.min),
                        float(center_window.max),
                    ),
                )

            def cw_restore(cw0=cw0):
                _set_quiet(center_window, "value", cw0)

            _, kept = _try_adjust(
                getter=lambda: center_window.value,
                setter=cw_plus,
                decrementer=cw_minus,
                restore=cw_restore,
                label="center_window",
            )
            if kept:
                improved_any = True
                iteration_changes += 1
                try:
                    status_html.value = (
                        f"<span style='color:#555;'>iterating... "
                        f"(iterations so far: {iteration_changes})</span>"
                    )
                except Exception:
                    pass

            # 3) Initial sigma
            try:
                if cancel_event.is_set():
                    break
            except Exception:
                pass
            sg0 = float(init_sigma.value)

            def sg_plus(sg0=sg0):
                _set_quiet(
                    init_sigma,
                    "value",
                    _clamp(
                        sg0 + sigma_step, float(init_sigma.min), float(init_sigma.max)
                    ),
                )

            def sg_minus(sg0=sg0):
                _set_quiet(
                    init_sigma,
                    "value",
                    _clamp(
                        sg0 - sigma_step, float(init_sigma.min), float(init_sigma.max)
                    ),
                )

            def sg_restore(sg0=sg0):
                _set_quiet(init_sigma, "value", sg0)

            _, kept = _try_adjust(
                getter=lambda: init_sigma.value,
                setter=sg_plus,
                decrementer=sg_minus,
                restore=sg_restore,
                label="init_sigma",
            )
            if kept:
                improved_any = True
                iteration_changes += 1
                try:
                    status_html.value = (
                        f"<span style='color:#555;'>iterating... "
                        f"(iterations so far: {iteration_changes})</span>"
                    )
                except Exception:
                    pass

        # Final status: show old -> new comparison
        try:
            old_str = f"{start_rc:.4g}"
        except Exception:
            old_str = str(start_rc)
        try:
            new_str = f"{base_rc:.4g}"
        except Exception:
            new_str = str(base_rc)
        try:
            if cancel_event.is_set():
                status_html.value = (
                    f"<span style='color:#a00;'>Iterative correction cancelled. "
                    f"Reduced chi-square: ({old_str}) ---&gt; ({new_str})</span>"
                )
            else:
                status_html.value = (
                    f"<span style='color:#000;'>Iterative correction complete. "
                    f"Reduced chi-square: ({old_str}) ---&gt; ({new_str})</span>"
                )
        except Exception:
            if cancel_event.is_set():
                _log_once(
                    f"Iterative correction cancelled. Reduced chi-square: ("
                    f"{old_str}) -> ({new_str})"
                )
            else:
                _log_once(
                    f"Iterative correction complete. Reduced chi-square: ("
                    f"{old_str}) -> ({new_str})"
                )
        # Remember iteration summary for the next refit message
        iter_final_redchi = base_rc
        iter_summary_pending = True
        # Allow normal fit status updates again
        iterating_in_progress = False
        # Hide the Cancel Fit button when iteration ends (success or cancel)
        try:
            _on_main_thread(_update_cancel_fit_visibility)
        except Exception:
            pass

    # Wire the iterative correct to run in background and support cancellation
    def _on_iteratively_correct_click(b):
        if _recent_click("iteratively_correct"):
            return
        nonlocal iter_thread, iterating_in_progress, cancel_event, iter_start_redchi
        try:
            if iter_thread is not None and iter_thread.is_alive():
                try:
                    status_html.value = (
                        f"<span style='color:#a60;'>Iterative correction"
                        f" is already running.</span>"
                    )
                except Exception:
                    pass
                return
        except Exception:
            pass
        # Snapshot current redchi for this spectrum at the moment of click
        try:
            idx_snapshot = spectrum_sel.value
            iter_start_redchi = last_redchi_by_idx.get(idx_snapshot, np.inf)
        except Exception:
            iter_start_redchi = np.inf
        # Reset cancel flag and mark iteration as active
        try:
            cancel_event.clear()
        except Exception:
            pass
        iterating_in_progress = True
        try:
            status_html.value = (
                "<span style='color:#555;'>Starting iterative correction...</span>"
            )
        except Exception:
            pass
        iter_thread = threading.Thread(target=_iteratively_correct_worker, daemon=True)
        iter_thread.start()
        try:
            _update_cancel_fit_visibility()
        except Exception:
            pass

    iter_btn.on_click(_on_iteratively_correct_click)

    def _on_cancel_fit(b):
        if _recent_click("cancel_fit"):
            return
        # Signal cancellation for any active fit/iteration
        try:
            cancel_event.set()
        except Exception:
            pass
        try:
            if fit_cancel_token is not None:
                fit_cancel_token.set()
        except Exception:
            pass
        try:
            status_html.value = (
                "<span style='color:#a00;'>Cancellation requested...</span>"
            )
        except Exception:
            pass
        # Keep visibility consistent; it will auto-hide when work stops
        try:
            _update_cancel_fit_visibility()
        except Exception:
            pass

    cancel_fit_btn.on_click(_on_cancel_fit)

    def _on_mark_bad_click(_b=None):
        try:
            idx = spectrum_sel.value
            qcol = _quality_column_name(FTIR_DataFrame)
            FTIR_DataFrame.at[idx, qcol] = "bad"
            try:
                status_html.value = (
                    f"<span style='color:#a00;'>Marked row {idx} as bad quality.</span>"
                )
            except Exception:
                pass
            # Rebuild spectrum options to drop the bad row
            try:
                _rebuild_spectrum_options()
            except Exception:
                pass
        except Exception:
            pass
        try:
            _refresh_mark_buttons()
        except Exception:
            pass

    # Defer wiring mark_bad_btn until after it's instantiated below

    # Layout
    controls_row_filters = widgets.HBox([material_dd, conditions_dd, include_bad_cb])
    controls_row_spectrum = widgets.HBox([spectrum_sel])
    # Place the Fit X-range slider above the peak modification section
    fit_range_row = widgets.HBox([fit_range])
    # Keep other global parameters grouped below the peak controls
    globals_column = widgets.VBox(
        [
            widgets.HBox([center_window, reset_center_btn]),
            widgets.HBox([init_sigma, reset_sigma_btn]),
        ]
    )
    reset_all_row = widgets.HBox([reset_all_btn])
    # Mark buttons on their own row as a pair
    mark_good_btn = widgets.Button(
        description="Mark as good",
        button_style="success",
        layout=widgets.Layout(width="110px"),
        tooltip="Mark the currently selected spectrum as good",
    )

    def _on_mark_good_click(_b=None):
        try:
            idx = spectrum_sel.value
            qcol = _quality_column_name(FTIR_DataFrame)
            FTIR_DataFrame.at[idx, qcol] = "good"
            try:
                status_html.value = f"<span style='color:#0a0;'>Marked row {idx} as good quality.</span>"
            except Exception:
                pass
            try:
                _rebuild_spectrum_options()
            except Exception:
                pass
        except Exception:
            pass
        try:
            _refresh_mark_buttons()
        except Exception:
            pass

    mark_bad_btn = widgets.Button(
        description="Mark as bad",
        button_style="danger",
        layout=widgets.Layout(width="110px"),
        tooltip="Mark the currently selected spectrum as bad and remove from lists",
    )
    mark_bad_btn.on_click(_on_mark_bad_click)
    mark_good_btn.on_click(_on_mark_good_click)
    mark_row = widgets.HBox([mark_bad_btn, mark_good_btn])

    def _refresh_mark_buttons():
        try:
            idx = spectrum_sel.value
            qcol = _quality_column_name(FTIR_DataFrame)
            status = None
            try:
                if idx is not None:
                    status = FTIR_DataFrame.at[idx, qcol]
            except Exception:
                status = None
            is_bad = str(status).strip().lower() == "bad"
            mark_bad_btn.layout.display = "none" if is_bad else ""
            mark_good_btn.layout.display = "" if is_bad else "none"
        except Exception:
            pass

    buttons_row = widgets.HBox(
        [
            fit_btn,
            add_peaks_btn,
            accept_new_peaks_btn,
            redo_new_peaks_btn,
            cancel_new_peaks_btn,
            iter_btn,
            cancel_fit_btn,
            save_btn,
            close_btn,
        ]
    )
    status_row = widgets.HBox([status_html])
    ui = widgets.VBox(
        [
            controls_row_filters,
            controls_row_spectrum,
            fit_range_row,
            peak_controls_box,
            globals_column,
            reset_all_row,
            buttons_row,
            mark_row,
            status_row,
        ]
    )

    # --- Add-peaks workflow callbacks ---
    def _enter_add_mode(b=None):
        if _recent_click("enter_add_mode"):
            return
        nonlocal adding_mode
        adding_mode = True
        new_peak_xs.clear()
        _clear_add_peak_shapes()
        _hide(add_peaks_btn)
        _show(accept_new_peaks_btn)
        _show(redo_new_peaks_btn)
        _show(cancel_new_peaks_btn)
        # Hide parameter modifiers during add-peaks mode; keep Center ±window visible
        _hide(init_sigma)
        _hide(reset_sigma_btn)
        _hide(iter_btn)
        _hide(cancel_fit_btn)
        _hide(save_btn)
        _hide(close_btn)
        _hide(peak_controls_box)
        # Disable selection widgets to avoid changing spectrum/filters mid-selection
        try:
            spectrum_sel.disabled = True
            material_dd.disabled = True
            conditions_dd.disabled = True
        except Exception:
            pass
        _log_once(
            "Add-peaks mode: click one or more x-locations on the plot. Then accept/redo/cancel."
        )

    def _accept_new_peaks(b=None):
        if _recent_click("accept_new_peaks"):
            return
        nonlocal adding_mode, shared_peaks_x
        idx = spectrum_sel.value
        x_arr, y_arr = _get_xy(idx)
        if x_arr is None:
            _log_once("Cannot accept: current spectrum has no normalized data.")
            return
        if len(new_peak_xs) == 0:
            # Nothing selected; just exit mode
            _cancel_new_peaks()
            return
        # Merge with existing peaks and sort; reject new peaks too close to existing
        xs_existing, ys_existing = _get_peaks(idx)
        xs_list = [float(v) for v in (xs_existing or [])]
        ys_list = [float(v) for v in (ys_existing or [])]

        try:
            min_sep = float(center_window.value)
        except Exception:
            min_sep = 0.0

        rejected_close = []
        for x_new in new_peak_xs:
            # Reject if too close to any existing (committed) peak
            too_close = False
            for xe in xs_existing or []:
                try:
                    if abs(float(xe) - float(x_new)) <= min_sep:
                        too_close = True
                        break
                except Exception:
                    continue
            if too_close:
                rejected_close.append(float(x_new))
                continue
            # find nearest y
            try:
                i = int(np.argmin(np.abs(x_arr - x_new)))
                y_new = float(y_arr[i])
            except Exception:
                y_new = float("nan")
            xs_list.append(float(x_new))
            ys_list.append(y_new)

        # Sort by x
        try:
            pairs = sorted(zip(xs_list, ys_list), key=lambda t: float(t[0]))
            xs_sorted, ys_sorted = [list(t) for t in zip(*pairs)] if pairs else ([], [])
        except Exception:
            xs_sorted, ys_sorted = xs_list, ys_list

        # Persist back to DataFrame for this spectrum
        FTIR_DataFrame.at[idx, "Peak Wavenumbers"] = [float(v) for v in xs_sorted]
        FTIR_DataFrame.at[idx, "Peak Absorbances"] = [float(v) for v in ys_sorted]
        # Update the shared/template peaks for this filter group so they carry over
        try:
            shared_peaks_x = [float(v) for v in xs_sorted]
        except Exception:
            shared_peaks_x = xs_sorted

        # Peaks changed; clear any saved per-peak settings for this spectrum
        try:
            per_spec_alpha.pop(idx, None)
            per_spec_include.pop(idx, None)
        except Exception:
            pass

        # Rebuild UI and refit with the updated peaks
        _rebuild_alpha_sliders(idx)
        # Do not auto-fit; wait for user to click Fit
        try:
            status_html.value = (
                "<span style='color:#555;'>Peaks updated. Click Fit to update.</span>"
            )
        except Exception:
            pass

        # Exit add mode and clean up visuals
        _clear_add_peak_shapes()
        new_peak_xs.clear()
        adding_mode = False
        _show(add_peaks_btn)
        _hide(accept_new_peaks_btn)
        _hide(redo_new_peaks_btn)
        _hide(cancel_new_peaks_btn)
        # Restore previously hidden/disabled controls after exiting add-peaks mode
        _show(init_sigma)
        _show(reset_sigma_btn)
        _show(iter_btn)
        _update_cancel_fit_visibility()
        _show(save_btn)
        _show(close_btn)
        _show(peak_controls_box)
        try:
            spectrum_sel.disabled = False
            material_dd.disabled = False
            conditions_dd.disabled = False
        except Exception:
            pass
        # Status: show any rejections in red
        if rejected_close:
            joined = ", ".join(f"{v:.3f}" for v in rejected_close)
            msg = (
                "Rejected due to proximity (±{:.2f} cm⁻¹): {}. "
                "Tip: reduce the Center ±window to fit peaks in small spaces."
            ).format(min_sep, joined)
            try:
                status_html.value = f"<span style='color:#a00;'>{msg}</span>"
            except Exception:
                _log_once(msg)
        else:
            _log_once("New peaks accepted and added to the current spectrum.")

    def _redo_new_peaks(b=None):
        if _recent_click("redo_new_peaks"):
            return
        new_peak_xs.clear()
        _clear_add_peak_shapes()
        _log_once("Selection cleared. Click on the plot to select peaks again.")

    def _cancel_new_peaks(b=None):
        if _recent_click("cancel_new_peaks"):
            return
        nonlocal adding_mode
        adding_mode = False
        new_peak_xs.clear()
        _clear_add_peak_shapes()
        _show(add_peaks_btn)
        _hide(accept_new_peaks_btn)
        _hide(redo_new_peaks_btn)
        _hide(cancel_new_peaks_btn)
        # Restore previously hidden/disabled controls after cancelling add-peaks mode
        _show(init_sigma)
        _show(reset_sigma_btn)
        _show(iter_btn)
        _update_cancel_fit_visibility()
        _show(save_btn)
        _show(close_btn)
        _show(peak_controls_box)
        try:
            spectrum_sel.disabled = False
            material_dd.disabled = False
            conditions_dd.disabled = False
        except Exception:
            pass
        _log_once("Peak addition cancelled. No changes were made.")

    add_peaks_btn.on_click(_enter_add_mode)
    accept_new_peaks_btn.on_click(_accept_new_peaks)
    redo_new_peaks_btn.on_click(_redo_new_peaks)
    cancel_new_peaks_btn.on_click(_cancel_new_peaks)

    display(ui, fig, log_html)
    # Seed options with current filters (defaults 'any') and trigger initial updates
    _rebuild_spectrum_options()

    return FTIR_DataFrame


def time_series_fitting(FTIR_DataFrame):
    """Interactive time-series fitting and visualization.

    - Computes time-series fits where peak centers/shapes are shared within a series
      (Material + Conditions including 'unexposed'), and amplitudes vary per spectrum.
    - Provides a UI with Material and Conditions dropdowns and a 'Fit Time-Series'
      button, then plots the selected series by Time (ascending) and overlays fits.

    Parameters
    ----------
    FTIR_DataFrame : pd.DataFrame
        DataFrame containing FTIR spectral data and metadata.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with time-series fitting results.
    """
    # --- Validation and setup (shared for compute + UI) ---
    if FTIR_DataFrame is None or len(FTIR_DataFrame) == 0:
        raise ValueError("FTIR_DataFrame must be loaded and non-empty.")

    # Identify condition column name and required columns
    cond_col = (
        "Conditions"
        if "Conditions" in FTIR_DataFrame.columns
        else ("Condition" if "Condition" in FTIR_DataFrame.columns else None)
    )
    if cond_col is None:
        raise KeyError("Missing 'Conditions' (or 'Condition') column in DataFrame.")

    required_cols = [
        "Material",
        cond_col,
        "Time",
        "Deconvolution Results",
        "X-Axis",
        "Normalized and Corrected Data",
        "Time-Series Fit Results",
    ]
    missing = [c for c in required_cols if c not in FTIR_DataFrame.columns]
    if missing:
        raise KeyError(
            f"Missing required column(s): {missing}. Ensure your DataFrame is prepared "
            f"with prior steps."
        )

    # Ensure destination column can hold arbitrary Python objects
    try:
        FTIR_DataFrame["Time-Series Fit Results"] = FTIR_DataFrame[
            "Time-Series Fit Results"
        ].astype(object)
    except Exception:
        pass

    # -------------------------- Backend helpers --------------------------- #
    def _parse_deconv(val):
        """Parse a Deconvolution Results cell to a list[dict] or None."""
        if val is None:
            return None
        if isinstance(val, str):
            try:
                v = ast.literal_eval(val)
            except Exception:
                return None
        else:
            v = val
        if isinstance(v, list):
            # Ensure items are dicts and sort by center if present
            try:
                items = [dict(d) for d in v if isinstance(d, dict)]
            except Exception:
                return None
            try:
                items = sorted(
                    items, key=lambda d: float(d.get("center", float("nan")))
                )
            except Exception:
                # If centers not coercible, keep original order
                pass
            return items
        return None

    def _parse_xy(row):
        """Return (x_arr, y_arr) as 1D float arrays or (None, None)."""
        x = row.get("X-Axis")
        y = row.get("Normalized and Corrected Data")
        if isinstance(x, str):
            try:
                x = ast.literal_eval(x)
            except Exception:
                return None, None
        if isinstance(y, str):
            try:
                y = ast.literal_eval(y)
            except Exception:
                return None, None
        try:
            x_arr = np.asarray(x, dtype=float)
            y_arr = np.asarray(y, dtype=float)
            if x_arr.ndim != 1 or y_arr.ndim != 1 or x_arr.size != y_arr.size:
                return None, None
        except Exception:
            return None, None
        return x_arr, y_arr

    def _mode_peak_count(lists_of_peaks):
        """Return the most common positive length among lists; tie -> max length."""
        lengths = [len(p) for p in lists_of_peaks if isinstance(p, list) and len(p) > 0]
        if not lengths:
            return 0
        vals, counts = np.unique(lengths, return_counts=True)
        # Choose value with max count; if tie, the larger length wins
        max_count = np.max(counts)
        candidates = [v for v, c in zip(vals, counts) if c == max_count]
        return int(max(candidates))

    def _is_unexposed(val):
        try:
            return str(val).strip().lower() == "unexposed"
        except Exception:
            return False

    # Compute only for the currently selected series (Material + Condition incl. unexposed)
    def _compute_for_selection(material, condition, include_bad=False):
        # Build the series subset
        series_mask = (FTIR_DataFrame["Material"].astype(str) == str(material)) & (
            (FTIR_DataFrame[cond_col].astype(str) == str(condition))
            | FTIR_DataFrame[cond_col].apply(_is_unexposed)
        )
        series_df = FTIR_DataFrame[series_mask].copy()
        # Optionally exclude rows marked as bad quality
        if not include_bad:
            try:
                series_df = series_df[_quality_good_mask(series_df)]
            except Exception:
                pass
        if series_df.empty:
            print("No spectra found for the selected Material/Conditions.")
            return
        # Collect deconvolution peak lists for this series
        peak_lists = []
        peak_lists_by_idx = {}
        for idx, row in series_df.iterrows():
            peaks = _parse_deconv(row.get("Deconvolution Results"))
            if peaks is not None and len(peaks) > 0:
                peak_lists.append(peaks)
                peak_lists_by_idx[idx] = peaks
        k = _mode_peak_count(peak_lists)
        if k <= 0:
            print("Selected series has no usable deconvolution results.")
            return
        centers = []
        sigmas = []
        fracs = []
        for peaks in peak_lists:
            if len(peaks) != k:
                continue
            try:
                centers.append([float(p.get("center", np.nan)) for p in peaks])
                sigmas.append([float(p.get("sigma", np.nan)) for p in peaks])
                fracs.append([float(p.get("fraction", np.nan)) for p in peaks])
            except Exception:
                continue
        if not centers:
            print("Selected series has inconsistent peak counts; cannot average.")
            return
        centers = np.asarray(centers, dtype=float)
        sigmas = np.asarray(sigmas, dtype=float)
        fracs = np.asarray(fracs, dtype=float)
        with np.errstate(all="ignore"):
            avg_center = np.nanmean(centers, axis=0)
            avg_sigma = np.nanmean(sigmas, axis=0)
            avg_frac = np.nanmean(fracs, axis=0)
        for i in range(k):
            if not np.isfinite(avg_center[i]):
                vals = centers[:, i]
                avg_center[i] = (
                    np.nanmedian(vals) if np.isfinite(np.nanmedian(vals)) else 0.0
                )
            if not np.isfinite(avg_sigma[i]):
                vals = sigmas[:, i]
                avg_sigma[i] = (
                    np.nanmedian(vals) if np.isfinite(np.nanmedian(vals)) else 10.0
                )
            if not np.isfinite(avg_frac[i]):
                vals = fracs[:, i]
                avg_frac[i] = (
                    np.nanmedian(vals) if np.isfinite(np.nanmedian(vals)) else 0.5
                )
        # Optional average amplitude (for initial guesses only)
        amps = []
        for peaks in peak_lists:
            if len(peaks) != k:
                continue
            try:
                amps.append([float(p.get("amplitude", np.nan)) for p in peaks])
            except Exception:
                continue
        avg_amp = None
        if amps:
            amps = np.asarray(amps, dtype=float)
            with np.errstate(all="ignore"):
                avg_amp = np.nanmean(amps, axis=0)

        # Build fixed-parameter composite template for this series (center/sigma/fraction fixed)
        def _build_model_with_fixed_params():
            comp_model = None
            params = None
            for i in range(k):
                m = PseudoVoigtModel(prefix=f"p{i}_")
                p = m.make_params()
                p[f"p{i}_center"].set(value=float(avg_center[i]), vary=False)
                p[f"p{i}_sigma"].set(
                    value=float(avg_sigma[i]), min=1e-3, max=1e4, vary=False
                )
                p[f"p{i}_fraction"].set(
                    value=float(avg_frac[i]), min=0.0, max=1.0, vary=False
                )
                p[f"p{i}_amplitude"].set(
                    min=0.0,
                    value=(
                        float(avg_amp[i]) if isinstance(avg_amp, np.ndarray) else 1.0
                    ),
                )
                if comp_model is None:
                    comp_model = m
                    params = p
                else:
                    comp_model = comp_model + m
                    params.update(p)
            return comp_model, params

        # Fit only amplitudes per spectrum in this selected series
        fits_done = 0
        for idx, row in series_df.iterrows():
            x_arr, y_arr = _parse_xy(row)
            if x_arr is None or y_arr is None or x_arr.size == 0:
                continue
            comp_model, params = _build_model_with_fixed_params()
            peaks_row = _parse_deconv(row.get("Deconvolution Results"))
            if isinstance(peaks_row, list) and len(peaks_row) == k:
                for i in range(k):
                    try:
                        ai = float(
                            peaks_row[i].get(
                                "amplitude", params[f"p{i}_amplitude"].value
                            )
                        )
                        params[f"p{i}_amplitude"].set(value=max(0.0, ai))
                    except Exception:
                        pass
            else:
                try:
                    for i in range(k):
                        ci = float(avg_center[i])
                        nearest = int(np.argmin(np.abs(x_arr - ci)))
                        ai0 = max(
                            0.0, float(y_arr[nearest]) * max(1.0, float(avg_sigma[i]))
                        )
                        params[f"p{i}_amplitude"].set(value=ai0)
                except Exception:
                    pass
            try:
                result = comp_model.fit(y_arr, params, x=x_arr)
            except Exception:
                continue
            out = []
            try:
                for i in range(k):
                    amp = float(result.params.get(f"p{i}_amplitude").value)
                    out.append(
                        {
                            "amplitude": amp,
                            "center": float(avg_center[i]),
                            "sigma": float(avg_sigma[i]),
                            "fraction": float(avg_frac[i]),
                        }
                    )
                FTIR_DataFrame.at[idx, "Time-Series Fit Results"] = out
                fits_done += 1
            except Exception:
                pass
        print(
            f"Time-series fit complete for selection (Material={material}, Condition={condition}). Fitted {fits_done} spectra."
        )

    def _optimize_centers_for_selection(material, condition, include_bad=False):
        """Iteratively optimize shared centers and per-spectrum amplitudes to reduce SSE.

        Approach:
        - Coordinate descent on shared centers. For each peak, try +/- step shifts.
        - For every center trial, re-fit amplitudes per spectrum with centers/σ/α fixed
          against the 'Normalized and Corrected Data' to get the best amplitudes.
        - Accept moves that lower the total SSE; shrink step when no progress; stop
          after a few passes or when below tolerance.

        Returns a status message and the optimized centers list (or None on failure).
        """
        # Build series subset (Material matches; Conditions match or are 'unexposed')
        series_mask = (FTIR_DataFrame["Material"].astype(str) == str(material)) & (
            (FTIR_DataFrame[cond_col].astype(str) == str(condition))
            | FTIR_DataFrame[cond_col].apply(_is_unexposed)
        )
        series_df = FTIR_DataFrame[series_mask].copy()
        # Optionally exclude rows marked as bad quality
        if not include_bad:
            try:
                series_df = series_df[_quality_good_mask(series_df)]
            except Exception:
                pass
        if series_df.empty:
            return "No spectra found for the selected Material/Conditions.", None

        # Seed shared parameters from deconvolution results
        peak_lists = []
        for _idx, _row in series_df.iterrows():
            pk = _parse_deconv(_row.get("Deconvolution Results"))
            if pk:
                peak_lists.append(pk)
        k = _mode_peak_count(peak_lists)
        if k <= 0:
            return "Selected series has no usable deconvolution results.", None

        centers, sigmas, fracs = [], [], []
        for pk in peak_lists:
            if len(pk) != k:
                continue
            try:
                centers.append([float(p.get("center", np.nan)) for p in pk])
                sigmas.append([float(p.get("sigma", np.nan)) for p in pk])
                fracs.append([float(p.get("fraction", np.nan)) for p in pk])
            except Exception:
                continue
        if not centers:
            return "Selected series has inconsistent peak counts; cannot average.", None
        centers = np.asarray(centers, dtype=float)
        sigmas = np.asarray(sigmas, dtype=float)
        fracs = np.asarray(fracs, dtype=float)
        with np.errstate(all="ignore"):
            cen = np.nanmean(centers, axis=0)
            sig = np.nanmean(sigmas, axis=0)
            frc = np.nanmean(fracs, axis=0)
        for i in range(k):
            if not np.isfinite(cen[i]):
                vals = centers[:, i]
                cen[i] = np.nanmedian(vals) if np.isfinite(np.nanmedian(vals)) else 0.0
            if not np.isfinite(sig[i]):
                vals = sigmas[:, i]
                sig[i] = np.nanmedian(vals) if np.isfinite(np.nanmedian(vals)) else 10.0
            if not np.isfinite(frc[i]):
                vals = fracs[:, i]
                frc[i] = np.nanmedian(vals) if np.isfinite(np.nanmedian(vals)) else 0.5

        # Determine global x-bounds across the series
        try:
            x_min = float("inf")
            x_max = float("-inf")
            for _idx, _row in series_df.iterrows():
                x_arr, y_arr = _parse_xy(_row)
                if x_arr is None or y_arr is None or x_arr.size == 0:
                    continue
                x_min = min(x_min, float(np.nanmin(x_arr)))
                x_max = max(x_max, float(np.nanmax(x_arr)))
            if not np.isfinite(x_min) or not np.isfinite(x_max):
                x_min, x_max = 0.0, 1.0
        except Exception:
            x_min, x_max = 0.0, 1.0

        def _build_model_fixed(cen_arr):
            comp = None
            params = None
            for i in range(k):
                m = PseudoVoigtModel(prefix=f"p{i}_")
                p = m.make_params()
                p[f"p{i}_center"].set(value=float(cen_arr[i]), vary=False)
                p[f"p{i}_sigma"].set(value=float(sig[i]), min=1e-3, max=1e4, vary=False)
                p[f"p{i}_fraction"].set(
                    value=float(frc[i]), min=0.0, max=1.0, vary=False
                )
                p[f"p{i}_amplitude"].set(min=0.0, value=1.0)
                if comp is None:
                    comp = m
                    params = p
                else:
                    comp = comp + m
                    params.update(p)
            return comp, params

        def _fit_and_sse(cen_arr, assign=False, capture=None):
            comp, base = _build_model_fixed(cen_arr)
            total = 0.0
            cache = {}
            for _idx, _row in series_df.iterrows():
                # Always compute residuals against processed data
                # Explicitly read from 'X-Axis' and 'Normalized and Corrected Data'
                x_val = _row.get("X-Axis")
                y_val = _row.get("Normalized and Corrected Data")
                if isinstance(x_val, str):
                    try:
                        x_val = ast.literal_eval(x_val)
                    except Exception:
                        x_val = None
                if isinstance(y_val, str):
                    try:
                        y_val = ast.literal_eval(y_val)
                    except Exception:
                        y_val = None
                try:
                    x_arr = np.asarray(x_val, dtype=float)
                    y_arr = np.asarray(y_val, dtype=float)
                    if x_arr.ndim != 1 or y_arr.ndim != 1 or x_arr.size != y_arr.size:
                        continue
                except Exception:
                    continue
                p = base.copy()
                peaks_row = _parse_deconv(_row.get("Deconvolution Results"))
                if isinstance(peaks_row, list) and len(peaks_row) == k:
                    for i in range(k):
                        try:
                            ai = float(
                                peaks_row[i].get(
                                    "amplitude", p[f"p{i}_amplitude"].value
                                )
                            )
                            p[f"p{i}_amplitude"].set(value=max(0.0, ai))
                        except Exception:
                            pass
                else:
                    for i in range(k):
                        try:
                            ci = float(cen_arr[i])
                            nearest = int(np.argmin(np.abs(x_arr - ci)))
                            ai0 = max(
                                0.0, float(y_arr[nearest]) * max(1.0, float(sig[i]))
                            )
                            p[f"p{i}_amplitude"].set(value=ai0)
                        except Exception:
                            pass
                try:
                    res = comp.fit(y_arr, p, x=x_arr)
                    y_fit = comp.eval(res.params, x=x_arr)
                    err = y_arr - y_fit
                    total += float(np.nansum(err * err))
                    if assign:
                        fit_list = []
                        for j in range(k):
                            try:
                                amp = float(res.params.get(f"p{j}_amplitude").value)
                            except Exception:
                                amp = float("nan")
                            fit_list.append(
                                {
                                    "amplitude": amp,
                                    "center": float(cen_arr[j]),
                                    "sigma": float(sig[j]),
                                    "fraction": float(frc[j]),
                                }
                            )
                        cache[_idx] = fit_list
                except Exception:
                    total += 1e12
            if assign:
                for _idx, fit_list in cache.items():
                    try:
                        FTIR_DataFrame.at[_idx, "Time-Series Fit Results"] = fit_list
                    except Exception:
                        pass
                if capture is not None:
                    try:
                        capture.clear()
                        capture.update(cache)
                    except Exception:
                        pass
            return total

        # Initial evaluation: fit amplitudes with seed centers and write results
        cen_init = cen.copy()
        cen_curr = cen.copy()
        init_cache = {}
        best_sse = _fit_and_sse(cen_curr, assign=True, capture=init_cache)
        initial_sse = best_sse
        # Track the centers corresponding to the current best SSE explicitly
        best_centers = cen_curr.copy()

        # Step size based on sigma scale
        try:
            sig_med = (
                float(np.nanmedian(sig)) if np.isfinite(np.nanmedian(sig)) else 10.0
            )
        except Exception:
            sig_med = 10.0
        step = max(0.5, min(10.0, 0.2 * sig_med))
        min_step = 0.05
        max_passes = 10
        tol = 1e-6

        passes_done = 0
        moves_attempted = 0
        moves_accepted = 0
        for _pass in range(max_passes):
            improved = False
            for i in range(k):
                for direction in (-1.0, 1.0):
                    cen_try = cen_curr.copy()
                    cen_try[i] = float(
                        np.clip(cen_try[i] + direction * step, x_min, x_max)
                    )
                    sse_try = _fit_and_sse(cen_try, assign=False)
                    moves_attempted += 1
                    if sse_try + tol < best_sse:
                        cen_curr = cen_try
                        best_sse = sse_try
                        best_centers = cen_try.copy()
                        improved = True
                        moves_accepted += 1
                        break
            if not improved:
                step *= 0.5
                if step < min_step:
                    break
            passes_done += 1

        # Final amplitudes re-fit and saved using the optimized centers (best found)
        final_cache = {}
        # Ensure we evaluate and assign using the best centers discovered
        cen_curr = best_centers.copy()
        final_sse = _fit_and_sse(cen_curr, assign=True, capture=final_cache)
        # Report delta: negative means decreased SSE
        delta = final_sse - initial_sse
        msg = (
            f"Optimized centers and amplitudes over {series_df.shape[0]} spectra. SSE: {initial_sse:.6g} -> {final_sse:.6g}"
            + (f" (Δ={delta:.6g})" if np.isfinite(delta) else "")
        )
        # Build detailed summary
        try:
            center_deltas = (cen_curr - cen_init).tolist()
        except Exception:
            center_deltas = None

        # Compute mean amplitude per peak across spectra (before vs after)
        def _mean_amps(cache_dict):
            means = []
            try:
                for j in range(k):
                    vals = []
                    for _v in cache_dict.values():
                        try:
                            a = float(_v[j].get("amplitude", float("nan")))
                            if np.isfinite(a):
                                vals.append(a)
                        except Exception:
                            pass
                    means.append(float(np.nanmean(vals)) if len(vals) else float("nan"))
            except Exception:
                pass
            return means

        amp_init_means = _mean_amps(init_cache)
        amp_final_means = _mean_amps(final_cache)
        try:
            amp_delta_means = [
                (
                    (amp_final_means[i] - amp_init_means[i])
                    if i < len(amp_init_means)
                    and np.isfinite(amp_init_means[i])
                    and np.isfinite(amp_final_means[i])
                    else float("nan")
                )
                for i in range(max(len(amp_init_means), len(amp_final_means)))
            ]
        except Exception:
            amp_delta_means = []
        # Build per-spectrum amplitude deltas (final - initial) for each peak
        amp_delta_by_spectrum = []
        try:
            # Sort spectra by numeric Time if available
            df_order = series_df.copy()
            try:
                df_order["_sort_time"] = pd.to_numeric(
                    df_order.get("Time", np.nan), errors="coerce"
                )
                df_order["_sort_time"] = df_order["_sort_time"].fillna(float("inf"))
                df_order = df_order.sort_values(by=["_sort_time"], kind="mergesort")
            except Exception:
                pass
            for _idx, _row in df_order.iterrows():
                init_list = init_cache.get(_idx)
                final_list = final_cache.get(_idx)
                label = f"T={_row.get('Time')}"
                deltas = []
                if isinstance(init_list, list) and isinstance(final_list, list):
                    for j in range(k):
                        try:
                            a0 = float(init_list[j].get("amplitude", float("nan")))
                        except Exception:
                            a0 = float("nan")
                        try:
                            a1 = float(final_list[j].get("amplitude", float("nan")))
                        except Exception:
                            a1 = float("nan")
                        try:
                            d = (
                                a1 - a0
                                if np.isfinite(a0) and np.isfinite(a1)
                                else float("nan")
                            )
                        except Exception:
                            d = float("nan")
                        deltas.append(d)
                amp_delta_by_spectrum.append({"label": label, "deltas": deltas})
        except Exception:
            amp_delta_by_spectrum = []
        summary = {
            "initial_sse": float(initial_sse) if np.isfinite(initial_sse) else None,
            "final_sse": float(final_sse) if np.isfinite(final_sse) else None,
            "passes": int(passes_done),
            "moves_attempted": int(moves_attempted),
            "moves_accepted": int(moves_accepted),
            "initial_centers": cen_init.tolist(),
            "final_centers": cen_curr.tolist(),
            "center_deltas": center_deltas,
            "amp_initial_means": amp_init_means,
            "amp_final_means": amp_final_means,
            "amp_delta_means": amp_delta_means,
            "amp_delta_by_spectrum": amp_delta_by_spectrum,
        }
        return msg, cen_curr.tolist(), summary

    # ---------------------------- UI helpers ----------------------------- #
    def _series_df(material_val, condition_val, include_bad=False):
        df = FTIR_DataFrame.copy()
        # Optionally exclude rows marked as bad quality
        if not include_bad:
            try:
                df = df[_quality_good_mask(df)]
            except Exception:
                pass
        try:
            df = df[df["Normalized and Corrected Data"].notna()]
        except Exception:
            pass
        try:
            df = df[df["Material"].astype(str) == str(material_val)]
        except Exception:
            df = df[df.get("Material", "").astype(str) == str(material_val)]

        def _is_unexp(v):
            try:
                return str(v).strip().lower() == "unexposed"
            except Exception:
                return False

        try:
            mask = (df[cond_col].astype(str) == str(condition_val)) | df[
                cond_col
            ].apply(_is_unexp)
            df = df[mask]
        except Exception:
            pass
        try:
            df = df.copy()
            df["_sort_time"] = pd.to_numeric(df.get("Time", np.nan), errors="coerce")
            df["_sort_time"] = df["_sort_time"].fillna(float("inf"))
            df = df.sort_values(by=["_sort_time"], kind="mergesort")
            try:
                df = df.drop(columns=["_sort_time"])
            except Exception:
                pass
        except Exception:
            pass
        return df

    def _eval_timeseries_fit(row):
        x = row.get("X-Axis")
        y = row.get("Normalized and Corrected Data")
        if isinstance(x, str):
            try:
                x = ast.literal_eval(x)
            except Exception:
                return None, None, None
        if isinstance(y, str):
            try:
                y = ast.literal_eval(y)
            except Exception:
                return None, None, None
        try:
            x_arr = np.asarray(x, dtype=float)
            y_arr = np.asarray(y, dtype=float)
        except Exception:
            return None, None, None
        res = row.get("Time-Series Fit Results")
        if isinstance(res, dict):
            return x_arr, y_arr, None
        if isinstance(res, str):
            try:
                res = ast.literal_eval(res)
            except Exception:
                res = None
        if not isinstance(res, list) or len(res) == 0:
            return x_arr, y_arr, None
        comp_model = None
        params = None
        try:
            for i, p in enumerate(res):
                m = PseudoVoigtModel(prefix=f"p{i}_")
                pr = m.make_params()
                try:
                    pr[f"p{i}_center"].set(
                        value=float(p.get("center", 0.0)), vary=False
                    )
                except Exception:
                    pr[f"p{i}_center"].set(value=0.0, vary=False)
                try:
                    pr[f"p{i}_sigma"].set(
                        value=float(p.get("sigma", 10.0)), min=1e-3, max=1e4, vary=False
                    )
                except Exception:
                    pr[f"p{i}_sigma"].set(value=10.0, min=1e-3, max=1e4, vary=False)
                try:
                    pr[f"p{i}_fraction"].set(
                        value=float(p.get("fraction", 0.5)),
                        min=0.0,
                        max=1.0,
                        vary=False,
                    )
                except Exception:
                    pr[f"p{i}_fraction"].set(value=0.5, min=0.0, max=1.0, vary=False)
                try:
                    pr[f"p{i}_amplitude"].set(
                        value=max(0.0, float(p.get("amplitude", 1.0))), min=0.0
                    )
                except Exception:
                    pr[f"p{i}_amplitude"].set(value=1.0, min=0.0)
                if comp_model is None:
                    comp_model = m
                    params = pr
                else:
                    comp_model = comp_model + m
                    params.update(pr)
            if comp_model is None:
                return x_arr, y_arr, None
            try:
                y_fit = comp_model.eval(params, x=x_arr)
            except Exception:
                y_fit = None
            return x_arr, y_arr, y_fit
        except Exception:
            return x_arr, y_arr, None

    # ---------------------------- Build UI ------------------------------- #
    # Checkbox to include/exclude rows marked as bad quality
    include_bad_cb = widgets.Checkbox(
        value=False,
        description="Include bad spectra",
        indent=False,
        layout=widgets.Layout(width="auto"),
    )

    try:
        df_opts = FTIR_DataFrame.copy()
        if not include_bad_cb.value:
            try:
                df_opts = df_opts[_quality_good_mask(df_opts)]
            except Exception:
                pass
        try:
            df_opts = df_opts[df_opts["Normalized and Corrected Data"].notna()]
        except Exception:
            pass
        unique_materials = (
            sorted(
                {
                    str(v)
                    for v in df_opts.get("Material", pd.Series([], dtype=object))
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                }
            )
            if "Material" in df_opts.columns
            else []
        )
    except Exception:
        unique_materials = []
    if not unique_materials:
        raise ValueError("No materials found in DataFrame.")
    # Choose initial material from session defaults when available
    default_material = unique_materials[0]
    try:
        _sess_defaults = _get_session_defaults()
        _sess_mat = str(_sess_defaults.get("material", "any"))
        if _sess_mat in unique_materials:
            default_material = _sess_mat
    except Exception:
        pass
    try:
        dfm = FTIR_DataFrame[
            FTIR_DataFrame["Material"].astype(str) == str(default_material)
        ]
    except Exception:
        dfm = FTIR_DataFrame
    # Optionally exclude bad quality rows and require normalized data for condition options
    if not include_bad_cb.value:
        try:
            dfm = dfm[_quality_good_mask(dfm)]
        except Exception:
            pass
    try:
        dfm = dfm[dfm["Normalized and Corrected Data"].notna()]
    except Exception:
        pass
    cond_vals = [
        str(c)
        for c in dfm.get(cond_col, pd.Series([], dtype=object))
        .dropna()
        .astype(str)
        .unique()
        .tolist()
        if str(c).strip().lower() != "unexposed"
    ]
    cond_vals = sorted(cond_vals)
    # Choose initial conditions from session defaults when available
    default_condition = cond_vals[0] if cond_vals else None
    try:
        _sess_cond = str(_sess_defaults.get("conditions", "any"))  # reuse if set
    except Exception:
        _sess_cond = "any"
    if _sess_cond in cond_vals:
        default_condition = _sess_cond
    if not cond_vals:
        raise ValueError(
            f"No non-unexposed {cond_col} values found for material '{default_material}'."
        )

    material_dd = widgets.Dropdown(
        options=unique_materials,
        value=default_material,
        description="Material",
        layout=widgets.Layout(width="40%"),
    )
    conditions_dd = widgets.Dropdown(
        options=cond_vals,
        value=(default_condition if default_condition is not None else (cond_vals[0] if cond_vals else None)),
        description="Conditions",
        layout=widgets.Layout(width="40%"),
    )
    fit_btn = widgets.Button(
        description="Fit Time-Series",
        button_style="primary",
        layout=widgets.Layout(width="180px"),
        tooltip="Run time-series deconvolution for the selected Material/Conditions",
    )
    opt_btn = widgets.Button(
        description="Optimize",
        button_style="info",
        layout=widgets.Layout(width="180px"),
        tooltip=(
            "Iteratively optimize shared peak centers and per-spectrum amplitudes to "
            "minimize SSE against Normalized and Corrected Data"
        ),
    )
    close_btn = widgets.Button(description="Close", button_style="danger")
    status_html = widgets.HTML(value="")
    optimize_status_html = widgets.HTML(value="")
    # Tables: Peak Areas (per-time amplitudes) and Peak Wavenumbers (shared centers)
    A_table_html = widgets.HTML(value="")
    WN_table_html = widgets.HTML(value="")
    # Store the most recently built tables for optional export
    last_table_df = None  # areas
    last_centers_list = None  # wavenumbers (list[float])

    fig = go.FigureWidget()
    fig.update_layout(
        title="Time-Series Fit",
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="Absorbance (AU)",
        legend_title_text="Series",
    )

    def _plot_series(
        material_val=None, condition_val=None, with_fits=False, include_bad=None
    ):
        nonlocal last_table_df, last_centers_list
        try:
            fig.data = ()
        except Exception:
            pass
        if material_val is None:
            material_val = material_dd.value
        if condition_val is None:
            condition_val = conditions_dd.value
        if include_bad is None:
            include_bad = include_bad_cb.value
        df_series = _series_df(material_val, condition_val, include_bad=include_bad)
        if df_series.empty:
            try:
                status_html.value = (
                    "<span style='color:#a00;'>No spectra for this selection.</span>"
                )
            except Exception:
                pass
            try:
                A_table_html.value = ""
            except Exception:
                pass
            try:
                WN_table_html.value = ""
            except Exception:
                pass
            return
        # Only show final time-series fits (no raw data traces)
        has_any_fit = False
        for idx, row in df_series.iterrows():
            if not with_fits:
                continue
            x_f, _y_f, y_fit = _eval_timeseries_fit(row)
            if y_fit is None or x_f is None:
                continue
            has_any_fit = True
            t_val = row.get("Time")
            name = f"T={t_val}"
            try:
                fig.add_scatter(
                    x=x_f.tolist(),
                    y=(y_fit.tolist() if hasattr(y_fit, "tolist") else list(y_fit)),
                    mode="lines",
                    name=f"Fit {name}",
                    line=dict(width=2),
                )
            except Exception:
                pass
        # Build/update wavenumbers (shared centers), sigmas, alphas table and amplitude table
        try:
            if with_fits and has_any_fit:
                # Determine shared centers/sigmas/alphas from first available fit result
                centers_list = None
                sigmas_list = None
                fracs_list = None
                for _idx, _row in df_series.iterrows():
                    res0 = _row.get("Time-Series Fit Results")
                    if isinstance(res0, str):
                        try:
                            res0 = ast.literal_eval(res0)
                        except Exception:
                            res0 = None
                    if isinstance(res0, list) and len(res0) > 0:
                        try:
                            centers_list = [
                                float(p.get("center", float("nan"))) for p in res0
                            ]
                            sigmas_list = [
                                float(p.get("sigma", float("nan"))) for p in res0
                            ]
                            fracs_list = [
                                float(p.get("fraction", float("nan"))) for p in res0
                            ]
                        except Exception:
                            centers_list = None
                        break

                # If we have centers, render a compact table for peak wavenumbers
                if centers_list and len(centers_list) > 0:
                    try:
                        import pandas as pd  # local import safe here

                        # Build a table with rows: Center, Sigma, Alpha; columns: Peak 1..N
                        k = len(centers_list)
                        header = [f"Peak {i+1}" for i in range(k)]

                        def _fmt(v, fmt=".6g"):
                            try:
                                return f"{float(v):{fmt}}" if np.isfinite(v) else ""
                            except Exception:
                                return ""

                        rows = [
                            ["Center (cm⁻¹)"]
                            + [_fmt(centers_list[i]) for i in range(k)],
                            ["σ (cm⁻¹)"]
                            + [
                                _fmt(
                                    (
                                        sigmas_list[i]
                                        if sigmas_list and i < len(sigmas_list)
                                        else float("nan")
                                    )
                                )
                                for i in range(k)
                            ],
                            ["α"]
                            + [
                                _fmt(
                                    (
                                        fracs_list[i]
                                        if fracs_list and i < len(fracs_list)
                                        else float("nan")
                                    )
                                )
                                for i in range(k)
                            ],
                        ]
                        wn_df = pd.DataFrame(rows, columns=["Parameter"] + header)
                        html_wn = wn_df.to_html(index=False, escape=False)
                        WN_table_html.value = (
                            "<div style='margin-top:8px'><b>Peak Parameters</b></div>"
                            + f"<div style='max-height:180px; overflow:auto'>{html_wn}</div>"
                        )
                        last_centers_list = [
                            (
                                float(c)
                                if isinstance(c, (int, float)) and np.isfinite(c)
                                else float("nan")
                            )
                            for c in centers_list
                        ]
                    except Exception:
                        # Fallback manual HTML
                        k = len(centers_list or [])
                        header_cells_wn = "".join(
                            [f"<th>Parameter</th>"]
                            + [f"<th>Peak {i+1}</th>" for i in range(k)]
                        )

                        def _cell(v, fmt=".6g"):
                            try:
                                return (
                                    f"<td>{float(v):{fmt}}</td>"
                                    if np.isfinite(v)
                                    else "<td></td>"
                                )
                            except Exception:
                                return "<td></td>"

                        row_center = "".join(
                            ["<td>Center (cm⁻¹)</td>"]
                            + [_cell(c) for c in (centers_list or [])]
                        )
                        row_sigma = "".join(
                            ["<td>σ (cm⁻¹)</td>"]
                            + [
                                _cell(
                                    sigmas_list[i]
                                    if sigmas_list and i < len(sigmas_list)
                                    else float("nan")
                                )
                                for i in range(k)
                            ]
                        )
                        row_alpha = "".join(
                            ["<td>α</td>"]
                            + [
                                _cell(
                                    fracs_list[i]
                                    if fracs_list and i < len(fracs_list)
                                    else float("nan")
                                )
                                for i in range(k)
                            ]
                        )
                        WN_table_html.value = (
                            "<div style='margin-top:8px'><b>Peak Parameters</b></div>"
                            + "<div style='max-height:180px; overflow:auto'>"
                            + f"<table><thead><tr>{header_cells_wn}</tr></thead><tbody>"
                            + f"<tr>{row_center}</tr><tr>{row_sigma}</tr><tr>{row_alpha}</tr>"
                            + "</tbody></table>"
                            + "</div>"
                        )
                        try:
                            last_centers_list = [float(c) for c in (centers_list or [])]
                        except Exception:
                            last_centers_list = centers_list or []
                else:
                    WN_table_html.value = "<span style='color:#555;'>No peak parameters available to display for this selection.</span>"

                # Determine max number of peaks across available fits for areas table
                k_max = 0
                series_rows = []  # (label, amplitudes | None)
                for _idx, _row in df_series.iterrows():
                    label = f"T={_row.get('Time')}"
                    res = _row.get("Time-Series Fit Results")
                    if isinstance(res, str):
                        try:
                            res = ast.literal_eval(res)
                        except Exception:
                            res = None
                    if isinstance(res, list) and len(res) > 0:
                        try:
                            amps = [
                                float(p.get("amplitude", float("nan"))) for p in res
                            ]
                            k_max = max(k_max, len(amps))
                        except Exception:
                            amps = None
                    else:
                        amps = None
                    series_rows.append((label, amps))

                if k_max > 0:
                    try:
                        import pandas as pd  # local import safe here

                        table_records = []
                        for label, amps in series_rows:
                            rec = {"Series": label}
                            for i in range(k_max):
                                val = ""
                                try:
                                    if amps is not None and i < len(amps):
                                        a = amps[i]
                                        if np.isfinite(a):
                                            val = f"{a:.6g}"
                                except Exception:
                                    pass
                                rec[f"Peak {i+1}"] = val
                            table_records.append(rec)
                        df_table = pd.DataFrame(table_records)
                        html = df_table.to_html(index=False, escape=False)
                        A_table_html.value = (
                            "<div style='margin-top:8px'><b>Peak Areas</b></div>"
                            + f"<div style='max-height:300px; overflow:auto'>{html}</div>"
                        )
                        # Keep a copy for saving
                        try:
                            last_table_df = df_table.copy()
                        except Exception:
                            last_table_df = df_table
                    except Exception:
                        # Fallback manual HTML
                        header_cells = "".join(
                            [f"<th>Series</th>"]
                            + [f"<th>Peak {i+1}</th>" for i in range(k_max)]
                        )
                        body_rows = []
                        for label, amps in series_rows:
                            cells = [f"<td>{label}</td>"]
                            for i in range(k_max):
                                try:
                                    if (
                                        amps is not None
                                        and i < len(amps)
                                        and np.isfinite(amps[i])
                                    ):
                                        cells.append(f"<td>{amps[i]:.6g}</td>")
                                    else:
                                        cells.append("<td></td>")
                                except Exception:
                                    cells.append("<td></td>")
                            body_rows.append(f"<tr>{''.join(cells)}</tr>")
                        A_table_html.value = (
                            "<div style='margin-top:8px'><b>Peak Areas</b></div>"
                            + "<div style='max-height:300px; overflow:auto'>"
                            + f"<table><thead><tr>{header_cells}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
                            + "</div>"
                        )
                else:
                    A_table_html.value = "<span style='color:#555;'>No fitted peaks available to tabulate for this selection.</span>"
            else:
                A_table_html.value = "<span style='color:#555;'>Run 'Fit Time-Series' to populate the peak areas table.</span>"
                WN_table_html.value = "<span style='color:#555;'>Run 'Fit Time-Series' to populate the peak parameters table.</span>"
        except Exception:
            pass
        try:
            status_html.value = (
                "<span style='color:#555;'>Click 'Fit Time-Series' to compute and display fits for this selection.</span>"
                if not with_fits
                else "<span style='color:#000;'>Displayed time-series fits (amplitudes vary; centers/σ/α shared).</span>"
            )
        except Exception:
            pass

    def _on_material_change(*_):
        try:
            mat = material_dd.value
            dfm2 = FTIR_DataFrame[FTIR_DataFrame["Material"].astype(str) == str(mat)]
        except Exception:
            dfm2 = FTIR_DataFrame
        if not include_bad_cb.value:
            try:
                dfm2 = dfm2[_quality_good_mask(dfm2)]
            except Exception:
                pass
        try:
            dfm2 = dfm2[dfm2["Normalized and Corrected Data"].notna()]
        except Exception:
            pass
        new_conds = [
            str(c)
            for c in dfm2.get(cond_col, pd.Series([], dtype=object))
            .dropna()
            .astype(str)
            .unique()
            .tolist()
            if str(c).strip().lower() != "unexposed"
        ]
        new_conds = sorted(new_conds)
        try:
            conditions_dd.options = new_conds
            if new_conds:
                if conditions_dd.value not in new_conds:
                    conditions_dd.value = new_conds[0]
            _plot_series(
                mat,
                conditions_dd.value,
                with_fits=False,
                include_bad=include_bad_cb.value,
            )
        except Exception:
            pass

    def _on_conditions_change(*_):
        _plot_series(with_fits=False)

    def _on_fit_click(_b=None):
        try:
            status_html.value = (
                "<span style='color:#555;'>Running time-series fit...</span>"
            )
        except Exception:
            pass
        try:
            _compute_for_selection(
                material_dd.value, conditions_dd.value, include_bad=include_bad_cb.value
            )
        except Exception as e:
            try:
                status_html.value = f"<span style='color:#a00;'>Fit failed: {e}</span>"
            except Exception:
                pass
            return
        _plot_series(with_fits=True, include_bad=include_bad_cb.value)

    def _on_optimize_click(_b=None):
        try:
            status_html.value = "<span style='color:#555;'>Optimizing centers and amplitudes (this may take a moment)...</span>"
        except Exception:
            pass

        # Compute SSE for the current selection BEFORE optimization using current stored fits
        def _series_sse(material_val, condition_val):
            try:
                df_sel = _series_df(
                    material_val, condition_val, include_bad=include_bad_cb.value
                )
            except Exception:
                return float("nan")
            total = 0.0
            any_added = False
            for _idx, _row in df_sel.iterrows():
                try:
                    x_arr, y_arr, y_fit = _eval_timeseries_fit(_row)
                    if x_arr is None or y_arr is None or y_fit is None:
                        continue
                    # ensure arrays
                    y_fit_arr = np.asarray(y_fit, dtype=float)
                    y_arr = np.asarray(y_arr, dtype=float)
                    if y_fit_arr.shape != y_arr.shape:
                        n = min(y_fit_arr.size, y_arr.size)
                        if n <= 0:
                            continue
                        y_fit_arr = y_fit_arr[:n]
                        y_arr = y_arr[:n]
                    mask = np.isfinite(y_arr) & np.isfinite(y_fit_arr)
                    if not np.any(mask):
                        continue
                    err = y_arr[mask] - y_fit_arr[mask]
                    total += float(np.sum(err * err))
                    any_added = True
                except Exception:
                    continue
            return total if any_added else float("nan")

        try:
            sse_before_calc = _series_sse(material_dd.value, conditions_dd.value)
        except Exception:
            sse_before_calc = float("nan")
        try:
            result = _optimize_centers_for_selection(
                material_dd.value, conditions_dd.value, include_bad=include_bad_cb.value
            )
            # Backward compatibility if tuple size differs
            if isinstance(result, tuple) and len(result) == 3:
                msg, centers_out, summary = result
            elif isinstance(result, tuple) and len(result) == 2:
                msg, centers_out = result
                summary = None
            else:
                msg = str(result)
                centers_out = None
                summary = None
            if centers_out is None:
                status_html.value = f"<span style='color:#a00;'>{msg}</span>"
                try:
                    optimize_status_html.value = f"<div style='color:#a00'>{msg}</div>"
                except Exception:
                    pass
            else:
                status_html.value = f"<span style='color:#0a0;'>{msg}</span>"
                # Build a persistent, detailed optimization log
                try:
                    # Compute SSE AFTER optimization using updated fits stored in the DataFrame
                    try:
                        sse_after_calc = _series_sse(
                            material_dd.value, conditions_dd.value
                        )
                    except Exception:
                        sse_after_calc = float("nan")
                    if summary is None:
                        optimize_status_html.value = (
                            f"<div><b>Optimization</b>: {msg}</div>"
                        )
                    else:
                        # Compose a compact table of per-peak changes
                        centers0 = summary.get("initial_centers", []) or []
                        centers1 = summary.get("final_centers", []) or []
                        deltas = summary.get("center_deltas", []) or []
                        amp0 = summary.get("amp_initial_means", []) or []
                        amp1 = summary.get("amp_final_means", []) or []
                        ampd = summary.get("amp_delta_means", []) or []
                        rows = []
                        klen = max(len(centers0), len(centers1), len(deltas))
                        for i in range(klen):
                            c0 = centers0[i] if i < len(centers0) else None
                            c1 = centers1[i] if i < len(centers1) else None
                            dd = deltas[i] if i < len(deltas) else None

                            def _fmt(v):
                                try:
                                    return f"{float(v):.6g}"
                                except Exception:
                                    return ""

                            rows.append(
                                f"<tr><td>Peak {i+1}</td><td>{_fmt(c0)}</td><td>{_fmt(c1)}</td><td>{_fmt(dd)}</td></tr>"
                            )
                        table_html = (
                            "<table style='border-collapse:collapse'>"
                            "<thead><tr><th></th><th>Initial center</th><th>Final center</th><th>Δ center</th></tr></thead>"
                            f"<tbody>{''.join(rows)}</tbody></table>"
                        )
                        # Amplitude change table by spectrum (Δ per peak)
                        delta_by_spec = summary.get("amp_delta_by_spectrum", []) or []
                        # Determine max number of peaks to render
                        kmax = 0
                        for item in delta_by_spec:
                            try:
                                kmax = max(kmax, len(item.get("deltas", [])))
                            except Exception:
                                pass
                        # Build header and body rows
                        header_cells = "".join(
                            ["<th>Series</th>"]
                            + [f"<th>Peak {i+1} Δ</th>" for i in range(kmax)]
                        )
                        body_rows = []

                        def _fmtd(v):
                            try:
                                return f"{float(v):.6g}"
                            except Exception:
                                return ""

                        for item in delta_by_spec:
                            label = item.get("label", "")
                            deltas = item.get("deltas", []) or []
                            cells = [f"<td>{label}</td>"]
                            for i in range(kmax):
                                val = deltas[i] if i < len(deltas) else None
                                cells.append(f"<td>{_fmtd(val)}</td>")
                            body_rows.append(f"<tr>{''.join(cells)}</tr>")
                        amp_table_html = (
                            "<table style='border-collapse:collapse'>"
                            f"<thead><tr>{header_cells}</tr></thead>"
                            f"<tbody>{''.join(body_rows)}</tbody></table>"
                        )
                        # Prefer recomputed SSE for consistent selection and evaluation; fallback to optimizer-reported values
                        sse0 = (
                            sse_before_calc
                            if np.isfinite(sse_before_calc)
                            else summary.get("initial_sse")
                        )
                        if sse0 is None or not np.isfinite(sse0):
                            sse0 = None
                        sse1 = (
                            sse_after_calc
                            if np.isfinite(sse_after_calc)
                            else summary.get("final_sse")
                        )
                        if sse1 is None or not np.isfinite(sse1):
                            sse1 = None
                        try:
                            # Display delta as (final - initial): negative means decreased SSE
                            delta_disp = (
                                (sse1 - sse0)
                                if (sse0 is not None and sse1 is not None)
                                else None
                            )
                        except Exception:
                            delta_disp = None
                        passes_done = summary.get("passes")
                        m_att = summary.get("moves_attempted")
                        m_acc = summary.get("moves_accepted")
                        optimize_status_html.value = (
                            "<div style='margin-top:8px'><b>Optimization Summary</b></div>"
                            + (
                                f"<div>SSE: {sse0:.6g} → {sse1:.6g} (Δ={delta_disp:.6g}) | Passes: {passes_done} | Moves: {m_acc}/{m_att} accepted</div>"
                                if (sse0 is not None and sse1 is not None)
                                else f"<div>Passes: {passes_done} | Moves: {m_acc}/{m_att} accepted</div>"
                            )
                            + "<div style='color:#555; margin-top:2px'>(Amplitudes re-fit per spectrum during optimization)</div>"
                            + f"<div style='margin-top:6px'><b>Peak Centers</b></div>"
                            + f"<div style='margin-top:2px'>{table_html}</div>"
                            + f"<div style='margin-top:8px'><b>Amplitude Changes by Spectrum (Δ = final − initial)</b></div>"
                            + f"<div style='margin-top:2px'>{amp_table_html}</div>"
                        )
                except Exception:
                    pass
        except Exception as e:
            try:
                status_html.value = (
                    f"<span style='color:#a00;'>Optimize failed: {e}</span>"
                )
            except Exception:
                pass
            return

    _plot_series(with_fits=True, include_bad=include_bad_cb.value)

    def _on_save_click(_b=None):
        """Save per-row peak wavenumbers (centers), alphas (fractions), sigmas, and areas (amplitudes) into 'Time-Series Fit Results'."""
        try:
            mat = str(material_dd.value)
            cond = str(conditions_dd.value)
        except Exception:
            mat = "material"
            cond = "condition"
        # Build a fresh table to ensure we capture latest fits, independent of UI
        df_series = _series_df(mat, cond, include_bad=include_bad_cb.value)
        if df_series.empty:
            try:
                status_html.value = "<span style='color:#a00;'>Nothing to save for this selection.</span>"
            except Exception:
                pass
            return
        # Determine peak count and build records for areas, and collect centers
        k_max = 0
        series_rows = []  # (time, amplitudes | None)
        centers_list = None
        sigmas_list = None
        fracs_list = None
        for _idx, _row in df_series.iterrows():
            t_val = _row.get("Time")
            res = _row.get("Time-Series Fit Results")
            if isinstance(res, str):
                try:
                    res = ast.literal_eval(res)
                except Exception:
                    res = None
            if isinstance(res, list) and len(res) > 0:
                try:
                    amps = [float(p.get("amplitude", float("nan"))) for p in res]
                    k_max = max(k_max, len(amps))
                    if centers_list is None:
                        try:
                            centers_list = [
                                float(p.get("center", float("nan"))) for p in res
                            ]
                            sigmas_list = [
                                float(p.get("sigma", float("nan"))) for p in res
                            ]
                            fracs_list = [
                                float(p.get("fraction", float("nan"))) for p in res
                            ]
                        except Exception:
                            centers_list = None
                except Exception:
                    amps = None
            else:
                amps = None
            series_rows.append((t_val, amps))
        if k_max <= 0:
            try:
                status_html.value = "<span style='color:#a00;'>No fitted peaks to save for this selection.</span>"
            except Exception:
                pass
            return
        # Normalize and persist per-row results back into 'Time-Series Fit Results'
        try:
            dest_col = "Time-Series Fit Results"
            if dest_col not in FTIR_DataFrame.columns:
                try:
                    FTIR_DataFrame[dest_col] = None
                except Exception:
                    pass
            try:
                FTIR_DataFrame[dest_col] = FTIR_DataFrame[dest_col].astype(object)
            except Exception:
                pass

            updated = 0
            for idx, _row in df_series.iterrows():
                res = _row.get(dest_col)
                if isinstance(res, str):
                    try:
                        res = ast.literal_eval(res)
                    except Exception:
                        res = None
                if not isinstance(res, list) or len(res) == 0:
                    # If no result exists, but we have centers and series_rows amplitudes aligned by time
                    # attempt to construct a minimal record for this row
                    try:
                        t_val = _row.get("Time")
                        # find matching amplitudes for this time
                        amps = None
                        for t_it, a_it in series_rows:
                            if t_it == t_val:
                                amps = a_it
                                break
                        if centers_list and amps and len(centers_list) == len(amps):
                            res = []
                            for i in range(len(centers_list)):
                                try:
                                    c = float(centers_list[i])
                                except Exception:
                                    c = float("nan")
                                try:
                                    a = float(amps[i])
                                except Exception:
                                    a = float("nan")
                                res.append(
                                    {
                                        "amplitude": a,
                                        "center": c,
                                        # preserve optional keys with defaults or shared lists
                                        "sigma": (
                                            float(sigmas_list[i])
                                            if (sigmas_list and i < len(sigmas_list))
                                            else float("nan")
                                        ),
                                        "fraction": (
                                            float(fracs_list[i])
                                            if (fracs_list and i < len(fracs_list))
                                            else float("nan")
                                        ),
                                    }
                                )
                        else:
                            res = None
                    except Exception:
                        res = None

                # Clean/normalize the structure to ensure plain Python floats
                if isinstance(res, list) and len(res) > 0:
                    cleaned = []
                    for p in res:
                        try:
                            amp = float(p.get("amplitude", float("nan")))
                        except Exception:
                            amp = float("nan")
                        try:
                            cen = float(p.get("center", float("nan")))
                        except Exception:
                            cen = float("nan")
                        try:
                            sig = float(
                                p.get(
                                    "sigma",
                                    (
                                        sigmas_list[res.index(p)]
                                        if sigmas_list
                                        and res.index(p) < len(sigmas_list)
                                        else float("nan")
                                    ),
                                )
                            )
                        except Exception:
                            sig = float("nan")
                        try:
                            frac = float(
                                p.get(
                                    "fraction",
                                    (
                                        fracs_list[res.index(p)]
                                        if fracs_list and res.index(p) < len(fracs_list)
                                        else float("nan")
                                    ),
                                )
                            )
                        except Exception:
                            frac = float("nan")
                        cleaned.append(
                            {
                                "amplitude": amp,
                                "center": cen,
                                "sigma": sig,
                                "fraction": frac,
                            }
                        )
                    try:
                        FTIR_DataFrame.at[idx, dest_col] = cleaned
                        updated += 1
                    except Exception:
                        pass
            try:
                status_html.value = f"<span style='color:#0a0;'>Saved per-row peak parameters to '{dest_col}' for {updated} row(s).</span>"
            except Exception:
                pass
        except Exception as e:
            try:
                status_html.value = f"<span style='color:#a00;'>Failed to save per-row results: {e}</span>"
            except Exception:
                pass

    def _on_close(_b=None):
        try:
            material_dd.close()
            conditions_dd.close()
            fit_btn.close()
            close_btn.close()
            status_html.close()
            optimize_status_html.close()
            ui.close()
            fig.close()
        except Exception:
            pass

    def _refresh_materials_and_conditions():
        """Recompute material and condition options based on include_bad toggle and current data."""
        try:
            df_opts_local = FTIR_DataFrame.copy()
            if not include_bad_cb.value:
                try:
                    df_opts_local = df_opts_local[_quality_good_mask(df_opts_local)]
                except Exception:
                    pass
            try:
                df_opts_local = df_opts_local[
                    df_opts_local["Normalized and Corrected Data"].notna()
                ]
            except Exception:
                pass
            new_materials = (
                sorted(
                    {
                        str(v)
                        for v in df_opts_local.get(
                            "Material", pd.Series([], dtype=object)
                        )
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    }
                )
                if "Material" in df_opts_local.columns
                else []
            )
        except Exception:
            new_materials = []
        # Update materials dropdown
        try:
            cur_mat = (
                material_dd.value
                if material_dd.value in getattr(material_dd, "options", [])
                else None
            )
            material_dd.options = new_materials
            if not new_materials:
                return
            if cur_mat not in new_materials:
                material_dd.value = new_materials[0]
        except Exception:
            pass
        # Update conditions based on current material
        try:
            mat = material_dd.value
            dfm2 = FTIR_DataFrame[FTIR_DataFrame["Material"].astype(str) == str(mat)]
        except Exception:
            dfm2 = FTIR_DataFrame
        if not include_bad_cb.value:
            try:
                dfm2 = dfm2[_quality_good_mask(dfm2)]
            except Exception:
                pass
        try:
            dfm2 = dfm2[dfm2["Normalized and Corrected Data"].notna()]
        except Exception:
            pass
        new_conds = [
            str(c)
            for c in dfm2.get(cond_col, pd.Series([], dtype=object))
            .dropna()
            .astype(str)
            .unique()
            .tolist()
            if str(c).strip().lower() != "unexposed"
        ]
        new_conds = sorted(new_conds)
        try:
            cur_cond = (
                conditions_dd.value
                if conditions_dd.value in getattr(conditions_dd, "options", [])
                else None
            )
            conditions_dd.options = new_conds
            if new_conds:
                if cur_cond not in new_conds:
                    conditions_dd.value = new_conds[0]
        except Exception:
            pass
        # Re-plot
        _plot_series(with_fits=False, include_bad=include_bad_cb.value)

    def _on_include_bad_toggle(*_):
        _refresh_materials_and_conditions()

    material_dd.observe(_on_material_change, names="value")
    conditions_dd.observe(_on_conditions_change, names="value")
    # Persist Material/Conditions selections to session
    def _persist_ts_filters(_=None):
        try:
            _set_session_selection(material=material_dd.value, conditions=conditions_dd.value)
        except Exception:
            pass
    material_dd.observe(_persist_ts_filters, names="value")
    conditions_dd.observe(_persist_ts_filters, names="value")
    fit_btn.on_click(_on_fit_click)
    opt_btn.on_click(_on_optimize_click)
    include_bad_cb.observe(_on_include_bad_toggle, names="value")
    # Add Save button to save per-row results (wavenumbers + areas) back into 'Time-Series Fit Results'
    save_btn = widgets.Button(
        description="Save",
        button_style="success",
        layout=widgets.Layout(width="120px"),
        tooltip="Save per-row peak parameters to 'Time-Series Fit Results'",
    )
    save_btn.on_click(_on_save_click)
    close_btn.on_click(_on_close)

    # Keep dropdowns/checkbox in one row; buttons in a separate row
    controls = widgets.HBox([material_dd, conditions_dd, include_bad_cb])
    buttons = widgets.HBox([fit_btn, opt_btn, save_btn, close_btn])
    ui = widgets.VBox([controls, buttons, status_html, optimize_status_html])
    display(ui, fig, WN_table_html, A_table_html)
    # Ensure options reflect current include_bad state
    _refresh_materials_and_conditions()
    _plot_series(with_fits=False, include_bad=include_bad_cb.value)

    return FTIR_DataFrame


def display_DataFrame(FTIR_DataFrame, height: int = 500):
    """Display the DataFrame in a scrollable table with dropdown filters.

    - Defaults to showing the entire DataFrame in a scrollable HTML table.
    - Provides dropdowns for Material, Conditions/Condition, and Time to filter.

    Parameters
    ----------
    FTIR_DataFrame : pd.DataFrame
        The DataFrame to display.
    height : int, optional
        Height in pixels for the scrollable container (default 500).

    Returns
    -------
    pd.DataFrame
        The (unchanged) input DataFrame.
    """
    # Basic validation
    if FTIR_DataFrame is None:
        raise ValueError("FTIR_DataFrame must be provided.")

    # Local imports for HTML display to avoid relying on module-level imports
    try:
        from IPython.display import display as _ip_display, HTML as _ip_HTML
    except Exception:
        _ip_display = None
        _ip_HTML = None

    # Determine the conditions column name (supports either 'Conditions' or 'Condition')
    cond_col = (
        "Conditions"
        if "Conditions" in FTIR_DataFrame.columns
        else ("Condition" if "Condition" in FTIR_DataFrame.columns else None)
    )

    # Helper to build dropdown options with an 'All' entry
    def _options_for(colname):
        try:
            vals = (
                FTIR_DataFrame.get(colname, None).dropna().astype(str).unique().tolist()
                if colname in FTIR_DataFrame.columns
                else []
            )
            # Unique, sorted, stringified
            vals = sorted({str(v) for v in vals})
        except Exception:
            vals = []
        return ["All"] + vals

    # Special case: Time options sorted in descending numeric order (then non-numeric)
    def _options_for_time():
        try:
            if "Time" not in FTIR_DataFrame.columns:
                return ["All"]
            ser = FTIR_DataFrame.get("Time", pd.Series([], dtype=object)).dropna()
            # Work with string forms for stable filtering downstream
            str_vals = list({str(v) for v in ser.astype(str).tolist()})

            def _sort_key(s):
                try:
                    num = pd.to_numeric(pd.Series([s]), errors="coerce").iloc[0]
                except Exception:
                    num = float("nan")
                # Numeric first (ascending), then non-numeric (ascending lexicographic)
                if pd.isna(num):
                    return (1, s)
                return (0, float(num))

            sorted_vals = sorted(str_vals, key=_sort_key)
        except Exception:
            sorted_vals = []
        return ["All"] + sorted_vals

    # Build dropdowns
    mat_dd = widgets.Dropdown(
        options=_options_for("Material"),
        value="All",
        description="Material",
        layout=widgets.Layout(width="33%"),
        disabled=("Material" not in FTIR_DataFrame.columns),
    )
    cond_label = "Conditions" if cond_col != "Condition" else "Condition"
    cond_dd = widgets.Dropdown(
        options=_options_for(cond_col) if cond_col else ["All"],
        value="All",
        description=cond_label,
        layout=widgets.Layout(width="33%"),
        disabled=(cond_col is None),
    )
    time_dd = widgets.Dropdown(
        options=_options_for_time(),
        value="All",
        description="Time",
        layout=widgets.Layout(width="33%"),
        disabled=("Time" not in FTIR_DataFrame.columns),
    )
    # Quality filter dropdown (canonical column name)
    quality_dd = widgets.Dropdown(
        options=_options_for("Quality"),
        value="All",
        description="Quality",
        layout=widgets.Layout(width="33%"),
        disabled=("Quality" not in FTIR_DataFrame.columns),
    )

    # Output area for the HTML table
    out = widgets.Output()

    # Render helper
    def _render(*_):
        with out:
            out.clear_output()
            # Compose filter mask
            try:
                mask = pd.Series(True, index=FTIR_DataFrame.index)
            except Exception:
                # Fallback: no filtering if mask creation fails
                mask = None

            # Apply Material filter
            try:
                if not mat_dd.disabled and mat_dd.value != "All":
                    mask = mask & (
                        FTIR_DataFrame["Material"].astype(str) == str(mat_dd.value)
                    )
            except Exception:
                pass

            # Apply Conditions/Condition filter
            try:
                if cond_col and not cond_dd.disabled and cond_dd.value != "All":
                    mask = mask & (
                        FTIR_DataFrame[cond_col].astype(str) == str(cond_dd.value)
                    )
            except Exception:
                pass

            # Apply Time filter
            try:
                if not time_dd.disabled and time_dd.value != "All":
                    mask = mask & (
                        FTIR_DataFrame["Time"].astype(str) == str(time_dd.value)
                    )
            except Exception:
                pass

            # Apply Quality filter
            try:
                if not quality_dd.disabled and quality_dd.value != "All":
                    # Case-insensitive compare to be tolerant to stored values
                    mask = mask & (
                        FTIR_DataFrame["Quality"].astype(str).str.lower()
                        == str(quality_dd.value).lower()
                    )
            except Exception:
                pass

            try:
                df_view = (
                    FTIR_DataFrame[mask].copy()
                    if isinstance(mask, pd.Series)
                    else FTIR_DataFrame.copy()
                )
            except Exception:
                df_view = FTIR_DataFrame.copy()

            # Preserve and temporarily expand display options
            prev_rows = pd.get_option("display.max_rows")
            prev_cols = pd.get_option("display.max_columns")
            try:
                pd.set_option("display.max_rows", None)
                pd.set_option("display.max_columns", None)
                html = df_view.to_html(max_rows=None, max_cols=None, notebook=True)
            finally:
                try:
                    pd.set_option("display.max_rows", prev_rows)
                    pd.set_option("display.max_columns", prev_cols)
                except Exception:
                    pass

            if _ip_display is not None and _ip_HTML is not None:
                _ip_display(
                    _ip_HTML(
                        f"<div style='height:{int(height)}px;overflow:auto;'>{html}</div>"
                    )
                )

    # Wire up events
    mat_dd.observe(_render, names="value")
    cond_dd.observe(_render, names="value")
    time_dd.observe(_render, names="value")
    quality_dd.observe(_render, names="value")

    # Action buttons
    reset_btn = widgets.Button(
        description="Reset Filters",
        button_style="warning",
        layout=widgets.Layout(width="150px"),
        tooltip="Reset all filters to 'All'",
    )
    close_btn = widgets.Button(
        description="Close",
        button_style="danger",
        layout=widgets.Layout(width="120px"),
        tooltip="Close this view",
    )

    def _on_reset(_b=None):
        # Reset dropdowns to 'All' when available and re-render
        try:
            if not mat_dd.disabled and "All" in mat_dd.options:
                mat_dd.value = "All"
        except Exception:
            pass
        try:
            if not cond_dd.disabled and "All" in cond_dd.options:
                cond_dd.value = "All"
        except Exception:
            pass
        try:
            if not time_dd.disabled and "All" in time_dd.options:
                time_dd.value = "All"
        except Exception:
            pass
        try:
            if not quality_dd.disabled and "All" in quality_dd.options:
                quality_dd.value = "All"
        except Exception:
            pass
        # Ensure a refresh even if values were already 'All'
        _render()

    def _on_close(_b=None):
        # Close all widgets and clear output area
        try:
            mat_dd.close()
        except Exception:
            pass
        try:
            cond_dd.close()
        except Exception:
            pass
        try:
            time_dd.close()
        except Exception:
            pass
        try:
            quality_dd.close()
        except Exception:
            pass
        try:
            reset_btn.close()
        except Exception:
            pass
        try:
            close_btn.close()
        except Exception:
            pass
        try:
            out.clear_output()
            out.close()
        except Exception:
            pass
        try:
            ui.close()
        except Exception:
            pass

    reset_btn.on_click(_on_reset)
    close_btn.on_click(_on_close)

    # Layout and initial render
    controls = widgets.HBox([mat_dd, cond_dd, time_dd, quality_dd])
    buttons = widgets.HBox([reset_btn, close_btn])
    ui = widgets.VBox([controls, buttons, out])
    display(ui)
    _render()

    return FTIR_DataFrame


# Inline DataFrame display removed. Use display_DataFrame(FTIR_DataFrame) to view the
# table with scroll and filters inside a notebook environment.
