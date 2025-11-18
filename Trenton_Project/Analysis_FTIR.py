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
import ast
import html
import contextlib
import plotly.graph_objs as go
import ipywidgets as widgets
from IPython.display import clear_output, display
from math import ceil
from plotly.subplots import make_subplots
import json
from scipy.signal import find_peaks
import importlib
import threading
import time
from lmfit.models import PseudoVoigtModel

# Environment detection (used for some interactive behaviors)
try:
    import google.colab  # type: ignore
    _IN_COLAB = True
except Exception:
    _IN_COLAB = False

    
def _convert_dates_iso(directory: str, dry_run: bool = False):
    """Convert date substrings in folder and file names under ``directory`` to ISO
    format (YYYY-MM-DD). Recognizes MM-DD-YYYY (US) patterns and variants with
    separators '-', '_', '.', or space. Already-ISO dates are left unchanged.

    Parameters:
        directory: Root directory to scan.
        dry_run: If True, only print planned changes.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Regex to capture potential date tokens (month/day/year with varied separators)
    date_token = re.compile(r"\b\d{1,2}[-_\. ]\d{1,2}[-_\. ]\d{4}\b")

    def _normalize_token(token: str) -> str:
        # unify separators to '-'
        clean = re.sub(r"[-_\. ]", "-", token)
        parts = clean.split("-")
        if len(parts) != 3:
            return token
        a, b, c = parts
        # If already ISO (year first)
        if len(a) == 4:
            return f"{a}-{b.zfill(2)}-{c.zfill(2)}"  # ensure zero padding
        # Otherwise treat as month-day-year
        if len(c) == 4 and len(a) <= 2 and len(b) <= 2:
            # Basic bounds check
            try:
                m = int(a); d = int(b)
                if not (1 <= m <= 12 and 1 <= d <= 31):
                    return token
            except Exception:
                return token
            return f"{c}-{a.zfill(2)}-{b.zfill(2)}"
        return token

    def _rename_entry(parent: str, name: str) -> str:
        new_name = name
        for match in date_token.findall(name):
            iso = _normalize_token(match)
            if iso != match:
                print(f"Found date '{match}' -> '{iso}' in '{name}'")
                new_name = new_name.replace(match, iso)
        return new_name

    print("Converting date substrings to ISO format..." if not dry_run else "(dry-run) Simulating date conversion...")
    # Rename directories first so file paths remain valid
    for current_root, dirnames, filenames in os.walk(directory):
        # Directories
        for d in list(dirnames):
            new_d = _rename_entry(current_root, d)
            if new_d != d:
                old_path = os.path.join(current_root, d)
                new_path = os.path.join(current_root, new_d)
                if dry_run:
                    print(f"(dry-run) Would rename directory: {old_path} -> {new_path}")
                else:
                    print(f"Renaming directory: {old_path} -> {new_path}")
                    os.rename(old_path, new_path)
                # Update dirnames list for continued traversal
                try:
                    dir_index = dirnames.index(d)
                    dirnames[dir_index] = new_d
                except ValueError:
                    pass
        # Files
        for fname in filenames:
            new_fname = _rename_entry(current_root, fname)
            if new_fname != fname:
                old_fp = os.path.join(current_root, fname)
                new_fp = os.path.join(current_root, new_fname)
                if dry_run:
                    print(f"(dry-run) Would rename file: {old_fp} -> {new_fp}")
                else:
                    print(f"Renaming file: {old_fp} -> {new_fp}")
                    os.rename(old_fp, new_fp)
    print("Date renaming to ISO format complete." if not dry_run else "(dry-run) Date renaming simulation complete.")

def rename_files(
    directory=None,
    replace_spaces=None,
    iso_date_rename=None,
    file_rename=None,
    character_to_use=None,
    pairs_input=None,
    dry_run: bool = False,
):
    """Rename files (and folder/file dates) within a directory.

    Parameters (None prompts interactively):
        directory: root folder to scan.
        replace_spaces: replace spaces in filenames.
        iso_date_rename: convert date substrings to ISO.
        file_rename: perform old:new word replacements.
        character_to_use: replacement for spaces (default prompted when needed).
        pairs_input: comma-separated old:new pairs (prompted if needed).
        dry_run: when True, only print planned changes (no filesystem writes).

    Actions (each optional):
        - Replace spaces in filenames with chosen character.
        - Convert date substrings in folder & file names to ISO (YYYY-MM-DD).
        - Replace specified substrings via old:new pairs.
    """
    # Directory
    if directory is None:
        directory = input("Enter the directory to scan: ").strip()
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    print(f"Scanning directory: {directory}")

    # Replace spaces
    if replace_spaces is None:
        ans = input("Replace spaces in filenames? (y/n): ").strip().lower()
        replace_spaces = ans == "y"
    if replace_spaces:
        if character_to_use is None:
            character_to_use = input("Separator to use (e.g. _): ").strip() or "_"
        print("Replacing spaces in filenames...")
        for root, _dirs, files in os.walk(directory):
            for fname in files:
                if " " in fname:
                    old_fp = os.path.join(root, fname)
                    new_fname = fname.replace(" ", character_to_use)
                    new_fp = os.path.join(root, new_fname)
                    if dry_run:
                        print(f"(dry-run) Would rename: {old_fp} -> {new_fp}")
                    else:
                        print(f"Renaming: {old_fp} -> {new_fp}")
                        os.rename(old_fp, new_fp)
        print("Space replacement complete." if not dry_run else "(dry-run) Space replacement simulation complete.")
    else:
        print("Spaces will not be replaced.")

    # Date conversion
    if iso_date_rename is None:
        ans = input("Convert dates in names to ISO (YYYY-MM-DD)? (y/n): ").strip().lower()
        iso_date_rename = ans == "y"
    if iso_date_rename:
        _convert_dates_iso(directory, dry_run=dry_run)
    else:
        print("Date conversion skipped.")

    # Word replacement
    if file_rename is None:
        ans = input("Perform word replacements (old:new)? (y/n): ").strip().lower()
        file_rename = ans == "y"
    if file_rename:
        if pairs_input is None:
            pairs_input = input("Enter old:new pairs (comma-separated): ").strip()
        word_pairs = [p.split(":") for p in pairs_input.split(",") if ":" in p]
        print("Replacing specified substrings in filenames...")
        for root, _dirs, files in os.walk(directory):
            for fname in files:
                new_fname = fname
                for old, new in word_pairs:
                    new_fname = new_fname.replace(old, new)
                if new_fname != fname:
                    old_fp = os.path.join(root, fname)
                    new_fp = os.path.join(root, new_fname)
                    if dry_run:
                        print(f"(dry-run) Would rename: {old_fp} -> {new_fp}")
                    else:
                        print(f"Renaming: {old_fp} -> {new_fp}")
                        os.rename(old_fp, new_fp)
        print("Batch word replacement complete." if not dry_run else "(dry-run) Batch word replacement simulation complete.")
    else:
        print("Word replacement skipped.")


def extract_file_info(
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

    # --- Helper functions (scoped to extract_file_info) --- #
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

        Helps extract_file_info() create a structured DataFrame by extracting details
        from filenames and parent folder names.
        If "ignore" is in the filename, the file will be skipped.
        """
        # Info is first derived from parent folder names, then filenames if not found

        data = []
        grouped_files = {}
        # Consistent spaced ellipsis formatting
        print("Scanning directory for spectral files . . .")
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
        if track_replicates:
                # Build replicate groups including BOTH existing rows already present in
                # FTIR_DataFrame and any newly discovered rows in `data`.
                # This allows the replicate reporting to reflect the full dataset state,
                # not just files added in this invocation.
                replicate_groups = {}

                def _add_row_like(mat, cond, t, fname, fpath):
                    group_key = (mat, cond, t)
                    replicate_groups.setdefault(group_key, []).append(
                        (fname, os.path.basename(fpath))
                    )

                # 1. Existing DataFrame rows (if provided)
                try:
                    if FTIR_DataFrame is not None and len(FTIR_DataFrame) > 0:
                        # Only consider rows that have the required grouping columns
                        needed = {"Material", "Conditions", "Time", "File Name", "File Location"}
                        available = set(FTIR_DataFrame.columns)
                        if needed.issubset(available):
                            for _idx, _r in FTIR_DataFrame.iterrows():
                                _add_row_like(
                                    _r.get("Material"),
                                    _r.get("Conditions"),
                                    _r.get("Time"),
                                    _r.get("File Name"),
                                    _r.get("File Location"),
                                )
                except Exception:
                    pass

                # 2. Newly gathered rows (in-memory `data` list of dicts)
                for row in data:
                    _add_row_like(
                        row.get("Material"),
                        row.get("Conditions"),
                        row.get("Time"),
                        row.get("File Name"),
                        row.get("File Location"),
                    )

                # Emit only groups with more than one member (sorted by material, conditions, time)
                print("Replicate groups (>=2 files across existing + new):")
                any_groups = False
                def _group_sort_key(k):
                    m, c, t = k
                    def s(v):
                        return "" if v is None else str(v).lower()
                    def tkey(v):
                        if v is None:
                            return (1, "")
                        try:
                            return (0, float(v))
                        except Exception:
                            return (1, str(v).lower())
                    return (s(m), s(c), tkey(t))
                for group_key, file_list in sorted(replicate_groups.items(), key=lambda item: _group_sort_key(item[0])):
                    if len(file_list) > 1:
                        any_groups = True
                        formatted = [
                            f"{fname} (parent folder: {pfolder})" for fname, pfolder in file_list
                        ]
                        print(f"  {group_key}: {formatted}")
                if not any_groups:
                    print("  (No replicate groups found.)")

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

    # --- Enforce canonical column ordering (always run, even if no new data) ---
    try:
        existing_cols = list(FTIR_DataFrame.columns)
        desired_order = [
            "File Location",
            "File Name",
            "Date",
            "Conditions",
            "Material",
            "Time",
            "Quality",
            "X-Axis",
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
        ordered = [c for c in desired_order if c in existing_cols]
        others = [c for c in existing_cols if c not in ordered]
        FTIR_DataFrame = FTIR_DataFrame[ordered + others]
    except Exception:
        pass

    # Replicate tracking after ordering (so columns exist in final DataFrame form).
    if track_replicates and FTIR_DataFrame is not None and len(FTIR_DataFrame) > 0:
        try:
            needed = {"Material", "Conditions", "Time", "File Name", "File Location"}
            if needed.issubset(set(FTIR_DataFrame.columns)):
                replicate_groups_df = {}
                for _idx, _r in FTIR_DataFrame.iterrows():
                    key = (_r.get("Material"), _r.get("Conditions"), _r.get("Time"))
                    replicate_groups_df.setdefault(key, []).append(
                        (
                            _r.get("File Name"),
                            os.path.basename(_r.get("File Location", "")),
                        )
                    )
                print("Replicate groups in DataFrame (>=2 files):")
                any_df = False
                def _group_sort_key2(k):
                    m, c, t = k
                    def s(v):
                        return "" if v is None else str(v).lower()
                    def tkey(v):
                        if v is None:
                            return (1, "")
                        try:
                            return (0, float(v))
                        except Exception:
                            return (1, str(v).lower())
                    return (s(m), s(c), tkey(t))
                for gk, flist in sorted(replicate_groups_df.items(), key=lambda item: _group_sort_key2(item[0])):
                    if len(flist) > 1:
                        any_df = True
                        formatted = [
                            f"{fn} (parent folder: {pf})" for fn, pf in flist
                        ]
                        print(f"  {gk}: {formatted}")
                if not any_df:
                    print("  (No replicate groups found in DataFrame.)")
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
    """Return a normalized, lowercase quality Series; defaults to 'good' when missing."""
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
    """Boolean mask where True indicates rows not marked as 'bad'.

    Falls back to all True if the Quality column is missing or parsing fails.
    """
    try:
        qs = _quality_series(df)
        return qs != "bad"
    except Exception:
        return pd.Series([True] * len(df), index=getattr(df, "index", None))


# ------------------------ Session summary helpers ------------------------- #
def _session_summary_lines(changes: dict, *, context: str = ""):
    """Build concise summary lines from a per-session changes dict.

    The changes dict may contain keys like:
      - range_material: list[(material, count_rows, range_str)]
      - normalized_materials: list[(material, updated_count, skipped_count_or_None)]
      - saved_file: list[(idx, count)]
      - saved_filtered: int
      - quality: list[(idx, new_quality)]
      - saved: list[(idx, count_components)] (deconvolution)
      - iter: list[(idx, start_rc, final_rc, changes)] (deconvolution)

    Returns a list of human-friendly strings.
    """
    lines = []
    try:
        rm = changes.get("range_material") or []
        if rm:
            lines.append(
                "Saved normalization range for {} materials: {}".format(
                    len(rm), ", ".join([f"{m} ({c} rows)" for m, c, _ in rm])
                )
            )
    except Exception:
        pass
    try:
        nm = changes.get("normalized_materials") or []
        if nm:
            mats = [m for m, _u, _s in nm]
            lines.append("Normalized materials: " + ", ".join(mats))
    except Exception:
        pass
    # Baseline-correction: emit grouped summary by function with filenames/materials
    try:
        _ctx = str(context).lower().strip()
        if _ctx.startswith("baseline"):  # only for baseline_correct_spectra context
            bcm = changes.get("baseline_corrected_material") or []  # list[(material, function, updated_count)]
            bcf = changes.get("baseline_corrected_file") or []      # list[(idx, function[, filename])]

            # Normalize bcf tuples to (function, filename)
            per_file = []
            try:
                for t in bcf:
                    if len(t) >= 3:
                        _idx, fnc, fname = t[0], str(t[1]), str(t[2])
                    elif len(t) == 2:
                        _idx, fnc = t
                        fname = str(_idx)
                    else:
                        continue
                    per_file.append((fnc.upper(), fname))
            except Exception:
                pass

            # Group by function
            grouped = {}
            total = 0
            # Material-level summaries
            try:
                for mat, fnc, cnt in bcm:
                    fnc_u = str(fnc).upper()
                    grouped.setdefault(fnc_u, {"materials": set(), "files": []})
                    grouped[fnc_u]["materials"].add(str(mat))
                    try:
                        total += int(cnt)
                    except Exception:
                        pass
            except Exception:
                pass
            # File-level summaries
            try:
                for fnc_u, fname in per_file:
                    grouped.setdefault(fnc_u, {"materials": set(), "files": []})
                    grouped[fnc_u]["files"].append(str(fname))
                    total += 1
            except Exception:
                pass

            noun = "spectrum" if total == 1 else "spectra"
            lines.append(f"Baseline-corrected {total} {noun}:")
            # Emit per-function subheadings with items
            for fnc_u in sorted(grouped.keys()):
                lines.append(f"{fnc_u}:")
                items = []
                try:
                    if grouped[fnc_u]["materials"]:
                        for m in sorted(grouped[fnc_u]["materials"]):
                            items.append(f"  - {m}")
                    if grouped[fnc_u]["files"]:
                        for fname in grouped[fnc_u]["files"]:
                            items.append(f"  - {fname}")
                except Exception:
                    pass
                if not items:
                    items.append("  - (none)")
                lines.extend(items)
        else:
            # Non-baseline tools keep their original saved summaries
            try:
                sf = changes.get("saved_file") or []
                if sf:
                    head = ", ".join([f"{i}:{n}" for i, n in sf[:5]])
                    tail = " ..." if len(sf) > 5 else ""
                    lines.append(
                        f"Saved results for {len(sf)} spectra (first 5: {head}{tail})"
                    )
            except Exception:
                pass
            try:
                sfilt = int(changes.get("saved_filtered") or 0)
                if sfilt:
                    lines.append(f"Bulk-saved results for {sfilt} filtered spectra.")
            except Exception:
                pass
    except Exception:
        pass
    try:
        sv = changes.get("saved") or []
        if sv:
            head = ", ".join([f"{i}:{c}" for i, c in sv[:5]])
            tail = " ..." if len(sv) > 5 else ""
            lines.append(
                f"Saved deconvolution for {len(sv)} spectra (first 5: {head}{tail})"
            )
    except Exception:
        pass
    try:
        iters = changes.get("iter") or []
        if iters:
            lines.append(f"Ran iterative correction {len(iters)} time(s).")
    except Exception:
        pass
    try:
        qev = changes.get("quality") or []
        if qev:
            bad = [i for i, v in qev if str(v).lower() == "bad"]
            good = [i for i, v in qev if str(v).lower() == "good"]
            if bad:
                head = bad[:10]
                tail = " ..." if len(bad) > 10 else ""
                lines.append(f"Marked {len(bad)} spectra bad: {head}{tail}")
            if good:
                head = good[:10]
                tail = " ..." if len(good) > 10 else ""
                lines.append(f"Marked {len(good)} spectra good: {head}{tail}")
    except Exception:
        pass
    if not lines:
        lines.append("No DataFrame modifications were made in this session.")
    return lines


def _emit_session_summary(target, lines, *, title: str = "Session Summary"):
    """Emit summary lines into either an Output widget (msg_out) or an HTML widget.

    - target: ipywidgets.Output | ipywidgets.HTML
    - lines: list[str]
    - For Output: prints plain text lines.
    - For HTML: sets monospaced text with escaping.
    """
    try:
        from ipywidgets import Output, HTML
    except Exception:
        Output, HTML = None, None
    # Normalize title capitalization and add underline
    try:
        # Replace leading 'Session summary' (any case) with 'Session Summary'
        if title.lower().startswith("session summary"):
            # Preserve any suffix after the phrase (e.g. '(Normalization)')
            suffix = title[len("Session summary"):]
            title = "Session Summary" + suffix
        elif title.lower().startswith("session summary"):
            title = "Session Summary" + title[len("session summary"):]
    except Exception:
        pass
    underline = "-" * len(title)

    # Output
    try:
        if Output is not None and isinstance(target, Output):
            with target:
                clear_output(wait=True)
                print(title)
                print(underline)
                for line in lines:
                    print(" - " + str(line))
            return
    except Exception:
        pass
    # HTML
    try:
        if HTML is not None and isinstance(target, HTML):
            safe_lines = "\n".join([html.escape(str(l)) for l in lines])
            target.value = (
                "<div style='font-family:monospace; white-space:pre-wrap;'><strong>"
                + html.escape(title)
                + "</strong>\n"
                + html.escape(underline)
                + "\n"
                + safe_lines
                + "</div>"
            )
            return
    except Exception:
        pass
    # Fallback to print
    try:
        print(title)
        print(underline)
        for line in lines:
            print(" - " + str(line))
    except Exception:
        pass


# ----------------------- Session selection persistence ----------------------- #
# Persist last-used selections across interactive tools within this module.
_SESSION_SELECTIONS = {"material": "any", "conditions": "any", "time": "any"}
# ^ Persist last-used filter selections across interactive tools so a user's context
#   (material / conditions / time) carries between normalization, peak finding, etc.

# Track active widgets/figures created by baseline_correct_spectra (interactive) to ensure clean re-entry
_TB_WIDGETS = []
# ^ Bookkeeping list of active ipywidgets objects created by baseline_correct_spectra so they can
#   be cleanly closed before rebuilding the UI (prevents stale comm warnings).

# Persist the "Parameter Details" toggle state across baseline_correct_spectra rebuilds
_TB_PARAM_DETAILS_OPEN = False
# ^ Remembers whether the "Parameter Details" toggle was open in baseline_correct_spectra to keep
#   user preference when the parameter UI is dynamically rebuilt.

# Session change trackers for interactive tools
# Used by plot_spectra to collect per-session quality marks across separate plot UIs
_PLOT_SPECTRA_SESSION_CHANGES = None  # type: ignore[var-annotated]
# ^ Per-session change log (dict) for plot_spectra interactive mode. Collects
#   quality mark events so a summary can be rendered on Close.


# ----------------------- Reusable Quality Button Helper ----------------------- #
def _make_quality_controls(df, row_getter, *, margin="10px 10px 0 0"):
    """Return mutually exclusive quality buttons ("Mark spectrum as bad" / "Mark spectrum as good").

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a quality column (auto-detected or created).
    row_getter : Callable[[], pd.Series | None]
        Function returning the currently selected DataFrame row or None.
    margin : str, optional
        CSS margin applied to each button for consistent spacing.

    Returns
    -------
    (mark_bad_btn, mark_good_btn, refresh_fn)
        Widgets and a refresh function to sync visibility with current row state.

    Notes
    -----
    - If no row selected, only "Mark spectrum as bad" is shown (default assumption is good).
    - When a spectrum is marked bad, the bad button hides and the good button appears (and vice versa).
    - Quality column name resolved via _quality_column_name; created if missing.
    - Exceptions are swallowed for notebook UI resilience.
    """
    # Ensure sufficient width so full text is visible in all contexts
    _btn_width = "210px"
    mark_bad_btn = widgets.Button(
        description="Mark spectrum as bad",
        button_style="danger",
        layout=widgets.Layout(margin=margin, width=_btn_width),
    )
    mark_good_btn = widgets.Button(
        description="Mark spectrum as good",
        button_style="success",
        layout=widgets.Layout(margin=margin, width=_btn_width),
    )

    def _quality_col():
        try:
            return _quality_column_name(df)
        except Exception:
            return "Quality"

    def refresh():
        try:
            row = row_getter()
        except Exception:
            row = None
        # Re-fetch the latest row from the DataFrame so quality changes made via df.at
        # are visible immediately (selected_row Series objects are stale snapshots).
        try:
            if row is not None:
                row = df.loc[row.name]
        except Exception:
            pass
        # If no row selected, show only bad button
        if row is None:
            try:
                mark_bad_btn.layout.display = ""
                mark_good_btn.layout.display = "none"
            except Exception:
                pass
            return
        qcol = _quality_col()
        try:
            val = str(row.get(qcol, "good")).strip().lower()
        except Exception:
            val = "good"
        is_bad = val == "bad"
        try:
            mark_bad_btn.layout.display = "none" if is_bad else ""
            mark_good_btn.layout.display = "" if is_bad else "none"
        except Exception:
            pass

    def _set_quality(status):
        try:
            row = row_getter()
            if row is None:
                return
            qcol = _quality_col()
            df.at[row.name, qcol] = status
        except Exception:
            pass
        refresh()

    try:
        mark_bad_btn.on_click(lambda _b=None: _set_quality("bad"))
        mark_good_btn.on_click(lambda _b=None: _set_quality("good"))
    except Exception:
        pass

    refresh()
    return mark_bad_btn, mark_good_btn, refresh


# ----------------------- Quality dropdown helper (decoupling) ----------------------- #
def _quality_dropdown_handle(action, *, dropdown, include_bad_flag, idx, label_builder, observer_fn):
    """Remove or reinsert a dropdown option for a spectrum while keeping its plot visible.

    Parameters
    ----------
    action : str
        'bad' or 'good'.
    dropdown : ipywidgets.Dropdown
        Spectrum selection dropdown.
    include_bad_flag : bool
        Checkbox state indicating whether bad spectra are shown.
    idx : Any
        DataFrame index of spectrum.
    label_builder : Callable[[Any], str]
        Builds label string for reinsertion.
    observer_fn : Callable[[dict], None]
        Function registered via dropdown.observe; temporarily detached during mutation.

    Notes
    -----
    - Skips modification when include_bad_flag is True.
    - On 'bad': removes option matching idx if present and clears dropdown value if it was selected.
    - On 'good': reinserts option if missing and selects it.
    - All exceptions swallowed for resilience in interactive notebooks.
    """
    try:
        if dropdown is None or include_bad_flag:
            return
        opts = list(getattr(dropdown, 'options', []))
        if action == 'bad':
            try:
                dropdown.unobserve(observer_fn, names='value')
            except Exception:
                pass
            try:
                opts = [o for o in opts if not (isinstance(o, tuple) and o[1] == idx)]
                dropdown.options = opts
                if getattr(dropdown, 'value', None) == idx:
                    dropdown.value = None if opts else None
            except Exception:
                pass
            try:
                dropdown.observe(observer_fn, names='value')
            except Exception:
                pass
        elif action == 'good':
            ids = [o[1] for o in opts if isinstance(o, tuple)]
            if idx not in ids:
                try:
                    dropdown.unobserve(observer_fn, names='value')
                except Exception:
                    pass
                try:
                    label = label_builder(idx)
                except Exception:
                    label = f"Row {idx}"
                try:
                    # Reinsert option without changing current selection to avoid flicker
                    dropdown.options = opts + [(label, idx)] if opts else [(label, idx)]
                    # Do NOT set dropdown.value here; keep current selection stable
                except Exception:
                    pass
                try:
                    dropdown.observe(observer_fn, names='value')
                except Exception:
                    pass
    except Exception:
        pass


# ----------------------- Common Dataset/Parsing Helpers ----------------------- #
def _conditions_column_name(df):
    """Return the conditions column name if present: 'Conditions' | 'Condition' | None."""
    try:
        if df is None or not hasattr(df, "columns"):
            return None
        if "Conditions" in df.columns:
            return "Conditions"
        if "Condition" in df.columns:
            return "Condition"
    except Exception:
        pass
    return None


def _extract_material_condition_lists(df, *, exclude_unexposed=True):
    """Return (materials, conditions) lists from df with optional 'unexposed' filtering.

    - materials: sorted unique string values of 'Material' if present, else []
    - conditions: sorted unique string values of Conditions/Condition column if present, else []
    - exclude_unexposed: drop any condition equal to 'unexposed' (case-insensitive)
    """
    materials = []
    conditions = []
    if df is None or len(df) == 0:
        return materials, conditions
    try:
        if "Material" in df.columns:
            materials = sorted(
                {str(v) for v in df["Material"].dropna().astype(str).unique().tolist()}
            )
    except Exception:
        pass
    try:
        ccol = _conditions_column_name(df)
        if ccol and ccol in df.columns:
            vals = [str(v) for v in df[ccol].dropna().astype(str).unique().tolist()]
            if exclude_unexposed:
                vals = [v for v in vals if v.strip().lower() != "unexposed"]
            conditions = sorted(vals)
    except Exception:
        pass
    return materials, conditions


def _parse_seq(val):
    """Parse a value into a 1D list/array of floats, or return None if invalid.

    Accepts python-literal strings (e.g., "[1,2,3]") via _safe_literal_eval.
    """
    try:
        v = _safe_literal_eval(val, value_name="sequence")
    except Exception:
        v = val
    try:
        if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
            arr = np.asarray(v, dtype=float).ravel()
            if arr.size == 0:
                return None
            return arr
        return None
    except Exception:
        return None


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


def _processing_column_errors(df, want_baseline, want_baseline_corrected, want_normalized):
    """Return list of user-facing error messages only when there is ZERO usable data.

    Previous logic required all rows to be populated (flagged errors if ANY NaN / None
    existed). This prevented plotting partially processed DataFrames. Now we instead
    emit an error only if there are no valid (non-empty, non-NaN) entries in the
    requested column.
    """
    errors = []

    checks = [
        (want_baseline, "Baseline", "You need to baseline-correct the spectra before this will be available for plotting."),
        (want_baseline_corrected, "Baseline-Corrected Data", "You need to baseline-correct the spectra before this will be available for plotting."),
        (want_normalized, "Normalized and Corrected Data", "You need to normalize the spectra before this will be available for plotting."),
    ]

    for flag, col_name, msg in checks:
        if not flag:
            continue  # Column not requested for plotting
        if col_name not in df.columns:
            errors.append(msg)
            continue
        col = df[col_name]
        if not isinstance(col, pd.Series):
            errors.append(msg)
            continue

        usable = 0
        try:
            for v in col:
                if v is None:
                    continue
                # Accept list/tuple/ndarray with at least one non-NaN value
                if isinstance(v, (list, tuple)):
                    if len(v) == 0:
                        continue
                    if any(pd.isna(x) for x in v):
                        # Allow partially valid lists as long as at least one value is not NaN
                        if all(pd.isna(x) for x in v):
                            continue
                    usable += 1
                    continue
                try:
                    import numpy as np  # local import to avoid issues if numpy missing earlier
                    if isinstance(v, np.ndarray):
                        if v.size == 0:
                            continue
                        if np.isnan(v).all():
                            continue
                        usable += 1
                        continue
                except Exception:
                    pass
                # Scalar numeric
                if isinstance(v, (int, float)) and not pd.isna(v):
                    usable += 1
        except Exception:
            # On unexpected failure treat as no usable data
            usable = 0

        if usable == 0:
            errors.append(msg)

    return errors


def plot_spectra(
    FTIR_DataFrame,
    materials=None,
    conditions=None,
    times=None,
    raw_data=True,
    baseline=False,
    baseline_corrected=False,
    normalized=False,
    downsample=False,
    separate_plots=False,
    include_replicates=True,
    mark_bad=None,
    mark_good=None,
    show_bad=False,
    interactive=True,
    colorblind_mode=False,
):
    """
    Plot spectra grouped by material, condition, and time.

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
    downsample : bool, optional
        When True, adaptively decimate x/y to reduce points per trace (default False).
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
    # Interactive widget UI
    # - When interactive=True (default) or no filters are provided, build an ipywidgets
    #   control panel for filtering (Material/Conditions/Time), toggling which traces
    #   to show, and choosing separate vs grouped plots.
    # - This branch also tracks per-session quality changes and shows a Close summary.
    if interactive or (materials is None and conditions is None and times is None):
        try:
            # Initialize per-session change log (quality marks only for this UI)
            global _PLOT_SPECTRA_SESSION_CHANGES
            # Initialize or reuse per-session quality change log for interactive plotting
            if _PLOT_SPECTRA_SESSION_CHANGES is None:
                _PLOT_SPECTRA_SESSION_CHANGES = {"quality": []}
            else:
                # Reuse existing dict but don't wipe previous events until summary emitted on Close
                _PLOT_SPECTRA_SESSION_CHANGES.setdefault("quality", [])
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
            time_options = [("any", "any")] + [(str(v), v) for v in times_opts]
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
            downsample_cb = widgets.Checkbox(
                value=False if materials is None else bool(downsample),
                description="Downsample spectra",
                layout=widgets.Layout(width="auto"),
            )
            colorblind_cb = widgets.Checkbox(
                value=False,
                description="Colorblind mode",
                layout=widgets.Layout(width="auto"),
            )
            # Traces as a column (left block) with a subtle frame
            traces_col = widgets.VBox(
                [
                    widgets.HTML(value="<b>Traces</b>"),
                    raw_cb,
                    base_cb,
                    blc_cb,
                    norm_cb,
                ],
                layout=widgets.Layout(
                    border="1px solid #ddd",
                    padding="8px",
                    margin="0 10px 0 0",
                ),
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

            # Actions & summary area
            plot_button = widgets.Button(description="Plot", button_style="primary")
            close_button = widgets.Button(description="Close", button_style="danger")
            out = widgets.Output()  # persists session summary on Close

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
                        tr_down = bool(downsample_cb.value)
                        # --- Validate requested trace types within the CURRENT FILTERED SUBSET ---
                        def _valid_mask_for_col(df, col):
                            if df is None or len(df) == 0:
                                return pd.Series([False] * (0 if df is None else len(df)))
                            if col not in df.columns:
                                return pd.Series([False] * len(df), index=df.index)
                            vals = df[col]
                            valid = []
                            for v in vals:
                                # Try to parse sequences robustly (handles stringified lists, etc.)
                                seq = None
                                try:
                                    seq = _parse_seq(v)
                                except Exception:
                                    seq = None
                                if seq is None:
                                    # Fallback for already-sequence values
                                    try:
                                        import numpy as np
                                        if isinstance(v, np.ndarray):
                                            seq = v
                                        elif isinstance(v, (list, tuple)):
                                            seq = v
                                    except Exception:
                                        pass
                                if seq is not None:
                                    try:
                                        import numpy as np
                                        arr = np.asarray(seq, dtype=float).ravel()
                                        valid.append(bool(arr.size > 0 and not np.isnan(arr).all()))
                                        continue
                                    except Exception:
                                        try:
                                            ok = any((x is not None) and not pd.isna(x) for x in seq)
                                            valid.append(bool(ok))
                                            continue
                                        except Exception:
                                            valid.append(False)
                                            continue
                                # Scalar numeric fallback
                                if isinstance(v, (int, float)) and not pd.isna(v):
                                    valid.append(True)
                                else:
                                    valid.append(False)
                            return pd.Series(valid, index=vals.index)

                        # Build filtered subset for validation using current UI selections
                        try:
                            m_val = _materials_value()
                            c_val = _conditions_value()
                            t_val = _times_value()
                            dfv = FTIR_DataFrame
                            mask_val = pd.Series([True] * len(dfv))
                            # Respect quality filter
                            try:
                                if not bool(show_bad_chk.value):
                                    mask_val &= _quality_good_mask(dfv).values
                            except Exception:
                                pass
                            if isinstance(m_val, str) and m_val.strip().lower() != "any":
                                mats_list = [s.strip() for s in m_val.split(",") if s.strip()]
                                try:
                                    mask_val &= dfv["Material"].astype(str).isin(mats_list)
                                except Exception:
                                    mask_val &= dfv.get("Material", pd.Series([])).astype(str).isin(mats_list)
                            if isinstance(c_val, str) and c_val.strip().lower() != "any":
                                cond_list = [s.strip() for s in c_val.split(",") if s.strip()]
                                try:
                                    cond_mask = dfv["Conditions"].astype(str).isin(cond_list)
                                except Exception:
                                    cond_mask = dfv.get("Conditions", pd.Series([])).astype(str).isin(cond_list)
                                # If Time == 'any', include 'unexposed' across conditions
                                if isinstance(t_val, str) and t_val.strip().lower() == "any":
                                    try:
                                        cond_series = dfv["Conditions"].astype(str).str.lower()
                                        cond_mask = cond_mask | (cond_series == "unexposed")
                                    except Exception:
                                        cond_mask = cond_mask | (dfv.get("Conditions", pd.Series([])) == "unexposed")
                                mask_val &= cond_mask
                            if isinstance(t_val, str) and t_val.strip().lower() != "any":
                                t_list = []
                                for t in t_val.split(","):
                                    ts = t.strip()
                                    if not ts:
                                        continue
                                    try:
                                        t_list.append(int(ts))
                                    except Exception:
                                        t_list.append(ts)
                                try:
                                    mask_val &= dfv["Time"].isin(t_list)
                                except Exception:
                                    mask_val &= dfv.get("Time", pd.Series([])).isin(t_list)
                            filtered_val = dfv[mask_val]
                        except Exception:
                            filtered_val = FTIR_DataFrame

                        validation_errors = []
                        try:
                            if tr_base and not _valid_mask_for_col(filtered_val, "Baseline").any():
                                validation_errors.append(
                                    "You need to baseline-correct the spectra before this will be available for plotting."
                                )
                            if tr_blc and not _valid_mask_for_col(filtered_val, "Baseline-Corrected Data").any():
                                validation_errors.append(
                                    "You need to baseline-correct the spectra before this will be available for plotting."
                                )
                            if tr_norm and not _valid_mask_for_col(filtered_val, "Normalized and Corrected Data").any():
                                validation_errors.append(
                                    "You need to normalize the spectra before this will be available for plotting."
                                )
                        except Exception:
                            validation_errors = []  # fail open on unexpected issues

                        if validation_errors:
                            for msg in validation_errors:
                                try:
                                    display(widgets.HTML(value=f"<b>Error:</b> {msg}"))
                                except Exception:
                                    print(f"Error: {msg}")
                            return
                        # Prepare a plotting function we can call after any warnings
                        def _do_plot():
                            try:
                                plot_spectra(
                                    FTIR_DataFrame=FTIR_DataFrame,
                                    materials=_materials_value(),
                                    conditions=_conditions_value(),
                                    times=_times_value(),
                                    raw_data=tr_raw,
                                    baseline=tr_base,
                                    baseline_corrected=tr_blc,
                                    normalized=tr_norm,
                                    downsample=tr_down,
                                    separate_plots=separate_plots_chk.value,
                                    include_replicates=include_replicates_chk.value,
                                    show_bad=show_bad_chk.value,
                                    interactive=False,
                                    colorblind_mode=bool(colorblind_cb.value),
                                )
                            except Exception as e:
                                with out:
                                    print(f"Error while plotting: {e}")

                        # Estimate how many spectra will be plotted with current filters
                        try:
                            m_val = _materials_value()
                            c_val = _conditions_value()
                            t_val = _times_value()
                            df = FTIR_DataFrame
                            mask_est = pd.Series([True] * len(df))
                            # Respect quality filter
                            try:
                                if not bool(show_bad_chk.value):
                                    mask_est &= _quality_good_mask(df).values
                            except Exception:
                                pass
                            if isinstance(m_val, str) and m_val.strip().lower() != "any":
                                mats_list = [s.strip() for s in m_val.split(",") if s.strip()]
                                try:
                                    mask_est &= df["Material"].astype(str).isin(mats_list)
                                except Exception:
                                    mask_est &= df.get("Material", pd.Series([])).astype(str).isin(mats_list)
                            if isinstance(c_val, str) and c_val.strip().lower() != "any":
                                cond_list = [s.strip() for s in c_val.split(",") if s.strip()]
                                try:
                                    cond_mask = df["Conditions"].astype(str).isin(cond_list)
                                except Exception:
                                    cond_mask = df.get("Conditions", pd.Series([])).astype(str).isin(cond_list)
                                # If Time == 'any', include 'unexposed' across conditions
                                if isinstance(t_val, str) and t_val.strip().lower() == "any":
                                    try:
                                        cond_series = df["Conditions"].astype(str).str.lower()
                                        cond_mask = cond_mask | (cond_series == "unexposed")
                                    except Exception:
                                        cond_mask = cond_mask | (df.get("Conditions", pd.Series([])) == "unexposed")
                                mask_est &= cond_mask
                            if isinstance(t_val, str) and t_val.strip().lower() != "any":
                                t_list = []
                                for t in t_val.split(","):
                                    ts = t.strip()
                                    if not ts:
                                        continue
                                    try:
                                        t_list.append(int(ts))
                                    except Exception:
                                        t_list.append(ts)
                                try:
                                    mask_est &= df["Time"].isin(t_list)
                                except Exception:
                                    mask_est &= df.get("Time", pd.Series([])).isin(t_list)
                            filtered_est = df[mask_est]
                            if not bool(include_replicates_chk.value) and not filtered_est.empty:
                                try:
                                    filtered_est = filtered_est.sort_values(by=["Material", "Conditions", "Time"]).drop_duplicates(
                                        subset=["Material", "Conditions", "Time"], keep="first"
                                    )
                                except Exception:
                                    pass
                            match_count = int(len(filtered_est))
                        except Exception:
                            match_count = 0

                        # Choose thresholds (lower in Colab)
                        warn_threshold = 40 if _IN_COLAB else 75

                        if match_count > warn_threshold:
                            # Build a warning prompt with proceed/cancel options
                            try:
                                # Disable original Plot button (grey out) while warning is active
                                try:
                                    plot_button.disabled = True
                                    plot_button.button_style = ''  # remove primary styling
                                    plot_button.description = 'Plot (warning active)'
                                except Exception:
                                    pass
                                msg = (
                                    f"This selection will plot {match_count} spectra"
                                    + (" as separate figures" if bool(separate_plots_chk.value) else "")
                                    + ". This may be slow or cause a crash."
                                )
                                tips = (
                                    "Tips: narrow filters, disable replicates, disable separate plots, or enable the downsample option."
                                )
                                display(
                                    widgets.VBox(
                                        [
                                            widgets.HTML(
                                                value=f"<b>Warning:</b> {msg}<br><i>{tips}</i>"
                                            ),
                                            widgets.HBox(
                                                [
                                                    widgets.Button(
                                                        description="Plot Anyways",
                                                        button_style="primary",
                                                        layout=widgets.Layout(margin="5px 10px 5px 0"),
                                                    ),
                                                    widgets.Button(
                                                        description="Redo Selection",
                                                        button_style="warning",
                                                        layout=widgets.Layout(margin="5px 0 5px 0"),
                                                    ),
                                                ]
                                            ),
                                        ]
                                    )
                                )

                                # Wire up buttons after creation
                                # Need references; rebuild HBox to capture them
                                proceed_btn = widgets.Button(
                                    description="Plot Anyways", button_style="primary"
                                )
                                redo_btn = widgets.Button(
                                    description="Redo Selection", button_style="warning"
                                )
                                # Re-render with wired instances and replace previous prompt
                                clear_output(wait=True)
                                display(
                                    widgets.VBox(
                                        [
                                            widgets.HTML(
                                                value=f"<b>Warning:</b> {msg}<br><i>{tips}</i>"
                                            ),
                                            widgets.HBox([proceed_btn, redo_btn]),
                                        ]
                                    )
                                )

                                def _proceed(_b=None):
                                    with out:
                                        clear_output(wait=True)
                                        # Re-enable Plot button
                                        try:
                                            plot_button.disabled = False
                                            plot_button.button_style = 'primary'
                                            plot_button.description = 'Plot'
                                        except Exception:
                                            pass
                                        _do_plot()

                                def _redo(_b=None):
                                    with out:
                                        clear_output(wait=True)
                                        # Re-enable Plot button so user can adjust and plot again
                                        try:
                                            plot_button.disabled = False
                                            plot_button.button_style = 'primary'
                                            plot_button.description = 'Plot'
                                        except Exception:
                                            pass
                                        display(
                                            widgets.HTML(
                                                value="Selection not plotted. Adjust filters above and click Plot."
                                            )
                                        )

                                proceed_btn.on_click(_proceed)
                                redo_btn.on_click(_redo)
                                return
                            except Exception:
                                # If warning UI fails, fall back to plotting directly
                                pass

                        # If below threshold (or warning failed), plot directly
                        _do_plot()
                        # After plotting separate plots, info: individual mark buttons
                        # append to the global session dict in the non-interactive branch below.
                        try:
                            if separate_plots_chk.value:
                                pass
                        except Exception:
                            pass
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
                    # Emit session summary before closing if any quality changes occurred
                    global _PLOT_SPECTRA_SESSION_CHANGES
                    if isinstance(_PLOT_SPECTRA_SESSION_CHANGES, dict):
                        lines = _session_summary_lines(
                            _PLOT_SPECTRA_SESSION_CHANGES,
                            context="plot_spectra",
                        )
                        # Use the existing output widget if still present; else fallback to print.
                        try:
                            _emit_session_summary(out, lines, title="Session Summary (Plot Spectra)")
                        except Exception:
                            print("plot_spectra session summary:")
                            for l in lines:
                                print(" - " + str(l))
                        # Reset tracker for next session
                        _PLOT_SPECTRA_SESSION_CHANGES = {"quality": []}
                except Exception:
                    pass
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
                    # Trace checkboxes and their column container
                    raw_cb,
                    base_cb,
                    blc_cb,
                    norm_cb,
                    downsample_cb,
                    traces_col,
                    options_col,
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
                    "selectors_placeholder",  # will be swapped below if defined
                    "toggles_placeholder",
                    "controls_placeholder",
                ]:
                    # Replace placeholder strings with actual widget objects if they exist
                    if w == "selectors_placeholder":
                        w = selectors
                    elif w == "toggles_placeholder":
                        w = toggles
                    elif w == "controls_placeholder":
                        w = controls
                    if w is None:
                        continue
                    try:
                        # Keep 'out' visible to preserve the session summary
                        if w is out:
                            continue
                        w.close()
                    except Exception:
                        pass

            close_button.on_click(_on_close)

            # Layout and display the controls and the summary output area (out)
            selectors = widgets.HBox([materials_dd, conditions_dd, times_dd])
            # Options as a column (right block) with a subtle frame
            options_col = widgets.VBox(
                [
                    widgets.HTML(value="<b>Options</b>"),
                    separate_plots_chk,
                    include_replicates_chk,
                    show_bad_chk,
                    colorblind_cb,
                    downsample_cb,
                ],
                layout=widgets.Layout(
                    border="1px solid #ddd",
                    padding="8px",
                ),
            )
            toggles = widgets.HBox([traces_col, options_col])
            controls = widgets.VBox(
                [
                    selectors,
                    toggles,
                    widgets.HBox([plot_button, close_button]),
                ]
            )
            display(controls, out)
            return
        except Exception as e:
            # If widgets are unavailable or something fails, fall back to non-interactive path with a note
            try:
                print(
                    f"Interactive controls unavailable, falling back to static plot: {e}"
                )
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

    # Validate within filtered_data; show descriptive errors only if zero usable rows
    def _valid_mask_for_col(df, col):
        if df is None or len(df) == 0:
            return pd.Series([False] * (0 if df is None else len(df)))
        if col not in df.columns:
            return pd.Series([False] * len(df), index=df.index)
        vals = df[col]
        valid = []
        for v in vals:
            seq = None
            try:
                seq = _parse_seq(v)
            except Exception:
                seq = None
            if seq is None:
                try:
                    import numpy as np
                    if isinstance(v, np.ndarray):
                        seq = v
                    elif isinstance(v, (list, tuple)):
                        seq = v
                except Exception:
                    pass
            if seq is not None:
                try:
                    import numpy as np
                    arr = np.asarray(seq, dtype=float).ravel()
                    valid.append(bool(arr.size > 0 and not np.isnan(arr).all()))
                    continue
                except Exception:
                    try:
                        ok = any((x is not None) and not pd.isna(x) for x in seq)
                        valid.append(bool(ok))
                        continue
                    except Exception:
                        valid.append(False)
                        continue
            if isinstance(v, (int, float)) and not pd.isna(v):
                valid.append(True)
            else:
                valid.append(False)
        return pd.Series(valid, index=vals.index)

    noninteractive_validation_errors = []
    try:
        if baseline and not _valid_mask_for_col(filtered_data, "Baseline").any():
            noninteractive_validation_errors.append(
                "You need to baseline-correct the spectra before this will be available for plotting."
            )
        if baseline_corrected and not _valid_mask_for_col(filtered_data, "Baseline-Corrected Data").any():
            noninteractive_validation_errors.append(
                "You need to baseline-correct the spectra before this will be available for plotting."
            )
        if normalized and not _valid_mask_for_col(filtered_data, "Normalized and Corrected Data").any():
            noninteractive_validation_errors.append(
                "You need to normalize the spectra before this will be available for plotting."
            )
    except Exception:
        noninteractive_validation_errors = []

    if noninteractive_validation_errors:
        for msg in noninteractive_validation_errors:
            try:
                display(widgets.HTML(value=f"<b>Error:</b> {msg}"))
            except Exception:
                print(f"Error: {msg}")
        return

    # Row-level warnings (non-blocking): enumerate rows in filtered subset lacking required data
    try:
        row_level_messages = []
        # Build per-trace validity masks only for requested trace types
        trace_specs = [
            (baseline, "Baseline"),
            (baseline_corrected, "Baseline-Corrected Data"),
            (normalized, "Normalized and Corrected Data"),
        ]
        # Precompute validity masks (avoid re-parsing column multiple times)
        validity_cache = {}
        for flag, col_name in trace_specs:
            if not flag:
                continue
            try:
                validity_cache[col_name] = _valid_mask_for_col(filtered_data, col_name)
            except Exception:
                validity_cache[col_name] = pd.Series([False] * len(filtered_data), index=filtered_data.index)

        for idx, row in filtered_data.iterrows():
            mat = str(row.get("Material", "NA"))
            cond = str(row.get("Conditions", row.get("Condition", "NA")))
            t_val = row.get("Time", "NA")
            for flag, col_name in trace_specs:
                if not flag:
                    continue
                mask_series = validity_cache.get(col_name)
                is_valid = False
                try:
                    is_valid = bool(mask_series.loc[idx])
                except Exception:
                    is_valid = False
                if not is_valid:
                    # Generic user-facing guidance (no granular reason details)
                    advice_map = {
                        "Baseline": "You need to baseline-correct the spectra before this will be available for plotting.",
                        "Baseline-Corrected Data": "You need to baseline-correct the spectra before this will be available for plotting.",
                        "Normalized and Corrected Data": "You need to normalize the spectra before this will be available for plotting.",
                    }
                    advice = advice_map.get(col_name, "Required processing step missing.")
                    row_level_messages.append(
                        f"Row {idx} (Material={mat}, Conditions={cond}, Time={t_val}) missing {col_name}: {advice}"
                    )
        if row_level_messages:
            # Present as a collapsible-ish block; simple HTML formatting
            try:
                warn_html = (
                    "<div style='border:1px solid #e0a800;padding:8px;margin:6px 0;background:#fffbe6'>"
                    "<b>Warning:</b> Some requested processed traces are missing for specific rows.<br>"
                    + "<br>".join(row_level_messages)
                    + "</div>"
                )
                display(widgets.HTML(value=warn_html))
            except Exception:
                print("Warning: Some requested processed traces are missing for specific rows:")
                for m in row_level_messages:
                    print(" - " + m)
    except Exception:
        # Swallow any unexpected issues; plotting should continue
        pass

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

    # Helper: adaptive stride to cap points per trace when downsampling
    def _stride_for(n):
        try:
            n = int(n)
        except Exception:
            return 1
        target = 2000
        try:
            from math import ceil as _ceil

            return max(1, _ceil(n / float(target)))
        except Exception:
            return 1

    # Precompute a consistent color (and dash) per spectrum (row) to reuse across plots
    try:
        from plotly.colors import qualitative as _qual

        if bool(colorblind_mode):
            # Favor colorblind-friendly palettes; combine for more distinct colors
            base_palette = []
            try:
                base_palette += list(_qual.Safe)
            except Exception:
                pass
            try:
                base_palette += list(_qual.G10)
            except Exception:
                pass
            try:
                base_palette += list(_qual.T10)
            except Exception:
                pass
            _palette = base_palette if base_palette else list(_qual.Safe)
        else:
            # Larger default palette: Dark24 + Alphabet (~50 colors)
            base_palette = []
            try:
                base_palette += list(_qual.Dark24)
            except Exception:
                pass
            try:
                base_palette += list(_qual.Alphabet)
            except Exception:
                pass
            _palette = base_palette if base_palette else list(_qual.Plotly)
    except Exception:
        # Fallback basic palette
        _palette = [
            "#636EFA",
            "#EF553B",
            "#00CC96",
            "#AB63FA",
            "#FFA15A",
            "#19D3F3",
            "#FF6692",
            "#B6E880",
            "#FF97FF",
            "#FECB52",
        ]

    _dashes = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
    idx_list = filtered_data_sorted.index.tolist()
    _row_line = {}
    for j, i in enumerate(idx_list):
        color = _palette[j % len(_palette)] if len(_palette) > 0 else None
        dash = _dashes[(j // max(1, len(_palette))) % len(_dashes)]
        _row_line[i] = {k: v for k, v in (("color", color), ("dash", dash)) if v is not None}

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
                x_list = list(x_axis)
                y_list = list(y_v)
                if downsample:
                    s = _stride_for(len(x_list))
                    if s > 1:
                        x_list = x_list[::s]
                        y_list = y_list[::s]
                fig_group.add_scatter(
                    x=x_list,
                    y=y_list,
                    mode="lines",
                    name=f"{name_suffix}: {spectrum_label}",
                    line=_row_line.get(idx),
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
        # Place legend on the right to avoid overlapping x-axis title
        legend=dict(
            orientation="v",
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
        ),
        margin=dict(r=140),
    )
    display(fig_group)

    # Display plotted file locations in the exact order they were plotted
    try:
        file_paths = []
        for _i, _row in filtered_data_sorted.iterrows():
            folder = str(_row.get("File Location", "") or "")
            fname = str(_row.get("File Name", "") or "")
            full_path = (
                os.path.join(folder, fname) if folder and fname else (folder or fname)
            )
            if full_path:
                file_paths.append(full_path)
        if file_paths:
            header = f"Plotted file order ({len(file_paths)}):"
            html = "<b>{}</b><br><pre style='margin:0'>{}</pre>".format(
                header, "\n".join(f"{i+1}. {p}" for i, p in enumerate(file_paths))
            )
            display(widgets.HTML(value=html))
    except Exception as e:
        try:
            print(f"Note: could not display file list: {e}")
        except Exception:
            pass

    # Optional: mark selected rows as good/bad after plotting
    try:
        qcol = _quality_column_name(FTIR_DataFrame)

        # Local function to append quality change to session log
        def _log_quality(idx_val, status_val):
            try:
                global _PLOT_SPECTRA_SESSION_CHANGES
                if isinstance(_PLOT_SPECTRA_SESSION_CHANGES, dict):
                    _PLOT_SPECTRA_SESSION_CHANGES.setdefault(
                        "quality", []
                    ).append((idx_val, status_val))
            except Exception:
                pass

        # Mark good first
        if mark_good is not None and not filtered_data.empty:
            if isinstance(mark_good, str) and mark_good.strip().lower() == "all":
                FTIR_DataFrame.loc[filtered_data.index, qcol] = "good"
                for _i in filtered_data.index.tolist():
                    _log_quality(_i, "good")
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
                    for _i in set(to_mark_idx):
                        _log_quality(_i, "good")
        # Then mark bad
        if mark_bad is not None and not filtered_data.empty:
            if isinstance(mark_bad, str) and mark_bad.strip().lower() == "all":
                FTIR_DataFrame.loc[filtered_data.index, qcol] = "bad"
                for _i in filtered_data.index.tolist():
                    _log_quality(_i, "bad")
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
                    for _i in set(to_mark_idx):
                        _log_quality(_i, "bad")
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
                    x_list = list(x_axis)
                    y_list = list(y_v)
                    if downsample:
                        s = _stride_for(len(x_list))
                        if s > 1:
                            x_list = x_list[::s]
                            y_list = y_list[::s]
                    fig_i.add_scatter(
                        x=x_list,
                        y=y_list,
                        mode="lines",
                        name=name_suffix,
                        line=_row_line.get(idx),
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

            # Unified quality controls via shared helper (closure captures this row index)
            mark_bad_btn, mark_good_btn, _refresh_row_btns = _make_quality_controls(
                FTIR_DataFrame, lambda i=idx: FTIR_DataFrame.loc[i]
            )
            # Add logging callbacks (helper already updates DataFrame & toggles buttons)
            try:
                mark_bad_btn.on_click(lambda _b=None, i=idx: _log_quality(i, "bad"))
                mark_good_btn.on_click(lambda _b=None, i=idx: _log_quality(i, "good"))
            except Exception:
                pass
            # Ensure initial visibility reflects current row status (helper ran refresh once)
            try:
                _refresh_row_btns()
            except Exception:
                pass
            # Display figure with its quality mark buttons BELOW the plot inside a bordered container
            try:
                container = widgets.VBox(
                    [
                        fig_i,
                        widgets.HBox([mark_bad_btn, mark_good_btn], layout=widgets.Layout(margin="4px 0 0 0")),
                    ],
                    layout=widgets.Layout(
                        border="1px solid #ccc",
                        padding="8px",
                        margin="6px 0",
                    ),
                )
                display(container)
            except Exception:
                # Fallback: show buttons with upward arrow indicating association if styling fails
                mark_bad_btn.description = "Mark spectrum as bad ↑"
                mark_good_btn.description = "Mark spectrum as good ↑"
                display(widgets.VBox([fig_i, widgets.HBox([mark_bad_btn, mark_good_btn])]))


def baseline_correct_spectra(
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

    # Per-session change log for this baseline_correct_spectra (interactive) session
    # Added baseline_corrected_* keys so the summary can explicitly report
    # which files/materials had baseline-corrected data generated inline.
    baseline_session_changes = {
        "quality": [],
        "saved_file": [],
        "saved_filtered": 0,
        "baseline_corrected_file": [],  # list[(idx, function_name)]
        "baseline_corrected_material": [],  # list[(material, function_name, count_updated)]
    }

    # --- Helper: robustly extract numeric x/y arrays from a DataFrame row ---
    def _row_xy(
        row,
    ):  # local helper (only used inside interactive baseline_correct_spectra)
        """Return (x_array, y_array) as float numpy arrays for a row.

        Tries literal_eval for string-stored lists; tolerates already-list/array.
        Falls back to empty arrays on any failure (so caller can skip row cleanly).
        """
        try:
            x_raw = row.get("X-Axis")
            y_raw = row.get("Raw Data")
        except Exception:
            return np.array([], dtype=float), np.array([], dtype=float)

        # Decode strings representing arrays; try literal_eval first, then a robust fallback
        def _to_array_from_string(s):
            # First attempt: literal_eval on python-literal list strings
            try:
                return ast.literal_eval(s)
            except Exception:
                pass
            # Fallback: find bracketed content and parse with numpy.fromstring
            try:
                txt = str(s)
                lb = txt.find("[")
                rb = txt.rfind("]")
                if lb != -1 and rb != -1 and rb > lb:
                    inner = txt[lb + 1 : rb]
                    # Normalize common tokens
                    inner = (
                        inner.replace("NaN", "nan")
                        .replace("INF", "inf")
                        .replace("-INF", "-inf")
                    )
                    arr = np.fromstring(inner, sep=",")
                    return arr.tolist()
            except Exception:
                pass
            return s

        if isinstance(x_raw, str):
            x_raw = _to_array_from_string(x_raw)
        if isinstance(y_raw, str):
            y_raw = _to_array_from_string(y_raw)
        # Coerce iterables to list first to avoid pandas Series dtype surprises
        try:
            if hasattr(x_raw, "__iter__") and not isinstance(x_raw, (str, bytes)):
                x_raw = list(x_raw)
        except Exception:
            x_raw = []
        try:
            if hasattr(y_raw, "__iter__") and not isinstance(y_raw, (str, bytes)):
                y_raw = list(y_raw)
        except Exception:
            y_raw = []
        try:
            x_arr = np.asarray(x_raw, dtype=float)
        except Exception:
            x_arr = np.array([], dtype=float)
        try:
            y_arr = np.asarray(y_raw, dtype=float)
        except Exception:
            y_arr = np.array([], dtype=float)
        # Basic shape validation; allow proceeding only if 1D and same length
        if x_arr.ndim != 1 or y_arr.ndim != 1 or x_arr.size == 0 or y_arr.size == 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        if x_arr.size != y_arr.size:
            return np.array([], dtype=float), np.array([], dtype=float)
        return x_arr, y_arr

    if baseline_function is None:
        # Default to ARPLS when not specified; user can change via dropdown below
        baseline_function = "ARPLS"
    # Do NOT auto-launch manual baseline; user must still select a spectrum first (minimal mode preserved)
    # Initialize selection placeholders; user will pick a spectrum via dropdowns
    selected_row = None
    # Track currently displayed spectrum independent of dropdown selection so plot persists
    # when its option is removed after marking bad quality with 'Include bad spectra' unchecked.
    current_idx_bc = None
    x_values = np.array([])
    y_values = np.array([])
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
            filtered_df = FTIR_DataFrame[
                (FTIR_DataFrame["File Location"] == folder)
                & (FTIR_DataFrame["File Name"] == fname)
            ]
        else:
            filtered_df = FTIR_DataFrame[FTIR_DataFrame["File Name"] == filepath]
        # Exclude rows marked as bad quality
        try:
            filtered_df = filtered_df[_quality_good_mask(filtered_df)]
        except Exception:
            pass
        if filtered_df.empty:
            raise ValueError(f"No entry found for file '{filepath}'.")
        selected_row = filtered_df.iloc[0]
        material = selected_row.get("Material", "Unknown")
        # Persist this selection to session state
        try:
            _set_session_selection(
                material=selected_row.get("Material"),
                conditions=selected_row.get("Conditions"),
                time=selected_row.get("Time"),
            )
        except Exception:
            pass
    # If a specific file is selected, compute x/y; otherwise wait for user selection
    if selected_row is not None:
        x_values = (
            ast.literal_eval(selected_row["X-Axis"])
            if isinstance(selected_row["X-Axis"], str)
            else selected_row["X-Axis"]
        )
        y_values = (
            ast.literal_eval(selected_row["Raw Data"])
            if isinstance(selected_row["Raw Data"], str)
            else selected_row["Raw Data"]
        )
        y_values = np.array(y_values, dtype=float)

    parameters = _get_default_parameters(baseline_function)
    parameters = _cast_parameter_types(baseline_function, parameters)

    # Print selected file path only after a specific file is chosen
    if selected_row is not None:
        file_path = os.path.join(
            selected_row.get("File Location", ""), selected_row.get("File Name", "")
        )
        try:
            print(f"Plotting: {file_path}")
        except Exception:
            pass

    # Widget setup for live parameter editing
    baseline_parameter_widgets = {}
    # Explicitly define widgets for each baseline function and parameter
    if baseline_function.upper() == "ARPLS":
        # lam: float, iterations: int (diff_order fixed internally; not user-editable)
        baseline_parameter_widgets["lam"] = widgets.FloatSlider(
            value=parameters.get("lam", 1e5),
            min=1e4,
            max=1e6,
            step=1e4,
            description="Smoothness (lam)",
            readout_format=".1e",
            style={"description_width": "auto"},
        )
        baseline_parameter_widgets["max_iter"] = widgets.IntSlider(
            value=parameters.get("max_iter", 50),
            min=1,
            max=200,
            step=1,
            description="Max Iterations",
            style={"description_width": "auto"},
        )
        baseline_parameter_widgets["tol"] = widgets.FloatSlider(
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
        baseline_parameter_widgets["lam"] = widgets.FloatSlider(
            value=parameters.get("lam", 1e6),
            min=1e5,
            max=1e7,
            step=1e5,
            description="Smoothness (lam)",
            readout_format=".1e",
            style={"description_width": "auto"},
        )
        baseline_parameter_widgets["quantile"] = widgets.FloatSlider(
            value=parameters.get("quantile", 0.05),
            min=0.001,
            max=0.5,
            step=0.001,
            description="Quantile",
            readout_format=".3f",
            style={"description_width": "auto"},
        )
        baseline_parameter_widgets["num_knots"] = widgets.IntSlider(
            value=parameters.get("num_knots", 100),
            min=5,
            max=500,
            step=5,
            description="Knots",
            style={"description_width": "auto"},
        )
        baseline_parameter_widgets["spline_degree"] = widgets.IntSlider(
            value=parameters.get("spline_degree", 3),
            min=1,
            max=5,
            step=1,
            description="Spline Degree",
            style={"description_width": "auto"},
        )
        baseline_parameter_widgets["diff_order"] = widgets.IntSlider(
            value=parameters.get("diff_order", 3),
            min=1,
            max=3,
            step=1,
            description="Differential Order",
            style={"description_width": "auto"},
        )
        baseline_parameter_widgets["max_iter"] = widgets.IntSlider(
            value=parameters.get("max_iter", 100),
            min=1,
            max=1000,
            step=1,
            description="Max Iterations",
            style={"description_width": "auto"},
        )
        baseline_parameter_widgets["tol"] = widgets.FloatSlider(
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
        baseline_parameter_widgets["lam"] = widgets.FloatSlider(
            value=parameters.get("lam", 1e6),
            min=1e4,
            max=1e7,
            step=1e5,
            description="Smoothness (lam)",
            readout_format=".1e",
            style={"description_width": "auto"},
        )
        # If no spectrum is selected yet, use a generic default for scale; recomputed on selection
        if selected_row is not None and y_values is not None and len(y_values) > 0:
            try:
                scale_default = int(
                    np.clip(ceil(optimize_window(y_values) / 2), 2, 500)
                )
            except Exception:
                scale_default = 50
        else:
            scale_default = 50
        scale_val = parameters.get("scale", None)
        if scale_val is None:
            scale_val = scale_default
        baseline_parameter_widgets["scale"] = widgets.IntSlider(
            value=int(scale_val),
            min=2,
            max=500,
            step=1,
            description="Scale",
            style={"description_width": "auto"},
        )
        baseline_parameter_widgets["num_std"] = widgets.FloatSlider(
            value=parameters.get("num_std", 3.0),
            min=1.5,
            max=4.5,
            step=0.1,
            description="Standard Deviations",
            readout_format=".2f",
            style={"description_width": "auto"},
        )
        baseline_parameter_widgets["diff_order"] = widgets.IntSlider(
            value=parameters.get("diff_order", 2),
            min=1,
            max=3,
            step=1,
            description="Differential Order",
            style={"description_width": "auto"},
        )
        baseline_parameter_widgets["min_length"] = widgets.IntSlider(
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
        base_df = filtered_df.copy()
    else:
        base_df = FTIR_DataFrame.copy()
        try:
            base_df = base_df[_quality_good_mask(base_df)]
        except Exception:
            pass
    # Unique materials
    try:
        unique_materials = (
            sorted(
                {
                    str(v)
                    for v in base_df.get("Material", pd.Series([], dtype=object))
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                }
            )
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
    # Conditions list (exclude 'unexposed' from selector values)
    try:
        cond_series = (
            base_df["Conditions"]
            if "Conditions" in base_df.columns
            else (
                base_df["Condition"]
                if "Condition" in base_df.columns
                else pd.Series([], dtype=object)
            )
        )
        _all_conditions = [
            str(v) for v in cond_series.dropna().astype(str).unique().tolist()
        ]
        unique_conditions = sorted(
            [c for c in _all_conditions if c.strip().lower() != "unexposed"]
        )
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
        value=(
            str(baseline_function).upper()
            if str(baseline_function).upper() in ["ARPLS", "IRSQR", "FABC", "MANUAL"]
            else "ARPLS"
        ),
        description="Baseline",
        layout=widgets.Layout(width="30%"),
    )
    # Normalize baseline_function to a valid string and sync with dropdown to avoid None-related crashes on first selection
    try:
        if not isinstance(baseline_function, str) or str(
            baseline_function
        ).upper() not in ("ARPLS", "IRSQR", "FABC", "MANUAL"):
            baseline_function = str(baseline_dd.value).upper()
        else:
            baseline_function = str(baseline_function).upper()
    except Exception:
        baseline_function = "ARPLS"
    # Spectrum dropdown (built via helper)
    spectrum_sel = widgets.Dropdown(
        options=[("Select a spectrum…", None)],
        value=None,
        description="Spectrum",
        layout=widgets.Layout(width="70%"),
    )

    def _rebuild_conditions_options():
        """Rebuild the Conditions dropdown based on current Material filter and data."""
        try:
            if material_dd.value == "any":
                dfm = base_df
            else:
                dfm = base_df[
                    base_df.get("Material", "").astype(str) == str(material_dd.value)
                ]
            cs = (
                dfm["Conditions"]
                if "Conditions" in dfm.columns
                else (
                    dfm["Condition"]
                    if "Condition" in dfm.columns
                    else pd.Series([], dtype=object)
                )
            )
            cvals = [str(v) for v in cs.dropna().astype(str).unique().tolist()]
            cvals = sorted([c for c in cvals if c.strip().lower() != "unexposed"])
            curr = (
                conditions_dd.value
                if conditions_dd.value in (["any"] + cvals)
                else "any"
            )
            conditions_dd.options = ["any"] + cvals
            conditions_dd.value = curr
        except Exception:
            pass

    def _build_spectrum_options():
        """Recompute the Spectrum dropdown options using current filters and session defaults."""
        # Update current selection
        nonlocal selected_row, x_values, y_values, material
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
                cond_col = (
                    "Conditions"
                    if "Conditions" in df.columns
                    else ("Condition" if "Condition" in df.columns else None)
                )
                if cond_col is not None:
                    # Always include 'unexposed' spectra in addition to the selected condition
                    sel_val = str(conditions_dd.value)
                    cond_series = df.get(cond_col, pd.Series([], dtype=object)).astype(
                        str
                    )
                    unexp_mask = cond_series.str.strip().str.lower() == "unexposed"
                    cond_mask = cond_series == sel_val
                    df = df[cond_mask | unexp_mask]
            # Sort by time if present
            if "Time" in df.columns:
                df["_sort_time"] = pd.to_numeric(df["Time"], errors="coerce").fillna(
                    float("inf")
                )
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
                    for _l, v in options:
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
                selected_row = rsel
                material = rsel.get("Material", material)
                x_values = (
                    ast.literal_eval(rsel["X-Axis"])
                    if isinstance(rsel["X-Axis"], str)
                    else rsel["X-Axis"]
                )
                y_values = (
                    ast.literal_eval(rsel["Raw Data"])
                    if isinstance(rsel["Raw Data"], str)
                    else rsel["Raw Data"]
                )
                y_values = np.array(y_values, dtype=float)
                # Persist session
                try:
                    _set_session_selection(
                        material=selected_row.get("Material"),
                        conditions=selected_row.get("Conditions"),
                        time=selected_row.get("Time"),
                    )
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
            selected_row = None
    except Exception:
        pass

    baseline_output_area = widgets.Output()
    # Persist a single Plotly FigureWidget and update its traces for low flicker
    baseline_figure_widget = None

    def _plot_baseline(**widget_params):
        """Compute and display baseline and residual using current params and selected spectrum."""
        nonlocal baseline_figure_widget
        # Merge and cast widget parameters
        param_vals = parameters.copy()
        param_vals.update(widget_params)
        param_vals = _cast_parameter_types(baseline_function, param_vals)
        with baseline_output_area:
            # If no spectrum has been selected yet, prompt once
            # Allow continued display even if dropdown value cleared after marking bad.
            if selected_row is None:
                # Do not clear an existing plot; only show prompt if nothing rendered yet.
                if baseline_figure_widget is None:
                    try:
                        clear_output(wait=True)
                    except Exception:
                        pass
                    print("Select a spectrum to preview the baseline.")
                else:
                    # Leave existing figure visible; optionally could append a note.
                    try:
                        print("(Select a spectrum to update the preview.)")
                    except Exception:
                        pass
                return

            # Compute baseline safely
            try:
                if baseline_function.upper() == "ARPLS":
                    baseline_result = arpls(y_values, **param_vals)
                elif baseline_function.upper() == "IRSQR":
                    baseline_result = irsqr(y_values, **param_vals, x_data=x_values)
                elif baseline_function.upper() == "FABC":
                    baseline_result = fabc(y_values, **param_vals)
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
                x_arr = np.asarray(x_values)
                y_arr = np.asarray(y_values)
                baseline_arr = np.asarray(baseline)
                residual = y_arr - baseline_arr

                # Build or update Plotly FigureWidget
                title_top = "Raw Data and Baseline"
                if (
                    baseline_figure_widget is None
                    or len(getattr(baseline_figure_widget, "data", [])) < 3
                ):
                    # Do not clear the whole cell output; this figure renders outside the Output widget.
                    # Create subplots: top (raw + baseline), bottom (baseline-corrected)
                    base_fig = make_subplots(
                        rows=2,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.12,
                        subplot_titles=(title_top, "Baseline-Corrected"),
                    )
                    baseline_figure_widget = go.FigureWidget(base_fig)
                    # Raw spectrum
                    baseline_figure_widget.add_scatter(
                        x=x_arr,
                        y=y_arr,
                        mode="lines",
                        name="Spectrum",
                        line=dict(color="black"),
                        row=1,
                        col=1,
                    )
                    # Baseline
                    baseline_figure_widget.add_scatter(
                        x=x_arr,
                        y=baseline_arr,
                        mode="lines",
                        name="Baseline",
                        line=dict(color="red", width=1.5, dash="dash"),
                        row=1,
                        col=1,
                    )
                    # Baseline-corrected (spectrum - baseline)
                    baseline_figure_widget.add_scatter(
                        x=x_arr,
                        y=residual,
                        mode="lines",
                        name="Baseline-Corrected",
                        line=dict(color="blue"),
                        row=2,
                        col=1,
                    )
                    # Axes labels and layout
                    baseline_figure_widget.update_yaxes(
                        title_text="Absorbance (AU)", row=1, col=1
                    )
                    baseline_figure_widget.update_yaxes(title_text="", row=2, col=1)
                    baseline_figure_widget.update_xaxes(
                        title_text="Wavenumber (cm⁻¹)", row=2, col=1
                    )
                    baseline_figure_widget.update_layout(
                        legend=dict(orientation="h", y=-0.2), height=800
                    )
                    display(baseline_figure_widget)
                else:
                    # Update data traces in-place (no redraw flicker)
                    try:
                        baseline_figure_widget.data[0].x = x_arr
                        baseline_figure_widget.data[0].y = y_arr
                        baseline_figure_widget.data[1].x = x_arr
                        baseline_figure_widget.data[1].y = baseline_arr
                        baseline_figure_widget.data[2].x = x_arr
                        baseline_figure_widget.data[2].y = residual
                    except Exception:
                        # Fall back to rebuild if trace shapes changed unexpectedly
                        baseline_figure_widget = None
                        _plot_baseline(**widget_params)
                        return
                    # Update subplot titles
                    try:
                        if (
                            hasattr(baseline_figure_widget.layout, "annotations")
                            and len(baseline_figure_widget.layout.annotations) >= 2
                        ):
                            baseline_figure_widget.layout.annotations[0].text = (
                                title_top
                            )
                            baseline_figure_widget.layout.annotations[1].text = (
                                "Baseline-Corrected"
                            )
                    except Exception:
                        pass
            except Exception as e:
                # Keep any existing figure; report error in the Output widget only.
                try:
                    clear_output(wait=True)
                    print(f"Plot error: {e}")
                except Exception:
                    pass

    # Minimal UI when no spectrum is selected: show only filters, spectrum dropdown, and Close button
    try:
        _no_selection = (selected_row is None) or (spectrum_sel.value is None)
    except Exception:
        _no_selection = True
    if _no_selection:
        # Simple message prompting selection
        with baseline_output_area:
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
    filters_row = widgets.HBox(
        [material_dd, conditions_dd, baseline_dd]
    )
    spectrum_row = widgets.HBox([spectrum_sel, include_bad_cb])
    ui = widgets.VBox([filters_row, spectrum_row, close_btn])

    container = widgets.VBox([ui, baseline_output_area])
    display(container)
    try:
        _TB_WIDGETS.extend([container])
    except Exception:
        pass

    # Wire minimal interactions: rebuild options on filter change
    def _on_mat_min(change):
        """Minimal-mode observer: when Material changes, refresh Conditions and Spectrum lists."""
        if change.get("name") == "value":
            _rebuild_conditions_options()
            _build_spectrum_options()

    def _on_cond_min(change):
        """Minimal-mode observer: when Conditions changes, refresh Spectrum list."""
        if change.get("name") == "value":
            _build_spectrum_options()

    def _on_inc_min(change):
        """Minimal-mode observer: when Include-bad toggles, refresh Spectrum list."""
        if change.get("name") == "value":
            _build_spectrum_options()

    def _on_base_min(change):
        """Minimal-mode observer: sync baseline function selection; manual builds later upon selection."""
        if change.get("name") == "value":
            nonlocal baseline_function
            try:
                baseline_function = str(change.get("new")).upper()
            except Exception:
                baseline_function = "ARPLS"
            # For MANUAL, defer anchor point UI until a spectrum is chosen.

    def _on_spec_min(change):
        """Minimal-mode observer: on Spectrum select, build the full parameter or manual UI in place."""
        if change.get("name") == "value" and change.get("new") is not None:
            # A spectrum has been chosen; build full UI in-place without recursive re-entry
            try:
                sel_idx = change.get("new")
                nonlocal selected_row, x_values, y_values, material, current_idx_bc
                selected_row = FTIR_DataFrame.loc[sel_idx]
                current_idx_bc = sel_idx
                material = selected_row.get("Material", material)
                x_values = (
                    ast.literal_eval(selected_row["X-Axis"])
                    if isinstance(selected_row["X-Axis"], str)
                    else selected_row["X-Axis"]
                )
                y_values = (
                    ast.literal_eval(selected_row["Raw Data"])
                    if isinstance(selected_row["Raw Data"], str)
                    else selected_row["Raw Data"]
                )
                y_values = np.array(y_values, dtype=float)
                try:
                    _set_session_selection(
                        material=selected_row.get("Material"),
                        conditions=selected_row.get("Conditions"),
                        time=selected_row.get("Time"),
                    )
                except Exception:
                    pass
            except Exception:
                return
            # Build integrated MANUAL mode or parameter UI depending on selection
            # Ensure baseline_function is a valid string in sync with dropdown (fresh-kernel safety)
            try:
                nonlocal baseline_function
            except Exception:
                pass
            try:
                baseline_function = (
                    str(baseline_function).upper()
                    if isinstance(baseline_function, str)
                    else str(baseline_dd.value).upper()
                )
                if baseline_function not in ("ARPLS", "IRSQR", "FABC", "MANUAL"):
                    baseline_function = str(baseline_dd.value).upper()
            except Exception:
                baseline_function = "ARPLS"

            def _build_manual_ui():
                """Build or rebuild the inline manual baseline UI, reusing the existing container.

                Replaces prior parameter UI without closing the shared parent container to allow
                switching between MANUAL and automated baselines without UI disappearance.
                """
                nonlocal baseline_parameter_widgets, baseline_function
                # Clear parameter widgets (manual mode uses anchor workflow instead)
                baseline_parameter_widgets = {}
                baseline_function = "MANUAL"
                # Detach any automated baseline observer while in manual mode to avoid double firing
                try:
                    baseline_dd.unobserve(_on_base_full, names="value")
                except Exception:
                    pass
                manual_out = widgets.Output()
                anchor_points = []
                baseline_active = False  # Flag: baseline preview active after Continue
                # Buttons
                continue_btn = widgets.Button(
                    description="Continue", button_style="success"
                )
                redo_btn = widgets.Button(
                    description="Undo all", button_style="warning"
                )
                undo_btn = widgets.Button(description="Undo last")
                save_file_btn_m = widgets.Button(
                    description="Save for file", button_style="success"
                )
                save_mat_btn_m = widgets.Button(
                    description="Save for material", button_style="info"
                )
                close_btn_m = widgets.Button(description="Close", button_style="danger")
                # Reusable quality controls
                mark_bad_btn_m, mark_good_btn_m, _refresh_mark_btns_m = (
                    _make_quality_controls(FTIR_DataFrame, lambda: selected_row)
                )

                # Track quality changes for session summary
                def _log_mark_bad_m(_b=None):
                    try:
                        if selected_row is not None:
                            baseline_session_changes.setdefault("quality", []).append((selected_row.name, "bad"))
                            _quality_dropdown_handle(
                                "bad",
                                dropdown=spectrum_sel,
                                include_bad_flag=include_bad_cb.value,
                                idx=selected_row.name,
                                label_builder=lambda i: f"{FTIR_DataFrame.loc[i].get('Material','')} | {FTIR_DataFrame.loc[i].get('Conditions', FTIR_DataFrame.loc[i].get('Condition',''))} | T={FTIR_DataFrame.loc[i].get('Time','')} | {FTIR_DataFrame.loc[i].get('File Name','')}",
                                observer_fn=_on_spec_m,
                            )
                    except Exception:
                        pass

                def _log_mark_good_m(_b=None):
                    try:
                        if selected_row is not None:
                            baseline_session_changes.setdefault("quality", []).append((selected_row.name, "good"))
                            _quality_dropdown_handle(
                                "good",
                                dropdown=spectrum_sel,
                                include_bad_flag=include_bad_cb.value,
                                idx=selected_row.name,
                                label_builder=lambda i: f"{FTIR_DataFrame.loc[i].get('Material','')} | {FTIR_DataFrame.loc[i].get('Conditions', FTIR_DataFrame.loc[i].get('Condition',''))} | T={FTIR_DataFrame.loc[i].get('Time','')} | {FTIR_DataFrame.loc[i].get('File Name','')}",
                                observer_fn=_on_spec_m,
                            )
                    except Exception:
                        pass

                try:
                    mark_bad_btn_m.on_click(_log_mark_bad_m)
                    mark_good_btn_m.on_click(_log_mark_good_m)
                except Exception:
                    pass
                # Figures
                fig_m = go.FigureWidget()
                fig_m.add_scatter(
                    x=np.asarray(x_values, dtype=float),
                    y=np.asarray(y_values, dtype=float),
                    mode="lines",
                    name="Spectrum",
                    line=dict(color="black"),
                )
                fig_m.add_scatter(
                    x=[],
                    y=[],
                    mode="markers",
                    name="Anchor Points",
                    marker=dict(color="red", size=10),
                )
                fig_m.update_layout(
                    title="Manual Baseline: click to add anchor points",
                    xaxis_title="Wavenumber (cm⁻¹)",
                    yaxis_title="Absorbance (AU)",
                    height=450,
                )
                # Prevent autoscaling on subsequent baseline updates by fixing initial ranges
                try:
                    _x_min = float(np.min(x_values))
                    _x_max = float(np.max(x_values))
                    _y_min = float(np.min(y_values))
                    _y_max = float(np.max(y_values))
                    fig_m.update_xaxes(range=[_x_min, _x_max], autorange=False)
                    fig_m.update_yaxes(range=[_y_min, _y_max], autorange=False)
                except Exception:
                    pass
                fig_corr = go.FigureWidget()
                fig_corr.add_scatter(
                    x=[],
                    y=[],
                    mode="lines",
                    name="Baseline-Corrected",
                    line=dict(color="blue"),
                )
                fig_corr.update_layout(
                    title="Baseline-Corrected",
                    xaxis_title="Wavenumber (cm⁻¹)",
                    yaxis_title="Absorbance (AU)",
                    height=350,
                )
                # Match corrected figure axes to primary figure to avoid autoscale jumps
                try:
                    fig_corr.update_xaxes(range=[_x_min, _x_max], autorange=False)
                    fig_corr.update_yaxes(range=[_y_min, _y_max], autorange=False)
                except Exception:
                    pass

                # (Handlers wired within helper)

                # --- Colab-safe manual anchor input ---
                # In Colab, Plotly click callbacks are unreliable; provide a
                # Text input for anchor x-values and an Add button instead.
                anchor_input = widgets.Text(
                    placeholder="e.g., 400, 750, 1080",
                    description="Anchors",
                    layout=widgets.Layout(width="50%"),
                )
                add_anchor_btn = widgets.Button(
                    description="Add",
                    button_style="info",
                    tooltip="Add anchor points from the text box",
                )

                def _add_anchors_from_text(_b=None):
                    raw = str(anchor_input.value or "").strip()
                    if not raw:
                        return
                    try:
                        vals = [float(v) for v in raw.split(",") if v.strip()]
                    except Exception:
                        with manual_out:
                            print("Invalid anchor list. Use comma-separated numbers.")
                        return
                    # Merge into anchor_points, keep unique, sorted
                    for v in vals:
                        try:
                            if v not in anchor_points:
                                anchor_points.append(float(v))
                        except Exception:
                            continue
                    ap_sorted = sorted(anchor_points)
                    # Update anchor markers on raw figure
                    try:
                        xs = np.asarray(x_values, dtype=float)
                        ys = np.asarray(y_values, dtype=float)
                        fig_m.data[1].x = ap_sorted
                        fig_m.data[1].y = [
                            float(ys[int(np.nanargmin(np.abs(xs - ax)))])
                            for ax in ap_sorted
                        ]
                    except Exception:
                        pass
                    # If preview active, recompute
                    if baseline_active and len(anchor_points) >= 2:
                        _preview_baseline()

                add_anchor_btn.on_click(_add_anchors_from_text)

                # Click handler to add anchor point at nearest x
                def _on_click(trace, points, selector):
                    try:
                        if not points.xs:
                            return
                        x_click = float(points.xs[0])
                        xs = np.asarray(x_values, dtype=float)
                        ys = np.asarray(y_values, dtype=float)
                        idx_near = int(np.nanargmin(np.abs(xs - x_click)))
                        apx = float(xs[idx_near])
                        # Avoid duplicates
                        if apx not in anchor_points:
                            anchor_points.append(apx)
                            ap_sorted = sorted(anchor_points)
                            fig_m.data[1].x = ap_sorted
                            fig_m.data[1].y = [
                                float(ys[int(np.nanargmin(np.abs(xs - ax)))])
                                for ax in ap_sorted
                            ]
                            # If baseline already active, recompute immediately for live update
                            if baseline_active and len(anchor_points) >= 2:
                                _preview_baseline()
                    except Exception:
                        pass

                # Attach click to raw trace only outside Colab
                try:
                    if not _IN_COLAB:
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
                        if (
                            fig_m.layout.xaxis.autorange is not True
                            and fig_m.layout.xaxis.range
                        ):
                            x_range_main = list(fig_m.layout.xaxis.range)
                        if (
                            fig_m.layout.yaxis.autorange is not True
                            and fig_m.layout.yaxis.range
                        ):
                            y_range_main = list(fig_m.layout.yaxis.range)
                    except Exception:
                        pass
                    try:
                        if (
                            fig_corr.layout.xaxis.autorange is not True
                            and fig_corr.layout.xaxis.range
                        ):
                            x_range_corr = list(fig_corr.layout.xaxis.range)
                        if (
                            fig_corr.layout.yaxis.autorange is not True
                            and fig_corr.layout.yaxis.range
                        ):
                            y_range_corr = list(fig_corr.layout.yaxis.range)
                    except Exception:
                        pass
                    xs = np.asarray(x_values, dtype=float)
                    ys = np.asarray(y_values, dtype=float)
                    if len(anchor_points) < 2:
                        with manual_out:
                            print("Select at least two anchor points to preview.")
                        return
                    ap_sorted = np.array(sorted(anchor_points), dtype=float)
                    y_anchor = np.array(
                        [ys[int(np.nanargmin(np.abs(xs - ap)))] for ap in ap_sorted],
                        dtype=float,
                    )
                    try:
                        spline = CubicSpline(
                            ap_sorted, y_anchor, bc_type=((1, 0.0), (1, 0.0))
                        )
                    except Exception:
                        spline = CubicSpline(ap_sorted, y_anchor)
                    baseline_vals = spline(xs)
                    corrected = ys - baseline_vals
                    # Update raw/baseline figure (keep 0: raw, 1: anchors)
                    while len(fig_m.data) > 2:
                        fig_m.data = tuple(fig_m.data[:2])
                    fig_m.add_scatter(
                        x=xs,
                        y=baseline_vals,
                        mode="lines",
                        name="Baseline",
                        line=dict(color="red", width=1.5, dash="dash"),
                    )
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
                    # Rebuild manual UI inline without closing shared container
                    try:
                        fig_m.close()
                    except Exception:
                        pass
                    try:
                        fig_corr.close()
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
                            if (
                                fig_m.layout.xaxis.autorange is not True
                                and fig_m.layout.xaxis.range
                            ):
                                x_range_main = list(fig_m.layout.xaxis.range)
                            if (
                                fig_m.layout.yaxis.autorange is not True
                                and fig_m.layout.yaxis.range
                            ):
                                y_range_main = list(fig_m.layout.yaxis.range)
                        except Exception:
                            pass
                        try:
                            if (
                                fig_corr.layout.xaxis.autorange is not True
                                and fig_corr.layout.xaxis.range
                            ):
                                x_range_corr = list(fig_corr.layout.xaxis.range)
                            if (
                                fig_corr.layout.yaxis.autorange is not True
                                and fig_corr.layout.yaxis.range
                            ):
                                y_range_corr = list(fig_corr.layout.yaxis.range)
                        except Exception:
                            pass
                        # Pop last added point (reverse chronological)
                        last = anchor_points.pop()
                        xs = np.asarray(x_values, dtype=float)
                        ys = np.asarray(y_values, dtype=float)
                        ap_sorted = sorted(anchor_points)
                        # Update anchor markers
                        try:
                            fig_m.data[1].x = ap_sorted
                            fig_m.data[1].y = [
                                float(ys[int(np.nanargmin(np.abs(xs - ax)))])
                                for ax in ap_sorted
                            ]
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
                    if selected_row is None or len(anchor_points) < 2:
                        with manual_out:
                            print("Need at least two anchor points to save.")
                        return
                    xs = np.asarray(x_values, dtype=float)
                    ys = np.asarray(y_values, dtype=float)
                    ap_sorted = np.array(sorted(anchor_points), dtype=float)
                    y_anchor = np.array(
                        [ys[int(np.nanargmin(np.abs(xs - ap)))] for ap in ap_sorted],
                        dtype=float,
                    )
                    try:
                        spline = CubicSpline(
                            ap_sorted, y_anchor, bc_type=((1, 0.0), (1, 0.0))
                        )
                    except Exception:
                        spline = CubicSpline(ap_sorted, y_anchor)
                    baseline_vals = spline(xs).astype(float)
                    corrected = (ys - baseline_vals).astype(float)
                    # Persist to DataFrame for this file
                    try:
                        FTIR_DataFrame.at[selected_row.name, "Baseline Function"] = (
                            "Manual"
                        )
                        FTIR_DataFrame.at[selected_row.name, "Baseline Parameters"] = (
                            str(
                                {
                                    "anchor_points": [
                                        float(v) for v in ap_sorted.tolist()
                                    ]
                                }
                            )
                        )
                        FTIR_DataFrame.at[selected_row.name, "Baseline"] = (
                            baseline_vals.tolist()
                        )
                        FTIR_DataFrame.at[
                            selected_row.name, "Baseline-Corrected Data"
                        ] = corrected.tolist()
                    except Exception:
                        pass
                    with manual_out:
                        print("Saved manual baseline for this file.")
                    try:
                        if selected_row is not None:
                            baseline_session_changes.setdefault(
                                "saved_file", []
                            ).append((selected_row.name, None))
                            # Also record as baseline-corrected (Manual) with filename for summary output
                            try:
                                fn = FTIR_DataFrame.at[selected_row.name, "File Name"]
                                if not isinstance(fn, str) or not fn.strip():
                                    _loc = FTIR_DataFrame.at[selected_row.name, "File Location"]
                                    fn = os.path.basename(_loc) if isinstance(_loc, str) else str(selected_row.name)
                            except Exception:
                                fn = str(selected_row.name)
                            baseline_session_changes.setdefault(
                                "baseline_corrected_file", []
                            ).append((selected_row.name, "MANUAL", str(fn)))
                    except Exception:
                        pass

                def _save_material(_b=None):
                    """Save manual baseline anchor points and compute baseline/corrected arrays for all rows of the material.

                    For each spectrum of the selected material, we:
                      1. Re-sample anchor point y-values from that spectrum's raw data.
                      2. Fit a cubic spline through those anchor points.
                      3. Store the spline-evaluated baseline and (raw - baseline) in the DataFrame.
                    """
                    if selected_row is None or len(anchor_points) < 2:
                        with manual_out:
                            print("Need at least two anchor points to save.")
                        return
                    mat_val = selected_row.get("Material", material)
                    ap_sorted = sorted(anchor_points)
                    # Ensure destination columns exist and are object dtype (lists per row)
                    try:
                        for _col in ("Baseline", "Baseline-Corrected Data"):
                            if _col not in FTIR_DataFrame.columns:
                                FTIR_DataFrame[_col] = None
                        FTIR_DataFrame["Baseline"] = FTIR_DataFrame["Baseline"].astype(
                            object
                        )
                        FTIR_DataFrame["Baseline-Corrected Data"] = FTIR_DataFrame[
                            "Baseline-Corrected Data"
                        ].astype(object)
                    except Exception:
                        pass
                    # Identify rows for this material
                    try:
                        mat_series = FTIR_DataFrame.get(
                            "Material",
                            pd.Series(index=FTIR_DataFrame.index, dtype=object),
                        ).astype(str)
                        msk = (
                            mat_series.str.strip().str.casefold()
                            == str(mat_val).strip().casefold()
                        )
                        # Coerce to boolean ndarray with same length to avoid index alignment pitfalls
                        msk = pd.Series(msk, index=FTIR_DataFrame.index)
                    except Exception:
                        msk = pd.Series(
                            [False] * len(FTIR_DataFrame), index=FTIR_DataFrame.index
                        )
                    matched_total = int(msk.sum())
                    updated_count = 0
                    skipped_count = 0
                    # Persist function + parameters first
                    try:
                        FTIR_DataFrame.loc[msk, "Baseline Function"] = "Manual"
                        FTIR_DataFrame.loc[msk, "Baseline Parameters"] = str(
                            {"anchor_points": [float(v) for v in ap_sorted]}
                        )
                    except Exception:
                        pass
                    ap_sorted_arr = np.array(ap_sorted, dtype=float)
                    for _ridx in FTIR_DataFrame.loc[msk].index:
                        try:
                            rloc = FTIR_DataFrame.loc[_ridx]
                            xs, ys = _row_xy(rloc)
                            if xs.size == 0:
                                skipped_count += 1
                                continue
                            y_anchor = np.array(
                                [
                                    ys[int(np.nanargmin(np.abs(xs - ap)))]
                                    for ap in ap_sorted_arr
                                ],
                                dtype=float,
                            )
                            try:
                                spline = CubicSpline(
                                    ap_sorted_arr,
                                    y_anchor,
                                    bc_type=((1, 0.0), (1, 0.0)),
                                )
                            except Exception:
                                spline = CubicSpline(ap_sorted_arr, y_anchor)
                            baseline_vals = spline(xs).astype(float)
                            if baseline_vals.size != ys.size:
                                skipped_count += 1
                                continue
                            corrected_vals = (ys - baseline_vals).astype(float)
                            FTIR_DataFrame.at[_ridx, "Baseline"] = (
                                baseline_vals.tolist()
                            )
                            FTIR_DataFrame.at[_ridx, "Baseline-Corrected Data"] = (
                                corrected_vals.tolist()
                            )
                            updated_count += 1
                        except Exception:
                            skipped_count += 1
                            continue
                    with manual_out:
                        print(
                            f"Saved manual baseline (anchors -> arrays) for material '{mat_val}' on {updated_count}/{matched_total} rows (skipped {skipped_count})."
                        )
                    try:
                        baseline_session_changes["saved_filtered"] = (
                            int(baseline_session_changes.get("saved_filtered", 0))
                            + matched_total
                        )
                    except Exception:
                        pass
                    try:
                        baseline_session_changes.setdefault(
                            "baseline_corrected_material", []
                        ).append((mat_val, "Manual", updated_count))
                    except Exception:
                        pass

                def _close_m(_b=None):
                    # Show session summary and collapse UI to summary only
                    try:
                        lines = _session_summary_lines(
                            baseline_session_changes, context="baseline_correct_spectra"
                        )
                        _emit_session_summary(
                            manual_out, lines, title="Session Summary (Baseline Correction)"
                        )
                    except Exception:
                        pass
                    try:
                        # Replace UI with summary output
                        container.children = (widgets.VBox([manual_out]),)
                    except Exception:
                        # Fallback: close figures
                        try:
                            fig_m.close()
                            fig_corr.close()
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
                        _rebuild_conditions_options()
                        _build_spectrum_options()
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
                            nonlocal selected_row, x_values, y_values, material, current_idx_bc
                            selected_row = r3
                            current_idx_bc = sel_idx3
                            material = r3.get("Material", material)
                            x_values = (
                                ast.literal_eval(r3["X-Axis"])
                                if isinstance(r3["X-Axis"], str)
                                else r3["X-Axis"]
                            )
                            y_values = (
                                ast.literal_eval(r3["Raw Data"])
                                if isinstance(r3["Raw Data"], str)
                                else r3["Raw Data"]
                            )
                            y_arr = np.asarray(y_values, dtype=float)
                            x_arr = np.asarray(x_values, dtype=float)
                            fig_m.data[0].x = x_arr
                            fig_m.data[0].y = y_arr
                            # reset anchors and preview
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
                            _set_session_selection(
                                material=selected_row.get("Material"),
                                conditions=selected_row.get("Conditions"),
                                time=selected_row.get("Time"),
                            )
                        except Exception:
                            pass

                def _on_base_m(change):
                    if change.get("name") == "value":
                        new_b = str(change.get("new")).upper()
                        if new_b == "MANUAL":
                            return  # already in manual mode
                        # Switch to automated baseline inline: detach manual observer, attach full observer
                        try:
                            baseline_dd.unobserve(_on_base_m, names="value")
                        except Exception:
                            pass
                        # Close manual figures
                        try:
                            fig_m.close()
                            fig_corr.close()
                        except Exception:
                            pass
                        # Reattach automated observer if not present
                        try:
                            baseline_dd.observe(_on_base_full, names="value")
                        except Exception:
                            pass
                        # Invoke automated baseline rebuild logic
                        try:
                            _on_base_full({"name": "value", "new": new_b})
                        except Exception:
                            # Fallback: set dropdown value triggers original observer
                            try:
                                baseline_dd.value = new_b
                            except Exception:
                                pass
                        return

                material_dd.observe(_on_mat_m, names="value")
                conditions_dd.observe(_on_cond_m, names="value")
                include_bad_cb.observe(_on_inc_m, names="value")
                spectrum_sel.observe(_on_spec_m, names="value")
                baseline_dd.observe(_on_base_m, names="value")

                # Compose UI
                controls_row_top = widgets.HBox(
                    [material_dd, conditions_dd, baseline_dd]
                )
                spec_row = widgets.HBox([spectrum_sel, include_bad_cb])
                mark_row_m = widgets.HBox([mark_bad_btn_m, mark_good_btn_m])
                # Row for Save/Close actions (outside bordered plot area)
                btn_row_m = widgets.HBox(
                    [save_file_btn_m, save_mat_btn_m, close_btn_m]
                )
                # Split action rows so anchor input can sit between Continue and Redo/Undo
                manual_continue_row = widgets.HBox([continue_btn])
                anchor_row_m = widgets.HBox([anchor_input, add_anchor_btn])
                manual_redo_undo_row = widgets.HBox([redo_btn, undo_btn])
                # Bordered plot + mark section: Continue row, anchor entry, redo/undo, then plots and mark buttons
                bordered_manual = widgets.VBox(
                    [
                        anchor_row_m,
                        manual_redo_undo_row,
                        manual_continue_row,
                        fig_m,
                        fig_corr,
                        mark_row_m,
                    ],
                    layout=widgets.Layout(
                        border="1px solid #ccc",
                        padding="8px",
                        margin="6px 0",
                    ),
                )
                manual_ui = widgets.VBox(
                    [
                        controls_row_top,
                        spec_row,
                        btn_row_m,
                        bordered_manual,
                        manual_out,
                    ]
                )
                try:
                    container.children = (manual_ui,)
                except Exception:
                    display(manual_ui)
                try:
                    _TB_WIDGETS.extend([fig_m, fig_corr])
                except Exception:
                    pass
                _refresh_mark_btns_m()

            # If MANUAL, build manual UI and return
            if baseline_function.upper() == "MANUAL":
                try:
                    _build_manual_ui()
                except Exception as e:
                    with baseline_output_area:
                        try:
                            clear_output(wait=True)
                            print(f"Error building manual UI: {e}")
                        except Exception:
                            pass
                return

            # Otherwise, proceed to rebuild parameter widgets (refresh defaults for selected baseline)
            nonlocal baseline_parameter_widgets
            baseline_parameter_widgets = {}
            parameters_local = _get_default_parameters(baseline_function)
            parameters_local = _cast_parameter_types(
                baseline_function, parameters_local
            )
            if baseline_function.upper() == "ARPLS":
                baseline_parameter_widgets["lam"] = widgets.FloatSlider(
                    value=parameters_local.get("lam", 1e5),
                    min=1e4,
                    max=1e6,
                    step=1e4,
                    description="Smoothness (lam)",
                    readout_format=".1e",
                    continuous_update=False,
                    style={"description_width": "auto"},
                )
                baseline_parameter_widgets["max_iter"] = widgets.IntSlider(
                    value=parameters_local.get("max_iter", 50),
                    min=1,
                    max=200,
                    step=1,
                    description="Max Iterations",
                    continuous_update=False,
                    style={"description_width": "auto"},
                )
                baseline_parameter_widgets["tol"] = widgets.FloatSlider(
                    value=parameters_local.get("tol", 1e-3),
                    min=1e-6,
                    max=1e-1,
                    step=1e-4,
                    description="Tolerance",
                    readout_format=".1e",
                    continuous_update=False,
                    style={"description_width": "auto"},
                )
            elif baseline_function.upper() == "IRSQR":
                baseline_parameter_widgets["lam"] = widgets.FloatSlider(
                    value=parameters_local.get("lam", 1e6),
                    min=1e5,
                    max=1e7,
                    step=1e5,
                    description="Smoothness (lam)",
                    readout_format=".1e",
                    continuous_update=False,
                    style={"description_width": "auto"},
                )
                baseline_parameter_widgets["quantile"] = widgets.FloatSlider(
                    value=parameters_local.get("quantile", 0.05),
                    min=0.001,
                    max=0.5,
                    step=0.001,
                    description="Quantile",
                    readout_format=".3f",
                    continuous_update=False,
                    style={"description_width": "auto"},
                )
                baseline_parameter_widgets["num_knots"] = widgets.IntSlider(
                    value=parameters_local.get("num_knots", 100),
                    min=5,
                    max=500,
                    step=5,
                    description="Knots",
                    continuous_update=False,
                    style={"description_width": "auto"},
                )
                baseline_parameter_widgets["spline_degree"] = widgets.IntSlider(
                    value=parameters_local.get("spline_degree", 3),
                    min=1,
                    max=5,
                    step=1,
                    description="Spline Degree",
                    continuous_update=False,
                    style={"description_width": "auto"},
                )
                baseline_parameter_widgets["diff_order"] = widgets.IntSlider(
                    value=parameters_local.get("diff_order", 3),
                    min=1,
                    max=3,
                    step=1,
                    description="Differential Order",
                    continuous_update=False,
                    style={"description_width": "auto"},
                )
                baseline_parameter_widgets["max_iter"] = widgets.IntSlider(
                    value=parameters_local.get("max_iter", 100),
                    min=1,
                    max=1000,
                    step=1,
                    description="Max Iterations",
                    continuous_update=False,
                    style={"description_width": "auto"},
                )
                baseline_parameter_widgets["tol"] = widgets.FloatSlider(
                    value=parameters_local.get("tol", 1e-6),
                    min=1e-10,
                    max=1e-2,
                    step=1e-7,
                    description="Tolerance",
                    readout_format=".1e",
                    continuous_update=False,
                    style={"description_width": "auto"},
                )
            elif baseline_function.upper() == "FABC":
                baseline_parameter_widgets["lam"] = widgets.FloatSlider(
                    value=parameters_local.get("lam", 1e6),
                    min=1e4,
                    max=1e7,
                    step=1e5,
                    description="Smoothness (lam)",
                    readout_format=".1e",
                    continuous_update=False,
                    style={"description_width": "auto"},
                )
                # scale recompute from selected spectrum
                try:
                    scale_default2 = int(
                        np.clip(ceil(optimize_window(y_values) / 2), 2, 500)
                    )
                except Exception:
                    scale_default2 = 50
                scale_val2 = parameters_local.get("scale") or scale_default2
                baseline_parameter_widgets["scale"] = widgets.IntSlider(
                    value=int(scale_val2),
                    min=2,
                    max=500,
                    step=1,
                    description="Scale",
                    continuous_update=False,
                    style={"description_width": "auto"},
                )
                baseline_parameter_widgets["num_std"] = widgets.FloatSlider(
                    value=parameters_local.get("num_std", 3.0),
                    min=1.5,
                    max=4.5,
                    step=0.1,
                    description="Standard Deviations",
                    readout_format=".2f",
                    continuous_update=False,
                    style={"description_width": "auto"},
                )
                baseline_parameter_widgets["diff_order"] = widgets.IntSlider(
                    value=parameters_local.get("diff_order", 2),
                    min=1,
                    max=3,
                    step=1,
                    description="Differential Order",
                    continuous_update=False,
                    style={"description_width": "auto"},
                )
                baseline_parameter_widgets["min_length"] = widgets.IntSlider(
                    value=parameters_local.get("min_length", 2),
                    min=1,
                    max=6,
                    step=1,
                    description="Min Baseline Span Length",
                    continuous_update=False,
                    style={"description_width": "auto"},
                )
            # Build full UI identical to main branch
            defaults_full = _get_default_parameters(baseline_function)
            widget_rows_full = []
            for k, w in baseline_parameter_widgets.items():
                rb = widgets.Button(
                    description="Reset",
                    button_style="info",
                    layout=widgets.Layout(width="70px", margin="0 0 6px 8px"),
                )
                if k == "scale":

                    def _reset_scale2(_b=None, w=w):
                        try:
                            new_def = int(
                                np.clip(ceil(optimize_window(y_values) / 2), 2, 500)
                            )
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

            # Add a collapsible 'Parameter Details' toggle below the parameter widgets.
            # Clicking toggles visibility of the explanatory text; button persists for hide/show.
            def _make_param_details_row(name: str):
                name = str(name).upper().strip()
                if name == "ARPLS":
                    txt = (
                        "lam (float): Smoothness parameter (higher = smoother baseline).\n\n"
                        "max_iter (integer): Max number of fit iterations.\n\n"
                        "tol (float): Exit criteria (accuracy goal)."
                    )
                elif name == "IRSQR":
                    txt = (
                        "lam (float): The smoothing parameter (higher = smoother baseline).\n\n"
                        "quantile (float): The quantile at which to fit the baseline (0 < quantile < 1).\n\n"
                        "num_knots (integer): The number of knots for the spline.\n\n"
                        "spline_degree (integer): The degree of the spline.\n\n"
                        "diff_order (integer): The order of the differential matrix. Typical values are 3, 2, or 1.\n\n"
                        "max_iter (integer): The max number of fit iterations.\n\n"
                        "tol (float): Exit criteria (accuracy goal)."
                    )
                elif name == "FABC":
                    txt = (
                        "lam (float): The smoothing parameter (higher = smoother baseline).\n\n"
                        "scale (integer): The scale at which to calculate the continuous wavelet transform. Should be approximately equal to the index-based full-width-at-half-maximum of the peaks or features in the data. Default is None, which will use half of the value from :func:`.optimize_window`, which is not always a good value, but at least scales with the number of data points and gives a starting point for tuning the parameter.\n\n"
                        "num_std (float): The number of standard deviations to include when thresholding. Higher values\n"
                        "will assign more points as baseline.\n\n"
                        "diff_order (integer): The order of the differential matrix. Must be greater than 0. Typical values are 2 or 1.\n\n"
                        "min_length (integer): Any region of consecutive baseline points less than `min_length` is considered to be a false positive and all points in the region are converted to peak points. A higher `min_length` ensures less points are falsely assigned as baseline points. Default is 2, which only removes lone baseline points."
                    )
                else:
                    txt = ""

                toggle = widgets.ToggleButton(
                    value=_TB_PARAM_DETAILS_OPEN,
                    description="Parameter Details",
                    button_style="info",
                    icon="chevron-down",
                )
                details = widgets.HTML(
                    value=f"<pre style='white-space:pre-wrap;margin:0'>{txt}</pre>"
                )
                # Hide details by default
                try:
                    details.layout.display = "" if _TB_PARAM_DETAILS_OPEN else "none"
                except Exception:
                    pass

                def _on_toggle(change):
                    if change.get("name") == "value":
                        show = bool(change.get("new"))
                        global _TB_PARAM_DETAILS_OPEN
                        _TB_PARAM_DETAILS_OPEN = show
                        try:
                            details.layout.display = "" if show else "none"
                        except Exception:
                            pass
                        try:
                            toggle.icon = "chevron-up" if show else "chevron-down"
                        except Exception:
                            pass

                try:
                    toggle.observe(_on_toggle, names="value")
                except Exception:
                    pass
                return widgets.VBox([toggle, details])

            # Append the details row as its own row in the UI
            try:
                widget_rows_full.append(_make_param_details_row(baseline_function))
            except Exception:
                pass
            reset_all_btn2 = widgets.Button(
                description="Reset All",
                button_style="warning",
                # Align with per-parameter Reset button (same vertical spacing) and small left gap
                layout=widgets.Layout(width="90px", margin="0 0 6px 8px"),
            )

            def _reset_all2(_b=None):
                for kk, ww in baseline_parameter_widgets.items():
                    if kk == "scale":
                        try:
                            ww.value = int(
                                np.clip(ceil(optimize_window(y_values) / 2), 2, 500)
                            )
                        except Exception:
                            pass
                    elif kk in defaults_full:
                        ww.value = defaults_full[kk]

            reset_all_btn2.on_click(_reset_all2)
            save_file_btn2 = widgets.Button(
                description="Save for file",
                button_style="success",
                layout=widgets.Layout(margin="10px 10px 0 0"),
            )
            save_material_btn2 = widgets.Button(
                description="Save for material",
                button_style="info",
                layout=widgets.Layout(margin="10px 10px 0 0"),
            )
            mark_bad_btn2, mark_good_btn2, _refresh_mark_btns2 = _make_quality_controls(
                FTIR_DataFrame, lambda: selected_row, margin="10px 10px 0 0"
            )
            close_btn2 = widgets.Button(
                description="Close",
                button_style="danger",
                layout=widgets.Layout(margin="10px 0 0 0"),
            )

            # Track quality changes
            def _log_mark_bad2(_b=None):
                try:
                    if selected_row is not None:
                        baseline_session_changes.setdefault("quality", []).append((selected_row.name, "bad"))
                        _quality_dropdown_handle(
                            "bad",
                            dropdown=spectrum_sel,
                            include_bad_flag=include_bad_cb.value,
                            idx=selected_row.name,
                            label_builder=lambda i: f"{FTIR_DataFrame.loc[i].get('Material','')} | {FTIR_DataFrame.loc[i].get('Conditions', FTIR_DataFrame.loc[i].get('Condition',''))} | T={FTIR_DataFrame.loc[i].get('Time','')} | {FTIR_DataFrame.loc[i].get('File Name','')}",
                            observer_fn=_on_spec_full,
                        )
                except Exception:
                    pass

            def _log_mark_good2(_b=None):
                try:
                    if selected_row is not None:
                        baseline_session_changes.setdefault("quality", []).append((selected_row.name, "good"))
                        _quality_dropdown_handle(
                            "good",
                            dropdown=spectrum_sel,
                            include_bad_flag=include_bad_cb.value,
                            idx=selected_row.name,
                            label_builder=lambda i: f"{FTIR_DataFrame.loc[i].get('Material','')} | {FTIR_DataFrame.loc[i].get('Conditions', FTIR_DataFrame.loc[i].get('Condition',''))} | T={FTIR_DataFrame.loc[i].get('Time','')} | {FTIR_DataFrame.loc[i].get('File Name','')}",
                            observer_fn=_on_spec_full,
                        )
                except Exception:
                    pass

            try:
                mark_bad_btn2.on_click(_log_mark_bad2)
                mark_good_btn2.on_click(_log_mark_good2)
            except Exception:
                pass

            def _current_params2():
                cur = parameters_local.copy()
                for kk, ww in baseline_parameter_widgets.items():
                    cur[kk] = ww.value
                return _cast_parameter_types(baseline_function, cur)

            def _serialize2(d):
                def to_plain(v):
                    try:
                        if isinstance(v, (np.integer,)):
                            return int(v)
                        if isinstance(v, (np.floating,)):
                            return float(v)
                        if isinstance(v, np.ndarray):
                            return v.tolist()
                    except Exception:
                        pass
                    return v

                return {kk: to_plain(vv) for kk, vv in d.items()}

            def _save_file2(_b=None):
                if selected_row is None:
                    return
                pv = _serialize2(_current_params2())
                FTIR_DataFrame.at[selected_row.name, "Baseline Function"] = (
                    baseline_function.upper()
                )
                FTIR_DataFrame.at[selected_row.name, "Baseline Parameters"] = str(pv)
                # Compute and persist baseline + corrected arrays for this row
                try:
                    # Parse current spectrum arrays
                    xs = (
                        ast.literal_eval(selected_row["X-Axis"])
                        if isinstance(selected_row.get("X-Axis"), str)
                        else selected_row.get("X-Axis")
                    )
                    ys = (
                        ast.literal_eval(selected_row["Raw Data"])
                        if isinstance(selected_row.get("Raw Data"), str)
                        else selected_row.get("Raw Data")
                    )
                    xs = np.asarray(xs, dtype=float)
                    ys = np.asarray(ys, dtype=float)
                    if xs.size and ys.size:
                        # Choose computation path by function
                        params_exec = _cast_parameter_types(
                            baseline_function, pv.copy()
                        )
                        if baseline_function.upper() == "ARPLS":
                            bres = arpls(ys, **params_exec)
                        elif baseline_function.upper() == "IRSQR":
                            bres = irsqr(ys, **params_exec, x_data=xs)
                        elif baseline_function.upper() == "FABC":
                            bres = fabc(ys, **params_exec)
                        else:
                            bres = None
                        if isinstance(bres, tuple):
                            baseline_arr = np.asarray(bres[0], dtype=float)
                        elif isinstance(bres, dict):
                            baseline_arr = np.asarray(bres.get("baseline"), dtype=float)
                        else:
                            baseline_arr = np.asarray(bres, dtype=float)
                        if baseline_arr.size == ys.size:
                            corrected_arr = ys - baseline_arr
                            FTIR_DataFrame.at[selected_row.name, "Baseline"] = (
                                baseline_arr.tolist()
                            )
                            FTIR_DataFrame.at[
                                selected_row.name, "Baseline-Corrected Data"
                            ] = corrected_arr.tolist()
                            try:
                                try:
                                    fn = FTIR_DataFrame.at[
                                        selected_row.name, "File Name"
                                    ]
                                    if not isinstance(fn, str) or not fn.strip():
                                        _loc = FTIR_DataFrame.at[
                                            selected_row.name, "File Location"
                                        ]
                                        fn = os.path.basename(_loc) if isinstance(_loc, str) else str(selected_row.name)
                                except Exception:
                                    fn = str(selected_row.name)
                                baseline_session_changes.setdefault(
                                    "baseline_corrected_file", []
                                ).append((selected_row.name, baseline_function.upper(), str(fn)))
                            except Exception:
                                pass
                except Exception:
                    pass
                with baseline_output_area:
                    print("Saved baseline (parameters + arrays) for this file.")
                try:
                    baseline_session_changes.setdefault("saved_file", []).append(
                        (selected_row.name, None)
                    )
                except Exception:
                    pass

            def _save_material2(_b=None):
                if selected_row is None:
                    return
                pv = _serialize2(_current_params2())
                mat_val = selected_row.get("Material", material)
                try:
                    mat_series2 = FTIR_DataFrame.get(
                        "Material",
                        pd.Series(index=FTIR_DataFrame.index, dtype=object),
                    ).astype(str)
                    msk = (
                        mat_series2.str.strip().str.casefold()
                        == str(mat_val).strip().casefold()
                    )
                    msk = pd.Series(msk, index=FTIR_DataFrame.index)
                except Exception:
                    msk = pd.Series(
                        [False] * len(FTIR_DataFrame), index=FTIR_DataFrame.index
                    )
                # Persist function/parameters
                FTIR_DataFrame.loc[msk, "Baseline Function"] = baseline_function.upper()
                FTIR_DataFrame.loc[msk, "Baseline Parameters"] = str(pv)
                # Ensure destination columns exist and are object dtype (lists per row)
                for _col in ("Baseline", "Baseline-Corrected Data"):
                    if _col not in FTIR_DataFrame.columns:
                        FTIR_DataFrame[_col] = None
                try:
                    FTIR_DataFrame["Baseline"] = FTIR_DataFrame["Baseline"].astype(
                        object
                    )
                    FTIR_DataFrame["Baseline-Corrected Data"] = FTIR_DataFrame[
                        "Baseline-Corrected Data"
                    ].astype(object)
                except Exception:
                    pass
                # Iterate material rows and compute baseline arrays
                matched_total = int(msk.sum())
                updated_count = 0
                skipped_count = 0
                params_exec_global = _cast_parameter_types(baseline_function, pv.copy())
                for _ridx in FTIR_DataFrame.loc[msk].index:
                    try:
                        rloc = FTIR_DataFrame.loc[_ridx]
                        xs, ys = _row_xy(rloc)
                        if xs.size == 0:
                            skipped_count += 1
                            continue
                        # Compute baseline
                        if baseline_function.upper() == "ARPLS":
                            bres = arpls(ys, **params_exec_global)
                        elif baseline_function.upper() == "IRSQR":
                            bres = irsqr(ys, **params_exec_global, x_data=xs)
                        elif baseline_function.upper() == "FABC":
                            bres = fabc(ys, **params_exec_global)
                        else:
                            bres = None
                        if isinstance(bres, tuple):
                            baseline_arr = np.asarray(bres[0], dtype=float)
                        elif isinstance(bres, dict):
                            baseline_arr = np.asarray(bres.get("baseline"), dtype=float)
                        else:
                            baseline_arr = np.asarray(bres, dtype=float)
                        if baseline_arr.size != ys.size:
                            skipped_count += 1
                            continue
                        corrected_arr = ys - baseline_arr
                        FTIR_DataFrame.at[_ridx, "Baseline"] = baseline_arr.tolist()
                        FTIR_DataFrame.at[_ridx, "Baseline-Corrected Data"] = (
                            corrected_arr.tolist()
                        )
                        updated_count += 1
                    except Exception:
                        skipped_count += 1
                        continue
                with baseline_output_area:
                    print(
                        f"Saved baseline (parameters + arrays) for material '{mat_val}' on {updated_count}/{matched_total} rows (skipped {skipped_count})."
                    )
                try:
                    baseline_session_changes["saved_filtered"] = (
                        int(baseline_session_changes.get("saved_filtered", 0))
                        + matched_total
                    )
                except Exception:
                    pass
                try:
                    baseline_session_changes.setdefault(
                        "baseline_corrected_material", []
                    ).append((mat_val, baseline_function.upper(), updated_count))
                except Exception:
                    pass

            save_file_btn2.on_click(_save_file2)
            save_material_btn2.on_click(_save_material2)

            def _close_full2(_b=None):
                # Emit summary and collapse UI to summary output
                try:
                    lines = _session_summary_lines(
                        baseline_session_changes, context="baseline_correct_spectra"
                    )
                    _emit_session_summary(
                        baseline_output_area, lines, title="Session Summary (Baseline Correction)"
                    )
                except Exception:
                    pass
                try:
                    container.children = (baseline_output_area,)
                except Exception:
                    try:
                        plt.close("all")
                    except Exception:
                        pass

            close_btn2.on_click(_close_full2)

            # Observers for filters in full mode
            def _on_mat_full(change):
                """Full-mode observer: Material changed -> rebuild Conditions & Spectrum and refresh plot."""
                if change.get("name") == "value":
                    _rebuild_conditions_options()
                    _build_spectrum_options()
                    _plot_baseline(
                        **{
                            kk: ww.value
                            for kk, ww in baseline_parameter_widgets.items()
                        }
                    )
                    _refresh_mark_btns2()

            def _on_cond_full(change):
                """Full-mode observer: Conditions changed -> rebuild Spectrum and refresh plot."""
                if change.get("name") == "value":
                    _build_spectrum_options()
                    _plot_baseline(
                        **{
                            kk: ww.value
                            for kk, ww in baseline_parameter_widgets.items()
                        }
                    )
                    _refresh_mark_btns2()

            def _on_inc_full(change):
                """Full-mode observer: Include bad spectra toggled -> rebuild Spectrum and refresh plot."""
                if change.get("name") == "value":
                    _build_spectrum_options()
                    _plot_baseline(
                        **{
                            kk: ww.value
                            for kk, ww in baseline_parameter_widgets.items()
                        }
                    )
                    _refresh_mark_btns2()

            def _on_base_full(change):
                """Full-mode observer: Baseline method changed -> rebuild parameter UI or switch to MANUAL."""
                if change.get("name") != "value":
                    return
                new_val = str(change.get("new")).upper()
                nonlocal baseline_function, baseline_parameter_widgets
                if new_val == "MANUAL":
                    baseline_function = new_val
                    _build_manual_ui()
                    return
                # Rebuild the parameter UI inline for the selected baseline without re-entering the function
                baseline_function = new_val
                # Recreate parameter widgets for the new method
                baseline_parameter_widgets = {}
                params_local = _get_default_parameters(baseline_function)
                params_local = _cast_parameter_types(baseline_function, params_local)
                if baseline_function == "ARPLS":
                    baseline_parameter_widgets["lam"] = widgets.FloatSlider(
                        value=params_local.get("lam", 1e5),
                        min=1e4,
                        max=1e6,
                        step=1e4,
                        description="Smoothness (lam)",
                        readout_format=".1e",
                        continuous_update=False,
                        style={"description_width": "auto"},
                    )
                    baseline_parameter_widgets["max_iter"] = widgets.IntSlider(
                        value=params_local.get("max_iter", 50),
                        min=1,
                        max=200,
                        step=1,
                        description="Max Iterations",
                        continuous_update=False,
                        style={"description_width": "auto"},
                    )
                    baseline_parameter_widgets["tol"] = widgets.FloatSlider(
                        value=params_local.get("tol", 1e-3),
                        min=1e-6,
                        max=1e-1,
                        step=1e-4,
                        description="Tolerance",
                        readout_format=".1e",
                        continuous_update=False,
                        style={"description_width": "auto"},
                    )
                elif baseline_function == "IRSQR":
                    baseline_parameter_widgets["lam"] = widgets.FloatSlider(
                        value=params_local.get("lam", 1e6),
                        min=1e5,
                        max=1e7,
                        step=1e5,
                        description="Smoothness (lam)",
                        readout_format=".1e",
                        continuous_update=False,
                        style={"description_width": "auto"},
                    )
                    baseline_parameter_widgets["quantile"] = widgets.FloatSlider(
                        value=params_local.get("quantile", 0.05),
                        min=0.001,
                        max=0.5,
                        step=0.001,
                        description="Quantile",
                        readout_format=".3f",
                        continuous_update=False,
                        style={"description_width": "auto"},
                    )
                    baseline_parameter_widgets["num_knots"] = widgets.IntSlider(
                        value=params_local.get("num_knots", 100),
                        min=5,
                        max=500,
                        step=5,
                        description="Knots",
                        continuous_update=False,
                        style={"description_width": "auto"},
                    )
                    baseline_parameter_widgets["spline_degree"] = widgets.IntSlider(
                        value=params_local.get("spline_degree", 3),
                        min=1,
                        max=5,
                        step=1,
                        description="Spline Degree",
                        continuous_update=False,
                        style={"description_width": "auto"},
                    )
                    baseline_parameter_widgets["diff_order"] = widgets.IntSlider(
                        value=params_local.get("diff_order", 3),
                        min=1,
                        max=3,
                        step=1,
                        description="Differential Order",
                        continuous_update=False,
                        style={"description_width": "auto"},
                    )
                    baseline_parameter_widgets["max_iter"] = widgets.IntSlider(
                        value=params_local.get("max_iter", 100),
                        min=1,
                        max=1000,
                        step=1,
                        description="Max Iterations",
                        continuous_update=False,
                        style={"description_width": "auto"},
                    )
                    baseline_parameter_widgets["tol"] = widgets.FloatSlider(
                        value=params_local.get("tol", 1e-6),
                        min=1e-10,
                        max=1e-2,
                        step=1e-7,
                        description="Tolerance",
                        readout_format=".1e",
                        continuous_update=False,
                        style={"description_width": "auto"},
                    )
                elif baseline_function == "FABC":
                    baseline_parameter_widgets["lam"] = widgets.FloatSlider(
                        value=params_local.get("lam", 1e6),
                        min=1e4,
                        max=1e7,
                        step=1e5,
                        description="Smoothness (lam)",
                        readout_format=".1e",
                        continuous_update=False,
                        style={"description_width": "auto"},
                    )
                    try:
                        scale_default3 = int(
                            np.clip(ceil(optimize_window(y_values) / 2), 2, 500)
                        )
                    except Exception:
                        scale_default3 = 50
                    scale_val3 = params_local.get("scale") or scale_default3
                    baseline_parameter_widgets["scale"] = widgets.IntSlider(
                        value=int(scale_val3),
                        min=2,
                        max=500,
                        step=1,
                        description="Scale",
                        continuous_update=False,
                        style={"description_width": "auto"},
                    )
                    baseline_parameter_widgets["num_std"] = widgets.FloatSlider(
                        value=params_local.get("num_std", 3.0),
                        min=1.5,
                        max=4.5,
                        step=0.1,
                        description="Standard Deviations",
                        readout_format=".2f",
                        continuous_update=False,
                        style={"description_width": "auto"},
                    )
                    baseline_parameter_widgets["diff_order"] = widgets.IntSlider(
                        value=params_local.get("diff_order", 2),
                        min=1,
                        max=3,
                        step=1,
                        description="Differential Order",
                        continuous_update=False,
                        style={"description_width": "auto"},
                    )
                    baseline_parameter_widgets["min_length"] = widgets.IntSlider(
                        value=params_local.get("min_length", 2),
                        min=1,
                        max=6,
                        step=1,
                        description="Min Baseline Span Length",
                        continuous_update=False,
                        style={"description_width": "auto"},
                    )

                defaults_new = _get_default_parameters(baseline_function)
                rows = []
                for k, w in baseline_parameter_widgets.items():
                    rb = widgets.Button(
                        description="Reset",
                        button_style="info",
                        layout=widgets.Layout(width="70px", margin="0 0 6px 8px"),
                    )
                    if k == "scale":

                        def _reset_scale3(_b=None, w=w):
                            try:
                                w.value = int(
                                    np.clip(ceil(optimize_window(y_values) / 2), 2, 500)
                                )
                            except Exception:
                                pass

                        rb.on_click(_reset_scale3)
                    else:
                        rv = defaults_new.get(k, w.value)

                        def make_reset_f2(w2, val2):
                            return lambda _b=None: setattr(w2, "value", val2)

                        rb.on_click(make_reset_f2(w, rv))
                    rows.append(widgets.HBox([w, rb]))

                # Parameter details toggle
                try:
                    rows.append(_make_param_details_row(baseline_function))
                except Exception:
                    pass

                reset_all_btn3 = widgets.Button(
                    description="Reset All",
                    button_style="warning",
                    layout=widgets.Layout(width="90px", margin="0 0 6px 8px"),
                )

                def _reset_all3(_b=None):
                    for kk, ww in baseline_parameter_widgets.items():
                        if kk == "scale":
                            try:
                                ww.value = int(
                                    np.clip(ceil(optimize_window(y_values) / 2), 2, 500)
                                )
                            except Exception:
                                pass
                        elif kk in defaults_new:
                            ww.value = defaults_new[kk]

                reset_all_btn3.on_click(_reset_all3)

                save_file_btn3 = widgets.Button(
                    description="Save for file",
                    button_style="success",
                    layout=widgets.Layout(margin="10px 10px 0 0"),
                )
                save_material_btn3 = widgets.Button(
                    description="Save for material",
                    button_style="info",
                    layout=widgets.Layout(margin="10px 10px 0 0"),
                )
                mark_bad_btn3, mark_good_btn3, _refresh_mark_btns3 = (
                    _make_quality_controls(
                        FTIR_DataFrame, lambda: selected_row, margin="10px 10px 0 0"
                    )
                )
                close_btn3 = widgets.Button(
                    description="Close",
                    button_style="danger",
                    layout=widgets.Layout(margin="10px 0 0 0"),
                )

                # Track quality changes for session summary
                def _log_mark_bad3(_b=None):
                    try:
                        if selected_row is not None:
                            baseline_session_changes.setdefault("quality", []).append(
                                (selected_row.name, "bad")
                            )
                        if selected_row is not None and not include_bad_cb.value:
                            try:
                                spectrum_sel.unobserve(_on_spec_full, names="value")
                                spectrum_sel.options = [opt for opt in spectrum_sel.options if opt[1] != selected_row.name]
                                spectrum_sel.value = None if spectrum_sel.options else None
                                spectrum_sel.observe(_on_spec_full, names="value")
                            except Exception:
                                pass
                    except Exception:
                        pass

                def _log_mark_good3(_b=None):
                    try:
                        if selected_row is not None:
                            baseline_session_changes.setdefault("quality", []).append(
                                (selected_row.name, "good")
                            )
                        if selected_row is not None and not include_bad_cb.value:
                            try:
                                ids = [v for (_l,v) in spectrum_sel.options]
                                if selected_row.name not in ids:
                                    spectrum_sel.unobserve(_on_spec_full, names="value")
                                    _mat = selected_row.get('Material','')
                                    _cond = selected_row.get('Conditions', selected_row.get('Condition',''))
                                    _t = selected_row.get('Time','')
                                    _fn = selected_row.get('File Name','')
                                    _label = f"{_mat} | {_cond} | T={_t} | {_fn}"
                                    spectrum_sel.options = spectrum_sel.options + [(_label, selected_row.name)] if spectrum_sel.options else [(_label, selected_row.name)]
                                    spectrum_sel.value = selected_row.name
                                    spectrum_sel.observe(_on_spec_full, names="value")
                            except Exception:
                                pass
                    except Exception:
                        pass

                try:
                    mark_bad_btn3.on_click(_log_mark_bad3)
                    mark_good_btn3.on_click(_log_mark_good3)
                except Exception:
                    pass

                # Wire save/close handlers (quality handled by helper)
                save_file_btn3.on_click(_save_file2)
                save_material_btn3.on_click(_save_material2)
                close_btn3.on_click(_close_full2)

                mark_row3 = widgets.HBox([mark_bad_btn3, mark_good_btn3])
                # Attach reset-all (rebuild path) to bottom-most parameter row
                try:
                    for _j in range(len(rows) - 1, -1, -1):
                        _row = rows[_j]
                        if isinstance(_row, widgets.HBox) and hasattr(_row, "children") and len(_row.children) == 2:
                            rows[_j] = widgets.HBox(list(_row.children) + [reset_all_btn3])
                            break
                except Exception:
                    pass
                footer3 = widgets.HBox(
                    [save_file_btn3, save_material_btn3, close_btn3]
                )
                # Bordered parameter cluster (rebuild path) for clarity
                try:
                    param_cluster_new = widgets.VBox(
                        rows,
                        layout=widgets.Layout(
                            border="1px solid #aaa",
                            padding="6px",
                            margin="6px 0",
                        ),
                    )
                except Exception:
                    param_cluster_new = widgets.VBox(rows)
                ui_full_new = widgets.VBox(
                    [
                        widgets.HBox(
                            [
                                material_dd,
                                conditions_dd,
                                baseline_dd,
                            ]
                        ),
                        widgets.HBox([spectrum_sel, include_bad_cb]),
                        param_cluster_new,
                        mark_row3,
                        footer3,
                    ]
                )

                # Swap UI inline
                try:
                    container.children = (ui_full_new, baseline_output_area)
                except Exception:
                    display(widgets.VBox([ui_full_new, baseline_output_area]))

                # Observe parameter changes for live updates
                def _on_param_change3(change):
                    """Full-mode: parameter widget changed -> update baseline preview with current values."""
                    if change.get("name") == "value":
                        _plot_baseline(
                            **{
                                kk: ww.value
                                for kk, ww in baseline_parameter_widgets.items()
                            }
                        )

                for _pw in baseline_parameter_widgets.values():
                    try:
                        _pw.observe(_on_param_change3, names="value")
                    except Exception:
                        pass

                # Refresh mark buttons and plot with new method
                try:
                    _refresh_mark_btns3()
                except Exception:
                    pass
                _plot_baseline(
                    **{kk: ww.value for kk, ww in baseline_parameter_widgets.items()}
                )

            material_dd.observe(_on_mat_full, names="value")
            conditions_dd.observe(_on_cond_full, names="value")
            include_bad_cb.observe(_on_inc_full, names="value")
            baseline_dd.observe(_on_base_full, names="value")

            def _on_spec_full(change):
                """Full-mode observer: Spectrum changed -> update selection, recompute scale (FABC), refresh plot."""
                if change.get("name") == "value" and change.get("new") is not None:
                    try:
                        sel_idx2 = change.get("new")
                        rsel2 = FTIR_DataFrame.loc[sel_idx2]
                        nonlocal selected_row, x_values, y_values, material, current_idx_bc
                        selected_row = rsel2
                        current_idx_bc = sel_idx2
                        material = rsel2.get("Material", material)
                        x_values = (
                            ast.literal_eval(rsel2["X-Axis"])
                            if isinstance(rsel2["X-Axis"], str)
                            else rsel2["X-Axis"]
                        )
                        y_values = (
                            ast.literal_eval(rsel2["Raw Data"])
                            if isinstance(rsel2["Raw Data"], str)
                            else rsel2["Raw Data"]
                        )
                        y_values = np.array(y_values, dtype=float)
                        # If currently using MANUAL baseline, switch into the integrated inline manual UI
                        if baseline_function.upper() == "MANUAL":
                            _build_manual_ui()
                            return
                        if (
                            baseline_function.upper() == "FABC"
                            and "scale" in baseline_parameter_widgets
                        ):
                            try:
                                new_scale3 = int(
                                    np.clip(ceil(optimize_window(y_values) / 2), 2, 500)
                                )
                                baseline_parameter_widgets["scale"].value = new_scale3
                            except Exception:
                                pass
                        _plot_baseline(
                            **{
                                kk: ww.value
                                for kk, ww in baseline_parameter_widgets.items()
                            }
                        )
                        _refresh_mark_btns2()
                        _set_session_selection(
                            material=selected_row.get("Material"),
                            conditions=selected_row.get("Conditions"),
                            time=selected_row.get("Time"),
                        )
                    except Exception:
                        pass

            spectrum_sel.observe(_on_spec_full, names="value")
            # Build final UI inside existing container (replace children to avoid flicker/disappearance)
            if baseline_function.upper() != "MANUAL":
                mark_row2 = widgets.HBox([mark_bad_btn2, mark_good_btn2])
                # Attach reset-all button to the bottom-most parameter row (after its individual Reset)
                try:
                    for _j in range(len(widget_rows_full) - 1, -1, -1):
                        _row = widget_rows_full[_j]
                        if isinstance(_row, widgets.HBox) and hasattr(_row, "children") and len(_row.children) == 2:
                            # Append the global reset button to this HBox
                            widget_rows_full[_j] = widgets.HBox(list(_row.children) + [reset_all_btn2])
                            break
                except Exception:
                    pass
                controls_footer2 = widgets.HBox(
                    [save_file_btn2, save_material_btn2, close_btn2]
                )
                # Build UI with bordered plot+mark section below parameter controls
                # Bordered parameter cluster for clarity (individual rows + details + global reset)
                try:
                    param_cluster_full = widgets.VBox(
                        widget_rows_full,
                        layout=widgets.Layout(
                            border="1px solid #aaa",
                            padding="6px",
                            margin="6px 0",
                        ),
                    )
                except Exception:
                    param_cluster_full = widgets.VBox(widget_rows_full)
                ui_controls_full = widgets.VBox(
                    [
                        widgets.HBox(
                            [
                                material_dd,
                                conditions_dd,
                                baseline_dd,
                            ]
                        ),
                        widgets.HBox([spectrum_sel, include_bad_cb]),
                        param_cluster_full,
                        controls_footer2,
                    ]
                )
                plot_and_mark_full = widgets.VBox(
                    [baseline_output_area, mark_row2],
                    layout=widgets.Layout(
                        border="1px solid #ccc",
                        padding="8px",
                        margin="6px 0",
                    ),
                )
                try:
                    container.children = (ui_controls_full, plot_and_mark_full)
                except Exception:
                    display(widgets.VBox([ui_controls_full, plot_and_mark_full]))
                _refresh_mark_btns2()

                # Parameter slider -> live plot updates
                def _on_param_change(change):
                    """Full-mode: parameter widget changed -> update baseline preview with current values."""
                    if change.get("name") == "value":
                        _plot_baseline(
                            **{
                                kk: ww.value
                                for kk, ww in baseline_parameter_widgets.items()
                            }
                        )

                for _pw in baseline_parameter_widgets.values():
                    try:
                        _pw.observe(_on_param_change, names="value")
                    except Exception:
                        pass
                # Initial plot
                _plot_baseline(
                    **{kk: ww.value for kk, ww in baseline_parameter_widgets.items()}
                )
            else:
                # If MANUAL, delegate to manual UI builder
                _build_manual_ui()

    def _on_close_min(_b=None):
        """Close minimal UI and emit a concise per-session summary (runs regardless of spectrum selection)."""
        try:
            lines = _session_summary_lines(
                baseline_session_changes, context="baseline_correct_spectra"
            )
            _emit_session_summary(
                baseline_output_area, lines, title="Session Summary (Baseline Correction)"
            )
        except Exception:
            pass
        try:
            container.children = (baseline_output_area,)
        except Exception:
            try:
                plt.close("all")
            except Exception:
                pass

    # Attach minimal-mode observers immediately so dropdowns work before spectrum selection
    material_dd.observe(_on_mat_min, names="value")
    conditions_dd.observe(_on_cond_min, names="value")
    include_bad_cb.observe(_on_inc_min, names="value")
    baseline_dd.observe(_on_base_min, names="value")
    spectrum_sel.observe(_on_spec_min, names="value")
    close_btn.on_click(_on_close_min)

    return FTIR_DataFrame


def populate_output_dictionary(
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
                            "σg": 0,
                            "σl": 0,
                            "α": 0,
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
        context="FTIR_DataFrame (populate_output_dictionary)",
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

        # Ensure special "unexposed" structure is present with per-condition and final
        # Build the per-condition keys from all conditions except 'unexposed'
        try:
            exposure_conditions = [
                c for c in cond_map.keys() if str(c).strip().lower() != "unexposed"
            ]
        except Exception:
            exposure_conditions = []
        # Revised unexposed struct: always store 'A' as a list to allow multiple entries over time
        # Convert any legacy scalar values to list form later during merges.
        unexposed_struct = {
            "per-condition": {c: {"A": []} for c in sorted(exposure_conditions)},
            "final": {"A": []},
        }
        # Override any DataFrame-provided 'unexposed' to use the new structure
        cond_map["unexposed"] = unexposed_struct

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
                "σg": 0,
                "σl": 0,
                "α": 0,
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
                    f"[populate_output_dictionary] Overwrite blocked for {code}.name: keeping existing '{payload.get('name')}', observed '{material}'."
                )
        if "alias" not in payload:
            payload["alias"] = material
        else:
            if str(payload.get("alias")) != str(material):
                print(
                    f"[populate_output_dictionary] Overwrite blocked for {code}.alias: keeping existing '{payload.get('alias')}', observed '{material}'."
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
        peak1.setdefault("σg", 0)
        peak1.setdefault("σl", 0)
        peak1.setdefault("α", 0)

        # Merge conditions: union times; keep existing A arrays intact. If an existing
        # condition is found, we do not overwrite its values—report actions taken.
        existing_conditions = peak1.get("conditions", {})
        if not isinstance(existing_conditions, dict):
            existing_conditions = {}
        for cond_str, new_payload in cond_map.items():
            # Special handling for the new-format 'unexposed' block
            if str(cond_str).strip().lower() == "unexposed":
                # Merge/initialize unexposed with per-condition keys for all exposure conditions
                try:
                    existing_unexp = existing_conditions.get("unexposed", {})
                    if not isinstance(existing_unexp, dict):
                        existing_unexp = {}
                except Exception:
                    existing_unexp = {}
                # Build the set of exposure condition names from both existing and new
                try:
                    all_cond_names = set(
                        k
                        for k in list(existing_conditions.keys())
                        + list(cond_map.keys())
                        if str(k).strip().lower() != "unexposed"
                    )
                except Exception:
                    all_cond_names = set()
                per_cond_old = existing_unexp.get("per-condition", {})
                if not isinstance(per_cond_old, dict):
                    per_cond_old = {}
                per_cond_new = {}
                for cname in sorted(all_cond_names):
                    try:
                        # Preserve existing scalar A if present; else default 0
                        a_val = per_cond_old.get(cname, {}).get("A", 0)
                        # Coerce non-numeric to 0
                        try:
                            a_val = float(a_val)
                        except Exception:
                            a_val = 0
                        per_cond_new[cname] = {"A": 0 if a_val is None else a_val}
                    except Exception:
                        per_cond_new[cname] = {"A": 0}
                final_old = existing_unexp.get("final", {})
                if not isinstance(final_old, dict):
                    final_old = {}
                try:
                    final_a = final_old.get("A", 0)
                    try:
                        final_a = float(final_a)
                    except Exception:
                        final_a = 0
                except Exception:
                    final_a = 0
                existing_conditions["unexposed"] = {
                    "per-condition": per_cond_new,
                    "final": {"A": 0 if final_a is None else final_a},
                }
                continue
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
                        "[populate_output_dictionary] Existing entry preserved for "
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


def normalize_spectra(FTIR_DataFrame, filepath=None):
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
    cond_col = _conditions_column_name(FTIR_DataFrame)

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

    materials, conditions = _extract_material_condition_lists(
        df_all, exclude_unexposed=True
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

    # Helper: perform normalization for a material
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
    # New user-facing display mode toggle: Single vs Time-series
    display_mode = widgets.ToggleButtons(
        options=[("Single spectrum", "single"), ("Time-series", "series")],
        value="single",
        description="Display",
        layout=widgets.Layout(width="40%"),
    )
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
    # Track currently displayed spectrum independently of dropdown so a bad-marked spectrum can remain visible
    current_idx = None

    # --- Plot ---
    # Initialize figure (name will be updated per selection depending on data used)
    fig = go.FigureWidget(data=[go.Scatter(x=[], y=[], mode="lines", name="Spectrum")])
    fig.update_layout(
        title="Select Normalization Range",
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="Absorbance (AU)",
    )

    def _clear_selection_visuals():
        fig.layout.shapes = ()

    def _current_y_bounds():
        """Return (ymin, ymax) for selection visuals.

        In time-series mode, compute from all plotted traces; otherwise use current y_data.
        """
        try:
            if display_mode.value == "series" and len(fig.data) > 0:
                ys = []
                for tr in fig.data:
                    try:
                        ys.extend([float(v) for v in tr.y if v is not None and np.isfinite(v)])
                    except Exception:
                        pass
                if ys:
                    return float(np.nanmin(ys)), float(np.nanmax(ys))
        except Exception:
            pass
        # Fallback to current single-spectrum data bounds
        y0min = float(np.nanmin(y_data)) if len(y_data) else 0.0
        y0max = float(np.nanmax(y_data)) if len(y_data) else 1.0
        return y0min, y0max

    def _draw_first_click(x0: float):
        y0min, y0max = _current_y_bounds()
        vline = dict(
            type="line",
            x0=x0,
            x1=x0,
            y0=y0min,
            y1=y0max,
            line=dict(color="red", dash="dot"),
            name="norm_vline_first",
        )
        fig.add_shape(vline)

    def _draw_selection_visuals(x0, x1):
        y0min, y0max = _current_y_bounds()
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

    # Decide interaction mode (click vs slider) based on Colab detection.
    range_slider = None
    range_lo_text = None
    range_hi_text = None
    range_apply_btn = None
    if _IN_COLAB:
        try:
            # Determine global xmin/xmax across all spectra for slider bounds.
            low_vals, high_vals = [], []
            for idx_tmp in FTIR_DataFrame.index:
                xv_tmp = _parse_seq(FTIR_DataFrame.loc[idx_tmp].get("X-Axis"))
                if xv_tmp:
                    low_vals.append(np.nanmin(xv_tmp))
                    high_vals.append(np.nanmax(xv_tmp))
            if low_vals and high_vals:
                global_low = float(np.nanmin(low_vals))
                global_high = float(np.nanmax(high_vals))
            else:
                global_low, global_high = 0.0, 1.0
        except Exception:
            global_low, global_high = 0.0, 1.0
        step = (global_high - global_low) / 1000.0 or 1.0
        range_slider = widgets.FloatRangeSlider(
            value=[global_low, global_high],
            min=global_low,
            max=global_high,
            step=step,
            description="Range (cm⁻¹)",
            continuous_update=False,
            readout_format=".1f",
            layout=widgets.Layout(width="90%"),
        )
        range_lo_text = widgets.FloatText(
            value=global_low,
            description="Low (cm⁻¹)",
            layout=widgets.Layout(width="160px"),
        )
        range_hi_text = widgets.FloatText(
            value=global_high,
            description="High (cm⁻¹)",
            layout=widgets.Layout(width="160px"),
        )
        range_apply_btn = widgets.Button(
            description="Apply",
            button_style="info",
            layout=widgets.Layout(width="90px"),
        )
        range_help = widgets.HTML(
            "<span style='color:#555;font-size:12px;'>Colab: Type Low/High and click Apply, or drag the slider. Selection is clamped to data bounds.</span>"
        )

        def _on_range_slider(change):
            if change.get("name") != "value":
                return
            try:
                lo, hi = change.get("new")
                # Sync text boxes
                try:
                    range_lo_text.value = float(lo)
                    range_hi_text.value = float(hi)
                except Exception:
                    pass
                selected_points.clear()
                selected_points.extend([float(lo), float(hi)])
                _clear_selection_visuals()
                _draw_selection_visuals(float(lo), float(hi))
                with msg_out:
                    clear_output(wait=True)
                    lo2, hi2 = (min(lo, hi), max(lo, hi))
                    print(
                        f"Selected normalization range (slider): [{lo2:.3f}, {hi2:.3f}] cm⁻¹"
                    )
            except Exception:
                pass

        def _apply_text_range(_b=None):
            try:
                lo = float(range_lo_text.value)
                hi = float(range_hi_text.value)
            except Exception:
                with msg_out:
                    clear_output(wait=True)
                    print("Invalid numeric input for Low/High.")
                return
            # Clamp/order
            if lo > hi:
                lo, hi = hi, lo
            try:
                lo_clamped = max(global_low, min(global_high, lo))
                hi_clamped = max(global_low, min(global_high, hi))
            except Exception:
                lo_clamped, hi_clamped = lo, hi
            # Update slider (which triggers its observer to update visuals & messages)
            try:
                range_slider.value = [float(lo_clamped), float(hi_clamped)]
            except Exception:
                pass
            # In case observer didn't fire (same values), manually reflect
            if (
                len(selected_points) != 2
                or selected_points[0] != lo_clamped
                or selected_points[1] != hi_clamped
            ):
                selected_points.clear()
                selected_points.extend([float(lo_clamped), float(hi_clamped)])
                _clear_selection_visuals()
                _draw_selection_visuals(float(lo_clamped), float(hi_clamped))
                with msg_out:
                    clear_output(wait=True)
                    print(
                        f"Selected normalization range (typed): [{lo_clamped:.3f}, {hi_clamped:.3f}] cm⁻¹"
                    )

        try:
            range_apply_btn.on_click(_apply_text_range)
        except Exception:
            pass

        range_slider.observe(_on_range_slider, names="value")
    else:
        try:
            fig.data[0].on_click(_on_click)
        except Exception:
            pass

    def _current_range():
        if len(selected_points) != 2:
            return None
        a, b = selected_points
        return [float(min(a, b)), float(max(a, b))]

    def _get_xy(idx):
        """Return (x, y, used_baseline) for the selected row.

        Uses baseline-corrected data when available; falls back to raw data otherwise.
        """
        row = FTIR_DataFrame.loc[idx]
        x = row.get("X-Axis")
        y_bc = row.get("Baseline-Corrected Data")
        raw = row.get("Raw Data")
        # Parse x if stored as literal string
        if isinstance(x, str):
            try:
                x = ast.literal_eval(x)
            except Exception:
                pass
        # Normalize baseline-corrected value
        if isinstance(y_bc, str):
            if y_bc.strip().lower() == "nan":
                y_bc = None
            else:
                try:
                    y_bc = ast.literal_eval(y_bc)
                except Exception:
                    # leave as original string if cannot eval
                    pass
        # If baseline is a pandas Series convert to list
        try:
            import pandas as _pd
            if isinstance(y_bc, _pd.Series):
                y_bc = y_bc.tolist()
        except Exception:
            pass
        # Determine validity
        y_bc_valid = False
        try:
            import numpy as _np
            if isinstance(y_bc, _np.ndarray):
                y_bc_valid = y_bc.ndim == 1 and y_bc.size > 1 and not _np.isnan(y_bc).all()
            elif isinstance(y_bc, (list, tuple)):
                y_bc_valid = len(y_bc) > 1 and not all(
                    (isinstance(v, float) and np.isnan(v)) for v in y_bc
                )
        except Exception:
            if isinstance(y_bc, (list, tuple)) and len(y_bc) > 1:
                y_bc_valid = True
        # Fallback to raw if baseline invalid
        y = y_bc if y_bc_valid else raw
        # Parse raw if needed for potential fallback
        if y is raw and isinstance(raw, str):
            try:
                y = ast.literal_eval(raw)
            except Exception:
                pass
        if isinstance(y, str):
            try:
                y = ast.literal_eval(y)
            except Exception:
                pass
        # Final guard: fallback again if y still not a proper sequence
        if not isinstance(y, (list, tuple, np.ndarray)) or (isinstance(y, (list, tuple)) and len(y) <= 1):
            # Attempt to parse raw more aggressively
            raw_fallback = raw
            if isinstance(raw_fallback, str):
                try:
                    raw_fallback = ast.literal_eval(raw_fallback)
                except Exception:
                    pass
            if isinstance(raw_fallback, (list, tuple, np.ndarray)) and len(raw_fallback) > 1:
                y = raw_fallback
                y_bc_valid = False
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float), bool(y_bc_valid)

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
            # If in time-series mode, also clear the plot so it's obvious
            try:
                if display_mode.value == "series":
                    with fig.batch_update():
                        fig.data = tuple()
                        fig.update_layout(title="Time Series | No data to display")
            except Exception:
                pass
            return
        spectrum_sel.options = opts
        # Preserve existing selection if still valid; otherwise require manual user selection
        if spectrum_sel.value not in [v for _, v in opts]:
            spectrum_sel.value = None
            # Only show dropdown guidance in single-spectrum mode
            if display_mode.value == "single":
                with info_out:
                    clear_output(wait=True)
                    print("Select a spectrum from the dropdown to begin normalization.")
        # If time-series mode is active, refresh plot to reflect current filters
        try:
            if display_mode.value == "series":
                _update_plot_for_selection()
        except Exception:
            pass

    def _update_plot_for_selection(*_):
        idx = spectrum_sel.value
        # Time-series mode: plot all spectra currently present in dropdown options (filtered set)
        if display_mode.value == "series":
            # Require specific Material and (if present) Conditions before plotting
            try:
                if material_dd.value == "any" or (cond_col and conditions_dd.value == "any"):
                    with info_out:
                        clear_output(wait=True)
                        print("Select specific Material and Conditions (not 'any') to view time-series.")
                    try:
                        mark_row.layout.display = "none"
                    except Exception:
                        pass
                    with fig.batch_update():
                        fig.data = tuple()
                        fig.update_layout(title="Time Series | Awaiting selections")
                    return
            except Exception:
                pass
            # Build filtered DataFrame directly (dropdown hidden in series mode)
            try:
                filtered_ts = FTIR_DataFrame.copy()
                if material_dd.value != "any":
                    filtered_ts = filtered_ts[filtered_ts.get("Material", pd.Series([])).astype(str) == str(material_dd.value)]
                if cond_col and conditions_dd.value != "any":
                    cond_series = filtered_ts.get(cond_col, pd.Series([None]*len(filtered_ts))).astype(str)
                    sel_cond = str(conditions_dd.value)
                    mask = (cond_series == sel_cond) | (cond_series.str.strip().str.lower() == "unexposed")
                    filtered_ts = filtered_ts[mask]
                if not include_bad_cb.value and len(filtered_ts):
                    try:
                        filtered_ts = filtered_ts[_quality_good_mask(filtered_ts)]
                    except Exception:
                        pass
                if "Time" in filtered_ts.columns:
                    try:
                        filtered_ts = filtered_ts.sort_values(by="Time")
                    except Exception:
                        pass
            except Exception:
                filtered_ts = FTIR_DataFrame.head(0)
            if filtered_ts is None or len(filtered_ts) == 0:
                with info_out:
                    clear_output(wait=True)
                    print("No spectra match the current filters.")
                try:
                    mark_row.layout.display = "none"
                except Exception:
                    pass
                with fig.batch_update():
                    fig.data = tuple()
                    fig.update_layout(title="Time Series | No spectra")
                return
            series_data = []  # list of dicts: {x, y, name}
            count_plotted = 0
            for i in filtered_ts.index:
                try:
                    r = FTIR_DataFrame.loc[i]
                    x_arr, y_arr, _used_bc = _get_xy(i)
                    if x_arr.size < 2 or y_arr.size < 2 or x_arr.size != y_arr.size:
                        continue
                    tval = r.get("Time", "?")
                    cond_val = r.get(cond_col, "?") if cond_col else None
                    parts = [f"t={tval}"]
                    if cond_col:
                        parts.append(str(cond_val))
                    name = " | ".join(parts)
                    series_data.append({
                        "x": x_arr.tolist(),
                        "y": y_arr.tolist(),
                        "name": name,
                    })
                    count_plotted += 1
                except Exception:
                    continue
            if len(series_data) == 0:
                with info_out:
                    clear_output(wait=True)
                    print("No spectra match the current filters.")
                try:
                    mark_row.layout.display = "none"
                except Exception:
                    pass
                with fig.batch_update():
                    fig.data = tuple()
                    fig.update_layout(title="Time Series | No data to display")
                # Still reveal container so user sees the state
                try:
                    bordered_plot.layout.display = ""
                except Exception:
                    pass
                try:
                    action_row.layout.display = "none"
                except Exception:
                    pass
                return
            title_mat = material_dd.value
            title_cond_sel = conditions_dd.value if (cond_col and conditions_dd.value != "any") else None
            with fig.batch_update():
                # Shrink to desired number of traces by taking a subset (allowed by FigureWidget)
                while len(fig.data) > len(series_data):
                    fig.data = fig.data[:-1]
                # Update existing traces or add new ones
                for i, d in enumerate(series_data):
                    if i < len(fig.data):
                        fig.data[i].x = d["x"]
                        fig.data[i].y = d["y"]
                        fig.data[i].mode = "lines"
                        fig.data[i].name = d["name"]
                    else:
                        fig.add_scatter(x=d["x"], y=d["y"], mode="lines", name=d["name"])
                title = f"Time Series | {title_mat}"
                if title_cond_sel:
                    title += f" | Condition: {title_cond_sel} (+ unexposed)"
                fig.update_layout(title=title)
            # Attach click handlers to all traces (non-Colab) for range selection
            try:
                if not _IN_COLAB:
                    for tr in fig.data:
                        try:
                            tr.on_click(_on_click)
                        except Exception:
                            pass
            except Exception:
                pass
            _clear_selection_visuals()
            selected_points.clear()
            try:
                mark_row.layout.display = "none"
            except Exception:
                pass
            with info_out:
                clear_output(wait=True)
                print(f"Plotted {count_plotted} trace(s). Click two points to select the tip of the normalization peak.")
            # Reveal plot container and show Save/Redo in time-series mode
            try:
                bordered_plot.layout.display = ""
            except Exception:
                pass
            try:
                action_row.layout.display = ""
            except Exception:
                pass
            return
        # Single-spectrum mode below
        if idx is None:
            # Leaving time-series mode or no spectrum chosen yet -> clear figure for clarity
            try:
                mark_row.layout.display = "none"
            except Exception:
                pass
            try:
                if display_mode.value != "series":
                    with fig.batch_update():
                        fig.data = tuple()
                        fig.update_layout(title="Select a spectrum (or enable Time-Series)")
                    with info_out:
                        clear_output(wait=True)
                        print("Choose a spectrum from the dropdown to plot, or enable Time-Series.")
            except Exception:
                pass
            return
        # Update current displayed index
        nonlocal current_idx
        current_idx = idx
        nonlocal x_data, y_data
        x_arr, y_arr, used_bc = _get_xy(idx)
        x_data, y_data = x_arr, y_arr
        # update trace (create if missing)
        try:
            if len(fig.data) == 0:
                fig.add_scatter(x=x_data.tolist(), y=y_data.tolist(), mode="lines", name="Spectrum")
            else:
                fig.data[0].x = x_data.tolist()
                fig.data[0].y = y_data.tolist()
        except Exception:
            pass
        try:
            fig.data[0].name = (
                "Baseline-Corrected" if used_bc else "Raw Data (baseline not saved)"
            )
        except Exception:
            pass
        _clear_selection_visuals()
        selected_points.clear()
        # update title and info
        mat_val = FTIR_DataFrame.loc[idx].get("Material", "?")
        fig.update_layout(title=f"Select Normalization Range | Material: {mat_val}")
        with info_out:
            clear_output(wait=True)
            print(f"Plotting: {_row_filepath(idx)}")
            if used_bc:
                print("Using baseline-corrected data for range selection.")
            else:
                print(
                    "Baseline-corrected data unavailable; showing raw data (apply baseline first)."
                )
            print("Click two points to select the tip of the normalization peak.")
        # Reveal plot and action buttons for single-spectrum mode
        try:
            bordered_plot.layout.display = ""
        except Exception:
            pass
        try:
            action_row.layout.display = ""
        except Exception:
            pass
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
        # Show mark row now that a spectrum is actively plotted
        try:
            mark_row.layout.display = ""
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

    # Delayed plotting: wait for user to manually choose a spectrum.

    # --- Buttons ---
    # Widen Save button so full description is visible (was truncated previously)
    save_mat_btn = widgets.Button(
        description="Save range and normalize material",
        button_style="info",
        layout=widgets.Layout(width="300px"),
    )
    redo_btn = widgets.Button(description="Redo", button_style="warning")
    cancel_btn = widgets.Button(description="Close", button_style="danger")
    # --- Change tracking (session summary on close) ---
    # Store structured events so we can summarize intelligently.
    _norm_changes = {
        "range_material": [],  # list[(material, count_rows, range_str)] saved per material
        "normalized_materials": [],  # list[(material, updated_count, skipped_count)]
        "quality": [],  # list[(idx, new_quality)]
    }
    # Use shared quality controls (mutually exclusive buttons)
    mark_bad_btn, mark_good_btn, _refresh_mark_buttons = _make_quality_controls(
        FTIR_DataFrame,
        lambda: (
            FTIR_DataFrame.loc[current_idx]
            if current_idx is not None
            else None
        ),
    )
    # Main control row excludes mark buttons; they go on their own row as a pair
    # Separate action buttons from Close so Close is always visible
    action_row = widgets.HBox([save_mat_btn, redo_btn])
    close_row = widgets.HBox([cancel_btn])
    mark_row = widgets.HBox([mark_bad_btn, mark_good_btn])
    # Hide Mark buttons until a spectrum is selected; place below plot inside border
    try:
        mark_row.layout.display = "none"
    except Exception:
        pass
    bordered_plot = widgets.VBox(
        [fig, mark_row],
        layout=widgets.Layout(border="1px solid #ccc", padding="8px", margin="6px 0"),
    )
    # Hide plot and action buttons until actual plotting occurs
    try:
        bordered_plot.layout.display = "none"
    except Exception:
        pass
    try:
        action_row.layout.display = "none"
    except Exception:
        pass

    # Refresh function provided by helper; keep name for local uses

    def _finalize_and_clear():
        # Detach click handler and close figure
        try:
            fig.data[0].on_click(None)
        except Exception:
            pass
        try:
            fig.close()
        except Exception:
            pass
        # Close widgets created in this UI
        widget_list = [
            save_mat_btn,
            redo_btn,
            cancel_btn,
            mark_bad_btn,
            mark_good_btn,
            spectrum_sel,
            material_dd,
            conditions_dd,
            include_bad_cb,
            display_mode,
            info_out,  # ensure info output removed after close
        ]
        if range_slider is not None:
            widget_list.append(range_slider)
        for w in widget_list:
            try:
                w.close()
            except Exception:
                pass
        # Close layout containers (including bordered plot frame) so only summary msg_out remains
        for container in (controls_row, mode_row, spectrum_row, mark_row, action_row, close_row, bordered_plot):
            try:
                container.close()
            except Exception:
                pass


    def _save_for_this_material(_b=None):
        # Determine material context and validate selection range
        rng = _current_range()
        if rng is None:
            with msg_out:
                clear_output(wait=True)
                print("Please select two points before saving.")
            return
        mat = None
        if display_mode.value == "series":
            # In time-series mode, use the chosen Material directly
            try:
                mat = material_dd.value if material_dd.value != "any" else None
            except Exception:
                mat = None
        else:
            # In single-spectrum mode, infer from the selected row
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
                print("Material not determined; select a material/spectrum first.")
            return
        mask = FTIR_DataFrame["Material"] == mat
        FTIR_DataFrame.loc[mask, target_col] = str(rng)
        try:
            _norm_changes["range_material"].append((mat, int(mask.sum()), str(rng)))
        except Exception:
            pass
        # Normalize immediately after saving range for material
        try:
            _normalize(mat)
            try:
                updated_count = int((FTIR_DataFrame["Material"] == mat).sum())
            except Exception:
                updated_count = 0
            try:
                _norm_changes["normalized_materials"].append((mat, updated_count, None))
            except Exception:
                pass
        except Exception:
            pass
        with msg_out:
            clear_output(wait=True)
            print(f"Saved normalization peak range {rng} for material '{mat}' and normalized its spectra.")


    def _redo(_b=None):
        selected_points.clear()
        _clear_selection_visuals()
        with msg_out:
            clear_output(wait=True)
            print("Selection cleared. Click two points to select the tip of the normalization peak.")

    def _close(_b=None):
        # Clear all prior info/messages so only session summary remains
        try:
            with info_out:
                clear_output(wait=True)
        except Exception:
            pass
        try:
            with msg_out:
                clear_output(wait=True)
        except Exception:
            pass
        # Build and display session summary, then remove other widgets
        try:
            lines = _session_summary_lines(_norm_changes, context="normalization")
            _emit_session_summary(msg_out, lines, title="Session summary (Normalization)")
        except Exception:
            pass
        # Close remaining interactive widgets except msg_out (preserve summary)
        _finalize_and_clear()

    # Add extra UI effects after helper toggles quality
    def _post_mark_update(status_label: str):
        nonlocal current_idx
        try:
            idx = current_idx
            if idx is None:
                return
            with msg_out:
                clear_output(wait=True)
                print(f"Marked row {idx} as {status_label} quality.")
            try:
                _norm_changes["quality"].append((idx, status_label))
            except Exception:
                pass
            # Decouple dropdown from plot via centralized helper
            if status_label in ("bad", "good"):
                try:
                    # Keep current plot index stable so display doesn't flicker
                    if status_label == "bad":
                        current_idx = idx
                    _quality_dropdown_handle(
                        status_label,
                        dropdown=spectrum_sel,
                        include_bad_flag=include_bad_cb.value,
                        idx=idx,
                        label_builder=_row_label,
                        observer_fn=_update_plot_for_selection,
                    )
                except Exception:
                    pass
            else:
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

    # Wire events
    material_dd.observe(_rebuild_spectrum_options, names="value")
    conditions_dd.observe(_rebuild_spectrum_options, names="value")
    spectrum_sel.observe(_update_plot_for_selection, names="value")
    include_bad_cb.observe(_rebuild_spectrum_options, names="value")
    include_bad_cb.observe(_update_plot_for_selection, names="value")
    # Ensure time-series re-plots immediately on filter changes
    material_dd.observe(_update_plot_for_selection, names="value")
    conditions_dd.observe(_update_plot_for_selection, names="value")
    save_mat_btn.on_click(_save_for_this_material)
    redo_btn.on_click(_redo)
    # The helper already wires core quality changes; add post-effects
    mark_bad_btn.on_click(lambda _b=None: _post_mark_update("bad"))
    mark_good_btn.on_click(lambda _b=None: _post_mark_update("good"))
    cancel_btn.on_click(_close)

    # New behaviors: mode toggle and reveal-on-plot
    def _on_display_mode(change):
        is_series = change.get("new") == "series"
        # Toggle spectrum selector visibility
        try:
            spectrum_row.layout.display = "none" if is_series else ""
        except Exception:
            pass
        # Hide plot and action buttons until an actual plot is produced
        try:
            bordered_plot.layout.display = "none"
        except Exception:
            pass
        try:
            action_row.layout.display = "none"
        except Exception:
            pass
        try:
            mark_row.layout.display = "none"
        except Exception:
            pass
        # Auto-trigger update when switching to time-series
        if is_series:
            try:
                _rebuild_spectrum_options()
                _update_plot_for_selection(None)
            except Exception:
                pass

    def _on_spectrum_change_reveal(change):
        # Reveal plot and action buttons when a single spectrum is chosen
        if display_mode.value == "single" and change.get("new") is not None:
            try:
                bordered_plot.layout.display = ""
            except Exception:
                pass
            try:
                action_row.layout.display = ""
            except Exception:
                pass
            try:
                mark_row.layout.display = ""
            except Exception:
                pass

    def _on_filter_change_reveal(_change):
        # In time-series mode, only reveal plot after both Material and Conditions chosen (if conditions exist)
        if display_mode.value == "series":
            material_ready = material_dd.value != "any"
            cond_ready = True if not cond_col else conditions_dd.value != "any"
            if material_ready and cond_ready:
                try:
                    bordered_plot.layout.display = ""
                except Exception:
                    pass
                try:
                    action_row.layout.display = ""  # show Save/Redo in series mode
                except Exception:
                    pass
                try:
                    mark_row.layout.display = "none"
                except Exception:
                    pass
            else:
                # Hide until selections complete
                try:
                    bordered_plot.layout.display = "none"
                except Exception:
                    pass
                try:
                    action_row.layout.display = "none"
                except Exception:
                    pass
                try:
                    mark_row.layout.display = "none"
                except Exception:
                    pass

    display_mode.observe(_on_display_mode, names="value")
    spectrum_sel.observe(_on_spectrum_change_reveal, names="value")
    material_dd.observe(_on_filter_change_reveal, names="value")
    conditions_dd.observe(_on_filter_change_reveal, names="value")

    # Layout: controls on top, then plot, then info and messages, then buttons
    controls_row = widgets.HBox([material_dd, conditions_dd])
    mode_row = widgets.HBox([display_mode])
    spectrum_row = widgets.HBox([spectrum_sel, include_bad_cb])
    if range_slider is not None:
        # Compose Colab range selection row with texts + slider + Apply button and help
        try:
            colab_range_row = widgets.HBox(
                [range_lo_text, range_hi_text, range_apply_btn]
            )
            display(
                controls_row,
                mode_row,
                spectrum_row,
                colab_range_row,
                range_slider,
                range_help,
                bordered_plot,
                info_out,
                msg_out,
                action_row,
                close_row,
            )
        except Exception:
            display(
                controls_row,
                mode_row,
                spectrum_row,
                range_slider,
                bordered_plot,
                info_out,
                msg_out,
                action_row,
                close_row,
            )
    else:
        display(
            controls_row,
            mode_row,
            spectrum_row,
            bordered_plot,
            info_out,
            msg_out,
            action_row,
            close_row,
        )
    try:
        _refresh_mark_buttons()
    except Exception:
        pass
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

    # Use shared _parse_seq helper defined at module scope

    # Determine the conditions column via shared helper
    cond_col = _conditions_column_name(filtered)

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
            _set_session_selection(
                material=material_dd.value, conditions=conditions_dd.value
            )
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
    # --- Change tracking for session summary on Close ---
    _peak_changes = {
        "saved_file": [],  # list[(idx, n_peaks)]
        "saved_filtered": 0,  # count of spectra updated via filtered save
        "quality": [],  # list[(idx, new_quality)]
    }
    # Use shared quality controls
    current_idx_fp = None  # track displayed spectrum independently of dropdown
    mark_bad_btn, mark_good_btn, _refresh_mark_buttons = _make_quality_controls(
        FTIR_DataFrame,
        lambda: (
            FTIR_DataFrame.loc[current_idx_fp]
            if current_idx_fp is not None
            else None
        ),
    )
    include_bad_cb = widgets.Checkbox(value=False, description="Include bad spectra")
    close_btn = widgets.Button(description="Close", button_style="danger")
    msg_out = widgets.Output()

    # Refresh provided by helper

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
        nonlocal current_idx_fp
        current_idx_fp = idx
        x_arr, y_arr = _get_xy(idx)
        if x_arr is None:
            with msg_out:
                msg_out.clear_output()
                print("Selected spectrum missing or invalid normalized data.")
            # Hide mark row if we have no valid selection/data
            try:
                mark_row.layout.display = "none"
            except Exception:
                pass
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
        # Show mark row now that a spectrum is actively plotted
        try:
            mark_row.layout.display = ""
        except Exception:
            pass

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
        try:
            _peak_changes["saved_file"].append((idx, int(peaks_idx.size)))
        except Exception:
            pass
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
        try:
            _peak_changes["saved_filtered"] += int(updated)
        except Exception:
            pass
        with msg_out:
            msg_out.clear_output()
            print(
                f"Updated {updated} spectra; skipped {skipped} (missing/invalid data)."
            )

    # Additional UI updates after helper toggles quality
    def _post_mark_update_find_peaks(_b=None):
        try:
            _refresh_mark_buttons()
        except Exception:
            pass
        try:
            idx = current_idx_fp
            if idx is not None:
                # Determine current quality
                qcol = _quality_column_name(FTIR_DataFrame)
                qval = FTIR_DataFrame.loc[idx].get(qcol, None)
                if qval in ("bad", "good"):
                    # Use centralized helper to remove/reinsert without disturbing current plot
                    _quality_dropdown_handle(
                        qval,
                        dropdown=spectrum_sel,
                        include_bad_flag=include_bad_cb.value,
                        idx=idx,
                        label_builder=lambda i: f"{FTIR_DataFrame.loc[i].get('Material','?')} | {FTIR_DataFrame.loc[i].get('Conditions', FTIR_DataFrame.loc[i].get('Condition','?'))} | {FTIR_DataFrame.loc[i].get('File Name','?')}",
                        observer_fn=_update_plot,
                    )
                else:
                    try:
                        if not include_bad_cb.value:
                            _on_filters_change()
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            idx = current_idx_fp
            if idx is not None:
                # capture current quality value
                qcol = _quality_column_name(FTIR_DataFrame)
                qval = FTIR_DataFrame.loc[idx].get(qcol, None)
                _peak_changes["quality"].append((idx, qval))
        except Exception:
            pass

    def _close_ui(b):
        # Emit session summary before closing figure (leave msg_out visible)
        try:
            lines = _session_summary_lines(_peak_changes, context="peaks")
            _emit_session_summary(msg_out, lines, title="Session summary (Peak Finder)")
        except Exception:
            pass
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
            # ui variable not defined in this scope; remove stale close attempt
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
            # Hide mark row when nothing is selectable
            try:
                mark_row.layout.display = "none"
            except Exception:
                pass
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
    # Helper wires core behavior; add post-effects
    mark_bad_btn.on_click(_post_mark_update_find_peaks)
    mark_good_btn.on_click(_post_mark_update_find_peaks)
    include_bad_cb.observe(_on_filters_change, names="value")
    close_btn.on_click(_close_ui)

    controls_row1 = widgets.HBox([spectrum_sel, include_bad_cb])
    controls_row2 = widgets.HBox([use_r1, x_range1])
    controls_row3 = widgets.HBox([use_r2, x_range2])
    controls_row4 = widgets.HBox([use_r3, x_range3])
    controls_row5 = widgets.HBox([prominence, min_height, distance])
    mark_row = widgets.HBox([mark_bad_btn, mark_good_btn])
    # Hide mark buttons until a spectrum is selected; place below plot in border
    try:
        mark_row.layout.display = "none"
    except Exception:
        pass
    # Keep sliders separate from buttons per requirements
    controls_row6 = widgets.HBox([width, max_peaks])
    buttons_row = widgets.HBox([save_file_btn, save_all_btn, close_btn])
    # Prepend filter controls row
    filters_row = widgets.HBox([material_dd, conditions_dd])
    # Bordered plot + mark container
    plot_and_mark_pf = widgets.VBox(
        [fig, mark_row],
        layout=widgets.Layout(border="1px solid #ccc", padding="8px", margin="6px 0"),
    )
    # Display the Peak-Finding UI components
    try:
        display(
            filters_row,
            controls_row1,
            controls_row2,
            controls_row3,
            controls_row4,
            controls_row5,
            plot_and_mark_pf,
            controls_row6,
            buttons_row,
            msg_out,
        )
    except Exception:
        # Fallback: display essential parts if batch display fails
        try:
            display(filters_row, controls_row1, plot_and_mark_pf, msg_out)
        except Exception:
            pass
    return FTIR_DataFrame


def deconvolute_peaks(FTIR_DataFrame, filepath=None):
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
            "lmfit is required for deconvolute_peaks. Please install it (e.g., pip "
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

    # Use shared _parse_seq helper (module-level)

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
    unique_materials, unique_conditions = _extract_material_condition_lists(
        filterable_df, exclude_unexposed=True
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
    # Per-peak default seeds (no global center/sigma controls)
    PER_PEAK_DEFAULT_CENTER_WINDOW = 15.0
    PER_PEAK_DEFAULT_SIGMA = 10.0
    SESSION_WINDOW_MARGIN = 1.0  # cm⁻¹ margin used when sizing session peak window vs existing peaks
    # Defaults for reset operations
    DEFAULT_ALPHA = 0.5
    DEFAULT_INCLUDE = True
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
    # --- Change tracking for session summary (on Close) ---
    _deconv_changes = {
        "saved": [],  # list[(idx, count_components)]
        "quality": [],  # list[(idx, new_quality)]
        "iter": [],  # list[(idx, start_redchi, final_redchi, improvements)]
    }

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
    reset_all_btn = widgets.Button(
        description="Reset all peak parameters to defaults",
        button_style="warning",
        tooltip="Reset all per-peak sliders and selections to defaults",
        layout=widgets.Layout(width="340px"),
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
    cancel_fit_btn_frozen = False  # suppress show/hide toggles to prevent flicker
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

    # Dynamic per-peak controls: include checkbox + per-peak parameter sliders/locks
    alpha_sliders = []  # list[widgets.FloatSlider]
    include_checkboxes = []  # list[widgets.Checkbox]
    center_window_sliders = []  # list[widgets.FloatSlider] per-peak center ± window
    sigma_sliders = []  # list[widgets.FloatSlider]
    lock_alpha_checkboxes = []  # list[widgets.Checkbox]
    lock_center_checkboxes = []  # list[widgets.Checkbox]
    lock_sigma_checkboxes = []  # list[widgets.Checkbox]
    peak_controls_box = widgets.VBox([])

    # Persisted per-spectrum settings so switching spectra preserves choices
    per_spec_alpha = {}  # idx -> list[float]
    per_spec_include = {}  # idx -> list[bool]
    per_spec_center_bounds = {}  # idx -> list[(minus, plus)]
    per_spec_sigma = {}  # idx -> list[float]
    per_spec_locks = {}  # idx -> { 'alpha': list[bool], 'center': list[bool], 'sigma': list[bool] }
    per_spec_globals = {}  # idx -> { 'fit_range': (lo,hi) }
    # Track the last active (Material, Conditions) filter to scope the above caches
    current_filter_key = (None, None)
    # Group-level templates (by current (Material, Conditions)) so parameter changes
    # carry over across spectra within the same selection.
    group_alpha_template = {}  # (Material, Conditions) -> list[float]
    group_include_template = {}  # (Material, Conditions) -> list[bool]
    group_center_bounds_template = {}  # (Material, Conditions) -> list[(minus, plus)]
    group_sigma_template = {}  # (Material, Conditions) -> list[float]
    group_locks_template = {}  # (Material, Conditions) -> { 'alpha': list[bool], 'center': list[bool], 'sigma': list[bool] }
    group_globals_template = ({})  # (Material, Conditions) -> {fit_range}
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
            # Store as (w, w) tuples for backward compatibility
            per_spec_center_bounds[idx] = [
                (float(sw.value), float(sw.value)) for sw in center_window_sliders
            ]
        except Exception:
            per_spec_center_bounds[idx] = []
        try:
            per_spec_sigma[idx] = [float(sg.value) for sg in sigma_sliders]
        except Exception:
            per_spec_sigma[idx] = []
        try:
            per_spec_locks[idx] = {
                'alpha': [bool(cb.value) for cb in lock_alpha_checkboxes],
                'center': [bool(cb.value) for cb in lock_center_checkboxes],
                'sigma': [bool(cb.value) for cb in lock_sigma_checkboxes],
            }
        except Exception:
            per_spec_locks[idx] = {'alpha': [], 'center': [], 'sigma': []}
        try:
            lo, hi = _current_fit_range()
            per_spec_globals[idx] = {
                "fit_range": (float(lo), float(hi)),
            }
        except Exception:
            pass
        # Also persist group-level templates for this (Material, Conditions)
        try:
            key = current_filter_key
            if isinstance(key, tuple) and any(v is not None for v in key):
                try:
                    group_alpha_template[key] = list(per_spec_alpha.get(idx, []))
                except Exception:
                    group_alpha_template[key] = []
                try:
                    group_include_template[key] = list(per_spec_include.get(idx, []))
                except Exception:
                    group_include_template[key] = []
                try:
                    group_center_bounds_template[key] = list(per_spec_center_bounds.get(idx, []))
                except Exception:
                    group_center_bounds_template[key] = []
                try:
                    group_sigma_template[key] = list(per_spec_sigma.get(idx, []))
                except Exception:
                    group_sigma_template[key] = []
                try:
                    group_locks_template[key] = dict(per_spec_locks.get(idx, {'alpha': [], 'center': [], 'sigma': []}))
                except Exception:
                    group_locks_template[key] = {'alpha': [], 'center': [], 'sigma': []}
                try:
                    group_globals_template[key] = dict(per_spec_globals.get(idx, {}))
                except Exception:
                    pass
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
    new_peak_windows = {}  # x_new -> effective session window (min(default, min_dist_to_existing - margin))

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
        # If an iterative optimization is in progress or visibility is frozen,
        # keep the button visible and enabled without toggling to avoid flicker
        # between the many short internal fits.
        try:
            if iterating_in_progress or cancel_fit_btn_frozen:
                try:
                    if getattr(cancel_fit_btn.layout, "display", "") == "none":
                        _show(cancel_fit_btn)
                except Exception:
                    _show(cancel_fit_btn)
                try:
                    if cancel_fit_btn.disabled:
                        cancel_fit_btn.disabled = False
                except Exception:
                    pass
                return  # never hide while iterating/frozen
        except Exception:
            pass
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

    # Explicit helpers to force show/hide independent of thread state.
    # These provide a deterministic UI state when fits/iterations start or end,
    # avoiding races where the thread reference may still appear alive briefly.
    def _force_cancel_fit_shown():
        try:
            _show(cancel_fit_btn)
            cancel_fit_btn.disabled = False
        except Exception:
            pass

    def _force_cancel_fit_hidden():
        # Do not hide while an iteration is active or visibility frozen
        try:
            if iterating_in_progress or cancel_fit_btn_frozen:
                return
        except Exception:
            pass
        try:
            if not cancel_fit_btn.disabled:
                cancel_fit_btn.disabled = True
        except Exception:
            pass
        try:
            if getattr(cancel_fit_btn.layout, "display", "") != "none":
                _hide(cancel_fit_btn)
        except Exception:
            pass

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

    # Click handler for adding peaks when in adding mode (desktop / non-Colab)
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
        # Reject if within any existing visible peak's per-peak Center ± window
        try:
            vis_xs, _vis_ys = _get_visible_peaks(idx)
        except Exception:
            vis_xs = []
        try:
            default_win = float(PER_PEAK_DEFAULT_CENTER_WINDOW)
        except Exception:
            default_win = 0.0
        try:
            for i, cx in enumerate(vis_xs or []):
                try:
                    w_i = float(center_window_sliders[i].value)
                except Exception:
                    w_i = default_win
                if abs(float(cx) - float(x_new)) <= abs(w_i):
                    try:
                        status_html.value = (
                            f"<span style='color:#a00;'>Rejected: {x_new:.3f} cm⁻¹ overlaps existing peak @ {cx:.3f} ±{w_i:.1f} cm⁻¹. "
                            f"Adjust selection or that peak’s Center ± window.</span>"
                        )
                    except Exception:
                        _log_once(
                            f"Rejected: {x_new:.3f} cm⁻¹ overlaps existing peak @ {cx:.3f} ±{w_i:.1f} cm⁻¹. "
                            f"Adjust selection or that peak’s Center ± window."
                        )
                    return
        except Exception:
            pass
        # Enforce proximity using Center ±window as minimum separation (customizable)
        try:
            min_sep = float(PER_PEAK_DEFAULT_CENTER_WINDOW)
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
                        f"Tip: reduce the per-peak Center ± slider to fit peaks in small "
                        f"spaces.</span>"
                    )
                except Exception:
                    _log_once(
                        f"Rejected: {x_new:.3f} cm⁻¹ is within ±{min_sep:.2f} cm⁻¹ of "
                        f"another selected peak ({existing_x:.3f}). Tip: reduce the "
                        f"per-peak Center ± slider to fit peaks in small spaces."
                    )
                return
        # Compute an effective session window so the new peak's window won't cover existing peaks
        try:
            xs_existing, _ys_existing = _get_peaks(idx)
        except Exception:
            xs_existing = []
        try:
            if xs_existing:
                min_dist = min(abs(float(xe) - x_new) for xe in xs_existing)
            else:
                min_dist = float('inf')
        except Exception:
            min_dist = float('inf')
        try:
            eff_window = float(PER_PEAK_DEFAULT_CENTER_WINDOW)
            if np.isfinite(min_dist):
                eff_window = min(eff_window, max(0.0, float(min_dist) - float(SESSION_WINDOW_MARGIN)))
        except Exception:
            eff_window = float(PER_PEAK_DEFAULT_CENTER_WINDOW)
        try:
            new_peak_windows[x_new] = float(eff_window)
        except Exception:
            pass
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

    # Register click handler only outside Colab; Colab uses slider-based fallback
    try:
        if not _IN_COLAB:
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
        nonlocal alpha_sliders, include_checkboxes, center_window_sliders, sigma_sliders
        nonlocal lock_alpha_checkboxes, lock_center_checkboxes, lock_sigma_checkboxes
        # Build controls for ALL peaks, grouping by Fit X-range membership.
        # Only in-range peaks contribute to fitting/optimization state (lists).
        peaks_x_all, peaks_y_all = _get_peaks(row_idx)
        alpha_sliders = []
        include_checkboxes = []
        center_window_sliders = []
        sigma_sliders = []
        lock_alpha_checkboxes = []
        lock_center_checkboxes = []
        lock_sigma_checkboxes = []
        children = []
        if not peaks_x_all:
            peak_controls_box.children = [
                widgets.HTML(
                    "<b>No peaks found.</b> Run find_peak_info first."
                )
            ]
            return
        try:
            lo_rng, hi_rng = _current_fit_range()
        except Exception:
            lo_rng, hi_rng = float("-inf"), float("inf")
        lo_rng_v = float(min(lo_rng, hi_rng))
        hi_rng_v = float(max(lo_rng, hi_rng))
        saved_alphas = per_spec_alpha.get(row_idx)
        saved_includes = per_spec_include.get(row_idx)
        saved_bounds = per_spec_center_bounds.get(row_idx)
        saved_sigma = per_spec_sigma.get(row_idx)
        saved_locks = per_spec_locks.get(row_idx)
        # Fallback to group-level templates if no per-spectrum values exist
        if not saved_alphas:
            try:
                saved_alphas = group_alpha_template.get(current_filter_key)
            except Exception:
                saved_alphas = None
        if not saved_includes:
            try:
                saved_includes = group_include_template.get(current_filter_key)
            except Exception:
                saved_includes = None
        if not saved_bounds:
            try:
                saved_bounds = group_center_bounds_template.get(current_filter_key)
            except Exception:
                saved_bounds = None
        if not saved_sigma:
            try:
                saved_sigma = group_sigma_template.get(current_filter_key)
            except Exception:
                saved_sigma = None
        if not saved_locks:
            try:
                saved_locks = group_locks_template.get(current_filter_key)
            except Exception:
                saved_locks = {'alpha': None, 'center': None, 'sigma': None}
        if not isinstance(saved_locks, dict):
            saved_locks = {'alpha': None, 'center': None, 'sigma': None}
        included_boxes = []
        excluded_boxes = []
        for i, (cx, cy) in enumerate(zip(peaks_x_all, peaks_y_all)):
            in_range = False
            try:
                in_range = lo_rng_v <= float(cx) <= hi_rng_v
            except Exception:
                in_range = False
            header = widgets.HTML(
                value=f"<b>Peak {i+1} @ {float(cx):.1f} cm⁻¹</b>" + (" <span style='color:#777;'>(out of range)</span>" if not in_range else ""),
            )
            include_checkbox = widgets.Checkbox(
                value=(
                    (
                        bool(saved_includes[i])
                        if saved_includes is not None and i < len(saved_includes)
                        else DEFAULT_INCLUDE
                    ) if in_range else False
                ),
                description="Include peak in Fit",
                indent=False,
                layout=widgets.Layout(width="200px"),
                disabled=(not in_range),
            )
            alpha_slider = widgets.FloatSlider(
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
                layout=widgets.Layout(width="220px"),
                disabled=(not in_range),
            )
            # Center ± window per-peak (uses per-peak default constant)
            default_bound = float(PER_PEAK_DEFAULT_CENTER_WINDOW)
            # Enforce a strictly positive lower bound for the window width
            min_cwin = 0.5  # cm⁻¹, must be > 0
            w_saved = (
                float(saved_bounds[i][0])
                if saved_bounds is not None and i < len(saved_bounds)
                and isinstance(saved_bounds[i], (list, tuple)) and len(saved_bounds[i]) == 2
                else default_bound
            )
            if not np.isfinite(w_saved) or w_saved <= 0.0:
                w_saved = max(default_bound, min_cwin)
            else:
                w_saved = max(w_saved, min_cwin)
            center_window_slider = widgets.FloatSlider(
                value=w_saved,
                min=min_cwin,
                max=max(default_bound * 2.0, 1.0),
                step=0.5,
                description="Center ± (cm⁻¹)",
                continuous_update=False,
                style={"description_width": "auto"},
                readout_format=".1f",
                layout=widgets.Layout(width="260px"),
                disabled=(not in_range),
            )
            # Sigma per-peak (uses per-peak default constant)
            default_sigma = float(PER_PEAK_DEFAULT_SIGMA)
            sigma_slider = widgets.FloatSlider(
                value=(
                    float(saved_sigma[i]) if saved_sigma is not None and i < len(saved_sigma) else default_sigma
                ),
                min=1.0,
                max=100.0,
                step=0.5,
                description="σ (cm⁻¹)",
                continuous_update=False,
                style={"description_width": "auto"},
                readout_format=".1f",
                layout=widgets.Layout(width="220px"),
                disabled=(not in_range),
            )
            # Per-parameter reset buttons (compact)
            reset_alpha_btn = widgets.Button(
                description="Reset",
                tooltip="Reset α to default (0.5)",
                layout=widgets.Layout(width="70px"),
            )
            reset_cwin_btn = widgets.Button(
                description="Reset",
                tooltip="Reset Center ± to default",
                layout=widgets.Layout(width="70px"),
            )
            reset_sigma_p_btn = widgets.Button(
                description="Reset",
                tooltip="Reset σ to default",
                layout=widgets.Layout(width="70px"),
            )

            # Wire reset handlers (capture references via defaults)
            def _mk_alpha_reset(sl=alpha_slider):
                return lambda _b=None: (_set_quiet(sl, "value", DEFAULT_ALPHA), _snapshot_current_controls())

            def _mk_cwin_reset(sl=center_window_slider, defb=lambda: float(PER_PEAK_DEFAULT_CENTER_WINDOW)):
                return lambda _b=None: (_set_quiet(sl, "value", float(defb())), _snapshot_current_controls())

            def _mk_sigma_reset(sl=sigma_slider, defs=lambda: float(PER_PEAK_DEFAULT_SIGMA)):
                return lambda _b=None: (_set_quiet(sl, "value", float(defs())), _snapshot_current_controls())

            try:
                reset_alpha_btn.on_click(_mk_alpha_reset())
                reset_cwin_btn.on_click(_mk_cwin_reset())
                reset_sigma_p_btn.on_click(_mk_sigma_reset())
            except Exception:
                pass
            # Locks row
            lock_alpha_checkbox = widgets.Checkbox(
                value=(
                    bool(saved_locks.get('alpha')[i])
                    if isinstance(saved_locks.get('alpha'), list) and i < len(saved_locks.get('alpha'))
                    else False
                ),
                description="Lock α",
                indent=False,
                layout=widgets.Layout(width="120px"),
                disabled=(not in_range),
            )
            lock_center_checkbox = widgets.Checkbox(
                value=(
                    bool(saved_locks.get('center')[i])
                    if isinstance(saved_locks.get('center'), list) and i < len(saved_locks.get('center'))
                    else False
                ),
                description="Lock center",
                indent=False,
                layout=widgets.Layout(width="140px"),
                disabled=(not in_range),
            )
            lock_sigma_checkbox = widgets.Checkbox(
                value=(
                    bool(saved_locks.get('sigma')[i])
                    if isinstance(saved_locks.get('sigma'), list) and i < len(saved_locks.get('sigma'))
                    else False
                ),
                description="Lock σ",
                indent=False,
                layout=widgets.Layout(width="120px"),
                disabled=(not in_range),
            )

            # Observers
            if in_range:
                include_checkbox.observe(_on_include_toggle, names="value")
                alpha_slider.observe(_on_alpha_change, names="value")
                center_window_slider.observe(_on_center_sigma_change, names="value")
                sigma_slider.observe(_on_center_sigma_change, names="value")
                for _cb in (lock_alpha_checkbox, lock_center_checkbox, lock_sigma_checkbox):
                    _cb.observe(lambda *_: _snapshot_current_controls(), names="value")

            # Only in-range peaks participate in fitting/optimization state lists
            if in_range:
                include_checkboxes.append(include_checkbox)
                alpha_sliders.append(alpha_slider)
                center_window_sliders.append(center_window_slider)
                sigma_sliders.append(sigma_slider)
                lock_alpha_checkboxes.append(lock_alpha_checkbox)
                lock_center_checkboxes.append(lock_center_checkbox)
                lock_sigma_checkboxes.append(lock_sigma_checkbox)

            sliders_row = widgets.HBox([
                widgets.HBox([alpha_slider, reset_alpha_btn], layout=widgets.Layout(align_items="center")),
                widgets.HBox([center_window_slider, reset_cwin_btn], layout=widgets.Layout(align_items="center")),
                widgets.HBox([sigma_slider, reset_sigma_p_btn], layout=widgets.Layout(align_items="center")),
            ])
            locks_row = widgets.HBox([lock_alpha_checkbox, lock_center_checkbox, lock_sigma_checkbox])
            include_row = widgets.HBox([include_checkbox])
            box = widgets.VBox([header, include_row, sliders_row, locks_row], layout=widgets.Layout(border="1px solid #eee", padding="6px", margin="6px 0"))
            if in_range:
                included_boxes.append(box)
            else:
                excluded_boxes.append(box)

        # Assemble with headings: Included Peaks -> Excluded Peaks (dark containers for visibility)
        heading_included = widgets.HTML(
            "<div style='background:#222; color:#fff; padding:6px 10px; border-radius:6px;"
            " border:1px solid #111; font-weight:600;'>Included Peaks</div>"
        )
        heading_excluded = widgets.HTML(
            "<div style='background:#2f2f2f; color:#fff; padding:6px 10px; border-radius:6px;"
            " border:1px solid #1a1a1a; font-weight:600;'>Excluded Peaks</div>"
        )
        note_none = widgets.HTML("<span style='color:#777;'>No peaks in selected range. Adjust 'Fit X-range' or run find_peak_info.</span>")
        final_children = [heading_included]
        if included_boxes:
            final_children.extend(included_boxes)
        else:
            final_children.append(note_none)
        final_children.append(heading_excluded)
        final_children.extend(excluded_boxes)
        peak_controls_box.children = final_children

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
                per_spec_center_bounds.clear()
                per_spec_sigma.clear()
                per_spec_locks.clear()
                last_redchi_by_idx.clear()
                last_result_by_idx.clear()
                # Reset shared peaks template when filter changes
                shared_peaks_x = None
                # No global sliders to reset; keep only per-peak/state caches
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
                for _lbl, _v in new_options:
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
        # If no valid values remain, hide mark row until user selects something later
        try:
            if not valid_values:
                mark_row.layout.display = "none"
        except Exception:
            pass

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

        # Determine which peaks are included (respect Fit X-range explicitly)
        included = [i for i, cb in enumerate(include_checkboxes) if cb.value]
        try:
            lo_chk, hi_chk = _current_fit_range()
        except Exception:
            lo_chk, hi_chk = float("-inf"), float("inf")
        lo_v_chk = float(min(lo_chk, hi_chk))
        hi_v_chk = float(max(lo_chk, hi_chk))
        # Auto-exclude any selected peaks whose centers fall outside the current range
        try:
            included_in_range = [i for i in included if lo_v_chk <= float(peaks_x[i]) <= hi_v_chk]
        except Exception:
            included_in_range = list(included)
        if len(included_in_range) != len(included):
            included = included_in_range
            try:
                status_html.value = (
                    "<span style='color:#555;'>Some peaks were auto-excluded by the Fit X-range.</span>"
                )
            except Exception:
                pass
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
                    # Center bounds from per-peak center ± window slider if present
                    try:
                        w = float(center_window_sliders[i].value)
                    except Exception:
                        w = float(PER_PEAK_DEFAULT_CENTER_WINDOW)
                    # Center lock
                    lock_center_i = False
                    try:
                        lock_center_i = bool(lock_center_checkboxes[i].value)
                    except Exception:
                        lock_center_i = False
                    p[f"p{i}_center"].set(
                        value=float(cx),
                        min=float(cx) - abs(w),
                        max=float(cx) + abs(w),
                        vary=(not lock_center_i),
                    )
                    # Sigma from per-peak slider; lock controls vary
                    try:
                        sg_val = float(sigma_sliders[i].value)
                    except Exception:
                        sg_val = float(PER_PEAK_DEFAULT_SIGMA)
                    lock_sigma_i = False
                    try:
                        lock_sigma_i = bool(lock_sigma_checkboxes[i].value)
                    except Exception:
                        lock_sigma_i = False
                    p[f"p{i}_sigma"].set(value=sg_val, min=1e-3, max=1e3, vary=(not lock_sigma_i))
                    # Alpha (fraction): controlled by slider; kept fixed during lmfit; lock applies to iterative tuning only
                    alpha_val = (
                        float(alpha_sliders[i].value) if i < len(alpha_sliders) else 0.5
                    )
                    p[f"p{i}_fraction"].set(value=alpha_val, min=0.0, max=1.0, vary=False)
                    amp0 = abs(float(peaks_y[i])) * max(1.0, float(sg_val))
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
                # Only hide at the end if not iterating/frozen
                try:
                    _on_main_thread(_force_cancel_fit_hidden)
                except Exception:
                    pass
                try:
                    _on_main_thread(_force_cancel_fit_hidden)
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
                    # Keep visible during iteration; outer loop will hide at end
                    try:
                        if iterating_in_progress or cancel_fit_btn_frozen:
                            _force_cancel_fit_shown()
                        else:
                            _force_cancel_fit_hidden()
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
                    # Keep visible during iteration; final cleanup handles hide.
                    try:
                        if iterating_in_progress or cancel_fit_btn_frozen:
                            _force_cancel_fit_shown()
                        else:
                            _force_cancel_fit_hidden()
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
        # Show the Cancel Fit button immediately when a fit starts
        try:
            _on_main_thread(_force_cancel_fit_shown)
        except Exception:
            try:
                _update_cancel_fit_visibility()
            except Exception:
                pass
        _finish_fit_guard()
        return None

    # Track displayed spectrum independently for deconvolution
    current_idx_deconv = None

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
            nonlocal current_idx_deconv
            current_idx_deconv = idx
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
                        fr = g.get("fit_range")
                        if isinstance(fr, (list, tuple)) and len(fr) == 2:
                            lo, hi = float(fr[0]), float(fr[1])
                            # clamp to bounds
                            lo = max(float(fit_range.min), min(lo, float(fit_range.max)))
                            hi = max(lo, min(hi, float(fit_range.max)))
                            fit_range.value = [lo, hi]
                    except Exception:
                        pass
                else:
                    # Use group-level globals as a fallback when switching within the same
                    # Material/Conditions selection
                    try:
                        gg = group_globals_template.get(current_filter_key)
                    except Exception:
                        gg = None
                    if isinstance(gg, dict) and gg:
                        try:
                            fr = gg.get("fit_range")
                            if isinstance(fr, (list, tuple)) and len(fr) == 2:
                                lo, hi = float(fr[0]), float(fr[1])
                                lo = max(float(fit_range.min), min(lo, float(fit_range.max)))
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
            # Show mark row now that a spectrum is actively selected
            try:
                mark_row.layout.display = ""
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
        try:
            _deconv_changes["saved"].append((idx, len(out)))
        except Exception:
            pass

    def _close_ui(b):
        # Emit a session summary before closing widgets; keep log_html visible
        try:
            lines = _session_summary_lines(_deconv_changes, context="deconvolution")
            _emit_session_summary(
                log_html, lines, title="Session summary (Deconvolution)"
            )
        except Exception:
            pass
        # Signal cancellation and close widgets promptly
        try:
            cancel_event.set()
        except Exception:
            pass
        try:
            spectrum_sel.close()
            material_dd.close()
            conditions_dd.close()
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
            _set_session_selection(
                material=material_dd.value, conditions=conditions_dd.value
            )
        except Exception:
            pass

    material_dd.observe(_persist_pd_filters, names="value")
    conditions_dd.observe(_persist_pd_filters, names="value")
    # Observers
    fit_range.observe(_on_fit_range_change, names="value")

    # Wire reset buttons
    def _reset_all(_b=None):
        nonlocal bulk_update_in_progress
        bulk_update_in_progress = True
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
        try:
            for s in center_window_sliders:
                s.value = float(PER_PEAK_DEFAULT_CENTER_WINDOW)
        except Exception:
            pass
        try:
            for s in sigma_sliders:
                s.value = float(PER_PEAK_DEFAULT_SIGMA)
        except Exception:
            pass
        _snapshot_current_controls()
        bulk_update_in_progress = False
        try:
            status_html.value = "<span style='color:#555;'>All peak parameters reset. Click Fit to update.</span>"
        except Exception:
            pass
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
        nonlocal cancel_fit_btn_frozen

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
                # Allow normal fit status updates again; unfreeze and hide now
                iterating_in_progress = False
                try:
                    cancel_fit_btn_frozen = False
                except Exception:
                    pass
                try:
                    _on_main_thread(_force_cancel_fit_hidden)
                except Exception:
                    pass
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
            # End iteration lifecycle cleanly: unfreeze, clear flag, and hide
            try:
                iterating_in_progress = False
                cancel_fit_btn_frozen = False
            except Exception:
                pass
            try:
                _on_main_thread(_force_cancel_fit_hidden)
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

            # 1) Per-peak α sliders (only included and unlocked peaks)
            unlocked_alpha_idxs = []
            try:
                unlocked_alpha_idxs = [i for i in included_idxs if not (i < len(lock_alpha_checkboxes) and bool(lock_alpha_checkboxes[i].value))]
            except Exception:
                unlocked_alpha_idxs = included_idxs
            for i in unlocked_alpha_idxs:
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

            # 2) Per-peak Center ± window (only included and unlocked peaks)
            try:
                if cancel_event.is_set():
                    break
            except Exception:
                pass
            unlocked_center_idxs = []
            try:
                unlocked_center_idxs = [i for i in included_idxs if not (i < len(lock_center_checkboxes) and bool(lock_center_checkboxes[i].value))]
            except Exception:
                unlocked_center_idxs = included_idxs
            for i in unlocked_center_idxs:
                try:
                    if cancel_event.is_set():
                        break
                except Exception:
                    pass
                if i >= len(center_window_sliders):
                    continue
                sld = center_window_sliders[i]
                if sld is None:
                    continue
                try:
                    v0 = float(sld.value)
                except Exception:
                    continue
                step = float(center_step)

                def set_plus(v0=v0, sld=sld, step=step):
                    _set_quiet(sld, "value", _clamp(v0 + step, float(getattr(sld, 'min', 0.0) or 0.0), float(getattr(sld, 'max', 1e3) or 1e3)))

                def set_minus(v0=v0, sld=sld, step=step):
                    _set_quiet(sld, "value", _clamp(v0 - step, float(getattr(sld, 'min', 0.0) or 0.0), float(getattr(sld, 'max', 1e3) or 1e3)))

                def restore(v0=v0, sld=sld):
                    _set_quiet(sld, "value", v0)

                _, kept = _try_adjust(
                    getter=lambda: sld.value,
                    setter=set_plus,
                    decrementer=set_minus,
                    restore=restore,
                    label=f"center_window[{i}]",
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

            # 3) Per-peak σ sliders (only included and unlocked peaks)
            try:
                if cancel_event.is_set():
                    break
            except Exception:
                pass
            unlocked_sigma_idxs = []
            try:
                unlocked_sigma_idxs = [i for i in included_idxs if not (i < len(lock_sigma_checkboxes) and bool(lock_sigma_checkboxes[i].value))]
            except Exception:
                unlocked_sigma_idxs = included_idxs
            for i in unlocked_sigma_idxs:
                try:
                    if cancel_event.is_set():
                        break
                except Exception:
                    pass
                if i >= len(sigma_sliders):
                    continue
                sld = sigma_sliders[i]
                if sld is None:
                    continue
                try:
                    v0 = float(sld.value)
                except Exception:
                    continue
                step = float(sigma_step)

                def set_plus(v0=v0, sld=sld, step=step):
                    _set_quiet(sld, "value", _clamp(v0 + step, float(getattr(sld, 'min', 0.1) or 0.1), float(getattr(sld, 'max', 1e3) or 1e3)))

                def set_minus(v0=v0, sld=sld, step=step):
                    _set_quiet(sld, "value", _clamp(v0 - step, float(getattr(sld, 'min', 0.1) or 0.1), float(getattr(sld, 'max', 1e3) or 1e3)))

                def restore(v0=v0, sld=sld):
                    _set_quiet(sld, "value", v0)

                _, kept = _try_adjust(
                    getter=lambda: sld.value,
                    setter=set_plus,
                    decrementer=set_minus,
                    restore=restore,
                    label=f"sigma[{i}]",
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

        # Allow normal fit status updates again and unfreeze button visibility.
        # Perform the hide only once here to prevent mid-loop flicker.
        iterating_in_progress = False
        cancel_fit_btn_frozen = False
        try:
            _on_main_thread(_force_cancel_fit_hidden)
        except Exception:
            pass

    # Wire the iterative correct to run in background and support cancellation
    def _on_iteratively_correct_click(b):
        if _recent_click("iteratively_correct"):
            return
        nonlocal iter_thread, iterating_in_progress, cancel_event, iter_start_redchi, cancel_fit_btn_frozen
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
        cancel_fit_btn_frozen = True
        try:
            status_html.value = (
                "<span style='color:#555;'>Starting iterative correction...</span>"
            )
        except Exception:
            pass
        iter_thread = threading.Thread(target=_iteratively_correct_worker, daemon=True)
        iter_thread.start()
        try:
            _on_main_thread(_force_cancel_fit_shown)
        except Exception:
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
    controls_row_filters = widgets.HBox([material_dd, conditions_dd])
    controls_row_spectrum = widgets.HBox([spectrum_sel, include_bad_cb])
    # Place the Fit X-range slider above the peak modification section
    fit_range_row = widgets.HBox([fit_range])
    # Keep other global parameters grouped below the peak controls
    reset_all_row = widgets.HBox([reset_all_btn])
    # Mark buttons (use shared helper for mutually exclusive controls)
    mark_bad_btn, mark_good_btn, _refresh_mark_buttons = _make_quality_controls(
        FTIR_DataFrame,
        lambda: (
            FTIR_DataFrame.loc[current_idx_deconv]
            if current_idx_deconv is not None
            else None
        ),
        margin="0 8px 0 0",
    )

    # Add additional status and list refresh after helper toggles quality
    def _post_mark_update_deconv(status: str):
        nonlocal current_idx_deconv
        try:
            idx = current_idx_deconv
            if status == "bad":
                try:
                    status_html.value = f"<span style='color:#a00;'>Marked row {idx} as bad quality.</span>"
                except Exception:
                    pass
            else:
                try:
                    status_html.value = f"<span style='color:#0a0;'>Marked row {idx} as good quality.</span>"
                except Exception:
                    pass
            # Removal/reinsertion logic delegated to centralized helper to avoid flicker
            if idx is not None:
                try:
                    if status == "bad":
                        current_idx_deconv = idx
                    _quality_dropdown_handle(
                        status,
                        dropdown=spectrum_sel,
                        include_bad_flag=include_bad_cb.value,
                        idx=idx,
                        label_builder=lambda i: f"{FTIR_DataFrame.loc[i].get('Material','?')} | {FTIR_DataFrame.loc[i].get('Conditions', FTIR_DataFrame.loc[i].get('Condition','?'))} | {FTIR_DataFrame.loc[i].get('File Name','?')}",
                        observer_fn=_on_spectrum_change,
                    )
                except Exception:
                    pass
        except Exception:
            pass
        try:
            _refresh_mark_buttons()
        except Exception:
            pass
        try:
            _deconv_changes["quality"].append((current_idx_deconv, status))
        except Exception:
            pass

    mark_bad_btn.on_click(lambda _b=None: _post_mark_update_deconv("bad"))
    mark_good_btn.on_click(lambda _b=None: _post_mark_update_deconv("good"))
    mark_row = widgets.HBox([mark_bad_btn, mark_good_btn])
    # Hide mark buttons until a spectrum is selected; will show on selection
    try:
        mark_row.layout.display = "none"
    except Exception:
        pass

    # Refresh function provided by helper

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
    # Colab fallback slider + typed input + Add button for peak addition
    add_peaks_slider = widgets.FloatSlider(
        value=float((xmin + xmax) / 2.0),
        min=float(xmin),
        max=float(xmax),
        step=(float(xmax) - float(xmin)) / 1000.0 or 1.0,
        description="Select (cm⁻¹)",
        continuous_update=False,
        readout_format=".1f",
        layout=widgets.Layout(width="70%"),
    )
    add_peaks_text = widgets.FloatText(
        value=float((xmin + xmax) / 2.0),
        description="Type (cm⁻¹)",
        layout=widgets.Layout(width="160px"),
    )
    add_peaks_add_btn = widgets.Button(
        description="Add", button_style="info", layout=widgets.Layout(width="80px")
    )

    # Keep text and slider in sync (slider drives text)
    def _sync_text_from_slider(change):
        if change.get("name") != "value":
            return
        try:
            add_peaks_text.value = float(change.get("new"))
        except Exception:
            pass

    add_peaks_slider.observe(_sync_text_from_slider, names="value")
    colab_add_row = widgets.HBox([add_peaks_slider, add_peaks_text, add_peaks_add_btn])
    colab_add_help = widgets.HTML(
        "<span style='color:#555;font-size:12px;'>Colab: Use the slider or type a wavenumber, then click Add. Peaks snap to nearest data point; Accept to commit.</span>"
    )
    # Hide row if not in Colab or not in add-peaks mode
    if not _IN_COLAB:
        colab_add_row.layout.display = "none"
        colab_add_help.layout.display = "none"
    # Place plot + mark row in a bordered container; keep mark below the plot
    plot_and_mark_deconv = widgets.VBox(
        [fig, mark_row],
        layout=widgets.Layout(border="1px solid #ccc", padding="8px", margin="6px 0"),
    )
    ui = widgets.VBox(
        [
            # 1) Material and Conditions
            controls_row_filters,
            # 2) Spectrum dropdown + Include bad checkbox
            controls_row_spectrum,
            # 3) Plot container (with mark buttons below plot)
            plot_and_mark_deconv,
            # Status (reduced chi-square and operation updates) directly under plot
            status_row,
            # 4) Fit/Save/etc buttons
            buttons_row,
            #    Colab add controls (only shown in Colab when adding)
            colab_add_row,
            colab_add_help,
            # 5) Reset button
            reset_all_row,
            # 6) X range selector bar
            fit_range_row,
            # 7) List of peaks and per-peak controls
            peak_controls_box,
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
        # Hide parameter modifiers during add-peaks mode
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
        if _IN_COLAB:
            # Show slider row/help and set bounds from current spectrum
            try:
                x_arr, _y_arr = _get_xy(spectrum_sel.value)
                if x_arr is not None and x_arr.size > 0:
                    x_min = float(np.nanmin(x_arr))
                    x_max = float(np.nanmax(x_arr))
                    add_peaks_slider.min = x_min
                    add_peaks_slider.max = x_max
                    add_peaks_slider.step = (x_max - x_min) / 1000.0 or 1.0
                    mid = float((x_min + x_max) / 2.0)
                    add_peaks_slider.value = mid
                    try:
                        add_peaks_text.value = mid
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                colab_add_row.layout.display = ""
                colab_add_help.layout.display = ""
            except Exception:
                pass
            _log_once(
                "Add-peaks mode (Colab): use the slider then click Add; Accept/Redo/Cancel when done."
            )
        else:
            _log_once(
                "Add-peaks mode: click x-locations on the plot; then Accept/Redo/Cancel."
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

        # Use per-peak windows from visible peaks for overlap checks
        rejected_close = []
        try:
            peaks_x, _ = _get_visible_peaks(idx)
        except Exception:
            peaks_x = []
        for x_new in new_peak_xs:
            # Reject if new point falls inside any existing peak's per-peak window
            overlapped = False
            try:
                for i, cx in enumerate(peaks_x or []):
                    try:
                        w_i = float(center_window_sliders[i].value)
                    except Exception:
                        w_i = float(PER_PEAK_DEFAULT_CENTER_WINDOW)
                    if abs(float(cx) - float(x_new)) <= abs(w_i):
                        overlapped = True
                        break
            except Exception:
                overlapped = False
            if overlapped:
                rejected_close.append(float(x_new))
                continue
            # find nearest y
            try:
                i = int(np.argmin(np.abs(x_arr - x_new)))
                y_new = float(y_arr[i])
            except Exception:
                # If we cannot compute a nearest point, skip this new peak
                continue
        _hide(accept_new_peaks_btn)
        _hide(redo_new_peaks_btn)
        _hide(cancel_new_peaks_btn)
        # Restore previously hidden/disabled controls after exiting add-peaks mode
        _show(iter_btn)
        _update_cancel_fit_visibility()
        _show(save_btn)
        _show(close_btn)
        _show(peak_controls_box)
        # Hide Colab slider row when exiting add-peaks mode via Accept
        if _IN_COLAB:
            try:
                colab_add_row.layout.display = "none"
                colab_add_help.layout.display = "none"
            except Exception:
                pass
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
                "Rejected: {} overlap existing peaks' Center ± windows. "
                "Tip: adjust per-peak Center ± sliders to make room."
            ).format(joined)
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
        if _IN_COLAB:
            _log_once("Selection cleared. Use slider to select peaks again, then Add.")
        else:
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
        _show(iter_btn)
        _update_cancel_fit_visibility()
        _show(save_btn)
        _show(close_btn)
        _show(peak_controls_box)
        if _IN_COLAB:
            try:
                colab_add_row.layout.display = "none"
                colab_add_help.layout.display = "none"
            except Exception:
                pass
        try:
            spectrum_sel.disabled = False
            material_dd.disabled = False
            conditions_dd.disabled = False
        except Exception:
            pass
        _log_once("Peak addition cancelled. No changes were made.")

    add_peaks_btn.on_click(_enter_add_mode)

    # Colab slider-driven addition
    def _colab_add_peak(_b=None):
        try:
            # Prefer typed value when provided; fall back to slider
            x_target = float(add_peaks_text.value)
        except Exception:
            try:
                x_target = float(add_peaks_slider.value)
            except Exception:
                return
        idx = spectrum_sel.value
        x_arr, y_arr = _get_xy(idx)
        if x_arr is None or y_arr is None or x_arr.size == 0:
            _log_once("Cannot add peak: spectrum has no normalized data.")
            return
        # Snap to nearest existing x
        try:
            nearest_i = int(np.argmin(np.abs(x_arr - x_target)))
            x_new = float(x_arr[nearest_i])
        except Exception:
            _log_once("Could not determine nearest x for selected location.")
            return
        # Reject if within any existing visible peak's per-peak Center ± window
        try:
            vis_xs, _vis_ys = _get_visible_peaks(idx)
        except Exception:
            vis_xs = []
        try:
            default_win = float(PER_PEAK_DEFAULT_CENTER_WINDOW)
        except Exception:
            default_win = 0.0
        try:
            for i, cx in enumerate(vis_xs or []):
                try:
                    w_i = float(center_window_sliders[i].value)
                except Exception:
                    w_i = default_win
                if abs(float(cx) - float(x_new)) <= abs(w_i):
                    _log_once(
                        f"Rejected: {x_new:.3f} cm⁻¹ overlaps existing peak @ {cx:.3f} ±{w_i:.1f} cm⁻¹. Adjust selection or that peak’s Center ± window."
                    )
                    return
        except Exception:
            pass
        # Check against session-selected peaks using default window
        try:
            min_sep_default = float(PER_PEAK_DEFAULT_CENTER_WINDOW)
        except Exception:
            min_sep_default = 0.0
        for existing_x in new_peak_xs:
            if abs(existing_x - x_new) <= min_sep_default:
                _log_once(
                    f"Rejected: {x_new:.3f} cm⁻¹ overlaps the ±{min_sep_default:.2f} cm⁻¹ window of selected peak {existing_x:.3f}."
                )
                return
        # Compute session-effective window vs existing peaks and store
        try:
            xs_existing, _ys_existing = _get_peaks(idx)
        except Exception:
            xs_existing = []
        try:
            if xs_existing:
                min_dist = min(abs(float(xe) - x_new) for xe in xs_existing)
            else:
                min_dist = float('inf')
        except Exception:
            min_dist = float('inf')
        try:
            eff_window = float(PER_PEAK_DEFAULT_CENTER_WINDOW)
            if np.isfinite(min_dist):
                eff_window = min(eff_window, max(0.0, float(min_dist) - float(SESSION_WINDOW_MARGIN)))
        except Exception:
            eff_window = float(PER_PEAK_DEFAULT_CENTER_WINDOW)
        try:
            new_peak_windows[x_new] = float(eff_window)
        except Exception:
            pass
        # Accept
        new_peak_xs.append(x_new)
        try:
            y_min = float(np.nanmin(y_arr))
            y_max = float(np.nanmax(y_arr))
        except Exception:
            y_min, y_max = 0.0, 1.0
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
        _log_once(f"Selected new peak at x = {x_new:.3f} cm⁻¹. Add more or Accept.")

    try:
        if _IN_COLAB:
            add_peaks_add_btn.on_click(_colab_add_peak)
    except Exception:
        pass
    accept_new_peaks_btn.on_click(_accept_new_peaks)
    redo_new_peaks_btn.on_click(_redo_new_peaks)
    cancel_new_peaks_btn.on_click(_cancel_new_peaks)

    # Only display the UI (which already contains the figure) and the log
    display(ui, log_html)
    # Seed options with current filters (defaults 'any') and trigger initial updates
    _rebuild_spectrum_options()

    return FTIR_DataFrame


def fit_time_series(FTIR_DataFrame):
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
        sigmas_l = []  # model sigma (Lorentzian width in lmfit PseudoVoigt)
        fracs = []  # fraction (Lorentzian fraction) from lmfit
        for peaks in peak_lists:
            if len(peaks) != k:
                continue
            try:
                centers.append([float(p.get("center", np.nan)) for p in peaks])
                sigmas_l.append([float(p.get("sigma", np.nan)) for p in peaks])
                fracs.append([float(p.get("fraction", np.nan)) for p in peaks])
            except Exception:
                continue
        if not centers:
            print("Selected series has inconsistent peak counts; cannot average.")
            return
        centers = np.asarray(centers, dtype=float)
        sigmas_l = np.asarray(sigmas_l, dtype=float)
        fracs = np.asarray(fracs, dtype=float)
        with np.errstate(all="ignore"):
            avg_center = np.nanmean(centers, axis=0)
            avg_sigma_l = np.nanmean(sigmas_l, axis=0)
            avg_frac = np.nanmean(fracs, axis=0)
        for i in range(k):
            if not np.isfinite(avg_center[i]):
                vals = centers[:, i]
                avg_center[i] = (
                    np.nanmedian(vals) if np.isfinite(np.nanmedian(vals)) else 0.0
                )
            if not np.isfinite(avg_sigma_l[i]):
                vals = sigmas_l[:, i]
                avg_sigma_l[i] = (
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
                # sigma in lmfit PseudoVoigt is Lorentzian width; store avg_sigma_l
                p[f"p{i}_sigma"].set(
                    value=float(avg_sigma_l[i]), min=1e-3, max=1e4, vary=False
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
                            0.0, float(y_arr[nearest]) * max(1.0, float(avg_sigma_l[i]))
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
                    sigma_l_val = float(avg_sigma_l[i])
                    # alpha is the Lorentzian fraction (lmfit 'fraction')
                    alpha_lorentz = float(avg_frac[i])
                    # Compute Gaussian width from Lorentzian width: σg = σl / sqrt(2 ln 2)
                    try:
                        sigma_g_val = sigma_l_val / float(np.sqrt(2.0 * np.log(2.0)))
                    except Exception:
                        sigma_g_val = sigma_l_val
                    out.append(
                        {
                            "amplitude": amp,  # A
                            "center": float(avg_center[i]),
                            "alpha": alpha_lorentz,  # Lorentzian fraction (same as lmfit 'fraction')
                            "sigma_l": sigma_l_val,  # Lorentzian width
                            "sigma_g": sigma_g_val,  # Derived Gaussian width
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

        centers, sigmas_l, fracs = [], [], []
        for pk in peak_lists:
            if len(pk) != k:
                continue
            try:
                centers.append([float(p.get("center", np.nan)) for p in pk])
                sigmas_l.append([float(p.get("sigma", np.nan)) for p in pk])
                fracs.append([float(p.get("fraction", np.nan)) for p in pk])
            except Exception:
                continue
        if not centers:
            return "Selected series has inconsistent peak counts; cannot average.", None
        centers = np.asarray(centers, dtype=float)
        sigmas_l = np.asarray(sigmas_l, dtype=float)
        fracs = np.asarray(fracs, dtype=float)
        with np.errstate(all="ignore"):
            cen = np.nanmean(centers, axis=0)
            sig_l = np.nanmean(sigmas_l, axis=0)
            frc = np.nanmean(fracs, axis=0)
        for i in range(k):
            if not np.isfinite(cen[i]):
                vals = centers[:, i]
                cen[i] = np.nanmedian(vals) if np.isfinite(np.nanmedian(vals)) else 0.0
            if not np.isfinite(sig_l[i]):
                vals = sigmas_l[:, i]
                sig_l[i] = (
                    np.nanmedian(vals) if np.isfinite(np.nanmedian(vals)) else 10.0
                )
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
                p[f"p{i}_sigma"].set(
                    value=float(sig_l[i]), min=1e-3, max=1e4, vary=False
                )
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
                                0.0, float(y_arr[nearest]) * max(1.0, float(sig_l[i]))
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
                                    "alpha": float(frc[j]),
                                    "sigma_l": float(sig_l[j]),
                                    "sigma_g": float(sig_l[j])
                                    / float(np.sqrt(2.0 * np.log(2.0))),
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
                float(np.nanmedian(sig_l)) if np.isfinite(np.nanmedian(sig_l)) else 10.0
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
                # Sigma_l preferred; fallback to legacy 'sigma' if present
                try:
                    sig_val = p.get("sigma_l")
                    if sig_val is None:
                        sig_val = p.get("sigma", 10.0)
                    pr[f"p{i}_sigma"].set(
                        value=float(sig_val), min=1e-3, max=1e4, vary=False
                    )
                except Exception:
                    pr[f"p{i}_sigma"].set(value=10.0, min=1e-3, max=1e4, vary=False)
                # Fraction parameter (Lorentzian fraction): prefer 'alpha', fallback to legacy 'fraction'
                try:
                    frac_val = p.get("alpha")
                    if frac_val is None:
                        frac_val = p.get("fraction")
                    fv = float(frac_val if frac_val is not None else 0.5)
                except Exception:
                    fv = 0.5
                pr[f"p{i}_fraction"].set(
                    value=float(np.clip(fv, 0.0, 1.0)), min=0.0, max=1.0, vary=False
                )
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
        value=(
            default_condition
            if default_condition is not None
            else (cond_vals[0] if cond_vals else None)
        ),
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
                            # Prefer sigma_l; fallback to legacy 'sigma'
                            sigmas_list = [
                                float(
                                    (
                                        p.get("sigma_l")
                                        if p.get("sigma_l") is not None
                                        else p.get("sigma", float("nan"))
                                    )
                                )
                                for p in res0
                            ]
                            # Alpha is stored; fallback to legacy 'fraction' if present
                            fracs_list = [
                                float(
                                    (
                                        p.get("alpha")
                                        if p.get("alpha") is not None
                                        else p.get("fraction", float("nan"))
                                    )
                                )
                                for p in res0
                            ]
                        except Exception:
                            centers_list = None
                        break

                # If we have centers, render a compact table for peak wavenumbers
                if centers_list and len(centers_list) > 0:
                    try:
                        import pandas as pd  # local import safe here

                        # Build a table with rows: Center, Sigma (σ), Alpha (Lorentz); columns: Peak 1..N
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
                            ["α (Lorentz frac)"]
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
                            ["<td>α (Lorentz frac)</td>"]
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
        # Persist results to DataFrame as before (existing code below) AND update materials.json
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
                        # Also store component widths and alpha if available, preserving prior keys
                        try:
                            alpha_val = float(
                                p.get(
                                    "alpha",
                                    (1.0 - frac) if np.isfinite(frac) else float("nan"),
                                )
                            )
                        except Exception:
                            alpha_val = (
                                (1.0 - frac) if np.isfinite(frac) else float("nan")
                            )
                        try:
                            sg = float(p.get("sigma_g", p.get("sigma", float("nan"))))
                        except Exception:
                            sg = float("nan")
                        try:
                            sl = float(p.get("sigma_l", p.get("sigma", float("nan"))))
                        except Exception:
                            sl = float("nan")
                        cleaned.append(
                            {
                                "amplitude": amp,
                                "center": cen,
                                "sigma": sig,
                                "fraction": frac,
                                "alpha": alpha_val,
                                "sigma_g": sg,
                                "sigma_l": sl,
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
            # --- JSON augmentation: add/update peak entries + unexposed A values --- #
            try:
                base_dir_js = os.path.dirname(__file__)
                materials_json_path = os.path.join(base_dir_js, "materials.json")
                with open(materials_json_path, "r", encoding="utf-8") as jf:
                    _content = json.load(jf)
                if not isinstance(_content, list) or not _content:
                    raise ValueError("materials.json unexpected structure (not list)")
                _top = _content[0]

                # Locate material code key (M###) by alias/name
                def _lookup_code(mname):
                    for _k, _payload in _top.items():
                        if not isinstance(_payload, dict):
                            continue
                        if (
                            str(_payload.get("alias")) == mname
                            or str(_payload.get("name")) == mname
                        ):
                            return _k
                    return None

                code_key = _lookup_code(mat)
                if code_key is not None:
                    mat_payload = _top.get(code_key, {})
                    peaks_payload = mat_payload.get("peaks", {})
                    # --- Derive unexposed peak amplitudes (average across any unexposed rows) ---
                    unexposed_peak_amps = (
                        []
                    )  # length k_max; each entry corresponds to peak index
                    try:
                        # Build subset containing unexposed rows for this material present in the current df_series
                        unexp_rows = (
                            df_series[
                                df_series[cond_col].astype(str).str.strip().str.lower()
                                == "unexposed"
                            ]
                            if cond_col in df_series.columns
                            else pd.DataFrame([])
                        )
                        amp_accum = (
                            []
                        )  # list of lists; each inner list amplitudes for a row (length k_max)
                        for _u_idx, _u_row in unexp_rows.iterrows():
                            _u_res = _u_row.get("Time-Series Fit Results")
                            if isinstance(_u_res, str):
                                try:
                                    _u_res = ast.literal_eval(_u_res)
                                except Exception:
                                    _u_res = None
                            if isinstance(_u_res, list) and len(_u_res) == k_max:
                                try:
                                    amp_list = [
                                        float(p.get("amplitude", float("nan")))
                                        for p in _u_res
                                    ]
                                except Exception:
                                    amp_list = []
                                if len(amp_list) == k_max:
                                    amp_accum.append(amp_list)
                        if amp_accum:
                            # Average across rows, ignoring NaNs
                            try:
                                arr = np.asarray(amp_accum, dtype=float)
                                with np.errstate(all="ignore"):
                                    unexposed_peak_amps = [
                                        (
                                            float(np.nanmean(arr[:, i]))
                                            if arr.shape[1] > i
                                            and np.isfinite(np.nanmean(arr[:, i]))
                                            else 0.0
                                        )
                                        for i in range(k_max)
                                    ]
                            except Exception:
                                unexposed_peak_amps = [0.0] * k_max
                        else:
                            unexposed_peak_amps = [0.0] * k_max
                    except Exception:
                        unexposed_peak_amps = [0.0] * k_max
                    # Determine how many peaks the time-series fit produced (k_max)
                    # Ensure peak entries 1..k_max exist
                    for p_idx in range(1, k_max + 1):
                        pk_key = str(p_idx)
                        pk_entry = peaks_payload.get(pk_key)
                        if not isinstance(pk_entry, dict):
                            pk_entry = {}
                        pk_entry.setdefault("name", "")
                        # Preserve existing center_wavenumber if present; otherwise derive from first centers_list if available
                        try:
                            if (
                                "center_wavenumber" not in pk_entry
                                and centers_list
                                and p_idx - 1 < len(centers_list)
                            ):
                                pk_entry["center_wavenumber"] = float(
                                    centers_list[p_idx - 1]
                                )
                        except Exception:
                            pk_entry.setdefault("center_wavenumber", 0)
                        # Maintain legacy keys for shape factors if already present
                        pk_entry.setdefault("sg", 0)
                        pk_entry.setdefault("sl", 0)
                        pk_entry.setdefault("f", 0)
                        # Ensure conditions mapping exists
                        conds_entry = pk_entry.get("conditions")
                        if not isinstance(conds_entry, dict):
                            conds_entry = {}
                        # Build or merge standard exposure conditions from cond_map-like info (use current DF selection rows)
                        # Gather all condition names (excluding unexposed) present for this material in DF (filtered by normalized data availability)
                        try:
                            mat_rows = FTIR_DataFrame[
                                FTIR_DataFrame["Material"].astype(str) == mat
                            ]
                        except Exception:
                            mat_rows = FTIR_DataFrame
                        try:
                            mat_rows = mat_rows[
                                mat_rows["Normalized and Corrected Data"].notna()
                            ]
                        except Exception:
                            pass
                        cond_names_all = sorted(
                            {
                                str(v)
                                for v in mat_rows.get(
                                    cond_col, pd.Series([], dtype=object)
                                )
                                .dropna()
                                .astype(str)
                                .unique()
                                .tolist()
                                if str(v).strip().lower() != "unexposed"
                            }
                        )
                        # Merge each exposure condition: ensure it has time list & A list
                        for c_name in cond_names_all:
                            c_block = conds_entry.get(c_name)
                            if not isinstance(c_block, dict):
                                # Build times from DF subset
                                try:
                                    times_c = (
                                        mat_rows[
                                            mat_rows[cond_col].astype(str) == c_name
                                        ]["Time"]
                                        .dropna()
                                        .astype(float)
                                        .astype(int)
                                        .sort_values()
                                        .unique()
                                        .tolist()
                                    )
                                except Exception:
                                    times_c = []
                                conds_entry[c_name] = {"time": times_c, "A": []}
                            else:
                                # Ensure keys exist
                                c_block.setdefault("time", [])
                                c_block.setdefault("A", [])
                        # Update 'unexposed' structure (per-condition A values) using latest per-condition exposures
                        unexp = conds_entry.get("unexposed")
                        if not isinstance(unexp, dict):
                            unexp = {"per-condition": {}, "final": {"A": []}}
                        per_cond = unexp.get("per-condition")
                        if not isinstance(per_cond, dict):
                            per_cond = {}
                        # Populate per-condition A values for this peak using derived unexposed amplitudes
                        # Only update the currently selected condition; leave others as defaults/preserved values
                        for c_name in cond_names_all:
                            pc_entry = per_cond.get(c_name)
                            if not isinstance(pc_entry, dict):
                                per_cond[c_name] = {"A": []}
                                pc_entry = per_cond[c_name]
                            else:
                                # Coerce legacy scalar to list
                                if "A" not in pc_entry:
                                    pc_entry["A"] = []
                                elif not isinstance(pc_entry["A"], list):
                                    pc_entry["A"] = (
                                        [pc_entry["A"]]
                                        if pc_entry["A"] is not None
                                        else []
                                    )
                            if c_name == cond:
                                amp_val = (
                                    unexposed_peak_amps[p_idx - 1]
                                    if (p_idx - 1) < len(unexposed_peak_amps)
                                    else 0.0
                                )
                                # Always append latest amplitude value
                                try:
                                    pc_entry["A"].append(amp_val)
                                except Exception:
                                    pc_entry["A"] = [amp_val]
                        # Preserve existing 'final' block but do not modify it here; a later function will populate it.
                        final_block = unexp.get("final")
                        if not isinstance(final_block, dict):
                            final_block = {"A": []}
                        # Intentionally do not append to 'final'["A"] in fit_time_series; handled by a later aggregation step.
                        unexp["per-condition"] = per_cond
                        unexp["final"] = final_block
                        conds_entry["unexposed"] = unexp
                        pk_entry["conditions"] = conds_entry
                        peaks_payload[pk_key] = pk_entry
                    mat_payload["peaks"] = peaks_payload
                    _top[code_key] = mat_payload
                    # Write back JSON
                    try:
                        with open(materials_json_path, "w", encoding="utf-8") as jf:
                            json.dump(_content, jf, indent=4, ensure_ascii=False)
                        print(
                            f"[fit_time_series] materials.json updated (added/merged peaks 1..{k_max} for {mat})."
                        )
                    except Exception as _je:
                        print(
                            f"[fit_time_series] Failed to write materials.json: {_je}"
                        )
                else:
                    print(
                        f"[fit_time_series] Material '{mat}' not found in materials.json; skip JSON update."
                    )
            except Exception as _json_err:
                print(
                    f"[fit_time_series] JSON update skipped due to error: {_json_err}"
                )
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
            _set_session_selection(
                material=material_dd.value, conditions=conditions_dd.value
            )
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


def fit_material(FTIR_DataFrame, materials_json_path=None):
    """
    Aggregate time-series parameters across conditions for a selected material,
    compute average alpha and center per peak, and plot the selected material/condition
    time-series using these averaged parameters. Optionally overlay error (RMSE vs
    normalized-and-corrected spectra) in red on a secondary y-axis.

    Inputs
    - FTIR_DataFrame: pd.DataFrame with columns:
        - 'Material', 'Conditions' or 'Condition', 'Time'
        - 'X-Axis', 'Normalized and Corrected Data'
        - 'Time-Series Fit Results' (list[dict] per row with keys like 'amplitude', 'alpha', 'center')
    - materials_json_path: Optional path to materials.json. If None, attempts to locate
        Trenton_Project/materials.json relative to this file, else './materials.json'.

    Outputs
    - Displays interactive widgets and a plotly FigureWidget with:
        - Primary y: peak area (A) vs time for each peak (uses per-row amplitudes)
        - Secondary y: RMSE residual vs time (red), comparing aggregate model vs normalized data
    """

    # --- Helpers --- #
    def _cond_col(df):
        try:
            if "Conditions" in df.columns:
                return "Conditions"
            if "Condition" in df.columns:
                return "Condition"
        except Exception:
            pass
        return None

    def _safe_eval_list(val):
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except Exception:
                return None
        return val if isinstance(val, (list, tuple)) else None

    def _get_json_path():
        if materials_json_path and os.path.isfile(materials_json_path):
            return materials_json_path
        # Try alongside this file under Trenton_Project/materials.json
        try:
            here = os.path.dirname(__file__)
            candidate = os.path.join(here, "materials.json")
            if os.path.isfile(candidate):
                return candidate
        except Exception:
            pass
        # Fallback current working dir
        if os.path.isfile("materials.json"):
            return "materials.json"
        return None

    def _load_materials_json():
        path = _get_json_path()
        if not path:
            return None, None
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = json.load(f)
            return content, path
        except Exception:
            return None, path

    def _map_material_name_to_code(json_content, name):
        """Return (code_key, mat_payload) by matching 'name' or 'alias' to provided name."""
        if json_content is None:
            return None, None
        # materials.json structure appears to be a list with a dict containing M001, M002, ...
        try:
            top = json_content
            if isinstance(top, list) and top:
                top = top[0]
            if isinstance(top, dict):
                for code_key, mat_payload in top.items():
                    try:
                        if not isinstance(mat_payload, dict):
                            continue
                        n = str(mat_payload.get("name", "")).strip()
                        a = str(mat_payload.get("alias", "")).strip()
                        if name.strip() in (n, a):
                            return code_key, mat_payload
                    except Exception:
                        continue
        except Exception:
            pass
        return None, None

    def _get_json_peak_sigma(json_mat_payload, peak_idx):
        """Return Gaussian sigma (σg) default for a peak index from JSON if available.
        Looks for legacy 'sg' or unicode 'σg'. Fallback 10.0.
        """
        sigma = 10.0
        try:
            peaks = json_mat_payload.get("peaks", {})
            pe = peaks.get(str(peak_idx), {})
            sigma = float(pe.get("sg", pe.get("σg", sigma)))
        except Exception:
            pass
        return sigma

    def _get_json_peak_sigma_l(json_mat_payload, peak_idx):
        """Return Lorentzian sigma (σl) for a peak from JSON if available.
        Preference order: explicit 'sl' or 'σl'; else derive from 'sg'/'σg' via σl = σg * sqrt(2 ln 2).
        Fallback 10.0.
        """
        sig_l = None
        try:
            peaks = json_mat_payload.get("peaks", {})
            pe = peaks.get(str(peak_idx), {})
            # Prefer explicit sigma_l
            if "sl" in pe:
                sig_l = float(pe.get("sl"))
            elif "σl" in pe:
                sig_l = float(pe.get("σl"))
            else:
                # Derive from sigma_g
                if "sg" in pe or "σg" in pe:
                    sg = float(pe.get("sg", pe.get("σg")))
                    sig_l = float(sg * np.sqrt(2.0 * np.log(2.0)))
        except Exception:
            sig_l = None
        if sig_l is None or not np.isfinite(sig_l):
            return 10.0
        return float(sig_l)

    def _parse_tsf(row):
        v = row.get("Time-Series Fit Results")
        if isinstance(v, str):
            try:
                v = ast.literal_eval(v)
            except Exception:
                v = None
        return v if isinstance(v, list) else None

    def _avg_params(df_rows):
        """Average alpha (Lorentz frac), center, sigma_g, sigma_l per peak index across provided rows.

        Returns
        -------
        k_max : int
            Maximum peak count across rows.
        alpha_avg : list[float]
            Averaged Gaussian fraction (alpha) per peak.
        center_avg : list[float]
            Averaged center wavenumber per peak.
        sigma_g_avg : list[float]
            Averaged Gaussian width per peak (sigma_g).
        sigma_l_avg : list[float]
            Averaged Lorentzian width per peak (sigma_l).
        """
        # Determine k_max as max length present
        k_max = 0
        parsed = []
        for r in df_rows:
            lst = _parse_tsf(r)
            if isinstance(lst, list):
                parsed.append(lst)
                k_max = max(k_max, len(lst))
        if k_max == 0:
            return 0, [], [], [], []
        alphas = [[] for _ in range(k_max)]
        centers = [[] for _ in range(k_max)]
        sigmas_g = [[] for _ in range(k_max)]
        sigmas_l = [[] for _ in range(k_max)]
        for lst in parsed:
            for i in range(min(k_max, len(lst))):
                p = lst[i] if isinstance(lst[i], dict) else {}
                try:
                    # Accept multiple spellings/keys; fit_time_series stores 'fraction'
                    a = p.get("alpha")
                    if a is None:
                        a = p.get("α")
                    if a is None:
                        a = p.get("fraction")
                    if a is None:
                        a = p.get("f")
                except Exception:
                    a = None
                try:
                    c = p.get("center")
                    if c is None:
                        c = p.get("center_wavenumber")
                except Exception:
                    c = None
                # Collect sigma_g / sigma_l (preferred) else derive from single width (sigma_l)
                try:
                    sg_val = p.get("sigma_g")
                except Exception:
                    sg_val = None
                try:
                    sl_val = p.get("sigma_l")
                except Exception:
                    sl_val = None
                # Fallback: derive from single 'sigma' (Lorentzian) if component-specific widths missing
                if sg_val is None or sl_val is None:
                    try:
                        base_sig = p.get("sigma")
                        if base_sig is not None:
                            base_sig = float(base_sig)
                            if sl_val is None:
                                sl_val = base_sig
                            if sg_val is None:
                                sg_val = base_sig / float(np.sqrt(2.0 * np.log(2.0)))
                    except Exception:
                        pass
                if a is not None:
                    try:
                        aval = float(a)
                        # Clamp to [0,1] just in case
                        if aval < 0.0:
                            aval = 0.0
                        elif aval > 1.0:
                            aval = 1.0
                        alphas[i].append(aval)
                    except Exception:
                        pass
                if c is not None:
                    try:
                        centers[i].append(float(c))
                    except Exception:
                        pass
                if sg_val is not None:
                    try:
                        sigmas_g[i].append(float(sg_val))
                    except Exception:
                        pass
                if sl_val is not None:
                    try:
                        sigmas_l[i].append(float(sl_val))
                    except Exception:
                        pass
        alpha_avg = [float(np.nanmean(v)) if v else 0.5 for v in alphas]
        center_avg = [float(np.nanmean(v)) if v else 0.0 for v in centers]
        sigma_g_avg = [float(np.nanmean(v)) if v else float("nan") for v in sigmas_g]
        sigma_l_avg = [float(np.nanmean(v)) if v else float("nan") for v in sigmas_l]
        return k_max, alpha_avg, center_avg, sigma_g_avg, sigma_l_avg

    def _get_unexposed_A_from_json(json_content, material_name, condition_name, k_max):
        """Return list of unexposed A values per peak for selected material and condition.
        Picks the last entry per A list; returns list length k_max (missing -> 0.0)."""
        if json_content is None:
            return [0.0] * k_max
        code_key, mat_payload = _map_material_name_to_code(json_content, material_name)
        if mat_payload is None:
            return [0.0] * k_max
        peaks = mat_payload.get("peaks", {})
        out = []
        for i in range(1, k_max + 1):
            pk = peaks.get(str(i), {})
            conds = pk.get("conditions", {})
            unexp = conds.get("unexposed", {})
            per_cond = unexp.get("per-condition", {}) if isinstance(unexp, dict) else {}
            a_list = []
            try:
                ent = per_cond.get(str(condition_name), {})
                a_list = ent.get("A", []) if isinstance(ent, dict) else []
                if not isinstance(a_list, list):
                    a_list = [a_list]
            except Exception:
                a_list = []
            out.append(float(a_list[-1]) if a_list else 0.0)
        return out

    def _build_model_y(
        x, amps, alpha_avg, center_avg, json_mat_payload, sigma_l_avg=None
    ):
        """
        Sum of pseudo-Voigt components using per-time amplitudes and averaged alpha/center.
        - alpha parameter is Lorentzian fraction (same as lmfit PseudoVoigt 'fraction')
        - sigma uses averaged σl (Lorentzian) when available; else derive from JSON
          (prefer σl if present; else compute from σg via σl = σg * sqrt(2 ln 2)); fallback 10.0
        Returns array y_pred (same length as x) or None if x invalid.
        """
        if x is None or len(x) == 0:
            return None
        x_arr = np.asarray(x, dtype=float)
        y_sum = np.zeros_like(x_arr, dtype=float)
        k_max_local = min(len(amps or []), len(alpha_avg), len(center_avg))
        for i in range(k_max_local):
            try:
                amp = float(amps[i]) if i < len(amps) else 0.0
            except Exception:
                amp = 0.0
            if not np.isfinite(amp) or amp == 0:
                continue
            # Determine sigma preference: averaged sigma_l else JSON-derived σl fallback
            if (
                isinstance(sigma_l_avg, (list, tuple))
                and i < len(sigma_l_avg)
                and np.isfinite(sigma_l_avg[i])
            ):
                sigma = float(sigma_l_avg[i])
            else:
                sigma = (
                    _get_json_peak_sigma_l(json_mat_payload, i + 1)
                    if json_mat_payload
                    else 10.0
                )
            # Lorentzian fraction is alpha
            frac_lorentz = float(alpha_avg[i] if i < len(alpha_avg) else 0.5)
            center = float(center_avg[i] if i < len(center_avg) else 0.0)
            try:
                m = PseudoVoigtModel(prefix=f"p{i}_")
                pars = m.make_params()
                pars[f"p{i}_amplitude"].set(value=amp, min=0)
                pars[f"p{i}_center"].set(value=center)
                pars[f"p{i}_sigma"].set(value=max(sigma, 1e-6), min=1e-6)
                pars[f"p{i}_fraction"].set(value=np.clip(frac_lorentz, 0.0, 1.0))
                y_sum = y_sum + m.eval(pars, x=x_arr)
            except Exception:
                # Fallback simple Gaussian
                try:
                    gauss = amp * np.exp(
                        -0.5 * ((x_arr - center) / max(sigma, 1e-6)) ** 2
                    )
                    y_sum = y_sum + gauss
                except Exception:
                    pass
        return y_sum

    # --- UI setup --- #
    cond_col = _cond_col(FTIR_DataFrame)
    if cond_col is None:
        raise KeyError("Conditions/Condition column not found in DataFrame")

    # Materials and conditions lists
    try:
        materials = sorted(
            [
                str(v)
                for v in FTIR_DataFrame["Material"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            ]
        )
    except Exception:
        materials = []
    mat_dd = widgets.Dropdown(options=materials or [""], description="Material:")

    def _condition_options(mat):
        try:
            dfm = FTIR_DataFrame[FTIR_DataFrame["Material"].astype(str) == str(mat)]
            conds = [
                str(v)
                for v in dfm.get(cond_col, pd.Series([], dtype=object))
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            ]
            conds = sorted([c for c in conds if c.strip().lower() != "unexposed"])
            return conds or [""]
        except Exception:
            return [""]

    cond_dd = widgets.Dropdown(
        options=_condition_options(materials[0] if materials else ""),
        description="Condition:",
    )
    show_err_cb = widgets.Checkbox(value=False, description="Show error")
    status_html = widgets.HTML(value="")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(height=500, margin=dict(l=40, r=40, t=40, b=40))
    figw = go.FigureWidget(fig)
    # Table output for averaged parameters (updates on each replot)
    avg_params_out = widgets.Output(
        layout=widgets.Layout(
            border="1px solid #ccc", padding="4px", max_height="220px", overflow="auto"
        )
    )

    json_content, json_path = _load_materials_json()

    def _replot(*_):
        mat = str(mat_dd.value)
        cond = str(cond_dd.value)
        if not mat or not cond:
            status_html.value = (
                "<span style='color:#a00;'>Select a material and condition.</span>"
            )
            with figw.batch_update():
                figw.data = []
            with avg_params_out:
                avg_params_out.clear_output()
                print("No selection.")
            return

        # Filter rows for selection and valid normalized data + TSF results
        try:
            subset = FTIR_DataFrame[
                (FTIR_DataFrame["Material"].astype(str) == mat)
                & (FTIR_DataFrame[cond_col].astype(str) == cond)
            ].copy()
        except Exception:
            subset = FTIR_DataFrame.iloc[0:0].copy()

        try:
            subset = subset.dropna(
                subset=["Normalized and Corrected Data"]
            )  # require y
        except Exception:
            pass
        if subset.empty:
            status_html.value = f"<span style='color:#a00;'>No rows found for {mat} / {cond} with normalized data.</span>"
            with figw.batch_update():
                figw.data = []
            return

        subset = subset.sort_values(by="Time", kind="mergesort")

        # Compute averaged parameters across ALL conditions for this material
        try:
            mat_rows_all = FTIR_DataFrame[FTIR_DataFrame["Material"].astype(str) == mat]
        except Exception:
            mat_rows_all = subset
        k_max, alpha_avg, center_avg, sigma_g_avg, sigma_l_avg = _avg_params(
            [r for _, r in mat_rows_all.iterrows()]
        )
        if k_max == 0:
            status_html.value = "<span style='color:#a00;'>No time-series fit results found to average.</span>"
            with figw.batch_update():
                figw.data = []
            with avg_params_out:
                avg_params_out.clear_output()
                print("No averaged parameters available.")
            return

        # Get JSON material payload (for sigma, and unexposed As)
        json_mat_code, json_mat_payload = _map_material_name_to_code(json_content, mat)

        # Build amplitude vs time for each peak; include unexposed A at t=0 from JSON if available
        times = subset.get("Time", pd.Series([], dtype=float)).astype(float).tolist()
        tsf_parsed = [_parse_tsf(r) for _, r in subset.iterrows()]
        amps_by_peak = [[] for _ in range(k_max)]
        for lst in tsf_parsed:
            if not isinstance(lst, list) or not lst:
                for i in range(k_max):
                    amps_by_peak[i].append(0.0)
                continue
            for i in range(k_max):
                v = 0.0
                if i < len(lst) and isinstance(lst[i], dict):
                    try:
                        v = float(lst[i].get("amplitude", 0.0))
                    except Exception:
                        v = 0.0
                amps_by_peak[i].append(v)

        # Prepend unexposed point at t=0
        unexp_As = _get_unexposed_A_from_json(json_content, mat, cond, k_max)
        times_with_unexp = [0.0] + times
        amps_with_unexp = [[unexp_As[i]] + amps_by_peak[i] for i in range(k_max)]

        # Compute error (RMSE across wavenumber) per time using averaged params
        err_times = []
        err_vals = []
        if show_err_cb.value:
            for (idx, row), lst in zip(subset.iterrows(), tsf_parsed):
                x = _safe_eval_list(row.get("X-Axis"))
                y = _safe_eval_list(row.get("Normalized and Corrected Data"))
                if x is None or y is None:
                    continue
                # Per-time amplitudes for peaks
                amps = []
                if isinstance(lst, list):
                    for i in range(k_max):
                        try:
                            amps.append(
                                float(lst[i].get("amplitude", 0.0))
                                if i < len(lst)
                                else 0.0
                            )
                        except Exception:
                            amps.append(0.0)
                y_pred = _build_model_y(
                    np.asarray(x, dtype=float),
                    amps,
                    alpha_avg,
                    center_avg,
                    json_mat_payload or {},
                    sigma_l_avg,
                )
                if y_pred is None:
                    continue
                try:
                    resid = np.asarray(y, dtype=float) - np.asarray(y_pred, dtype=float)
                    rmse = float(np.sqrt(np.nanmean(resid**2)))
                    err_times.append(float(row.get("Time", np.nan)))
                    err_vals.append(rmse)
                except Exception:
                    pass

        # Build averaged parameter table (Peak, Center, α (Lorentz), σl(avg), σg(avg), σl(JSON fallback))
        try:
            # Fetch JSON material payload once (already computed above)
            rows_df = []
            for i in range(k_max):
                try:
                    sigma_json_l = (
                        _get_json_peak_sigma_l(json_mat_payload, i + 1)
                        if json_mat_payload
                        else 10.0
                    )
                except Exception:
                    sigma_json_l = 10.0
                a_val = float(alpha_avg[i]) if i < len(alpha_avg) else 0.5
                c_val = float(center_avg[i]) if i < len(center_avg) else 0.0
                sg_avg = sigma_g_avg[i] if i < len(sigma_g_avg) else float("nan")
                sl_avg = sigma_l_avg[i] if i < len(sigma_l_avg) else float("nan")
                rows_df.append(
                    {
                        "Peak": i + 1,
                        "Center (cm⁻¹)": round(c_val, 4),
                        "α (Lorentz frac)": round(a_val, 4),
                        "σl (avg)": (
                            round(float(sl_avg), 4) if np.isfinite(sl_avg) else ""
                        ),
                        "σg (avg)": (
                            round(float(sg_avg), 4) if np.isfinite(sg_avg) else ""
                        ),
                        "σl (JSON fallback)": round(float(sigma_json_l), 4),
                    }
                )
            df_avg = pd.DataFrame(rows_df)
        except Exception:
            df_avg = None
        with avg_params_out:
            avg_params_out.clear_output()
            if df_avg is not None and not df_avg.empty:
                display(df_avg)
            else:
                print("Averaged parameter table unavailable.")

        # Plot
        with figw.batch_update():
            figw.data = []
            # Amplitude traces (primary y)
            for i in range(k_max):
                figw.add_trace(
                    go.Scatter(
                        x=times_with_unexp,
                        y=amps_with_unexp[i],
                        mode="lines+markers",
                        name=f"Peak {i+1} A",
                    ),
                    secondary_y=False,
                )
            # Error trace (secondary y)
            if show_err_cb.value and err_times and err_vals:
                figw.add_trace(
                    go.Scatter(
                        x=err_times,
                        y=err_vals,
                        mode="lines+markers",
                        name="RMSE (model vs normalized)",
                        line=dict(color="red"),
                        marker=dict(color="red"),
                    ),
                    secondary_y=True,
                )
            figw.update_layout(
                title=f"Material: {mat} | Condition: {cond} | Averaged α & centers applied",
                xaxis_title="Time (h)",
                yaxis_title="Area (A)",
            )
            figw.update_yaxes(title_text="Area (A)", secondary_y=False)
            figw.update_yaxes(title_text="RMSE", secondary_y=True)

        try:
            nrows = mat_rows_all.shape[0]
        except Exception:
            nrows = 0
        status_html.value = (
            f"<span style='color:#060;'>Averaged across {nrows} rows; k={k_max} peaks. "
            + (
                f"materials.json: {json_path}"
                if json_path
                else "materials.json not found; used defaults."
            )
            + "</span>"
        )

    def _on_material_change(change):
        cond_dd.options = (
            _condition_options(change["new"])
            if isinstance(change, dict)
            else _condition_options(mat_dd.value)
        )
        if cond_dd.options:
            cond_dd.value = cond_dd.options[0]
        _replot()

    mat_dd.observe(lambda ch: _on_material_change(ch), names="value")
    cond_dd.observe(lambda ch: _replot(), names="value")
    show_err_cb.observe(lambda ch: _replot(), names="value")

    controls = widgets.HBox([mat_dd, cond_dd, show_err_cb])
    ui = widgets.VBox([controls, avg_params_out, figw, status_html])
    display(ui)
    # Initial draw
    _replot()
    return FTIR_DataFrame
