#Created: 9-18-2025
#Author: Trenton Wells
#Organization: NREL
#NREL Contact: trenton.wells@nrel.gov
#Personal Contact: trentonwells73@gmail.com

## This script scans a specified root directory and its subdirectories to find and rename files. Folder names will not be changed,
## except in the case of date renaming to ISO format (e.g., 2025-09-18) (optional).
## It works by replacing spaces and/or specified words in the filenames. (e.g., replacing spaces with underscores).
## Suggested to use this tool if file names have inconsistent naming conventions that may cause issues in downstream processing.

def Date_change_ISO(directory):
    """
    Renames all dates in filenames and folder names in the given directory and its subdirectories to ISO format (YYYY-MM-DD).

    Parameters:
    -----------
    directory (str): Directory to scan. Must be a valid directory path.

    Returns:
    -----------
    Renamed files and folders in place; prints changes to console.
    """
    import re
    date_patterns = [
        # MM-DD-YYYY or M-D-YYYY
        r'(\b\d{1,2}-\d{1,2}-\d{4}\b)',
        # YYYY-MM-DD
        r'(\b\d{4}-\d{1,2}-\d{1,2}\b)',
        # MMDDYYYY
        r'(\b\d{2}\d{2}\d{4}\b)',
        # YYYYMMDD
        r'(\b\d{4}\d{2}\d{2}\b)'
    ]
    def convert_to_iso(date_str):
        # If already in ISO format (YYYY-MM-DD), return as is
        if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            return date_str
        # MM-DD-YYYY or M-D-YYYY
        match = re.match(r'^(\d{1,2})-(\d{1,2})-(\d{4})$', date_str)
        if match:
            m, d, y = match.groups()
            return f"{y}-{int(m):02d}-{int(d):02d}"
        # MMDDYYYY
        match = re.match(r'^(\d{2})(\d{2})(\d{4})$', date_str)
        if match:
            m, d, y = match.groups()
            return f"{y}-{m}-{d}"
        # YYYYMMDD
        match = re.match(r'^(\d{4})(\d{2})(\d{2})$', date_str)
        if match:
            y, m, d = match.groups()
            return f"{y}-{m}-{d}"
        return date_str
    print("Renaming dates in filenames to ISO format...")
    for root, dirs, files in os.walk(directory):
        # Rename folders in current directory
        for current_dir in dirs:
            new_dirname = current_dir
            for pattern in date_patterns:
                for date_match in re.findall(pattern, current_dir):
                    iso_date = convert_to_iso(date_match)
                    if iso_date != date_match:
                        print(f"In folder '{current_dir}': changing date '{date_match}' to '{iso_date}'")
                        new_dirname = new_dirname.replace(date_match, iso_date)
            if new_dirname != current_dir:
                old_dirpath = os.path.join(root, current_dir)
                new_dirpath = os.path.join(root, new_dirname)
                print(f"Renaming folder: {old_dirpath} to {new_dirpath}")
                os.rename(old_dirpath, new_dirpath)

        # Rename files in current directory
        for current_filename in files:
            new_filename = current_filename
            for pattern in date_patterns:
                for date_match in re.findall(pattern, current_filename):
                    iso_date = convert_to_iso(date_match)
                    if iso_date != date_match:
                        print(f"In file '{current_filename}': changing date '{date_match}' to '{iso_date}'")
                        new_filename = new_filename.replace(date_match, iso_date)
            if new_filename != current_filename:
                old_filepath = os.path.join(root, current_filename)
                new_filepath = os.path.join(root, new_filename)
                print(f"Renaming: {old_filepath} to {new_filepath}")
                os.rename(old_filepath, new_filepath)
    print("Date renaming to ISO format complete.")

import os

def batch_rename_files(directory=None, replace_spaces=None, iso_date_rename=None, file_rename=None, character_to_use=None, pairs_input=None):
    """
    Scans a directory and its subdirectories to rename files by replacing spaces and/or specified words in filenames.
    Folder names will not be changed.
    Parameters:
    -----------
        directory (str): Directory to scan. If None, prompts user for input.
    
    Returns:
    -----------
        Renamed files in place; prints changes to console.
    """
    ## If no directory is provided, prompt the user for input
    if directory is None:
        directory = input("Enter the directory to scan: ").strip()
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    print(f"Scanning directory: {directory}")

    ## Option to replace spaces in filenames with different separator character
    if replace_spaces is None:
        replace_spaces_input = input("Do you want to replace spaces in filenames? (y/n): ").strip().lower()
        if replace_spaces_input == 'y':
            replace_spaces = True
    if replace_spaces:
        if character_to_use is None:
            character_to_use = input("Enter the separator to use instead of spaces (e.g. _): ").strip()
        print("Renaming files now!")
        for root, dirs, files in os.walk(directory):
            for current_filename in files:
                if ' ' in current_filename:
                    old_filepath = os.path.join(root, current_filename)
                    new_filename = current_filename.replace(' ', character_to_use)
                    new_filepath = os.path.join(root, new_filename)
                    print(f"Renaming: {old_filepath} to {new_filepath}")
                    os.rename(old_filepath, new_filepath)
        print("File renaming complete.")
    else:
        print("No spaces will be replaced in filenames.")
    
    ## Option to batch rename dates to ISO format (YYYY-MM-DD)
    if iso_date_rename is None:
        iso_date_input = input("Do you want to convert all dates in filenames to ISO format (YYYY-MM-DD)? (y/n): ").strip().lower()
        if iso_date_input == 'y':
            iso_date_rename = True
    if iso_date_rename:
        Date_change_ISO(directory)
    else:
        print("No dates will be changed in filenames.")

    if file_rename:
        if pairs_input is None:
            pairs_input = input("Enter words to find and their replacements as comma-separated pairs (e.g. old1:new1,old2:new2): ").strip()
        word_pairs = [pair.split(':') for pair in pairs_input.split(',') if ':' in pair]
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

if __name__ == "__main__":
    batch_rename_files()
