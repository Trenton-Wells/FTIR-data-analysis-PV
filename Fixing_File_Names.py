#Created: 9-18-2025
#Author: Trenton Wells
#Organization: NREL
#NREL Contact: trenton.wells@nrel.gov
#Personal Contact: trentonwells73@gmail.com

## This script scans a specified root directory and its subdirectories to find and rename files. Folder names will not be changed.
## by replacing spaces and/or specified words in the filenames. (e.g., replacing spaces with underscores).
## Suggested to use this tool if file names have inconsistent naming conventions that may cause issues in downstream processing.
import os

def batch_rename_files(directory=None):
    """
    Scans a directory and its subdirectories to rename files by replacing spaces and/or specified words in filenames.
    Folder names will not be changed.
    Args:
        directory (str): Directory to scan. If None, prompts user for input.
    """
    ## If no directory is provided, prompt the user for input
    if directory is None:
        directory = input("Enter the directory to scan: ").strip()
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    print(f"Scanning directory: {directory}")

    ## Option to replace spaces in filenames with different separator character
    replace_spaces = input("Do you want to replace spaces in filenames? (y/n): ").strip().lower()
    if replace_spaces == 'y':
        char_to_use = input("Enter the separator to use instead of spaces (e.g. _): ").strip()
        print("Renaming files now!")
        for root, dirs, files in os.walk(directory):
            for current_filename in files:
                if ' ' in current_filename:
                    old_filepath = os.path.join(root, current_filename)
                    new_filename = current_filename.replace(' ', char_to_use)
                    new_filepath = os.path.join(root, new_filename)
                    print(f"Renaming: {old_filepath} to {new_filepath}")
                    os.rename(old_filepath, new_filepath)
        print("File renaming complete.")
    else:
        print("No spaces will be replaced in filenames.")
    
    ## Option to replace other specified words in filenames-- will loop until user opts out
    while True:
        batch_replace = input("Do you want to replace other words in filenames? (y/n): ").strip().lower()
        if batch_replace != 'y':
            print("No other words will be replaced in filenames.")
            break
        print("Enter words to find and their replacements as comma-separated pairs (e.g. old1:new1,old2:new2)")
        print("Careful! This will rename files in bulk. Suggested to back up files first, or test on a small set.")
        pairs_input = input("Enter pairs: ").strip()
        word_pairs = [pair.split(':') for pair in pairs_input.split(',') if ':' in pair]
        batch_replace_check = input(f"You wrote: {word_pairs}, do you want to proceed? (y/n): ").strip().lower()
        if batch_replace_check == 'y':
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
            print("Batch word replacement complete. Would you like to replace more words?")
        else:
            print("Batch word replacement canceled.")

# To use interactively:
# batch_rename_files()