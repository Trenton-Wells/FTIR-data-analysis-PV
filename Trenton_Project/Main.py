import os
import pandas as pd
from prompt_toolkit import prompt
from Dataframe_Modification import baseline_selection, cast_param_types, get_default_params, parse_parameters, prompt_parameters
from File_Info_Gathering import file_info_extractor

if __name__ == "__main__":
    # Change files (y/n)
    File_Change = input("Do you want to change any file names? (y/n): ").lower()
    if File_Change.lower() == 'quit':
        print("Exiting program.")
        quit()
    if File_Change == 'y':
        from Fixing_File_Names import batch_rename_files
        directory = input("Enter the directory to scan for file renaming (or press Enter to specify later): ")
        if directory.lower() == 'quit':
            print("Exiting program.")
            quit()
        if not directory:
            directory = None
            print("No file renaming will be done.")
        batch_rename_files(directory)
    else:
        print("No file renaming will be done.")
    # Create new or modify existing dataframe
    while True:
        New_or_Existing = input("Do you want to create a new dataframe or append new files into an existing one? (new/append): ").lower()
        if New_or_Existing.lower() == 'quit':
            print("Exiting program.")
            raise SystemExit
        elif New_or_Existing == 'new':
            print("Creating a new dataframe.")
            file_info_extractor()
            break
        elif New_or_Existing == 'append':
            print("Appending new files into an existing dataframe.")
            file_info_extractor()
            break
        else:
            print("Invalid option. Please enter 'new' or 'append'.")
    # Baseline correction options
    while True:
        modify_choice = input("Do you want to modify the baseline function and parameters in the dataframe? (y/n): ").lower()
        if modify_choice == 'quit':
            print("Exiting program.")
            quit()
        elif modify_choice == 'n':
            print("No modifications will be made to the dataframe.")
            break
        elif modify_choice == 'y':
            while True:
                csv_path = input("Enter the path to the existing CSV file (or type 'quit' to exit): ")
                if csv_path.lower() == 'quit':
                    print("Exiting program.")
                    raise SystemExit
                if not os.path.isfile(csv_path):
                    print(f"File not found: {csv_path}. Please enter a valid path or type 'quit' to exit.")
                else:
                    break
            baseline_selection(csv_path)
            break
        else:
            print("Invalid option. Please enter 'y' or 'n'.")
