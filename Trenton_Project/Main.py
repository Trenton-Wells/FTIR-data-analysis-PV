import os
import pandas as pd
from prompt_toolkit import prompt
from Dataframe_Modification import modify_dataframe, cast_param_types, get_default_params, parse_parameters, prompt_parameters
from File_Info_Gathering import file_info_extractor

if __name__ == "__main__":
    # Change files (y/n)
    File_Change = input("Do you want to change any file names? (y/n): ").lower()
    if File_Change.lower() == 'quit':
        print("Exiting program.")
        raise SystemExit
    if File_Change == 'y':
        from Fixing_File_Names import batch_rename_files
        directory = input("Enter the directory to scan for file renaming (or press Enter to specify later): ")
        if directory.lower() == 'quit':
            print("Exiting program.")
            raise SystemExit
        if not directory:
            directory = None
            print("No file renaming will be done.")
        batch_rename_files(directory)
    else:
        print("No file renaming will be done.")
    # Create new or modify existing dataframe
    while True:
        New_or_Existing = input("Do you want to create a new dataframe or modify an existing one? (new/modify): ").lower()
        if New_or_Existing.lower() == 'quit':
            print("Exiting program.")
            raise SystemExit
        elif New_or_Existing == 'new':
            print("Creating a new dataframe.")
            file_info_extractor()
            break
        elif New_or_Existing == 'modify':
            Info_Extraction = input("Do you want to extract file info to create/append dataframe? (y/n): ").lower()
            if Info_Extraction.lower() == 'quit':
                print("Exiting program.")
                raise SystemExit
            if Info_Extraction == 'y':
                file_info_extractor()
            else:
                print("No file info extraction will be done.")
        while True:
            csv_path = input("Enter the path to the dataframe CSV file: (default is Trenton_Project\\dataframe.csv)").strip()
            if csv_path.lower() == 'quit':
                print("Exiting program.")
                raise SystemExit
            if not csv_path:
                csv_path = r"Trenton_Project\dataframe.csv"  # Default path if none provided
            if not os.path.isfile(csv_path):
                print(f"File not found: {csv_path}. Please enter a valid path or type 'quit' to exit.")
            else:
                break
            modify_dataframe(csv_path)
            break
        else:
            print("Invalid option. Please enter 'new' or 'modify'.")
    # Baseline correction
