import os

root_dir = input("Enter the root directory to scan: ").strip()

for dirpath, _, filenames in os.walk(root_dir):
    for fname in filenames:
        if fname.endswith('.0.dpt'):
            old_path = os.path.join(dirpath, fname)
            new_fname = fname[:-6] + '.dpt'  # Remove '.0.dpt', add '.dpt'
            new_path = os.path.join(dirpath, new_fname)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")