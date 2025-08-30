# data_loader.py

import pandas as pd

def read_csv_file(file_path):
    """Reads a CSV file and returns a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path} with {len(df)} records.")
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        raise