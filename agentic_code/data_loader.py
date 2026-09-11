# data_loader.py
import pandas as pd
import os

def read_csv_file(file_path):
    """Reads a CSV file and returns a DataFrame."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at {file_path}")
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path} with {len(df)} records.")
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        raise