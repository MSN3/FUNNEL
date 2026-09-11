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

def load_data_with_costs(patients_path, labels_path, config):
    """
    Loads patients and test labels, including cost data.
    Returns: df_patients, label_to_code, available_tests, test_costs
    """
    df_patients = read_csv_file(patients_path)
    # Remove empty rows
    df_patients.dropna(subset=['Patient ID'], how='all', inplace=True)
    
    df_test_labels = read_csv_file(labels_path)
    
    # 1. Map Labels to Codes
    label_to_code = dict(zip(df_test_labels['lab_label'], df_test_labels['Test Code']))
    available_tests = df_test_labels['lab_label'].tolist()
    
    # 2. Extract Costs
    # Check if 'Cost(USD)' exists, otherwise default to empty
    if 'Cost(USD)' in df_test_labels.columns:
        test_costs = dict(zip(df_test_labels['lab_label'], df_test_labels['Cost(USD)']))
    else:
        print("Warning: 'Cost(USD)' column not found in labels file. Costs will default to $50.")
        test_costs = {}

    # 3. Add default costs for known Radiology tests (heuristic)
    for rad_test in config.ALLOWED_RADIOLOGY_TESTS:
        if rad_test not in test_costs:
            test_costs[rad_test] = 100.0

    return df_patients, label_to_code, available_tests, test_costs