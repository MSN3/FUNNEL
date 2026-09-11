import importlib.util
import sys
from data_loader import read_csv_file
import pandas as pd
import os

def load_config_from_path(config_path: str):
    """Dynamically loads a .py file as a config module."""
    try:
        config_path = os.path.abspath(config_path)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
        spec = importlib.util.spec_from_file_location("config", config_path)
        if spec is None:
            raise ImportError(f"Could not load spec from {config_path}")
            
        config = importlib.util.module_from_spec(spec)
        sys.modules["config"] = config 
        spec.loader.exec_module(config)
        print(f"Successfully loaded config from: {config_path}")
        return config
    except Exception as e:
        print(f"Failed to load config file: {e}")
        raise

def load_data_from_config(config):
    """Loads patient data and test labels/costs."""
    print("Loading patient data and test costs...")
    
    if not hasattr(config, 'PATIENTS_FILE_PATH') or not hasattr(config, 'TEST_LABELS_FILE_PATH'):
        raise AttributeError("Config file must define PATIENTS_FILE_PATH and TEST_LABELS_FILE_PATH")

    df_patients = read_csv_file(config.PATIENTS_FILE_PATH)
    df_test_labels = read_csv_file(config.TEST_LABELS_FILE_PATH) # Load the cost CSV
    
    df_patients.dropna(subset=['Patient ID'], how='all', inplace=True)
    
    # Standardize ground_truth_disease column using Discharge Diagnosis
    if 'ground_truth_disease' not in df_patients.columns:
        if 'Discharge Diagnosis' in df_patients.columns:
            df_patients['ground_truth_disease'] = df_patients['Discharge Diagnosis'].astype(str).str.split('|').str[-1].str.strip()
            print("Standardized 'ground_truth_disease' column created from Discharge Diagnosis.")

    # Dynamically extract available tests from the Excel to ensure names match exactly
    available_tests_set = set()
    for lab_str in df_patients['Laboratory Tests'].dropna():
        lab_str_clean = str(lab_str).strip()
        if lab_str_clean.lower() not in ['not available', 'nan', '']:
            for item in lab_str_clean.split(';'):
                if ':' in item:
                    test_name = item.split(':')[0].strip()
                    if test_name.startswith('-'): test_name = test_name[1:]
                    available_tests_set.add(test_name)
    
    available_tests = list(available_tests_set)
    label_to_code = {test: test for test in available_tests}
    
    # --- RE-INTEGRATED COST LOGIC ---
    test_costs = dict(zip(df_test_labels['lab_label'], df_test_labels['Cost (USD)']))
    
    # Add costs for radiology tests (default heuristic if not in file)
    for rad_test in config.ALLOWED_RADIOLOGY_TESTS:
        if rad_test not in test_costs:
            test_costs[rad_test] = 100.0
    
    patient_records = df_patients.to_dict('records')
    print(f"Loaded {len(patient_records)} patient records.")
    
    return patient_records, label_to_code, available_tests, test_costs