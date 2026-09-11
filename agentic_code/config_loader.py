# config_loader.py
import importlib.util
import sys
from data_loader import read_csv_file
import pandas as pd
import os

def load_config_from_path(config_path: str):
    """Dynamically loads a .py file as a config module."""
    try:
        # Ensure path is absolute
        config_path = os.path.abspath(config_path)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
        spec = importlib.util.spec_from_file_location("config", config_path)
        if spec is None:
            raise ImportError(f"Could not load spec from {config_path}")
            
        config = importlib.util.module_from_spec(spec)
        
        # Add to sys.modules to allow deep imports if config imports other files
        sys.modules["config"] = config 
        
        spec.loader.exec_module(config)
        print(f"Successfully loaded config from: {config_path}")
        return config
    except Exception as e:
        print(f"Failed to load config file: {e}")
        raise

def load_data_from_config(config):
    """Loads all data files specified in the config module."""
    print("Loading patient and test data...")
    
    if not hasattr(config, 'PATIENTS_FILE_PATH') or not hasattr(config, 'TEST_LABELS_FILE_PATH'):
        raise AttributeError("Config file must define PATIENTS_FILE_PATH and TEST_LABELS_FILE_PATH")

    df_patients = read_csv_file(config.PATIENTS_FILE_PATH)
    df_test_labels = read_csv_file(config.TEST_LABELS_FILE_PATH)
    
    # --- Pre-processing logic from main.py ---
    # Remove empty rows
    df_patients.dropna(subset=['Patient ID'], how='all', inplace=True)
    
    # Create standardized ground_truth_disease column
    if 'ground_truth_disease' not in df_patients.columns:
        mask = df_patients['Patient ID'].str.contains('_', regex=False, na=False)
        extracted_disease_str = df_patients.loc[mask, 'Patient ID'].str.split('_', n=1).str[1]
        standardized_disease = extracted_disease_str.str.replace('_', ' ').str.strip()
        df_patients.loc[mask, 'ground_truth_disease'] = standardized_disease
        print("Standardized 'ground_truth_disease' column created.")
    else:
        print("'ground_truth_disease' column already exists.")

    # Create test mappings
    label_to_code = dict(zip(df_test_labels['lab_label'], df_test_labels['Test Code']))
    available_tests = df_test_labels['lab_label'].tolist()
    
    #---------------------------------Adding Cost factor---------------------------------
    # Include test costs
    test_costs = dict(zip(df_test_labels['lab_label'], df_test_labels['Cost(USD)']))
    
    # Add costs for radiology tests (default heuristic if not in file)
    for rad_test in config.ALLOWED_RADIOLOGY_TESTS:
        if rad_test not in test_costs:
            test_costs[rad_test] = 100.0
    #--------------------------------------------------------------------------------------s
    
    # Convert patient df to list of dicts for processing
    patient_records = df_patients.to_dict('records')
    print(f"Loaded {len(patient_records)} patient records.")
    
    return patient_records, label_to_code, available_tests, test_costs