# import pandas as pd
# import os
# import glob
# import re

# # --- Configuration ---
# # 1. Path to the folder containing your data files.
# #    Use '.' to represent the current directory where the script is running.
# input_folder = 'D:/Frugal_LLM/data' 

# # 2. Name for the final combined CSV file.
# output_filename = 'D:/Frugal_LLM/data/unified_samples.csv'

# # 3. Number of random samples to take from each file.
# samples_per_file = 120

# # --- Script Logic ---

# def extract_disease_from_filename(filename):
#     """
#     Extracts a disease name from a filename, handling single and multi-word diseases.
#     Examples:
#     '..._stroke_patients.xlsx' -> 'Stroke'
#     '..._renal_failure_patients.csv' -> 'renal_failure'
#     '..._myocardial_infarction_patients.csv' -> 'myocardial_infarction'
#     """
#     base_name = os.path.splitext(filename)[0]
    
#     # Define start and end markers for complex, multi-word names
#     start_marker = '_no_discharge_'
#     end_marker = '_patients'
    
#     # First, try to find the multi-word pattern
#     if start_marker in base_name and end_marker in base_name:
#         try:
#             start_index = base_name.find(start_marker) + len(start_marker)
#             end_index = base_name.rfind(end_marker) # Use rfind to be safe
#             if start_index < end_index:
#                 # Extract the multi-word disease name, e.g., 'renal_failure'
#                 return base_name[start_index:end_index]
#         except Exception:
#             # If indexing fails, fall through to the next method
#             pass

#     # Fallback to the original single-word logic for simpler filenames
#     parts = base_name.split('_')
#     try:
#         patients_index = parts.index('patients')
#         if patients_index > 0:
#             disease_name = parts[patients_index - 1]
#             # Capitalize the single word for consistency, e.g., 'Stroke'
#             return disease_name.capitalize()
#     except ValueError:
#         print(f"  -> Warning: Could not find '_patients' pattern in '{filename}'. Disease name will be 'Unknown'.")
#         pass
        
#     # Final fallback if no patterns match
#     return 'Unknown'


# def sample_and_combine_files(folder_path, output_path, num_samples):
#     """
#     Finds all CSV and Excel files in a folder, takes a random sample from each,
#     modifies the first column to include the disease name, and combines them
#     into a single CSV file.
#     """
#     # Find all .csv and .xlsx files in the specified folder
#     csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
#     excel_files = glob.glob(os.path.join(folder_path, '*.xlsx'))
#     all_files = csv_files + excel_files

#     if not all_files:
#         print(f"No CSV or Excel files found in the directory: {folder_path}")
#         return

#     print(f"Found {len(all_files)} files to process.")
    
#     # A list to hold the sampled data from each file
#     list_of_samples = []

#     # Loop through each found file
#     for file_path in all_files:
#         filename = os.path.basename(file_path)
#         print(f"Processing '{filename}'...")
#         try:
#             # Read the file into a pandas DataFrame
#             if file_path.endswith('.csv'):
#                 df = pd.read_csv(file_path, low_memory=False)
#             else: # .xlsx
#                 df = pd.read_excel(file_path)

#             # --- UPDATED LOGIC: Filter by 'Laboratory Tests' column ---
#             if 'Laboratory Tests' in df.columns:
#                 original_row_count = len(df)
#                 # First, drop rows where 'Laboratory Tests' is truly NaN (empty)
#                 df.dropna(subset=['Laboratory Tests'], inplace=True)
                
#                 # Next, filter out rows where the value is an empty dictionary string '{}'
#                 # We convert to string and strip whitespace to handle variations safely
#                 df = df[df['Laboratory Tests'].astype(str).str.strip() != '{}']
                
#                 print(f"  -> Filtered by 'Laboratory Tests', kept {len(df)} of {original_row_count} rows.")
#             else:
#                 print("  -> 'Laboratory Tests' column not found, skipping filter.")

#             # Determine the number of samples to take
#             n_samples = min(num_samples, len(df))
#             # n_samples = 2

#             if n_samples > 0:
#                 # Take a random sample
#                 sampled_df = df.sample(n=n_samples)
                
#                 # --- Modify the first column ---
#                 # 1. Extract the disease name from the filename
#                 disease = extract_disease_from_filename(filename)
                
#                 # 2. Get the name of the first column
#                 first_column_name = sampled_df.columns[0]
                
#                 # 3. Append the disease name to the patient IDs in that column
#                 #    .astype(str) ensures that IDs are treated as text
#                 sampled_df[first_column_name] = sampled_df[first_column_name].astype(str) + '_' + disease
                
#                 list_of_samples.append(sampled_df)
#                 print(f"  -> Sampled {n_samples} rows and appended disease '{disease}'.")
#             else:
#                 print("  -> File is empty or no rows remained after filtering, skipping.")

#         except Exception as e:
#             print(f"  -> Error processing file: {e}. Skipping.")
    
#     # Combine all the sampled DataFrames into one
#     if list_of_samples:
#         print("\nCombining all samples...")
#         unified_df = pd.concat(list_of_samples, ignore_index=True, sort=False)
        
#         # --- NEW LOGIC: Rename the first column ---
#         if not unified_df.empty:
#             original_first_col = unified_df.columns[0]
#             unified_df.rename(columns={original_first_col: 'Patient ID'}, inplace=True)
#             print(f"Renamed first column to 'Patient ID'.")

#         # Save the final DataFrame to a CSV file
#         unified_df.to_csv(output_path, index=False)
#         print(f"\nSuccess! Unified data with {len(unified_df)} total rows saved to '{output_path}'")
#     else:
#         print("\nNo data was sampled. Output file not created.")

# # --- Run the main function ---
# if __name__ == '__main__':
#     sample_and_combine_files(input_folder, output_filename, samples_per_file)

import pandas as pd
import os
import glob
import re

# --- Configuration ---
# 1. Path to the folder containing your data files.
input_folder = 'D:/Frugal_LLM/data'  

# 2. Name for the final combined CSV file.
output_filename = 'D:/Frugal_LLM/data/unified_full_mimic.csv'

# --- MODIFIED ---
# The 'samples_per_file' variable is no longer needed, as we are extracting all matching rows.

# --- Script Logic ---

def extract_disease_from_filename(filename):
    """
    Extracts a disease name from a filename, handling single and multi-word diseases.
    """
    base_name = os.path.splitext(filename)[0]
    
    # Define start and end markers for complex, multi-word names
    start_marker = '_no_discharge_'
    end_marker = '_patients'
    
    # First, try to find the multi-word pattern
    if start_marker in base_name and end_marker in base_name:
        try:
            start_index = base_name.find(start_marker) + len(start_marker)
            end_index = base_name.rfind(end_marker) # Use rfind to be safe
            if start_index < end_index:
                # Extract the multi-word disease name, e.g., 'renal_failure'
                return base_name[start_index:end_index]
        except Exception:
            # If indexing fails, fall through to the next method
            pass

    # Fallback to the original single-word logic for simpler filenames
    parts = base_name.split('_')
    try:
        patients_index = parts.index('patients')
        if patients_index > 0:
            disease_name = parts[patients_index - 1]
            # Capitalize the single word for consistency, e.g., 'Stroke'
            return disease_name.capitalize()
    except ValueError:
        print(f"  -> Warning: Could not find '_patients' pattern in '{filename}'. Disease name will be 'Unknown'.")
        pass
        
    # Final fallback if no patterns match
    return 'Unknown'


def extract_and_combine_files(folder_path, output_path):
    """
    Finds all CSV and Excel files in a folder, applies strict filtering for 
    multiple key columns, and combines all resulting valid rows into a single CSV.
    """
    # Define the list of columns that MUST NOT have missing (NaN) values
    REQUIRED_COLS = ['Patient History', 'ICD Diagnosis', 'Laboratory Tests']
    
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    excel_files = glob.glob(os.path.join(folder_path, '*.xlsx'))
    all_files = csv_files + excel_files

    if not all_files:
        print(f"No CSV or Excel files found in the directory: {folder_path}")
        return

    print(f"Found {len(all_files)} files to process.")
    
    list_of_dataframes = []

    for file_path in all_files:
        filename = os.path.basename(file_path)
        print(f"Processing '{filename}'...")
        try:
            # Read the file into a pandas DataFrame
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, low_memory=False)
            else: # .xlsx
                df = pd.read_excel(file_path)

            original_row_count = len(df)
            
            # --- MODIFIED: Check if all four required columns exist ---
            if not all(col in df.columns for col in REQUIRED_COLS):
                missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
                print(f"  -> Skipping file: Missing essential columns: {', '.join(missing_cols)}")
                continue 
            
            # 1. Drop rows where ANY of the required columns are truly NaN (empty)
            df.dropna(subset=REQUIRED_COLS, inplace=True)
            
            # 2. Further filter 'Laboratory Tests' for string representations of empty data
            if not df.empty:
                stripped_series = df['Laboratory Tests'].astype(str).str.strip()
                empty_values = ['{}', '[]', ''] # Check for empty dictionary, list, or whitespace
                
                # Keep rows where the value is NOT one of the empty strings
                mask_to_keep = ~stripped_series.isin(empty_values)
                df = df[mask_to_keep]
            
            print(f"  -> Filtered, kept {len(df)} of {original_row_count} rows.")

            if not df.empty:
                # We use .copy() to ensure we are working with a clean, independent DataFrame
                processed_df = df.copy()
                
                # --- Modify the first column (Original Patient ID) ---
                disease = extract_disease_from_filename(filename)
                first_column_name = processed_df.columns[0]
                
                # Append the disease name (extracted from the filename) to the IDs
                processed_df[first_column_name] = processed_df[first_column_name].astype(str) + '_' + disease
                
                list_of_dataframes.append(processed_df)
                print(f"  -> Extracted {len(processed_df)} rows and appended disease '{disease}'.")
            else:
                print("  -> No rows remained after strict filtering, skipping.")

        except Exception as e:
            print(f"  -> Error processing file '{filename}': {e}. Skipping.")
    
    # Combine all the DataFrames into one
    if list_of_dataframes:
        print("\nCombining all dataframes...")
        unified_df = pd.concat(list_of_dataframes, ignore_index=True, sort=False)
        
        # --- Rename the first column ---
        if not unified_df.empty:
            original_first_col = unified_df.columns[0]
            unified_df.rename(columns={original_first_col: 'Patient ID'}, inplace=True)
            print(f"Renamed first column to 'Patient ID'.")

        # Save the final DataFrame to a CSV file
        unified_df.to_csv(output_path, index=False)
        print(f"\nSuccess! Unified data with {len(unified_df)} total rows saved to '{output_path}'")
    else:
        print("\nNo data was extracted. Output file not created.")

# --- Run the main function ---
if __name__ == '__main__':
    extract_and_combine_files(input_folder, output_filename)