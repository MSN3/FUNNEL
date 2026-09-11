

import pandas as pd
import os

ROOT_DIR = ""
OUTPUT_FILE = ""

# --- Formatting Functions (as defined above) ---
def process_labs(filepath):
    print("Processing lab.csv (Keeping latest results)...")
    # Load offset to determine chronological order
    df = pd.read_csv(filepath, usecols=['patientunitstayid', 'labname', 'labresult', 'labresultoffset'])
    df = df.dropna(subset=['labname', 'labresult'])
    
    # Sort chronologically (oldest to newest)
    df = df.sort_values(by=['patientunitstayid', 'labname', 'labresultoffset'])
    
    # Drop duplicates, keeping the 'last' (newest) result for each specific lab name
    df = df.drop_duplicates(subset=['patientunitstayid', 'labname'], keep='last')
    
    df['formatted_text'] = df['labname'].astype(str) + ": " + df['labresult'].astype(str)
    grouped = df.groupby('patientunitstayid')['formatted_text'].apply(lambda x: '; '.join(x)).reset_index()
    grouped.rename(columns={'formatted_text': 'Laboratory Tests'}, inplace=True)
    return grouped

def process_exams(filepath):
    print("Processing physicalExam.csv (Keeping latest per category)...")
    # MODIFIED: Load physicalexamtext (the actual result) instead of physicalexamvalue
    df = pd.read_csv(filepath, usecols=['patientunitstayid', 'physicalexampath', 'physicalexamtext', 'physicalexamoffset'])
    df = df.dropna(subset=['physicalexampath', 'physicalexamtext'])
    
    # 1. Filter out administrative metadata rows
    ignore_terms = ['scored', 'performed - structured', 'performed', 'done']
    df = df[~df['physicalexamtext'].astype(str).str.strip().str.lower().isin(ignore_terms)]
    
    # 2. Extract a clean category name from the long path
    def extract_exam_category(path_str):
        if pd.isna(path_str): return "Unknown"
        parts = str(path_str).split('/')
        # Remove generic fluff from the path
        clean_parts = [p for p in parts if p not in ['notes', 'Progress Notes', 'Physical Exam']]
        
        if len(clean_parts) >= 2:
            # e.g., "Vital Signs - HR Current"
            return f"{clean_parts[-2]} - {clean_parts[-1]}"
        elif len(clean_parts) == 1:
            return clean_parts[0]
        else:
            return parts[-1]
            
    df['clean_category'] = df['physicalexampath'].apply(extract_exam_category)
    
    # 3. Sort chronologically
    df = df.sort_values(by=['patientunitstayid', 'clean_category', 'physicalexamoffset'])
    
    # 4. Keep only the newest exam finding for each specific category
    df = df.drop_duplicates(subset=['patientunitstayid', 'clean_category'], keep='last')
    
    # 5. Format as "Category: Result"
    df['formatted_text'] = df['clean_category'].astype(str) + ": " + df['physicalexamtext'].astype(str)
    
    grouped = df.groupby('patientunitstayid')['formatted_text'].apply(lambda x: '; '.join(x)).reset_index()
    grouped.rename(columns={'formatted_text': 'Physical Examination'}, inplace=True)
    
    return grouped

def process_simple_table(filepath, text_col, target_col_name):
    print(f"Processing {os.path.basename(filepath)}...")
    df = pd.read_csv(filepath, usecols=['patientunitstayid', text_col])
    df = df.dropna(subset=[text_col])
    grouped = df.groupby('patientunitstayid')[text_col].apply(lambda x: '; '.join(x.astype(str))).reset_index()
    grouped.rename(columns={text_col: target_col_name}, inplace=True)
    return grouped

def process_diagnosis(filepath):
    print("Processing diagnosis.csv...")
    df = pd.read_csv(filepath, usecols=['patientunitstayid', 'diagnosisstring', 'diagnosispriority'])
    # Filter for Primary only
    is_primary = df['diagnosispriority'].astype(str).str.strip().str.lower() == 'primary'
    df = df[is_primary].dropna(subset=['diagnosisstring'])
    grouped = df.groupby('patientunitstayid')['diagnosisstring'].apply(lambda x: '; '.join(x.astype(str))).reset_index()
    grouped.rename(columns={'diagnosisstring': 'Discharge Diagnosis'}, inplace=True)
    return grouped

def process_past_history(filepath):
    print("Processing pastHistory.csv (Keeping all unique items)...")
    # We don't strictly need the offset here since we are just collecting a unique set, 
    # but we load it to ensure consistent chronological reading if needed.
    df = pd.read_csv(filepath, usecols=['patientunitstayid', 'pasthistorypath', 'pasthistoryoffset'])
    df = df.dropna(subset=['pasthistorypath'])
    
    # Sort by offset just to keep the list in chronological order of discovery
    df = df.sort_values(by=['patientunitstayid', 'pasthistoryoffset'])
    
    # Drop exact duplicate history entries for the same patient
    df = df.drop_duplicates(subset=['patientunitstayid', 'pasthistorypath'])
    
    grouped = df.groupby('patientunitstayid')['pasthistorypath'].apply(lambda x: '; '.join(x.astype(str))).reset_index()
    grouped.rename(columns={'pasthistorypath': 'past_hist_temp'}, inplace=True)
    return grouped

# --- Main Execution ---
def main():
    print("Loading patient.csv...")
    df_main = pd.read_csv(os.path.join(ROOT_DIR, "patient.csv"), usecols=['uniquepid', 'patientunitstayid', 'age', 'gender'])
    df_main.rename(columns={'uniquepid': 'Patient ID', 'patientunitstayid': 'Visit ID', 'age': 'Age', 'gender': 'Gender'}, inplace=True)

    # Process all tables
    df_labs = process_labs(os.path.join(ROOT_DIR, "lab.csv"))
    df_exams = process_exams(os.path.join(ROOT_DIR, "physicalExam.csv"))
    df_past = process_past_history(os.path.join(ROOT_DIR, "pastHistory.csv"))
    df_admit = process_simple_table(os.path.join(ROOT_DIR, "admissionDx.csv"), 'admitdxname', 'admit_dx_temp')
    df_dx = process_diagnosis(os.path.join(ROOT_DIR, "diagnosis.csv"))

    # Process microLab if it exists, else create empty
    micro_path = os.path.join(ROOT_DIR, "microLab.csv")
    if os.path.exists(micro_path):
        print("Processing microLab.csv...")
        df_micro = pd.read_csv(micro_path, usecols=['patientunitstayid', 'culturesite', 'organism', 'antibiotic', 'sensitivitylevel'])
        
        # 1. Drop rows where there is no organism identified at all
        df_micro = df_micro.dropna(subset=['organism'])
        
        # 2. Fill missing antibiotic/sensitivity data so we don't get "nan" strings
        df_micro['antibiotic'] = df_micro['antibiotic'].fillna("Unknown")
        df_micro['sensitivitylevel'] = df_micro['sensitivitylevel'].fillna("Unknown")
        
        # 3. Create the formatted string
        df_micro['formatted_text'] = df_micro['culturesite'].astype(str) + " - " + df_micro['organism'].astype(str) + " (" + df_micro['antibiotic'].astype(str) + ": " + df_micro['sensitivitylevel'].astype(str) + ")"
        
        # 4. Group by patient and safely join ONLY valid strings
        df_micro = df_micro.groupby('patientunitstayid')['formatted_text'].apply(
            lambda x: '; '.join(x.dropna().astype(str))
        ).reset_index()
        
        df_micro.rename(columns={'formatted_text': 'Microbiology'}, inplace=True)
    else:
        df_micro = pd.DataFrame(columns=['patientunitstayid', 'Microbiology'])

    # Process treatment if it exists
    tx_path = os.path.join(ROOT_DIR, "treatment.csv")
    if os.path.exists(tx_path):
        df_tx = process_simple_table(tx_path, 'treatmentstring', 'Procedures Discharge')
    else:
        df_tx = pd.DataFrame(columns=['patientunitstayid', 'Procedures Discharge'])


    print("Merging tables...")
    # INNER JOIN labs to guarantee patient has lab data
    df_final = pd.merge(df_main, df_labs, left_on='Visit ID', right_on='patientunitstayid', how='inner').drop(columns=['patientunitstayid'])
    
    # LEFT JOIN everything else
    for t in [df_exams, df_past, df_admit, df_dx, df_micro, df_tx]:
        if not t.empty:
            df_final = pd.merge(df_final, t, left_on='Visit ID', right_on='patientunitstayid', how='left').drop(columns=['patientunitstayid'])

    # Clean up Patient History
    df_final['past_hist_temp'] = df_final['past_hist_temp'].fillna("None")
    df_final['admit_dx_temp'] = df_final['admit_dx_temp'].fillna("None")
    df_final['Patient History'] = "Admission: " + df_final['admit_dx_temp'] + " | History: " + df_final['past_hist_temp']

    # Finalize missing data
    for col in ['Microbiology', 'Physical Examination', 'Discharge Diagnosis', 'Procedures Discharge']:
        if col in df_final.columns:
            df_final[col] = df_final[col].fillna("Not Available")
    
    df_final['Radiology'] = "Not Available"
    
    # Target ordering
    target_order = ['Patient ID', 'Visit ID', 'Age', 'Gender', 'Patient History', 'Physical Examination', 'Laboratory Tests', 'Radiology', 'Microbiology', 'Discharge Diagnosis', 'Procedures Discharge']
    df_final = df_final[[c for c in target_order if c in df_final.columns]]

    # Drop missing diagnoses
    initial_len = len(df_final)
    df_final = df_final[(df_final['Discharge Diagnosis'] != "Not Available") & (df_final['Discharge Diagnosis'] != "")]
    print(f"Dropped {initial_len - len(df_final)} patients due to missing primary diagnosis.")
    
    # Drop missing labs
    initial_len = len(df_final)
    df_final = df_final[(df_final['Laboratory Tests'] != "Not Available") & (df_final['Laboratory Tests'] != "")]
    print(f"Dropped {initial_len - len(df_final)} patients due to missing labs.")

    df_final.to_excel(OUTPUT_FILE, index=False)
    print(f"Success! Saved {len(df_final)} fully populated records.")

if __name__ == "__main__":
    main()