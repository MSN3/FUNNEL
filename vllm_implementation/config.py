# config.py

# ----------------- MODEL CONFIGURATION ----------------- #
# This name MUST match the model loaded by your vLLM server.
# Example: "meta-llama/Llama-3-8B-Instruct" or "medicalai/MedGemma-7B"
MODEL_NAME = "google/medgemma-4b-it"

# ----------------- PROCESSING CONFIGURATION ----------------- #
# Number of patient records to process in each concurrent batch.
CHUNK_SIZE = 50

# ----------------- FILE PATHS ----------------- #
PATIENTS_FILE_PATH = '/scratch/nshanb2/Frugal_LLM/data/unified_samples.csv'
TEST_LABELS_FILE_PATH = '/scratch/nshanb2/Frugal_LLM/data/40_cases_unique_test_codes_with_labels.csv'

# ----------------- MEDICAL KNOWLEDGE ----------------- #
# (This section remains unchanged)
ALLOWED_RADIOLOGY_TESTS = [
    "CT ABD & PELVIS WITH CONTRAST", "LIVER OR GALLBLADDER US (SINGLE ORGAN)",
    "ABDOMEN (SUPINE & ERECT)", "CT ABDOMEN W/CONTRAST", "CHEST (PRE-OP PA & LAT)",
    "US ABD LIMIT, SINGLE ORGAN", "CTA CHEST W&W/O C&RECONS, NON-CORONARY",
    "MRCP (MR ABD W&W/OC)", "ERCP BILIARY&PANCREAS BY GI UNIT", "CT ABD & PELVIS W/O CONTRAST",
    "CHEST (PA & LAT)", "CHEST (PORTABLE AP)", "CT ABDOMEN W/O CONTRAST",
    "CT ABD W&W/O C", "UNILAT UP EXT VEINS US LEFT", "UNILAT UP EXT VEINS US RIGHT",
    "US APPENDIX"
]
TEST_MAPPING = {
    'Stool Culture': 'Microbiology',
    'Physical Examination': 'Physical Examination',
}
RELATED_DIAGNOSES = {
    'cholecystitis': ['cholelithiasis', 'biliary colic', 'gallstones', 'gallbladder inflammation'],
    'choledocholithiasis': ['common bile duct stones', 'biliary obstruction'],
    'pancreatitis': ['pancreatic inflammation', 'acute pancreatitis'],
    'appendicitis': ['appendiceal inflammation', 'acute appendicitis', 'chronic appendicitis', 'ruptured appendix', 'perforated appendix', 'inflamed appendix'],
    'diverticulitis': ['diverticular disease', 'colonic diverticulosis'],
    'peptic ulcer': ['gastric ulcer', 'duodenal ulcer', 'gastritis'],
    'hepatitis': ['liver inflammation', 'viral hepatitis'],
    'stroke': ['cerebrovascular accident', 'CVA', 'ischemic stroke', 'hemorrhagic stroke', 'brain infarction', 'transient ischemic attack', 'TIA'],
    'myocardial infarction': ['heart attack', 'MI', 'acute coronary syndrome', 'ACS', 'STEMI', 'NSTEMI', 'Pulmonary Edema'],
    'renal failure': ['kidney failure', 'acute kidney injury', 'AKI', 'chronic kidney disease', 'CKD', 'end stage renal disease', 'ESRD'],
    'COPD': ['chronic obstructive pulmonary disease', 'chronic bronchitis', 'emphysema', 'chronic lung disease'],
    'pneumonia': ['lung infection', 'bronchopneumonia', 'lobar pneumonia', 'community acquired pneumonia', 'CAP', 'hospital acquired pneumonia', 'HAP'],
    'sepsis': ['septicemia', 'blood infection', 'septic shock', 'systemic infection', 'bacteremia']
}