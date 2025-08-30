
# ----------------- MODEL CONFIGURATION ----------------- #
MODEL_NAME = 'alibayram/medgemma:4b' # Switch between 'llama3', 'deepseek-coder', 'medgemma', etc.

# ----------------- FILE PATHS ----------------- #
PATIENTS_FILE_PATH = 'data/unified_samples_2.csv'
TEST_LABELS_FILE_PATH = '40_cases_unique_test_codes_with_labels.csv'

# ----------------- MEDICAL KNOWLEDGE ----------------- #
# Define which radiology tests the LLM is allowed to suggest
ALLOWED_RADIOLOGY_TESTS = [
    "CT ABD & PELVIS WITH CONTRAST", "LIVER OR GALLBLADDER US (SINGLE ORGAN)",
    "ABDOMEN (SUPINE & ERECT)", "CT ABDOMEN W/CONTRAST", "CHEST (PRE-OP PA & LAT)",
    "US ABD LIMIT, SINGLE ORGAN", "CTA CHEST W&W/O C&RECONS, NON-CORONARY",
    "MRCP (MR ABD W&W/OC)", "ERCP BILIARY&PANCREAS BY GI UNIT", "CT ABD & PELVIS W/O CONTRAST",
    "CHEST (PA & LAT)", "CHEST (PORTABLE AP)", "CT ABDOMEN W/O CONTRAST",
    "CT ABD W&W/O C", "UNILAT UP EXT VEINS US LEFT", "UNILAT UP EXT VEINS US RIGHT",
    "US APPENDIX"
]

# Map special test categories to data sources
TEST_MAPPING = {
    'Stool Culture': 'Microbiology',
    'Physical Examination': 'Physical Examination',
}

# Medical knowledge for smart diagnosis matching
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