# config.py

# --- SELECT YOUR PROVIDER HERE ---
# Options: "vllm", "google_vertex", "anthropic"
AGENT_PROVIDER = "vllm" 

# --- MODEL NAMES ---
# vLLM Example: "google/medgemma-27b-text-it" (or whatever you serve)
# Vertex Example: "gemini-1.5-pro-preview-0409"
# Anthropic Example: "claude-3-5-sonnet-20240620"
AGENT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# --- API CREDENTIALS ---
# vLLM
AGENT_BASE_URL = "http://..." 
AGENT_API_KEY = "EMPTY"

# Google Vertex (GCP)
GOOGLE_PROJECT_ID = "project_name"
GOOGLE_LOCATION = "us-central1" 

# # Anthropic
# ANTHROPIC_API_KEY = "sk-ant-..."

# The separate, more powerful model used for evaluation
JUDGE_PROVIDER = "vllm"
JUDGE_MODEL_NAME = "google/medgemma-27b-text-it" 
JUDGE_BASE_URL = "http://..."

# SIMILARITY_MODEL_NAME = 'Qwen/Qwen3-Embedding-4B'
# # The cosine similarity score above which two diagnoses are considered equivalent.
# # Typically, a value between 0.75 and 0.9 is a good starting point.
# SIMILARITY_THRESHOLD = 0.8

# Set to False to disable cost checking (accepts all tests)
USE_LLM_FRUGALITY = True

# Maximum budget per patient case (in dollars)
BUDGET_CAP = 400.0

# ----------------- PROCESSING CONFIGURATION ----------------- #
# Number of patient records to process in each concurrent batch.
CHUNK_SIZE = 100

# ----------------- FILE PATHS ----------------- #
PATIENTS_FILE_PATH = ''
TEST_LABELS_FILE_PATH = ''

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
    'cholecystitis': ['cholelithiasis', 'biliary colic', 'gallstones', 'gallbladder inflammation', 'common bile duct stones', 'choledocholithiasis', 'biliary obstruction'],
    #'choledocholithiasis': ['common bile duct stones', 'biliary obstruction'],
    'pancreatitis': ['pancreatic inflammation', 'acute pancreatitis'],
    'appendicitis': ['appendiceal inflammation', 'acute appendicitis', 'chronic appendicitis', 'ruptured appendix', 'perforated appendix', 'inflamed appendix'],
    'diverticulitis': ['diverticular disease', 'colonic diverticulosis'],
    #'peptic ulcer': ['gastric ulcer', 'duodenal ulcer', 'gastritis'],
    #'hepatitis': ['liver inflammation', 'viral hepatitis'],
    'stroke': ['cerebrovascular accident', 'CVA', 'ischemic stroke', 'hemorrhagic stroke', 'brain infarction', 'transient ischemic attack', 'TIA'],
    'myocardial infarction': ['heart attack', 'MI', 'acute coronary syndrome', 'ACS', 'STEMI', 'NSTEMI', 'Pulmonary Edema'],
    'renal failure': ['kidney failure', 'acute kidney injury', 'AKI', 'chronic kidney disease', 'CKD', 'end stage renal disease', 'ESRD'],
    'copd': ['chronic obstructive pulmonary disease', 'chronic bronchitis', 'emphysema', 'chronic lung disease'],
    'pneumonia': ['lung infection', 'bronchopneumonia', 'lobar pneumonia', 'community acquired pneumonia', 'CAP', 'hospital acquired pneumonia', 'HAP'],
    'sepsis': ['septicemia', 'blood infection', 'septic shock', 'systemic infection', 'bacteremia']
}