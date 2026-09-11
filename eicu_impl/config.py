# config.py

# --- SELECT YOUR PROVIDER HERE ---
# Options: "vllm", "google_vertex", "anthropic", "bedrock"
AGENT_PROVIDER = "vllm" 

# --- MODEL NAMES ---
# vLLM Example: "google/medgemma-27b-text-it" (or whatever you serve)
# Vertex Example: "gemini-1.5-pro-preview-0409"
# Anthropic Example: "claude-3-5-sonnet-20240620"
AGENT_MODEL_NAME = "google/medgemma-27b-text-it"

# --- API CREDENTIALS ---
# vLLM
AGENT_BASE_URL = "http://..."
AGENT_API_KEY = "EMPTY"

# Google Vertex (GCP)
GOOGLE_PROJECT_ID = "project_name"
GOOGLE_LOCATION = "us-central1"

# The separate, more powerful model used for evaluation
JUDGE_PROVIDER = "vllm"
JUDGE_MODEL_NAME = "google/medgemma-27b-text-it" 
JUDGE_BASE_URL = "http://..."

# SIMILARITY_MODEL_NAME = 'Qwen/Qwen3-Embedding-4B'
# # The cosine similarity score above which two diagnoses are considered equivalent.
# # Typically, a value between 0.75 and 0.9 is a good starting point.
# SIMILARITY_THRESHOLD = 0.8

# Set to False to disable cost checking (accepts all tests)
USE_LLM_FRUGALITY = False

# Maximum budget per patient case (in dollars)
BUDGET_CAP = 150.0

# ----------------- PROCESSING CONFIGURATION ----------------- #
# Number of patient records to process in each concurrent batch.
CHUNK_SIZE = 100

# ----------------- FILE PATHS ----------------- #
PATIENTS_FILE_PATH = ''
TEST_LABELS_FILE_PATH = ''

# ----------------- MEDICAL KNOWLEDGE ----------------- #
# (This section remains unchanged)
ALLOWED_RADIOLOGY_TESTS = []
# Updated to match the exact string prefixes found in the eICU Microbiology column
TEST_MAPPING = {
    'Blood, Venipuncture': 'Microbiology',
    'Blood, Central Line': 'Microbiology',
    'Sputum, Tracheal Specimen': 'Microbiology',
    'Sputum, Expectorated': 'Microbiology',
    'Urine, Catheter Specimen': 'Microbiology',
    'Nasopharynx': 'Microbiology',
    'Bronchial Lavage': 'Microbiology',
    'Wound, Decubitus': 'Microbiology',
    'Stool': 'Microbiology',
    'Physical Examination': 'Physical Examination',
    # Adding generic fallbacks in case the LLM suggests standard names
    'Blood Culture': 'Microbiology',
    'Urine Culture': 'Microbiology',
    'Sputum Culture': 'Microbiology'
}

# Updated to perfectly match the top 40 exact string formulations in eICU Discharge Diagnosis
RELATED_DIAGNOSES = {
    'sepsis': ['septic shock', 'severe sepsis', 'septicemia', 'blood infection', 'systemic infection'],
    'respiratory failure': ['acute respiratory failure', 'acute respiratory distress', 'ards', 'respiratory arrest'],
    'heart failure': ['congestive heart failure', 'chf', 'acute pulmonary edema', 'fluid overload'],
    'myocardial infarction': ['acute myocardial infarction (no st elevation)', 'acute myocardial infarction (with st elevation)', 'acute coronary syndrome', 'mi', 'stemi', 'nstemi', 'chest pain'],
    'stroke': ['ischemic stroke', 'hemorrhagic stroke', 'cerebrovascular accident', 'cva', 'brain infarction', 'transient ischemic attack', 'tia'],
    'gastrointestinal bleeding': ['gi bleeding', 'upper gi bleeding', 'lower gi bleeding'],
    'copd': ['acute copd exacerbation', 'chronic obstructive pulmonary disease', 'emphysema', 'chronic bronchitis'],
    'renal failure': ['acute renal failure', 'kidney failure', 'aki', 'chronic kidney disease', 'ckd', 'end stage renal disease'],
    'atrial fibrillation': ['with rapid ventricular response', 'afib', 'rvr', 'supraventricular tachycardia', 'arrhythmias'],
    'diabetic ketoacidosis': ['dka', 'diabetic crisis', 'severe hyperglycemia'],
    'pneumonia': ['lung infection', 'bronchopneumonia', 'lobar pneumonia', 'community acquired pneumonia', 'hospital acquired pneumonia'],
    'cardiac arrest': ['pulseless electrical activity', 'ventricular fibrillation', 'asystole'],
    'altered mental status': ['change in mental status', 'encephalopathy', 'delirium'],
    'seizures': ['status epilepticus', 'epilepsy', 'convulsions']
}