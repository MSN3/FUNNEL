# utils.py

import pandas as pd
import numpy as np
import re
import ast
import torch
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from config import RELATED_DIAGNOSES, TEST_MAPPING, ALLOWED_RADIOLOGY_TESTS
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingEvaluator:
    """
    Handles loading a sentence transformer model and comparing diagnoses
    based on the cosine similarity of their embeddings.
    """
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        print(f"Loading sentence transformer model: {model_name}...")
        self.model = SentenceTransformer(model_name,
            model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto", "dtype": torch.float16},
            tokenizer_kwargs={"padding_side": "left"},
            cache_folder='/scratch/nshanb2/cache_custom/models')

        print("Sentence transformer model loaded successfully.")
    
    def fit(self, related_diagnoses_dict):
        """
        Pre-computes and stores embeddings for the keys of the related diagnoses dictionary.
        """
        print("Pre-computing embeddings for related diagnosis keys...")
        self.known_diagnoses_keys = list(related_diagnoses_dict.keys())
        self.known_diagnoses_embeddings = self.model.encode(self.known_diagnoses_keys)
        print(f"{len(self.known_diagnoses_keys)} unique diagnosis key embeddings pre-computed.")

    def evaluate_diagnosis(self, llm_pred, ground_truth, threshold=0.80):
        llm_pred = str(llm_pred).lower().strip() if llm_pred else ""
        ground_truth = str(ground_truth).lower().strip() if ground_truth else ""

        if not llm_pred or 'unable' in llm_pred:
            return False, 0.0, "No valid diagnosis provided"

        # ✓ 1. Exact match
        if llm_pred == ground_truth:
            return True, 1.0, ground_truth

        # ✓ 2. Fuzzy string matching
        if fuzz.token_set_ratio(llm_pred, ground_truth) >= 80:
            return True, 0.9, ground_truth
        
        for main_key, synonyms in RELATED_DIAGNOSES.items():
            all_terms = [main_key] + [term.lower() for term in synonyms]

            # If the ground truth belongs to this canonical group
            if ground_truth in all_terms:
                # Check if LLM prediction matches any synonym in this group
                for term in all_terms:
                    if fuzz.token_set_ratio(llm_pred, term) >= 80:
                        return True, 0.9, main_key

        # ✓ 3. Embedding-based semantic similarity
        try:
            emb_pred, emb_true = self.model.encode([llm_pred, ground_truth])
            sim = cosine_similarity([emb_pred], [emb_true])[0][0]
        except Exception as e:
            print(f"Embedding error: {e}")
            return False, 0.0, "Error in embedding"

        # ✓ 4. Find nearest known diagnosis group if incorrect
        if self.known_diagnoses_embeddings is not None:
            sims = cosine_similarity([emb_pred], self.known_diagnoses_embeddings)[0]
            best_idx = sims.argmax()
            best_key = self.known_diagnoses_keys[best_idx]
            if best_key == ground_truth: 
                return True, float(sim), best_key
            return False, float(sim), best_key

        return False, float(sim), "No match in known groups"

def estimate_real_life_tests(record):
    """Estimates number of tests performed in real life, excluding physical examination."""
    lab_tests = ast.literal_eval(record['Laboratory Tests']) if pd.notna(record['Laboratory Tests']) else {}
    radiology = record['Radiology'] if pd.notna(record['Radiology']) else ""
    microbiology = record['Microbiology'] if pd.notna(record['Microbiology']) else ""

    num_lab_tests = len(lab_tests)
    num_radiology = 1 if radiology.strip() and "no" not in radiology.lower() else 0
    num_micro = 1 if microbiology.strip() and "no" not in microbiology.lower() else 0
    
    return num_lab_tests + num_radiology + num_micro

def get_real_life_test_names(record):
    """
    Extracts specific NAMES of tests performed in real life.
    This is required for the 'calculate_cost_metrics' function to lookup prices.
    """
    test_names = []

    # 1. Laboratory Tests
    if pd.notna(record.get('Laboratory Tests')):
        try:
            labs = ast.literal_eval(record['Laboratory Tests'])
            if isinstance(labs, dict):
                test_names.extend(labs.keys())
        except:
            pass

    # 2. Radiology
    rad = record.get('Radiology')
    if pd.notna(rad) and isinstance(rad, str):
        clean_rad = rad.strip()
        # Ensure we don't process empty data or "[]"
        if clean_rad and "no" not in clean_rad.lower() and clean_rad != "[]":
            parsed_successfully = False
            
            # Case A: Try to parse as a structured list
            if clean_rad.startswith('['):
                try:
                    rad_list = ast.literal_eval(clean_rad)
                    if isinstance(rad_list, list):
                        parsed_successfully = True # Mark as parsed so we don't fuzzy match later
                        for r in rad_list:
                            if isinstance(r, dict) and 'Exam Name' in r:
                                test_names.append(r['Exam Name'])
                except: 
                    pass
            
            # Case B: Unstructured String (e.g. "CT Scan performed")
            # Only run this if it WASN'T a list (to avoid matching "[]" or "[{}]")
            if not parsed_successfully:
                # Check if there is actual alpha-numeric content
                if re.search(r'[a-zA-Z0-9]', clean_rad):
                    match = process.extractOne(clean_rad, ALLOWED_RADIOLOGY_TESTS, scorer=fuzz.partial_ratio)
                    if match and match[1] > 60:
                        test_names.append(match[0]) 
                    else:
                        test_names.append("Unspecified Radiology")

    # 3. Microbiology
    micro = record.get('Microbiology')
    if pd.notna(micro) and isinstance(micro, str):
        if micro.strip() and "no" not in micro.lower():
            test_names.append("Microbiology") 

    return test_names

def calculate_cost_metrics(llm_test_list, real_test_list, test_costs, is_correct):
    """
    Calculates Cost, CES, UTR, and Frugality Index.
    """
    # 1. Costs
    llm_cost = sum([test_costs.get(t, 50.0) for t in llm_test_list])
    
    real_cost = 0.0
    for t in real_test_list:
        cost = test_costs.get(t)
        if cost is None:
            # Fallback logic
            if "Radiology" in t: cost = 100.0
            elif "Microbiology" in t: cost = 50.0
            else: cost = 50.0
        real_cost += cost

    # 2. CES
    denominator = real_cost if real_cost > 0 else 1.0
    ces = llm_cost / denominator
    
    # 3. UTR
    llm_set = set(llm_test_list)
    real_set = set(real_test_list)
    intersection = llm_set.intersection(real_set)
    unnecessary_count = len(llm_set) - len(intersection)
    utr = unnecessary_count / len(llm_set) if len(llm_set) > 0 else 0.0
    
    # 4. FI
    acc_val = 1.0 if is_correct else 0.0
    fi = acc_val / ces if ces > 0 else 0.0
    
    return {
        "llm_cost": llm_cost,
        "real_cost": real_cost,
        "CES": ces,
        "UTR": utr,
        "FI": fi
    }

def generate_patient_scenario(record):
    """Creates a patient scenario string from a patient record."""
    age = record['Age'] if pd.notna(record['Age']) else "unknown age"
    gender = record['Gender'].lower() if pd.notna(record['Gender']) else "unknown gender"
    gender = 'female' if gender == 'f' else 'male' if gender == 'm' else 'unknown gender'
    
    history = record['Patient History']
    exam_findings = record['Physical Examination'] if pd.notna(record['Physical Examination']) else "No physical exam findings."
    return f"A {age}-year-old {gender} presents with {history}. Physical exam: {exam_findings}"

def get_test_results(record, suggested_tests, label_to_code, available_tests, config):
    """
    Retrieves actual test results from the dataset for the tests suggested by the LLM.
    """
    ALLOWED_RADIOLOGY_TESTS = config.ALLOWED_RADIOLOGY_TESTS
    TEST_MAPPING = config.TEST_MAPPING

    lab_results_str = record.get('Laboratory Tests')
    lab_results = {}
    if pd.notna(lab_results_str) and lab_results_str != '{}':
        try:
            lab_results = ast.literal_eval(lab_results_str)
            if not isinstance(lab_results, dict): lab_results = {}
        except (SyntaxError, ValueError):
            lab_results = {}

    ref_lower_str = record.get('Reference Range Lower')
    ref_lower = {}
    if pd.notna(ref_lower_str) and ref_lower_str != '{}':
        try:
            ref_lower = ast.literal_eval(ref_lower_str)
            if not isinstance(ref_lower, dict): ref_lower = {}
        except (SyntaxError, ValueError):
            ref_lower = {}

    ref_upper_str = record.get('Reference Range Upper')
    ref_upper = {}
    if pd.notna(ref_upper_str) and ref_upper_str != '{}':
        try:
            ref_upper = ast.literal_eval(ref_upper_str)
            if not isinstance(ref_upper, dict): ref_upper = {}
        except (SyntaxError, ValueError):
            ref_upper = {}

    microbiology = record.get('Microbiology', "No microbiology data.") if pd.notna(record.get('Microbiology')) else "No microbiology data."
    physical_exam = record.get('Physical Examination', "No physical exam data.") if pd.notna(record.get('Physical Examination')) else "No physical exam data."
    
    radiology_reports = []
    radiology_data = record.get('Radiology')
    if pd.notna(radiology_data):
        try:
            if isinstance(radiology_data, str) and radiology_data.strip().startswith('['):
                radiology_reports = ast.literal_eval(radiology_data)
            else:
                radiology_reports = [{"Exam Name": "Unknown", "Report": radiology_data}]
            if not isinstance(radiology_reports, list):
                radiology_reports = []
        except (SyntaxError, ValueError):
            radiology_reports = []

    results = {}
    for test in suggested_tests:
        if test in TEST_MAPPING:
            source = TEST_MAPPING[test]
            if source == 'Microbiology':
                results[test] = microbiology
            elif source == 'Physical Examination':
                results[test] = physical_exam
            else:
                results[test] = "Test not available in dataset."
        else:
            if test in label_to_code:
                code = label_to_code[test]
                if code in lab_results:
                    value = lab_results[code]
                    try:
                        num_value = float(re.sub(r'[^\d.]', '', str(value)))
                        lower = ref_lower.get(code, float('-inf'))
                        upper = ref_upper.get(code, float('inf'))
                        if lower is not None and upper is not None:
                            status = "Abnormal" if not (lower <= num_value <= upper) else "Normal"
                            results[test] = f"{value} ({status})"
                        else:
                            results[test] = str(value)
                    except (ValueError, TypeError):
                        results[test] = str(value)
                else:
                    results[test] = "Test result not found."
            else:
                rad_match = process.extractOne(test, ALLOWED_RADIOLOGY_TESTS, scorer=fuzz.partial_ratio)
                if rad_match and rad_match[1] > 50:
                    matched_test = rad_match[0]
                    if radiology_reports:
                        exam_names = [report.get('Exam Name', '') for report in radiology_reports if 'Exam Name' in report]
                        if exam_names:
                            exam_match = process.extractOne(matched_test, exam_names, scorer=fuzz.partial_ratio)
                            if exam_match and exam_match[1] > 80:
                                matched_exam = exam_match[0]
                                for report in radiology_reports:
                                    if report.get('Exam Name') == matched_exam:
                                        results[test] = report.get('Report', 'No report available.')
                                        break
                            else:
                                for report in radiology_reports:
                                    report_text = report.get('Report', '')
                                    if any(keyword.lower() in report_text.lower() for keyword in matched_test.split()):
                                        results[test] = report_text
                                        break
                                else:
                                    results[test] = f"Radiology test '{matched_test}' suggested but no matching report found."
                        else:
                            if radiology_reports and len(radiology_reports) > 0 and 'Report' in radiology_reports[0]:
                                results[test] = radiology_reports[0]['Report']
                            else:
                                results[test] = "No radiology exam names available."
                    else:
                        results[test] = "No radiology data available."
                else:
                    lab_match = process.extractOne(test, available_tests, scorer=fuzz.partial_ratio)
                    if lab_match and lab_match[1] > 80:
                        matched_label = lab_match[0]
                        code = label_to_code[matched_label]
                        if code in lab_results:
                            value = lab_results[code]
                            try:
                                num_value = float(re.sub(r'[^\d.]', '', str(value)))
                                lower = ref_lower.get(code, float('-inf'))
                                upper = ref_upper.get(code, float('inf'))
                                if lower is not None and upper is not None:
                                    status = "Abnormal" if not (lower <= num_value <= upper) else "Normal"
                                    results[test] = f"{value} ({status}) (matched to {matched_label})"
                                else:
                                    results[test] = f"{value} (matched to {matched_label})"
                            except (ValueError, TypeError):
                                results[test] = f"{value} (matched to {matched_label})"
                        else:
                            results[test] = f"Test result not found (closest match: {matched_label})"
                    else:
                        results[test] = "Test not available in dataset."
    return results