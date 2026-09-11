# utils.py
import pandas as pd
import numpy as np
import re
import ast
import torch
import os
import json
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from typing import List, Dict, Any, Tuple
from llm_client import UniversalLLMClient
from tool_wrapper import SearchTool

def parse_json_list_from_key(response_text: str, key: str) -> List[str]:
    """
    Extracts a JSON object from a messy string, then gets a list from a key.
    """
    try:
        start_index = response_text.find('{')
        end_index = response_text.rfind('}')
        if start_index == -1 or end_index == -1 or end_index < start_index:
            raise ValueError("No valid JSON object found in response.")
        json_str = response_text[start_index:end_index+1]
        response_json = json.loads(json_str)
        result = response_json.get(key, [])
        return result if isinstance(result, list) else ([str(result)] if result else [])
    except Exception as e:
        print(f"Warning: JSON list parsing failed for key '{key}': {e}. Falling back to regex.")
        matches = re.findall(r'\"([^\"]+)\'', response_text)
        return [m for m in matches if m.lower() not in [key, "diagnoses", "tests"]]

def parse_json_dict(response_text: str) -> Dict[str, Any]:
    """
    Extracts a JSON object (a dictionary) from a messy string.
    """
    try:
        start_index = response_text.find('{')
        end_index = response_text.rfind('}')
        if start_index == -1 or end_index == -1 or end_index < start_index:
            raise ValueError("No valid JSON object found in response.")
        json_str = response_text[start_index:end_index+1]
        response_json = json.loads(json_str)
        return response_json if isinstance(response_json, dict) else {}
    except Exception as e:
        print(f"Warning: JSON dict parsing failed: {e}")
        return {}
# --- END PARSERS ---

def _parse_test_container(value):
    """Safely parse dictionary/list/string-encoded test containers into a list of test names."""
    if pd.isna(value):
        return []
    if isinstance(value, dict):
        return [str(k) for k in value.keys()]
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            if isinstance(item, dict) and 'Exam Name' in item:
                out.append(str(item['Exam Name']))
            elif isinstance(item, dict):
                out.extend([str(v) for v in item.values() if isinstance(v, str)])
            else:
                out.append(str(item))
        return out
    text = str(value).strip()
    if not text or text.lower() in {'nan', 'none'} or text in {'{}', '[]'}:
        return []
    try:
        parsed = ast.literal_eval(text)
        return _parse_test_container(parsed)
    except Exception:
        if '|' in text or ';' in text:
            return [p for p in re.split(r'\s*[|;]\s*', text) if p]
        return [text]


def is_radiology_test(test_name: str, allowed_radiology_tests: List[str] = None) -> bool:
    """Classify whether a test name is a radiology/imaging study."""
    if test_name is None or pd.isna(test_name):
        return False
    t = str(test_name).strip().lower()
    if not t:
        return False
    allowed = [x.lower() for x in (allowed_radiology_tests or [])]
    if t in allowed:
        return True
    keywords = [
        'ct ', 'cta', 'mri', 'mr ', 'mra', 'x-ray', 'xray', 'radiograph',
        'ultrasound', ' us', 'echo', 'chest (', 'abdomen', 'pelvis', 'head',
        'brain', 'spine', 'portable ap', 'pa & lat', 'mrcp', 'ercp', 'angiography',
        'duplex', 'venous', 'arterial'
    ]
    return any(k in f" {t} " for k in keywords)

def estimate_real_life_tests(record):
    """Estimates number of tests performed in real life, excluding physical examination."""
    lab_tests_str = record.get('Laboratory Tests')
    lab_tests = {}
    if pd.notna(lab_tests_str) and lab_tests_str != '{}':
        try:
            lab_tests = ast.literal_eval(lab_tests_str)
            if not isinstance(lab_tests, dict): lab_tests = {}
        except (SyntaxError, ValueError):
            lab_tests = {} # Handle malformed string
            
    radiology = record.get('Radiology', "") if pd.notna(record.get('Radiology')) else ""
    microbiology = record.get('Microbiology', "") if pd.notna(record.get('Microbiology')) else ""
    num_lab_tests = len(lab_tests)
    num_radiology = 1 if radiology.strip() and "no" not in radiology.lower() else 0
    num_micro = 1 if microbiology.strip() and "no" not in microbiology.lower() else 0
    return num_lab_tests + num_radiology + num_micro

def get_real_life_test_names(record, allowed_radiology_tests: List[str] = None):
    """
    Extract names of real-life laboratory, radiology and microbiology tests for cost and radiology-audit calculations.

    Parameters
    ----------
    record : dict-like
        Patient record with Laboratory Tests, Radiology and Microbiology fields.
    allowed_radiology_tests : list[str], optional
        Canonical radiology names from config.ALLOWED_RADIOLOGY_TESTS. If provided, unstructured
        radiology text is fuzzy matched to this list.
    """
    tests = []

    # 1. Laboratory Tests
    tests.extend(_parse_test_container(record.get('Laboratory Tests')))

    # 2. Radiology Tests
    rad = record.get('Radiology')
    if pd.notna(rad):
        clean_rad = str(rad).strip()
        if clean_rad and clean_rad.lower() not in {'no', 'none', 'nan'} and clean_rad != '[]':
            parsed = _parse_test_container(clean_rad)
            if parsed:
                for r in parsed:
                    if allowed_radiology_tests:
                        match = process.extractOne(str(r), allowed_radiology_tests, scorer=fuzz.partial_ratio)
                        tests.append(match[0] if match and match[1] > 60 else str(r))
                    else:
                        tests.append(str(r))
            elif allowed_radiology_tests:
                match = process.extractOne(clean_rad, allowed_radiology_tests, scorer=fuzz.partial_ratio)
                tests.append(match[0] if match and match[1] > 60 else 'Unspecified Radiology')
            else:
                tests.append('Unspecified Radiology')

    # 3. Microbiology
    micro = record.get('Microbiology')
    if pd.notna(micro) and isinstance(micro, str):
        if micro.strip() and 'no' not in micro.lower():
            tests.append('Microbiology')

    return [t for t in tests if str(t).strip()]

def generate_patient_scenario(record):
    """Creates a patient scenario string from a patient record."""
    # ... (this function is unchanged) ...
    age = record.get('Age', "unknown age") if pd.notna(record.get('Age')) else "unknown age"
    gender = record.get('Gender', "unknown gender").lower() if pd.notna(record.get('Gender')) else "unknown gender"
    gender = 'female' if gender == 'f' else 'male' if gender == 'm' else 'unknown gender'
    history = record.get('Patient History', "No history provided.") if pd.notna(record.get('Patient History')) else "No history provided."
    exam_findings = record.get('Physical Examination', "No physical exam findings.") if pd.notna(record.get('Physical Examination')) else "No physical exam findings."
    return f"A {age}-year-old {gender} presents with {history}. Physical exam: {exam_findings}"

def get_test_results(record, suggested_tests, label_to_code, available_tests, config):
    """Retrieves actual test results from the dataset for the tests suggested by the LLM."""
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

class LLMJudgeEvaluator:
    """
    Evaluates diagnoses using an LLM-as-a-Judge, augmented with a search tool.
    """
    def __init__(self, judge_vllm_client: UniversalLLMClient, config: Any):
        print("Initializing LLM-as-a-Judge Evaluator...")
        self.judge_vllm_client = judge_vllm_client 
        self.search_tool = SearchTool(max_results=3)
        self.related_diagnoses = config.RELATED_DIAGNOSES
        self.known_diagnoses_keys = list(self.related_diagnoses.keys())
        print("LLM-as-a-Judge Evaluator initialized.")

    def fit(self, *args, **kwargs):
        pass

    async def _run_llm_judge(self, llm_pred: str, ground_truth_list: List[str], fallback_ground_truth: str) -> Tuple[bool, float, str, str, str, int]:
        """
        Runs the full RAG-based LLM-as-a-Judge evaluation.
        Compares one prediction against a list of ground truths.
        """
        tokens = 0
        # --- 1. MODIFIED SEARCH QUERY ---
        # Search for the relationship between the prediction and all GT
        search_query = f'medical relationship between "{llm_pred}" and the following diagnoses: {", ".join(ground_truth_list)}'
        search_context = self.search_tool.search(search_query)

        # 2. Build the Judge Prompt
        system_prompt = (
            "You are an expert medical evaluator. Your job is to classify the relationship between a single predicted diagnosis and a list of ground truth diagnoses. "
            "You MUST respond ONLY with a single valid JSON object with two keys: "
            "'score' (str, 'A', 'B', or 'C') and 'reasoning' (str)."
        )

        # --- 3. MODIFIED PROMPT WITH LIST-BASED RULES ---
        user_prompt = f"""
        Please evaluate if the 'LLM Diagnosis' is medically equivalent or broadly related to ANY of the diagnoses in the 'Ground Truth List'.

        LLM Diagnosis: "{llm_pred}"
        Ground Truth List: {ground_truth_list}

        Search Context:
        ---
        {search_context}
        ---

        SCORING RULES:
        - Score 'A' (Equivalent): The 'LLM Diagnosis' is an exact match, common synonym, or clear symptom of *ANY* diagnosis in the 'Ground Truth List'.
          (e.g., 'heart attack' vs ['myocardial infarction', 'hypertension'])
        - Score 'B' (Broadly Related): The 'LLM Diagnosis' is a *correctly related concept* (like a treatment, procedure, or direct complication) of *ANY* diagnosis in the list.
          (e.g., 'cholecystectomy' vs ['cholecystitis', 'diabetes'])
        - Score 'C' (Incorrect): The 'LLM Diagnosis' is not related to *ANY* diagnosis in the list.
          (e.g., 'pneumonia' vs ['cholecystitis', 'diabetes'])

        Example (Score A):
        {{"score": "A", "reasoning": "heart attack is a common synonym for myocardial infarction, which is in the list."}}

        Example (Score B):
        {{"score": "B", "reasoning": "Cholecystectomy is the surgical procedure for cholecystitis, which is in the list."}}
        
        Example (Score C):
        {{"score": "C", "reasoning": "Pneumonia is a lung infection and is not related to cholecystitis or diabetes."}}
        """

        # 4. Call LLM
        response_text, usage = await self.judge_vllm_client.invoke(system_prompt, user_prompt, temperature=0.0)
        tokens += usage['total']

        # 5. Parse Response (and use Python for logic)
        try:
            res_json = parse_json_dict(response_text)
            if not res_json:
                raise ValueError("Parsing returned an empty dictionary.")

            score = res_json.get("score", "C")
            reasoning = res_json.get("reasoning", "No reasoning provided.")
            
            is_equivalent = (score == "A" or score == "B")
            similarity_score = 1.0 if is_equivalent else 0.0
            
            effective_diagnosis = ""
            if is_equivalent:
                # If it's correct (A or B), we try to find what it matched
                # For simplicity, we'll just use the first ground truth
                effective_diagnosis = fallback_ground_truth
            else:
                # If it's incorrect (C), find best match for the *bad prediction*
                best_match, match_score = process.extractOne(llm_pred, self.known_diagnoses_keys)
                effective_diagnosis = best_match
            
            print(f"--- LLM Judge Result ---")
            print(f"  Pred: {llm_pred} | GT List: {ground_truth_list[:3]}... (Total: {len(ground_truth_list)})")
            print(f"  Verdict: {is_equivalent} (Score: {score}) | Reasoning: {reasoning}")
            
            return is_equivalent, similarity_score, effective_diagnosis, score, reasoning, tokens
            
        except Exception as e:
            print(f"Error parsing LLM Judge response: {e}. Defaulting to False. Response text: {response_text}")
            return False, 0.0, llm_pred, "C", "Parsing error in LLM Judge response", tokens

    async def _filter_icd_list(self, fallback_ground_truth: str, icd_list: List[str]) -> Tuple[List[str], int]:
        """
        Uses an LLM to filter a long ICD list down to only those diagnoses
        medically related to the primary ground truth.
        """
        system_prompt = (
            "You are a medical expert. Your task is to identify which ICD diagnoses from a list are medically related "
            "to a given reference diagnosis. "
            "Respond ONLY with a valid JSON object with one key: 'relevant_diagnoses'." # <-- Fix 2
        )

        user_prompt = f"""
        Ground Truth Diagnosis: "{fallback_ground_truth}"

        ICD Diagnosis Candidates:
        {icd_list}

        Return ONLY the ICD diagnoses from the list that are medically related to the Ground Truth Diagnosis.
        Include the Ground Truth Diagnosis if it's in the list.
        
        Respond using ONLY a JSON object with a single 'relevant_diagnoses' key.

        Example:
        {{"relevant_diagnoses": ["acute kidney failure, unspecified", "renal failure"]}}
        """

        response, usage = await self.judge_vllm_client.invoke(system_prompt, user_prompt, temperature=0.0)
        filtered = parse_json_list_from_key(response, "relevant_diagnoses") 
        return (filtered or [], usage['total'])

    async def evaluate_diagnosis(self, llm_pred: str, icd_diagnoses_str: str, fallback_ground_truth: str) -> Tuple[bool, float, str, str, str, int]:
        """
        Evaluates the diagnosis, using fast checks against the full ICD list first, 
        then the LLM judge.
        """
        total_judge_tokens = 0
        llm_pred_clean = str(llm_pred).lower().strip() if llm_pred else ""
        gt_dx = str(fallback_ground_truth).lower().strip() if fallback_ground_truth else ""
        
        if not llm_pred_clean or 'unable' in llm_pred_clean or "error:" in llm_pred_clean:
            return False, 0.0, "No valid diagnosis provided", "C", "Invalid LLM prediction", total_judge_tokens
        
        # 1. Parse the ICD list
        cleaned_list = []
        try:
            diagnosis_list = ast.literal_eval(icd_diagnoses_str)
            if not isinstance(diagnosis_list, list):
                raise ValueError("Not a list")
            # Clean the list: "5849 - Acute kidney failure" -> "acute kidney failure"
            cleaned_list = [re.sub(r'^\S+\s*-\s*', '', dx).lower().strip() for dx in diagnosis_list]
        except (SyntaxError, ValueError, TypeError):
            # Fallback to the single ground_truth if parsing fails
            cleaned_list = [str(fallback_ground_truth).lower().strip()]
        
        if not cleaned_list:
             cleaned_list = [str(fallback_ground_truth).lower().strip()]

        # Obtain relevant ICD diagnoses using LLM filtering
        relevant_icd_list, filter_tokens = await self._filter_icd_list(gt_dx, cleaned_list)
        total_judge_tokens += filter_tokens

        if not relevant_icd_list:
            relevant_icd_list = [gt_dx]  # Fallback

        print("Relevant ICD diagnoses selected by LLM:", relevant_icd_list)

        # 2. Fast-Path Checks (Loop through the full list)
        # for gt_dx in cleaned_list:
        if llm_pred_clean == gt_dx:
            return True, 1.0, gt_dx, "A", "Exact match", total_judge_tokens
        if fuzz.token_set_ratio(llm_pred_clean, gt_dx) >= 90:
            return True, 0.9, gt_dx, "B", "High token set similarity", total_judge_tokens
        for main_key, synonyms in self.related_diagnoses.items():
            all_terms = [main_key] + [term.lower() for term in synonyms]
            if gt_dx in all_terms and llm_pred_clean in all_terms:
                return True, 0.9, main_key, "B", "Related diagnosis", total_judge_tokens
        # 3. Slow-Path (LLM Judge)
        # Call the judge with the prediction and the *full list*
        return await self._run_llm_judge(llm_pred_clean, relevant_icd_list, gt_dx)