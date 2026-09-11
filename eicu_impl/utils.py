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

def estimate_real_life_tests(record):
    lab_tests_str = str(record.get('Laboratory Tests', ''))
    num_lab_tests = 0
    if lab_tests_str and lab_tests_str.lower() not in ['not available', 'nan']:
        num_lab_tests = len([x for x in lab_tests_str.split(';') if x.strip()])
            
    radiology = str(record.get('Radiology', ''))
    microbiology = str(record.get('Microbiology', ''))
    num_radiology = 0 if not radiology.strip() or radiology.lower() in ['not available', 'nan'] else 1
    num_micro = 0 if not microbiology.strip() or microbiology.lower() in ['not available', 'nan'] else 1
    return num_lab_tests + num_radiology + num_micro

def get_real_life_test_names(record):
    tests = []
    
    # 1. Laboratory Tests
    lab_tests_str = str(record.get('Laboratory Tests', ''))
    if lab_tests_str and lab_tests_str.lower() not in ['not available', 'nan']:
        for item in lab_tests_str.split(';'):
            if ':' in item:
                test_name = item.split(':')[0].strip()
                if test_name.startswith('-'): test_name = test_name[1:]
                tests.append(test_name)
                 
    # 2. Radiology Tests
    rad = str(record.get('Radiology', ''))
    if rad and rad.lower() not in ['not available', 'nan']:
        tests.append("Radiology Exam")

    # 3. Microbiology
    micro = str(record.get('Microbiology', ''))
    if micro and micro.lower() not in ['not available', 'nan']:
        tests.append("Microbiology") 
            
    return tests

def generate_patient_scenario(record):
    age = record.get('Age', "unknown age") if pd.notna(record.get('Age')) else "unknown age"
    gender = record.get('Gender', "unknown gender").lower() if pd.notna(record.get('Gender')) else "unknown gender"
    gender = 'female' if gender == 'f' else 'male' if gender == 'm' else 'unknown gender'
    history = record.get('Patient History', "No history provided.") if pd.notna(record.get('Patient History')) else "No history provided."
    exam_findings = record.get('Physical Examination', "No physical exam findings.") if pd.notna(record.get('Physical Examination')) else "No physical exam findings."
    return f"A {age}-year-old {gender} presents with {history}. Physical exam: {exam_findings}"

def get_test_results(record, suggested_tests, label_to_code, available_tests, config):
    ALLOWED_RADIOLOGY_TESTS = config.ALLOWED_RADIOLOGY_TESTS
    TEST_MAPPING = config.TEST_MAPPING
    
    lab_results_str = str(record.get('Laboratory Tests', ''))
    lab_results = {}
    if lab_results_str and lab_results_str.lower() not in ['not available', 'nan']:
        for item in lab_results_str.split(';'):
            if ':' in item:
                k, v = item.split(':', 1)
                clean_k = k.strip()
                if clean_k.startswith('-'): clean_k = clean_k[1:]
                lab_results[clean_k] = v.strip()

    microbiology = record.get('Microbiology', "No microbiology data.")
    if str(microbiology).lower() in ['not available', 'nan']: 
        microbiology = "No microbiology data."
    
    physical_exam = record.get('Physical Examination', "No physical exam data.")
    if str(physical_exam).lower() in ['not available', 'nan']:
        physical_exam = "No physical exam data."
    
    radiology_reports = []
    radiology_data = str(record.get('Radiology', ''))
    if radiology_data and radiology_data.lower() not in ['not available', 'nan']:
        radiology_reports = [{"Exam Name": "Unknown", "Report": radiology_data}]

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
            # Fuzzy match directly against the parsed lab results from the string
            matched = False
            for k, v in lab_results.items():
                if fuzz.partial_ratio(test.lower(), k.lower()) > 80:
                    results[test] = str(v)
                    matched = True
                    break
            
            if matched:
                continue
                
            # Fallback to Radiology search
            rad_match = process.extractOne(test, ALLOWED_RADIOLOGY_TESTS, scorer=fuzz.partial_ratio)
            if rad_match and rad_match[1] > 50:
                if radiology_reports:
                    results[test] = radiology_reports[0]['Report']
                else:
                    results[test] = "No radiology data available."
            else:
                results[test] = "Test result not found."
                
    return results

class LLMJudgeEvaluator:
    def __init__(self, judge_vllm_client: UniversalLLMClient, config: Any):
        print("Initializing LLM-as-a-Judge Evaluator...")
        self.judge_vllm_client = judge_vllm_client 
        self.search_tool = SearchTool(max_results=2)
        self.related_diagnoses = config.RELATED_DIAGNOSES
        self.known_diagnoses_keys = list(self.related_diagnoses.keys())
        print("LLM-as-a-Judge Evaluator initialized.")

    def fit(self, *args, **kwargs):
        pass

    async def _run_llm_judge(self, llm_pred: str, ground_truth_list: List[str], fallback_ground_truth: str) -> Tuple[bool, float, str, str, str, int]:
        tokens = 0
        search_query = f'medical relationship between "{llm_pred}" and the following diagnoses: {", ".join(ground_truth_list)}'
        search_context = self.search_tool.search(search_query)

        system_prompt = (
            "You are an expert critical care medical evaluator. Your job is to classify the relationship between a single predicted diagnosis and a list of ground truth diagnoses for an ICU patient. "
            "You MUST respond ONLY with a single valid JSON object with two keys: "
            "'score' (str, 'A', 'B', or 'C') and 'reasoning' (str)."
        )

        # --- UPDATED PROMPT WITH ICU-SPECIFIC EXAMPLES ---
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
        - Score 'B' (Broadly Related): The 'LLM Diagnosis' is a *correctly related concept* (like a treatment, procedure, or direct complication) of *ANY* diagnosis in the list.
        - Score 'C' (Incorrect): The 'LLM Diagnosis' is not related to *ANY* diagnosis in the list.

        Example (Score A):
        {{"score": "A", "reasoning": "Septic shock is a severe progression and direct equivalent to sepsis, which is in the list."}}

        Example (Score B):
        {{"score": "B", "reasoning": "Mechanical ventilation is the primary life-support procedure for acute respiratory failure, which is in the list."}}
        
        Example (Score C):
        {{"score": "C", "reasoning": "Acute kidney injury is a renal condition and is not medically equivalent to atrial fibrillation or stroke."}}
        """

        response_text, usage = await self.judge_vllm_client.invoke(system_prompt, user_prompt, temperature=0.0)
        tokens += usage['total']

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
                effective_diagnosis = fallback_ground_truth
            else:
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
        system_prompt = (
            "You are a critical care medical expert. Your task is to identify which diagnoses from a list are medically related "
            "to a given reference diagnosis in an ICU context. "
            "Respond ONLY with a valid JSON object with one key: 'relevant_diagnoses'."
        )

        user_prompt = f"""
        Ground Truth Diagnosis: "{fallback_ground_truth}"

        ICD Diagnosis Candidates:
        {icd_list}

        Return ONLY the ICD diagnoses from the list that are medically related to the Ground Truth Diagnosis.
        Include the Ground Truth Diagnosis if it's in the list.
        
        Respond using ONLY a JSON object with a single 'relevant_diagnoses' key.
        """

        response, usage = await self.judge_vllm_client.invoke(system_prompt, user_prompt, temperature=0.0)
        filtered = parse_json_list_from_key(response, "relevant_diagnoses") 
        return (filtered or [], usage['total'])

    async def evaluate_diagnosis(self, llm_pred: str, icd_diagnoses_str: str, fallback_ground_truth: str) -> Tuple[bool, float, str, str, str, int]:
        total_judge_tokens = 0
        llm_pred_clean = str(llm_pred).lower().strip() if llm_pred else ""
        
        if not llm_pred_clean or 'unable' in llm_pred_clean or "error:" in llm_pred_clean:
            return False, 0.0, "No valid diagnosis provided", "C", "Invalid LLM prediction", total_judge_tokens
        
        # 1. Parse the full eICU list (splitting by pipes and semicolons)
        cleaned_list = []
        try:
            normalized_str = str(icd_diagnoses_str).replace('|', ';')
            cleaned_list = [dx.lower().strip() for dx in normalized_str.split(';') if dx.strip()]
        except Exception:
            cleaned_list = [str(fallback_ground_truth).lower().strip()]
        
        if not cleaned_list:
             cleaned_list = [str(fallback_ground_truth).lower().strip()]

        print("Evaluating against full list of patient diagnoses:", cleaned_list)

        # 2. Fast-Path Checks against ALL patient diagnoses
        for gt_dx in cleaned_list:
            if llm_pred_clean == gt_dx:
                return True, 1.0, gt_dx, "A", f"Exact match with {gt_dx}", total_judge_tokens
            if fuzz.token_set_ratio(llm_pred_clean, gt_dx) >= 90:
                return True, 0.9, gt_dx, "B", f"High fuzzy match with {gt_dx}", total_judge_tokens
            
            # Check synonyms
            for main_key, synonyms in self.related_diagnoses.items():
                all_terms = [main_key] + [term.lower() for term in synonyms]
                if gt_dx in all_terms and llm_pred_clean in all_terms:
                    return True, 0.9, main_key, "B", f"Related diagnosis match with {gt_dx}", total_judge_tokens
                    
        # 3. Slow-Path (LLM Judge) - Pass the ENTIRE list
        # (You can completely delete the _filter_icd_list function from the class)
        return await self._run_llm_judge(llm_pred_clean, cleaned_list, fallback_ground_truth)