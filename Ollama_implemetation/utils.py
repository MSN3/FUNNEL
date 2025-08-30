# utils.py

import pandas as pd
import re
import ast
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from config import RELATED_DIAGNOSES, TEST_MAPPING, ALLOWED_RADIOLOGY_TESTS

def are_diagnoses_related(diagnosis1, diagnosis2):
    """Determines if two diagnoses are related based on keywords or medical knowledge."""
    if not diagnosis1 or not diagnosis2:
        return False
    diagnosis1 = diagnosis1.lower()
    diagnosis2 = diagnosis2.lower()

    if fuzz.partial_ratio(diagnosis1, diagnosis2) > 80:
        return True

    for key, related_terms in RELATED_DIAGNOSES.items():
        if key in diagnosis1 or any(term in diagnosis1 for term in related_terms):
            if key in diagnosis2 or any(term in diagnosis2 for term in related_terms):
                return True
    return False

def estimate_real_life_tests(record):
    """Estimates the number of tests performed in real life."""
    lab_tests = ast.literal_eval(record['Laboratory Tests']) if pd.notna(record['Laboratory Tests']) else {}
    radiology = record['Radiology'] if pd.notna(record['Radiology']) else ""
    microbiology = record['Microbiology'] if pd.notna(record['Microbiology']) else ""
    exam = record['Physical Examination'] if pd.notna(record['Physical Examination']) else ""

    num_lab_tests = len(lab_tests)
    num_radiology = 1 if radiology.strip() and "no" not in radiology.lower() else 0
    num_micro = 1 if microbiology.strip() and "no" not in microbiology.lower() else 0
    num_exam = 1 if exam.strip() and "no" not in exam.lower() else 0
    return num_lab_tests + num_radiology + num_micro + num_exam

def generate_patient_scenario(record):
    """Creates a patient scenario string from a patient record."""
    age = record['Age'] if pd.notna(record['Age']) else "unknown age"
    gender = record['Gender'].lower() if pd.notna(record['Gender']) else "unknown gender"
    gender = 'female' if gender == 'f' else 'male' if gender == 'm' else 'unknown gender'
    
    history = record['Patient History']
    exam_findings = record['Physical Examination'] if pd.notna(record['Physical Examination']) else "No physical exam findings."
    return f"A {age}-year-old {gender} presents with {history}. Physical exam: {exam_findings}"

def get_test_results(record, suggested_tests, label_to_code, available_tests):
    """Retrieves actual test results from the dataset for the tests suggested by the LLM."""
    lab_results_str = record['Laboratory Tests'] if pd.notna(record['Laboratory Tests']) else '{}'
    try:
        lab_results = ast.literal_eval(lab_results_str)
        if not isinstance(lab_results, dict):
            print(f"Warning: Laboratory Tests for patient {record.name} is not a dictionary. Using empty dict.")
            lab_results = {}
    except (SyntaxError, ValueError) as e:
        print(f"Warning: Could not parse Laboratory Tests for patient {record.name}: {e}. Using empty dict.")
        lab_results = {}

    ref_lower_str = record['Reference Range Lower'] if pd.notna(record['Reference Range Lower']) else '{}'
    try:
        ref_lower = ast.literal_eval(ref_lower_str)
        if not isinstance(ref_lower, dict):
            print(f"Warning: Reference Range Lower for patient {record.name} is not a dictionary. Using empty dict.")
            ref_lower = {}
    except (SyntaxError, ValueError) as e:
        print(f"Warning: Could not parse Reference Range Lower for patient {record.name}: {e}. Using empty dict.")
        ref_lower = {}

    ref_upper_str = record['Reference Range Upper'] if pd.notna(record['Reference Range Upper']) else '{}'
    try:
        ref_upper = ast.literal_eval(ref_upper_str)
        if not isinstance(ref_upper, dict):
            print(f"Warning: Reference Range Upper for patient {record.name} is not a dictionary. Using empty dict.")
            ref_upper = {}
    except (SyntaxError, ValueError) as e:
        print(f"Warning: Could not parse Reference Range Upper for patient {record.name}: {e}. Using empty dict.")
        ref_upper = {}

    microbiology = record['Microbiology'] if pd.notna(record['Microbiology']) else "No microbiology data."
    physical_exam = record['Physical Examination'] if pd.notna(record['Physical Examination']) else "No physical exam data."

    radiology_reports = []
    if pd.notna(record['Radiology']):
        try:
            radiology_data = record['Radiology']
            if isinstance(radiology_data, str) and radiology_data.strip().startswith('['):
                radiology_reports = ast.literal_eval(radiology_data)
            else:
                radiology_reports = [{"Exam Name": "Unknown", "Report": radiology_data}]
            if not isinstance(radiology_reports, list):
                radiology_reports = []
        except (SyntaxError, ValueError) as e:
            print(f"Warning: Could not parse Radiology for patient {record.name}: {e}. Using empty list.")
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
                            if radiology_reports and 'Report' in radiology_reports[0]:
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