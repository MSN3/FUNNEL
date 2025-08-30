import ollama
import json
import re
from config import MODEL_NAME

def get_differential_diagnoses(scenario):
    """Queries a local LLM via Ollama for differential diagnoses."""
    prompt = (
        f"Based on the following patient scenario, provide a list of possible differential diagnoses. "
        f"Format your response as a valid JSON array with the key 'diagnoses'. "
        f"Example format: {{\"diagnoses\": [\"Diagnosis 1\", \"Diagnosis 2\"]}}. "
        f"Patient Scenario: {scenario}\n"
    )

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}],
            format='json'
        )
        response_text = response['message']['content']
        response_json = json.loads(response_text)
        return response_json.get("diagnoses", [])
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: JSON error in differential diagnosis: {e}. Falling back to text parsing.")
        # Fallback for models that struggle with strict JSON
        matches = re.findall(r'\"([^\"]+)\"', response_text)
        return [m for m in matches if m.lower() != "diagnoses"]

def get_test_suggestions(scenario, diagnoses_list, available_tests, allowed_radiology_tests, test_results_str=""):
    """Queries a local LLM for test suggestions."""
    prompt = (
        f"Based on the patient scenario and differential diagnoses, suggest 2–6 unique, cost-effective tests. "
        f"Format your response as a valid JSON array with the key 'tests'. "
        f"Example format: {{\"tests\": [\"Test 1\", \"Test 2\"]}}. "
        f"Patient Scenario: {scenario}\n"
        f"Differential Diagnoses: {', '.join(diagnoses_list)}\n"
        f"Previous test results:\n{test_results_str}\n"
        f"For lab and microbiology tests, choose ONLY from this list: {', '.join(available_tests)}. "
        f"For radiology tests, choose ONLY from this list: {', '.join(allowed_radiology_tests)}. "
        f"Prioritize high-yield tests that will most efficiently narrow down the diagnosis.\n"
    )

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}],
            format='json'
        )
        response_text = response['message']['content']
        response_json = json.loads(response_text)
        return response_json.get("tests", [])
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: JSON error in test suggestions: {e}. Falling back to text parsing.")
        matches = re.findall(r'\"([^\"]+)\"', response_text)
        return [m for m in matches if m.lower() != "tests"]

def get_final_decision(scenario, diagnoses_list, test_results_str):
    """Queries a local LLM for a final diagnosis or more tests."""
    prompt = (
        f"Based on the scenario, diagnoses, and test results, provide a final diagnosis or suggest more tests.\n"
        f"Scenario: {scenario}\nDiagnoses: {', '.join(diagnoses_list)}\nTest Results:\n{test_results_str}\n"
        f"Your response MUST start with either 'Final Diagnosis:' or 'Additional Tests:'. "
        f"Provide only one of these, not both."
    )
    
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response['message']['content'].strip()