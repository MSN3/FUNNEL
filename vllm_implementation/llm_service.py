# llm_service.py

# import json
# import re
# from vllm import LLM, SamplingParams
# from config import MODEL_NAME

# # 1. Initialize the LLM model once when the script starts.
# # This loads the model into GPU memory.
# print(f"Loading model: {MODEL_NAME}...")
# llm = LLM(model=MODEL_NAME)
# print("Model loaded successfully.")

# def query_vllm(prompt, temperature=0.1, max_tokens=1024):
#     """Helper function to query the loaded vLLM model."""
#     sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
#     # llm.generate expects a list of prompts
#     outputs = llm.generate([prompt], sampling_params)
#     # Return the text from the first (and only) output
#     return outputs[0].outputs[0].text

# def get_differential_diagnoses(scenario):
#     """Queries the loaded vLLM model for differential diagnoses."""
#     prompt = (
#         f"Based on the following patient scenario, provide a list of possible differential diagnoses. "
#         f"You must respond with only a valid JSON object containing a single key 'diagnoses' with a list of strings as the value. "
#         f"Example: {{\"diagnoses\": [\"Diagnosis 1\", \"Diagnosis 2\"]}}. "
#         f"Patient Scenario: {scenario}\n"
#     )
    
#     response_text = query_vllm(prompt, temperature=0.1, max_tokens=4096)
#     try:
#         # Clean up potential markdown formatting that models sometimes add
#         cleaned_text = re.sub(r"```json\n?|```", "", response_text).strip()
#         response_json = json.loads(cleaned_text)
#         return response_json.get("diagnoses", [])
#     except Exception as e:
#         print(f"Warning: JSON error in differential diagnosis: {e}. Falling back to text parsing.")
#         matches = re.findall(r'\"([^\"]+)\"', response_text)
#         return [m for m in matches if m.lower() not in ["diagnoses", "tests"]]

# def get_test_suggestions(scenario, diagnoses_list, available_tests, allowed_radiology_tests, test_results_str=""):
#     """Queries the loaded vLLM model for test suggestions."""
#     prompt = (
#         f"Based on the patient scenario and differential diagnoses, suggest 2–6 unique, cost-effective tests. "
#         f"You must respond with only a valid JSON object containing a single key 'tests' with a list of strings as the value. "
#         f"Example: {{\"tests\": [\"Test 1\", \"Test 2\"]}}. "
#         f"Patient Scenario: {scenario}\n"
#         f"Differential Diagnoses: {', '.join(diagnoses_list)}\n"
#         f"Previous test results:\n{test_results_str}\n"
#         f"For lab and microbiology tests, choose ONLY from this list: {', '.join(available_tests)}. "
#         f"For radiology tests, choose ONLY from this list: {', '.join(allowed_radiology_tests)}. "
#         f"Prioritize high-yield tests that will most efficiently narrow down the diagnosis.\n"
#     )

#     response_text = query_vllm(prompt, temperature=0.1, max_tokens=4096)
#     try:
#         cleaned_text = re.sub(r"```json\n?|```", "", response_text).strip()
#         response_json = json.loads(cleaned_text)
#         return response_json.get("tests", [])
#     except Exception as e:
#         print(f"Warning: JSON error in test suggestions: {e}. Falling back to text parsing.")
#         matches = re.findall(r'\"([^\"]+)\"', response_text)
#         return [m for m in matches if m.lower() not in ["diagnoses", "tests"]]

# def get_final_decision(scenario, diagnoses_list, test_results_str):
#     """Queries the loaded vLLM model for a final diagnosis or more tests."""
#     prompt = (
#         f"Based on the scenario, diagnoses, and test results, provide a final diagnosis or suggest more tests.\n"
#         f"Scenario: {scenario}\nDiagnoses: {', '.join(diagnoses_list)}\nTest Results:\n{test_results_str}\n"
#         f"Your response MUST start with either 'Final Diagnosis:' or 'Additional Tests:'. "
#         f"Provide only one of these, not both."
#     )
    
#     # Use slightly higher temperature for more natural text generation
#     return query_vllm(prompt, temperature=0.1, max_tokens=512)

from openai import AsyncOpenAI # Use the async client
import json
import re
from config import MODEL_NAME

# Configure the async client to connect to the vLLM server
client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="vllm"
)

async def get_differential_diagnoses(scenario):
    """Asynchronously queries vLLM for differential diagnoses."""
    prompt = (
        f"Based on the following patient scenario, provide a list of possible differential diagnoses. "
        f"You must respond with only a valid JSON object containing a single key 'diagnoses' with a list of strings as the value. "
        f"Example: {{\"diagnoses\": [\"Diagnosis 1\", \"Diagnosis 2\"]}}. "
        f"Patient Scenario: {scenario}\n"
    )
    
    try:
        response = await client.chat.completions.create( # Use 'await'
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=4096,
        )
        response_text = response.choices[0].message.content
        cleaned_text = re.sub(r"```json\n?|```", "", response_text).strip()
        response_json = json.loads(cleaned_text)
        return response_json.get("diagnoses", [])
    except Exception as e:
        print(f"Warning: JSON error in differential diagnosis: {e}. Falling back to text parsing.")
        if not response_text:
            response_text = str(e)
        matches = re.findall(r'\"([^\"]+)\"', response_text)
        return [m for m in matches if m.lower() not in ["diagnoses", "tests"]]

async def get_test_suggestions(scenario, diagnoses_list, available_tests, allowed_radiology_tests, test_results_str=""):
    """Asynchronously queries vLLM for test suggestions."""
    prompt = (
        f"Based on the patient scenario and differential diagnoses, suggest 2–6 unique, cost-effective tests. "
        f"You must respond with only a valid JSON object containing a single key 'tests' with a list of strings as the value. "
        f"Example: {{\"tests\": [\"Test 1\", \"Test 2\"]}}. "
        f"Patient Scenario: {scenario}\n"
        f"Differential Diagnoses: {', '.join(diagnoses_list)}\n"
        f"Previous test results:\n{test_results_str}\n"
        f"For lab and microbiology tests, choose ONLY from this list: {', '.join(available_tests)}. "
        f"For radiology tests, choose ONLY from this list: {', '.join(allowed_radiology_tests)}. "
        f"Prioritize high-yield tests that will most efficiently narrow down the diagnosis.\n"
    )

    try:
        response = await client.chat.completions.create( # Use 'await'
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=4096,
        )
        response_text = response.choices[0].message.content
        cleaned_text = re.sub(r"```json\n?|```", "", response_text).strip()
        response_json = json.loads(cleaned_text)
        return response_json.get("tests", [])
    except Exception as e:
        print(f"Warning: JSON error in test suggestions: {e}. Falling back to text parsing.")
        if not response_text:
            response_text = str(e)
        matches = re.findall(r'\"([^\"]+)\"', response_text)
        return [m for m in matches if m.lower() not in ["diagnoses", "tests"]]

async def get_final_decision(scenario, diagnoses_list, test_results_str):
    """Asynchronously queries vLLM for a final decision."""
    prompt = (
        f"Based on the scenario, diagnoses, and test results, provide a final diagnosis or suggest more tests.\n"
        f"Scenario: {scenario}\nDiagnoses: {', '.join(diagnoses_list)}\nTest Results:\n{test_results_str}\n"
        f"Your response MUST start with either 'Final Diagnosis:' or 'Additional Tests:'. "
        f"Provide only one of these, not both."
    )
    
    response = await client.chat.completions.create( # Use 'await'
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()