from openai import AsyncOpenAI # Use the async client
import json
import re
import ast
# from config import MODEL_NAME, 
from config import (
    MODEL_NAME, JUDGE_MODEL_NAME, RELATED_DIAGNOSES,
    BASE_URL, JUDGE_BASE_URL
)
from tool_wrapper import SearchTool
from fuzzywuzzy import fuzz

# --- NEW: Create two separate clients ---
# Client for "worker" tasks (diagnosis, test suggestion)
worker_client = AsyncOpenAI(
    base_url=BASE_URL, # <-- Uses worker URL
    api_key="vllm"
)

# Client for "judge" tasks (evaluation)
judge_client = AsyncOpenAI(
    base_url=JUDGE_BASE_URL, # <-- Uses judge URL
    api_key="vllm"
)
# --- END NEW CLIENTS ---

search_tool = SearchTool(max_results=3)
        
async def get_differential_diagnoses(scenario):
    """Asynchronously queries vLLM for differential diagnoses."""
    prompt = (
        f"You are a medical expert focused on diagnostic accuracy and resource frugality.\n"
        f"Given the patient scenario below, list only the most probable differential diagnoses — "
        f"no more than 5 items — that would most efficiently guide testing decisions.\n\n"
        f"Respond ONLY as a valid JSON object with a single key 'diagnoses' containing a list of strings.\n"
        f"Do not include explanations, reasoning, or extra text.\n"
        f"Example: {{\"diagnoses\": [\"Diagnosis 1\", \"Diagnosis 2\"]}}\n\n"
        f"Patient Scenario: {scenario}\n /no_think"
    )
    
    try:
        response = await worker_client.chat.completions.create( # Use 'await'
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
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

    is_first_iteration = not test_results_str.strip()

    prompt = (
        f"/no_think\n"
        f"You are a cost-conscious medical diagnostician. Based on the patient scenario and differential diagnoses, "
        f"suggest ONLY the most essential tests that will efficiently confirm or exclude key possibilities.\n\n"
        f"Respond ONLY as a valid JSON object with a single key 'tests' containing a list of strings.\n"
        f"Do not include explanations, reasoning, or extra text.\n"
        f"Example: {{\"tests\": [\"Test 1\", \"Test 2\"]}}\n\n"
        f"Patient Scenario: {scenario}\n"
        f"Differential Diagnoses: {', '.join(diagnoses_list)}\n"
        f"Previous test results:\n{test_results_str if not is_first_iteration else 'None'}\n\n"
        f"For lab and microbiology tests, choose ONLY from this list: {', '.join(available_tests)}.\n"
        f"For radiology tests, choose ONLY from this list: {', '.join(allowed_radiology_tests)}.\n\n"
        # f"CRITICAL RULES:\n"
        # f"- Choose tests that have the highest information gain and lowest cost.\n"
        # f"- Do NOT order overlapping or confirmatory tests unless necessary.\n"
        # f"- Prefer one high-yield test over multiple redundant ones.\n"
        # f"- Stop suggesting once diagnostic uncertainty is acceptably low.\n /no_think\n"
    )

    # NEW: Add a specific rule for the first iteration
    if is_first_iteration:
        prompt += f"- This is the first request. You MUST suggest initial, high-yield tests to start the diagnosis.\n /no_think"
    else:
        prompt += f"- Based on previous results, suggest 1-3 *new* tests to narrow the differential diagnosis.\n /no_think"
        
    prompt += f"- Choose tests that have the highest information gain.\n- Do NOT order overlapping tests.\n /no_think\n"

    try:
        response = await worker_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )
        response_text = response.choices[0].message.content
        cleaned_text = re.sub(r"```json\n?|```", "", response_text).strip()
        response_json = json.loads(cleaned_text)
        return response_json.get("tests", [])
    except Exception as e:
        print(f"Warning: JSON error in test suggestions: {e}. Falling back to text parsing.")
        if 'response_text' not in locals():
            response_text = str(e)
        matches = re.findall(r'\"([^\"]+)\'', response_text)
        return [m for m in matches if m.lower() not in ["diagnoses", "tests"]]

# async def get_final_decision(scenario, diagnoses_list, test_results_str):
#     """Asynchronously queries vLLM for a final decision."""
#     prompt = (
#         f"Based on the scenario, diagnoses, and test results, provide a final diagnosis or suggest more tests.\n"
#         f"Do not provide reasoning, explanations, or extra text. "
#         f"Scenario: {scenario}\nDiagnoses: {', '.join(diagnoses_list)}\nTest Results:\n{test_results_str}\n"
#         f"Your response MUST start with either 'Final Diagnosis:' or 'Additional Tests:'. "
#         f"Provide only one of these, not both. /no_think"
#     )
    
#     response = await client.chat.completions.create( # Use 'await'
#         model=MODEL_NAME,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.0,
#         max_tokens=256,
#     )
#     # Return the response directly, ensuring it's not None
#     if response.choices[0].message.content is None:
#         return "<no response>"
#     return response.choices[0].message.content.strip()

async def get_final_decision(scenario, diagnoses_list, test_results_str):
    """Enhanced prompting for structured diagnosis extraction."""
    
    # NEW: Check if test results are empty and create a clear string
    test_results_display = test_results_str.strip() if test_results_str.strip() else "No test results yet."
    
    prompt = (
        f"/no_think\n"
        f"Based on the scenario, diagnoses, and test results, provide a final decision.\n"
        f"Scenario: {scenario}\nDiagnoses: {', '.join(diagnoses_list)}\nTest Results:\n{test_results_display}\n\n"
        
        # Enhanced format instructions
        f"CRITICAL: Your response must follow EXACTLY one of these formats:\n"
        f"Format 1: 'Final Diagnosis: [DISEASE NAME ONLY - maximum 3 words]'\n"
        f"Format 2: 'Additional Tests: [test1, test2, test3]'\n\n"
        
        f"--- NEW RULE ---\n"
        f"Rule: If 'Test Results' is 'No test results yet.', you *must* use Format 2 and request tests. "
        f"Do not attempt a final diagnosis.\n"
        f"--- END RULE ---\n\n"
        
        f"Examples of correct diagnosis format:\n"
        f"- Final Diagnosis: myocardial infarction\n"
        f"- Final Diagnosis: acute appendicitis\n\n"
        
        f"Do NOT include explanations, reasoning, or extra text."
    )
    
    response = await worker_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a medical expert. Follow the exact format requested. Be concise.\n /no_think"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    
    if response.choices[0].message.content is None:
        return "<no response>"
    return response.choices[0].message.content.strip()

# async def get_final_decision(scenario, diagnoses_list, test_results_str):
#     """Enhanced prompting for structured diagnosis extraction."""
#     prompt = (
#         f"Based on the scenario, diagnoses, and test results, provide a final decision.\n"
#         f"Scenario: {scenario}\nDiagnoses: {', '.join(diagnoses_list)}\nTest Results:\n{test_results_str}\n\n"
        
#         # Enhanced format instructions
#         f"CRITICAL: Your response must follow EXACTLY one of these formats:\n"
#         f"Format 1: 'Final Diagnosis: [DISEASE NAME ONLY - maximum 3 words]'\n"
#         f"Format 2: 'Additional Tests: [test1, test2, test3]'\n\n"
        
#         f"Examples of correct diagnosis format:\n"
#         f"- Final Diagnosis: myocardial infarction\n"
#         f"- Final Diagnosis: acute appendicitis\n"
#         f"- Final Diagnosis: renal failure\n\n"
        
#         f"Do NOT include explanations, reasoning, or extra text."# /no_think"
#     )
    
#     response = await worker_client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=[
#             {"role": "system", "content": "You are a medical expert. Follow the exact format requested. Be concise."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.0,  # Add this line
#         max_tokens=1024,    # Add this line
#     )
    
#     if response.choices[0].message.content is None:
#         return "<no response>"
#     return response.choices[0].message.content.strip()

# SCORING RULES:
#     - Score 'A' (Equivalent): Exact match, synonym, or clear symptom.
#     - Score 'B' (Broadly Related): Correctly related concept (e.g., treatment, procedure, common complication).
#     - Score 'C' (Incorrect): Unrelated.
#     Rules for 'effective_diagnosis':
#     - If Score is 'A' or 'B', 'effective_diagnosis' MUST be the 'Ground Truth Diagnosis'.
#     - If Score is 'C', 'effective_diagnosis' MUST be the canonical disease from 'Known Disease Categories' that the 'LLM Diagnosis' *best* matches. If no match, use the 'LLM Diagnosis' as-is.

def _parse_json_list_from_key(response_text: str, key: str):
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
    
def _parse_json_dict(response_text: str):
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

async def _filter_icd_list(fallback_ground_truth: str, icd_list: list):
    """
    Uses an LLM to filter a long ICD list down to only those diagnoses
    medically related to the primary ground truth.
    """
    system_prompt = (
        "You are a medical expert. Your task is to identify which ICD diagnoses from a list are medically related "
        "to a given reference diagnosis. "
        "Respond ONLY with a valid JSON object with one key: 'relevant_diagnoses'."
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

    response_text = await judge_client.chat.completions.create(
        model=JUDGE_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        max_tokens=1024 # Allow for potentially long lists
    )
    
    filtered = _parse_json_list_from_key(response_text.choices[0].message.content, "relevant_diagnoses")
    return filtered or []
    
async def _run_llm_judge(llm_pred: str, ground_truth_list: list, fallback_ground_truth: str):
    """
    Runs the full RAG-based LLM-as-a-Judge evaluation.
    Compares one prediction against a list of ground truths.
    """
    # 1. MODIFIED SEARCH QUERY
    search_query = f'medical relationship between "{llm_pred}" and the following diagnoses: {", ".join(ground_truth_list)}'
    search_context = search_tool.search(search_query)

    # 2. Build the Judge Prompt
    system_prompt = (
        "You are an expert medical evaluator. Your job is to classify the relationship between a single predicted diagnosis and a list of ground truth diagnoses. "
        "You MUST respond ONLY with a single valid JSON object with two keys: "
        "'score' (str, 'A', 'B', or 'C') and 'reasoning' (str)."
    )

    # 3. MODIFIED PROMPT WITH LIST-BASED RULES
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
    {{"score": "A", "reasoning": "heart attack is a common synonym for myocardial infarction, which is in the list."}}
    
    Example (Score B):
    {{"score": "B", "reasoning": "Cholecystectomy is the surgical procedure for cholecystitis, which is in the list."}}
    
    Example (Score C):
    {{"score": "C", "reasoning": "Pneumonia is a lung infection and is not related to cholecystitis or diabetes."}}
    """

    # 4. Call LLM
    response_text = ""
    try:
        response = await judge_client.chat.completions.create(
            model=JUDGE_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=512
        )
        response_text = response.choices[0].message.content

        # 5. Parse Response (and use Python for logic)
        res_json = _parse_json_dict(response_text)
        if not res_json:
            raise ValueError("Parsing returned an empty dictionary.")

        score = res_json.get("score", "C")
        reasoning = res_json.get("reasoning", "No reasoning provided.")
        
        is_equivalent = (score == "A" or score == "B")
        similarity_score = 1.0 if is_equivalent else 0.0
        
        effective_diagnosis = ""
        if is_equivalent:
            # If it's correct (A or B), use the primary ground truth
            effective_diagnosis = fallback_ground_truth
        else:
            # If it's incorrect (C), find best match for the *bad prediction*
            # This requires the list of keys from the config
            known_diagnoses_keys = list(RELATED_DIAGNOSES.keys())
            best_match, match_score = process.extractOne(llm_pred, known_diagnoses_keys)
            effective_diagnosis = best_match
        
        print(f"--- LLM Judge Result ---")
        print(f"  Pred: {llm_pred} | GT List: {ground_truth_list[:3]}... (Total: {len(ground_truth_list)})")
        print(f"  Verdict: {is_equivalent} (Score: {score}) | Reasoning: {reasoning}")
        
        return is_equivalent, similarity_score, effective_diagnosis, score, reasoning
        
    except Exception as e:
        print(f"Error parsing LLM Judge response: {e}. Defaulting to False. Response text: {response_text}")
        return False, 0.0, llm_pred, "C", "Parsing error in LLM Judge response"

async def evaluate_diagnosis_with_judge(llm_pred: str, icd_diagnoses_str: str, fallback_ground_truth: str):
    """
    Evaluates the diagnosis, using fast checks against the full ICD list first, 
    then the LLM judge.
    """
    llm_pred_clean = str(llm_pred).lower().strip() if llm_pred else ""
    gt_primary_clean = str(fallback_ground_truth).lower().strip() if fallback_ground_truth else ""
    
    if not llm_pred_clean or 'unable' in llm_pred_clean or "error:" in llm_pred_clean:
        return False, 0.0, "No valid diagnosis provided", "C", "Invalid LLM prediction"
    
    # 1. Parse the ICD list
    cleaned_list = set() # Use a set for efficiency
    try:
        diagnosis_list = ast.literal_eval(icd_diagnoses_str)
        if not isinstance(diagnosis_list, list):
            raise ValueError("Not a list")
        # Clean the list: "5849 - Acute kidney failure" -> "acute kidney failure"
        for dx in diagnosis_list:
            cleaned_list.add(re.sub(r'^\S+\s*-\s*', '', dx).lower().strip())
    except (SyntaxError, ValueError, TypeError):
        cleaned_list.add(gt_primary_clean)
    
    # Always add the primary ground truth
    # if gt_primary_clean:
    #     cleaned_list.add(gt_primary_clean)
    
    if not cleaned_list:
        cleaned_list.add(gt_primary_clean)

    # --- BUG FIX: Run fast-path checks against ALL cleaned diagnoses ---
    if llm_pred_clean == gt_primary_clean:
        return True, 1.0, gt_primary_clean, "A", "Exact match"
    if fuzz.token_set_ratio(llm_pred_clean, gt_primary_clean) >= 90:
        return True, 0.9, gt_primary_clean, "B", "High token set similarity"
    for main_key, synonyms in RELATED_DIAGNOSES.items():
        all_terms = [main_key] + [term.lower() for term in synonyms]
        if gt_primary_clean in all_terms and llm_pred_clean in all_terms:
            return True, 0.9, main_key, "B", "Related diagnosis"

    # 3. Slow-Path (LLM Judge)
    # Filter the list *before* sending it to the judge
    # Convert set back to list for filtering
    list_to_filter = list(cleaned_list)
    relevant_icd_list = await _filter_icd_list(gt_primary_clean, list_to_filter)
    
    if not relevant_icd_list:
        relevant_icd_list = [gt_primary_clean] # Fallback
        
    print("Relevant ICD diagnoses selected by LLM:", relevant_icd_list)

    # Call the judge with the prediction and the *filtered list*
    return await _run_llm_judge(llm_pred_clean, relevant_icd_list, gt_primary_clean)


# Used only for GPT OSS:

# from openai import AsyncOpenAI
# import json
# import re
# from config import MODEL_NAME

# # Configure the async client to connect to the vLLM server
# client = AsyncOpenAI(
#     base_url="http://0.0.0.0:8000/v1",
#     api_key="vllm"
# )

# async def get_differential_diagnoses(scenario):
#     """Asynchronously gets differential diagnoses using a one-shot example prompt."""
#     # This system prompt tells the model its role and forbids conversational text.
#     system_prompt = (
#         "You are a medical data processor. Your only task is to generate a JSON object. "
#         "Do not provide any explanation, preamble, or conversational text. "
#         "Your entire response must be only the requested JSON."
#     )
#     # This human prompt shows the model a clear example of what is expected.
#     human_prompt = (
#         "Based on the patient scenario, provide a list of possible differential diagnoses. "
#         "Respond with a valid JSON object with a single key 'diagnoses'.\n\n"
#         "--- EXAMPLE START ---\n"
#         "Patient Scenario: A 45-year-old male presents with sharp, stabbing right lower quadrant pain that started 12 hours ago. He has a fever of 101.5°F, nausea, and anorexia.\n"
#         '{"diagnoses": ["Acute Appendicitis", "Meckel\'s Diverticulitis", "Right-sided Colonic Diverticulitis", "Infectious Ileocolitis"]}\n'
#         "--- EXAMPLE END ---\n\n"
#         "--- TASK START ---\n"
#         "Patient Scenario: {scenario}\n"
#     )
    
#     response_text = ""
#     try:
#         response = await client.chat.completions.create(
#             model=MODEL_NAME,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": human_prompt.format(scenario=scenario)}
#             ],
#             temperature=0.0, # Set to 0.0 for deterministic JSON output
#             max_tokens=1500  # A safe limit for a list of diagnoses
#         )
#         response_text = response.choices[0].message.content
#         cleaned_text = re.sub(r"```json\n?|```", "", response_text).strip()
#         response_json = json.loads(cleaned_text)
#         return response_json.get("diagnoses", [])
#     except Exception as e:
#         print(f"Warning: JSON error in differential diagnosis: {e}. Falling back to text parsing.")
#         # The fallback logic now safely handles an empty response_text
#         matches = re.findall(r'\"([^\"]+)\'', response_text or "")
#         return [m for m in matches if m.lower() not in ["diagnoses", "tests"]]

# async def get_test_suggestions(scenario, diagnoses_list, available_tests, allowed_radiology_tests, test_results_str=""):
#     """Asynchronously gets test suggestions using a one-shot example prompt."""
#     system_prompt = (
#         "You are a medical data processor. Your only task is to generate a JSON object listing tests. "
#         "Do not explain your choices or add any extra text. "
#         "Your entire response must be only the requested JSON."
#     )
#     human_prompt = (
#         "Based on the scenario and diagnoses, suggest 2-6 unique, cost-effective tests. "
#         "Respond with a valid JSON object with a single key 'tests'.\n"
#         "--- EXAMPLE START ---\n"
#         "Patient Scenario: A 45-year-old male with RLQ pain.\n"
#         "Differential Diagnoses: Acute Appendicitis, Meckel's Diverticulitis\n"
#         "Previous test results:\n\n"
#         "Choose lab tests ONLY from: White Blood Cells, Hemoglobin, Platelet Count. "
#         "Choose radiology tests ONLY from: CT ABD & PELVIS WITH CONTRAST, US APPENDIX.\n"
#         '{"tests": ["White Blood Cells", "CT ABD & PELVIS WITH CONTRAST"]}\n'
#         "--- EXAMPLE END ---\n\n"
#         "--- TASK START ---\n"
#         "Patient Scenario: {scenario}\n"
#         "Differential Diagnoses: {diagnoses}\n"
#         "Previous test results:\n{test_results}\n"
#         "For lab and microbiology tests, choose ONLY from this list: {lab_tests}. "
#         "For radiology tests, choose ONLY from this list: {rad_tests}."
#     )

#     response_text = ""
#     try:
#         response = await client.chat.completions.create(
#             model=MODEL_NAME,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": human_prompt.format(
#                     scenario=scenario,
#                     diagnoses=', '.join(diagnoses_list),
#                     test_results=test_results_str,
#                     lab_tests=', '.join(available_tests),
#                     rad_tests=', '.join(allowed_radiology_tests)
#                 )}
#             ],
#             temperature=0.0, # Set to 0.0 for deterministic JSON output
#             max_tokens=512   # A safe limit for a list of tests
#         )
#         response_text = response.choices[0].message.content
#         cleaned_text = re.sub(r"```json\n?|```", "", response_text).strip()
#         response_json = json.loads(cleaned_text)
#         return response_json.get("tests", [])
#     except Exception as e:
#         print(f"Warning: JSON error in test suggestions: {e}. Falling back to text parsing.")
#         matches = re.findall(r'\"([^\"]+)\'', response_text or "")
#         return [m for m in matches if m.lower() not in ["diagnoses", "tests"]]

# async def get_final_decision(scenario, diagnoses_list, test_results_str):
#     """Asynchronously gets a final decision using a two-shot example prompt."""
#     system_prompt = (
#         "You are a medical decision engine. Your task is to output a single line of text in a specific format. "
#         "Do not provide any explanation or conversational text."
#     )
#     human_prompt = (
#         "Based on the provided information, make a final decision. "
#         "Your response MUST start with either 'Final Diagnosis:' or 'Additional Tests:'.\n\n"
#         "--- EXAMPLE 1 (Final Diagnosis) ---\n"
#         "Test Results:\n- White Blood Cells: 18.5 (Abnormal)\n- CT ABD & PELVIS WITH CONTRAST: Findings consistent with acute appendicitis.\n"
#         "Final Diagnosis: Acute Appendicitis\n"
#         "--- EXAMPLE 2 (More Tests Needed) ---\n"
#         "Test Results:\n- White Blood Cells: 11.2 (Normal)\n"
#         "Additional Tests: CT ABD & PELVIS WITH CONTRAST\n"
#         "--- END EXAMPLES ---\n\n"
#         "--- TASK START ---\n"
#         "Scenario: {scenario}\n"
#         "Diagnoses: {diagnoses}\n"
#         "Test Results:\n{test_results}"
#     )
    
#     response = await client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": human_prompt.format(
#                 scenario=scenario,
#                 diagnoses=', '.join(diagnoses_list),
#                 test_results=test_results_str
#             )}
#         ],
#         temperature=0.1, # A little creativity is okay for the final text response
#         max_tokens=512   # A safe limit for the decision
#     )
#     # Return the response directly, ensuring it's not None
#     return response.choices[0].message.content.strip() or "<no response>"