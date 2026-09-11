# agents.py
import json
import re
from typing import Dict, List, Any
from llm_client import UniversalLLMClient
from graph_state import AgentState
from utils import get_test_results, parse_json_list_from_key, parse_json_dict
# from tool_wrapper import SearchTool

def update_token_stats(state: AgentState, usage: Dict[str, int]) -> Dict[str, int]:
    return {
        "total_input_tokens": state.get("total_input_tokens", 0) + usage.get('input', 0),
        "total_output_tokens": state.get("total_output_tokens", 0) + usage.get('output', 0),
        "total_tokens": state.get("total_tokens", 0) + usage.get('total', 0)
    }

# --- DDx Sub-System ---
async def differential_diagnosis_agent(state: AgentState, vllm_client: UniversalLLMClient) -> Dict:
    print("--- Sub-System Node: Differential Diagnosis ---")
    scenario = state["patient_scenario"]
    feedback = state.get("sub_loop_feedback", "None") 
    system_prompt = (
        "You are a medical expert... Respond ONLY as a valid JSON object with two keys:\n"
        "1. 'diagnoses': A list of string diagnoses.\n"
        "2. 'explanation': A brief (1-2 sentence) explanation for your choices."
    )
    user_prompt = (
        f"Generate a differential diagnosis for the case below.\n"
        f"**CRITICAL FEEDBACK ON PREVIOUS ATTEMPT**: {feedback}\n"
        f"If feedback is present, YOU MUST provide a new, corrected list.\n\n"
        f"Patient Scenario: {scenario}\n\n"
        f"Example Response:\n"
        f"{{\"diagnoses\": [\"Diagnosis 1\"], \"explanation\": \"...\"}}"
    )
    response_text, usage = await vllm_client.invoke(system_prompt, user_prompt, temperature=0.3, max_tokens=4096)
    response_json = parse_json_dict(response_text)
    
    updates = update_token_stats(state, usage)
    updates.update({
        "differential_diagnoses": response_json.get("diagnoses", []),
        "ddx_explanation": response_json.get("explanation", "No explanation"),
        "ddx_loop_count": state.get("ddx_loop_count", 0) + 1
    })
    return updates

    # return {
    #     "differential_diagnoses": response_json.get("diagnoses", []),
    #     "ddx_explanation": response_json.get("explanation", "No explanation"),
    #     "ddx_loop_count": state.get("ddx_loop_count", 0) + 1
    # }

def ddx_verifier_agent(state: AgentState, vllm_client: UniversalLLMClient = None) -> Dict:
    print("--- Sub-System Node: DDx Verifier (Rule-Based) ---")
    loop_count = state.get("ddx_loop_count", 0)
    diagnoses = state.get("differential_diagnoses", [])
    if loop_count > 2:
        print("DDx loop limit reached. Forcing acceptance.")
        return {"sub_loop_decision": "accept"}
    if not diagnoses:
        print("DDx Verifier: List is empty. Requesting redo.")
        return {
            "sub_loop_feedback": "No diagnoses were provided. Please generate a list.",
            "sub_loop_decision": "redo_ddx"
        }
    if len(diagnoses) > 5:
        print(f"DDx Verifier: List has {len(diagnoses)} items. Requesting trim to 3-5.")
        return {
            "sub_loop_feedback": "The list is too long. Please provide only the top 3-5 most likely diagnoses.",
            "sub_loop_decision": "redo_ddx"
        }
    print("DDx Verifier: List is valid (1-5 items). Accepting.")
    return {"sub_loop_decision": "accept"}

# --- Test Selection Sub-System ---
async def test_selection_agent(state: AgentState, vllm_client: UniversalLLMClient) -> Dict:
    print("--- Sub-System Node: Test Selection ---")
    config = state["config"]
    feedback = state.get("sub_loop_feedback", "None")
    system_prompt = (
        "You are a medical diagnostician. Respond ONLY as a valid JSON object with two keys:\n"
        "1. 'tests': A list of string test names (can be empty).\n"
        "2. 'explanation': A brief explanation. If 'tests' is empty, you MUST explain why (e.g., 'Clinical diagnosis')."
    )
    prompt = (
        f"Based on the scenario and diagnoses, suggest the most essential tests.\n"
        f"**CRITICAL FEEDBACK ON PREVIOUS ATTEMPT**: {feedback}\n"
        f"If feedback is present, provide a new, corrected list.\n"
        f"If no tests are needed, return an empty 'tests' list and explain why.\n\n"
        f"Patient Scenario: {state['patient_scenario']}\n"
        f"Differential Diagnoses: {', '.join(state['differential_diagnoses'])}\n"
        f"Previous test results:\n{state.get('test_results_str', 'None')}\n\n"
        f"For lab/microbiology, choose ONLY from: {', '.join(state['available_tests'])}.\n"
        f"For radiology, choose ONLY from: {', '.join(config.ALLOWED_RADIOLOGY_TESTS)}.\n"
    )
    response_text, usage = await vllm_client.invoke(system_prompt, prompt, temperature=0.7, max_tokens=4096)
    response_json = parse_json_dict(response_text)
    # return {
    #     "newly_suggested_tests": response_json.get("tests", []),
    #     "tests_explanation": response_json.get("explanation", "No explanation"),
    #     "test_loop_count": state.get("test_loop_count", 0) + 1
    # }
    tests = response_json.get("tests", [])
    
    updates = update_token_stats(state, usage)
    updates.update({
        "newly_suggested_tests": tests,
        "initial_proposed_test_count": len(tests), # Metric capture
        "tests_explanation": response_json.get("explanation", "No explanation"),
        "test_loop_count": state.get("test_loop_count", 0) + 1
    })
    return updates

async def test_frugality_agent(state: AgentState, vllm_client: UniversalLLMClient = None) -> Dict:
    print("--- Sub-System Node: Cost-Aware Frugality Agent ---")
    config = state["config"]
    tests = state.get("newly_suggested_tests", [])
    test_costs = state.get("test_costs", {})
    loop_count = state.get("test_loop_count", 0)
    budget = config.BUDGET_CAP
    
    if loop_count >= 2:
        print(f"   > Speed Limit: Loop count {loop_count}. Auto-accepting to prevent infinite loops.")
        current_total = sum([test_costs.get(t, 50.0) for t in tests])
        return {
            "sub_loop_decision": "accept", 
            "final_kept_test_count": len(tests),
            "newly_suggested_tests": tests,
            "total_case_cost": current_total
        }
    
    # 1. Non-Frugal Mode Check
    if not getattr(config, 'USE_LLM_FRUGALITY', True):
        current_total = sum([test_costs.get(t, 50.0) for t in tests])
        return {"sub_loop_decision": "accept", "final_kept_test_count": len(tests), "total_case_cost": current_total, "newly_suggested_tests": tests}

    # 2. Empty List Check
    if not tests:
        return {"sub_loop_decision": "accept_no_tests", "final_kept_test_count": 0}

    # 3. Enrich with Costs (Search if missing)
    # search_tool = SearchTool(max_results=1)
    detailed_costs = []
    current_total = 0.0
    
    for test in tests:
        cost = test_costs.get(test)
        if cost is None:
            cost = 50.0  # Default cost for unknown tests
            # Assist the agent with a web search for missing prices
            # (In future, cache this to avoid repeated searches)
            # try:
            #     res = search_tool.search(f"average medicare reimbursement cost of {test} lab test")
            #     # Look for patterns like $15 or $15.50
            #     prices = re.findall(r'\$\s?(\d+(?:\.\d+)?)', res)
            #     if prices:
            #         cost = float(prices[0])
            #     else:
            #         cost = 50.0
            # except:
            #     cost = 50.0
        current_total += cost
        detailed_costs.append(f"- {test}: ${cost:.2f}")

    # 4. Detailed Structured Prompt
    system_prompt = (
        "You are a cost-conscious medical reviewer. Your job is to ensure diagnostic accuracy while adhering to a strict budget. "
        "Respond ONLY with a valid JSON object with keys 'tests' (list), 'decision' (str), and 'explanation' (str)."
    )
    
    user_prompt = f"""
    Review the 'Proposed Tests' and their associated costs against the 'Budget Cap'.
    
    Budget Cap: ${budget}
    Proposed Tests Total: ${current_total:.2f}
    
    Proposed Tests (with costs): 
    {chr(10).join(detailed_costs)}
    
    Clinical Context (DDx): {state['differential_diagnoses']}
    Doctor's Rationale: {state['tests_explanation']}

    Rules:
    1. If 'Proposed Tests Total' <= 'Budget Cap':
       - "decision": "accept"
       - "tests": {tests} 
       - "explanation": "Total cost is within the budget."
    
    2. If 'Proposed Tests Total' > 'Budget Cap':
       - You MUST attempt to filter the list.
       - Remove low-value, redundant, or "nice-to-have" tests.
       - Keep "Gold Standard" tests for the suspected diagnosis, even if they are expensive.
       - If you can create a valid list under (or close to) budget:
         - "decision": "accept"
         - "tests": [Your filtered list]
         - "explanation": "Removed [Test X] to meet budget; retained essential tests."
       
    3. If you CANNOT reduce the cost without missing a critical diagnosis (e.g., all tests are vital life-saving measures):
       - "decision": "redo_tests"
       - "tests": [] 
       - "explanation": "Cannot meet budget without compromising patient safety. All proposed tests are critical."

    Example Response:
    {{"decision": "accept", "tests": ["Test A", "Test B"], "explanation": "Removed Test C (redundant) to stay within budget."}}
    """
    
    response_text, usage = await vllm_client.invoke(system_prompt, user_prompt, temperature=0.4, max_tokens=4096)
    review = parse_json_dict(response_text)
    
    filtered_tests = review.get("tests", tests)
    
    # Calculate final cost of the filtered list
    final_cost = sum([test_costs.get(t, 50.0) for t in filtered_tests])

    updates = update_token_stats(state, usage)
    updates.update({
        "newly_suggested_tests": filtered_tests,
        "final_kept_test_count": len(filtered_tests),
        "total_case_cost": final_cost,
        "sub_loop_decision": review.get("decision", "accept"),
        "sub_loop_feedback": review.get("explanation", "Processed by Frugality Agent")
    })
    return updates

# --- Final Diagnosis Sub-System ---
async def final_diagnosis_agent(state: AgentState, vllm_client: UniversalLLMClient) -> Dict:
    print("--- Sub-System Node: Final Diagnosis ---")
    # feedback = state.get("sub_loop_feedback", "None")
    system_prompt = (
        "You are a senior diagnostician. Respond ONLY as a valid JSON object with two keys:\n"
        "1. 'diagnosis': A single string...\n"
        "2. 'confidence': A float between 0.0 (uncertain) and 1.0 (certain).\n"
        "3. 'explanation': A 1-2 sentence explanation..."
    )
    prompt = (
        f"Based on all information, provide a final decision.\n"
        # f"**CRITICAL FEEDBACK ON PREVIOUS ATTEMPT**: {feedback}\n"
        f"If 'Test Results' is empty or 'None', you MUST make a clinical diagnosis based on the Scenario and DDx alone.\n\n"
        f"Patient Scenario: {state['patient_scenario']}\n"
        f"Differential Diagnoses: {', '.join(state['differential_diagnoses'])}\n"
        f"Test Results:\n{state.get('test_results_str', 'None')}\n\n"
        f"Example Response: {{\"diagnosis\": \"Myocardial Infarction\", \"confidence\": 0.95, \"explanation\": \"...\"}}\n"
    )
    response_text, usage = await vllm_client.invoke(system_prompt, prompt, temperature=0.3, max_tokens=4096)
    response_json = parse_json_dict(response_text)
    # return {
    #     "final_diagnosis_text": response_json.get("diagnosis", "Error: Parsing failed"),
    #     "final_dx_explanation": response_json.get("explanation", "No explanation"),
    #     # "final_dx_loop_count": state.get("final_dx_loop_count", 0) + 1
    #     "final_dx_loop_count": 1 # No looping for final dx
    # }
    updates = update_token_stats(state, usage)
    updates.update({
        "final_diagnosis_text": response_json.get("diagnosis", "Error: Parsing failed"),
        "final_dx_confidence": response_json.get("confidence", 0.5),
        "final_dx_explanation": response_json.get("explanation", "No explanation"),
        "final_dx_loop_count": 1 
    })
    return updates

async def get_test_results_node(state: AgentState, vllm_client: UniversalLLMClient = None) -> Dict:
    print("--- Main Graph Node: Get Test Results ---")
    new_tests = state["newly_suggested_tests"]
    if not new_tests:
        print("No new tests to get results for (Clinical diagnosis).")
        return {} 
    results_dict = get_test_results(
        record=state["patient_case"],
        suggested_tests=new_tests,
        label_to_code=state["label_to_code"],
        available_tests=state["available_tests"],
        config=state["config"]
    )
    new_results_str = "\n".join([f"- {k}: {v}" for k, v in results_dict.items()])
    updated_results_str = state.get("test_results_str", "") + "\n" + new_results_str
    updated_cumulative_tests = state.get("cumulative_suggested_tests", []) + new_tests
    print(f"New Results:\n{new_results_str}")
    return {
        "test_results_str": updated_results_str.strip(),
        "cumulative_suggested_tests": list(set(updated_cumulative_tests))
    }