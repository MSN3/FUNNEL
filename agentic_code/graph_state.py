# graph_state.py
from typing import List, Dict, TypedDict, Optional, Any

class AgentState(TypedDict):
    """
    Represents the main "Orchestrator" state for a single patient.
    """
    # --- Inputs (set at start) ---
    patient_case: Dict[str, Any]
    config: Any
    label_to_code: Dict[str, str]
    available_tests: List[str]
    test_costs: Dict[str, float]
    patient_scenario: str

    # --- Agent Outputs (accumulated) ---
    differential_diagnoses: List[str]
    ddx_explanation: str
    
    cumulative_suggested_tests: List[str]
    newly_suggested_tests: List[str]
    tests_explanation: str
    
    test_results_str: str
    
    final_diagnosis_text: str
    final_dx_confidence: float
    final_dx_explanation: str
    
    # --- Control Flow ---
    # Feedback to be passed *into* a sub-loop
    sub_loop_feedback: str 
    # Decision from a sub-loop
    sub_loop_decision: str 
    
    # Granular loop counters
    ddx_loop_count: int
    test_loop_count: int
    #final_dx_loop_count: int
    
    # New Metrics
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    
    initial_proposed_test_count: int
    final_kept_test_count: int
    total_case_cost: float