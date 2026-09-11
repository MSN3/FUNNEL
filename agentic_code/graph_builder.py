# graph_builder.py
from langgraph.graph import StateGraph, END
from functools import partial
from graph_state import AgentState
from agents import (
    # DDx Sub-system
    differential_diagnosis_agent,
    ddx_verifier_agent,
    test_selection_agent,
    test_frugality_agent,
    # Final Dx Sub-system
    final_diagnosis_agent,
    # final_dx_reason_checker_agent
    # Tool Node
    get_test_results_node
)
from llm_client import UniversalLLMClient
import os

# --- Sub-Graph Router Functions ---

def route_ddx_sub_loop(state: AgentState) -> str:
    """Router for the DDx sub-loop."""
    decision = state.get("sub_loop_decision", "redo_ddx")
    if decision == "accept":
        return END # Exit the sub-graph
    return "differential_diagnosis" # Loop back

def route_test_sub_loop(state: AgentState) -> str:
    """Router for the Test sub-loop."""
    decision = state.get("sub_loop_decision", "redo_tests")
    if decision == "accept" or decision == "accept_no_tests":
        return END # Exit the sub-graph
    return "test_selection" # Loop back

# def route_final_dx_sub_loop(state: AgentState) -> str:
#     """Router for the Final Dx sub-loop."""
#     decision = state.get("sub_loop_decision", "redo_final_dx")
#     if decision == "redo_final_dx":
#         return "final_diagnosis" # Loop back
#     return END # Exit the sub-graph (decision is 'accept')

# --- Graph Assembly Function ---

def create_graph(vllm_client: UniversalLLMClient):
    """
    Assembles the hierarchical graph system with rule-based verifiers.
    """
    
    # 1. Bind the client to all *LLM-based* agent functions
    ddx_agent = partial(differential_diagnosis_agent, vllm_client=vllm_client)
    test_agent = partial(test_selection_agent, vllm_client=vllm_client)
    frugal_agent_llm = partial(test_frugality_agent, vllm_client=vllm_client)
    final_dx_agent = partial(final_diagnosis_agent, vllm_client=vllm_client)
    # reason_checker_agent = partial(final_dx_reason_checker_agent, vllm_client=vllm_client)

    # === DEFINE DDx SUB-GRAPH ===
    ddx_workflow = StateGraph(AgentState)
    ddx_workflow.add_node("differential_diagnosis", ddx_agent)
    ddx_workflow.add_node("ddx_verifier", ddx_verifier_agent) # <-- Use Python function
    ddx_workflow.set_entry_point("differential_diagnosis")
    ddx_workflow.add_edge("differential_diagnosis", "ddx_verifier")
    ddx_workflow.add_conditional_edges(
        "ddx_verifier",
        route_ddx_sub_loop,
        {"differential_diagnosis": "differential_diagnosis", END: END}
    )
    ddx_system = ddx_workflow.compile()
    print("DDx Sub-System Compiled (Rule-Based Verifier).")

    # === DEFINE TEST SUB-GRAPH ===
    test_workflow = StateGraph(AgentState)
    test_workflow.add_node("test_selection", test_agent)
    # test_workflow.add_node("test_frugality_agent", test_frugality_agent) # <-- Use Python function
    test_workflow.add_node("test_frugality_agent", frugal_agent_llm)
    test_workflow.set_entry_point("test_selection")
    test_workflow.add_edge("test_selection", "test_frugality_agent")
    test_workflow.add_conditional_edges(
        "test_frugality_agent",
        route_test_sub_loop,
        {"test_selection": "test_selection", END: END}
    )
    test_system = test_workflow.compile()
    print("Test Sub-System Compiled (Rule-Based Verifier).")

    # === DEFINE FINAL DX SUB-GRAPH ===
    # final_dx_workflow = StateGraph(AgentState)
    # final_dx_workflow.add_node("final_diagnosis", final_dx_agent)
    # final_dx_workflow.add_node("final_dx_reason_checker", reason_checker_agent)
    # final_dx_workflow.set_entry_point("final_diagnosis")
    # final_dx_workflow.add_edge("final_diagnosis", "final_dx_reason_checker")
    # final_dx_workflow.add_conditional_edges(
    #     "final_dx_reason_checker",
    #     route_final_dx_sub_loop,
    #     {"final_diagnosis": "final_diagnosis", END: END}
    # )
    # final_dx_system = final_dx_workflow.compile()
    # print("Final Dx Sub-System Compiled.")

    # === DEFINE MAIN ORCHESTRATOR (HEALTH MANAGER) ===
    main_workflow = StateGraph(AgentState)
    
    main_workflow.add_node("ddx_system", ddx_system)
    main_workflow.add_node("test_system", test_system)
    main_workflow.add_node("get_test_results", get_test_results_node)
    main_workflow.add_node("final_diagnosis", final_dx_agent)

    main_workflow.set_entry_point("ddx_system")
    
    # --- THIS IS THE NEW, SIMPLIFIED FLOW ---
    # A -> B -> C -> D -> END
    main_workflow.add_edge("ddx_system", "test_system")
    main_workflow.add_edge("test_system", "get_test_results")
    main_workflow.add_edge("get_test_results", "final_diagnosis")
    main_workflow.add_edge("final_diagnosis", END) # End after the final system runs
    
    # Compile the final, hierarchical graph
    print("Compiling Main Orchestrator Graph (Linear Flow)...")
    app = main_workflow.compile() 
    print("Main Orchestrator compiled successfully.")
            
    return app