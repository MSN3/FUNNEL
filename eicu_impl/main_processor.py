#--------------------------------------Type 3 with LLM as a judge--------------------------------------#
# main_processor.py
import pandas as pd
import asyncio
import time
from config_loader import load_config_from_path, load_data_from_config
from llm_client import UniversalLLMClient
from graph_state import AgentState
from graph_builder import create_graph
from evaluator import run_evaluation_and_plotting
from utils import LLMJudgeEvaluator, estimate_real_life_tests, generate_patient_scenario, get_real_life_test_names

def calculate_complex_metrics(state: AgentState, is_correct: bool, record: dict):
    # 1. CES (Cost Efficiency Score)
    model_tests = state.get('cumulative_suggested_tests', [])
    
    real_tests = get_real_life_test_names(record)
    # ------------------------------------------------------------
    
    test_costs = state.get('test_costs', {})
    model_cost = sum([test_costs.get(t, 50.0) for t in model_tests])
    
    # Calculate Real Cost
    real_cost = 0.0
    for t in real_tests:
        cost = test_costs.get(t)
        if cost is None:
            if "Radiology" in t: cost = 100.0
            elif "Microbiology" in t: cost = 50.0
            else: cost = 50.0
        real_cost += cost
    
    # Avoid div/0
    ces = model_cost / real_cost if real_cost > 0 else (model_cost / 50.0 if model_cost > 0 else 1.0)

    # 2. UTR (Unnecessary Test Rate)
    model_tests = set(state.get('cumulative_suggested_tests', []))
    
    intersection = model_tests.intersection(set(real_tests))
    unnecessary_count = len(model_tests) - len(intersection)
    utr = unnecessary_count / len(model_tests) if len(model_tests) > 0 else 0.0

    # 3. FI (Frugality Index) = Accuracy / CES
    fi = (1.0 if is_correct else 0.0) / ces if ces > 0 else 0.0

    # 4. StD (Steps to Diagnosis)
    std = state.get('ddx_loop_count', 0) + state.get('test_loop_count', 0) + 1

    return {"CES": ces, "UTR": utr, "FI_New": fi, "StD": std, "Real_Cost": real_cost, "total_case_cost": model_cost}   

async def process_data_in_chunks(
    app, 
    patient_records, 
    config, 
    label_to_code, 
    available_tests,
    test_costs,
    embedding_evaluator, # This is now an LLMJudgeEvaluator
    progress_callback=None
    ):
    """
    Processes all patient records in chunks, running the graph for each.
    """
    all_results = []
    chunk_size = config.CHUNK_SIZE
    num_to_run = len(patient_records)
    total_chunks = (num_to_run + chunk_size - 1) // chunk_size
    
    start_time = time.time()
    run_config = {"recursion_limit": 75} 

    for i in range(0, num_to_run, chunk_size):
        chunk_records = patient_records[i:i + chunk_size]
        print(f"\n--- Processing chunk {i // chunk_size + 1}/{total_chunks} (Patients {i+1} to {i + len(chunk_records)}) ---\n")
        
        graph_tasks = []
        for record in chunk_records:
            initial_state = AgentState(
                patient_case=record, config=config, label_to_code=label_to_code,
                available_tests=available_tests, patient_scenario=generate_patient_scenario(record),
                differential_diagnoses=[], ddx_explanation="",
                cumulative_suggested_tests=[], newly_suggested_tests=[],
                tests_explanation="", test_results_str="",
                final_diagnosis_text="Unable to reach diagnosis", final_dx_explanation="",
                final_dx_confidence=0.0,
                sub_loop_feedback="", sub_loop_decision="",
                ddx_loop_count=0, test_loop_count=0,#, final_dx_loop_count=0
                total_input_tokens=0, total_output_tokens=0, total_tokens=0,
                initial_proposed_test_count=0, final_kept_test_count=0, total_case_cost=0.0
            )
            graph_tasks.append(app.ainvoke(initial_state, config=run_config))

        # 1. Run all graph invocations concurrently
        chunk_final_states = await asyncio.gather(*graph_tasks, return_exceptions=True)

        # ---CONCURRENT EVALUATION ---
        evaluation_tasks = []
        valid_final_states = []

        for j, final_state in enumerate(chunk_final_states):
            record_index = i + j
            if isinstance(final_state, Exception):
                print(f"Error processing record {record_index} (Graph): {final_state}")
                continue 
            if not isinstance(final_state, dict):
                 print(f"Error: Invalid final state for record {record_index}. Skipping.")
                 continue 

            llm_diagnosis = final_state.get('final_diagnosis_text', 'Error: Missing Diagnosis')
            record = final_state.get('patient_case', patient_records[record_index])
            
            # Get BOTH the full ICD list (as a string) and the fallback
            icd_list_str = record.get('Discharge Diagnosis')
            fallback_gt = record.get('ground_truth_disease')
            
            if not icd_list_str or not fallback_gt:
                print(f"Skipping record {record_index}: Missing ground truth data.")
                continue

            # Add the async judge call to the task list
            evaluation_tasks.append(
                embedding_evaluator.evaluate_diagnosis(
                    llm_diagnosis, icd_list_str, fallback_gt
                )
            )
            valid_final_states.append(final_state)
        
        # 2. Run all judge evaluations concurrently
        print(f"--- Running LLM-as-a-Judge for {len(evaluation_tasks)} records ---")
        evaluation_results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
        print(f"--- LLM-as-a-Judge complete ---")

        # 3. Now, build the final results list
        for k, eval_result in enumerate(evaluation_results):
            final_state = valid_final_states[k]
            record = final_state.get('patient_case')
            
            if isinstance(eval_result, Exception):
                print(f"Error processing record {record.get('Patient ID')} (Judge): {eval_result}")
                continue
            
            (is_correct, similarity_score, effective_diagnosis, judge_score, judge_reason, judge_tokens) = eval_result
            
            # Complex Metrics
            metrics = calculate_complex_metrics(final_state, is_correct, record)
            
            # Collect token usage
            agent_tokens = final_state.get('total_tokens', 0)
            total_tokens = agent_tokens + judge_tokens
            
            # (Rest of the logic is the same)
            cumulative_tests = final_state.get('cumulative_suggested_tests', [])
            num_tests_llm = len(set(cumulative_tests))
            # num_tests_real = estimate_real_life_tests(record)
            # frugality_ratio = num_tests_llm / num_tests_real if num_tests_real > 0 else 0
            final_kept = final_state.get('final_kept_test_count', 0)
            num_tests_real = estimate_real_life_tests(record)
            frugality_ratio = final_kept / num_tests_real if num_tests_real > 0 else 0
            
            # Obtain confidence value
            confidence = final_state.get('final_dx_confidence', 0.0)

            all_results.append({
                'patient_id': record.get('Patient ID', 'N/A'),
                'ground_truth_disease': record.get('ground_truth_disease'),
                'detailed_actual_diagnosis': record.get('Discharge Diagnosis', 'N/A'),
                'llm_diagnosis': effective_diagnosis,
                'detailed_llm_diagnosis': final_state.get('final_diagnosis_text', 'Error'),
                'similarity_score': similarity_score, # Will be 1.0 or 0.0
                'model_confidence': confidence,
                'is_correct': is_correct,
                'tests_ordered': cumulative_tests,
                # Metrics
                'num_tests_llm': num_tests_llm,
                'num_tests_real': num_tests_real,
                'frugality_ratio': frugality_ratio,
                'ddx_explanation': final_state.get('ddx_explanation', 'N/A'),
                'tests_explanation': final_state.get('tests_explanation', 'N/A'),
                'final_test_count': final_kept,
                'final_dx_explanation': final_state.get('final_dx_explanation', 'N/A'),
                'judge_score': judge_score,
                'judge_reason': judge_reason,
                # Token usage
                'total_tokens': total_tokens,
                'agent_tokens': agent_tokens,
                'judge_tokens': judge_tokens,
                # Test loop info
                'initial_test_count': final_state.get('initial_proposed_test_count', 0),
                'final_kept_test_count': final_kept,
                'final_loop_count': final_state.get('ddx_loop_count', 0) + final_state.get('test_loop_count', 0),
                'total_case_cost_old': final_state.get('total_case_cost', 0.0),
                # Complex Metrics
                **metrics
            })
        
        processed_count = len(all_results)
        if progress_callback and num_to_run > 0:
            progress_callback(processed_count / num_to_run)
            
    end_time = time.time()
    print(f"\n--- Graph processing complete. Time taken: {end_time - start_time:.2f} seconds ---")
    
    if not all_results:
        print("Warning: No results were successfully processed.")
        return pd.DataFrame()

    return pd.DataFrame(all_results)


async def run_full_analysis_pipeline(config_path: str, progress_callback=None):
    total_start_time = time.time()
    try:
        print("Loading config...")
        config = load_config_from_path(config_path)
        patient_records, label_to_code, available_tests, test_costs = load_data_from_config(config)

        # --- THIS IS THE CHANGE ---
        print("Initializing vLLM clients...")
        # 1. Create the Agent Client
        agent_vllm_client = UniversalLLMClient(
            provider=config.AGENT_PROVIDER, 
            model_name=config.AGENT_MODEL_NAME, 
            base_url=getattr(config, 'AGENT_BASE_URL', None), 
            api_key=getattr(config, 'AGENT_API_KEY', None),
            project_id=getattr(config, 'GOOGLE_PROJECT_ID', None),
            location=getattr(config, 'GOOGLE_LOCATION', 'us-central1')
        )
        
        # 2. Create the (separate) Judge Client
        judge_vllm_client = UniversalLLMClient(
            provider=config.JUDGE_PROVIDER, 
            model_name=config.JUDGE_MODEL_NAME, 
            base_url=getattr(config, 'JUDGE_BASE_URL', None), 
            api_key=getattr(config, 'JUDGE_API_KEY', None),
            project_id=getattr(config, 'GOOGLE_PROJECT_ID', None)
        )
        print("vLLM Clients initialized.")
        print("vLLM Clients initialized.")
        
        # 3. Pass the correct client to each component
        # app = create_graph(agent_vllm_client, judge_vllm_client) # Graph uses the AGENT client
        app = create_graph(agent_vllm_client) # Graph uses the AGENT client

        print("Initializing LLMJudgeEvaluator...")
        embedding_evaluator = LLMJudgeEvaluator(
            judge_vllm_client=judge_vllm_client, # Evaluator uses the JUDGE client
            config=config
        )
        embedding_evaluator.fit()
        print("LLMJudgeEvaluator initialized.")
        # --- END OF CHANGE ---
        
        print("Starting data processing chunks...")
        results_df = await process_data_in_chunks(
            app, 
            patient_records, 
            config, 
            label_to_code, 
            available_tests,
            test_costs, 
            embedding_evaluator,
            progress_callback
        )

        if results_df.empty:
            print("No results to evaluate.")
            return None, {"error": "No results processed"}, {}
            
        print("Running final evaluation and plotting...")
        summary_metrics, generated_plots = run_evaluation_and_plotting(
            results_df, 
            config.AGENT_MODEL_NAME
        )
        
        total_end_time = time.time()
        print(f"Full pipeline finished in {total_end_time - total_start_time:.2f} seconds.")
        
        return results_df, summary_metrics, generated_plots
        
    except Exception as e:
        print(f"FATAL ERROR in run_full_analysis_pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None, {"error": str(e)}, {}