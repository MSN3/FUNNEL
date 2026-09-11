# main.py

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import asyncio
import time
import ast
from sklearn.metrics import confusion_matrix
import config
from data_loader import load_data_with_costs
from utils import (
    # are_diagnoses_related,
    estimate_real_life_tests,
    generate_patient_scenario,
    get_test_results,
    get_real_life_test_names, # New Import
    calculate_cost_metrics     # New Import
)
from llm_service import (
    get_differential_diagnoses,
    get_test_suggestions,
    get_final_decision,
    evaluate_diagnosis_with_judge
)

# This async workflow now includes the full, original logic
# Added 'config' as a parameter
async def frugal_llm_workflow_async(record, record_idx, label_to_code, available_tests, test_costs, config, evaluator=None, max_iterations=3):
    """Asynchronously processes a single patient record with full original logic."""
    scenario = generate_patient_scenario(record)
    print(f"### Step 1: Patient Scenario\n{scenario}\n")
    
    diagnoses_list = await get_differential_diagnoses(scenario)
    print(f"### Step 2: Differential Diagnosis\n{', '.join(diagnoses_list)}\n")

    all_suggested_tests = []
    test_results_str = ""
    final_diagnosis_text = "Unable to reach a final diagnosis." 

    for iteration in range(max_iterations):
        print(f"--- Iteration {iteration+1} ---\n")
        suggested_tests = await get_test_suggestions(
            scenario, diagnoses_list, available_tests, 
            config.ALLOWED_RADIOLOGY_TESTS, test_results_str
        )
        
        new_tests = [test for test in suggested_tests if test not in all_suggested_tests]
        all_suggested_tests.extend(new_tests)
        print(f"### Step 3.{iteration+1}: Suggested Tests\n{', '.join(new_tests)}\n")
        
        if not new_tests and iteration > 0:
            print("No new tests suggested. Moving to final diagnosis.\n")
            final_response = await get_final_decision(scenario, diagnoses_list, test_results_str)
            final_response = final_response or ""
            if "final diagnosis:" in final_response.lower():
                final_diagnosis_text = final_response.split(":", 1)[-1].strip().split('.')[0]
            break

        test_results = get_test_results(record, new_tests, label_to_code, available_tests, config)
        test_results_str += "\n".join([f"- {k}: {v}" for k, v in test_results.items()]) + "\n"
        print(f"### Step 4.{iteration+1}: Test Results Found\n")

        final_response = await get_final_decision(scenario, diagnoses_list, test_results_str)
        final_response = final_response or ""
        print(f"### Step 5.{iteration+1}: LLM Decision\n{final_response}\n")

        if "final diagnosis" in final_response.lower():
            final_diagnosis_text = final_response.split(":", 1)[-1].strip().split('.')[0]
            break
        elif iteration == max_iterations - 1:
            final_diagnosis_text = "Unable to reach a final diagnosis."
        elif "additional tests" in final_response.lower():
            additional_tests_str = final_response.split(":", 1)[-1].strip()
            additional_tests = [test.strip() for test in additional_tests_str.split(',')]
            all_suggested_tests.extend(additional_tests)
            print(f"--- LLM requested additional tests: {', '.join(additional_tests)} ---\n")

    patient_id = record['Patient ID']
    
    # Get the differential diagnoses
    ddx_tests = diagnoses_list
    
    llm_diagnosis = final_diagnosis_text.lower().strip()
    
    # 1. Get the primary (fallback) ground truth
    primary_ground_truth = str(record['ground_truth_disease']).lower().strip()
    
    # 2. Get the *raw string* of the ICD list
    icd_list_str = record['ICD Diagnosis']
    
    # 3. Call the new evaluator ONCE
    # It handles parsing, fast-checks, filtering, and the batch LLM judge call
    eval_result_tuple = await evaluate_diagnosis_with_judge(
        llm_diagnosis,
        icd_list_str,
        primary_ground_truth
    )

    # 4. Unpack the 5-tuple response
    # (is_equivalent, similarity_score, effective_diagnosis, score, reasoning)
    is_correct = eval_result_tuple[0]
    similarity_score = eval_result_tuple[1]
    effective_diagnosis = eval_result_tuple[2]
    judge_score = eval_result_tuple[3]
    judge_reasoning = eval_result_tuple[4]
    
    # --- END OF MODIFIED EVALUATION LOGIC ---

    detailed_actual_diagnosis = record['ICD Diagnosis'] # Keep the original list string
    
    num_tests_llm = len(set(all_suggested_tests))
    num_tests_real = estimate_real_life_tests(record)
    frugality_ratio = num_tests_llm / num_tests_real if num_tests_real > 0 else 0

    # Cost metrics calculation
    real_test_names = get_real_life_test_names(record)
    unique_llm_tests = list(set(all_suggested_tests))
    cost_metrics = calculate_cost_metrics(
        unique_llm_tests,
        real_test_names,
        test_costs,
        is_correct
    )

    print(f"### Step 6: Evaluation for Patient {patient_id}\n"
          f"- Patient Id: {patient_id}\n"
          f"- Primary Diagnosis: {primary_ground_truth}\n"
          f"- LLM Diagnosis: {effective_diagnosis}\n"
          f"- Judge Score: {judge_score} (Reason: {judge_reasoning})\n"
          f"- Correct (A or B): {is_correct}\n"
          f"- Tests Ordered by LLM: {num_tests_llm}\n"
          f"- Tests in Real Life: {num_tests_real}\n"
          f"- Frugality Ratio: {frugality_ratio:.2f}\n"
          f"- CES: {cost_metrics['CES']:.2f} | UTR: {cost_metrics['UTR']:.2%}\n"
          f"- Frugality Index (New): {cost_metrics['FI']:.2f}\n"
          f"---------------------------------------------------\n")

    return {
        'patient_id': patient_id,
        'ddx_tests': ddx_tests,
        'ground_truth_disease': primary_ground_truth, # Store the primary GT
        'detailed_actual_diagnosis': detailed_actual_diagnosis, # Store the full list
        'llm_diagnosis': effective_diagnosis,
        'detailed_llm_diagnosis': final_diagnosis_text,
        'similarity_score': similarity_score,
        'is_correct': is_correct,
        'judge_score': judge_score,
        'judge_reasoning': judge_reasoning,
        'tests_ordered': all_suggested_tests,
        'tests_ordered_real': real_test_names,
        'num_tests_llm': num_tests_llm,
        'num_tests_real': num_tests_real,
        'frugality_ratio': frugality_ratio,
        'total_case_cost': cost_metrics['llm_cost'],
        'real_case_cost': cost_metrics['real_cost'],
        'CES': cost_metrics['CES'],
        'UTR': cost_metrics['UTR'],
        'FI_New': cost_metrics['FI']
    }

async def main():
    """Main function to run the simulation and analysis asynchronously."""
    start_time = time.time()

    df_patients, label_to_code, available_tests, test_costs = load_data_with_costs(
        config.PATIENTS_FILE_PATH,
        config.TEST_LABELS_FILE_PATH,
        config
    )
    # df_patients.dropna(subset=['Patient ID'], how='all', inplace=True)
    # df_test_labels = read_csv_file(config.TEST_LABELS_FILE_PATH)
    # label_to_code = dict(zip(df_test_labels['lab_label'], df_test_labels['Test Code']))
    # available_tests = df_test_labels['lab_label'].tolist()

    mask = df_patients['Patient ID'].str.contains('_', regex=False, na=False)
    extracted_disease_str = df_patients.loc[mask, 'Patient ID'].str.split('_', n=1).str[1]
    standardized_disease = extracted_disease_str.str.replace('_', ' ').str.strip()
    df_patients.loc[mask, 'ground_truth_disease'] = standardized_disease

    print("Standardized ground truth disease column created successfully.")

    num_patients_to_process = 20000
    num_to_run = min(num_patients_to_process, len(df_patients))
    
    all_results = []
    total_chunks = (num_to_run + config.CHUNK_SIZE - 1) // config.CHUNK_SIZE
    for i in range(0, num_to_run, config.CHUNK_SIZE):
        chunk_df = df_patients.iloc[i:i + config.CHUNK_SIZE]
        print(f"\n--- Processing chunk {i // config.CHUNK_SIZE + 1}/{total_chunks} (Patients {i+1} to {i + len(chunk_df)}) ---\n")

        tasks = []
        for idx, record in chunk_df.iterrows():
            # Pass the 'config' object to the workflow
            tasks.append(frugal_llm_workflow_async(record, idx, label_to_code, available_tests, test_costs, config, evaluator=None))

        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
        all_results.extend(chunk_results)
        
        successful_count = len([res for res in all_results if not isinstance(res, Exception)])
        print(f"--- Finished chunk. Total successful patients processed: {successful_count} ---")

    successful_results = [res for res in all_results if not isinstance(res, Exception)]
    errors = [res for res in all_results if isinstance(res, Exception)]
    if errors:
        print(f"\nEncountered {len(errors)} errors during processing.")
        for err in errors[:5]:
            print(err)

    if not successful_results:
        print("No successful results to analyze. Exiting.")
        return

    results_df = pd.DataFrame(successful_results)
    results_filename = f'frugal_llm_results_{config.MODEL_NAME.replace("/", "_")}.csv'
    results_df.to_csv(results_filename, index=False)
    print(f"Results saved to '{results_filename}'")

    total_patients = len(results_df)
    accuracy = results_df['is_correct'].mean() if total_patients > 0 else 0
    frugality_index = results_df['frugality_ratio'].mean() if total_patients > 0 else 0
    frugality_std = results_df['frugality_ratio'].std() if total_patients > 0 else 0
    accuracy_std_err = math.sqrt((accuracy * (1 - accuracy)) / total_patients) if total_patients > 0 else 0

    avg_cost_llm = results_df['total_case_cost'].mean()
    avg_ces = results_df['CES'].mean()
    avg_utr = results_df['UTR'].mean()
    avg_fi = results_df['FI_New'].mean()

    print("\n================ Overall Summary ================")
    print(f"MODEL USED: {config.MODEL_NAME}")
    print(f"JUDGE MODEL USED: {config.JUDGE_MODEL_NAME}")
    print(f"- Total Patients Processed: {total_patients}")
    print(f"- Accuracy: {accuracy*100:.2f}% (± {accuracy_std_err*100:.2f}%)")
    print(f"- Frugality Index: {frugality_index:.2f} (± {frugality_std:.2f}) (Lower is better)")
    print(f"- Average Cost of LLM-Ordered Tests: ${avg_cost_llm:.2f}")
    print(f"- Average Cost Efficiency Score (CES): {avg_ces:.2f}")
    print(f"- Average Utilization to True Requirement (UTR): {avg_utr*100:.2f}%")
    print(f"- Average Frugality Index (FI): {avg_fi:.2f}")
    print("==============================================\n")

    # --- Plotting Functions ---
    def plot_metric_per_disease(df, metric_col, metric_label, filename_suffix):
        try:
            plt.figure(figsize=(14, 8))
            grouped = df.groupby('ground_truth_disease')[metric_col].mean().sort_values()
            
            sns.barplot(x=grouped.index, y=grouped.values, palette='viridis')
            plt.title(f'Average {metric_label} by Disease', fontsize=16)
            plt.xlabel('Disease', fontsize=12)
            plt.ylabel(f'Avg {metric_label}', fontsize=12)
            plt.xticks(rotation=45, ha="right")
            
            for i, v in enumerate(grouped.values):
                plt.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
                
            plt.tight_layout()
            plt.savefig(f'disease_{filename_suffix}_{config.MODEL_NAME.replace("/", "_")}.png')
            print(f"Saved plot: disease_{filename_suffix}_{config.MODEL_NAME.replace('/', '_')}.png")
        except Exception as e:
            print(f"Error plotting {metric_label}: {e}")

    # Generate Standard Plots
    plot_metric_per_disease(results_df, 'is_correct', 'Accuracy', 'accuracy')
    plot_metric_per_disease(results_df, 'frugality_ratio', 'Frugality Index', 'frugality')
    plot_metric_per_disease(results_df, 'total_case_cost', 'Total Cost ($)', 'cost')
    plot_metric_per_disease(results_df, 'CES', 'Cost Efficiency Score (CES)', 'ces')
    plot_metric_per_disease(results_df, 'UTR', 'Unnecessary Test Rate (UTR)', 'utr')
    plot_metric_per_disease(results_df, 'FI_New', 'Frugality Index (FI)', 'fi_new')
    # def plot_confusion_matrix(results_df, model_name):
    #     print("Generating confusion matrix...")
    #     y_true = results_df['ground_truth_disease'].fillna("Unknown").astype(str).str.strip().str.lower()
    #     y_pred = results_df['llm_diagnosis'].fillna("Unknown").astype(str).str.strip().str.lower()
    #     labels = sorted(list(set(y_true.unique()) | set(y_pred.unique())))
    #     if not labels:
    #         print("No labels found for confusion matrix. Skipping.")
    #         return
    #     cm = confusion_matrix(y_true, y_pred, labels=labels)
    #     plt.figure(figsize=(12, 10))
    #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
    #                 xticklabels=labels, yticklabels=labels)
    #     plt.title(f'Confusion Matrix ({model_name}) – Counts', fontsize=16)
    #     plt.ylabel('True Label', fontsize=12)
    #     plt.xlabel('Predicted Label', fontsize=12)
    #     plt.xticks(rotation=45, ha="right")
    #     plt.yticks(rotation=0)
    #     plt.tight_layout()
    #     plt.savefig(f'confusion_matrix_counts_{model_name.replace("/", "_")}.png')
    #     print(f"Raw count confusion matrix saved as 'confusion_matrix_counts_{model_name.replace('/', '_')}.png'")
    #     cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     cm_normalized = np.nan_to_num(cm_normalized) 
    #     plt.figure(figsize=(12, 10))
    #     sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues',
    #                 xticklabels=labels, yticklabels=labels)
    #     plt.title(f'Confusion Matrix ({model_name}) – Normalized', fontsize=16)
    #     plt.ylabel('True Label', fontsize=12)
    #     plt.xlabel('Predicted Label', fontsize=12)
    #     plt.xticks(rotation=45, ha="right")
    #     plt.yticks(rotation=0)
    #     plt.tight_layout()
    #     plt.savefig(f'confusion_matrix_normalized_{model_name.replace("/", "_")}.png')
    #     print(f"Normalized confusion matrix saved as 'confusion_matrix_normalized_{model_name.replace('/', '_')}.png'")

    
    # plot_subset = min(20, total_patients)
    # plt.figure(figsize=(12, 7))
    # bar_width = 0.35
    # index = np.arange(plot_subset)
    # plt.bar(index, results_df['num_tests_real'].head(plot_subset), bar_width, label='Real-Life Tests', color='lightcoral')
    # plt.bar(index + bar_width, results_df['num_tests_llm'].head(plot_subset), bar_width, label='LLM Tests', color='skyblue')
    # plt.title(f'Test Comparison for First {plot_subset} Patients ({config.MODEL_NAME})')
    # plt.xlabel('Patient')
    # plt.ylabel('Number of Tests')
    # plt.xticks(index + bar_width / 2, [f"Pt {i+1}" for i in range(plot_subset)], rotation=45, ha="right")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f'test_comparison_{config.MODEL_NAME.replace("/", "_")}.png')
    # print(f"Comparison plot saved as 'test_comparison_{config.MODEL_NAME.replace('/', '_')}.png'")
    
    # print("Generating grouped box plot for test counts...")
    # plot_data = pd.melt(results_df, 
    #                     id_vars=['ground_truth_disease'], 
    #                     value_vars=['num_tests_real', 'num_tests_llm'], 
    #                     var_name='Test_Type', 
    #                     value_name='Number_of_Tests')
    # plot_data['Test_Type'] = plot_data['Test_Type'].map({'num_tests_real': 'Real-Life', 'num_tests_llm': 'LLM'})
    # plt.figure(figsize=(16, 9))
    # sns.boxplot(x='ground_truth_disease', y='Number_of_Tests', hue='Test_Type', data=plot_data, palette="Set2")
    # plt.title(f'Distribution of Tests Ordered per Disease ({config.MODEL_NAME})', fontsize=16)
    # plt.xlabel('Ground Truth Disease', fontsize=12)
    # plt.ylabel('Number of Tests Ordered', fontsize=12)
    # plt.xticks(rotation=45, ha="right")
    # plt.legend(title='Test Type')
    # plt.tight_layout()
    # plt.savefig(f'test_distribution_by_disease_{config.MODEL_NAME.replace("/", "_")}.png')
    # print(f"Test distribution plot saved as 'test_distribution_by_disease_{config.MODEL_NAME.replace('/', '_')}.png'")
    
    # disease_metrics = results_df.groupby('ground_truth_disease').agg(
    #     accuracy=('is_correct', 'mean'),
    #     frugality=('frugality_ratio', 'mean'),
    #     count=('ground_truth_disease', 'size')
    # ).reset_index()
    # disease_metrics = disease_metrics[disease_metrics['count'] > 0].sort_values(by='count', ascending=False)
    # disease_metrics['plot_label'] = disease_metrics['ground_truth_disease'] + " (n=" + disease_metrics['count'].astype(str) + ")"
    
    # plt.figure(figsize=(14, 8))
    # plt.bar(disease_metrics['plot_label'], disease_metrics['accuracy'], color='mediumseagreen')
    # plt.xlabel('Ground Truth Disease (with patient count)')
    # plt.ylabel('Average Accuracy')
    # plt.title(f'Average Accuracy per Disease ({config.MODEL_NAME})')
    # plt.xticks(rotation=45, ha="right")
    # plt.ylim(0, 1.05)
    # plt.tight_layout()
    # plt.savefig(f'accuracy_per_disease_{config.MODEL_NAME.replace("/", "_")}.png')
    # print(f"Accuracy per disease plot saved as 'accuracy_per_disease_{config.MODEL_NAME.replace('/', '_')}.png'")
    
    # plt.figure(figsize=(14, 8))
    # plt.bar(disease_metrics['plot_label'], disease_metrics['frugality'], color='mediumpurple')
    # plt.xlabel('Ground Truth Disease (with patient count)')
    # plt.ylabel('Average Frugality Index')
    # plt.title(f'Average Frugality per Disease ({config.MODEL_NAME})')
    # plt.xticks(rotation=45, ha="right")
    # plt.tight_layout()
    # plt.savefig(f'frugality_per_disease_{config.MODEL_NAME.replace("/", "_")}.png')
    # print(f"Frugality per disease plot saved as 'frugality_per_disease_{config.MODEL_NAME.replace('/', '_')}.png'")

    # plot_confusion_matrix(results_df, config.MODEL_NAME)

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())