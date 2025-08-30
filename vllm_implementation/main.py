# # main.py

# import pandas as pd
# import numpy as np
# import math
# import os
# import matplotlib.pyplot as plt
# from config import PATIENTS_FILE_PATH, TEST_LABELS_FILE_PATH, ALLOWED_RADIOLOGY_TESTS, MODEL_NAME
# from data_loader import read_csv_file
# from utils import (
#     are_diagnoses_related, 
#     estimate_real_life_tests, 
#     generate_patient_scenario,
#     get_test_results
# )
# from llm_service import (
#     get_differential_diagnoses, 
#     get_test_suggestions, 
#     get_final_decision
# )

# os.environ['HF_HOME'] = "/scratch/nshanb2/cache_custom/models"
# os.environ['HF_TOKEN'] = "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"

# def frugal_llm_workflow(record, label_to_code, available_tests, max_iterations=3):
#     """Processes a single patient record through the frugal diagnostic workflow."""
#     scenario = generate_patient_scenario(record)
#     print(f"### Step 1: Patient Scenario\n{scenario}\n")
#     diagnoses_list = get_differential_diagnoses(scenario)
#     print(f"### Step 2: Differential Diagnosis\n{', '.join(diagnoses_list)}\n")
#     all_suggested_tests, test_results_str = set(), ""
#     final_diagnosis_text = "Unable to reach a final diagnosis."

#     for iteration in range(max_iterations):
#         suggested_tests = get_test_suggestions(scenario, diagnoses_list, available_tests, ALLOWED_RADIOLOGY_TESTS, test_results_str)
#         new_tests = [test for test in suggested_tests if test not in all_suggested_tests]
#         all_suggested_tests.update(new_tests)
#         print(f"### Step 3.{iteration+1}: Suggested Tests\n{', '.join(new_tests)}\n")

#         if not new_tests:
#             print("No new tests suggested. Moving to final diagnosis.\n")
#             final_response = get_final_decision(scenario, diagnoses_list, test_results_str)
#             if "final diagnosis:" in final_response.lower():
#                 final_diagnosis_text = final_response.split(":", 1)[1].strip()
#             break

#         test_results = get_test_results(record, new_tests, label_to_code, available_tests)
#         current_results_str = "\n".join([f"- {k}: {v}" for k, v in test_results.items()])
#         test_results_str += current_results_str + "\n"
#         print(f"### Step 4.{iteration+1}: Test Results\n{current_results_str}\n")
        
#         final_response = get_final_decision(scenario, diagnoses_list, test_results_str)
#         print(f"### Step 5.{iteration+1}: LLM Decision\n{final_response}\n")

#         if "final diagnosis:" in final_response.lower():
#             final_diagnosis_text = final_response.split(":", 1)[1].strip()
#             break
    
#     actual_diagnosis = record['Discharge Diagnosis']
#     is_correct = are_diagnoses_related(final_diagnosis_text, actual_diagnosis)
#     num_tests_llm = len(all_suggested_tests)
#     num_tests_real = estimate_real_life_tests(record)
#     frugality_ratio = num_tests_llm / num_tests_real if num_tests_real > 0 else 0

#     print(f"### Step 6: Evaluation\n"
#           f"- Actual Diagnosis: {actual_diagnosis}\n"
#           f"- LLM Diagnosis: {final_diagnosis_text}\n"
#           f"- Correct Diagnosis: {is_correct}\n"
#           f"- Tests Ordered by LLM: {num_tests_llm}\n"
#           f"- Tests in Real Life: {num_tests_real}\n"
#           f"- Frugality Ratio: {frugality_ratio:.2f}\n"
#           f"---------------------------------------------------\n")

#     return {
#         'actual_diagnosis': actual_diagnosis,
#         'is_correct': is_correct,
#         'num_tests_llm': num_tests_llm,
#         'num_tests_real': num_tests_real,
#         'frugality_ratio': frugality_ratio
#     }

# def main():
#     """Main function to run the simulation and analysis."""
#     df_patients = read_csv_file(PATIENTS_FILE_PATH)
#     df_test_labels = read_csv_file(TEST_LABELS_FILE_PATH)
#     label_to_code = dict(zip(df_test_labels['lab_label'], df_test_labels['Test Code']))
#     available_tests = df_test_labels['lab_label'].tolist()

#     results = []
#     # Updated to process 1000 patients
#     num_patients_to_process = 5
#     # Ensure we don't try to process more patients than exist in the file
#     num_to_run = min(num_patients_to_process, len(df_patients))

#     for idx, record in df_patients.head(num_to_run).iterrows():
#         print(f"================ Processing Patient {idx + 1}/{num_to_run} ================\n")
#         try:
#             result = frugal_llm_workflow(record, label_to_code, available_tests)
#             results.append(result)
#         except Exception as e:
#             print(f"!!!!!! An error occurred processing Patient {idx + 1}: {e} !!!!!!")
#             results.append({'actual_diagnosis': 'ERROR', 'is_correct': False, 'num_tests_llm': 0, 'num_tests_real': 0, 'frugality_ratio': 0})

#     if not results:
#         print("No results to analyze. Exiting.")
#         return

#     results_df = pd.DataFrame(results)
#     total_patients = len(results_df)
#     accuracy = results_df['is_correct'].mean()
#     frugality_index = results_df['frugality_ratio'].mean()
#     frugality_std = results_df['frugality_ratio'].std()
#     accuracy_std_err = math.sqrt((accuracy * (1 - accuracy)) / total_patients) if total_patients > 0 else 0

#     print("\n================ Overall Summary ================")
#     print(f"MODEL USED: {MODEL_NAME}")
#     print(f"- Total Patients Processed: {total_patients}")
#     print(f"- Accuracy: {accuracy*100:.2f}% (± {accuracy_std_err*100:.2f}%)")
#     print(f"- Frugality Index: {frugality_index:.2f} (± {frugality_std:.2f}) (Lower is better)")
#     print("==============================================\n")
    
#     # (Plotting code remains the same as previous version)
#     # Plot 1: Overall Test Comparison (Plotting all 1000 would be unreadable, so we plot the first 20 for visualization)
#     plot_subset = min(20, total_patients)
#     plt.figure(figsize=(12, 7))
#     bar_width = 0.35
#     index = np.arange(plot_subset)
#     plt.bar(index, results_df['num_tests_real'].head(plot_subset), bar_width, label='Real-Life Tests', color='lightcoral')
#     plt.bar(index + bar_width, results_df['num_tests_llm'].head(plot_subset), bar_width, label='LLM Tests', color='skyblue')
#     plt.xlabel('Patient')
#     plt.ylabel('Number of Tests')
#     plt.title(f'Test Comparison for First {plot_subset} Patients ({MODEL_NAME})')
#     plt.xticks(index + bar_width / 2, [f"Pt {i+1}" for i in range(plot_subset)], rotation=45, ha="right")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'test_comparison_{MODEL_NAME.replace("/", "_")}.png')
#     print(f"Comparison plot saved as 'test_comparison_{MODEL_NAME.replace('/', '_')}.png'")

#     # --- Group by disease for new plots ---
#     disease_metrics = results_df.groupby('actual_diagnosis').agg(
#         accuracy=('is_correct', 'mean'),
#         frugality=('frugality_ratio', 'mean'),
#         count=('actual_diagnosis', 'size')
#     ).reset_index()
#     disease_metrics = disease_metrics[disease_metrics['count'] > 0]

#     # Plot 2: Accuracy per Disease
#     plt.figure(figsize=(12, 8))
#     plt.bar(disease_metrics['actual_diagnosis'], disease_metrics['accuracy'], color='mediumseagreen')
#     plt.xlabel('Disease')
#     plt.ylabel('Average Accuracy')
#     plt.title(f'Average Accuracy per Disease ({MODEL_NAME})')
#     plt.xticks(rotation=45, ha="right")
#     plt.ylim(0, 1)
#     plt.tight_layout()
#     plt.savefig(f'accuracy_per_disease_{MODEL_NAME.replace("/", "_")}.png')
#     print(f"Accuracy per disease plot saved as 'accuracy_per_disease_{MODEL_NAME.replace('/', '_')}.png'")

#     # Plot 3: Frugality per Disease
#     plt.figure(figsize=(12, 8))
#     plt.bar(disease_metrics['actual_diagnosis'], disease_metrics['frugality'], color='mediumpurple')
#     plt.xlabel('Disease')
#     plt.ylabel('Average Frugality Index')
#     plt.title(f'Average Frugality per Disease ({MODEL_NAME})')
#     plt.xticks(rotation=45, ha="right")
#     plt.tight_layout()
#     plt.savefig(f'frugality_per_disease_{MODEL_NAME.replace("/", "_")}.png')
#     print(f"Frugality per disease plot saved as 'frugality_per_disease_{MODEL_NAME.replace('/', '_')}.png'")

# if __name__ == "__main__":
#     main()

# main.py

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import asyncio
import time
from config import PATIENTS_FILE_PATH, TEST_LABELS_FILE_PATH, ALLOWED_RADIOLOGY_TESTS, MODEL_NAME, CHUNK_SIZE
from data_loader import read_csv_file
from utils import (
    are_diagnoses_related,
    estimate_real_life_tests,
    generate_patient_scenario,
    get_test_results
)
from llm_service import (
    get_differential_diagnoses,
    get_test_suggestions,
    get_final_decision
)

# This async workflow now includes the full, original logic
async def frugal_llm_workflow_async(record, record_idx, label_to_code, available_tests, max_iterations=3):
    """Asynchronously processes a single patient record with full original logic."""
    scenario = generate_patient_scenario(record)
    print(f"### Step 1: Patient Scenario\n{scenario}\n")
    
    diagnoses_list = await get_differential_diagnoses(scenario)
    print(f"### Step 2: Differential Diagnosis\n{', '.join(diagnoses_list)}\n")

    all_suggested_tests = [] # Using a list to preserve order, will use set() for counting later
    test_results_str = ""
    final_diagnosis_text = "Unable to reach a final diagnosis." # Default value

    for iteration in range(max_iterations):
        # Get new suggestions based on all prior information
        suggested_tests = await get_test_suggestions(scenario, diagnoses_list, available_tests, ALLOWED_RADIOLOGY_TESTS, test_results_str)
        
        # Filter out tests that have already been suggested
        new_tests = [test for test in suggested_tests if test not in all_suggested_tests]
        all_suggested_tests.extend(new_tests)
        print(f"### Step 3.{iteration+1}: Suggested Tests\n{', '.join(new_tests)}\n")
        if not new_tests:
            print("No new tests suggested. Moving to final diagnosis.\n")
            # If no new tests are suggested, we should ask for a final diagnosis
            final_response = await get_final_decision(scenario, diagnoses_list, test_results_str)
            if "final diagnosis:" in final_response.lower():
                final_diagnosis_text = final_response.split(":", 1)[-1].strip().split('.')[0]
            break

        # Get results only for the newly suggested tests
        test_results = get_test_results(record, new_tests, label_to_code, available_tests)
        test_results_str += "\n".join([f"- {k}: {v}" for k, v in test_results.items()]) + "\n"
        print(f"### Step 4.{iteration+1}: Test Results Found\n")

        # Make a decision based on the cumulative results
        final_response = await get_final_decision(scenario, diagnoses_list, test_results_str)
        print(f"### Step 5.{iteration+1}: LLM Decision\n{final_response}\n")

        # --- RESTORED LOGIC BLOCK ---
        if "final diagnosis" in final_response.lower():
            # Refined diagnosis parsing to get only the first sentence
            final_diagnosis_text = final_response.split(":", 1)[-1].strip().split('.')[0]
            # final_diagnosis_text = final_response.split(":", 1)[1].strip()
            break
        # elif iteration == max_iterations - 1:
        #     # Fallback if max iterations are reached without a diagnosis
        #     final_diagnosis_text = "Unable to reach a final diagnosis."
        elif "additional tests" in final_response.lower():
            additional_tests_str = final_response.split(":", 1)[-1].strip()
            additional_tests = [test.strip() for test in additional_tests_str.split(',')]
            all_suggested_tests.extend(additional_tests)
            print(f"--- LLM requested additional tests: {', '.join(additional_tests)} ---\n")
        # --- END OF RESTORED LOGIC ---

    actual_diagnosis = record['Discharge Diagnosis']
    is_correct = are_diagnoses_related(final_diagnosis_text, actual_diagnosis)
    
    # Use set() here to get the count of unique tests
    num_tests_llm = len(set(all_suggested_tests))
    num_tests_real = estimate_real_life_tests(record)
    frugality_ratio = num_tests_llm / num_tests_real if num_tests_real > 0 else 0

    print(f"### Step 6: Evaluation for Patient {record['Patient ID']}\n"
          f"- Patient Id: {record['Patient ID']}\n"
          f"- Actual Diagnosis: {actual_diagnosis}\n"
          f"- LLM Diagnosis: {final_diagnosis_text}\n"
          f"- Correct Diagnosis: {is_correct}\n"
          f"- Tests Ordered by LLM: {num_tests_llm}\n"
          f"- Tests in Real Life: {num_tests_real}\n"
          f"- Frugality Ratio: {frugality_ratio:.2f}\n"
          f"---------------------------------------------------\n")

    return {
        'patient_id': record['Patient ID'],
        'actual_diagnosis': actual_diagnosis,
        'llm_diagnosis': final_diagnosis_text,
        'is_correct': is_correct,
        'num_tests_llm': num_tests_llm,
        'num_tests_real': num_tests_real,
        'frugality_ratio': frugality_ratio
    }

async def main():
    """Main function to run the simulation and analysis asynchronously."""
    start_time = time.time()

    df_patients = read_csv_file(PATIENTS_FILE_PATH)
    df_test_labels = read_csv_file(TEST_LABELS_FILE_PATH)
    label_to_code = dict(zip(df_test_labels['lab_label'], df_test_labels['Test Code']))
    available_tests = df_test_labels['lab_label'].tolist()

    # print(f"Columns in patient data: {df_patients.columns.tolist()}")
    # Print the first few rows to verify data loading
    # print("First few rows of patient data:")
    # print(df_patients.head())

    mask = df_patients['Patient ID'].str.contains('_', regex=False, na=False)

    # **CHANGE**: Split by '_' instead of ' + '
    df_patients.loc[mask, 'ground_truth_disease'] = df_patients.loc[mask, 'Patient ID'].str.split('_').str[1].str.strip()
    print("Standardized ground truth disease column created successfully.")
    # print(df_patients[['Patient ID', 'ground_truth_disease']].head(10)) # Print first 10 for verification


    num_patients_to_process = 2
    num_to_run = min(num_patients_to_process, len(df_patients))

    # # Create tasks for concurrent processing and include chunking
    # tasks = []
    # for idx, record in df_patients.head(num_to_run).iterrows():
    #     tasks.append(frugal_llm_workflow_async(record, idx, label_to_code, available_tests))

    # print(f"Starting concurrent processing for {len(tasks)} patients...")
    # results = await asyncio.gather(*tasks, return_exceptions=True)

    # successful_results = [res for res in results if not isinstance(res, Exception)]
    # errors = [res for res in results if isinstance(res, Exception)]
    # if errors:
    #     print(f"\nEncountered {len(errors)} errors during processing.")
    #     for err in errors[:5]: # Print first 5 errors for debugging
    #         print(err)

    # if not successful_results:
    #     print("No successful results to analyze. Exiting.")
    #     return
     # --- CHUNKING LOGIC ---
    all_results = []
    total_chunks = (num_to_run + CHUNK_SIZE - 1) // CHUNK_SIZE
    for i in range(0, num_to_run, CHUNK_SIZE):
        chunk_df = df_patients.iloc[i:i + CHUNK_SIZE]
        print(f"\n--- Processing chunk {i // CHUNK_SIZE + 1}/{total_chunks} (Patients {i+1} to {i + len(chunk_df)}) ---\n")

        tasks = []
        for idx, record in chunk_df.iterrows():
            tasks.append(frugal_llm_workflow_async(record, idx, label_to_code, available_tests))

        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
        all_results.extend(chunk_results)
        
        successful_count = len([res for res in all_results if not isinstance(res, Exception)])
        print(f"--- Finished chunk. Total successful patients processed: {successful_count} ---")
    # --- END OF CHUNKING LOGIC ---

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
    # Save the results to a CSV file for further analysis
    results_df.to_csv(f'frugal_llm_results_{MODEL_NAME.replace("/", "_")}.csv', index=False)
    print(f"Results saved to 'frugal_llm_results_{MODEL_NAME.replace('/', '_')}.csv'")

    # results_df = pd.merge(results_df, df_patients[['ground_truth_disease']], left_on='patient_index', right_index=True)
    # Merge based on Patient ID  instead to ensure correct alignment. Don't include patient index.
    results_df = pd.merge(results_df, df_patients[['ground_truth_disease']], on='patient_id', how='left')

    total_patients = len(results_df)
    accuracy = results_df['is_correct'].mean() if total_patients > 0 else 0
    frugality_index = results_df['frugality_ratio'].mean() if total_patients > 0 else 0
    frugality_std = results_df['frugality_ratio'].std() if total_patients > 0 else 0
    accuracy_std_err = math.sqrt((accuracy * (1 - accuracy)) / total_patients) if total_patients > 0 else 0

    print("\n================ Overall Summary ================")
    # ... (rest of the main function, including analysis and plotting, remains the same) ...
    print(f"MODEL USED: {MODEL_NAME}")
    print(f"- Total Patients Processed: {total_patients}")
    print(f"- Accuracy: {accuracy*100:.2f}% (± {accuracy_std_err*100:.2f}%)")
    print(f"- Frugality Index: {frugality_index:.2f} (± {frugality_std:.2f}) (Lower is better)")
    print("==============================================\n")
    
    plot_subset = min(20, total_patients)
    plt.figure(figsize=(12, 7))
    bar_width = 0.35
    index = np.arange(plot_subset)
    plt.bar(index, results_df['num_tests_real'].head(plot_subset), bar_width, label='Real-Life Tests', color='lightcoral')
    plt.bar(index + bar_width, results_df['num_tests_llm'].head(plot_subset), bar_width, label='LLM Tests', color='skyblue')
    plt.title(f'Test Comparison for First {plot_subset} Patients ({MODEL_NAME})')
    plt.xlabel('Patient')
    plt.ylabel('Number of Tests')
    plt.xticks(index + bar_width / 2, [f"Pt {i+1}" for i in range(plot_subset)], rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'test_comparison_{MODEL_NAME.replace("/", "_")}.png')
    print(f"Comparison plot saved as 'test_comparison_{MODEL_NAME.replace('/', '_')}.png'")

    disease_metrics = results_df.groupby('ground_truth_disease').agg(
        accuracy=('is_correct', 'mean'),
        frugality=('frugality_ratio', 'mean'),
        count=('ground_truth_disease', 'size')
    ).reset_index()
    disease_metrics = disease_metrics[disease_metrics['count'] > 0].sort_values(by='count', ascending=False)

    plt.figure(figsize=(12, 8))
    plt.bar(disease_metrics['ground_truth_disease'], disease_metrics['accuracy'], color='mediumseagreen')
    plt.xlabel('Ground Truth Disease')
    plt.ylabel('Average Accuracy')
    plt.title(f'Average Accuracy per Disease ({MODEL_NAME})')
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f'accuracy_per_disease_{MODEL_NAME.replace("/", "_")}.png')
    print(f"Accuracy per disease plot saved as 'accuracy_per_disease_{MODEL_NAME.replace('/', '_')}.png'")

    plt.figure(figsize=(12, 8))
    plt.bar(disease_metrics['ground_truth_disease'], disease_metrics['frugality'], color='mediumpurple')
    plt.xlabel('Ground Truth Disease')
    plt.ylabel('Average Frugality Index')
    plt.title(f'Average Frugality per Disease ({MODEL_NAME})')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f'frugality_per_disease_{MODEL_NAME.replace("/", "_")}.png')
    print(f"Frugality per disease plot saved as 'frugality_per_disease_{MODEL_NAME.replace('/', '_')}.png'")

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())