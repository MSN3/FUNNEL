import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from config import PATIENTS_FILE_PATH, TEST_LABELS_FILE_PATH, ALLOWED_RADIOLOGY_TESTS, MODEL_NAME
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

def frugal_llm_workflow(record, label_to_code, available_tests, max_iterations=3):
    """Processes a single patient record through the frugal diagnostic workflow."""
    scenario = generate_patient_scenario(record)
    print(f"### Step 1: Patient Scenario\n{scenario}\n")

    diagnoses_list = get_differential_diagnoses(scenario)
    print(f"### Step 2: Differential Diagnosis\n{', '.join(diagnoses_list)}\n")

    all_suggested_tests = set()
    test_results_str = ""
    final_diagnosis_text = "Unable to reach a final diagnosis."

    for iteration in range(max_iterations):
        suggested_tests = get_test_suggestions(scenario, diagnoses_list, available_tests, ALLOWED_RADIOLOGY_TESTS, test_results_str)
        new_tests = [test for test in suggested_tests if test not in all_suggested_tests]
        all_suggested_tests.update(new_tests)

        print(f"### Step 3.{iteration+1}: Suggested Tests\n{', '.join(new_tests)}\n")

        if not new_tests:
            print("No new tests suggested. Moving to final diagnosis.\n")
            # If no new tests are suggested, we should ask for a final diagnosis
            final_response = get_final_decision(scenario, diagnoses_list, test_results_str)
            if "final diagnosis:" in final_response.lower():
                final_diagnosis_text = final_response.split(":", 1)[1].strip()
            break

        test_results = get_test_results(record, new_tests, label_to_code, available_tests)
        current_results_str = "\n".join([f"- {k}: {v}" for k, v in test_results.items()])
        test_results_str += current_results_str + "\n"
        print(f"### Step 4.{iteration+1}: Test Results\n{current_results_str}\n")

        final_response = get_final_decision(scenario, diagnoses_list, test_results_str)
        print(f"### Step 5.{iteration+1}: LLM Decision\n{final_response}\n")

        if "final diagnosis:" in final_response.lower():
            final_diagnosis_text = final_response.split(":", 1)[1].strip()
            break
        # elif "additional tests:" in final_response.lower():
        #     additional_tests = final_response.split("Additional Tests:", 1)[1].strip()
        #     all_suggested_tests.update([test.strip() for test in additional_tests.split(",")])

    actual_diagnosis = record['Discharge Diagnosis']
    is_correct = are_diagnoses_related(final_diagnosis_text, actual_diagnosis)
    num_tests_llm = len(all_suggested_tests)
    num_tests_real = estimate_real_life_tests(record)
    frugality_ratio = num_tests_llm / num_tests_real if num_tests_real > 0 else 0

    print(f"### Step 6: Evaluation\n"
          f"- Actual Diagnosis: {actual_diagnosis}\n"
          f"- LLM Diagnosis: {final_diagnosis_text}\n"
          f"- Correct Diagnosis: {is_correct}\n"
          f"- Tests Ordered by LLM: {num_tests_llm}\n"
          f"- Tests in Real Life: {num_tests_real}\n"
          f"- Frugality Ratio: {frugality_ratio:.2f}\n"
          f"---------------------------------------------------\n")

    return {
        'final_diagnosis': final_diagnosis_text,
        'actual_diagnosis': actual_diagnosis,
        'is_correct': is_correct,
        'num_tests_llm': num_tests_llm,
        'num_tests_real': num_tests_real,
        'frugality_ratio': frugality_ratio
    }

def main():
    """Main function to run the simulation and analysis."""
    df_patients = read_csv_file(PATIENTS_FILE_PATH)
    df_test_labels = read_csv_file(TEST_LABELS_FILE_PATH)
    label_to_code = dict(zip(df_test_labels['lab_label'], df_test_labels['Test Code']))
    available_tests = df_test_labels['lab_label'].tolist()

    mask = df_patients['Patient ID'].str.contains('_', regex=False, na=False)

    # **CHANGE**: Split by '_' instead of ' + '
    df_patients.loc[mask, 'ground_truth_disease'] = df_patients.loc[mask, 'Patient ID'].str.split('_').str[1].str.strip()
    print("Standardized ground truth disease column created successfully.")

    results = []
    # Set to 20 to process all your cases, or use len(df_patients) for the whole file
    num_patients_to_process = 50 #len(df_patients)
    for idx, record in df_patients.head(num_patients_to_process).iterrows():
        print(f"================ Processing Patient {idx + 1} ================\n")
        try:
            result = frugal_llm_workflow(record, label_to_code, available_tests)
            results.append(result)
        except Exception as e:
            print(f"!!!!!! An error occurred processing Patient {idx + 1}: {e} !!!!!!")
            # Optionally, append a failed result to keep counts accurate
            results.append({
                'final_diagnosis': 'Error',
                'actual_diagnosis': 'Error',
                'is_correct': False,
                'num_tests_llm': 0,
                'num_tests_real': 0,
                'frugality_ratio': 0
            })


    #Save the results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'frugal_llm_results.csv', index=False)

    results_df = pd.merge(results_df, df_patients[['ground_truth_disease']], left_on='patient_index', right_index=True)
    
    # total_patients = len(results)
    # correct_diagnoses = sum(res['is_correct'] for res in results)
    # accuracy = (correct_diagnoses / total_patients * 100) if total_patients > 0 else 0
    # avg_tests_llm = np.mean([res['num_tests_llm'] for res in results])
    # avg_tests_real = np.mean([res['num_tests_real'] for res in results])
    # frugality_index = np.mean([res['frugality_ratio'] for res in results])

    # print("\n================ Overall Summary ================")
    # print(f"MODEL USED: {MODEL_NAME}")
    # print(f"- Total Patients Processed: {total_patients}")
    # print(f"- Correct Diagnoses: {correct_diagnoses}")
    # print(f"- Accuracy: {accuracy:.2f}%")
    # print(f"- Average Tests Ordered by LLM: {avg_tests_llm:.2f}")
    # print(f"- Average Tests in Real Life: {avg_tests_real:.2f}")
    # print(f"- Frugality Index: {frugality_index:.2f} (Lower is better)")
    # print("==============================================\n")
    total_patients = len(results_df)
    accuracy = results_df['is_correct'].mean() if total_patients > 0 else 0
    correct_diagnoses = sum(res['is_correct'] for res in results_df)
    frugality_index = results_df['frugality_ratio'].mean() if total_patients > 0 else 0
    frugality_std = results_df['frugality_ratio'].std() if total_patients > 0 else 0
    accuracy_std_err = math.sqrt((accuracy * (1 - accuracy)) / total_patients) if total_patients > 0 else 0

    print("\n================ Overall Summary ================")
    # ... (rest of the main function, including analysis and plotting, remains the same) ...
    print(f"MODEL USED: {MODEL_NAME}")
    print(f"- Total Patients Processed: {total_patients}")
    print(f"- Correct Diagnoses: {correct_diagnoses}")
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
    plt.savefig(f'test_comparison.png')
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
    plt.savefig(f'accuracy_per_disease.png')
    print(f"Accuracy per disease plot saved as 'accuracy_per_disease.png'")

    plt.figure(figsize=(12, 8))
    plt.bar(disease_metrics['ground_truth_disease'], disease_metrics['frugality'], color='mediumpurple')
    plt.xlabel('Ground Truth Disease')
    plt.ylabel('Average Frugality Index')
    plt.title(f'Average Frugality per Disease ({MODEL_NAME})')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f'frugality_per_disease.png')
    print(f"Frugality per disease plot saved as 'frugality_per_disease.png'")

if __name__ == "__main__":
    main()