# evaluator.py
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import io
import os

def plot_confusion_matrix(results_df, model_name):
    """Generates and returns confusion matrix plots (with case-normalized labels)."""
    print("Generating confusion matrix...")
    y_true = results_df['ground_truth_disease'].fillna("Unknown").astype(str).str.strip().str.lower()
    y_pred = results_df['llm_diagnosis'].fillna("Unknown").astype(str).str.strip().str.lower()
    labels = sorted(list(set(y_true.unique()) | set(y_pred.unique())))
    
    if len(labels) == 0:
        print("No labels for confusion matrix. Skipping.")
        return {}
        
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plots = {}
    plt.ioff() # Turn off interactive plotting

    # --- Raw counts ---
    fig_counts, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f'Confusion Matrix ({model_name}) – Counts', fontsize=16)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig_counts.tight_layout()
    plots["confusion_matrix_counts"] = fig_counts

    # --- Normalized (percentages per true class) ---
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized) # Replace nan with 0

    fig_norm, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f'Confusion Matrix ({model_name}) – Normalized', fontsize=16)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig_norm.tight_layout()
    plots["confusion_matrix_normalized"] = fig_norm
    
    return plots

def run_evaluation_and_plotting(results_df: pd.DataFrame, model_name: str):
    """
    Takes the final results dataframe, calculates metrics,
    and generates all plots.
    Returns (summary_metrics_dict, plots_dict)
    """
    plt.ioff() # Turn off interactive plotting
    
    # --- 1. Save CSV ---
    # Create output directory if it doesn't exist
    output_dir = "./outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model_name_safe = model_name.replace("/", "_")
    output_csv_path = os.path.join(output_dir, f'frugal_llm_results_{model_name_safe}.csv')
    results_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to '{output_csv_path}'")

    # --- 2. Calculate Overall Metrics ---
    total_patients = len(results_df)
    if total_patients == 0:
        return {"error": "No results to analyze"}, {}
        
    accuracy = results_df['is_correct'].mean()
    frugality_index = results_df['frugality_ratio'].mean()
    frugality_std = results_df['frugality_ratio'].std()
    accuracy_std_err = math.sqrt((accuracy * (1 - accuracy)) / total_patients) if total_patients > 0 else 0
    # Token metrics
    avg_tokens = results_df['total_tokens'].mean()
    total_tokens_consumed = results_df['total_tokens'].sum()
    
    # Unnecessary test rate
    results_df['unnecessary_test_rate'] = np.where(
        results_df['initial_test_count'] > 0,
        (results_df['initial_test_count'] - results_df['final_test_count']) / results_df['initial_test_count'],
        0.0
    )
    avg_unnecessary_rate = results_df['unnecessary_test_rate'].mean()

    summary_metrics = {
        "MODEL_USED": model_name,
        "Total_Patients_Processed": total_patients,
        "Accuracy_Percent": f"{accuracy*100:.2f}% (± {accuracy_std_err*100:.2f}%)",
        "Frugality_Index": f"{frugality_index:.2f} (± {frugality_std:.2f}) (Lower is better)",
        "Avg_Confidence": f"{results_df['model_confidence'].mean():.2f}",
        "Avg_Tokens_Per_Patient": f"{avg_tokens:.0f}",
        "Total_Tokens_Run": f"{total_tokens_consumed:.0f}",
        "Avg_Unnecessary_Test_Rate": f"{avg_unnecessary_rate:.2%}",
        "Avg_CES": f"{results_df['CES'].mean():.2f}",
        "Avg_UTR": f"{results_df['UTR'].mean():.2%}",
        "Avg_Cost": f"${results_df['total_case_cost'].mean():.2f}",
        "Avg_Steps": f"{results_df['StD'].mean():.1f}"
        
    }
    print(f"Overall Summary: {summary_metrics}")

    # --- 3. Generate Plots ---
    all_plots = {}
    
    # Plot 1: Test Comparison Bar Chart
    plot_subset = min(20, total_patients)
    fig_bar, ax = plt.subplots(figsize=(12, 7))
    bar_width = 0.35
    index = np.arange(plot_subset)
    ax.bar(index, results_df['num_tests_real'].head(plot_subset), bar_width, label='Real-Life Tests', color='lightcoral')
    ax.bar(index + bar_width, results_df['num_tests_llm'].head(plot_subset), bar_width, label='LLM Tests', color='skyblue')
    ax.set_title(f'Test Comparison for First {plot_subset} Patients ({model_name})')
    ax.set_xlabel('Patient')
    ax.set_ylabel('Number of Tests')
    ax.set_xticks(index + bar_width / 2, [f"Pt {i+1}" for i in range(plot_subset)], rotation=45, ha="right")
    ax.legend()
    fig_bar.tight_layout()
    all_plots["test_comparison_bar"] = fig_bar

    # Plot 2: Grouped box plot for test counts per disease
    plot_data = pd.melt(results_df, 
                        id_vars=['ground_truth_disease'], 
                        value_vars=['num_tests_real', 'num_tests_llm'], 
                        var_name='Test_Type', 
                        value_name='Number_of_Tests')
    plot_data['Test_Type'] = plot_data['Test_Type'].map({'num_tests_real': 'Real-Life', 'num_tests_llm': 'LLM'})

    fig_box, ax = plt.subplots(figsize=(16, 9))
    sns.boxplot(x='ground_truth_disease', y='Number_of_Tests', hue='Test_Type', data=plot_data, palette="Set2", ax=ax)
    ax.set_title(f'Distribution of Tests Ordered per Disease ({model_name})', fontsize=16)
    ax.set_xlabel('Ground Truth Disease', fontsize=12)
    ax.set_ylabel('Number of Tests Ordered', fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.legend(title='Test Type')
    fig_box.tight_layout()
    all_plots["test_distribution_boxplot"] = fig_box
    
    # --- Per-Disease Metrics ---
    disease_metrics = results_df.groupby('ground_truth_disease').agg(
        accuracy=('is_correct', 'mean'),
        frugality=('frugality_ratio', 'mean'),
        count=('ground_truth_disease', 'size')
    ).reset_index()
    disease_metrics = disease_metrics[disease_metrics['count'] > 0].sort_values(by='count', ascending=False)
    disease_metrics['plot_label'] = disease_metrics['ground_truth_disease'] + " (n=" + disease_metrics['count'].astype(str) + ")"

    # Plot 3: Accuracy per Disease
    fig_acc_disease, ax = plt.subplots(figsize=(14, 8))
    ax.bar(disease_metrics['plot_label'], disease_metrics['accuracy'], color='mediumseagreen')
    ax.set_xlabel('Ground Truth Disease (with patient count)')
    ax.set_ylabel('Average Accuracy')
    ax.set_title(f'Average Accuracy per Disease ({model_name})')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 1.05)
    fig_acc_disease.tight_layout()
    all_plots["accuracy_per_disease_bar"] = fig_acc_disease

    # Plot 4: Frugality per Disease
    fig_frugal_disease, ax = plt.subplots(figsize=(14, 8))
    ax.bar(disease_metrics['plot_label'], disease_metrics['frugality'], color='mediumpurple')
    ax.set_xlabel('Ground Truth Disease (with patient count)')
    ax.set_ylabel('Average Frugality Index')
    ax.set_title(f'Average Frugality per Disease ({model_name})')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig_frugal_disease.tight_layout()
    all_plots["frugality_per_disease_bar"] = fig_frugal_disease
    
    # Bin frugality ratio into categories
    results_df['frugality_bin'] = pd.cut(
        results_df['frugality_ratio'], 
        bins=[-0.1, 0.5, 1.0, 5.0], 
        labels=['Frugal (<0.5)', 'Normal (0.5-1.0)', 'Expensive (>1.0)']
    )
    fig_acc_frugal, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='frugality_bin', y='is_correct', hue='frugality_bin', data=results_df, ax=ax, palette='viridis', errorbar=None, legend=False)
    ax.set_title('Accuracy vs Frugality Level')
    ax.set_ylabel('Average Accuracy')
    ax.set_xlabel('Frugality Ratio (LLM Tests / Real Tests)')
    all_plots["accuracy_vs_frugality"] = fig_acc_frugal
    
    # Plot 3: Loop Count vs Accuracy (NEW)
    # fig_loop_acc, ax = plt.subplots(figsize=(10, 6))
    # # Aggregate to get mean accuracy per loop count
    # loop_data = results_df.groupby('final_loop_count')['is_correct'].mean().reset_index()
    # sns.lineplot(x='final_loop_count', y='is_correct', data=loop_data, marker='o', ax=ax, color='salmon')
    # ax.set_title('Agent Loop Count vs Accuracy')
    # ax.set_ylabel('Average Accuracy')
    # ax.set_xlabel('Total Sub-Loops Triggered')
    # ax.grid(True)
    # all_plots["loop_count_vs_accuracy"] = fig_loop_acc
    
    # Plot 4: Unnecessary Test Rate Distribution (NEW)
    # fig_waste, ax = plt.subplots(figsize=(10, 6))
    # sns.histplot(results_df['unnecessary_test_rate'], bins=10, ax=ax, kde=True, color='teal')
    # ax.set_title('Distribution of Unnecessary Test Rates')
    # ax.set_xlabel('Rate: (Proposed - Kept) / Proposed')
    # all_plots["unnecessary_test_rate_dist"] = fig_waste

    # Plot 5: Confusion Matrices
    # cm_plots = plot_confusion_matrix(results_df, model_name)
    # all_plots.update(cm_plots) # Add dict to dict
    
    # Plot confidence vs accuracy
    fig_conf_acc, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='model_confidence', y='is_correct', data=results_df, ax=ax, color='orchid', alpha=0.6)
    ax.set_title('Model Confidence vs Accuracy')
    ax.set_xlabel('Model Confidence')
    ax.set_ylabel('Correct Diagnosis (1=Yes, 0=No)')
    all_plots["confidence_vs_accuracy"] = fig_conf_acc
    
    # Plot confidence plot per disease
    fig_conf_disease, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(x='ground_truth_disease', y='model_confidence', data=results_df, ax=ax, palette="Set3")
    ax.set_title(f'Model Confidence per Disease ({model_name})', fontsize=16)
    ax.set_xlabel('Ground Truth Disease', fontsize=12)
    ax.set_ylabel('Model Confidence', fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig_conf_disease.tight_layout()
    all_plots["confidence_per_disease_boxplot"] = fig_conf_disease
    
    # Key = Column Name, Value = Display Label for Y-Axis
    metrics_to_plot = {
        'is_correct': 'Accuracy',
        'CES': 'Cost Efficiency Score (CES)',
        'UTR': 'Unnecessary Test Rate (UTR)',
        'FI_New': 'Frugality Index (FI)',
        'StD': 'Steps to Diagnosis (StD)',
        'final_dx_confidence': 'Model Confidence',
        'total_case_cost': 'Total Case Cost ($)'
    }

    # # Create a Bar Plot for each metric
    # for col, label in metrics_to_plot.items():
    #     try:
    #         fig, ax = plt.subplots(figsize=(14, 8))
            
    #         # Group by disease and calculate mean (sorted by metric value for readability)
    #         order = results_df.groupby('ground_truth_disease')[col].mean().sort_values().index
            
    #         sns.barplot(
    #             data=results_df,
    #             x='ground_truth_disease',
    #             y=col,
    #             order=order,
    #             palette='viridis',
    #             ax=ax,
    #             errorbar=None # Remove error bars for cleaner look, or use 'sd' for standard deviation
    #         )
            
    #         ax.set_title(f'Average {label} by Disease', fontsize=16)
    #         ax.set_xlabel('Disease', fontsize=12)
    #         ax.set_ylabel(f'Avg {label}', fontsize=12)
    #         plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            
    #         # Add value labels on top of bars
    #         for container in ax.containers:
    #             ax.bar_label(container, fmt='%.2f', padding=3)

    #         fig.tight_layout()
    #         all_plots[f"disease_{col}"] = fig
            
    #     except Exception as e:
    #         print(f"Could not plot {col} by disease: {e}")

    # Save all plots
    for name, fig in all_plots.items():
        save_path = os.path.join(output_dir, f'{name}_{model_name_safe}.png')
        try:
            fig.savefig(save_path)
            print(f"Plot saved: {save_path}")
        except Exception as e:
            print(f"Error saving plot {name}: {e}")
        plt.close(fig) # Close the figure to save memory

    return summary_metrics, all_plots