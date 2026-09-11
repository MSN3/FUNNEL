# run_headless.py
import asyncio
import argparse
import os
import time
from dotenv import load_dotenv
load_dotenv()
from main_processor import run_full_analysis_pipeline

async def main(config_path):
    """
    Main async function to run the analysis pipeline.
    """
    print(f"Starting headless analysis with config: {config_path}")
    print("-" * 30)
    start_time = time.time()
    
    # Define a simple console-based progress callback
    def console_progress(fraction):
        # Create a simple text-based progress bar
        bar_length = 40
        filled_length = int(bar_length * fraction)
        bar = "█" * filled_length + "-" * (bar_length - filled_length)
        print(f"\rProcessing: |{bar}| {fraction*100:.1f}%", end="", flush=True)

    # Call the main pipeline function
    results_df, summary_metrics, generated_plots = await run_full_analysis_pipeline(
        config_path=config_path,
        progress_callback=console_progress
    )
    
    end_time = time.time()
    print(f"\n\n--- Analysis Complete in {end_time - start_time:.2f} seconds ---")
    
    # --- Print Results to Console ---
    
    if summary_metrics:
        print("\n--- Overall Summary ---")
        for key, value in summary_metrics.items():
            print(f"  - {key}: {value}")
    
    if generated_plots:
        print(f"\n{len(generated_plots)} plots were generated and saved to the 'outputs' directory.")
    
    if results_df is not None and not results_df.empty:
        print(f"\nProcessed {len(results_df)} records.")
        print("\n--- Sample of Results (first 5 rows) ---")
        print(results_df[['patient_id', 'ground_truth_disease', 'llm_diagnosis', 'model_confidence', 'is_correct', 'frugality_ratio']].head().to_string())
    else:
        print("\nError: No results were generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the agentic medical analysis without Streamlit.")
    
    # Set a default path for the config file
    default_config = os.path.abspath("./config.py")
    
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help=f"Path to the config.py file. Defaults to: {default_config}"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        print("Please create the file in the './example_config/' directory or provide a valid path using --config")
    else:
        # Run the main async function
        asyncio.run(main(args.config))