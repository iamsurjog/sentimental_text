import csv
import itertools
import sys
import io
from main import main as main_runner

# --- Configuration ---
MODEL_TYPES = ["lstm", "rnn", "transformer", "attention"]
OUTPUT_TYPES = ["text_classification", "next_word_prediction", "sentiment_analysis"]
INPUT_FILE = "input.txt"
OUTPUT_FILE = "results.csv"

def run_combinations():
    """
    Reads input.txt, runs the main script for all combinations of model
    and output types, and stores the results in a CSV file.
    """
    print("Starting analysis...")
    original_argv = sys.argv
    original_stdout = sys.stdout

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
            lines = [line.strip() for line in infile if line.strip()]
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    combinations = list(itertools.product(MODEL_TYPES, OUTPUT_TYPES))
    print(f"Found {len(lines)} lines in {INPUT_FILE}.")
    print(f"Generated {len(combinations)} option combinations.")
    total_runs = len(lines) * len(combinations)
    print(f"Total runs to execute: {total_runs}")

    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["input_text", "model_type", "output_type", "result"])

        current_run = 0
        for line in lines:
            for model_type, output_type in combinations:
                current_run += 1
                print(f"Running ({current_run}/{total_runs}): model={model_type}, output={output_type}...")

                # Redirect stdout to suppress prints from main_runner
                sys.stdout = io.StringIO()

                try:
                    # Set command-line arguments for the main script
                    sys.argv = [
                        'main.py',
                        line,
                        '--model_type',
                        model_type,
                        '--output_type',
                        output_type
                    ]
                    # Run the main function and get the result
                    result = main_runner()
                    writer.writerow([line, model_type, output_type, result])

                except Exception as e:
                    # If the script fails, log the error in the CSV
                    error_message = f"ERROR: {type(e).__name__} - {e}"
                    print(f"  -> Failed: {error_message}")
                    writer.writerow([line, model_type, output_type, error_message])
                finally:
                    # Restore stdout and original arguments
                    sys.stdout = original_stdout
                    sys.argv = original_argv

    print(f"\nAnalysis complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_combinations()