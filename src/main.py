"""Main entry point for MLP4CS pipeline."""
import time
from src.utils import print_separator, capture_and_save
from src.config import LOGS_DIR, RESULTS_DIR
from src.analysis.error_analysis import run_analysis
from src.experiments.exp1 import run_experiment_1
from src.experiments.exp2 import run_experiment_2
from src.experiments.exp3 import run_experiment_3


def main() -> None:
    """Run the full MLP4CS pipeline."""
    print_separator("MLP4CS: MODULAR PIPELINE FOR CUSTOMER SERVICE")
    start_time = time.time()

    # Experiment 1: Single-LLM baseline
    print("\n\n>>> Running Experiment 1: Single-LLM Baseline ...")
    run_experiment_1()

    # Experiment 2: Modular pipeline, zero-shot
    print("\n\n>>> Running Experiment 2: Modular Zero-shot LLM Pipeline ...")
    run_experiment_2()

    # Experiment 3: Modular pipeline, fine-tuned models
    # print("\n\n>>> Running Experiment 3: Modular LLM Pipeline with Fine-tuned LLMs ...")
    # run_experiment_3()

    total = time.time() - start_time
    print(f"\n>>> Total execution time: {total:.2f} seconds")
    print_separator("MLP4CS PIPELINE COMPLETED")


if __name__ == "__main__":
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Run everything and save all terminal output to log file
    capture_and_save(func=main, output_path=str(LOGS_DIR / f"run_{timestamp}.txt"))

    # Error analysis
    capture_and_save(func=run_analysis, output_path=RESULTS_DIR / f"error_analysis_{timestamp}.txt")


