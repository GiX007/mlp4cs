"""Experiment 3: Modular pipeline with fine-tuned LLMs."""
from src.config import EXP3_CONFIGS
from src.pipeline.runner import run_experiment
from src.evaluation.evaluator import evaluate_experiment
from src.evaluation.tomiinek import run_tomiinek
from src.evaluation.reporter import save_results, print_table


def run_experiment_3(split: str = "dev") -> None:
    """
    Run Experiment 3: Modular pipeline with fine-tuned LLMs.

    Args:
        split: dataset split to evaluate on, one of 'train', 'dev', 'test'
    """
    all_results = {}

    for config_name, model_config in EXP3_CONFIGS.items():
        experiment_name = f"exp3_{config_name}"

        # 1. Run pipeline
        tomiinek_results, custom_results = run_experiment(experiment_name=experiment_name, model_config=model_config, split=split, zeroshot=True, exp_id=3)

        # 2. Evaluate custom metrics
        dataset_metrics = evaluate_experiment(custom_results, split)

        # 3. Evaluate tomiinek
        tomiinek_metrics = run_tomiinek(tomiinek_results)
        dataset_metrics.update(tomiinek_metrics)

        # 4. Save 3 JSON files
        save_results(experiment_name, dataset_metrics, split)

        all_results[config_name] = dataset_metrics

    # 5. Print comparison table
    print_table(all_results, "Modular Fine-Tuned Pipeline", exp_id=3)
