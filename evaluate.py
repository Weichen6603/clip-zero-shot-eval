### evaluate.py
# Main script for running zero-shot evaluation
###

# !/usr/bin/env python3
"""
Main script for evaluating CLIP zero-shot classification performance.

This script loads datasets according to configuration, runs CLIP zero-shot
classification evaluation, and saves results.
"""

import argparse
import os
import sys

from dataset_factory import DatasetFactory
from evaluator import ZeroShotEvaluator
from config_parser import ConfigParser


def main():
    """Main evaluation function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Evaluate CLIP zero-shot classification performance'
    )
    parser.add_argument(
        'config',
        type=str,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--override-device',
        type=str,
        choices=['cuda', 'cpu'],
        help='Override device specified in config'
    )
    parser.add_argument(
        '--override-batch-size',
        type=int,
        help='Override batch size specified in config'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        help='Only evaluate specified datasets (by name)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Load datasets and show info without running evaluation'
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = ConfigParser.parse_yaml(args.config)

    # Apply overrides
    if args.override_device:
        config.device = args.override_device
    if args.override_batch_size:
        config.batch_size = args.override_batch_size

    # Validate configuration
    warnings = ConfigParser.validate_config(config)
    if warnings:
        print("\nConfiguration warnings:")
        for warning in warnings:
            print(f"  - {warning}")
        print()

    # Filter datasets if specified
    if args.datasets:
        config.datasets = [d for d in config.datasets if d.name in args.datasets]
        if not config.datasets:
            print(f"Error: No datasets matched the filter: {args.datasets}")
            sys.exit(1)

    print(f"\nExperiment Configuration:")
    print(f"  CLIP Model: {config.clip_model}")
    print(f"  Device: {config.device}")
    print(f"  Use Ensemble: {config.use_ensemble}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Datasets: {[d.name for d in config.datasets]}")

    # Create output directory
    if config.save_results:
        os.makedirs(config.output_dir, exist_ok=True)

        # Save configuration for reproducibility
        config_save_path = os.path.join(config.output_dir, 'config_used.yaml')
        ConfigParser.save_config(config, config_save_path)
        print(f"\nConfiguration saved to: {config_save_path}")

    # Load datasets
    print("\nLoading datasets...")
    datasets = {}

    for dataset_config in config.datasets:
        try:
            print(f"  Loading {dataset_config.name}...")

            # Prepare dataset configuration
            dataset_params = {
                'type': dataset_config.type,
                'root_path': dataset_config.root_path,
                'split': dataset_config.split,
                **dataset_config.params
            }

            # Create dataset
            dataset = DatasetFactory.create_dataset(dataset_params)
            datasets[dataset_config.name] = dataset

            print(f"    Loaded {len(dataset)} samples, {len(dataset.classes)} classes")

        except Exception as e:
            print(f"    Error loading dataset: {e}")
            if not args.dry_run:
                sys.exit(1)

    if args.dry_run:
        print("\nDry run complete. Exiting.")
        return

    # Create evaluator
    print("\nInitializing CLIP model...")
    evaluator = ZeroShotEvaluator(
        clip_model=config.clip_model,
        device=config.device,
        use_ensemble=config.use_ensemble
    )

    # Run evaluation
    print("\nStarting evaluation...")
    results = evaluator.evaluate_multiple_datasets(
        datasets,
        batch_size=config.batch_size,
        save_dir=config.output_dir if config.save_results else None
    )

    # Print final summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print("\nFinal Results Summary:")
    print(f"{'Dataset':<30} {'Accuracy':<10} {'Top-5':<10}")
    print("-" * 50)

    for name, result in results.items():
        if 'error' not in result:
            acc = result['accuracy'] * 100
            top5 = result['top5_accuracy'] * 100
            print(f"{name:<30} {acc:<10.2f}% {top5:<10.2f}%")
        else:
            print(f"{name:<30} ERROR")

    if config.save_results:
        print(f"\nDetailed results saved to: {config.output_dir}")


if __name__ == '__main__':
    main()