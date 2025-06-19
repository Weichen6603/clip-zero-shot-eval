### evaluator.py
# Main evaluation logic for zero-shot classification
###

import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional
import numpy as np
from tqdm import tqdm
import json
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report

from base_dataset import BaseDatasetAdapter
from clip_classifier import CLIPZeroShotClassifier


class ZeroShotEvaluator:
    """Evaluator for zero-shot classification tasks.

    This class handles the evaluation of CLIP models on various datasets
    for zero-shot classification, computing metrics and saving results.
    """

    def __init__(self,
                 clip_model: str = "ViT-B/32",
                 device: str = "cuda",
                 use_ensemble: bool = True):
        """
        Args:
            clip_model: CLIP model variant to use
            device: Device for computation
            use_ensemble: Whether to use template ensemble
        """
        self.classifier = CLIPZeroShotClassifier(
            model_name=clip_model,
            device=device,
            use_ensemble=use_ensemble
        )
        self.device = device

    def evaluate_dataset(self,
                         dataset: BaseDatasetAdapter,
                         batch_size: int = 32,
                         num_workers: int = 4,
                         save_predictions: bool = False) -> Dict[str, Any]:
        """Evaluate CLIP on a single dataset.

        Args:
            dataset: Dataset adapter instance
            batch_size: Batch size for evaluation
            num_workers: Number of data loading workers
            save_predictions: Whether to save individual predictions

        Returns:
            Dictionary containing evaluation metrics
        """
        # Set preprocessing
        dataset.transform = self.classifier.preprocess

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        # Get classes and templates
        classes = dataset.classes
        templates = dataset.get_templates()

        print(f"Evaluating on {len(dataset)} samples with {len(classes)} classes")
        print(f"Using {len(templates)} templates for text encoding")

        # Encode text descriptions
        text_features = self.classifier.encode_text_classes(classes, templates)

        # Evaluation loop
        all_predictions = []
        all_labels = []
        all_similarities = []

        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Get predictions
            predictions, similarities = self.classifier.classify_images(images, text_features)

            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            all_similarities.append(similarities.cpu())

        # Concatenate all results
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        all_similarities = torch.cat(all_similarities)

        # Compute metrics
        accuracy = (all_predictions == all_labels).float().mean().item()

        # Per-class accuracy
        per_class_correct = torch.zeros(len(classes))
        per_class_total = torch.zeros(len(classes))

        for pred, label in zip(all_predictions, all_labels):
            per_class_total[label] += 1
            if pred == label:
                per_class_correct[label] += 1

        per_class_accuracy = per_class_correct / per_class_total.clamp(min=1)

        # Top-k accuracy
        top5_correct = 0
        for i, label in enumerate(all_labels):
            top5_preds = all_similarities[i].argsort(descending=True)[:5]
            if label in top5_preds:
                top5_correct += 1
        top5_accuracy = top5_correct / len(all_labels)

        # Confusion matrix (only for small number of classes)
        conf_matrix = None
        if len(classes) <= 20:
            conf_matrix = confusion_matrix(
                all_labels.numpy(),
                all_predictions.numpy()
            ).tolist()

        # Prepare results
        results = {
            'dataset_name': type(dataset).__name__,
            'num_samples': len(dataset),
            'num_classes': len(classes),
            'accuracy': accuracy,
            'top5_accuracy': top5_accuracy,
            'per_class_accuracy': {
                classes[i]: acc.item()
                for i, acc in enumerate(per_class_accuracy)
            },
            'mean_per_class_accuracy': per_class_accuracy.mean().item(),
            'confusion_matrix': conf_matrix,
            'model_info': self.classifier.get_model_info(),
            'evaluation_time': datetime.now().isoformat(),
        }

        # Save predictions if requested
        if save_predictions:
            results['predictions'] = {
                'predicted_labels': all_predictions.tolist(),
                'true_labels': all_labels.tolist(),
                'prediction_scores': all_similarities.tolist(),
            }

        return results

    def evaluate_multiple_datasets(self,
                                   datasets: Dict[str, BaseDatasetAdapter],
                                   batch_size: int = 32,
                                   save_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Evaluate on multiple datasets.

        Args:
            datasets: Dictionary mapping dataset names to dataset adapters
            batch_size: Batch size for evaluation
            save_dir: Directory to save results

        Returns:
            Dictionary mapping dataset names to their results
        """
        all_results = {}

        for name, dataset in datasets.items():
            print(f"\n{'=' * 50}")
            print(f"Evaluating on {name}")
            print(f"{'=' * 50}")

            try:
                results = self.evaluate_dataset(dataset, batch_size=batch_size)
                all_results[name] = results

                # Print summary
                print(f"\nResults for {name}:")
                print(f"  Accuracy: {results['accuracy']:.2%}")
                print(f"  Top-5 Accuracy: {results['top5_accuracy']:.2%}")
                print(f"  Mean Per-Class Accuracy: {results['mean_per_class_accuracy']:.2%}")

            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                all_results[name] = {'error': str(e)}

        # Save results if directory provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

            # Save detailed results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(save_dir, f"results_{timestamp}.json")

            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)

            # Save summary
            summary_file = os.path.join(save_dir, f"summary_{timestamp}.txt")
            self._save_summary(all_results, summary_file)

            print(f"\nResults saved to {save_dir}")

        return all_results

    def _save_summary(self, results: Dict[str, Dict[str, Any]], filepath: str):
        """Save a human-readable summary of results."""
        with open(filepath, 'w') as f:
            f.write("Zero-Shot Classification Results Summary\n")
            f.write("=" * 60 + "\n\n")

            # Model info
            first_result = next(iter(results.values()))
            if 'model_info' in first_result:
                f.write("Model Information:\n")
                for key, value in first_result['model_info'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

            # Dataset results
            f.write("Dataset Results:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Dataset':<30} {'Accuracy':<10} {'Top-5':<10} {'Per-Class':<10}\n")
            f.write("-" * 60 + "\n")

            for dataset_name, result in results.items():
                if 'error' in result:
                    f.write(f"{dataset_name:<30} {'ERROR':<10} {'-':<10} {'-':<10}\n")
                else:
                    acc = result['accuracy'] * 100
                    top5 = result['top5_accuracy'] * 100
                    per_class = result['mean_per_class_accuracy'] * 100
                    f.write(f"{dataset_name:<30} {acc:<10.2f} {top5:<10.2f} {per_class:<10.2f}\n")