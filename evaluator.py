### evaluator.py
# Main evaluation logic for zero-shot classification with error handling
###

import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Iterator, Tuple
from tqdm import tqdm
import json
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix
import warnings
from PIL import Image
import traceback

from base_dataset import BaseDatasetAdapter
from clip_classifier import CLIPZeroShotClassifier

# Suppress warnings
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
warnings.filterwarnings("ignore", message=".*TIFF.*")


def worker_init_fn(worker_id):
    """Initialize worker process to suppress warnings."""
    warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
    warnings.filterwarnings("ignore", message=".*TIFF.*")
    warnings.filterwarnings("ignore", message=".*truncated.*")


class ErrorHandlingDataLoader:
    """DataLoader wrapper that handles errors gracefully."""
    
    def __init__(self, dataset, batch_size=32, shuffle=False, num_workers=4, 
                 pin_memory=True, drop_last=False):
        """Initialize error-handling DataLoader."""
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Create collate function that handles errors
        def safe_collate_fn(batch):
            """Collate function that filters out failed samples."""
            valid_samples = []
            
            for sample in batch:
                if sample is not None and isinstance(sample, (tuple, list)) and len(sample) == 2:
                    img, label = sample
                    # Check if image is valid tensor
                    if isinstance(img, torch.Tensor) and not torch.isnan(img).any():
                        valid_samples.append(sample)
            
            if not valid_samples:
                # Return minimal batch if all samples failed
                return torch.zeros(1, 3, 224, 224), torch.tensor([0])
            
            # Separate and stack valid samples
            images, labels = zip(*valid_samples)
            images = torch.stack(images)
            labels = torch.tensor(labels) if not isinstance(labels[0], torch.Tensor) else torch.stack(labels)
            
            return images, labels
        
        # Create underlying DataLoader
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=safe_collate_fn,
            worker_init_fn=worker_init_fn
        )
        
        self.failed_batches = 0
        self.total_batches = 0
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate with error handling."""
        for batch_idx, batch_data in enumerate(self.loader):
            try:
                self.total_batches += 1
                yield batch_data
            except Exception as e:
                self.failed_batches += 1
                error_msg = str(e)
                if 'TIFF' in error_msg or 'tiff' in error_msg.lower():
                    # Silently skip TIFF errors (common in TreeOfLife)
                    continue
                elif self.failed_batches <= 5:  # Only print first 5 errors
                    print(f"⚠️  Error in batch {batch_idx}: {type(e).__name__}: {error_msg[:100]}")
                continue
    
    def __len__(self):
        return len(self.loader)


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

        # Create error-handling dataloader
        dataloader = ErrorHandlingDataLoader(
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
        
        # Track progress and errors
        samples_processed = 0
        batch_errors = 0

        try:
            for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Evaluating")):
                try:
                    # Check for valid batch
                    if images.size(0) == 0:
                        continue
                    
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # Get predictions
                    predictions, similarities = self.classifier.classify_images(images, text_features)

                    all_predictions.append(predictions.cpu())
                    all_labels.append(labels.cpu())
                    all_similarities.append(similarities.cpu())
                    
                    samples_processed += labels.size(0)
                    
                except Exception as e:
                    batch_errors += 1
                    if batch_errors <= 5:  # Only print first 5 batch errors
                        print(f"\n⚠️  Error processing batch {batch_idx}: {type(e).__name__}: {str(e)[:100]}")
                    continue

        except KeyboardInterrupt:
            print("\n⚠️  Evaluation interrupted by user")
        except Exception as e:
            print(f"\n❌  Critical error during evaluation: {type(e).__name__}: {e}")
            traceback.print_exc()

        # Check if we have any results
        if not all_predictions:
            print("❌  No valid predictions obtained!")
            return {
                'dataset_name': type(dataset).__name__,
                'error': 'No valid predictions obtained',
                'samples_processed': 0,
                'batch_errors': batch_errors
            }

        # Concatenate all results
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        all_similarities = torch.cat(all_similarities)
        
        # Compute metrics
        accuracy = (all_predictions == all_labels).float().mean().item()
        
        print(f"\nDebug info:")
        print(f"  Samples processed: {samples_processed} / {len(dataset)}")
        print(f"  Batch errors: {batch_errors}")
        print(f"  Failed batches: {dataloader.failed_batches} / {dataloader.total_batches}")
        print(f"  all_predictions shape: {all_predictions.shape}")
        print(f"  all_labels shape: {all_labels.shape}")
        print(f"  all_similarities shape: {all_similarities.shape}")
        if len(all_predictions) > 0:
            print(f"  Sample prediction: {all_predictions[0].item()}")
            print(f"  Sample label: {all_labels[0].item()}")
            print(f"  Sample similarities (top 5): {all_similarities[0].argsort(descending=True)[:5]}")

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
            'samples_processed': samples_processed,
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
            'evaluation_stats': {
                'samples_processed': samples_processed,
                'batch_errors': batch_errors,
                'failed_batches': dataloader.failed_batches,
                'total_batches': dataloader.total_batches,
                'success_rate': (dataloader.total_batches - dataloader.failed_batches) / dataloader.total_batches if dataloader.total_batches > 0 else 0
            }
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
                                   save_dir: Optional[str] = None,
                                   save_individual: bool = False,
                                   config=None) -> Dict[str, Dict[str, Any]]:
        """Evaluate on multiple datasets.

        Args:
            datasets: Dictionary mapping dataset names to dataset adapters
            batch_size: Batch size for evaluation
            save_dir: Directory to save results (for comprehensive summary)
            save_individual: Whether to save each dataset to its own directory
            config: Configuration object to save with individual results

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
                if 'error' not in results:
                    print(f"  Accuracy: {results['accuracy']:.2%}")
                    print(f"  Top-5 Accuracy: {results['top5_accuracy']:.2%}")
                    print(f"  Mean Per-Class Accuracy: {results['mean_per_class_accuracy']:.2%}")
                    if 'evaluation_stats' in results:
                        stats = results['evaluation_stats']
                        print(f"  Samples processed: {stats['samples_processed']} / {results['num_samples']}")
                        print(f"  Success rate: {stats['success_rate']:.2%}")
                else:
                    print(f"  Error: {results['error']}")

                # Save individual dataset results immediately
                if save_individual:
                    self._save_individual_dataset_results(name, results, config)

            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                traceback.print_exc()
                all_results[name] = {'error': str(e)}

        # Save comprehensive results if directory provided
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

            print(f"\nComprehensive results saved to {save_dir}")

        return all_results

    def _save_individual_dataset_results(self, dataset_name: str, results: Dict[str, Any], config=None):
        """Save individual dataset results to its corresponding directory, using output_dir from config if available, else root_path for naming."""
        # Try to get output_dir from config (per-dataset)
        output_dir = None
        if config and isinstance(config, dict):
            # config can be the full config or a dataset config
            if 'output_dir' in config:
                output_dir = config['output_dir']
            elif 'datasets' in config:
                # Try to find the dataset config by name
                for ds in config['datasets']:
                    if ds.get('name', '').lower() == dataset_name.lower():
                        output_dir = ds.get('output_dir', None)
                        break
        if output_dir:
            individual_save_dir = output_dir
        else:
            # Fallback to root_path logic
            root_path = None
            if 'root_path' in results:
                root_path = results['root_path']
            elif 'model_info' in results and 'root_path' in results['model_info']:
                root_path = results['model_info']['root_path']
            if not root_path:
                dir_name = dataset_name.lower().replace('-', '_') + '_evaluation'
            else:
                last_part = os.path.basename(os.path.normpath(root_path))
                dir_name = f"{last_part}_evaluation"
            individual_save_dir = f"./results/{dir_name}"
        os.makedirs(individual_save_dir, exist_ok=True)
        # Save config file if provided
        if config:
            from config_parser import ConfigParser
            config_save_path = os.path.join(individual_save_dir, 'config_used.yaml')
            ConfigParser.save_config(config, config_save_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(individual_save_dir, f"results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump({dataset_name: results}, f, indent=2)
        summary_file = os.path.join(individual_save_dir, f"summary_{timestamp}.txt")
        self._save_summary({dataset_name: results}, summary_file)
        print(f"  → Results saved to {individual_save_dir}")
        return individual_save_dir

    def _save_summary(self, results: Dict[str, Dict[str, Any]], filepath: str):
        """Save a human-readable summary of results."""
        with open(filepath, 'w') as f:
            f.write("Zero-Shot Classification Results Summary\n")
            f.write("=" * 60 + "\n\n")

            # Model info
            first_result = next(iter(results.values()))
            if 'model_info' in first_result and 'error' not in first_result:
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
                    acc = result['accuracy']
                    top5 = result['top5_accuracy'] 
                    per_class = result['mean_per_class_accuracy']
                    f.write(f"{dataset_name:<30} {acc:<10.2%} {top5:<10.2%} {per_class:<10.2%}\n")
            
            # Add evaluation statistics if available
            f.write("\n" + "-" * 60 + "\n")
            f.write("Evaluation Statistics:\n")
            for dataset_name, result in results.items():
                if 'evaluation_stats' in result:
                    stats = result['evaluation_stats']
                    f.write(f"\n{dataset_name}:\n")
                    f.write(f"  Samples processed: {stats['samples_processed']}\n")
                    f.write(f"  Batch errors: {stats['batch_errors']}\n")
                    f.write(f"  Success rate: {stats['success_rate']:.2%}\n")