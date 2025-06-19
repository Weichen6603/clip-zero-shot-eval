### config_parser.py
# Configuration file parser and validator
###

import torch
import yaml
import os
from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    type: str
    root_path: str
    split: str = 'test'
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Configuration for the entire experiment."""
    # Model configuration
    clip_model: str = "ViT-B/32"
    device: str = "cuda"
    use_ensemble: bool = True

    # Evaluation configuration
    batch_size: int = 32
    num_workers: int = 4
    save_predictions: bool = False

    # Output configuration
    output_dir: str = "./results"
    save_results: bool = True

    # Dataset configurations
    datasets: List[DatasetConfig] = field(default_factory=list)


class ConfigParser:
    """Parser for experiment configuration files."""

    @staticmethod
    def parse_yaml(config_path: str) -> ExperimentConfig:
        """Parse YAML configuration file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            ExperimentConfig object
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return ConfigParser.parse_dict(config_dict)

    @staticmethod
    def parse_dict(config_dict: Dict[str, Any]) -> ExperimentConfig:
        """Parse configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            ExperimentConfig object
        """
        # Parse dataset configurations
        datasets = []
        for dataset_dict in config_dict.get('datasets', []):
            # Extract base parameters
            name = dataset_dict['name']
            dtype = dataset_dict['type']
            root_path = dataset_dict['root_path']
            split = dataset_dict.get('split', 'test')

            # All other parameters go into params
            params = {k: v for k, v in dataset_dict.items()
                      if k not in ['name', 'type', 'root_path', 'split']}

            datasets.append(DatasetConfig(
                name=name,
                type=dtype,
                root_path=root_path,
                split=split,
                params=params
            ))

        # Create experiment config
        return ExperimentConfig(
            clip_model=config_dict.get('clip_model', 'ViT-B/32'),
            device=config_dict.get('device', 'cuda'),
            use_ensemble=config_dict.get('use_ensemble', True),
            batch_size=config_dict.get('batch_size', 32),
            num_workers=config_dict.get('num_workers', 4),
            save_predictions=config_dict.get('save_predictions', False),
            output_dir=config_dict.get('output_dir', './results'),
            save_results=config_dict.get('save_results', True),
            datasets=datasets
        )

    @staticmethod
    def validate_config(config: ExperimentConfig) -> List[str]:
        """Validate configuration and return list of warnings.

        Args:
            config: ExperimentConfig object to validate

        Returns:
            List of warning messages (empty if no issues)
        """
        warnings = []

        # Check CLIP model
        valid_models = [
            'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64',
            'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
        ]
        if config.clip_model not in valid_models:
            warnings.append(f"Unknown CLIP model: {config.clip_model}")

        # Check device
        if config.device == 'cuda' and not torch.cuda.is_available():
            warnings.append("CUDA requested but not available")

        # Check datasets
        for dataset in config.datasets:
            if not os.path.exists(dataset.root_path):
                warnings.append(f"Dataset path not found: {dataset.root_path}")

        # Check output directory
        if config.save_results and not os.path.exists(config.output_dir):
            warnings.append(f"Output directory will be created: {config.output_dir}")

        return warnings

    @staticmethod
    def save_config(config: ExperimentConfig, filepath: str):
        """Save configuration to YAML file.

        Args:
            config: ExperimentConfig object
            filepath: Path to save the configuration
        """
        config_dict = {
            'clip_model': config.clip_model,
            'device': config.device,
            'use_ensemble': config.use_ensemble,
            'batch_size': config.batch_size,
            'num_workers': config.num_workers,
            'save_predictions': config.save_predictions,
            'output_dir': config.output_dir,
            'save_results': config.save_results,
            'datasets': []
        }

        # Add datasets
        for dataset in config.datasets:
            dataset_dict = {
                'name': dataset.name,
                'type': dataset.type,
                'root_path': dataset.root_path,
                'split': dataset.split,
                **dataset.params
            }
            config_dict['datasets'].append(dataset_dict)

        # Save to file
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)