# CLIP Zero-Shot Classification Evaluation Framework

A flexible framework for evaluating CLIP models on zero-shot classification tasks across multiple datasets.

## Features

- **Modular Design**: Easy to add new datasets through adapter pattern
- **Multiple Dataset Support**: Built-in support for CIFAR-10, ImageNet, and custom formats
- **Flexible Configuration**: YAML-based configuration for experiments
- **Comprehensive Metrics**: Accuracy, top-5 accuracy, per-class accuracy, confusion matrices
- **Template Ensemble**: Support for multiple prompt templates to improve performance
- **Result Management**: Automatic saving of results with timestamps

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

1. Prepare your configuration file (see `example_config.yaml`)
2. Run evaluation:

```bash
python evaluate.py config.yaml
```

## Adding New Datasets

1. Create a new adapter class inheriting from `BaseDatasetAdapter`
2. Implement required methods: `_load_data`, `_get_classes`, `get_templates`
3. Register the adapter in `DatasetFactory`

Example:
```python
class MyDatasetAdapter(BaseDatasetAdapter):
    def _load_data(self, **kwargs):
        # Load your dataset
        pass# dataset_factory.py
        from base_dataset import BaseDatasetAdapter
    
    def _get_classes(self):
        # Return list of class names
        pass
    
    def get_templates(self):
        # Return list of prompt templates
        return ["a photo of a {}.", "an image of a {}."]

# Register
DatasetFactory.register_adapter('mydataset', MyDatasetAdapter)
```

## Configuration Options

See `example_config.yaml` for all available options.

## Output Structure

```
results/
├── config_used.yaml       # Configuration used for reproducibility
├── results_YYYYMMDD_HHMMSS.json  # Detailed results
└── summary_YYYYMMDD_HHMMSS.txt   # Human-readable summary
```
```