# CLIP Zero-Shot Classification Evaluation Framework

A flexible framework for evaluating CLIP models on zero-shot classification tasks across multiple datasets.

## 🚀 Recent Updates

**Major Refactoring (June 2025)**: The project has been refactored with a new modular architecture:
- **Centralized Templates**: All prompt templates are now managed in `dataset_adapters.py` for consistency
- **Modular Adapters**: Each dataset adapter is in its own file within the `adapters/` directory
- **Auto-Registration**: New adapters are automatically discovered and registered
- **ImageNet-1K Support**: Added complete support for official HuggingFace ImageNet-1K dataset
- **Backward Compatible**: Existing configurations and usage remain unchanged

## Features

- **Modular Design**: Easy to add new datasets through adapter pattern with centralized template management
- **Multiple Dataset Support**: Built-in support for CIFAR-10, CIFAR-100, SUN397, and ImageNet-1K
- **Large Dataset Optimization**: Memory-efficient lazy loading for large datasets
- **Flexible Configuration**: YAML-based configuration for experiments
- **Comprehensive Metrics**: Accuracy, top-5 accuracy, per-class accuracy, confusion matrices
- **Template Ensemble**: Support for multiple prompt templates to improve performance
- **Result Management**: Automatic saving of results with timestamps
- **Cross-Platform Storage**: Support for WSL environments with Windows drive access

## Installation

```bash
pip install -r requirements.txt
```

### Additional Setup for ImageNet-1K

ImageNet-1K requires additional authentication setup:

```bash
# Install HuggingFace Hub (if not already installed)
pip install huggingface_hub

# Login to HuggingFace
huggingface-cli login

# Request access to imagenet-1k dataset at:
# https://huggingface.co/datasets/imagenet-1k
```

**Note**: You may need to wait for approval to access the ImageNet-1K dataset.

## Quick Start

1. Prepare your configuration file (see `example_config.yaml`)
2. Run evaluation:

```bash
python evaluate.py config.yaml
```

### ImageNet-1K Quick Start

1. **Setup authentication**:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```

2. **Request dataset access**: Visit [imagenet-1k on HuggingFace](https://huggingface.co/datasets/imagenet-1k) and request access

3. **Run evaluation**:
   ```bash
   python evaluate.py config/imagenet_config.yaml
   ```

4. **View results**: Check `./results/imagenet_evaluation/` for detailed results

## Adding New Datasets

The project uses a modular adapter system with centralized template management. To add a new dataset:

1. **Create adapter file**: Create a new file in the `adapters/` directory
2. **Implement adapter class**: Inherit from `BaseDatasetAdapter` and implement required methods
3. **Use shared templates**: Leverage centralized templates from `DatasetTemplates`
4. **Register adapter**: Add to the registry for automatic discovery

### Example: Creating a New Adapter

Create `adapters/my_dataset_adapter.py`:

```python
"""My dataset adapter."""

import os
import sys
from typing import List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_dataset import BaseDatasetAdapter

class MyDatasetAdapter(BaseDatasetAdapter):
    def _load_data(self, **kwargs):
        """Load your dataset."""
        data = []
        # Your data loading logic here
        # Return list of dicts with 'image_path' and 'label'
        return data
    
    def _get_classes(self) -> List[str]:
        """Return list of class names."""
        return ['class1', 'class2', 'class3']  # Your classes
    
    def get_templates(self) -> List[str]:
        """Use unified templates from centralized system."""
        from dataset_adapters import DatasetTemplates
        # All datasets now use the same comprehensive template set
        return DatasetTemplates.get_templates()
```

### Register the Adapter

Add to `adapters/__init__.py`:
```python
from .my_dataset_adapter import MyDatasetAdapter
```

Add to the registry in `dataset_adapters.py`:
```python
try:
    from adapters.my_dataset_adapter import MyDatasetAdapter
    DatasetAdapterRegistry.register('my_dataset', MyDatasetAdapter)
except ImportError:
    pass
```

### Unified Template System

The framework now uses a single comprehensive template set for all datasets, ensuring consistency across evaluations:

- **Unified Templates** (100 templates): A comprehensive set combining the best templates from object classification, scene classification, and ImageNet-style prompts
- **Consistent Performance**: All datasets benefit from the same diverse template variations  
- **Simplified Maintenance**: Single template set reduces complexity and ensures consistent updates

This unified approach eliminates the need to choose different template sets and provides optimal performance across all dataset types.

## Configuration Options

See `example_config.yaml` for all available options.

### Performance Tuning

Monitor your system resources and adjust:
- **Increase batch_size** if you have more GPU memory available
- **Increase num_workers** if you have more CPU cores
- **Use max_samples** for quick testing before full evaluation

### Supported Datasets

#### CIFAR-10
- **Classes**: 10 basic categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Images**: 32x32 color images, 10,000 test samples
- **Usage**: `type: "cifar10"`

#### CIFAR-100
- **Fine Labels**: 100 detailed classes (apple, aquarium_fish, baby, bear, etc.)
- **Coarse Labels**: 20 superclasses (aquatic_mammals, fish, flowers, etc.)
- **Images**: 32x32 color images, 10,000 test samples
- **Usage**: 
  - Fine-grained: `type: "cifar100"`, `use_coarse_labels: false`
  - Coarse-grained: `type: "cifar100"`, `use_coarse_labels: true`

#### SUN397
- **Classes**: 397 scene categories (abbey, airplane_cabin, airport_terminal, alley, etc.)
- **Images**: Variable size color images, ~87,000 samples
- **Challenge**: Large-scale scene understanding dataset
- **Memory Optimization**: Uses lazy loading to handle large dataset efficiently
- **Usage**: `type: "sun397"`
- **Configuration Example**:
  ```yaml
  - name: "SUN397"
    type: "sun397"
    root_path: "/path/to/sun397/cache"  # HuggingFace cache directory
    split: "train"  # Will auto-split for evaluation
    # Optional: limit samples for testing
    # max_samples: 1000
  ```

#### ImageNet-1K
- **Classes**: 1000 object classes from the ImageNet Large Scale Visual Recognition Challenge
- **Images**: Variable size color images, 50,000 validation samples
- **Source**: Official Hugging Face `imagenet-1k` dataset
- **Authentication**: Requires HuggingFace account and dataset access approval
- **Usage**: `type: "imagenet"`
- **Features**:
  - Complete ImageNet-1K validation set (50,000 images)
  - Full 1000-class taxonomy with synset mappings
  - Automatic caching to local storage
  - WSL/Windows cross-platform support
- **Setup Requirements**:
  1. **Authentication**: `huggingface-cli login`
  2. **Dataset Access**: Request access to `imagenet-1k` on HuggingFace
  3. **Storage**: ~7GB cache space required
- **Configuration Example**:
  ```yaml
  - name: "ImageNet-1K"
    type: "imagenet"
    root_path: "/mnt/d/data/imagenet"  # Cache directory
    split: "validation"  # Use validation split (50K images)
  ```
- **Performance**: 
  - Initial download: ~30-40 minutes (one-time)
  - Evaluation speed: ~2000+ samples/sec on RTX 4070
  - Memory usage: ~6GB RAM, ~0.8GB GPU memory

## Output Structure

```
results/
├── config_used.yaml       # Configuration used for reproducibility
├── results_YYYYMMDD_HHMMSS.json  # Detailed results
└── summary_YYYYMMDD_HHMMSS.txt   # Human-readable summary
```

## Project Structure

```
clip-zero-shot-eval/
├── adapters/                    # Dataset adapter implementations
│   ├── __init__.py             # Package initialization  
│   ├── cifar10_adapter.py      # CIFAR-10 dataset adapter
│   ├── cifar100_adapter.py     # CIFAR-100 dataset adapter
│   ├── imagenet_adapter.py     # ImageNet dataset adapter
│   └── sun397_adapter.py       # SUN397 dataset adapter
├── config/                     # Configuration files
├── data/                      # Dataset storage
├── results/                   # Evaluation results
├── dataset_adapters.py        # Central registry and templates
├── dataset_factory.py         # Factory for creating adapters
├── base_dataset.py           # Base adapter class
├── clip_classifier.py        # CLIP model wrapper
├── evaluator.py              # Evaluation logic
├── evaluate.py               # Main evaluation script
└── requirements.txt          # Dependencies
```
