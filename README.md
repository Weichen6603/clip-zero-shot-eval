# CLIP Zero-Shot Classification Evaluation Framework

A flexible framework for evaluating CLIP models on zero-shot classification tasks across multiple datasets.

**Included datasets**:
- CIFAR-10 (10 classes)
- CIFAR-100 Fine (100 classes) 
- CIFAR-100 Coarse (20 classes)
- SUN397 (397 scene categories)
- ImageNet-1K (1,000 object classes)
- Visual Genome Objects with Synsets (2,000+ semantic object types)
- TreeOfLife-10M (biological taxa)

## Features

- **Modular Design**: Easy to add new datasets through adapter pattern with centralized template management
- **Multiple Dataset Support**: Built-in support for CIFAR-10, CIFAR-100, SUN397, ImageNet-1K (Original ILSVRC2012), Visual Genome, and TreeOfLife-10M
- **Flexible Configuration**: YAML-based configuration for experiments
- **Comprehensive Metrics**: Accuracy, top-5 accuracy, per-class accuracy, confusion matrices
- **Template Ensemble**: Support for multiple prompt templates to improve performance
- **Result Management**: Automatic saving of results with timestamps

## Installation

You can install all dependencies with pip:

```bash
pip install -r requirements.txt
```

Or use the provided Conda environment (recommended for reproducibility):

```bash
conda env create -f env.yml
conda activate clip
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

### Data Loading

This framework downloads and caches datasets locally for evaluation. Different datasets use different loading mechanisms:

- **Small datasets** (CIFAR-10, CIFAR-100): Automatically downloaded via torchvision
- **Large datasets** (ImageNet-1K, SUN397, Visual Genome): Downloaded and cached via HuggingFace datasets
- **Special datasets** (TreeOfLife-10M): Supports multiple loading modes including WebDataset

For large datasets, the first run will download and cache the data locally, but subsequent runs will be much faster.

### Available Datasets

Choose from our pre-configured datasets and run evaluations immediately:

#### CIFAR-10 (10 classes, ~32x32 images)
```bash
python evaluate.py config/cifar10.yaml
```
- **Dataset**: Automatically downloaded via torchvision
- **Classes**: 10 basic categories (airplane, car, bird, cat, etc.)

#### CIFAR-100 (100 fine classes or 20 coarse superclasses, ~32x32 images)
```bash
python evaluate.py config/cifar100.yaml
```
- **Dataset**: Automatically downloaded via torchvision
- **Classes**: Supports both 100 fine-grained categories (e.g., apple, aquarium_fish, baby, ...) and 20 coarse superclasses (e.g., aquatic_mammals, flowers, vehicles, ...)
- **Configurable**: Use `use_coarse_labels: false` for fine labels, `use_coarse_labels: true` for coarse labels in the config file
- **Note**: The provided `cifar100.yaml` integrates both fine and coarse label evaluation—run once to get both results.

#### SUN397 (397 scene categories, large-scale)
```bash
python evaluate.py config/sun397.yaml
```
- **Dataset**: Automatically downloaded via HuggingFace
- **Classes**: 397 scene categories (abbey, airport, alley, etc.)

#### ImageNet-1K (Original ILSVRC2012, 1000 classes, requires access approval)
```bash
# First-time setup required (see detailed guide below)
python evaluate.py config/imagenet.yaml
```
- **Dataset**: Original ImageNet-1K (ILSVRC2012) via HuggingFace (requires approval)
- **Classes**: 1000 object categories from the ImageNet Large Scale Visual Recognition Challenge 2012
- **Note**: Full dataset will be downloaded and cached locally on first run

#### Visual Genome (2000+ object types, complex scenes)
```bash
python evaluate.py config/visual_genome.yaml
```
- **Dataset**: Automatically downloaded via HuggingFace
- **Classes**: 2000+ unique object types in rich visual scenes

#### TreeOfLife-10M (Biological Taxa Classification)

> ⚠️ **Warning**  
> Due to the large size of the dataset, local and WebDataset modes have not been tested. The provided code and usage instructions are for reference only. Streaming mode is recommended for quick visualization and experimentation.

TreeOfLife-10M is the largest ML-ready biological dataset with 10+ million images covering 454,000+ taxa. We provide flexible configuration options to accommodate different use cases and resource constraints.

```bash
# Quick start with streaming mode (no download required)
python evaluate_clip.py config/treeoflife_streaming.yaml

# Development with local files
python evaluate_clip.py config/treeoflife_local.yaml

# Production with WebDataset (best performance)
python evaluate_clip.py config/treeoflife_webdataset.yaml
```

**Key Features**:
- **Three Loading Modes**: WebDataset (production), Local (development), Streaming (quick start)
- **Flexible Scale**: From 10K samples (testing) to 10M+ images (full dataset)
- **Configurable Taxonomy**: Choose classification level from species to kingdom
- **Smart Data Loading**: Efficient memory usage with multiple backend options

**Important Notes**:
- Requires HuggingFace authentication: `huggingface-cli login`
- Streaming mode uses placeholder labels due to ID mismatch
- For accurate results, use WebDataset or Local mode

📖 **[See the complete TreeOfLife-10M guide](./docs/treeoflife.md)** for detailed setup instructions, configuration options, and best practices.

**Quick Configuration Examples**:
```yaml
# For testing (streaming mode, 10K samples)
mode: "streaming"
max_samples: 10000

# For development (local files, train_small)
mode: "local"  
split: "train_small"

# For production (WebDataset, full performance)
mode: "webdataset"
split: "train_small"  # or "train" for full dataset
```

### Custom Configuration

For custom settings, modify any config file or create your own:

```bash
python evaluate.py your_custom_config.yaml
```

**Available configurations**:
- `config/comprehensive.yaml` - All datasets except TreeOfLife-Full (recommended for benchmarking)
- `config/cifar10.yaml` - CIFAR-10 only
- `config/cifar100.yaml` - CIFAR-100 (both fine and coarse labels)
- `config/sun397.yaml` - SUN397 scene recognition
- `config/imagenet.yaml` - ImageNet-1K validation set
- `config/visual_genome.yaml` - Visual Genome with WordNet synsets
- `config/treeoflife.yaml` - TreeOfLife-10M lightweight version
- `config/treeoflife_full.yaml` - TreeOfLife-10M full dataset (research quality)

### ImageNet-1K Detailed Setup

1. **Setup authentication**:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```

2. **Request dataset access**: Visit [imagenet-1k on HuggingFace](https://huggingface.co/datasets/imagenet-1k) and request access

3. **Run evaluation**:
   ```bash
   python evaluate.py config/imagenet.yaml
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
- **Classes**: 1000 object classes from the ImageNet Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)
- **Images**: Variable size color images, 50,000 validation samples
- **Source**: Official Hugging Face `imagenet-1k` dataset (original ImageNet validation set)
- **Authentication**: Requires HuggingFace account and dataset access approval
- **Usage**: `type: "imagenet"`
- **Dataset Version**: This is the original ImageNet-1K dataset (ILSVRC2012), not variants like ImageNetV2, ImageNet-A, ImageNet-R, ImageNet-S, or ImageNet-Sketch
- **Features**:
  - Complete ImageNet-1K validation set (50,000 images)
  - Full 1000-class taxonomy with synset mappings
  - Automatic caching to local storage
- **Setup Requirements**:
  1. **Authentication**: `huggingface-cli login`
  2. **Dataset Access**: Request access to `imagenet-1k` on HuggingFace
- **Image Loading**:
  - Supports HuggingFace datasets for automatic downloading and caching

#### Visual Genome
- **Classes**: 2,000+ unique object types in rich visual scenes
- **Images**: ~108,000 variable size color images with complex object relationships
- **Source**: Visual Genome v1.2.0 Objects dataset via HuggingFace
- **Usage**: `type: "visual_genome"`
- **Features**:
  - **Lazy Loading**: Memory-efficient loading for large-scale evaluation
  - **Flexible Filtering**: Configurable object count and sample size limits
  - **Rich Annotations**: Multiple objects per image with detailed labels
  - **Zero-Shot Challenge**: Complex scenes with diverse object types
- **Configuration Parameters**:
  ```yaml
  - name: "VisualGenome-Objects"
    type: "visual_genome"
    root_path: "/path/to/cache"          # HuggingFace cache directory
    config_name: "objects_v1.2.0"       # Dataset version
    
    # Lazy loading optimization parameters
    max_samples: null        # Use all samples (~108K) or limit (e.g., 10000)
    min_objects: 1           # Minimum objects per image (filter empty images)
    max_objects: null        # Maximum objects per image (null = no limit)
    use_synsets: true       # Use object names (false) vs WordNet synsets (true)(recommmend)
  ```
- **Parameter Guide**:
  - **`max_samples`**: Control dataset size for faster testing or full evaluation
    - Remove or set `null` for full dataset (~80K-100K filtered samples)
    - Set to `10000` for medium-scale testing
    - Set to `1000` for quick validation
  - **`min_objects`**: Filter out images with too few objects (recommended: 1-2)
  - **`max_objects`**: Limit complexity by filtering highly complex images
    - Set `null` for no limit (include all complexity levels)
    - Set `20-30` for balanced complexity
  - **`use_synsets`**: Choose label type (typically `false` for readability)
  - **Samples**: ~80,000-100,000 images (after filtering)
  - **Classes**: 2,000-3,000 unique object types
  - **Memory Usage**: Low (thanks to lazy loading implementation)
- **Memory Optimization**: The adapter uses true lazy loading - only sample indices are kept in memory, with images and labels loaded on-demand and cached using LRU strategy


#### TreeOfLife-10M
📖 **[See the complete TreeOfLife-10M guide](./docs/treeoflife.md)** for detailed setup instructions, configuration options, and best practices.

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
│   ├── __init__.py             # Package initialization and auto-registration
│   ├── cifar10_adapter.py      # CIFAR-10 dataset adapter
│   ├── cifar100_adapter.py     # CIFAR-100 dataset adapter
│   ├── imagenet_adapter.py     # ImageNet-1K dataset adapter
│   ├── imagenet_classes.py     # ImageNet class mappings and synsets
│   ├── sun397_adapter.py       # SUN397 scene classification adapter
│   ├── visual_genome_adapter.py # Visual Genome objects adapter (lazy loading)
│   └── treeoflife_adapter.py   # TreeOfLife-10M dataset adapter (loads catalog.csv fully into memory)
├── config/                     # Pre-configured evaluation settings
│   ├── cifar10.yaml           # CIFAR-10 evaluation configuration
│   ├── cifar100.yaml          # CIFAR-100 evaluation configuration  
│   ├── imagenet.yaml          # ImageNet-1K evaluation configuration
│   ├── sun397.yaml            # SUN397 evaluation configuration
│   ├── visual_genome.yaml     # Visual Genome evaluation configuration
│   └── treeoflife.yaml        # TreeOfLife-10M evaluation configuration
├── data/                      # Dataset cache and storage
├── results/                   # Evaluation results and reports
├── dataset_adapters.py        # Central registry and unified templates
├── dataset_factory.py         # Factory for creating dataset adapters
├── base_dataset.py           # Base adapter class and interface
├── clip_classifier.py        # CLIP model wrapper and inference
├── evaluator.py              # Evaluation metrics and logic
├── evaluate.py               # Main evaluation script
└── requirements.txt          # Package dependencies
```
