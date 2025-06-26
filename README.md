# CLIP Zero-Shot Classification Evaluation Framework

A flexible framework for evaluating CLIP models on zero-shot classification tasks across multiple datasets.

## Features

- **Modular Design**: Easy to add new datasets through adapter pattern with centralized template management
- **Multiple Dataset Support**: Built-in support for CIFAR-10, CIFAR-100, SUN397, ImageNet-1K (Original ILSVRC2012), Visual Genome, and TreeOfLife-10M
- **Flexible Configuration**: YAML-based configuration for experiments
- **Comprehensive Metrics**: Accuracy, top-5 accuracy, per-class accuracy, confusion matrices
- **Template Ensemble**: Support for multiple prompt templates to improve performance
- **Result Management**: Automatic saving of results with timestamps

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

### Available Datasets

Choose from our pre-configured datasets and run evaluations immediately:

#### CIFAR-10 (10 classes, ~32x32 images)
```bash
python evaluate.py config/cifar10.yaml
```
- **Dataset**: Automatically downloaded via torchvision
- **Classes**: 10 basic categories (airplane, car, bird, cat, etc.)

#### CIFAR-100 (100 classes, ~32x32 images)
```bash
python evaluate.py config/cifar100.yaml
```
- **Dataset**: Automatically downloaded via torchvision
- **Classes**: 100 detailed categories with fine/coarse labels

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
- **Note**: This is the original ImageNet dataset, not variants like ImageNetV2, ImageNet-A, ImageNet-R, etc.

#### Visual Genome (2000+ object types, complex scenes)
```bash
python evaluate.py config/visual_genome.yaml
```
- **Dataset**: Automatically downloaded via HuggingFace
- **Classes**: 2000+ unique object types in rich visual scenes
- **Features**: Lazy loading for memory efficiency

#### TreeOfLife-10M (454K+ biological taxa, lightweight testing)
```bash
# Lightweight testing (recommended - 1,000 images, ~150MB)
python evaluate.py config/treeoflife.yaml

# Quick development testing (100 images, ~15MB)
# Edit config to set max_samples: 100
```
- **Dataset**: TreeOfLife-10M via HuggingFace (train_small subset)
- **Lightweight Approach**: Uses 1M image subset instead of full 10M+ dataset
- **Classes**: 454,000+ taxa across the entire tree of life (configurable taxonomic level)
- **Sources**: iNaturalist21, BIOSCAN-1M, Encyclopedia of Life
- **Features**: 
  - **Zero-shot ready**: Pre-computed BioCLIP text embeddings available
  - **Scalable testing**: 100 images (dev) → 1,000 images (testing) → 1M images (research)
  - **Fast loading**: train_small subset (~150-200GB vs 1.9TB full dataset)
  - **Lazy loading**: Only metadata in memory, images loaded on-demand
- **Storage**: ~150-200MB for 1,000 images (vs 1.9TB full dataset)

## 🚀 Lightweight Zero-Shot Testing (Recommended)

For quick evaluation and development, we recommend starting with lightweight configurations:

### TreeOfLife-10M Lightweight Testing Path

**✅ Goal**: Low-cost zero-shot classification experiments  
**✅ Method**: Image features + text similarity (no model training)  
**✅ Dataset**: `train_small` subset + BioCLIP text embeddings  
**✅ Storage**: ~150-200MB instead of 1.9TB

#### Quick Start Commands

```bash
# 🚀 Ultra-light testing (3 shards, 300 images, ~300MB, 5-10 min) - Perfect for dev/debug
python evaluate.py config/treeoflife_ultralight.yaml

# 🔬 Standard lightweight testing (5 shards, 1,000 images, ~500MB, 10-20 min) - Recommended
python evaluate.py config/treeoflife.yaml

# 📊 Custom lightweight testing
python test_lightweight.py --samples 100

# ⚡ Quick test without CLIP loading (dataset validation only)
python test_lightweight.py --skip-clip --samples 50
```

#### Testing Strategy

| Scale | Shards | Images | Storage | Time | Use Case |
|-------|--------|--------|---------|------|----------|
| **Ultra-light** | 3 | 300 | ~300MB | 5-10 min | Algorithm development, debugging |
| **Standard** | 5 | 1,000 | ~500MB | 10-20 min | Method evaluation, CI/CD testing |
| **Research** | 10 | 20,000 | ~1GB | 30-60 min | Paper results, model comparison |
| **Full** | 73 | 1M+ | 150-200GB | Hours | Production evaluation |

#### Benefits of Lightweight Approach

- **🔄 Quick iteration**: Test algorithm changes in minutes, not hours
- **💾 Ultra-low storage**: Download only 3-5 shards (~300-500MB) vs full dataset (3TB)
- **⚡ Fast loading**: Uses pre-filtered `train_small` subset with shard limiting
- **🧠 Low memory**: Lazy loading keeps only metadata in RAM
- **📊 Representative**: Subset is uniformly sampled from full dataset
- **🎯 Zero-shot focused**: Uses pre-computed BioCLIP text embeddings
- **🛠️ Configurable**: Easily adjust number of shards (3 for dev, 5 for testing, 10+ for research)

### Custom Configuration

For custom settings, modify any config file or create your own:

```bash
python evaluate.py your_custom_config.yaml
```

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
  - WSL/Windows cross-platform support
- **Setup Requirements**:
  1. **Authentication**: `huggingface-cli login`
  2. **Dataset Access**: Request access to `imagenet-1k` on HuggingFace
  3. **Storage**: ~7GB cache space required

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
    use_synsets: false       # Use object names (false) vs WordNet synsets (true)
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
- **Expected Results** (full dataset):
  - **Samples**: ~80,000-100,000 images (after filtering)
  - **Classes**: 2,000-3,000 unique object types
  - **Memory Usage**: Low (thanks to lazy loading implementation)
- **Memory Optimization**: The adapter uses true lazy loading - only sample indices are kept in memory, with images and labels loaded on-demand and cached using LRU strategy

#### TreeOfLife-10M (Two Configuration Options)

TreeOfLife-10M is the largest ML-ready biological dataset with 10+ million images covering 454,000+ taxa. Due to its massive size (~3TB), we provide two configuration options:

##### Option 1: Lightweight Version (Recommended for Testing)
- **Configuration File**: `config/treeoflife.yaml`
- **Dataset Size**: ~300MB (3 shards only)
- **Evaluation Time**: 30-60 minutes
- **Use Case**: Algorithm development, quick testing, CI/CD

```bash
python evaluate.py config/treeoflife.yaml
```

**Features**:
- Downloads only first 3 shards (~100K images each)
- Uses all samples from the 3 shards (no artificial limits)
- Perfect for rapid prototyping and testing
- Requires HuggingFace authentication: `huggingface-cli login`

##### Option 2: Full Dataset Version (Research Quality)
- **Configuration File**: `config/treeoflife_full.yaml`
- **Dataset Size**: ~3TB (all 73 shards)
- **Evaluation Time**: Days to weeks
- **Use Case**: Research publications, comprehensive evaluation

```bash
python evaluate.py config/treeoflife_full.yaml
```

**Features**:
- Complete TreeOfLife-10M dataset (10+ million images)
- All 454,000+ taxa available
- Suitable for publication-quality research
- Requires HuggingFace authentication: `huggingface-cli login`

##### Authentication Setup

Before using TreeOfLife-10M, you need to authenticate with HuggingFace:

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Login to HuggingFace
huggingface-cli login
```

##### Common Features (Both Versions)
- **Classes**: 454,000+ taxa across the entire tree of life (configurable by taxonomic level)
- **Source**: TreeOfLife-10M dataset via HuggingFace (iNaturalist21, BIOSCAN-1M, Encyclopedia of Life)
- **Authentication**: Requires HuggingFace login for dataset access
- **Usage**: `type: "treeoflife"`
- **Features**:
  - **Streaming Data Loading**: Memory-efficient loading for massive biological dataset
  - **Configurable Taxonomy**: Choose classification level from species to kingdom
  - **Multi-source Data**: Combines expert-labeled museum specimens, field photos, and curated images
  - **Biological Diversity**: Covers animals, plants, fungi, and microorganisms

##### Configuration Parameters (Both Versions)
```yaml
- name: "TreeOfLife-10M"
  type: "treeoflife"
  root_path: "/mnt/d/data/treeoflife"   # Cache directory for HuggingFace datasets
  split: "train"                        # TreeOfLife-10M only has 'train' split
  
  # Taxonomic configuration
  taxonomic_level: "species"            # species, genus, family, order, class, phylum, kingdom
  use_common_names: true                # Include common names in text templates
  exclude_partial_labels: false        # Filter samples without full taxonomy
  
  # Version-specific parameters
  max_shards: 3                         # Lightweight: 3, Full: null (all 73 shards)
  min_images_per_class: 1               # Minimum samples per taxonomic class
```

##### Parameter Guide
- **`taxonomic_level`**: Choose classification granularity
  - `"species"`: Finest-grained (454K+ classes, most challenging)
  - `"genus"`: Moderate granularity (good balance of diversity and feasibility)
  - `"family"`: Broader categories (manageable class count for testing)
  - `"kingdom"`: Coarsest level (animals, plants, fungi, etc.)
- **`max_shards`**: Control dataset size
  - `3`: Lightweight version (~300MB, 3 shards)
  - `null`: Full version (~3TB, all 73 shards)
- **`min_images_per_class`**: Filter rare taxa
  - Higher values = fewer but better-represented classes
  - Lower values = more taxonomic diversity but potential class imbalance

##### Expected Results Comparison

| Version | Data Size | Samples | Classes (Species) | Time | Use Case |
|---------|-----------|---------|-------------------|------|----------|
| **Lightweight** | ~300MB | ~300K | ~thousands | 30-60 min | Testing, Development |
| **Full** | ~3TB | ~10M+ | ~454K+ | Days-Weeks | Research, Publications |

##### Hardware Requirements
- **Lightweight Version**: 
  - Storage: 1GB+ available space
  - RAM: 8GB+ recommended
  - GPU: Any modern GPU
- **Full Version**:
  - Storage: 3TB+ available space
  - RAM: 32GB+ recommended
  - GPU: High-end GPU (V100, A100, RTX 4090, etc.)
  - Network: High-speed internet for initial download

##### Dataset Scope (Both Versions)
This is the largest ML-ready biological dataset, covering:
- **Animals**: Mammals, birds, reptiles, amphibians, fish, insects, marine life
- **Plants**: Flowering plants, trees, ferns, mosses, algae
- **Fungi**: Mushrooms, molds, yeasts, lichens
- **Microorganisms**: Bacteria, protists, archaea

**Use Cases**: Ideal for biodiversity research, conservation applications, and biological foundation model evaluation

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
│   └── treeoflife_adapter.py   # TreeOfLife-10M dataset adapter (lazy loading)
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
