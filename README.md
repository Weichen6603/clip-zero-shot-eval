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

### Data Loading Modes: Online (Streaming) vs. Offline (Local Caching)

This framework supports two main data loading modes for large datasets:

- **Online (Streaming) Mode**: Images are loaded on demand from HuggingFace during evaluation, not all downloaded at once. This keeps memory and disk usage low, and is recommended for most users and large datasets. Enable by setting `streaming: true` in the config.
- **Offline (Local Caching) Mode**: All images are downloaded and cached locally before evaluation. This can be faster for repeated runs if you have enough disk space, but requires significant storage. Enable by setting `streaming: false` (or omitting the option for some datasets).

Most adapters default to the recommended mode for their dataset size. You can override this in the config if needed.

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
- **Image loading supports HuggingFace streaming mode (recommended)**

#### ImageNet-1K (Original ILSVRC2012, 1000 classes, requires access approval)
```bash
# First-time setup required (see detailed guide below)
python evaluate.py config/imagenet.yaml
```
- **Dataset**: Original ImageNet-1K (ILSVRC2012) via HuggingFace (requires approval)
- **Classes**: 1000 object categories from the ImageNet Large Scale Visual Recognition Challenge 2012
- **Image loading supports HuggingFace streaming mode (recommended)**

#### Visual Genome (2000+ object types, complex scenes)
```bash
python evaluate.py config/visual_genome.yaml
```
- **Dataset**: Automatically downloaded via HuggingFace
- **Classes**: 2000+ unique object types in rich visual scenes
- **Image loading supports HuggingFace streaming mode (recommended)**

#### TreeOfLife-10M (454K+ biological taxa, lightweight testing)
```bash
# Lightweight testing (recommended - use official train_small split, ~265GB)
python evaluate.py config/treeoflife.yaml

# Full dataset version (all data, ~3TB)
python evaluate.py config/treeoflife_full.yaml
```
- **Dataset**: TreeOfLife-10M via HuggingFace 
- **Classes**: 454,000+ taxa across the entire tree of life (configurable taxonomic level)
- **Image loading supports HuggingFace streaming mode (recommended)**

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
  - Supports HuggingFace streaming (online) mode via the config

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

#### TreeOfLife-10M (Two Configuration Options)

TreeOfLife-10M is the largest ML-ready biological dataset with 10+ million images covering 454,000+ taxa. Due to its massive size (~3TB), we provide two configuration options:

##### Option 1: Lightweight Version (Recommended for Testing)
- **Configuration File**: `config/treeoflife.yaml`
- **Dataset Size**: ~265GB (official `train_small` split, 953K images)
- **Use Case**: Algorithm development, quick testing, CI/CD

```bash
python evaluate.py config/treeoflife.yaml
```

**Features**:
- Uses official `train_small` split for efficient testing
- Perfect for rapid prototyping and testing
- Requires HuggingFace authentication: `huggingface-cli login`
- **Ultra-fast taxonomy lookup**: The adapter loads the entire `catalog.csv` (taxonomic metadata, ~1.5GB) into memory as a pandas DataFrame for O(1) sample-to-taxonomy mapping. This enables extremely fast and scalable evaluation, at the cost of moderate RAM usage (typically 2-3GB peak for catalog; images are still streamed/lazy loaded).

##### Option 2: Full Dataset Version (Research Quality)
- **Configuration File**: `config/treeoflife_full.yaml`
- **Dataset Size**: ~3TB (all data, ~9.5M images)
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
- **Ultra-fast taxonomy lookup**: Same as above; catalog.csv is loaded fully into memory for all samples.

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
- **Source**: TreeOfLife-10M via HuggingFace (iNaturalist21, BIOSCAN-1M, Encyclopedia of Life)
- **Authentication**: Requires HuggingFace login for dataset access
- **Usage**: `type: "treeoflife"`
- **Features**:
  - **Streaming Data Loading**: Images are streamed from HuggingFace for memory efficiency; only taxonomy metadata is fully loaded into RAM.
  - **Configurable Taxonomy**: Choose classification level from species to kingdom
  - **Multi-source Data**: Combines expert-labeled museum specimens, field photos, and curated images
  - **Biological Diversity**: Covers animals, plants, fungi, and microorganisms
  - **Configurable label filtering**: By default, only empty/null labels are filtered. You can enable strict filtering of uncertain/confusing/hybrid labels (e.g., `confusor`, `sp.`, `x`, `unknown`) by setting `strict_label_filtering: true` in your config to ensure scientific validity and reproducibility.

---

### TreeOfLife-10M Data Loading, Catalog Indexing, and Memory Usage

- **Catalog Table (`catalog.csv`)**: On first run, the full catalog file is always downloaded locally (to the directory specified by `root_path`).
  - The adapter uses an ultra-lightweight text index, mapping each `sample_id` to its line number in the CSV for fast taxonomy lookup with minimal memory usage.

- **Image Data**: Images are processed using HuggingFace's streaming mode with **true on-demand memory management**.
  - During initial loading: Images are downloaded from HuggingFace, compressed to JPEG format (quality=85), and stored as compressed bytes in memory. Original image objects are immediately released.
  - During evaluation: Each compressed image is decoded to PIL Image only when accessed in `__getitem__`, processed/transformed, converted to tensor, then immediately garbage collected.
  - **True streaming behavior**: No image data persists in memory between accesses. Each image is decoded fresh from compressed bytes when needed, ensuring constant memory usage regardless of dataset size.
  - Memory footprint: ~50-90% reduction compared to storing uncompressed images, with fast decode times from compressed bytes.
  - This approach enables processing unlimited dataset sizes (full ~953K train_small or ~10M full dataset) with **predictable, constant memory usage**.

- **Advantages**:
  - **Memory efficiency**: True streaming with compressed storage enables processing massive datasets on modest hardware
  - **Predictable resource usage**: Constant memory footprint regardless of dataset size (train_small ~953K or full ~10M images)
  - **Fast taxonomy lookup**: Ultra-lightweight text indexing for O(1) sample-to-taxonomy mapping
  - **Configurable quality**: Strict label filtering optional for research vs. general use

- **Memory Usage Summary**:
  - **Catalog metadata**: ~100-200MB (text index) vs ~1.5-3GB (full pandas DataFrame)
  - **Image storage**: ~50-90% compression ratio (JPEG bytes vs uncompressed)
  - **Runtime memory**: Single image in memory at a time during evaluation
  - **Total footprint**: Constant ~500MB-1GB regardless of dataset size

- **Notes**:
  - The ultra-lightweight text index is built only once and reused for future runs.
  - If you run out of RAM when loading the catalog, the adapter will automatically fall back to the text index, but evaluation will be slower.
  - For full local image access, set `streaming=False` (not recommended except for special use cases).

##### Configuration Parameters (Both Versions)
```yaml
- name: "TreeOfLife-10M"
  type: "treeoflife"
  root_path: "/mnt/d/data/treeoflife"   # Cache directory for HuggingFace datasets
  split: "train"                        # TreeOfLife-10M only has 'train' split
  
  # Taxonomic configuration
  taxonomic_level: "species"            # species, genus, family, order, class, phylum, kingdom
  exclude_partial_labels: false        # Filter samples without full taxonomy
  strict_label_filtering: false        # Enable strict filtering of uncertain/confusing labels (default: false)
  
  # Version-specific parameters
  min_images_per_class: 1               # Minimum samples per taxonomic class
```

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
