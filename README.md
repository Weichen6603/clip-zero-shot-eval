# CLIP Zero-Shot Classification Evaluation Framework

A flexible framework for evaluating CLIP models on zero-shot classification tasks across multiple datasets.

## Features

- **Modular Design**: Easy to add new datasets through adapter pattern
- **Multiple Dataset Support**: Built-in support for CIFAR-10, CIFAR-100, SUN397, ImageNet, and custom formats
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

### Memory Optimization for Large Datasets

For large datasets (like SUN397 with 87k samples), the framework uses memory-efficient strategies:

- **Lazy Loading**: Images are loaded on-demand during evaluation, not pre-loaded into memory
- **Batch Processing**: Configurable batch size to balance memory usage and GPU utilization
- **Worker Optimization**: Adjustable number of data loading workers

**Note**: Small datasets (CIFAR-10, CIFAR-100) use traditional pre-loading for optimal performance, while large datasets automatically use lazy loading to prevent memory issues.


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

### Performance Optimization & Analysis

Based on comprehensive performance analysis on RTX 4070 Laptop GPU with 20 CPU cores:

#### Recommended Settings for Large Datasets (SUN397)
- **Batch size**: 32 (optimal balance between memory and speed)
- **Number of workers**: 8 (optimal data loading throughput)
- **Expected throughput**: ~200 samples/sec for data loading, ~2200 samples/sec for inference

#### Performance Analysis Results
- **Data loading bottleneck**: Use `batch_size=128, num_workers=8` for maximum data loading speed (198.2 samples/sec)
- **Inference optimization**: Use `batch_size=32` for maximum inference speed (2226.7 samples/sec) 
- **End-to-end performance**: batch_size=32 provides best overall performance (~14s for 1000 samples)
- **Memory usage**: ~6GB RAM, ~0.7GB GPU memory for batch_size=32

#### Hardware Requirements
- **GPU**: NVIDIA RTX 4070 or better recommended
- **RAM**: Minimum 8GB, 16GB recommended for large datasets
- **CPU**: Multi-core CPU benefits data loading (8+ cores recommended)

#### Scaling for Different Hardware
- **Lower-end GPU**: Reduce batch_size to 16 or 8
- **Less RAM**: Enable `max_samples` limit in config
- **Fewer CPU cores**: Reduce `num_workers` to 2-4

#### Performance Tools
- Run `python data_loading_performance.py` to analyze data loading performance
- Run `python performance_analysis.py` for comprehensive analysis including inference

#### ImageNet
- **Classes**: 1000 object classes
- **Images**: Variable size color images
- **Usage**: `type: "imagenet"`

#### Custom Datasets
- **Formats**: JSON, CSV, TXT annotations
- **Flexible**: Configurable image and label keys
- **Usage**: `type: "custom"`

## Output Structure

```
results/
├── config_used.yaml       # Configuration used for reproducibility
├── results_YYYYMMDD_HHMMSS.json  # Detailed results
└── summary_YYYYMMDD_HHMMSS.txt   # Human-readable summary
```
```